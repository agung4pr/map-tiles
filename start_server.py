from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import os
import json
import rasterio
from pathlib import Path
from tile_pyramid_generator import TilePyramidGenerator
import threading
import math
import io
from PIL import Image
import time
from threading import Lock

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
TILES_FOLDER = 'tiles'
ALLOWED_EXTENSIONS = {'tif', 'tiff'}

Path(UPLOAD_FOLDER).mkdir(exist_ok=True)
Path(TILES_FOLDER).mkdir(exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB max

# ====== CONCURRENCY PROTECTION ======

file_operation_lock = Lock()
generation_lock = Lock()
active_uploads = {}
upload_lock = Lock()

# Global reference to current generator (for immediate cancellation)
current_generator = None
generator_lock = Lock()

# Store tiling progress and control
tiling_progress = {
    'status': 'idle',  # idle, processing, cancelling, complete, error, cancelled
    'progress': 0, 
    'message': '', 
    'current': 0, 
    'total': 0,
    'should_cancel': False,
    'filename': None,
    'is_cancelling': False  # Track if cancellation is in progress
}

# ====== UTILITY FUNCTIONS ======

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_tiff_bounds(tiff_path):
    """Get geographic bounds of a TIFF file."""
    try:
        with rasterio.open(tiff_path) as src:
            bounds = src.bounds
            return {
                'west': bounds.left,
                'south': bounds.bottom,
                'east': bounds.right,
                'north': bounds.top,
                'crs': str(src.crs)
            }
    except Exception as e:
        print(f"[ERROR] Cannot read TIFF bounds: {e}")
        return None


def get_unique_filename(original_filename, folder):
    """Check if file exists and auto-number if duplicate."""
    filepath = os.path.join(folder, original_filename)
    
    if not os.path.exists(filepath):
        return original_filename
    
    base_name = Path(original_filename).stem
    extension = Path(original_filename).suffix
    counter = 1
    
    while True:
        new_filename = f"{base_name} ({counter}){extension}"
        filepath = os.path.join(folder, new_filename)
        
        if not os.path.exists(filepath):
            print(f"[UPLOAD] Duplicate detected! Renamed: {original_filename} -> {new_filename}")
            return new_filename
        
        counter += 1


def check_upload_in_progress(filename):
    """Check if file is currently being uploaded."""
    with upload_lock:
        return filename in active_uploads


def mark_upload_start(filename):
    """Mark that a file upload is starting."""
    with upload_lock:
        if filename in active_uploads:
            return False
        active_uploads[filename] = time.time()
        print(f"[UPLOAD] Upload started: {filename}")
        return True


def mark_upload_end(filename):
    """Mark that a file upload is complete."""
    with upload_lock:
        if filename in active_uploads:
            del active_uploads[filename]
            print(f"[UPLOAD] Upload completed: {filename}")


def generate_pyramid_background(tiff_path, output_dir, max_workers=4, min_zoom=None, max_zoom=None):
    """Run pyramid generation in background with IMMEDIATE cancellation support."""
    global tiling_progress, current_generator
    
    # Acquire generation lock
    acquired = generation_lock.acquire(blocking=False)
    
    if not acquired:
        print(f"[ERROR] Generation already in progress")
        tiling_progress['status'] = 'error'
        tiling_progress['message'] = 'Another generation is already in progress'
        return
    
    try:
        tiling_progress['status'] = 'processing'
        tiling_progress['message'] = 'Starting pyramid generation...'
        tiling_progress['progress'] = 0
        tiling_progress['should_cancel'] = False
        tiling_progress['filename'] = Path(tiff_path).name
        
        # Create generator instance
        generator = TilePyramidGenerator(tiff_path, output_dir, tile_size=256, max_workers=max_workers)
        
        # Store reference so /api/cancel can access it
        with generator_lock:
            current_generator = generator
        
        def progress_callback(current, total):
            """Called during generation to check for cancellation."""
            global tiling_progress, current_generator
            
            # Check cancellation flag
            if tiling_progress['should_cancel']:
                print(f"[CANCEL] User cancellation signal detected, stopping immediately")
                # Set the generator's internal cancel flag
                with generator_lock:
                    if current_generator:
                        current_generator.set_cancel_flag()
            
            tiling_progress['current'] = current
            tiling_progress['total'] = total
            tiling_progress['progress'] = int((current / total) * 100) if total > 0 else 0
        
        # Verify file still exists
        if not os.path.exists(tiff_path):
            raise FileNotFoundError(f"Source TIFF not found: {tiff_path}")
        
        # Use provided zoom levels or defaults
        gen_min_zoom = min_zoom if min_zoom is not None else generator.min_zoom
        gen_max_zoom = max_zoom if max_zoom is not None else generator.max_zoom
        
        # Ensure min doesn't exceed max
        if gen_min_zoom > gen_max_zoom:
            gen_min_zoom, gen_max_zoom = gen_max_zoom, gen_min_zoom
        
        # Clamp to valid range but respect user's choice to go lower
        gen_min_zoom = max(0, min(gen_min_zoom, 24))
        gen_max_zoom = max(0, min(gen_max_zoom, 24))
        
        print(f"[GENERATION] Starting: {Path(tiff_path).name} (Zoom {gen_min_zoom}-{gen_max_zoom})")
        print(f"[GENERATION] Requested zoom range: min={min_zoom}, max={max_zoom}")
        print(f"[GENERATION] Calculated defaults: min={generator.min_zoom}, max={generator.max_zoom}")
        print(f"[GENERATION] After processing: min={gen_min_zoom}, max={gen_max_zoom}")
        
        # Generate tiles - will stop immediately if should_cancel is set
        generator.generate(min_zoom=gen_min_zoom, max_zoom=gen_max_zoom, progress_callback=progress_callback)
        
        # Check if we were cancelled
        if tiling_progress['should_cancel']:
            print(f"[CANCEL] Generation was stopped by user")
            tiling_progress['status'] = 'cancelled'
            tiling_progress['message'] = 'Generation cancelled'
            tiling_progress['is_cancelling'] = False
        else:
            # Only save metadata if completed normally
            if not os.path.exists(output_dir):
                raise Exception("Output directory was not created")
            
            metadata = generator.generate_metadata()
            metadata_path = os.path.join(output_dir, 'metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            tiling_progress['status'] = 'complete'
            tiling_progress['message'] = 'Pyramid generation complete!'
            tiling_progress['progress'] = 100
            tiling_progress['is_cancelling'] = False
            print(f"[SUCCESS] Generation complete: {Path(tiff_path).name}")
        
        tiling_progress['should_cancel'] = False
        
    except Exception as e:
        print(f"[ERROR] Generation failed: {e}")
        import traceback
        traceback.print_exc()
        tiling_progress['status'] = 'error'
        tiling_progress['message'] = str(e)
        tiling_progress['should_cancel'] = False
        tiling_progress['is_cancelling'] = False
        
    finally:
        # Clean up
        with generator_lock:
            current_generator = None
        
        generation_lock.release()
        tiling_progress['filename'] = None


class GenerationCancelled(Exception):
    """Custom exception for cancellation."""
    pass


# ====== FLASK ROUTES ======

@app.route('/')
def index():
    """Serve main page."""
    return render_template('index_xyz.html')


@app.route('/inspector')
def inspector():
    """Serve tile inspector."""
    return render_template('tile_inspector.html')


@app.route('/api/files', methods=['GET'])
def list_files():
    """List uploaded files and their pyramid status."""
    files = []
    
    try:
        for filename in os.listdir(UPLOAD_FOLDER):
            if allowed_file(filename):
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                file_stem = Path(filename).stem
                pyramid_dir = os.path.join(TILES_FOLDER, file_stem)
                
                bounds = get_tiff_bounds(filepath)
                if bounds is None:
                    continue
                
                has_pyramid = os.path.exists(pyramid_dir) and os.path.exists(
                    os.path.join(pyramid_dir, 'metadata.json')
                )
                
                metadata = None
                if has_pyramid:
                    try:
                        with open(os.path.join(pyramid_dir, 'metadata.json')) as f:
                            metadata = json.load(f)
                    except:
                        pass
                
                is_uploading = check_upload_in_progress(filename)
                
                files.append({
                    'name': filename,
                    'path': filename,
                    'bounds': bounds,
                    'has_pyramid': has_pyramid,
                    'pyramid_dir': file_stem,
                    'metadata': metadata,
                    'is_uploading': is_uploading
                })
    except Exception as e:
        print(f"[ERROR] Error listing files: {e}")
    
    return jsonify(files)


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file upload with protection against concurrent uploads."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid TIFF file'}), 400
    
    original_filename = file.filename
    filename = get_unique_filename(original_filename, app.config['UPLOAD_FOLDER'])
    
    if not mark_upload_start(filename):
        return jsonify({'error': 'This file is already being uploaded'}), 409
    
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        with file_operation_lock:
            file.save(filepath)
            print(f"[UPLOAD] Saved file: {filename}")
        
        if not os.path.exists(filepath):
            raise Exception("File was not saved properly")
        
        bounds = get_tiff_bounds(filepath)
        if bounds is None:
            os.remove(filepath)
            return jsonify({'error': 'Invalid TIFF file - cannot read bounds'}), 400
        
        return jsonify({
            'success': True,
            'filename': filename,
            'bounds': bounds
        })
        
    except Exception as e:
        print(f"[ERROR] Upload failed: {e}")
        return jsonify({'error': str(e)}), 500
        
    finally:
        mark_upload_end(filename)


@app.route('/api/estimate-tiles', methods=['POST'])
def estimate_tiles():
    """Estimate number of tiles for given zoom range."""
    data = request.json
    filename = data.get('filename')
    min_zoom = data.get('min_zoom')
    max_zoom = data.get('max_zoom')
    
    if not filename or not allowed_file(filename):
        return jsonify({'error': 'Invalid filename'}), 400
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    
    try:
        generator = TilePyramidGenerator(filepath, TILES_FOLDER, tile_size=256, max_workers=1)
        
        # If custom zoom not provided, use defaults
        if min_zoom is None:
            min_zoom = generator.min_zoom
        if max_zoom is None:
            max_zoom = generator.max_zoom
        
        # Estimate tile count
        estimated_tiles = generator.estimate_tile_count(min_zoom, max_zoom)
        
        return jsonify({
            'estimated_tiles': estimated_tiles,
            'min_zoom': generator.min_zoom,
            'max_zoom': generator.max_zoom,
            'suggested_min': generator.min_zoom,
            'suggested_max': generator.max_zoom
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/pyramid', methods=['POST'])
def create_pyramid():
    """Create or regenerate tile pyramid from TIFF.
    
    Supports both initial generation and regeneration with different zoom levels.
    When regenerating, new tiles will be written to the same directory, 
    potentially overwriting existing tiles in modified zoom ranges.
    """
    global tiling_progress
    
    data = request.json
    filename = data.get('filename')
    max_workers = data.get('max_workers', 4)
    min_zoom = data.get('min_zoom')  # Optional custom zoom
    max_zoom = data.get('max_zoom')  # Optional custom zoom
    
    if not filename or not allowed_file(filename):
        return jsonify({'error': 'Invalid filename'}), 400
    
    if tiling_progress['status'] == 'processing':
        return jsonify({
            'error': 'Generation already in progress',
            'current_file': tiling_progress['filename']
        }), 409
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    
    output_dir = os.path.join(TILES_FOLDER, Path(filename).stem)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    tiling_progress['should_cancel'] = False
    
    thread = threading.Thread(
        target=generate_pyramid_background,
        args=(filepath, output_dir, max_workers, min_zoom, max_zoom),
        daemon=False,
        name=f"TileGen-{filename}"
    )
    thread.start()
    
    return jsonify({'success': True, 'message': 'Pyramid generation started'})


@app.route('/api/progress', methods=['GET'])
def get_progress():
    """Get generation progress."""
    return jsonify({
        'status': tiling_progress['status'],
        'progress': tiling_progress['progress'],
        'message': tiling_progress['message'],
        'current': tiling_progress['current'],
        'total': tiling_progress['total'],
        'filename': tiling_progress['filename'],
        'is_cancelling': tiling_progress['is_cancelling']
    })


@app.route('/api/cancel', methods=['POST'])
def cancel_generation():
    """Cancel the current generation - IMMEDIATELY."""
    global tiling_progress, current_generator
    
    if tiling_progress['status'] != 'processing':
        return jsonify({'error': 'No generation in progress'}), 400
    
    # Set the flag and mark as cancelling
    tiling_progress['should_cancel'] = True
    tiling_progress['is_cancelling'] = True
    tiling_progress['status'] = 'cancelling'
    tiling_progress['message'] = 'Cancellation in progress...'
    
    # ALSO immediately notify the generator
    with generator_lock:
        if current_generator:
            print(f"[CANCEL] Immediately cancelling generator thread")
            current_generator.set_cancel_flag()
    
    print(f"[CANCEL] Cancellation requested by user (immediate)")
    
    return jsonify({'success': True, 'message': 'Cancellation requested'})


@app.route('/tiles/<pyramid_name>/<int:z>/<int:x>/<int:y>.png', methods=['GET'])
def serve_tile(pyramid_name, z, x, y):
    """Serve XYZ tile."""
    tile_path = os.path.join(TILES_FOLDER, pyramid_name, str(z), str(x), f'{y}.png')
    
    if not os.path.exists(tile_path):
        img = Image.new('RGBA', (256, 256), (0, 0, 0, 0))
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        return send_file(img_bytes, mimetype='image/png')
    
    return send_file(tile_path, mimetype='image/png')


@app.route('/api/pyramid-info/<pyramid_name>', methods=['GET'])
def pyramid_info(pyramid_name):
    """Get pyramid metadata."""
    metadata_path = os.path.join(TILES_FOLDER, pyramid_name, 'metadata.json')
    
    if not os.path.exists(metadata_path):
        return jsonify({'error': 'Pyramid not found'}), 404
    
    try:
        with open(metadata_path) as f:
            metadata = json.load(f)
        return jsonify(metadata)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/delete/<filename>', methods=['DELETE'])
def delete_file(filename):
    """Delete file and pyramid with protection."""
    if not allowed_file(filename):
        return jsonify({'error': 'Invalid filename'}), 400
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    pyramid_dir = os.path.join(TILES_FOLDER, Path(filename).stem)
    
    if tiling_progress['filename'] == filename and tiling_progress['status'] == 'processing':
        return jsonify({
            'error': 'Cannot delete file while generation is in progress'
        }), 409
    
    try:
        with file_operation_lock:
            if os.path.exists(filepath):
                os.remove(filepath)
                print(f"[DELETE] Deleted file: {filename}")
            
            if os.path.exists(pyramid_dir):
                import shutil
                shutil.rmtree(pyramid_dir)
                print(f"[DELETE] Deleted pyramid: {Path(filename).stem}")
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("[SERVER] ====================================")
    print("[SERVER] XYZ Tile Pyramid Server (Hardened)")
    print("[SERVER] WITH IMMEDIATE CANCELLATION")
    print("[SERVER] ====================================")
    print("[SERVER] Features:")
    print("[SERVER]   Concurrent upload protection")
    print("[SERVER]   Single generation at a time")
    print("[SERVER]   IMMEDIATE cancellation")
    print("[SERVER]   File locking for safety")
    print("[SERVER]   Stress-test resilient")
    print("[SERVER] Visit: http://localhost:2000")
    print("[SERVER] ====================================\n")
    app.run(debug=True, host='0.0.0.0', port=2000, use_reloader=False)
