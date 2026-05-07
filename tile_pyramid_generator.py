import rasterio
import numpy as np
from rasterio.windows import Window
from rasterio.transform import from_bounds
from PIL import Image
import os
from pathlib import Path
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
from rasterio.crs import CRS
from rasterio.warp import calculate_default_transform, reproject, Resampling
import threading


class WebMercator:
    """Web Mercator projection utilities for XYZ tiles."""
    
    EARTH_RADIUS = 6378137  # meters
    MAX_LATITUDE = 85.051129
    
    @staticmethod
    def lat_lon_to_meters(lat, lon):
        """Convert lat/lon to Web Mercator meters."""
        x = lon * WebMercator.EARTH_RADIUS / 180.0
        y = math.log(math.tan((90.0 + lat) * math.pi / 360.0)) * WebMercator.EARTH_RADIUS
        return x, y
    
    @staticmethod
    def meters_to_lat_lon(x, y):
        """Convert Web Mercator meters to lat/lon."""
        lon = (x / WebMercator.EARTH_RADIUS) * 180.0
        lat = (360.0 / math.pi) * math.atan(math.exp(y / WebMercator.EARTH_RADIUS)) - 90.0
        return lat, lon
    
    @staticmethod
    def tile_bounds(z, x, y):
        """Get geographic bounds (meters in Web Mercator) for a tile."""
        resolution = (2 * math.pi * WebMercator.EARTH_RADIUS) / (256 * (2 ** z))
        
        meter_x = x * 256 * resolution - math.pi * WebMercator.EARTH_RADIUS
        meter_y = -(y * 256 * resolution - math.pi * WebMercator.EARTH_RADIUS)
        
        meter_width = 256 * resolution
        meter_height = 256 * resolution
        
        north = meter_y
        south = meter_y - meter_height
        west = meter_x
        east = meter_x + meter_width
        
        return {
            'north': north,
            'south': south,
            'west': west,
            'east': east
        }
    
    @staticmethod
    def lat_lon_to_tile(z, lat, lon):
        """Convert lat/lon to tile coordinates."""
        n = 2 ** z
        x = n * ((lon + 180) / 360)
        lat_rad = math.radians(lat)
        y = n * (1 - math.log(math.tan(lat_rad) + 1/math.cos(lat_rad)) / math.pi) / 2
        return int(x), int(y)


class TileStorage:
    """Handle tile image storage and caching."""
    
    def __init__(self, output_dir):
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.tile_cache = {}  # In-memory cache: (z, x, y) -> filepath
    
    def get_tile_path(self, z, x, y):
        """Get the file path for a tile."""
        tile_dir = Path(self.output_dir) / str(z) / str(x)
        return tile_dir / f"{y}.png"
    
    def save_tile(self, z, x, y, img):
        """
        Save a tile image to disk with verification.
        
        Args:
            z, x, y: Tile coordinates
            img: PIL Image object
            
        Returns:
            str: Path to saved tile, or None if save failed
        """
        try:
            tile_path = self.get_tile_path(z, x, y)
            tile_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Preserve alpha channel if present (CRITICAL for transparency!)
            if img.mode in ('RGB', 'RGBA', 'L'):
                img.save(tile_path, 'PNG', optimize=False)
            else:
                # Convert to RGBA to preserve transparency capability
                img = img.convert('RGBA')
                img.save(tile_path, 'PNG', optimize=False)
            
            # Verify file exists and has content
            if tile_path.exists() and tile_path.stat().st_size > 0:
                self.tile_cache[(z, x, y)] = str(tile_path)
                file_size = tile_path.stat().st_size
                print(f"  ✓ Saved z={z} x={x} y={y} ({file_size} bytes)")
                return str(tile_path)
            else:
                print(f"  ✗ Tile save failed verification: {tile_path}")
                return None
                
        except Exception as e:
            print(f"  ✗ Error saving tile z={z} x={x} y={y}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def tile_exists(self, z, x, y):
        """Check if tile exists in cache or on disk."""
        if (z, x, y) in self.tile_cache:
            return True
        
        tile_path = self.get_tile_path(z, x, y)
        return tile_path.exists() and tile_path.stat().st_size > 0
    
    def clear_cache(self):
        """Clear in-memory cache."""
        self.tile_cache.clear()


class TilePyramidGenerator:
    """Generate XYZ web tiles from GeoTIFF at multiple zoom levels."""
    
    def __init__(self, tiff_path, output_dir, tile_size=256, max_workers=4):
        """
        Initialize the pyramid generator.
        
        Args:
            tiff_path (str): Path to input GeoTIFF
            output_dir (str): Output directory for tiles
            tile_size (int): Tile size in pixels (default 256)
            max_workers (int): Number of parallel workers
        """
        self.tiff_path = tiff_path
        self.output_dir = output_dir
        self.tile_size = tile_size
        self.max_workers = max_workers
        
        # Cancellation flag for graceful shutdown
        self.should_cancel = False
        self.cancel_lock = threading.Lock()
        
        # Store actual generated zoom range
        self.generated_min_zoom = None
        self.generated_max_zoom = None
        
        # Initialize tile storage
        self.tile_storage = TileStorage(output_dir)
        
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Open source and get metadata
        with rasterio.open(tiff_path) as src:
            self.width = src.width
            self.height = src.height
            self.crs = src.crs
            self.transform = src.transform
            self.dtype = src.dtypes[0]
            self.count = src.count
            self.bounds = src.bounds
            
            print(f"\n📊 Source TIFF Info:")
            print(f"  Size: {self.width}x{self.height} pixels")
            print(f"  Bands: {self.count}")
            print(f"  Data type: {self.dtype}")
            print(f"  CRS: {self.crs}")
            print(f"  Bounds: {self.bounds}")
            
            # Get bounds in lat/lon
            self._get_latlon_bounds()
    
    def _get_latlon_bounds(self):
        """Get lat/lon bounds, reprojecting if necessary."""
        with rasterio.open(self.tiff_path) as src:
            bounds = src.bounds
            crs = src.crs
            
            print(f"Source CRS: {crs}")
            
            # Check if already in Web Mercator (EPSG:3857)
            if crs and (crs.to_epsg() == 3857 or 'EPSG:3857' in crs.to_string()):
                # Already in Web Mercator - bounds are in meters
                self.west, self.south, self.east, self.north = bounds.left, bounds.bottom, bounds.right, bounds.top
                print(f"✓ Source is already EPSG:3857")
                
                # Convert meter bounds back to lat/lon for zoom calculations
                from rasterio.warp import transform_bounds
                try:
                    self.west_latlon, self.south_latlon, self.east_latlon, self.north_latlon = transform_bounds(
                        CRS.from_epsg(3857),
                        CRS.from_epsg(4326),
                        bounds.left, bounds.bottom, bounds.right, bounds.top
                    )
                except:
                    self.west_latlon, self.south_latlon, self.east_latlon, self.north_latlon = bounds.left, bounds.bottom, bounds.right, bounds.top
            
            # Check if in lat/lon (EPSG:4326)
            elif crs and crs.to_epsg() == 4326:
                # In lat/lon - keep lat/lon bounds and convert to Web Mercator meters
                self.west_latlon, self.south_latlon, self.east_latlon, self.north_latlon = bounds.left, bounds.bottom, bounds.right, bounds.top
                
                print(f"Converting EPSG:4326 to EPSG:3857...")
                from rasterio.warp import transform_bounds
                try:
                    self.west, self.south, self.east, self.north = transform_bounds(
                        CRS.from_epsg(4326),
                        CRS.from_epsg(3857),
                        bounds.left, bounds.bottom, bounds.right, bounds.top
                    )
                    print(f"✓ Converted bounds (meters): {self.west:.0f}, {self.south:.0f}, {self.east:.0f}, {self.north:.0f}")
                except Exception as e:
                    print(f"⚠ Warning: Could not convert bounds: {e}")
                    self.west, self.south, self.east, self.north = bounds.left, bounds.bottom, bounds.right, bounds.top
                    self.west_latlon, self.south_latlon, self.east_latlon, self.north_latlon = bounds.left, bounds.bottom, bounds.right, bounds.top
            
            else:
                # Unknown or other projected CRS
                print(f"⚠ Warning: CRS is {crs}, not EPSG:3857 or EPSG:4326")
                print(f"   For best results, reproject your raster to EPSG:3857")
                self.west, self.south, self.east, self.north = bounds.left, bounds.bottom, bounds.right, bounds.top
                self.west_latlon, self.south_latlon, self.east_latlon, self.north_latlon = bounds.left, bounds.bottom, bounds.right, bounds.top
        
        # Calculate zoom levels (using lat/lon bounds)
        self.min_zoom = self._calculate_min_zoom()
        self.max_zoom = self._calculate_max_zoom()
        
        print(f"\n📈 Calculated Zoom Range: {self.min_zoom} - {self.max_zoom}")
    
    def set_cancel_flag(self):
        """Signal cancellation to the generator."""
        with self.cancel_lock:
            self.should_cancel = True
    
    def check_cancel(self):
        """Check if cancellation was requested."""
        with self.cancel_lock:
            return self.should_cancel
    
    def reset_cancel_flag(self):
        """Reset cancellation flag."""
        with self.cancel_lock:
            self.should_cancel = False
    
    def _calculate_min_zoom(self):
        """Calculate minimum zoom level where raster has meaningful coverage."""
        for z in range(0, 28):
            min_x, max_y = WebMercator.lat_lon_to_tile(z, self.north_latlon, self.west_latlon)
            max_x, min_y = WebMercator.lat_lon_to_tile(z, self.south_latlon, self.east_latlon)
            
            if (max_x - min_x) <= 2 and (max_y - min_y) <= 2:
                continue
            else:
                return max(0, z - 1)
        return 0  # Allow zoom 0 as absolute minimum
    
    def _calculate_max_zoom(self):
        """Calculate maximum zoom level (default to 20 for better detail)."""
        return min(24, 20)
    
    def estimate_tile_count(self, min_zoom=None, max_zoom=None):
        """Estimate total number of tiles to generate."""
        if min_zoom is None:
            min_zoom = self.min_zoom
        if max_zoom is None:
            max_zoom = self.max_zoom
        
        total_tiles = 0
        
        for z in range(min_zoom, max_zoom + 1):
            min_x, max_y = WebMercator.lat_lon_to_tile(z, self.north_latlon, self.west_latlon)
            max_x, min_y = WebMercator.lat_lon_to_tile(z, self.south_latlon, self.east_latlon)
            
            # Fix inverted Y coordinates
            if min_y > max_y:
                min_y, max_y = max_y, min_y
            if min_x > max_x:
                min_x, max_x = max_x, min_x
            
            tiles_in_level = (max_x - min_x + 1) * (max_y - min_y + 1)
            total_tiles += tiles_in_level
        
        return total_tiles
    
    def _read_tile_data(self, z, x, y):
        """
        Read raster data for a single XYZ tile.
        Handles both EPSG:4326 and EPSG:3857 source rasters.
        """
        try:
            # Calculate tile bounds in Web Mercator meters
            resolution = (2 * math.pi * WebMercator.EARTH_RADIUS) / (256 * (2 ** z))
            
            tile_minx = x * 256 * resolution - math.pi * WebMercator.EARTH_RADIUS
            tile_maxx = (x + 1) * 256 * resolution - math.pi * WebMercator.EARTH_RADIUS
            tile_maxy = math.pi * WebMercator.EARTH_RADIUS - y * 256 * resolution
            tile_miny = math.pi * WebMercator.EARTH_RADIUS - (y + 1) * 256 * resolution
            
            with rasterio.open(self.tiff_path) as src:
                # Determine source CRS and convert tile bounds if needed
                source_crs = src.crs
                read_minx, read_miny, read_maxx, read_maxy = tile_minx, tile_miny, tile_maxx, tile_maxy
                
                # If source is in lat/lon (EPSG:4326), convert tile bounds from meters to lat/lon
                if source_crs and source_crs.to_epsg() == 4326:
                    from rasterio.warp import transform_bounds
                    read_minx, read_miny, read_maxx, read_maxy = transform_bounds(
                        CRS.from_epsg(3857),
                        CRS.from_epsg(4326),
                        tile_minx, tile_miny, tile_maxx, tile_maxy
                    )
                
                # Fast reject: check if tile bounds intersect with source
                if (
                    read_maxx <= src.bounds.left or
                    read_minx >= src.bounds.right or
                    read_maxy <= src.bounds.bottom or
                    read_miny >= src.bounds.top
                ):
                    return None
                
                # Compute pixel window from geographic bounds
                window = rasterio.windows.from_bounds(
                    read_minx,
                    read_miny,
                    read_maxx,
                    read_maxy,
                    transform=src.transform
                )
                
                # Read data with boundless=True to handle edge tiles
                # masked=True will handle NoData values properly
                data = src.read(
                    window=window,
                    boundless=True,
                    fill_value=0,
                    masked=False  # Don't use masked arrays, just fill with 0
                )
                
                # Replace NoData with 0
                for i in range(data.shape[0]):
                    if src.nodata is not None:
                        # Handle uint16 with nodata values outside range
                        if data.dtype == np.uint16:
                            # For uint16, values > 65535 wrapped, so check for unlikely values
                            data[i] = np.where(data[i] >= 65535, 0, data[i])
                        else:
                            data[i] = np.where(data[i] == src.nodata, 0, data[i])
                
                # Check if tile has any real data
                if not np.any(data):
                    return None
                
                return data
        
        except Exception as e:
            print(f"Tile read error z={z} x={x} y={y}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _read_tile_direct(self, bounds):
        """Read tile data directly when source and destination CRS match."""
        from rasterio.windows import from_bounds
        
        try:
            with rasterio.open(self.tiff_path) as src:
                window = from_bounds(
                    bounds['west'], 
                    bounds['south'],
                    bounds['east'], 
                    bounds['north'],
                    src.transform
                )
                
                data = src.read(
                    window=window,
                    out_shape=(self.src_count, self.tile_size, self.tile_size)
                )
            
            return data
        except Exception as e:
            return None
    
    def _read_and_reproject_tile(self, bounds):
        """Read and reproject tile data when CRS conversion is needed."""
        try:
            with rasterio.open(self.tiff_path) as src:
                # Read entire raster
                data = src.read()
                
                # Create destination array
                dst_data = np.zeros(
                    (self.src_count, self.tile_size, self.tile_size),
                    dtype=data.dtype
                )
                
                # Reproject to tile bounds
                dst_transform = from_bounds(
                    bounds['west'],
                    bounds['south'],
                    bounds['east'],
                    bounds['north'],
                    self.tile_size,
                    self.tile_size
                )
                
                reproject(
                    data,
                    dst_data,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=CRS.from_epsg(3857),
                    resampling=Resampling.bilinear
                )
            
            return dst_data
        except Exception as e:
            return None
    
    def _render_tile(self, z, x, y):
        """
        Render a single tile using geospatial reprojection (not stretching).
        This ensures perfect alignment with the XYZ tile grid.
        Supports cancellation via check_cancel().
        """
        # Check cancellation BEFORE starting
        if self.check_cancel():
            return None
        
        try:
            # Calculate tile bounds in Web Mercator meters
            resolution = (2 * math.pi * WebMercator.EARTH_RADIUS) / (256 * (2 ** z))
            
            tile_minx = x * 256 * resolution - math.pi * WebMercator.EARTH_RADIUS
            tile_maxx = (x + 1) * 256 * resolution - math.pi * WebMercator.EARTH_RADIUS
            tile_maxy = math.pi * WebMercator.EARTH_RADIUS - y * 256 * resolution
            tile_miny = math.pi * WebMercator.EARTH_RADIUS - (y + 1) * 256 * resolution
            
            if z <= 7:
                print(f"  [Z{z} {x}/{y}] Tile bounds (Web Mercator meters): ({tile_minx:.0f}, {tile_miny:.0f}, {tile_maxx:.0f}, {tile_maxy:.0f})")
            
            with rasterio.open(self.tiff_path) as src:
                source_crs = src.crs
                
                # Determine source tile bounds (convert to source CRS if needed)
                read_minx, read_miny, read_maxx, read_maxy = tile_minx, tile_miny, tile_maxx, tile_maxy
                
                if source_crs and source_crs.to_epsg() == 4326:
                    from rasterio.warp import transform_bounds
                    read_minx, read_miny, read_maxx, read_maxy = transform_bounds(
                        CRS.from_epsg(3857),
                        CRS.from_epsg(4326),
                        tile_minx, tile_miny, tile_maxx, tile_maxy
                    )
                
                # Fast reject: check if tile bounds intersect with source
                # IMPORTANT: Only apply at higher zooms (z >= 8)
                # At low zooms (0-7), tiles are too large relative to raster footprint
                # Bounding box intersection is unreliable there - let reprojection handle it
                source_bounds = src.bounds
                
                # Check intersection: tiles must overlap in BOTH dimensions
                # For EPSG:4326 (lat/lon), remember that y increases northward
                # source_bounds.top is north (max latitude), source_bounds.bottom is south (min latitude)
                no_intersection = (
                    read_maxx <= source_bounds.left or      # tile is entirely west of source
                    read_minx >= source_bounds.right or     # tile is entirely east of source
                    read_maxy <= source_bounds.bottom or    # tile is entirely south of source
                    read_miny >= source_bounds.top          # tile is entirely north of source
                )
                
                # DEBUG: Log for low zoom levels
                if z <= 7:
                    print(f"  [Z{z} {x}/{y}] Web Mercator tile: ({tile_minx:.0f}, {tile_miny:.0f}, {tile_maxx:.0f}, {tile_maxy:.0f})")
                    print(f"  [Z{z} {x}/{y}] Converted to {source_crs}: ({read_minx:.6f}, {read_miny:.6f}, {read_maxx:.6f}, {read_maxy:.6f})")
                    print(f"  [Z{z} {x}/{y}] Source bounds: ({source_bounds.left:.6f}, {source_bounds.bottom:.6f}, {source_bounds.right:.6f}, {source_bounds.top:.6f})")
                    print(f"  [Z{z} {x}/{y}] Longitude: tile=[{read_minx:.6f}, {read_maxx:.6f}], source=[{source_bounds.left:.6f}, {source_bounds.right:.6f}]")
                    print(f"  [Z{z} {x}/{y}] Latitude:  tile=[{read_miny:.6f}, {read_maxy:.6f}], source=[{source_bounds.bottom:.6f}, {source_bounds.top:.6f}]")
                    print(f"  [Z{z} {x}/{y}] Intersection check result: {no_intersection}")
                    print(f"  [Z{z} {x}/{y}] Early rejection enabled for z >= 8, z={z} will proceed to reprojection")
                
                # Only early-reject at higher zooms (z >= 8)
                # Low zoom tiles must be rendered - NaN mask will handle empty areas
                if no_intersection and z >= 8:
                    if z <= 7:
                        print(f"  [Z{z} {x}/{y}] ✗ REJECTED (high-zoom, no intersection)")
                    return None
                
                # Create destination transform for the tile in Web Mercator (256x256 pixels)
                # The tile bounds are in Web Mercator meters, so use those for the transform
                tile_transform = from_bounds(
                    tile_minx, tile_miny, tile_maxx, tile_maxy,
                    self.tile_size, self.tile_size
                )
                
                # Create output array for reprojection with NaN initialization
                # NaN marks "no data" areas (outside raster), not color values
                # This allows us to distinguish between real black pixels (0,0,0) and empty areas
                dst_array = np.full(
                    (src.count, self.tile_size, self.tile_size),
                    np.nan,
                    dtype=np.float32
                )
                
                # Flag to track if we got any real data
                has_data = False
                
                # Reproject each band into the tile
                for band_idx in range(1, src.count + 1):
                    band_data = src.read(band_idx, masked=False)
                    
                    # Stretch this band to 0-255
                    stretched = self._stretch_band(band_data.astype(np.float32))
                    
                    if z <= 7:
                        print(f"  [Z{z} {x}/{y}] Band {band_idx}: read {stretched.shape}, stretching...")
                    
                    # Use nearest neighbor for ultra-low zoom (z <= 2) to preserve tiny rasters
                    # Use bilinear for higher zoom for smoother results
                    resampling_method = Resampling.nearest if z <= 2 else Resampling.bilinear
                    
                    # Reproject stretched band to Web Mercator tile coordinates
                    try:
                        reproject(
                            source=stretched,
                            destination=dst_array[band_idx - 1],
                            src_transform=src.transform,
                            src_crs=source_crs,
                            dst_transform=tile_transform,
                            dst_crs=CRS.from_epsg(3857),  # Always reproject to Web Mercator for XYZ tiles
                            resampling=resampling_method,  # Nearest at z<=2, bilinear at z>2
                            dst_nodata=np.nan  # Mark pixels outside raster as NaN (not 0!)
                        )
                        valid_pixels = ~np.isnan(dst_array[band_idx - 1])
                        if np.any(valid_pixels):
                            has_data = True
                            if z <= 7:
                                print(f"  [Z{z} {x}/{y}] Band {band_idx}: ✓ Reprojected ({resampling_method.name}), {np.count_nonzero(valid_pixels)} valid pixels")
                        else:
                            if z <= 7:
                                print(f"  [Z{z} {x}/{y}] Band {band_idx}: ⊘ No valid pixels after reproject (all NaN)")
                    except Exception as reproject_error:
                        print(f"  ⚠ Reproject error band {band_idx} z={z} x={x} y={y}: {reproject_error}")
                        import traceback
                        traceback.print_exc()
                        # Continue to next band
                
                if z <= 7:
                    print(f"  [Z{z} {x}/{y}] After all bands: has_data={has_data}")
                
                if not has_data:
                    if z <= 10:
                        print(f"  [Z{z} {x}/{y}] ✗ SKIP: No data from any band (0% coverage after reprojection)")
                        print(f"         → This is normal at low-zoom: tile might be too large relative to raster")
                    return None
                
                # CRITICAL: Create alpha mask from NaN values BEFORE converting to uint8
                # This is the key difference - alpha is based on the mask, NOT on RGB values
                # This allows real black pixels (0,0,0) to be preserved
                valid_mask = ~np.isnan(dst_array[0])  # True where data exists
                alpha = valid_mask.astype(np.uint8) * 255  # 255 where valid, 0 where NaN
                
                # Check if tile has ANY data at all
                # Create tiles at all zoom levels if there's any valid data
                # (even small tiles at low zoom levels are important for map continuity)
                coverage_ratio = np.count_nonzero(alpha) / alpha.size
                if coverage_ratio == 0:  # Completely empty - no data at all
                    return None
                
                # Log sparse tiles for debugging
                if coverage_ratio < 0.01 and coverage_ratio > 0:
                    print(f"  ✓ Sparse z={z} x={x} y={y} ({coverage_ratio*100:.4f}% coverage) → Created anyway (data found)")
                    print(f"     Comparison: Z6 got 0% (skipped), this Z{z} got {coverage_ratio*100:.4f}% (created)")
                
                # Check cancellation BEFORE saving
                if self.check_cancel():
                    return None
                
                # Now safely convert NaN -> 0 for the RGB channels
                dst_array = np.nan_to_num(dst_array, nan=0.0)
                dst_array = np.clip(dst_array, 0, 255).astype(np.uint8)
                
                # Get RGB from converted data
                rgb = dst_array[:3] if dst_array.shape[0] >= 3 else np.tile(dst_array[0], (3, 1, 1))
                
                # Stack into RGBA
                rgba_array = np.dstack([rgb[0], rgb[1], rgb[2], alpha])
                
                # Convert to image
                img = self._array_to_image_uint8(rgba_array)
                
                if img is None:
                    return None
                
                # Save using tile storage (with verification)
                result = self.tile_storage.save_tile(z, x, y, img)
                return result
        
        except Exception as e:
            if not self.check_cancel():  # Only print errors if not cancelling
                print(f"  ✗ Render error z={z} x={x} y={y}: {e}")
            return None
    
    def _array_to_image_uint8(self, arr):
        """Convert uint8 numpy array to PIL Image."""
        try:
            # Handle RGBA format (H, W, 4) - most common for tiles with transparency
            if len(arr.shape) == 3 and arr.shape[2] == 4:
                return Image.fromarray(arr.astype(np.uint8), mode='RGBA')
            # Handle RGB format (H, W, 3)
            elif len(arr.shape) == 3 and arr.shape[2] == 3:
                return Image.fromarray(arr.astype(np.uint8), mode='RGB')
            # Handle old format (bands, H, W)
            elif arr.shape[0] == 1:
                # Single band - grayscale
                return Image.fromarray(arr[0], mode='L')
            elif arr.shape[0] == 2:
                # Two bands - use first as grayscale
                return Image.fromarray(arr[0], mode='L')
            elif arr.shape[0] >= 3:
                # Three or more bands - use as RGB
                arr_rgb = np.stack([arr[0], arr[1], arr[2]], axis=2).astype(np.uint8)
                return Image.fromarray(arr_rgb, mode='RGB')
            else:
                return None
        except Exception as e:
            print(f"Error in _array_to_image_uint8: {e}")
            return None
    
    def _stretch_band(self, band_data):
        """Stretch a single band from its native range to 0-255."""
        # Clip to valid range first
        band_data = np.clip(band_data, 0, 65535)
        
        # Use percentiles to avoid extreme outliers
        min_val = np.percentile(band_data, 2)
        max_val = np.percentile(band_data, 98)
        
        if max_val <= min_val:
            # No variation - return middle gray
            return np.full_like(band_data, 128, dtype=np.uint8)
        
        # Linear stretch: map [min_val, max_val] -> [0, 255]
        stretched = (band_data - min_val) / (max_val - min_val) * 255
        stretched = np.clip(stretched, 0, 255)
        
        return stretched.astype(np.uint8)
    
    def _resize_data(self, data, width, height):
        """DEPRECATED - use reprojection-based rendering instead."""
        raise NotImplementedError("Use the new reprojection-based _render_tile() instead")
    
    def _array_to_image(self, arr):
        """Convert numpy array to PIL Image."""
        if len(arr.shape) == 2:
            return Image.fromarray(arr, mode='L')
        elif arr.shape[2] == 3:
            return Image.fromarray(arr.astype(np.uint8), mode='RGB')
        elif arr.shape[2] == 4:
            return Image.fromarray(arr.astype(np.uint8), mode='RGBA')
        else:
            return Image.fromarray(arr[:, :, 0].astype(np.uint8), mode='L')
    
    def generate(self, min_zoom=None, max_zoom=None, progress_callback=None):
        """
        Generate tile pyramid with proper cancellation support.
        
        Args:
            min_zoom: Minimum zoom level
            max_zoom: Maximum zoom level
            progress_callback: Progress callback function
        """
        # Reset cancel flag at start
        self.reset_cancel_flag()
        
        if min_zoom is None:
            min_zoom = self.min_zoom
        if max_zoom is None:
            max_zoom = self.max_zoom
        
        # Store the actual zoom range that will be generated
        self.generated_min_zoom = min_zoom
        self.generated_max_zoom = max_zoom
        
        print(f"\n🚀 Generating pyramid: zoom {min_zoom}-{max_zoom}")
        
        total_tiles = 0
        tiles_created = 0
        tiles_failed = 0
        
        # Count tiles
        for z in range(min_zoom, max_zoom + 1):
            min_x, max_y = WebMercator.lat_lon_to_tile(z, self.north_latlon, self.west_latlon)
            max_x, min_y = WebMercator.lat_lon_to_tile(z, self.south_latlon, self.east_latlon)
            
            # Fix inverted Y coordinates (north/south produces backwards Y)
            if min_y > max_y:
                min_y, max_y = max_y, min_y
            if min_x > max_x:
                min_x, max_x = max_x, min_x
            
            tiles_in_level = (max_x - min_x + 1) * (max_y - min_y + 1)
            total_tiles += tiles_in_level
        
        print(f"📊 Total tiles to create: {total_tiles}\n")
        
        # Generate tiles with proper cancellation support
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for z in range(min_zoom, max_zoom + 1):
                # Check cancellation before processing zoom level
                if self.check_cancel():
                    print(f"\n[CANCEL] Cancellation detected at zoom level {z}")
                    executor.shutdown(wait=False, cancel_futures=True)
                    break
                
                min_x, max_y = WebMercator.lat_lon_to_tile(z, self.north_latlon, self.west_latlon)
                max_x, min_y = WebMercator.lat_lon_to_tile(z, self.south_latlon, self.east_latlon)
                
                # Fix inverted coordinates
                if min_y > max_y:
                    min_y, max_y = max_y, min_y
                if min_x > max_x:
                    min_x, max_x = max_x, min_x
                
                tiles_at_zoom = (max_x - min_x + 1) * (max_y - min_y + 1)
                print(f"  Zoom {z}: Tiles x=[{min_x}-{max_x}] y=[{min_y}-{max_y}] ({tiles_at_zoom} total)")
                
                # Create futures dict for this zoom level
                futures = {}
                
                for x in range(min_x, max_x + 1):
                    for y in range(min_y, max_y + 1):
                        future = executor.submit(self._render_tile, z, x, y)
                        futures[future] = (z, x, y)
                
                # Process as they complete for this zoom level
                for future in as_completed(futures):
                    # Check cancellation while processing
                    if self.check_cancel():
                        print(f"\n[CANCEL] Cancellation detected during tile processing")
                        executor.shutdown(wait=False, cancel_futures=True)
                        break
                    
                    try:
                        result = future.result()
                        if result:
                            tiles_created += 1
                        else:
                            tiles_failed += 1
                    except Exception as e:
                        tiles_failed += 1
                        if not self.check_cancel():
                            print(f"  ✗ Tile error: {e}")
                    
                    if progress_callback:
                        progress_callback(tiles_created, total_tiles)
                    
                    if (tiles_created + tiles_failed) % max(1, total_tiles // 20) == 0:
                        pct = int(((tiles_created + tiles_failed) / total_tiles) * 100)
                        print(f"  Progress: {tiles_created + tiles_failed}/{total_tiles} ({pct}%) - Created: {tiles_created}, Failed: {tiles_failed}", end='\r')
                
                if self.check_cancel():
                    break
        
        print(f"\n✓ Created {tiles_created} tiles, {tiles_failed} failed in {self.output_dir}")
        return tiles_created
    
    def generate_metadata(self):
        """Generate metadata JSON for tile set."""
        # Use actual generated zoom range if available, otherwise use calculated
        min_zoom = self.generated_min_zoom if self.generated_min_zoom is not None else self.min_zoom
        max_zoom = self.generated_max_zoom if self.generated_max_zoom is not None else self.max_zoom
        
        metadata = {
            'name': Path(self.tiff_path).stem,
            'bounds': [self.west_latlon, self.south_latlon, self.east_latlon, self.north_latlon],
            'center': [(self.west_latlon + self.east_latlon) / 2, (self.south_latlon + self.north_latlon) / 2],
            'minZoom': min_zoom,
            'maxZoom': max_zoom,
            'tileSize': self.tile_size,
            'tiles': [f'{{baseUrl}}/{{z}}/{{x}}/{{y}}.png']
        }
        return metadata


if __name__ == "__main__":
    input_tiff = "res.tif"
    output_dir = "./tiles/"
    
    generator = TilePyramidGenerator(input_tiff, output_dir, tile_size=256, max_workers=4)
    generator.generate()
    
    metadata = generator.generate_metadata()
    print(f"\nMetadata: {metadata}")