# XYZ Tile Pyramid Viewer

A local web app for uploading GeoTIFF files, generating XYZ PNG tile pyramids, and previewing the result on an interactive Leaflet map.

The app is designed for geospatial raster inspection: upload a `.tif` or `.tiff`, choose zoom levels, generate web map tiles, then compare the raster against street or satellite basemaps with an opacity slider.

## Features

- Upload GeoTIFF files up to 500 MB.
- Generate XYZ tile pyramids as `{z}/{x}/{y}.png`.
- Preview generated rasters in a Leaflet map.
- Switch between Street and Satellite basemaps.
- Adjust raster opacity for alignment checks.
- Choose custom min/max zoom levels before generation.
- Estimate tile counts before generating.
- Cancel generation while it is running.
- Resume safely by skipping existing generated tiles.
- Supports common georeferenced CRSs through Rasterio/GDAL reprojection to Web Mercator.

## Project Structure

```text
.
|-- start_server.py              # Flask web server and API routes
|-- tile_pyramid_generator.py    # GeoTIFF to XYZ tile generation logic
|-- templates/
|   |-- index_xyz.html           # Main upload/generation/viewer UI
|   `-- tile_inspector.html      # Tile inspection page
|-- uploads/                     # Uploaded GeoTIFF files
|-- tiles/                       # Generated tile pyramids
`-- PERFORMANCE_OPTIMIZATIONS.md # Notes about performance work
```

## Requirements

- Python 3.10 or newer
- Rasterio with GDAL/PROJ support
- Flask
- Flask-CORS
- NumPy
- Pillow

Optional GPU-related packages are detected if installed, but the main bottleneck is GDAL reprojection and tile I/O, so GPU acceleration is not required.

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install flask flask-cors rasterio numpy pillow
```

If Rasterio installation fails on Windows, using Conda is often easier:

```bash
conda install -c conda-forge rasterio flask flask-cors numpy pillow
```

## Run

```bash
python start_server.py
```

Open:

```text
http://localhost:2000
```

The server listens on port `2000` by default.

## Usage

1. Open the app in your browser.
2. Upload a `.tif` or `.tiff` file.
3. Review or change the min/max zoom values.
4. Generate the tile pyramid.
5. Click `View` after generation completes.
6. Use `Street` / `Satellite` and the raster opacity slider to verify alignment.

Generated tiles are stored in:

```text
tiles/<raster-name>/<z>/<x>/<y>.png
```

Each generated pyramid also includes:

```text
tiles/<raster-name>/metadata.json
```

## CRS Support

The generator supports rasters with a valid CRS that Rasterio/GDAL can transform to:

- EPSG:4326 for tile coordinate calculations
- EPSG:3857 for Web Mercator XYZ tile rendering

This includes normal projected rasters such as UTM GeoTIFFs, for example `EPSG:26911`.

The generator cannot reliably process rasters with:

- no CRS
- broken CRS metadata
- custom CRSs that PROJ cannot transform
- ungeoreferenced image-only TIFFs

## Performance Notes

The current renderer uses thread-local Rasterio handles and `WarpedVRT` so each worker can read the relevant Web Mercator tile window instead of reopening and reprojecting the whole raster for every tile.

For very large rasters or many zoom levels, performance can still be limited by:

- GDAL reprojection
- PNG encoding
- many small tile writes
- disk I/O

For heavier production workflows, consider prewarping and building overviews:

```bash
gdalwarp -t_srs EPSG:3857 input.tif output_3857.tif
gdaladdo -r average output_3857.tif 2 4 8 16
```

## API Overview

Main routes:

- `GET /` - main web UI
- `GET /inspector` - tile inspector UI
- `GET /api/files` - list uploaded files and generated pyramids
- `POST /api/upload` - upload a GeoTIFF
- `POST /api/estimate-tiles` - estimate tile count for a zoom range
- `POST /api/pyramid` - start tile generation
- `GET /api/progress` - check generation progress
- `POST /api/cancel` - cancel active generation
- `GET /tiles/<pyramid>/<z>/<x>/<y>.png` - serve generated tiles
- `GET /api/pyramid-info/<pyramid>` - read generated metadata
- `DELETE /api/delete/<filename>` - delete upload and generated tiles

## Troubleshooting

If the generated raster appears shifted or scaled incorrectly, clear the existing output folder and regenerate. Existing tiles are skipped for resumable generation.

If a raster estimates an impossibly large tile count, check its CRS:

```bash
gdalinfo uploads/your-file.tif
```

or:

```bash
python -c "import rasterio; src=rasterio.open('uploads/your-file.tif'); print(src.crs, src.bounds)"
```

If the Satellite basemap does not appear, check browser internet access. The satellite layer uses Esri World Imagery.

If startup fails on Windows because of console encoding, make sure console output uses ASCII-only text.

## Development Notes

- `uploads/`, `tiles/`, and `__pycache__/` should not be committed.
- Regenerating tiles into an existing folder may skip files that are already present.
- Use a fresh output folder when validating tile alignment changes.
