# GIS Scripts

Small GIS automation repository for ArcGIS Pro, ArcPy, and related data-processing helpers.

This repo is meant to store reusable scripts and notebooks that speed up common GIS tasks such as importing files, organizing layers, and preparing data for mapping.

## Auto Data Importer

[AutoDataImporter.ipynb](https://github.com/yangchh17/GIS-automation-scripts/blob/main/AutoDataImporter.ipynb) is a notebook that scans a folder for spatial files like `.kml`, `.kmz`, and `.shp`, then adds them into the current ArcGIS Pro map.

It groups imported layers by folder, uses a scratch geodatabase, and prints a summary of what was added or failed.

Before running it, update the input folder path and scratch geodatabase path in the configuration cell.

## PDF to TIFF Converter

[PDFtoTIFF.ipynb](https://github.com/yangchh17/GIS-automation-scripts/blob/main/PDFtoTIFF.ipynb) is a notebook that converts single-page PDFs to TIFF using `arcpy.conversion.PDFToTIFF`.
 
 Configure the input and output folders in the configuration cell, or set these environment variables before running:
 - `PDF_TO_TIFF_INPUT_DIR`
 - `PDF_TO_TIFF_OUTPUT_DIR`
 - `PDF_TO_TIFF_DPI`

## Batch Land Use Segmentation
[landuse_langsam_pipiline.py](https://github.com/yangchh17/GIS-automation-scripts/blob/main/landuse_langsam_pipiline.py) A batch pipeline for land use classification of aerial/satellite imagery tiles using LangSAM (Language-Segment-Anything). Outputs per-class vector files (GeoPackage) merged across all tiles.

What It Does
 - `Scans an input directory for .tif tiles`
 - `Skips tiles that are mostly nodata/black`
 - `Runs LangSAM text-prompted segmentation for each land use class`
 - `Assigns any unclassified pixels to a "plantable" class`
 - `Vectorizes results and merges them per class into final GeoPackages`


## Dominant categorical-raster class per polygon

Summarises a single-band integer (categorical) raster over a set of polygons.
Set up for the DataBC Soil Parent Material raster (`HaBC_PM.tif`), but works for
any categorical raster if `CLASS_NAMES` matches your class codes. Assumes a
projected CRS in metres.

Outputs (written to `output/`):

- **`dominant_class.csv`** — dominant class per polygon (modal by pixel count;
  nearest-pixel inference for empty/sparse polygons).
- **`class_composition.csv`** — full area-weighted class breakdown per polygon.
- **`class_top3.csv`** — wide table, top 3 classes per polygon.

### Usage

```bash
pip install numpy pandas geopandas rasterio shapely scipy
python dominant_class_per_polygon.py
```

Edit the config block at the top first: `RASTER_PATH`, `VECTOR_PATH`, `ID_FIELD`
(unique polygon label), optional `NUM_FIELD`, and `CLASS_NAMES`. The file is
structured as `# %%` cells, so it also runs cell-by-cell in VS Code or Jupyter.
