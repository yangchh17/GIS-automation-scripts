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
A batch pipeline for land use classification of aerial/satellite imagery tiles using LangSAM (Language-Segment-Anything). Outputs per-class vector files (GeoPackage) merged across all tiles.

What It Does
 - `Scans an input directory for .tif tiles`
 - `Skips tiles that are mostly nodata/black`
 - `Runs LangSAM text-prompted segmentation for each land use class`
 - `Assigns any unclassified pixels to a "plantable" class`
 - `Vectorizes results and merges them per class into final GeoPackages`
