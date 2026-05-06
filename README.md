# GIS Scripts

Small GIS automation repository for ArcGIS Pro, ArcPy, and related data-processing helpers.

This repo is meant to store reusable scripts and notebooks that speed up common GIS tasks such as importing files, organizing layers, and preparing data for mapping.

## AutoDataImporter

`AutoDataImporter.ipynb` is a notebook that scans a folder for spatial files like `.kml`, `.kmz`, and `.shp`, then adds them into the current ArcGIS Pro map.

It groups imported layers by folder, uses a scratch geodatabase, and prints a summary of what was added or failed.

Before running it, update the input folder path and scratch geodatabase path in the configuration cell.

