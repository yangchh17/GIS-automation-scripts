[README.md](https://github.com/user-attachments/files/27452017/README.md)
# GIS Automation Scripts

This repository is a home for GIS-focused automation scripts, notebooks, and helper tools used to speed up repeatable mapping and data-management tasks.

The goal is to keep the repository focused on code and workflow, not on project-specific data. Scripts in this repo should be reusable, documented, and safe to share without exposing internal folder structures, network drives, client names, or local machine details.

## Script Overview: AutoDataImporter

`AutoDataImporter.ipynb` is an ArcGIS Pro / ArcPy notebook for **bulk-importing spatial data files from a folder tree into the current ArcGIS Pro project**, while organizing the imported layers by project folder.

In plain language, the notebook does this:

1. Reads a root input folder that contains multiple project subfolders.
2. Scans that folder tree for supported spatial files:
   - `.kml`
   - `.kmz`
   - `.shp`
3. Ignores non-import targets such as PDFs and ArcGIS layer/project files.
4. Groups discovered spatial files by their top-level subfolder.
5. Creates or reuses a scratch file geodatabase.
6. Connects to the **current ArcGIS Pro project** and active map.
7. Creates one group layer per project folder.
8. Adds each supported dataset into the map and places it into the matching group.
9. Prints an import summary showing successes and failures.
10. Generates a simple PDF inventory report by project folder.

## What Problem This Solves

This notebook is useful when you receive many GIS deliverables spread across many folders and need to bring them into ArcGIS Pro in a structured way without manually dragging files into the map one at a time.

It helps with:

- batch import of external spatial files
- organizing layers by source folder or project name
- keeping imports consistent
- quickly seeing what was processed
- checking for supporting PDF documents in the same directory tree

## Requirements

This workflow is designed for:

- ArcGIS Pro
- ArcPy
- a project open in ArcGIS Pro
- a current/active map available in that project

Because the notebook uses `arcpy.mp.ArcGISProject("CURRENT")`, it is meant to run **inside ArcGIS Pro**, not as a generic standalone Python script.

## Configuration

The main values to update before running are:

```python
ROOT_FOLDER = r"C:\path\to\input_folder"
SCRATCH_GDB = r"C:\path\to\scratch.gdb"
```

- `ROOT_FOLDER` is the parent folder containing the project subfolders to scan
- `SCRATCH_GDB` is the file geodatabase used for temporary or imported content

Supported file types:

```python
SUPPORTED_EXTENSIONS = {".kml", ".kmz", ".shp"}
SKIP_EXTENSIONS = {".pdf", ".lyr", ".lyrx", ".mxd", ".aprx"}
```

## Typical Workflow

1. Open ArcGIS Pro.
2. Open the target ArcGIS Pro project.
3. Open the notebook in ArcGIS Pro.
4. Update `ROOT_FOLDER` and `SCRATCH_GDB`.
5. Run the notebook cells in order.
6. Review the grouped layers created in the active map.
7. Review the printed import summary and any failed files.

## Expected Output

After a successful run, you should expect:

- spatial files added to the active map
- group layers created for each detected top-level folder
- imported content organized beneath those groups
- a summary of imported and failed items
- a list of PDF files found in the same folder structure

