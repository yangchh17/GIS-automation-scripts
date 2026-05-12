"""
Batch land use segmentation pipeline.
Processes all tiles in INPUT_DIR, then merges results per class.
"""

import os
import sys
import time
import glob
import traceback
import numpy as np
import rasterio
from rasterio.enums import Resampling
from samgeo.text_sam import LangSAM
from samgeo import raster_to_vector
import geopandas as gpd
import pandas as pd

# === CONFIG ===
INPUT_DIR  = r"path/to/input/tiles"   # Directory containing input .tif tiles
OUTPUT_DIR = r"path/to/output"        # Directory for output files

CLASSES = [
    ("water",         "water . pond . lake . river . ocean . shore water",                      0.22, 0.22, 4, (30, 144, 255)),   # blue
    ("forest",        "tree . forest . woodland . dense vegetation",                            0.28, 0.28, 6, (34, 139, 34)),    # green
    ("building",      "building",                                                               0.24, 0.24, 1, (220, 20, 60)),    # crimson
    ("ground_objects","car . truck . vehicle . van . pickup truck . sheds . trampoline . trash bin . container . tents",
                                                                                                0.22, 0.22, 2, (255, 165, 0)),    # orange
    ("road",          "road . paved road . asphalt road . driveway",                            0.22, 0.22, 3, (128, 128, 128)),  # grey
    ("non_plantable", "beach . sand . gravel pad . bare rock . exposed bedrock . rocky shore",  0.22, 0.22, 5, (210, 180, 140)),  # tan
]
PLANTABLE_ID    = 7
PLANTABLE_COLOR = (255, 235, 100)  # bright yellow — target areas
SIMPLIFY_TOLERANCE = 0.25

# Skip tiles where more than this fraction is nodata/black
BAD_TILE_THRESHOLD = 0.25

os.makedirs(OUTPUT_DIR, exist_ok=True)
log_path = os.path.join(OUTPUT_DIR, "batch_log.txt")
log_file = open(log_path, "w", encoding="utf-8")

def log(msg):
    print(msg)
    log_file.write(msg + "\n")
    log_file.flush()

def is_bad_tile(tif_path):
    """Return True if tile is mostly nodata/black."""
    try:
        with rasterio.open(tif_path) as src:
            sample = src.read(1, out_shape=(256, 256), resampling=Resampling.average)
            nodata_val = src.nodata
            if nodata_val is not None:
                bad_frac = np.mean(sample == nodata_val)
            else:
                bad_frac = np.mean(sample == 0)
            return bad_frac > BAD_TILE_THRESHOLD
    except Exception:
        return True

def process_tile(tile_path, tile_out_dir, sam):
    """Run full pipeline on one tile. Returns dict of class -> vector path."""
    os.makedirs(tile_out_dir, exist_ok=True)

    with rasterio.open(tile_path) as src:
        src_transform = src.transform
        src_crs       = src.crs
        src_w, src_h  = src.width, src.height

    label_map = np.zeros((src_h, src_w), dtype=np.uint8)

    for class_name, prompt, box_thr, text_thr, class_id, _ in CLASSES:
        mask_path = os.path.join(tile_out_dir, f"{class_name}_mask.tif")
        try:
            sam.predict(
                tile_path,
                text_prompt=prompt,
                box_threshold=box_thr,
                text_threshold=text_thr,
                output=mask_path,
            )
            with rasterio.open(mask_path) as msrc:
                mask = msrc.read(1, out_shape=(src_h, src_w), resampling=Resampling.nearest)
            detected = (mask > 0) & (label_map == 0)
            label_map[detected] = class_id
        except Exception as e:
            log(f"    !! {class_name} failed: {e}")

    label_map[label_map == 0] = PLANTABLE_ID

    # Save labelled raster
    label_tif_path = os.path.join(tile_out_dir, "landuse_labels.tif")
    profile = {
        "driver": "GTiff", "height": src_h, "width": src_w, "count": 1,
        "dtype": "uint8", "crs": src_crs, "transform": src_transform,
        "compress": "lzw", "nodata": 0,
    }
    with rasterio.open(label_tif_path, "w", **profile) as dst:
        dst.write(label_map, 1)

    # Vectorize each class
    class_vectors = {}
    all_classes = list(CLASSES) + [("plantable", "plantable", 0, 0, PLANTABLE_ID, PLANTABLE_COLOR)]

    for class_name, _p, _b, _t, class_id, _c in all_classes:
        binary = (label_map == class_id).astype(np.uint8) * 255
        binary_path = os.path.join(tile_out_dir, f"{class_name}_binary.tif")
        with rasterio.open(binary_path, "w", **profile) as dst:
            dst.write(binary, 1)

        vector_path = os.path.join(tile_out_dir, f"{class_name}.gpkg")
        try:
            raster_to_vector(binary_path, vector_path, simplify_tolerance=SIMPLIFY_TOLERANCE)
            class_vectors[class_name] = vector_path
        except Exception as e:
            log(f"    !! vectorize {class_name} failed: {e}")

    return class_vectors

# === MAIN ===
log(f"Batch run started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
tiles = sorted(glob.glob(os.path.join(INPUT_DIR, "*.tif")))
log(f"Found {len(tiles)} tiles in input directory")

log("Loading LangSAM (one time)...")
sam = LangSAM()
log("Model loaded.\n")

per_tile_vectors = {cls[0]: [] for cls in CLASSES}
per_tile_vectors["plantable"] = []

start_all = time.time()
skipped   = []
failed    = []

for i, tile_path in enumerate(tiles, 1):
    tile_name = os.path.splitext(os.path.basename(tile_path))[0]
    log(f"\n[{i}/{len(tiles)}] {tile_name}")

    if is_bad_tile(tile_path):
        log(f"  SKIP - mostly nodata/black")
        skipped.append(tile_name)
        continue

    tile_out_dir = os.path.join(OUTPUT_DIR, "per_tile", tile_name)
    t0 = time.time()
    try:
        class_vectors = process_tile(tile_path, tile_out_dir, sam)
        for cname, vpath in class_vectors.items():
            per_tile_vectors[cname].append(vpath)
        log(f"  OK ({time.time() - t0:.1f}s)")
    except Exception as e:
        log(f"  FAIL: {e}")
        log(traceback.format_exc())
        failed.append(tile_name)

# === MERGE ===
log("\n=== Merging per-class vectors ===")
for class_name, vector_paths in per_tile_vectors.items():
    if not vector_paths:
        log(f"  {class_name}: nothing to merge")
        continue
    gdfs = []
    for vp in vector_paths:
        try:
            gdf = gpd.read_file(vp)
            if len(gdf) > 0:
                gdfs.append(gdf)
        except Exception as e:
            log(f"    !! could not read {vp}: {e}")
    if not gdfs:
        log(f"  {class_name}: no valid geometry")
        continue
    merged      = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=gdfs[0].crs)
    merged_path = os.path.join(OUTPUT_DIR, f"merged_{class_name}.gpkg")
    merged.to_file(merged_path, driver="GPKG")
    log(f"  {class_name}: {len(merged)} features -> {merged_path}")

log(f"\nTotal time: {(time.time() - start_all) / 60:.1f} minutes")
log(f"Tiles processed: {len(tiles) - len(skipped) - len(failed)}")
log(f"Skipped (bad): {len(skipped)}")
log(f"Failed: {len(failed)}")
if skipped: log(f"  Skipped tiles: {skipped}")
if failed:  log(f"  Failed tiles: {failed}")
log_file.close()
print(f"\nLog saved: {log_path}")