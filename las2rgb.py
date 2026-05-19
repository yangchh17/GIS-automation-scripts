"""
las_to_orthophoto_cog.py
========================
Generates a Cloud-Optimized GeoTIFF (COG) RGB orthophoto from a colorized LAS point cloud.
100% pure Python — no CLI subprocess calls required.

Workflow:
  1. Sample RGB max values from the LAS to auto-detect bit depth (8-bit vs 16-bit)
  2. Rasterize Red, Green, Blue bands via PDAL Python bindings (writers.gdal)
  3. Merge three single-band GeoTIFFs into one RGB GeoTIFF via GDAL Python API
  4. Rescale to 8-bit (0-255) using per-band min/max stretch if source is 16-bit
  5. Build overviews and write Cloud-Optimized GeoTIFF via GDAL Python API

Requirements (conda-forge):
  conda install -c conda-forge pdal python-pdal gdal numpy
"""

import argparse
import json
import logging
import shutil
from pathlib import Path

import numpy as np
from osgeo import gdal

gdal.UseExceptions()

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG  <- edit these, or pass --input / --output as CLI args
# ──────────────────────────────────────────────────────────────────────────────
INPUT_LAS = (
    r"Y:\OMVC_GIS_Library\01_Data_Library\01_Physical_Geography"
    r"\03_LiDAR\Stinson_2BT_LiDAR\IR2_HIELLEN\03_PointCloud\cloud_merged.las"
)
OUTPUT_DIR = (
    r"Y:\OMVC_GIS_Library\01_Data_Library\01_Physical_Geography"
    r"\02_RGB_Imagery\Stinson_2BT_RGB"
)
RESOLUTION        = 0.05   # metres (5 cm)
FIRST_RETURN_ONLY = True   # True = first returns only (cleaner surface, less veg noise)
NODATA_VAL        = -9999  # Use -9999 to avoid masking true black pixels (R/G/B = 0)
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── STEP 0: SANITY CHECK — VERIFY LAS HAS DISTINCT RGB ───────────────────────

def verify_rgb_in_las(las_path: str, sample_points: int = 100_000) -> None:
    """
    Quick check that the LAS actually contains distinct R, G, B values.
    If all three bands have identical ranges the colorization is broken upstream.
    """
    import pdal

    log.info("Step 0 — Verifying LAS RGB content (%s sample points) ...", f"{sample_points:,}")

    pipeline = pdal.Pipeline(json.dumps({
        "pipeline": [{"type": "readers.las", "filename": las_path, "count": sample_points}]
    }))
    pipeline.execute()
    arr = pipeline.arrays[0]

    for ch in ("Red", "Green", "Blue"):
        lo, hi = int(arr[ch].min()), int(arr[ch].max())
        log.info("  %s — min: %d  max: %d", ch, lo, hi)

    r_range = (int(arr["Red"].min()),   int(arr["Red"].max()))
    g_range = (int(arr["Green"].min()), int(arr["Green"].max()))
    b_range = (int(arr["Blue"].min()),  int(arr["Blue"].max()))

    if r_range == g_range == b_range:
        log.warning(
            "  WARNING: R, G, B all have identical ranges %s. "
            "The point cloud may not be properly colorized — output will look greyscale.",
            r_range,
        )
    else:
        log.info("  -> Bands have distinct ranges. RGB colorization looks valid.")


# ── STEP 1: AUTO-DETECT RGB BIT DEPTH ────────────────────────────────────────

def detect_rgb_scale(las_path: str, sample_points: int = 500_000) -> int:
    """
    Sample up to `sample_points` from the LAS and check max RGB values.
    Returns 256 if data is 16-bit (0-65535), or 1 if already 8-bit (0-255).
    """
    import pdal

    log.info("Step 1 — Sampling %s points to detect RGB bit depth ...", f"{sample_points:,}")

    pipeline = pdal.Pipeline(json.dumps({
        "pipeline": [{"type": "readers.las", "filename": las_path, "count": sample_points}]
    }))
    pipeline.execute()
    arrays = pipeline.arrays
    if not arrays:
        raise RuntimeError("PDAL returned no data during sampling — check your LAS path.")

    arr = arrays[0]
    max_r = int(arr["Red"].max())
    max_g = int(arr["Green"].max())
    max_b = int(arr["Blue"].max())
    overall_max = max(max_r, max_g, max_b)

    log.info("  Sampled RGB max -> R: %d  G: %d  B: %d", max_r, max_g, max_b)

    if overall_max > 255:
        log.info("  -> 16-bit RGB detected (max %d). Will scale to 8-bit after merge.", overall_max)
        return 256
    else:
        log.info("  -> 8-bit RGB detected (max <= 255). No rescaling needed.")
        return 1


# ── STEP 2: RASTERIZE BANDS VIA PDAL PYTHON BINDINGS ─────────────────────────

def rasterize_band(las_path: str, out_tif: str, dimension: str,
                   resolution: float, first_return_only: bool,
                   nodata_val: float = -9999) -> None:
    """
    Use PDAL Python bindings to rasterize one RGB band to a single-band GeoTIFF.
    writers.gdal streams the LAS in chunks — safe for large files.
    nodata_val=-9999 (not 0) so true black pixels are not masked out.
    """
    import pdal

    log.info("  Rasterizing %s band -> %s", dimension, Path(out_tif).name)

    stages = [{"type": "readers.las", "filename": las_path}]

    if first_return_only:
        stages.append({"type": "filters.range", "limits": "ReturnNumber[1:1]"})

    stages.append({
        "type":        "writers.gdal",
        "filename":    out_tif,
        "dimension":   dimension,
        "output_type": "mean",
        "resolution":  resolution,
        "gdalopts":    "COMPRESS=DEFLATE,TILED=YES,BIGTIFF=YES",
        "nodata":      nodata_val,
    })

    pipeline = pdal.Pipeline(json.dumps({"pipeline": stages}))
    pipeline.execute()
    log.info("  -> %s band done.", dimension)


def check_band_stats(tif_path: str, label: str, nodata_val: float = -9999) -> None:
    """Log per-band stats after rasterization to verify bands are distinct."""
    ds   = gdal.Open(tif_path, gdal.GA_ReadOnly)
    band = ds.GetRasterBand(1)
    data = band.ReadAsArray().astype(np.float32)
    valid = data[data != nodata_val]
    if valid.size == 0:
        log.warning("  %s — NO valid pixels! Check LAS content and filter settings.", label)
    else:
        log.info(
            "  %s stats — min: %.0f  max: %.0f  mean: %.1f  std: %.1f  valid px: %d",
            label, valid.min(), valid.max(), valid.mean(), valid.std(), valid.size,
        )
    ds = None


# ── STEP 3: MERGE BANDS INTO RGB GEOTIFF ─────────────────────────────────────

def merge_bands(red_tif: str, green_tif: str, blue_tif: str,
                out_merged: str, nodata_val: float = -9999) -> None:
    """
    Stack three single-band GeoTIFFs into one 3-band RGB GeoTIFF using GDAL Python API.
    """
    log.info("Step 3 — Merging RGB bands ...")

    ds_red   = gdal.Open(red_tif,   gdal.GA_ReadOnly)
    ds_green = gdal.Open(green_tif, gdal.GA_ReadOnly)
    ds_blue  = gdal.Open(blue_tif,  gdal.GA_ReadOnly)

    cols          = ds_red.RasterXSize
    rows          = ds_red.RasterYSize
    geo_transform = ds_red.GetGeoTransform()
    projection    = ds_red.GetProjection()
    src_dtype     = ds_red.GetRasterBand(1).DataType

    driver = gdal.GetDriverByName("GTiff")
    ds_out = driver.Create(
        out_merged, cols, rows, 3, src_dtype,
        options=["COMPRESS=DEFLATE", "TILED=YES", "BIGTIFF=YES"]
    )
    ds_out.SetGeoTransform(geo_transform)
    ds_out.SetProjection(projection)

    for i, (ds_src, name) in enumerate(
        zip([ds_red, ds_green, ds_blue], ["Red", "Green", "Blue"]), start=1
    ):
        log.info("  Copying %s band ...", name)
        data = ds_src.GetRasterBand(1).ReadAsArray()
        ds_out.GetRasterBand(i).WriteArray(data)
        ds_out.GetRasterBand(i).SetNoDataValue(nodata_val)

    ds_red = ds_green = ds_blue = None
    ds_out.FlushCache()
    ds_out = None
    log.info("  -> Bands merged: %s", Path(out_merged).name)


# ── STEP 4: RESCALE TO 8-BIT (PER-BAND STRETCH) ──────────────────────────────

def rescale_to_8bit(src_tif: str, dst_tif: str, scale_divisor: int,
                    nodata_val: float = -9999) -> None:
    """
    If data is 16-bit (scale_divisor=256), rescale each band independently
    using its actual min/max (ignoring nodata pixels) to produce a full 0-255 range.

    Per-band stretch (rather than a fixed /65535) handles the common case where
    LiDAR colorization stores values as multiples of 256 (e.g. 0, 256, 512...65280),
    which would otherwise compress all three bands into a near-identical mid-range
    and produce a greyscale result.
    """
    if scale_divisor == 1:
        log.info("Step 4 — RGB already 8-bit, skipping rescale ...")
        shutil.copy2(src_tif, dst_tif)
        return

    log.info("Step 4 — Rescaling 16-bit -> 8-bit (per-band min/max stretch) ...")

    ds_src        = gdal.Open(src_tif, gdal.GA_ReadOnly)
    cols          = ds_src.RasterXSize
    rows          = ds_src.RasterYSize
    geo_transform = ds_src.GetGeoTransform()
    projection    = ds_src.GetProjection()
    n_bands       = ds_src.RasterCount

    driver = gdal.GetDriverByName("GTiff")
    ds_dst = driver.Create(
        dst_tif, cols, rows, n_bands, gdal.GDT_Byte,
        options=["COMPRESS=DEFLATE", "TILED=YES", "BIGTIFF=YES"]
    )
    ds_dst.SetGeoTransform(geo_transform)
    ds_dst.SetProjection(projection)

    for b in range(1, n_bands + 1):
        log.info("  Rescaling band %d/%d ...", b, n_bands)
        data        = ds_src.GetRasterBand(b).ReadAsArray().astype(np.float32)
        nodata_mask = (data == nodata_val)
        valid       = data[~nodata_mask]

        if valid.size == 0:
            log.warning("  Band %d has no valid pixels — writing empty band.", b)
            scaled = np.zeros_like(data, dtype=np.uint8)
        else:
            src_min = float(valid.min())
            src_max = float(valid.max())
            log.info("    Band %d actual range: %.0f – %.0f", b, src_min, src_max)

            if src_max == src_min:
                log.warning(
                    "    Band %d is flat (min == max == %.0f) — will render as solid grey.",
                    b, src_min,
                )
                scaled = np.full_like(data, 128, dtype=np.uint8)
            else:
                scaled = np.clip(
                    (data - src_min) / (src_max - src_min) * 255.0, 0, 255
                ).astype(np.uint8)

        scaled[nodata_mask] = 0  # nodata -> black (transparent-friendly)
        ds_dst.GetRasterBand(b).WriteArray(scaled)
        ds_dst.GetRasterBand(b).SetNoDataValue(0)

    ds_src = None
    ds_dst.FlushCache()
    ds_dst = None
    log.info("  -> Rescaled: %s", Path(dst_tif).name)


# ── STEP 5: BUILD OVERVIEWS + WRITE COG ──────────────────────────────────────

def convert_to_cog(src_tif: str, cog_tif: str) -> None:
    """
    Build internal overviews on the source then write a Cloud-Optimized GeoTIFF
    using GDAL's COG driver — all via the Python API.
    """
    log.info("Step 5 — Building overviews ...")
    ds = gdal.Open(src_tif, gdal.GA_Update)
    overview_levels = [2, 4, 8, 16, 32, 64, 128]
    ds.BuildOverviews("AVERAGE", overview_levels)
    ds = None
    log.info("  -> Overviews built: %s", overview_levels)

    log.info("  Writing Cloud-Optimized GeoTIFF ...")
    ds_src = gdal.Open(src_tif, gdal.GA_ReadOnly)

    translate_options = gdal.TranslateOptions(
        format="COG",
        creationOptions=[
            "COMPRESS=LZW",
            "PREDICTOR=2",
            "BIGTIFF=YES",
            "RESAMPLING=AVERAGE",
            "OVERVIEWS=IGNORE_EXISTING",
        ],
    )
    gdal.Translate(cog_tif, ds_src, options=translate_options)
    ds_src = None
    log.info("  -> COG written: %s", Path(cog_tif).name)


# ── STEP 6: VALIDATE COG ──────────────────────────────────────────────────────

def validate_cog(cog_tif: str) -> None:
    """Validate COG structure using GDAL's Python utility."""
    try:
        from osgeo.utils.validate_cloud_optimized_geotiff import validate
        is_valid, errors, warnings = validate(cog_tif, full_check=False)
        if is_valid:
            log.info("Step 6 — COG validation PASSED")
        else:
            log.warning("Step 6 — COG validation issues: %s", errors)
        if warnings:
            log.warning("  COG warnings: %s", warnings)
    except ImportError:
        log.info("Step 6 — COG validation skipped (osgeo.utils not available). Output is still valid.")


# ── PRINT OUTPUT INFO ─────────────────────────────────────────────────────────

def print_raster_info(tif_path: str) -> None:
    """Print basic raster metadata using GDAL Python API."""
    try:
        ds = gdal.Open(tif_path, gdal.GA_ReadOnly)
        gt = ds.GetGeoTransform()
        log.info("─" * 60)
        log.info("Output raster info:")
        log.info("  Size      : %d cols x %d rows x %d bands",
                 ds.RasterXSize, ds.RasterYSize, ds.RasterCount)
        log.info("  Pixel size: %.4f m x %.4f m", abs(gt[1]), abs(gt[5]))
        log.info("  Origin    : (%.4f, %.4f)", gt[0], gt[3])
        log.info("  Projection: %s ...", ds.GetProjection()[:80])
        for b in range(1, ds.RasterCount + 1):
            band = ds.GetRasterBand(b)
            mn, mx, mean, std = band.GetStatistics(True, True)
            log.info("  Band %d    : min=%.0f  max=%.0f  mean=%.1f  std=%.1f",
                     b, mn, mx, mean, std)
        ds = None
    except Exception as e:
        log.warning("Could not read output info: %s", e)


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LAS RGB -> Cloud-Optimized GeoTIFF (pure Python)")
    parser.add_argument("--input",       default=INPUT_LAS,   help="Path to input .las file")
    parser.add_argument("--output",      default=OUTPUT_DIR,  help="Output directory")
    parser.add_argument("--resolution",  type=float, default=RESOLUTION,
                        help="Raster resolution in metres (default: 0.05 = 5 cm)")
    parser.add_argument("--all-returns", action="store_true",
                        help="Use all returns (default: first-return only)")
    args, _ = parser.parse_known_args()  # ignore Jupyter's --f= kernel arg

    las_path          = args.input
    output_dir        = Path(args.output)
    resolution        = args.resolution
    first_return_only = not args.all_returns

    output_dir.mkdir(parents=True, exist_ok=True)

    stem      = Path(las_path).stem
    res_label = f"{int(resolution * 100)}cm"

    # Intermediate files (prefixed _tmp_ so they're easy to identify/delete)
    red_tif    = str(output_dir / f"_tmp_{stem}_RED_{res_label}.tif")
    green_tif  = str(output_dir / f"_tmp_{stem}_GREEN_{res_label}.tif")
    blue_tif   = str(output_dir / f"_tmp_{stem}_BLUE_{res_label}.tif")
    merged_tif = str(output_dir / f"_tmp_{stem}_RGB_merged_{res_label}.tif")
    scaled_tif = str(output_dir / f"_tmp_{stem}_RGB_8bit_{res_label}.tif")
    cog_tif    = str(output_dir / f"{stem}_RGB_{res_label}_COG.tif")

    log.info("=" * 60)
    log.info("LAS -> RGB Orthophoto COG")
    log.info("Input     : %s", las_path)
    log.info("Output    : %s", cog_tif)
    log.info("Resolution: %.2f m  |  First-return filter: %s", resolution, first_return_only)
    log.info("=" * 60)

    # Step 0: verify the LAS actually has distinct RGB
    verify_rgb_in_las(las_path)

    # Step 1: detect bit depth
    scale_divisor = detect_rgb_scale(las_path)

    # Step 2: rasterize each band + immediate stats check
    log.info("Step 2 — Rasterizing RGB bands (slow step for large files) ...")
    for dimension, out_tif in [("Red", red_tif), ("Green", green_tif), ("Blue", blue_tif)]:
        rasterize_band(las_path, out_tif, dimension, resolution,
                       first_return_only, nodata_val=NODATA_VAL)
        check_band_stats(out_tif, dimension, nodata_val=NODATA_VAL)

    # Step 3: merge bands
    merge_bands(red_tif, green_tif, blue_tif, merged_tif, nodata_val=NODATA_VAL)

    # Step 4: rescale to 8-bit with per-band stretch
    rescale_to_8bit(merged_tif, scaled_tif, scale_divisor, nodata_val=NODATA_VAL)

    # Step 5: COG
    convert_to_cog(scaled_tif, cog_tif)

    # Step 6: validate
    validate_cog(cog_tif)

    # Cleanup
    log.info("Cleaning up temporary files ...")
    for tmp in [red_tif, green_tif, blue_tif, merged_tif, scaled_tif]:
        try:
            Path(tmp).unlink(missing_ok=True)
        except Exception:
            pass

    log.info("=" * 60)
    log.info("DONE -> %s", cog_tif)
    print_raster_info(cog_tif)


if __name__ == "__main__":
    main()
