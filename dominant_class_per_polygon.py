# %% [markdown]
# Dominant categorical-raster class per polygon
#
# Summarises a single-band integer (categorical) raster over a set of polygons.
# Configured here for the DataBC Soil Parent Material raster (HaBC_PM.tif), but
# works for any categorical raster as long as CLASS_NAMES matches the class codes.
#
# Three outputs:
#   1. dominant_class.csv      - one dominant class per polygon
#        - majority (modal) class by pixel count for covered polygons
#        - nearest-pixel inference (KDTree, K-nearest majority) for empty/sparse ones
#   2. class_composition.csv   - FULL class breakdown per polygon,
#        area-weighted with partial pixels (so minority classes are kept)
#   3. class_top3.csv          - wide table, top-3 classes per polygon
#
# Assumes a projected CRS in metres so distances/areas are metres / metres^2.

# %% Imports & config -------------------------------------------------------
import math
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.windows import Window
from rasterio.transform import xy as rxy
from shapely.geometry import mapping, box
from scipy.spatial import cKDTree

# --- paths (edit these) ---
RASTER_PATH  = "data/HaBC_PM.tif"            # categorical raster (int class codes)
VECTOR_PATH  = "data/polygons.shp"           # polygon layer to summarise
ID_FIELD     = "NAME"                        # unique label field on the polygon layer
NUM_FIELD    = None                          # optional secondary field carried into the
                                             #   wide table and used to sort it; None to skip
OUT_CSV      = "output/dominant_class.csv"
OUT_COMP_CSV = "output/class_composition.csv"
OUT_WIDE_CSV = "output/class_top3.csv"
OUT_GPKG     = None                          # e.g. "output/polygons_with_class.gpkg", or None

# --- knobs ---
MIN_PIXELS   = 2      # polygons with fewer valid pixels -> nearest-pixel inference
N_NEAREST    = 5      # how many nearest valid pixels to majority-vote for inference
ALL_TOUCHED  = False  # False = count pixels whose CENTRE is inside polygon (area proxy)
                      # True  = count any pixel the polygon touches (over-weights edges)
LOWCONF_MULT = 3.0    # inference flagged low_confidence if nearest pixel > N*pixel size
WEAK_THRESH  = 70.0   # composition: print polygons whose top class is < this % of covered

# --- class code -> name (DataBC Soil Parent Material legend; "Anthroprogenic"
#     in the source legend corrected here to "Anthropogenic") ---
CLASS_NAMES = {
    2: "Anthropogenic", 3: "Colluvium", 4: "Weathered Bedrock", 5: "Eolian",
    6: "Fluvial", 7: "Glaciofluvial", 8: "Ice", 9: "Lacustrine",
    10: "Glaciolacustrine", 11: "Till", 12: "Organic", 13: "Rock",
    14: "Undifferentiated", 15: "Volcanic", 16: "Marine", 17: "Glaciomarine",
    18: "Water",
}


# %% Helper -----------------------------------------------------------------
def mode_stats(values):
    """Return (modal_value, count_of_mode, n_total) for a 1-D integer array."""
    vals, counts = np.unique(values, return_counts=True)
    i = int(counts.argmax())
    return int(vals[i]), int(counts[i]), int(values.size)


# %% Load + align CRS -------------------------------------------------------
gdf = gpd.read_file(VECTOR_PATH)

with rasterio.open(RASTER_PATH) as src:
    raster_crs = src.crs
    nodata     = src.nodata
    pixel_size = abs(src.res[0])

if gdf.crs != raster_crs:
    gdf = gdf.to_crs(raster_crs)

print(f"Polygons: {len(gdf)} | raster CRS: {raster_crs} | "
      f"pixel size: {pixel_size:g} | nodata: {nodata}")


# %% Majority class per polygon ---------------------------------------------
records = []
with rasterio.open(RASTER_PATH) as src:
    nodata = src.nodata
    for gidx, row in gdf.iterrows():
        name = row[ID_FIELD]
        geom = row.geometry

        try:
            arr, _ = mask(src, [mapping(geom)], crop=True,
                          all_touched=ALL_TOUCHED, filled=False)
        except ValueError:
            records.append({"_gidx": gidx, ID_FIELD: name,
                            "dominant_class": None, "pixel_count": 0,
                            "method": "needs_inference"})
            continue

        data  = arr[0]                       # single band -> masked array
        valid = data.compressed()            # drop pixels outside the polygon
        if nodata is not None:
            valid = valid[valid != nodata]
        if np.issubdtype(valid.dtype, np.floating):
            valid = valid[~np.isnan(valid)]

        if valid.size >= MIN_PIXELS:
            mval, mcount, n = mode_stats(valid.astype(int))
            records.append({"_gidx": gidx, ID_FIELD: name,
                            "dominant_class": mval, "pixel_count": n,
                            "mode_count": mcount,
                            "mode_fraction": round(mcount / n, 3),
                            "method": "majority"})
        else:
            records.append({"_gidx": gidx, ID_FIELD: name,
                            "dominant_class": None,
                            "pixel_count": int(valid.size),
                            "method": "needs_inference"})

result = pd.DataFrame(records)
print(result[[ID_FIELD, "dominant_class", "pixel_count", "method"]])


# %% Build KDTree of valid pixel centres (for inference) --------------------
with rasterio.open(RASTER_PATH) as src:
    band      = src.read(1, masked=True)
    transform = src.transform
    nodata    = src.nodata

arr_full   = band.astype(float).filled(np.nan)      # cast before fill: int8 can't hold NaN
valid_mask = ~np.ma.getmaskarray(band)
if nodata is not None:
    valid_mask &= (arr_full != nodata)
valid_mask &= ~np.isnan(arr_full)

rows, cols = np.where(valid_mask)
xs, ys     = rxy(transform, rows, cols, offset="center")   # cell centres
xy         = np.column_stack([np.asarray(xs), np.asarray(ys)])
pix_vals   = arr_full[rows, cols].astype(int)
tree       = cKDTree(xy)
print(f"Valid pixels available for inference: {len(pix_vals)}")


# %% Infer dominant class for sparse / empty polygons -----------------------
if "nearest_dist" not in result.columns:
    result["nearest_dist"] = np.nan
if "low_confidence" not in result.columns:
    result["low_confidence"] = pd.Series(pd.NA, index=result.index, dtype="object")

need = result["method"] == "needs_inference"
for ri in result.index[need]:
    gidx = result.at[ri, "_gidx"]
    cen  = gdf.loc[gidx, "geometry"].centroid

    k = min(N_NEAREST, len(pix_vals))
    dist, ii = tree.query([cen.x, cen.y], k=k)
    dist = np.atleast_1d(dist)
    ii   = np.atleast_1d(ii)

    mval, mcount, n = mode_stats(pix_vals[ii])
    nearest = float(dist.min())

    result.at[ri, "dominant_class"] = mval
    result.at[ri, "mode_count"]     = mcount
    result.at[ri, "mode_fraction"]  = round(mcount / n, 3)
    result.at[ri, "method"]         = f"nearest_{n}px"
    result.at[ri, "nearest_dist"]   = round(nearest, 1)
    result.at[ri, "low_confidence"] = nearest > LOWCONF_MULT * pixel_size


# %% Add names, save main result, QA summary --------------------------------
result = result.drop(columns="_gidx", errors="ignore")
result["dominant_class"] = result["dominant_class"].astype("Int64")  # nullable int
if "dominant_name" not in result.columns:
    result.insert(result.columns.get_loc("dominant_class") + 1,
                  "dominant_name", result["dominant_class"].map(CLASS_NAMES))

result.to_csv(OUT_CSV, index=False)
print(f"\nWrote {OUT_CSV}")
print(result[[ID_FIELD, "dominant_class", "dominant_name",
              "pixel_count", "mode_fraction", "method"]].to_string(index=False))

inferred = result[result["method"].str.startswith("nearest", na=False)]
if len(inferred):
    print(f"\n{len(inferred)} polygon(s) inferred from nearest pixels:")
    print(inferred[[ID_FIELD, "dominant_class", "dominant_name",
                    "nearest_dist", "low_confidence"]].to_string(index=False))

if OUT_GPKG:
    out = gdf.merge(result, on=ID_FIELD, how="left")
    out.to_file(OUT_GPKG, driver="GPKG")
    print(f"\nWrote {OUT_GPKG}")


# %% Full class composition (area-weighted, partial pixels included) --------
def composition_for_geom(src, geom, nodata):
    """Exact area (map units) of each class inside geom, partial cells included."""
    if not geom.is_valid:
        geom = geom.buffer(0)
    minx, miny, maxx, maxy = geom.bounds
    inv = ~src.transform
    pts = [inv * (x, y) for x, y in ((minx, miny), (minx, maxy),
                                     (maxx, miny), (maxx, maxy))]
    cs = [p[0] for p in pts]; rs = [p[1] for p in pts]
    c0 = max(0, math.floor(min(cs)) - 1); c1 = min(src.width,  math.ceil(max(cs)) + 1)
    r0 = max(0, math.floor(min(rs)) - 1); r1 = min(src.height, math.ceil(max(rs)) + 1)
    if c1 <= c0 or r1 <= r0:
        return {}
    win = Window(c0, r0, c1 - c0, r1 - r0)
    arr = src.read(1, window=win)
    wt  = src.window_transform(win)
    areas = {}
    for rr in range(arr.shape[0]):
        for cc in range(arr.shape[1]):
            val = arr[rr, cc]
            if nodata is not None and val == nodata:
                continue
            x0, y0 = wt * (cc, rr); x1, y1 = wt * (cc + 1, rr + 1)
            cell = box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
            if not geom.intersects(cell):
                continue
            a = geom.intersection(cell).area
            if a > 0:
                areas[int(val)] = areas.get(int(val), 0.0) + a
    return areas

rows_out = []
with rasterio.open(RASTER_PATH) as src:
    nodata = src.nodata
    for gidx, row in gdf.iterrows():
        name, geom = row[ID_FIELD], row.geometry
        areas     = composition_for_geom(src, geom, nodata)
        poly_area = geom.area
        covered   = sum(areas.values())
        if not areas:                       # off-raster -> see inference in main table
            rows_out.append({ID_FIELD: name, "class": None, "area_m2": 0.0,
                             "pct_of_covered": np.nan, "pct_of_polygon": 0.0,
                             "coverage": 0.0})
            continue
        for cls, a in sorted(areas.items(), key=lambda kv: kv[1], reverse=True):
            rows_out.append({
                ID_FIELD: name, "class": cls,
                "area_m2": round(a, 1),
                "pct_of_covered": round(100 * a / covered, 1),
                "pct_of_polygon": round(100 * a / poly_area, 1),
                "coverage": round(100 * covered / poly_area, 1),
            })

comp = pd.DataFrame(rows_out)


# %% Name + save composition, flag weak (mixed) polygons --------------------
comp["class"] = comp["class"].astype("Int64")
if "name" not in comp.columns:
    comp.insert(comp.columns.get_loc("class") + 1,
                "name", comp["class"].map(CLASS_NAMES))

comp.to_csv(OUT_COMP_CSV, index=False)
print(f"\nWrote {OUT_COMP_CSV}")

# polygons where the dominant class is a weak winner -> worth eyeballing
top  = (comp.dropna(subset=["class"])
            .sort_values("pct_of_covered", ascending=False)
            .groupby(ID_FIELD).head(1))
weak = top.loc[top["pct_of_covered"] < WEAK_THRESH, ID_FIELD].tolist()
if weak:
    print(f"\n{len(weak)} polygon(s) with a weak/mixed dominant class "
          f"(< {WEAK_THRESH:g}% of covered area):")
    print(comp[comp[ID_FIELD].isin(weak)].to_string(index=False))
else:
    print("\nNo weak/mixed polygons under the threshold.")


# %% Wide table: top-3 classes per polygon ----------------------------------
TOP_N = 3

def class_counts(src, geom, nodata):
    """Per-class pixel counts inside a polygon (same masking rules as main pass)."""
    try:
        arr, _ = mask(src, [mapping(geom)], crop=True,
                      all_touched=ALL_TOUCHED, filled=False)
    except ValueError:
        return {}
    valid = arr[0].compressed()
    if nodata is not None:
        valid = valid[valid != nodata]
    if np.issubdtype(valid.dtype, np.floating):
        valid = valid[~np.isnan(valid)]
    vals, counts = np.unique(valid.astype(int), return_counts=True)
    return dict(zip(vals.tolist(), counts.tolist()))

# (class_code_col, name_col, pixel_count_col) for ranks 1..3
LABELS = [
    ("dominant_class", "dominant_name", "pixel_count"),
    ("class_2nd",      "name_2nd",      "pixel_count_2nd"),
    ("class_3rd",      "name_3rd",      "pixel_count_3rd"),
]

method_by_name = dict(zip(result[ID_FIELD], result["method"]))
infer_class    = dict(zip(result[ID_FIELD], result["dominant_class"]))

wide_rows = []
with rasterio.open(RASTER_PATH) as src:
    nodata = src.nodata
    for _, row in gdf.iterrows():
        name, geom = row[ID_FIELD], row.geometry
        method = method_by_name.get(name)
        ranked = sorted(class_counts(src, geom, nodata).items(),
                        key=lambda kv: kv[1], reverse=True)[:TOP_N]

        if method != "majority":                 # inferred / no real pixels
            dc = infer_class.get(name)
            ranked = [(int(dc), pd.NA)] if pd.notna(dc) else []

        rec = {ID_FIELD: name, "method": method}
        if NUM_FIELD:
            rec[NUM_FIELD] = row[NUM_FIELD]
        for i, (ccol, ncol, pcol) in enumerate(LABELS):
            if i < len(ranked):
                cls, cnt = ranked[i]
                rec[ccol], rec[ncol], rec[pcol] = cls, CLASS_NAMES.get(cls), cnt
            else:
                rec[ccol] = rec[ncol] = rec[pcol] = pd.NA
        wide_rows.append(rec)

cols = [ID_FIELD]
if NUM_FIELD:
    cols.append(NUM_FIELD)
cols += ["dominant_class", "dominant_name", "pixel_count", "method",
         "class_2nd", "name_2nd", "pixel_count_2nd",
         "class_3rd", "name_3rd", "pixel_count_3rd"]
wide = pd.DataFrame(wide_rows)[cols]

if NUM_FIELD:
    wide[NUM_FIELD] = wide[NUM_FIELD].astype("Int64")
    wide = wide.sort_values(NUM_FIELD).reset_index(drop=True)

for c in ("dominant_class", "class_2nd", "class_3rd",
          "pixel_count", "pixel_count_2nd", "pixel_count_3rd"):
    wide[c] = wide[c].astype("Int64")

wide.to_csv(OUT_WIDE_CSV, index=False, na_rep="")
print(f"Wrote {OUT_WIDE_CSV}")
disp = wide.astype(object).where(wide.notna(), "")
print(disp.to_string(index=False))
# %%
