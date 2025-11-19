import argparse, json
from math import ceil, cos, radians, floor, sin, asin, sqrt
from typing import List, Dict, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

try:
    import geopandas as gpd
except Exception:
    gpd = None


KM_PER_DEG_LAT = 111.32  
BBOX_DEFAULT = (14.07, 49.00, 24.15, 54.84)  # Default bounding box (~Poland)

def parse_bbox(s: str) -> Tuple[float, float, float, float]:
    vals = [float(x.strip()) for x in s.split(",")]
    if len(vals) != 4: raise ValueError('BBOX as "minlon,minlat,maxlon,maxlat"')
    minlon, minlat, maxlon, maxlat = vals
    if not (minlon < maxlon and minlat < maxlat): raise ValueError("BBOX: min<max")
    return minlon, minlat, maxlon, maxlat

def parse_lonlat(s: str) -> Tuple[float, float]:
    vals = [float(x.strip()) for x in s.split(",")]
    if len(vals) != 2: raise ValueError('Coordinates as "lon,lat"')
    return vals[0], vals[1]

def km_per_deg_lon(lat_deg: float) -> float:
    return KM_PER_DEG_LAT * cos(radians(lat_deg))

def deg_from_km_lon(km: float, ref_lat_deg: float) -> float:
    return km / max(km_per_deg_lon(ref_lat_deg), 1e-9)

def deg_from_km_lat(km: float) -> float:
    return km / KM_PER_DEG_LAT


def compute_degrees(maxlat: float, tile_km: float, buffer_km: float):
    dlat_tile = deg_from_km_lat(tile_km)
    dlon_tile = deg_from_km_lon(tile_km, maxlat)
    dlat_buf  = deg_from_km_lat(buffer_km)
    dlon_buf  = deg_from_km_lon(buffer_km, maxlat)
    return dlon_tile, dlat_tile, dlon_buf, dlat_buf

def generate_tiles_with_buffers(
    bbox: Tuple[float, float, float, float],
    dlon_tile: float, dlat_tile: float,
    dlon_buf: float,  dlat_buf: float
) -> List[Dict]:
    minlon, minlat, maxlon, maxlat = bbox

    # Tile spacing (no overlap)
    dlon_step = dlon_tile
    dlat_step = dlat_tile

    # Center of the first tile (top-right corner)
    first_center_lon = maxlon - 0.5 * dlon_tile
    first_center_lat = maxlat - 0.5 * dlat_tile

    # Calculate how many tiles fit into the bounding box
    span_lon = (first_center_lon - (minlon - 0.5 * dlon_tile))
    span_lat = (first_center_lat - (minlat - 0.5 * dlat_tile))
    n_cols = int(ceil(span_lon / dlon_step)) + 1
    n_rows = int(ceil(span_lat / dlat_step)) + 1

    cells: List[Dict] = []
    for ci in range(n_cols):
        clon = first_center_lon - ci * dlon_step
        tile_minx = clon - 0.5 * dlon_tile
        tile_maxx = clon + 0.5 * dlon_tile
        if tile_maxx <= minlon or tile_minx >= maxlon:
            continue

        for ri in range(n_rows):
            clat = first_center_lat - ri * dlat_step
            tile_miny = clat - 0.5 * dlat_tile
            tile_maxy = clat + 0.5 * dlat_tile
            if tile_maxy <= minlat or tile_miny >= maxlat:
                continue

            # Add buffer around the tile
            buf_minx = tile_minx - dlon_buf
            buf_maxx = tile_maxx + dlon_buf
            buf_miny = tile_miny - dlat_buf
            buf_maxy = tile_maxy + dlat_buf

            cells.append({
                "grid_id": f"r{ri}_c{ci}",
                "row": ri, "col": ci,
                "centroid_wgs84": {"lon": float(clon), "lat": float(clat)},
                "tile_bbox_wgs84": {
                    "minlon": float(tile_minx), "minlat": float(tile_miny),
                    "maxlon": float(tile_maxx), "maxlat": float(tile_maxy),
                },
                "buffer_bbox_wgs84": {
                    "minlon": float(buf_minx), "minlat": float(buf_miny),
                    "maxlon": float(buf_maxx), "maxlat": float(buf_maxy),
                },
                "tile_deg":   {"dlon": float(dlon_tile), "dlat": float(dlat_tile)},
                "buffer_deg": {"dlon": float(dlon_buf),  "dlat": float(dlat_buf)},
            })
    return cells


def haversine_km(lon1, lat1, lon2, lat2):
    R = 6371.0088
    dlon = radians(lon2 - lon1); dlat = radians(lat2 - lat1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return 2 * R * asin(min(1.0, sqrt(a)))

def select_nearest(cells: List[Dict], lon: float, lat: float) -> Optional[int]:
    if not cells: return None
    best_i, best_d = 0, float("inf")
    for i, rec in enumerate(cells):
        c = rec["centroid_wgs84"]
        d = haversine_km(lon, lat, c["lon"], c["lat"])
        if d < best_d: best_d, best_i = d, i
    return best_i

def k_nearest_indices(cells: List[Dict], idx: int, k: int = 5) -> List[int]:
    base = cells[idx]["centroid_wgs84"]
    dists = []
    for i, rec in enumerate(cells):
        if i == idx: continue
        c = rec["centroid_wgs84"]
        d = haversine_km(base["lon"], base["lat"], c["lon"], c["lat"])
        dists.append((d, i))
    dists.sort(key=lambda x: x[0])
    return [i for _, i in dists[:k]]


def save_json(cells: List[Dict], path: str):
    if path == "-" or not path: print(json.dumps(cells, ensure_ascii=False))
    else:
        with open(path, "w", encoding="utf-8") as f: json.dump(cells, f, ensure_ascii=False)
        print(f"OK: JSON saved -> {path} (tiles: {len(cells)})")

def draw_rect(ax, bbox, *, edgewidth=0.8, fill=False, alpha=0.2):
    w = bbox["maxlon"] - bbox["minlon"]
    h = bbox["maxlat"] - bbox["minlat"]
    ax.add_patch(Rectangle((bbox["minlon"], bbox["minlat"]), w, h, fill=fill, alpha=alpha, linewidth=edgewidth))

def plot_overview(cells, bbox, out_png, country_geojson=None, pad_deg=1.0, grid_step_deg=1.0,
                  selected_idx: Optional[int]=None, selected_point: Optional[Tuple[float,float]]=None):
    if not out_png: return
    minlon, minlat, maxlon, maxlat = bbox
    x0, x1 = minlon - pad_deg, maxlon + pad_deg
    y0, y1 = minlat - pad_deg, maxlat + pad_deg

    fig, ax = plt.subplots(figsize=(11, 9))

    # Draw country outline
    if country_geojson and gpd is not None:
        try:
            country = gpd.read_file(country_geojson)
            if country.crs is None or str(country.crs).lower() not in ("epsg:4326","wgs84"):
                country = country.to_crs(4326)
            country.boundary.plot(ax=ax, linewidth=1.0)
        except Exception as e:
            print(f"Warning: failed to read {country_geojson} ({e}) — drawing without outline.")

    # Draw buffers (outer contours)
    for rec in cells:
        draw_rect(ax, rec["buffer_bbox_wgs84"], edgewidth=0.5, fill=False, alpha=0.15)

    # Draw main tiles
    for i, rec in enumerate(cells):
        lw = 1.2 if (selected_idx is not None and i == selected_idx) else 0.8
        draw_rect(ax, rec["tile_bbox_wgs84"], edgewidth=lw, fill=False, alpha=0.2)

    # Mark selected tile and/or point
    if selected_idx is not None:
        c = cells[selected_idx]["centroid_wgs84"]
        ax.plot([c["lon"]], [c["lat"]], marker="o", markersize=5)
    if selected_point is not None:
        ax.plot([selected_point[0]], [selected_point[1]], marker="*", markersize=7)

    # Draw coordinate grid
    if grid_step_deg and grid_step_deg > 0:
        xticks = np.arange(floor(x0), floor(x1)+1e-9, grid_step_deg)
        yticks = np.arange(floor(y0), floor(y1)+1e-9, grid_step_deg)
        ax.set_xticks(xticks); ax.set_yticks(yticks); ax.grid(True, which="both", linewidth=0.5)

    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Longitude (°)"); ax.set_ylabel("Latitude (°)")
    ax.set_title("10×10 km tiles (black) + buffers (contours). The buffer does not define tile boundaries.")
    fig.tight_layout(); fig.savefig(out_png, dpi=220); plt.close(fig)
    print(f"OK: overview saved -> {out_png}")

def plot_zoom(cells: List[Dict], out_png_zoom: str, selected_idx: int, neighbors_idx: List[int],
              selected_point: Optional[Tuple[float,float]]=None):
    if not out_png_zoom: return
    ids = [selected_idx] + neighbors_idx
    sub = [cells[i] for i in ids]

    # Determine drawing range
    minlon = min(r["buffer_bbox_wgs84"]["minlon"] for r in sub)
    minlat = min(r["buffer_bbox_wgs84"]["minlat"] for r in sub)
    maxlon = max(r["buffer_bbox_wgs84"]["maxlon"] for r in sub)
    maxlat = max(r["buffer_bbox_wgs84"]["maxlat"] for r in sub)
    padx = (maxlon - minlon) * 0.15 or 0.05
    pady = (maxlat - minlat) * 0.15 or 0.05

    fig, ax = plt.subplots(figsize=(8.5, 7.5))

    for i, rec in zip(ids, sub):
        # Highlight selected tile
        edge = 1.5 if i == selected_idx else 1.0
        fill = True if i == selected_idx else False
        draw_rect(ax, rec["buffer_bbox_wgs84"], edgewidth=edge, fill=fill, alpha=0.25 if fill else 0.2)
        draw_rect(ax, rec["tile_bbox_wgs84"], edgewidth=1.2, fill=False, alpha=0.2)

    # Draw centroids
    for rec in sub:
        c = rec["centroid_wgs84"]
        ax.plot([c["lon"]], [c["lat"]], marker="x", markersize=5)
    if selected_point is not None:
        ax.plot([selected_point[0]], [selected_point[1]], marker="*", markersize=8)

    ax.set_xlim(minlon - padx, maxlon + padx)
    ax.set_ylim(minlat - pady, maxlat + pady)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Longitude (°)"); ax.set_ylabel("Latitude (°)")
    ax.set_title("Zoom: selected tile (filled buffer) + 5 nearest (outline buffers)")
    fig.tight_layout(); fig.savefig(out_png_zoom, dpi=240); plt.close(fig)
    print(f"OK: zoom saved -> {out_png_zoom}")

def main():
    ap = argparse.ArgumentParser
    ap.add_argument("--bbox", default="{}, {}, {}, {}".format(*BBOX_DEFAULT),
                    help='BBOX WGS84: "minlon,minlat,maxlon,maxlat" (default ≈ Poland).')
    ap.add_argument("--tile-km", type=float, default=10.0, help="Tile size (km) – without overlap.")
    ap.add_argument("--buffer-km", type=float, default=2.0, help="Buffer radius (km) added to tile edges.")
    ap.add_argument("--out-json", required=True, help='JSON file or "-" (stdout).')
    ap.add_argument("--out-png", default="", help="PNG with overview of the grid.")
    ap.add_argument("--out-png-zoom", default="", help="PNG zoom of the selected tile + 5 nearest.")
    ap.add_argument("--country-geojson", default="", help="(Optional) Country outline (GeoJSON) on the plot.")
    ap.add_argument("--plot-pad-deg", type=float, default=1.0, help="Plot padding (± degrees).")
    ap.add_argument("--gridline-step-deg", type=float, default=1.0, help="Axis grid spacing in degrees.")
    ap.add_argument("--select", default="", help='Coordinates "lon,lat" to select the nearest tile.')
    args = ap.parse_args()

    bbox = parse_bbox(args.bbox)
    dlon_tile, dlat_tile, dlon_buf, dlat_buf = compute_degrees(
        maxlat=bbox[3], tile_km=args.tile_km, buffer_km=args.buffer_km
    )
    cells = generate_tiles_with_buffers(bbox, dlon_tile, dlat_tile, dlon_buf, dlat_buf)

    selected_idx = None; selected_point = None; neighbors_idx: List[int] = []
    if args.select.strip():
        lon, lat = parse_lonlat(args.select)
        selected_point = (lon, lat)
        idx = select_nearest(cells, lon, lat)
        if idx is not None:
            selected_idx = idx
            for i in range(len(cells)):
                cells[i]["selected"] = (i == idx)
            neighbors_idx = k_nearest_indices(cells, idx, k=5)

    save_json(cells, args.out_json)
    if args.out_png:
        plot_overview(
            cells, bbox, args.out_png,
            country_geojson=args.country_geojson or None,
            pad_deg=args.plot_pad_deg, grid_step_deg=args.gridline_step_deg,
            selected_idx=selected_idx, selected_point=selected_point
        )
    if args.out_png_zoom and selected_idx is not None:
        plot_zoom(cells, args.out_png_zoom, selected_idx, neighbors_idx, selected_point=selected_point)

if __name__ == "__main__":
    main()
