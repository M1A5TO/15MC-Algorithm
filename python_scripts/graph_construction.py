"""
Buduje graf pieszy z kafla .pbf i zapisuje w formacie CSR (.npz):

Przykład:
  python scripts/graph_construction.py --pbf data/gdansk_5km.pbf --out-prefix data/gdansk_5km
  python scripts/graph_construction.py --pbf data/gdansk_5km.pbf --out-prefix data/gdansk_5km --no-plot
"""

import argparse
import os
import time
from collections import defaultdict

import numpy as np
import pandas as pd

import matplotlib
if "DISPLAY" not in os.environ:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pyrosm import OSM
from scipy.spatial import cKDTree
from shapely import wkt

def tsec(t0):  
    return f"{(time.perf_counter()-t0):.3f}s"

def _norm(s):
    return str(s).strip().lower() if s is not None else ""

def _safe_coords(geom):
    if geom is None:
        return None
    try:
        return list(geom.coords)
    except Exception:
        try:
            g = wkt.loads(geom)
            return list(g.coords)
        except Exception:
            return None

def clean_walkable_edges(edges_gdf):
    """
    Reguły 'walkable':
    - Zawsze: footway, path, pedestrian, steps, platform, crossing, living_street,
              cycleway, track, residential, unclassified, service
    - Duże drogi (primary/secondary/tertiary i *_link) tylko gdy sidewalk=* lub foot in {yes,designated,permissive}
    - Wytnij motorway/trunk oraz motorroad=yes
    """
    edges = edges_gdf.copy()

    for col in ("highway", "foot", "sidewalk", "motorroad", "oneway"):
        if col in edges.columns:
            edges[col + "_n"] = edges[col].map(_norm)
        else:
            edges[col + "_n"] = ""

    always = {
        "footway","path","pedestrian","steps","platform","crossing",
        "living_street","cycleway","track","residential","unclassified","service",
    }
    big = {"primary","primary_link","secondary","secondary_link","tertiary","tertiary_link"}

    has_sidewalk = edges["sidewalk_n"].isin({"yes","both","left","right"})
    has_foot_ok  = edges["foot_n"].isin({"yes","designated","permissive"})
    is_fast      = edges["highway_n"].isin({"motorway","motorway_link","trunk","trunk_link"}) | (edges["motorroad_n"]=="yes")

    keep = (
        edges["highway_n"].isin(always)
        | (edges["highway_n"].isin(big) & (has_sidewalk | has_foot_ok))
    )
    return edges[keep & (~is_fast)].copy()

def make_uv_if_missing(edges_clean, nodes_gdf):
    """
    Jeśli w krawędziach nie ma 'u'/'v', mapujemy końce geometrii na najbliższe węzły (KDTree w EPSG:3857).
    """
    if "u" in edges_clean.columns and "v" in edges_clean.columns:
        return edges_clean

    nodes_3857 = nodes_gdf.to_crs(3857).copy()
    node_xy = np.c_[nodes_3857.geometry.x.values, nodes_3857.geometry.y.values]
    node_ids = nodes_gdf["node_id"].to_numpy()
    tree = cKDTree(node_xy)

    us, vs = [], []
    edges_3857 = edges_clean.to_crs(3857)
    for geom in edges_3857.geometry:
        coords = _safe_coords(geom)
        if not coords or len(coords) < 2:
            us.append(-1); vs.append(-1); continue
        x0,y0 = coords[0]; x1,y1 = coords[-1]
        _, i0 = tree.query([x0,y0], k=1)
        _, i1 = tree.query([x1,y1], k=1)
        us.append(int(node_ids[i0])); vs.append(int(node_ids[i1]))
    out = edges_clean.copy()
    out["u"] = us; out["v"] = vs
    return out

def save_network_png(edges_clean, nodes_df, out_prefix):
    try:
        fig, ax = plt.subplots(figsize=(9,9))
        uniq = edges_clean["highway"].dropna().astype(str).str.lower().unique() if "highway" in edges_clean.columns else []
        cmap = plt.get_cmap("tab20"); cmap_map = {h: cmap(i%20) for i,h in enumerate(uniq)}
        for _, r in edges_clean.iterrows():
            g = r.geometry
            if g is None:
                continue
            try:
                xy = list(g.coords)
            except Exception:
                continue
            xs = [p[0] for p in xy]; ys=[p[1] for p in xy]
            ax.plot(xs, ys, linewidth=0.6, color=cmap_map.get(str(r.get("highway","")).lower(), (0.5,0.5,0.5)))
        ax.scatter(nodes_df["lon"], nodes_df["lat"], s=5, color="k", alpha=0.3)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"Walking network (cleaned) from {os.path.basename(out_prefix)}")
        ax.set_xlabel("lon"); ax.set_ylabel("lat")
        out_png = f"{out_prefix}_plot.png"
        plt.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[PNG] {out_png}")
    except Exception as e:
        print("PNG failed:", e)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pbf", required=True, help="Ścieżka do kafla .osm.pbf (po filtrze osmium).")
    ap.add_argument("--out-prefix", default="tile", help="Prefix plików wyjściowych (bez rozszerzeń).")
    ap.add_argument("--no-plot", action="store_true", help="Nie zapisuj podglądowego PNG.")
    args = ap.parse_args()

    T0 = time.perf_counter()
    print(f"[0] Start  | pbf={args.pbf}")

    t = time.perf_counter()
    osm = OSM(args.pbf)
    nodes_gdf, edges_gdf = osm.get_network(nodes=True, network_type="walking")
    if nodes_gdf is None or edges_gdf is None or len(nodes_gdf)==0 or len(edges_gdf)==0:
        print("Brak danych 'walking' w PBF."); return
    if nodes_gdf.crs is None: nodes_gdf.set_crs(4326, inplace=True)
    if edges_gdf.crs is None: edges_gdf.set_crs(4326, inplace=True)
    print(f"[1] Wczytanie sieci: nodes={len(nodes_gdf)} edges={len(edges_gdf)}  ({tsec(t)})")

    t = time.perf_counter()
    edges_len = edges_gdf.to_crs(3857)
    edges_gdf = edges_gdf.copy()
    edges_gdf["length_m"] = edges_len.geometry.length.astype("float32")
    print(f"[2] Długości policzone ({tsec(t)})")

    t = time.perf_counter()
    edges_clean = clean_walkable_edges(edges_gdf)
    print(f"[3] Czyszczenie: edges {len(edges_gdf)} -> {len(edges_clean)}  ({tsec(t)})")

    t = time.perf_counter()
    if "id" in nodes_gdf.columns and "node_id" not in nodes_gdf.columns:
        nodes_gdf = nodes_gdf.rename(columns={"id":"node_id"})
    nodes_gdf["lon"] = nodes_gdf.geometry.x.astype("float32")
    nodes_gdf["lat"] = nodes_gdf.geometry.y.astype("float32")

    edges_clean = make_uv_if_missing(edges_clean, nodes_gdf)

    edges_clean = edges_clean[(edges_clean["u"]>=0) & (edges_clean["v"]>=0)].copy()
    edges_clean["u"] = edges_clean["u"].astype("int64")
    edges_clean["v"] = edges_clean["v"].astype("int64")
    print(f"[4] u/v gotowe  ({tsec(t)})")

    t = time.perf_counter()

    nodes_sorted = nodes_gdf.sort_values("node_id").reset_index(drop=True)
    osm_node_id = nodes_sorted["node_id"].to_numpy(dtype=np.int64)
    node_index = {int(n): i for i, n in enumerate(osm_node_id)}

    edges_sorted = edges_clean[["u","v","length_m","oneway","highway"]].copy()
    edges_sorted = edges_sorted.sort_values(["u","v"]).reset_index(drop=True)

    u_idx = edges_sorted["u"].map(node_index).to_numpy(dtype=np.int32, na_value=-1)
    v_idx = edges_sorted["v"].map(node_index).to_numpy(dtype=np.int32, na_value=-1)
    ok = (u_idx>=0) & (v_idx>=0)
    u_idx = u_idx[ok]; v_idx = v_idx[ok]
    w = edges_sorted.loc[ok, "length_m"].to_numpy(dtype=np.float32)

    N = len(osm_node_id)
    counts = np.bincount(u_idx, minlength=N)
    indptr = np.zeros(N+1, dtype=np.int32)
    indptr[1:] = np.cumsum(counts)
    indices = v_idx
    weights = w

    lon = nodes_sorted["lon"].to_numpy(dtype=np.float32)
    lat = nodes_sorted["lat"].to_numpy(dtype=np.float32)
    lonlat = np.stack([lon, lat], axis=1)  # [N,2]
    lon_e7 = (lon * 1e7).astype(np.int32)
    lat_e7 = (lat * 1e7).astype(np.int32)

    out_npz = f"{args.out_prefix}_csr.npz"
    np.savez_compressed(
        out_npz,
        indptr=indptr,
        indices=indices,
        weights=weights,
        lonlat=lonlat,
        lon_e7=lon_e7,
        lat_e7=lat_e7,
        osm_node_id=osm_node_id
    )
    print(f"[5] Zapisano CSR: {out_npz}  (N={N}, E={len(indices)})  ({tsec(t)})")

    if not args.no_plot:
        t = time.perf_counter()

        save_network_png(edges_clean, nodes_gdf[["lon","lat"]].assign(dummy=0).rename(columns={"dummy":"_"}), args.out_prefix)
        print(f"[6] PNG gotowy  ({tsec(t)})")

    print(f"[OK] Całość: {tsec(T0)}")

if __name__ == "__main__":
    main()
