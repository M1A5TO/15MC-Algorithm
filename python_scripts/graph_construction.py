"""
Builds a pedestrian graph from a .pbf tile and saves it in CSR (.npz) format:
  {out_prefix}_csr.npz  with fields:
    - indptr:int32, indices:int32, weights:float32 (meters)
    - lonlat:float32 [N,2] (WGS84)
    - lon_e7:int32, lat_e7:int32 (WGS84 scaled by 1e7; optional for snapping)
    - osm_node_id:int64 (OSM node IDs in the same order as lonlat)
Note: no CSV/JSON output – this is a runtime artifact used exclusively by A*/Dijkstra.

Example:
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


def build_bidirectional_csr_from_edges(edges_clean, N, w_col="length_m"):

    edges = edges_clean[(edges_clean["u"] >= 0) & (edges_clean["v"] >= 0)].copy()
    edges["u"] = edges["u"].astype("int64")
    edges["v"] = edges["v"].astype("int64")

    u_fwd = edges["u"].to_numpy(np.int64, copy=False)
    v_fwd = edges["v"].to_numpy(np.int64, copy=False)
    w_fwd = edges[w_col].to_numpy(np.float32, copy=False)

    u_all = np.concatenate([u_fwd, v_fwd]).astype(np.int64, copy=False)
    v_all = np.concatenate([v_fwd, u_fwd]).astype(np.int64, copy=False)
    w_all = np.concatenate([w_fwd, w_fwd]).astype(np.float32, copy=False)

    m = (u_all != v_all)
    u_all, v_all, w_all = u_all[m], v_all[m], w_all[m]

    ord_idx = np.lexsort((v_all, u_all))
    u_all = u_all[ord_idx]; v_all = v_all[ord_idx]; w_all = w_all[ord_idx]

    same = (np.r_[False, (u_all[1:] == u_all[:-1]) & (v_all[1:] == v_all[:-1])])
    starts = np.where(~same)[0]
    w_min = np.minimum.reduceat(w_all, starts)

    u_uv = u_all[starts].astype(np.int32, copy=False)
    v_uv = v_all[starts].astype(np.int32, copy=False)
    w_uv = w_min.astype(np.float32, copy=False)

    if (len(u_uv) and (u_uv.max() >= N or v_uv.max() >= N)) or (u_uv.min(initial=0) < 0 or v_uv.min(initial=0) < 0):
        raise ValueError("Edge endpoints reference node index out of [0, N). Check node indexing.")

    counts = np.bincount(u_uv, minlength=N).astype(np.int32, copy=False)
    indptr = np.empty(N + 1, dtype=np.int32)
    indptr[0] = 0
    np.cumsum(counts, out=indptr[1:])

    indices = v_uv
    weights = w_uv


    assert indptr[-1] == len(indices) == len(weights)
    return indptr, indices, weights


def compute_edge_weights(u_idx, v_idx, length_m, lonlat, mode="length", alpha=1.25):
    length_m = length_m.astype(np.float32, copy=False)

    if mode == "length":
        return length_m

    lon = lonlat[:,0]; lat = lonlat[:,1]
    dlon = np.radians(lon[v_idx] - lon[u_idx])
    dlat = np.radians(lat[v_idx] - lat[u_idx])
    a = np.sin(dlat/2.0)**2 + np.cos(np.radians(lat[u_idx]))*np.cos(np.radians(lat[v_idx]))*np.sin(dlon/2.0)**2
    hav = (2.0 * 6371000.0 * np.arcsin(np.sqrt(a))).astype(np.float32)
    hav = np.maximum(hav, 0.01)  

    if mode == "haversine":
        return hav
    elif mode == "min":
        return np.minimum(length_m, alpha * hav).astype(np.float32, copy=False)
    else:
        return length_m

def show_csr_graph(indptr, indices, lonlat, max_edges=50000, title="CSR graph"):
 
    N = len(indptr) - 1
    u = np.repeat(np.arange(N, dtype=np.int32), indptr[1:] - indptr[:-1])
    v = indices.astype(np.int32, copy=False)
    E = len(v)
    if E > max_edges:
        sel = np.random.RandomState(42).choice(E, size=max_edges, replace=False)
        u = u[sel]; v = v[sel]
    plt.figure(figsize=(8,8))
    for uu, vv in zip(u, v):
        plt.plot([lonlat[uu,0], lonlat[vv,0]], [lonlat[uu,1], lonlat[vv,1]], linewidth=0.3)
    plt.scatter(lonlat[:,0], lonlat[:,1], s=2, alpha=0.2)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title(title)
    plt.xlabel("lon"); plt.ylabel("lat")
    plt.tight_layout()
    plt.show()

# ------------------------- main -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pbf", required=True, help="Path to the .osm.pbf tile (after osmium filtering).")
    ap.add_argument("--out-prefix", default="tile", help="Prefix for output files (without extensions).")
    ap.add_argument("--no-plot", action="store_true", help="Do not save preview PNG.")
    ap.add_argument("--dump-csv", action="store_true",
                    help="Save auxiliary CSV files: nodes_sorted.csv, edges_clean_raw.csv, edges_uv_debug.csv")
    ap.add_argument("--weight-mode", choices=["length", "haversine", "min"],
                default="length",
                help="How to compute edge weights: 'length' (length_m), 'haversine' (straight line between u,v), "
                     "'min' (min(length_m, alpha*haversine)).")
    ap.add_argument("--alpha", type=float, default=1.25,
                    help="Multiplier for --weight-mode=min (default 1.25).")
    ap.add_argument("--show-plot", action="store_true",
                    help="Show the graph in a Matplotlib window after CSR construction (requires DISPLAY).")
    args = ap.parse_args()

    T0 = time.perf_counter()
    print(f"[0] Start  | pbf={args.pbf}")

    t = time.perf_counter()
    osm = OSM(args.pbf)
    nodes_gdf, edges_gdf = osm.get_network(nodes=True, network_type="walking")
    if nodes_gdf is None or edges_gdf is None or len(nodes_gdf) == 0 or len(edges_gdf) == 0:
        print("No 'walking' data found in PBF."); return
    if nodes_gdf.crs is None: nodes_gdf.set_crs(4326, inplace=True)
    if edges_gdf.crs is None: edges_gdf.set_crs(4326, inplace=True)
    print(f"[1] Network loaded: nodes={len(nodes_gdf)} edges={len(edges_gdf)}  ({tsec(t)})")

    t = time.perf_counter()
    edges_len = edges_gdf.to_crs(3857)
    edges_gdf = edges_gdf.copy()
    edges_gdf["length_m"] = edges_len.geometry.length.astype("float32")
    print(f"[2] Edge lengths computed ({tsec(t)})")

    t = time.perf_counter()
    edges_clean = clean_walkable_edges(edges_gdf)
    print(f"[3] Cleaning: edges {len(edges_gdf)} -> {len(edges_clean)}  ({tsec(t)})")
    if args.dump_csv:
        cols = [c for c in ["u", "v", "length_m", "highway", "foot", "sidewalk", "oneway"] if c in edges_clean.columns]
        edges_clean[cols].to_csv(f"{args.out_prefix}_edges_clean_raw.csv", index=False)
        print(f"[dump] saved {args.out_prefix}_edges_clean_raw.csv")

    t = time.perf_counter()
    if "id" in nodes_gdf.columns and "node_id" not in nodes_gdf.columns:
        nodes_gdf = nodes_gdf.rename(columns={"id": "node_id"})
    nodes_gdf["lon"] = nodes_gdf.geometry.x.astype("float32")
    nodes_gdf["lat"] = nodes_gdf.geometry.y.astype("float32")

    edges_clean = make_uv_if_missing(edges_clean, nodes_gdf)
    edges_clean = edges_clean[(edges_clean["u"] >= 0) & (edges_clean["v"] >= 0)].copy()
    edges_clean["u"] = edges_clean["u"].astype("int64")
    edges_clean["v"] = edges_clean["v"].astype("int64")
    print(f"[4] u/v columns ready  ({tsec(t)})")

    t = time.perf_counter()
    nodes_sorted = nodes_gdf.sort_values("node_id").reset_index(drop=True)
    osm_node_id = nodes_sorted["node_id"].to_numpy(dtype=np.int64)
    node_index = {int(n): i for i, n in enumerate(osm_node_id)}

    if args.dump_csv:
        nodes_sorted.assign(idx=np.arange(len(nodes_sorted), dtype=np.int32))[
            ["idx", "node_id", "lon", "lat"]
        ].to_csv(f"{args.out_prefix}_nodes_sorted.csv", index=False)
        print(f"[dump] saved {args.out_prefix}_nodes_sorted.csv")

    tmp = edges_clean[["u", "v", "length_m", "highway", "oneway"]].copy()
    tmp["u_idx"] = tmp["u"].map(node_index).astype("Int64")
    tmp["v_idx"] = tmp["v"].map(node_index).astype("Int64")
    tmp = tmp.dropna(subset=["u_idx", "v_idx"]).reset_index(drop=True)
    tmp["u_idx"] = tmp["u_idx"].astype("int32")
    tmp["v_idx"] = tmp["v_idx"].astype("int32")

    lonlat_ns = nodes_sorted[["lon", "lat"]].to_numpy(dtype=np.float32)
    weights_vec = compute_edge_weights(
        tmp["u_idx"].to_numpy(),
        tmp["v_idx"].to_numpy(),
        tmp["length_m"].to_numpy(dtype=np.float32),
        lonlat_ns,
        mode=args.weight_mode,
        alpha=args.alpha
    ).astype(np.float32)
    
    if args.dump_csv:
        lon_ns = nodes_sorted["lon"].to_numpy(dtype=np.float32)
        lat_ns = nodes_sorted["lat"].to_numpy(dtype=np.float32)

        u = tmp["u_idx"].to_numpy()
        v = tmp["v_idx"].to_numpy()
        dlon = np.radians(lon_ns[v] - lon_ns[u])
        dlat = np.radians(lat_ns[v] - lat_ns[u])
        a = np.sin(dlat/2.0)**2 + np.cos(np.radians(lat_ns[u])) * np.cos(np.radians(lat_ns[v])) * np.sin(dlon/2.0)**2
        hav = (2.0 * 6371000.0 * np.arcsin(np.sqrt(a))).astype(np.float32)
        eps = 1e-3
        ratio = (tmp["length_m"].to_numpy(dtype=np.float32) / np.maximum(hav, eps)).astype(np.float32)

        dbg = pd.DataFrame({
            "osm_u_id": tmp["u"].to_numpy(dtype=np.int64),
            "osm_v_id": tmp["v"].to_numpy(dtype=np.int64),
            "u_idx": u.astype(np.int32),
            "v_idx": v.astype(np.int32),
            "length_m": tmp["length_m"].to_numpy(dtype=np.float32),
            "haversine_m": hav,
            "ratio_len_over_hav": ratio,
            "weight_final": weights_vec,
            "highway": (tmp["highway"].astype(str) if "highway" in tmp.columns else ""),
            "oneway":  (tmp["oneway"].astype(str)  if "oneway"  in tmp.columns else "")
        })
        dbg.to_csv(f"{args.out_prefix}_edges_uv_debug.csv", index=False)
        print(f"[dump] saved {args.out_prefix}_edges_uv_debug.csv")

    tmp_csr = tmp[["u_idx", "v_idx"]].rename(columns={"u_idx": "u", "v_idx": "v"}).copy()
    tmp_csr["w"] = weights_vec

    N = len(osm_node_id)
    indptr, indices, weights = build_bidirectional_csr_from_edges(
        tmp_csr.rename(columns={"w": "length_m"}), N, w_col="length_m"
    )

    outdeg = indptr[1:] - indptr[:-1]
    print(f"[debug] N={N}, E={len(indices)}, outdeg_mean={outdeg.mean():.2f}, zero_outdeg={(outdeg==0).sum()}")

    lon = nodes_sorted["lon"].to_numpy(dtype=np.float32)
    lat = nodes_sorted["lat"].to_numpy(dtype=np.float32)
    lonlat = np.stack([lon, lat], axis=1)
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
    print(f"[5] CSR saved: {out_npz}  (N={N}, E={len(indices)})  ({tsec(t)})")

    if args.show_plot:
        try:
            show_csr_graph(indptr, indices, lonlat,
                        max_edges=5000000,
                        title=f"CSR ({args.weight_mode}, alpha={args.alpha})")
        except Exception as e:
            print("show-plot failed:", e)
    if not args.no_plot:
        t = time.perf_counter()
        save_network_png(edges_clean, nodes_gdf[["lon","lat"]].assign(dummy=0).rename(columns={"dummy":"_"}), args.out_prefix)
        print(f"[6] PNG ready  ({tsec(t)})")

    print(f"[OK] Completed in {tsec(T0)}")

if __name__ == "__main__":
    main()
