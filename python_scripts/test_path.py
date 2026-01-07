import argparse, os, heapq
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

def haversine_m(lon1, lat1, lon2, lat2):
    R = 6371000.0
    dlon = np.radians(lon2 - lon1)
    dlat = np.radians(lat2 - lat1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1))*np.cos(np.radians(lat2))*np.sin(dlon/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

def load_csr(path):
    d = np.load(path)
    indptr  = d["indptr"].astype(np.int32, copy=False)
    indices = d["indices"].astype(np.int32, copy=False)
    weights = (d["weights"] if "weights" in d.files else d["data"]).astype(np.float32, copy=False)
    lonlat  = d["lonlat"].astype(np.float32, copy=False)
    osm_ids = d["osm_node_id"] if "osm_node_id" in d.files else None
    return indptr, indices, weights, lonlat, osm_ids

def load_pois(path, cats=None):
    ext = os.path.splitext(path)[1].lower()
    df = pd.read_parquet(path) if ext==".parquet" else pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    for need in ("poi_id","category","node_idx"):
        if need not in {c.lower() for c in df.columns}:
            raise ValueError("POI musi mieć kolumny: poi_id, category, node_idx")
    df = df.rename(columns={cols["poi_id"]:"poi_id", cols["category"]:"category", cols["node_idx"]:"node_idx"})
    df["poi_id"] = pd.to_numeric(df["poi_id"], errors="coerce").fillna(-1).astype(np.int64)
    df["node_idx"] = pd.to_numeric(df["node_idx"], errors="coerce").fillna(-1).astype(np.int64)
    if cats:
        df = df[df["category"].isin(cats)].copy()
    return df

def snap_to_node(lonlat, lon, lat, max_snap_m=None):
    tree = cKDTree(lonlat)
    _, idx = tree.query([lon, lat], k=1)
    dm = float(haversine_m(lon, lat, lonlat[idx,0], lonlat[idx,1]))
    if max_snap_m is not None and dm > max_snap_m:
        return -1, dm
    return int(idx), dm

def multisource_dijkstra_with_prev(indptr, indices, weights, sources_nodes, limit_m=None):
    """Źródła = lista węzłów. Zwraca dist[N], src_idx[N], prev[N], gdzie src_idx to indeks w sources_nodes."""
    N = len(indptr) - 1
    INF = np.float32(np.inf)
    dist = np.full(N, INF, np.float32)
    srci = np.full(N, -1, np.int32)
    prev = np.full(N, -1, np.int32)
    h = []
    for i, s in enumerate(sources_nodes):
        if 0 <= s < N and dist[s] > 0:
            dist[s] = 0.0
            srci[s] = i
            prev[s] = -1
            heapq.heappush(h, (0.0, int(s), i))
    L = np.float32(limit_m) if limit_m is not None else np.float32(np.inf)
    while h:
        d_u, u, s_idx = heapq.heappop(h)
        if d_u > dist[u]: 
            continue
        if d_u > L:
            break
        for e in range(indptr[u], indptr[u+1]):
            v = int(indices[e])
            nd = d_u + float(weights[e])
            if nd < dist[v] and nd <= L:
                dist[v] = nd
                srci[v] = s_idx
                prev[v] = u
                heapq.heappush(h, (nd, v, s_idx))
    return dist, srci, prev

def backtrack_to_source(prev, node):
    path = []
    u = int(node)
    if u < 0: 
        return path
    while u != -1:
        path.append(u)
        u = int(prev[u])
    path.reverse()  # teraz: source -> ... -> node
    return path

def save_path_csv(path_nodes, lonlat, indptr, indices, weights, out_csv, poi_id=None, category=None):
    rows = []
    cum = 0.0
    for i, n in enumerate(path_nodes):
        lo, la = float(lonlat[n,0]), float(lonlat[n,1])
        if i == 0:
            step = 0.0
        else:
            u, v = path_nodes[i-1], n
            # znajdź wagę u->v
            w_uv = None
            for e in range(indptr[u], indptr[u+1]):
                if int(indices[e]) == v:
                    w_uv = float(weights[e]); break
            if w_uv is None:
                w_uv = float(haversine_m(lonlat[u,0], lonlat[u,1], lo, la))
            step = w_uv
        cum += step
        rows.append((i, int(n), lo, la, step, cum, poi_id if poi_id is not None else "", category if category else ""))
    df = pd.DataFrame(rows, columns=["seq","node_idx","lon","lat","step_m","cum_m","poi_id","category"])
    df.to_csv(out_csv, index=False)
    return df

def render_local_png(indptr, indices, lonlat, path_nodes, center_lon, center_lat, out_png, radius_m=1000, max_edges=60000):
    # wybór węzłów w promieniu
    d = haversine_m(center_lon, center_lat, lonlat[:,0], lonlat[:,1])
    keep = d <= float(radius_m)
    if not keep.any():
        idx = np.argsort(d)[:500]
        keep_mask = np.zeros(len(lonlat), dtype=bool); keep_mask[idx] = True
    else:
        keep_mask = keep
    # krawędzie w promieniu
    deg = indptr[1:] - indptr[:-1]
    u_all = np.repeat(np.arange(len(deg), dtype=np.int32), deg)
    v_all = indices
    m = keep_mask[u_all] & keep_mask[v_all]
    u = u_all[m]; v = v_all[m]
    if len(u) > max_edges:
        sel = np.random.RandomState(0).choice(len(u), size=max_edges, replace=False)
        u, v = u[sel], v[sel]
    # rysunek
    plt.figure(figsize=(8,8))
    for uu, vv in zip(u, v):
        plt.plot([lonlat[uu,0], lonlat[vv,0]], [lonlat[uu,1], lonlat[vv,1]], linewidth=0.25, alpha=0.6)
    if path_nodes:
        xs = lonlat[path_nodes,0]; ys = lonlat[path_nodes,1]
        plt.plot(xs, ys, linewidth=2.5)
        plt.scatter([xs[0]],[ys[0]], s=40)
        plt.scatter([xs[-1]],[ys[-1]], s=40)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("lon"); plt.ylabel("lat"); plt.title("path (multi-source Dijkstra from POI)")
    plt.tight_layout(); plt.savefig(out_png, dpi=180, bbox_inches="tight"); plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csr", required=True)
    ap.add_argument("--poi", required=True, help="tile_poi.parquet (lub .csv) z poi_id,category,node_idx")
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--cats", nargs="*", default=None, help="(opcjonalnie) lista kategorii do rozważenia")
    ap.add_argument("--limit-m", type=float, default=None, help="(opcjonalnie) limit promienia jak w precompute")
    ap.add_argument("--max-snap-m", type=float, default=300.0)
    ap.add_argument("--out-csv", default="ms_path.csv")
    ap.add_argument("--out-png", default="ms_path.png")
    ap.add_argument("--radius-m", type=float, default=1000.0)
    args = ap.parse_args()

    indptr, indices, weights, lonlat, _ = load_csr(args.csr)
    pois = load_pois(args.poi, cats=args.cats)

    # sanity POI
    N = len(indptr) - 1
    pois = pois[(pois["node_idx"] >= 0) & (pois["node_idx"] < N)].reset_index(drop=True)
    if pois.empty:
        print("[ERR] Brak poprawnych POI po filtrach."); return

    # SNAP startu
    s_idx, sdm = snap_to_node(lonlat, args.lon, args.lat, max_snap_m=args.max_snap_m)
    if s_idx < 0:
        print(f"[ERR] Snap startu nieudany (> {args.max_snap_m:.0f} m)."); return
    print(f"[SNAP] start node={s_idx} (d={sdm:.1f} m) | POI={len(pois)}")

    # multi-source Dijkstra z POI (jak w precompute)
    sources_nodes = pois["node_idx"].to_numpy(dtype=np.int32, copy=True)
    dist, srci, prev = multisource_dijkstra_with_prev(indptr, indices, weights, sources_nodes, limit_m=args.limit_m)

    if not np.isfinite(dist[s_idx]):
        print("[INFO] Brak osiągalnego POI z tego węzła (w limicie)."); return

    # wybrany POI = ten, którego źródło doszło do s_idx
    poi_row = int(srci[s_idx])
    poi_id = int(pois.iloc[poi_row]["poi_id"])
    poi_cat = str(pois.iloc[poi_row]["category"])
    poi_node = int(pois.iloc[poi_row]["node_idx"])
    print(f"[HIT] poi_id={poi_id}  cat={poi_cat}  node={poi_node}  dist={float(dist[s_idx]):.1f} m")

    # backtrack ścieżki: source -> ... -> start; odwróć na start -> poi
    path_src_to_start = backtrack_to_source(prev, s_idx)
    if not path_src_to_start or path_src_to_start[0] != poi_node:
        print("[WARN] Backtrack nie zaczyna się od węzła źródłowego – możliwe, że poza limitem."); 
    path_nodes = list(reversed(path_src_to_start))  # start -> ... -> source(POI)

    # CSV + PNG
    df = save_path_csv(path_nodes, lonlat, indptr, indices, weights, args.out_csv, poi_id=poi_id, category=poi_cat)
    print(f"[CSV] {args.out_csv}  rows={len(df)}  length={df['cum_m'].iloc[-1] if len(df) else 0:.1f} m")

    render_local_png(indptr, indices, lonlat, path_nodes, args.lon, args.lat, args.out_png, radius_m=args.radius_m)
    print(f"[PNG] {args.out_png}")

if __name__ == "__main__":
    main()
