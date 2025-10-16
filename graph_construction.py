"""
Usage:
  python extract_pyrosm_walk_full_plotfix.py --pbf gdynia_5km.pbf --out-prefix ped_pyrosm
  python extract_pyrosm_walk_full_plotfix.py --pbf gdynia_5km.pbf --out-prefix ped_pyrosm --no-plot
"""
import argparse
import json
import os
from pyrosm import OSM
import pandas as pd
import numpy as np
from shapely.geometry import LineString
from shapely import wkt
from pyproj import Transformer
from scipy.spatial import cKDTree
import matplotlib
# jeśli nie ma serwera X (DISPLAY), ustaw backend nie-interaktywny
if "DISPLAY" not in os.environ:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

def build_adjacency(edges_df, walk_speed_m_s=1.4):
    adj = defaultdict(list)
    for idx, row in edges_df.iterrows():
        u = int(row['u']); v = int(row['v'])
        length = float(row['length_m'])
        time_s = length / walk_speed_m_s if length is not None else None
        edge_id = int(row.get('way_id', idx))
        attr = {"edge_id": edge_id, "length_m": length, "cost_s": time_s, "highway": row.get('highway')}
        adj[u].append({"to": v, **attr})
        oneway = str(row.get('oneway')).lower()
        if oneway not in ("yes","true","1"):
            adj[v].append({"to": u, **attr})
    return adj

def safe_coords_from_geom(geom):
    """Zwróć listę (x,y) albo None jeśli nie da się."""
    if geom is None:
        return None
    # jeśli to już shapely geometry
    try:
        coords = list(geom.coords)
        return coords
    except Exception:
        # jeśli mamy WKT string
        try:
            g = wkt.loads(geom)
            return list(g.coords)
        except Exception:
            return None

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pbf", required=True, help="path to local .osm.pbf (should be a small extract)")
    p.add_argument("--out-prefix", default="ped_pyrosm")
    p.add_argument("--no-plot", action="store_true", help="skip plotting (useful on headless servers)")
    args = p.parse_args()

    print("Opening PBF with pyrosm (will load full walking network from the given PBF)...")
    osm = OSM(args.pbf)

    nodes_gdf, edges_gdf = osm.get_network(nodes=True, network_type="walking")

    if nodes_gdf is None or edges_gdf is None or len(nodes_gdf) == 0 or len(edges_gdf) == 0:
        print("Brak danych w pliku PBF lub sieć walking nie została znaleziona.")
        return

    # Ensure CRS defined; pyrosm returns EPSG:4326 usually
    if nodes_gdf.crs is None:
        nodes_gdf.set_crs(epsg=4326, inplace=True)
    if edges_gdf.crs is None:
        edges_gdf.set_crs(epsg=4326, inplace=True)

    # compute lengths in meters by reprojecting to EPSG:3857
    edges_3857 = edges_gdf.to_crs(epsg=3857).copy()
    edges_gdf = edges_gdf.copy()
    edges_gdf['length_m'] = edges_3857.geometry.length

    # ensure node id column exists (pyrosm uses 'id' for nodes)
    if 'id' in nodes_gdf.columns:
        nodes_gdf = nodes_gdf.rename(columns={'id': 'node_id'})
    nodes_gdf['lon'] = nodes_gdf.geometry.x
    nodes_gdf['lat'] = nodes_gdf.geometry.y

    # if edges have no 'u'/'v', map endpoints to nearest node (EPSG:3857 recommended)
    if 'u' not in edges_gdf.columns or 'v' not in edges_gdf.columns:
        print("Edges have no 'u'/'v' columns - mapping endpoints to nearest nodes (KDTree in EPSG:3857).")
        nodes_3857 = nodes_gdf.to_crs(epsg=3857).copy()
        node_coords = np.vstack([nodes_3857.geometry.x.values, nodes_3857.geometry.y.values]).T
        node_ids = nodes_gdf['node_id'].values
        tree = cKDTree(node_coords)
        us = []; vs = []
        edges_3857_local = edges_gdf.to_crs(epsg=3857)
        for geom in edges_3857_local.geometry:
            try:
                coords = list(geom.coords)
                p0 = coords[0]; p1 = coords[-1]
            except Exception:
                # defensywnie spróbuj parsować z WKT
                try:
                    coords_wkt = safe_coords_from_geom(geom)
                    if not coords_wkt or len(coords_wkt) < 2:
                        us.append(-1); vs.append(-1); continue
                    p0 = coords_wkt[0]; p1 = coords_wkt[-1]
                except Exception:
                    us.append(-1); vs.append(-1); continue
            _, i0 = tree.query([p0[0], p0[1]], k=1)
            _, i1 = tree.query([p1[0], p1[1]], k=1)
            us.append(int(node_ids[i0])); vs.append(int(node_ids[i1]))
        edges_gdf = edges_gdf.copy()
        edges_gdf['u'] = us; edges_gdf['v'] = vs

    # prepare save tables
    nodes_out = nodes_gdf[['node_id','lon','lat']].copy()
    edges_wgs = edges_gdf.copy()  # geometry jest w EPSG:4326
    # some edges may have geometry missing or invalid -> geometry_wkt safe
    edges_wgs['geometry_wkt'] = edges_wgs.geometry.apply(lambda g: g.wkt if g is not None else None)

    # Compose tags column: keep a JSON of select tags
    def collect_tags(row):
        tags = {}
        for k in ('highway','surface','oneway','name'):
            if k in row and row[k] is not None:
                tags[k] = row[k]
        return json.dumps(tags, ensure_ascii=False)

    tags_series = edges_wgs.apply(collect_tags, axis=1)

    edges_out = pd.DataFrame({
        'way_id': edges_wgs.get('osmid', edges_wgs.get('id', edges_wgs.index)),
        'u': edges_wgs['u'].astype(int),
        'v': edges_wgs['v'].astype(int),
        'length_m': edges_gdf['length_m'].astype(float),
        'highway': edges_wgs.get('highway', None),
        'surface': edges_wgs.get('surface', None),
        'oneway': edges_wgs.get('oneway', None),
        'geometry_wkt': edges_wgs['geometry_wkt'],
        'tags': tags_series
    })

    # Save CSVs
    nodes_csv = f"{args.out_prefix}_nodes.csv"
    edges_csv = f"{args.out_prefix}_edges.csv"
    nodes_out.to_csv(nodes_csv, index=False)
    edges_out.to_csv(edges_csv, index=False)
    print(f"Zapisano: {nodes_csv} ({len(nodes_out)} węzłów)")
    print(f"Zapisano: {edges_csv} ({len(edges_out)} krawędzi)")

    # Build adjacency (u -> list)
    adj = build_adjacency(edges_out)
    adj_file = f"{args.out_prefix}_adjacency.json"
    with open(adj_file, "w", encoding="utf8") as fh:
        json.dump({str(k): v for k,v in adj.items()}, fh, ensure_ascii=False)
    print(f"Zapisano adjacency: {adj_file}")

    # Plot / Save figure unless user asked to skip
    if args.no_plot:
        print("Pominięto rysowanie (--no-plot).")
        return

    try:
        # use shapely geometries directly (edges_wgs.geometry)
        plt.figure(figsize=(9,9))
        ax = plt.gca()
        unique_hw = edges_out['highway'].dropna().unique()
        cmap = plt.get_cmap('tab20')
        color_map = {hw: cmap(i % 20) for i,hw in enumerate(unique_hw)}
        for idx, row in edges_wgs.iterrows():
            geom = row.geometry
            if geom is None:
                # fallback to WKT parse
                if row.geometry_wkt:
                    try:
                        geom = wkt.loads(row.geometry_wkt)
                    except Exception:
                        continue
                else:
                    continue
            # shapely LineString -> coords
            try:
                coords = list(geom.coords)
            except Exception:
                continue
            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]
            hw = row.get('highway', None)
            c = color_map.get(hw, (0.5,0.5,0.5))
            ax.plot(xs, ys, linewidth=0.8, color=c)
        ax.scatter(nodes_out['lon'], nodes_out['lat'], s=6, color='k')
        ax.set_title(f"Walking network from PBF: {args.pbf}")
        ax.set_xlabel("lon"); ax.set_ylabel("lat")
        ax.set_aspect('equal', adjustable='box')

        # jeśli mamy DISPLAY to pokaż, w przeciwnym razie zapisz plik
        if "DISPLAY" in os.environ:
            plt.show()
        else:
            imgfile = f"{args.out_prefix}_plot.png"
            plt.savefig(imgfile, dpi=150, bbox_inches='tight')
            print(f"Brak DISPLAY -> zapisano wykres do {imgfile}")
        plt.close()
    except Exception as e:
        print("Błąd podczas rysowania:", e)

if __name__ == "__main__":
    main()
