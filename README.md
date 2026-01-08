# 15MC-Algorithm

## Project Overview
15 Minute City Algorithm repository aims to deliver an end-to-end algorithm for finding routes from a residential location to selected Points of Interest (POI). It covers the full data pipeline: from map acquisition and preprocessing, through POI detection and locality-aware data collection around the residence, to persisting results in a database.

## Project Structure

The 15MC-Algorithm contains the following main components:

### **python_scripts/** - Core Data Processing Scripts

#### 1. **graph_construction.py**
Builds a pedestrian graph from a .pbf (OpenStreetMap) tile and saves it in Compressed Sparse Row (CSR) format.
- **Input**: OSM .pbf tile file
- **Output**: `{prefix}_csr.npz` with graph data (indptr, indices, weights, lonlat, osm_node_id)
- **Usage**: `python graph_construction.py --pbf INPUT.pbf --out-prefix OUTPUT_PREFIX [--no-plot] [--dump-csv] [--show-plot]`

#### 2. **grid_creation.py**
Creates a grid of points across a geographic region based on bounding box and tile size.
- **Input**: Bounding box and tile/buffer sizes in km
- **Output**: JSON file with grid cell definitions and bounding boxes
- **Usage**: `python grid_creation.py --bbox MIN_LON,MIN_LAT,MAX_LON,MAX_LAT --tile-km SIZE --buffer-km BUFFER --out-json OUTPUT.json [--out-png PNG_FILE] [--select LON,LAT] [--country-geojson GEOJSON]`

#### 3. **grid_extraction_script.py**
Extracts a single circular geographic region from a large OSM file around a specific point.
- **Input**: Source .pbf file, center coordinates (lat/lon), radius in meters
- **Output**: Smaller .pbf file with extracted region
- **Usage**: `python grid_extraction_script.py --pbf INPUT.pbf --lat LATITUDE --lon LONGITUDE --radius METERS --out OUTPUT.pbf [--no-docker-fallback] [--relations]`

#### 4. **extract_map_from_json.py**
Batch extracts multiple geographic tiles from a large OSM file using a JSON grid definition.
- **Input**: Source .pbf file, JSON grid file (from `grid_creation.py`)
- **Output**: Multiple .pbf files (one per grid cell) in output directory
- **Usage**: `python extract_map_from_json.py --pbf INPUT.pbf --json GRID.json --out-dir OUTPUT_DIR [--use tile|buffer] [--limit N] [--grid-file GRID_IDS.txt] [--relations] [--delete-empty] [--no-docker-fallback]`

#### 5. **snap_poi_to_nodes.py**
Snaps Points of Interest (POIs) from filtered PBF files to the nearest nodes in the graph.
- **Input**: Filtered .pbf file (with POIs), CSR .npz file
- **Output**: Parquet file with POI data (poi_id, category, node_idx, lon, lat, name, dist_to_node_m)
- **Usage**: `python snap_poi_to_nodes.py --pbf INPUT.pbf --csr GRAPH.npz --out OUTPUT.parquet`

#### 6. **precompute_poi_reach.py**
Precomputes distances and reachability to POIs from every node using multi-source Dijkstra algorithm.
- **Input**: CSR .npz graph file, POI parquet/csv file
- **Output**: `{prefix}_precompute.npz` with distance, time, and nearest POI per category
- **Usage**: `python precompute_poi_reach.py --csr GRAPH.npz --pois POIS.parquet --out-prefix OUTPUT_PREFIX [--limit-m METERS] [--limit-min MINUTES] [--speed-mps SPEED] [--cats CATEGORY1 CATEGORY2 ...] [--no-summary]`

#### 7. **poi_query.py**
Queries POI availability from a given location based on precomputed data.
- **Input**: CSR .npz file, precomputed .npz file, query coordinates (lat/lon)
- **Output**: JSON with available POIs by category and distance/time to reach
- **Usage**: `python poi_query.py --csr GRAPH.npz --precompute PRECOMPUTE.npz --lat LATITUDE --lon LONGITUDE [--json]`

#### 8. **orchestrator.py**
Automated pipeline orchestrator that runs all processing steps in sequence for all tiles.
- **Input**: Directory with .pbf tile files
- **Output**: Workspace directory with graph, POI, and precomputed artifacts
- **Usage**: `python orchestrator.py --tiles-dir DIR --workspace DIR [--python PYTHON_BIN]--graph-script SCRIPT_Path --snap-script SCRIPT_Path --prec-script SCRIPT_Path [--limit-m METERS] [--speed-mps SPEED] [--cats CATEGORY ...] [--skip-plot] [--dump-csv] [--stop-on-error]`

#### 9. **automatic_poi_query.py**
Batch POI queries across multiple locations or points of interest.
- **Input**: CSR .npz file, precomputed .npz file, list of query coordinates
- **Output**: Results for each query location in JSON format
- **Usage**: `python automatic_poi_query.py --csr GRAPH.npz --precompute PRECOMPUTE.npz --lat LATITUDE --lon LONGITUDE [--json]`

#### 10. **json_to_txt_conversion.py**
Converts JSON-formatted data to text format for integration with other systems.
- **Input**: JSON file with data
- **Output**: Text file with converted data
- **Usage**: `python json_to_txt_conversion.py INPUT.json OUTPUT.txt`

#### 11. **test_path.py**
Testing utility for validating path calculations and route queries.
- **Input**: Graph files and test parameters
- **Output**: Validation results and test reports
- **Usage**: `python test_path.py [PARAMETERS]`

#### 12. **validate_csr.py**
Validates the integrity and correctness of CSR graph files.
- **Input**: CSR .npz file
- **Output**: Validation report and statistics
- **Usage**: `python validate_csr.py CSR_FILE.npz`

#### 13. **validate_dist.py**
Validates distance calculations and precomputed distance matrices.
- **Input**: Precomputed .npz file and CSR graph file
- **Output**: Validation report with distance metrics
- **Usage**: `python validate_dist.py PRECOMPUTE.npz CSR_FILE.npz`

###  **osm_scripts/** - OSM Data Filtering

#### 1. **osm_full_data_filter_script.ps1**
PowerShell script for filtering full OSM country datasets to remove non-pedestrian ways.
- **Input**: Full .pbf file for a country/region
- **Output**: Filtered .pbf file with only pedestrian-accessible ways
- **Usage**: `powershell osm_full_data_filter_script.ps1 INPUT.pbf OUTPUT.pbf`

#### 2. **osm_poi_filter_script.txt**
Text-based specifications and documentation for POI filtering from OSM data.
- **Input**: POI tag specifications (defines which OSM tags constitute POIs)
- **Output**: Configuration reference for POI extraction
- **Usage**: Reference file for POI category definitions

###  **integration/** - Backend Integration
- **main.py**: Integration point with backend services or external APIs

###  **Libs/** - Utility Libraries
- **environment.yml**: Conda environment specification with all dependencies

###  **data/** - Sample Data and Documentation
- Contains example datasets and supplementary files

###  **workspace/** - Output Directory
- Default location for all generated artifacts (graphs, precompute files, results)

---

## Typical Workflow

1. **Data Preparation**
   - Obtain OSM .pbf files for target regions
   - Filter using `osm_full_data_filter_script.ps1`

2. **Grid Setup**
   - Define bounding box using `grid_creation.py`
   - Extract tiles using `extract_map_from_json.py`

3. **Graph Construction**
   - Build pedestrian graphs: `graph_construction.py`
   - Output: CSR format graph files

4. **POI Snapping**
   - Snap POIs to graph nodes: `snap_poi_to_nodes.py`
   - Output: Parquet files with POI-node mappings

5. **Precomputation**
   - Run Dijkstra on all categories: `precompute_poi_reach.py`
   - Output: Precomputed distance/time matrices

6. **Automation (Optional)**
   - Use `orchestrator.py` to automate steps 3-5

7. **Querying**
   - Query POI accessibility: `poi_query.py` or `automatic_poi_query.py`
   - Integrate with backend via `integration/main.py`

---

## Setup & Requirements

- **Python Environment**: See `Libs/environment.yml`
- **Dependencies**: pyrosm, geopandas, shapely, scipy, numpy, pandas, networkx
- **Data**: OSM .pbf files for target regions

---


