"""
ML_Catchment_Splits - building train/val/test catchment splits, gauge networks and catchment maps.
Moved out of ML_Plots.py unchanged, apart from one matplotlib compatibility fix (see plot_geographic_network).
"""
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from sklearn.cluster import KMeans

import geopandas as gpd
import networkx as nx
from shapely.geometry import Point


#### Making Catchment Splits and Maps

def assign_splits_greedy(sets_df, target_ratios=(0.70, 0.15, 0.15),
                         optimize_by='gauges', random_seed=None,
                         top_n_locked= -1):
    """
    top_n_locked: keep the N largest sets deterministic (sorted), 
                  shuffle the rest. Set to 0 to shuffle everything.

    Assign sets to train/val/test splits using a greedy algorithm
    
    Parameters:
    -----------
    sets_df : pd.DataFrame
        DataFrame with columns ['set_id', 'Catchments', 'Total Gauges']
    target_ratios : tuple
        Target ratios for (train, val, test). Default is (0.70, 0.15, 0.15)
    optimize_by : str
        'gauges' or 'catchments' - what to optimize the split for
    
    Returns:
    --------
    pd.DataFrame with added 'split' column
    """

    
    rng = np.random.default_rng(random_seed)
    df = sets_df.copy()
    
    # Lock the largest N sets in place (they dominate the split anyway),
    # shuffle the tail for variability
    df = df.sort_values(by=optimize_by, ascending=False).reset_index(drop=True)
    tail = df.iloc[top_n_locked:].copy()
    tail = tail.iloc[rng.permutation(len(tail))].reset_index(drop=True)
    df = pd.concat([df.iloc[:top_n_locked], tail], ignore_index=True)
    
    
    # Initialize split assignments
    splits = {'train': [], 'val': [], 'test': []}
    split_counts = {'train': 0, 'val': 0, 'test': 0}
    
    total = df[optimize_by].sum()
    target_counts = {
        'train': total * target_ratios[0],
        'val': total * target_ratios[1],
        'test': total * target_ratios[2]
    }
    
    # Greedy assignment: assign each set to the split that needs it most
    for idx, row in df.iterrows():
        set_value = row[optimize_by]
        
        # Calculate how far each split is from its target
        deficits = {
            split: (target_counts[split] - split_counts[split])/target_counts[split]
            # split: target_counts[split] - split_counts[split]
            for split in splits.keys()
        }
        
        # Assign to the split with the largest deficit
        best_split = max(deficits, key=deficits.get)
        
        splits[best_split].append(row['set_id'])
        split_counts[best_split] += set_value
    
    # Create assignment mapping
    split_assignment = {}
    for split, set_ids in splits.items():
        for set_id in set_ids:
            split_assignment[set_id] = split
    
    # Add split column to original dataframe
    result_df = sets_df.copy()
    result_df['split'] = result_df['set_id'].map(split_assignment)
    
    # Print summary
    print(f"\nSplit Assignment Summary (optimized by {optimize_by}):")
    print("=" * 60)
    for split in ['train', 'val', 'test']:
        split_data = result_df[result_df['split'] == split]
        n_sets = len(split_data)
        n_catchments = split_data['Catchments'].sum()
        n_gauges = split_data['Total Gauges'].sum()
        
        print(f"{split.upper():6s}: {n_sets:2d} sets | "
              f"{n_catchments:3d} catchments ({n_catchments/result_df['Catchments'].sum()*100:5.1f}%) | "
              f"{n_gauges:3d} gauges ({n_gauges/result_df['Total Gauges'].sum()*100:5.1f}%)")
    
    return result_df


def plot_geographic_network(stats_df, gdf_catchments, uk_boundary_filtered):
    # 1. Filter for active catchments
    active_stats = stats_df[stats_df['total_gauges'] > 0].copy()
    
    # 2. Build the Graph
    G = nx.Graph()
    for idx, row in active_stats.iterrows():
        # Get centroid for geographic positioning
        centroid = gdf_catchments.loc[idx].geometry.centroid
        G.add_node(idx, pos=(centroid.x, centroid.y), count=int(row['total_gauges']))
    
    # Add Edges based on pair_ids
    exploded = active_stats[['pair_ids']].explode('pair_ids').dropna()
    pair_groups = exploded.reset_index().groupby('pair_ids')['index'].apply(list)
    
    for catch_list in pair_groups:
        unique_catch = list(set(catch_list))
        if len(unique_catch) > 1:
            for i in range(len(unique_catch)):
                for j in range(i + 1, len(unique_catch)):
                    G.add_edge(unique_catch[i], unique_catch[j])

    # 3. Component Coloring
    components = list(nx.connected_components(G))
    colors = plt.get_cmap('tab20')(np.linspace(0, 1, len(components)))
    color_map = {node: colors[i] for i, comp in enumerate(components) for node in comp}

    # 4. Plotting
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Draw the UK base layer
    uk_boundary_filtered.plot(ax=ax, color="lightgray", edgecolor="darkgray", linewidth=1.5, alpha=0.5)
    
    # Extract positions
    pos = nx.get_node_attributes(G, 'pos')
    node_colors = [color_map[n] for n in G.nodes]
    
    # Draw Edges (Links between catchments)
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.5, edge_color='black', width=1)
    
    # Draw Nodes (Catchments)
    nx.draw_networkx_nodes(G, pos, ax=ax, 
                           node_size=100, 
                           node_color=node_colors, 
                           edgecolors='white', 
                           linewidths=1)
    
    # Draw Labels (Gauge Counts)
    labels = {n: str(G.nodes[n]['count']) for n in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=7, font_weight='bold')

    ax.set_title(f"Geographic Gauge Network: {len(components)} Connected Sets", fontsize=16)
    ax.axis('off')
    plt.tight_layout()
    plt.show()


def get_component_stats(stats_df):
    """
    Analyze connected components and return statistics as a DataFrame
    
    Parameters:
    -----------
    stats_df : pd.DataFrame
        DataFrame with catchment statistics including 'total_gauges' and 'pair_ids'
    
    Returns:
    --------
    pd.DataFrame with columns ['set_id', 'Catchments', 'Total Gauges', 'pair_ids', 'catchment_indices']
    """
    # 1. Rebuild the graph (using the logic from the previous step)
    active_stats = stats_df[stats_df['total_gauges'] > 0].copy()
    G = nx.Graph()
    for idx, row in active_stats.iterrows():
        G.add_node(idx, count=int(row['total_gauges']))
    
    exploded = active_stats[['pair_ids']].explode('pair_ids').dropna()
    pair_groups = exploded.reset_index().groupby('pair_ids')['index'].apply(list)
    
    for catch_list in pair_groups:
        unique_catch = list(set(catch_list))
        if len(unique_catch) > 1:
            for i in range(len(unique_catch)):
                for j in range(i + 1, len(unique_catch)):
                    G.add_edge(unique_catch[i], unique_catch[j])
    
    # 2. Analyze Connected Components
    components = list(nx.connected_components(G))
    
    # 3. Build DataFrame with pair_ids included
    component_data = []
    for i, component in enumerate(components, 1):
        # Count catchments in this set
        num_catchments = len(component)
        
        # Sum gauges across all catchments in this set
        total_gauges = active_stats.loc[list(component), 'total_gauges'].sum()
        
        # Collect all pair_ids from catchments in this component
        all_pair_ids = []
        for catchment_idx in component:
            catchment_pair_ids = active_stats.loc[catchment_idx, 'pair_ids']
            if isinstance(catchment_pair_ids, list):
                all_pair_ids.extend(catchment_pair_ids)
            elif pd.notna(catchment_pair_ids):
                all_pair_ids.append(catchment_pair_ids)
        
        # Remove duplicates while preserving order
        unique_pair_ids = list(dict.fromkeys(all_pair_ids))
        
        component_data.append({
            'set_id': i,
            'Catchments': num_catchments,
            'Total Gauges': int(total_gauges),
            'pair_ids': unique_pair_ids
        })
    
    # Create DataFrame
    component_df = pd.DataFrame(component_data)
    
    return component_df


def plot_catchment_sets_distance(stats_df, gauge_gdf=None, distance_threshold_km=13):
    """
    Parameters:
    -----------
    stats_df : pd.DataFrame
        Must have columns: 'total_gauges', 'pair_ids'
    gauge_gdf : GeoDataFrame, optional
        Must have columns: 'set_id', 'geometry' (projected CRS, metres)
        If provided, sets whose gauges are within distance_threshold_m of each
        other will be connected by an edge even if they share no pair_ids.
    distance_threshold_m : float
        Distance in metres below which two sets are considered connected.
        Default 13 000 m (13 km).
    """
    G = nx.Graph()
    stats_df = stats_df[stats_df['total_gauges'] > 0].copy()

    # 1. Add nodes
    for idx, row in stats_df.iterrows():
        G.add_node(idx, count=int(row['total_gauges']))

    # 2. Edges from shared pair_ids
    exploded = stats_df[['pair_ids']].explode('pair_ids').dropna()
    exploded = exploded[exploded['pair_ids'] != 0]
    shared_pairs = exploded.reset_index().groupby('pair_ids')['index'].apply(list)

    for catchments in shared_pairs:
        unique_catchments = list(set(catchments))
        if len(unique_catchments) > 1:
            for i in range(len(unique_catchments)):
                for j in range(i + 1, len(unique_catchments)):
                    G.add_edge(unique_catchments[i], unique_catchments[j],
                               edge_type='shared_pair')

    # 3. NEW: Edges from gauge proximity
    if gauge_gdf is not None:
        pair_id_to_geom = gauge_gdf.groupby('pair_id')['geometry'].apply(list).to_dict()
        
        set_index_to_gauges = {}
        for idx, row in stats_df.iterrows():
            geoms = [geom for p in row['pair_ids']  # pair_ids stays plural here
                     if p in pair_id_to_geom
                     for geom in pair_id_to_geom[p]]
            if geoms:
                set_index_to_gauges[idx] = geoms

        
        # Only consider sets that are actually nodes in the graph
        node_ids = list(G.nodes)

        for i in range(len(node_ids)):
            for j in range(i + 1, len(node_ids)):
                # Skip if already connected
                if G.has_edge(node_ids[i], node_ids[j]):
                    continue

                gauges_i = set_index_to_gauges.get(node_ids[i], [])
                gauges_j = set_index_to_gauges.get(node_ids[j], [])

                if not gauges_i or not gauges_j:
                    continue

                # Find the minimum distance between any gauge in set i and set j
                min_dist = 1e-3 * min(
                    g_i.distance(g_j)
                    for g_i in gauges_i
                    for g_j in gauges_j
                )

                if min_dist < distance_threshold_km:
                                    G.add_edge(node_ids[i], node_ids[j],
                                               edge_type='proximity',
                                               distance_km=round(min_dist, 3))

    # 4. Connected components → colour map (unchanged)
    components = list(nx.connected_components(G))
    colors = cm.rainbow(np.linspace(0, 1, len(components)))
    color_map = {}
    for i, component in enumerate(components):
        for node in component:
            color_map[node] = colors[i]
    node_colors = [color_map[n] for n in G.nodes]

    # 5. Plot — draw proximity edges differently so you can see what fired
    proximity_edges = [(u, v) for u, v, d in G.edges(data=True)
                       if d.get('edge_type') == 'proximity']
    pair_edges      = [(u, v) for u, v, d in G.edges(data=True)
                       if d.get('edge_type') != 'proximity']

    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(G, k=0.3, iterations=50)

    nx.draw_networkx_nodes(G, pos, node_size=800, node_color=node_colors, alpha=0.8)
    nx.draw_networkx_edges(G, pos, edgelist=pair_edges,
                           edge_color='gray', alpha=0.4)
    nx.draw_networkx_edges(G, pos, edgelist=proximity_edges,
                           edge_color='red', alpha=0.6, style='dashed', width=1.5)
    labels = {n: str(G.nodes[n]['count']) for n in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_weight='bold')

    n_prox = len(proximity_edges)
    plt.title(
        f"Catchment Network: {len(components)} Distinct Sets  "
        f"(+{n_prox} proximity edge{'s' if n_prox != 1 else ''} < {distance_threshold_km:.0f} km)",
        fontsize=14
    )
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    
def get_component_stats_distance(stats_df, gauge_gdf=None, distance_threshold_km=13):
    active_stats = stats_df[stats_df['total_gauges'] > 0].copy()
    G = nx.Graph()
    for idx, row in active_stats.iterrows():
        G.add_node(idx, count=int(row['total_gauges']))
    
    # Edges from shared pair_ids
    exploded = active_stats[['pair_ids']].explode('pair_ids').dropna()
    pair_groups = exploded.reset_index().groupby('pair_ids')['index'].apply(list)
    
    for catch_list in pair_groups:
        unique_catch = list(set(catch_list))
        if len(unique_catch) > 1:
            for i in range(len(unique_catch)):
                for j in range(i + 1, len(unique_catch)):
                    G.add_edge(unique_catch[i], unique_catch[j])
    
    # Edges from gauge proximity
    if gauge_gdf is not None:
        pair_id_to_geom = gauge_gdf.groupby('pair_id')['geometry'].apply(list).to_dict()

        set_index_to_gauges = {}
        for idx, row in active_stats.iterrows():
            geoms = [geom for p in row['pair_ids']
                     if p in pair_id_to_geom
                     for geom in pair_id_to_geom[p]]
            if geoms:
                set_index_to_gauges[idx] = geoms

        
        node_ids = list(G.nodes)
        for i in range(len(node_ids)):
            for j in range(i + 1, len(node_ids)):
                if G.has_edge(node_ids[i], node_ids[j]):
                    continue
                gauges_i = set_index_to_gauges.get(node_ids[i], [])
                gauges_j = set_index_to_gauges.get(node_ids[j], [])
                if not gauges_i or not gauges_j:
                    continue
                min_dist = min(
                    g_i.distance(g_j)
                    for g_i in gauges_i
                    for g_j in gauges_j
                )
                if min_dist < distance_threshold_km * 1000:
                    G.add_edge(node_ids[i], node_ids[j])

    # Connected components → DataFrame (unchanged)
    components = list(nx.connected_components(G))
    component_data = []
    for i, component in enumerate(components, 1):
        total_gauges = active_stats.loc[list(component), 'total_gauges'].sum()
        all_pair_ids = []
        for catchment_idx in component:
            catchment_pair_ids = active_stats.loc[catchment_idx, 'pair_ids']
            if isinstance(catchment_pair_ids, list):
                all_pair_ids.extend(catchment_pair_ids)
            elif pd.notna(catchment_pair_ids):
                all_pair_ids.append(catchment_pair_ids)
        
        component_data.append({
            'set_id': i,
            'Catchments': len(component),
            'Total Gauges': int(total_gauges),
            'pair_ids': list(dict.fromkeys(all_pair_ids))
        })


    return pd.DataFrame(component_data)

def plot_catchment_sets(stats_df):
    G = nx.Graph()
    stats_df = stats_df[stats_df['total_gauges'] > 0].copy()
    
    # 1. Add nodes with metadata (gauge counts)
    for idx, row in stats_df.iterrows():
        G.add_node(idx, count=int(row['total_gauges']))
    
    # 2. Add edges for shared pair_ids
    exploded = stats_df[['pair_ids']].explode('pair_ids').dropna()
    # Filter out empty lists or rows with no pair_id
    exploded = exploded[exploded['pair_ids'] != 0] 
    
    shared_pairs = exploded.reset_index().groupby('pair_ids')['index'].apply(list)
    
    for catchments in shared_pairs:
        unique_catchments = list(set(catchments))
        if len(unique_catchments) > 1:
            for i in range(len(unique_catchments)):
                for j in range(i + 1, len(unique_catchments)):
                    G.add_edge(unique_catchments[i], unique_catchments[j])

    # 3. Identify Distinct Sets (Connected Components)
    components = list(nx.connected_components(G))
    color_map = {}
    # Generate colors for each distinct set
    colors = cm.rainbow(np.linspace(0, 1, len(components)))
    
    for i, component in enumerate(components):
        for node in component:
            color_map[node] = colors[i]

    node_colors = [color_map[n] for n in G.nodes]
    
    # 4. Plotting
    plt.figure(figsize=(14, 10))
    # Use 'spring_layout' to pull connected components together and push sets apart
    pos = nx.spring_layout(G, k=0.3, iterations=50)
    
    # Draw Nodes
    nx.draw_networkx_nodes(G, pos, 
                           node_size=800, 
                           node_color=node_colors, 
                           alpha=0.8)
    
    # Draw Edges
    nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.4)
    
    # Draw Labels (The number of gauges inside the node)
    labels = {n: str(G.nodes[n]['count']) for n in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_weight='bold')

    plt.title(f"Catchment Network: {len(components)} Distinct Sets Identified", fontsize=15)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    return components # Returns the list of catchment IDs in each set


def count_gauges_per_catchment(gdf_catchments, res_gdf, nonres_gdf):
    all_gauges = pd.concat([res_gdf, nonres_gdf], ignore_index=True)
    
    if all_gauges.crs != gdf_catchments.crs:
        all_gauges = all_gauges.to_crs(gdf_catchments.crs)

    # Use nearest join to ensure every gauge is assigned
    # We reset index to keep track of the catchment's original ID
    gauges_mapped = gpd.sjoin_nearest(
        all_gauges, 
        gdf_catchments[['geometry']].reset_index().rename(columns={'index': 'catch_id'}), 
        how="left"
    )

    # Aggregate: Count types AND collect pair_ids
    stats_df = gauges_mapped.groupby('catch_id').agg({
        'type': [lambda x: (x == 'reservoir').sum(), lambda x: (x == 'non_reservoir').sum(), 'count'],
        'pair_id': lambda x: list(x.unique())
    })

    # Clean up multi-index columns
    stats_df.columns = ['reservoir_gauges', 'non_reservoir_gauges', 'total_gauges', 'pair_ids']
    
    # Ensure all original catchments are present (even those with 0 gauges)
    stats_df = pd.DataFrame(index=gdf_catchments.index).join(stats_df).fillna(0)
    stats_df['pair_ids'] = stats_df['pair_ids'].apply(lambda x: x if isinstance(x, list) else [])
    stats_df['area'] = gdf_catchments.geometry.area
    
    return stats_df.sort_values('total_gauges', ascending=False)


def prepare_gauge_data(pairs_df, target_crs):
    """Prepare reservoir and non-reservoir gauge GeoDataFrames"""
    
    # Reservoir gauges
    res_geometry = [
        Point(lon, lat)
        for lon, lat in zip(pairs_df["reservoir_lon"], pairs_df["reservoir_lat"])
    ]
    
    # Handle optional 'split' column
    cols_to_keep = ["reservoir_gauge_id", "reservoir_gauge_name"]
    if "split" in pairs_df.columns:
        cols_to_keep.append("split")
    
    res_gdf = gpd.GeoDataFrame(
        pairs_df[cols_to_keep].copy(),
        geometry=res_geometry,
        crs="EPSG:4326",
    )
    res_gdf["type"] = "reservoir"
    res_gdf = res_gdf.rename(
        columns={"reservoir_gauge_id": "gauge_id", "reservoir_gauge_name": "gauge_name"}
    )
    
    # Non-reservoir gauges
    nonres_geometry = [
        Point(lon, lat)
        for lon, lat in zip(pairs_df["non_reservoir_lon"], pairs_df["non_reservoir_lat"])
    ]
    
    cols_to_keep = ["non_reservoir_gauge_id", "non_reservoir_gauge_name"]
    if "split" in pairs_df.columns:
        cols_to_keep.append("split")
    
    nonres_gdf = gpd.GeoDataFrame(
        pairs_df[cols_to_keep].copy(),
        geometry=nonres_geometry,
        crs="EPSG:4326",
    )
    nonres_gdf["type"] = "non_reservoir"
    nonres_gdf = nonres_gdf.rename(
        columns={
            "non_reservoir_gauge_id": "gauge_id",
            "non_reservoir_gauge_name": "gauge_name",
        }
    )
    
    # Convert to same CRS as catchments
    res_gdf = res_gdf.to_crs(target_crs)
    nonres_gdf = nonres_gdf.to_crs(target_crs)
    
    return res_gdf, nonres_gdf


def assign_gauges_to_catchments(gdf_catchments, res_gdf, nonres_gdf):
    """Assign each gauge to its containing catchment"""
    
    print("Assigning gauges to catchments...")
    
    # For each gauge, find which catchment it's in
    def find_catchment(gauge_row, catchments):
        for idx, catchment in catchments.iterrows():
            if gauge_row.geometry.within(catchment.geometry):
                return idx
        return None
    
    res_gdf['catchment_idx'] = res_gdf.apply(
        lambda row: find_catchment(row, gdf_catchments), axis=1
    )
    nonres_gdf['catchment_idx'] = nonres_gdf.apply(
        lambda row: find_catchment(row, gdf_catchments), axis=1
    )
    
    return res_gdf, nonres_gdf
    
def plot_catchments_with_gauges(catchments_gdf, res_gdf, nonres_gdf, 
                                 output_file="camelsgb_catchments_with_gauges.png"):
    """Plot catchments with gauge locations"""
    
    fig, ax = plt.subplots(figsize=(16, 14))
    MARKER_SIZE = 20
    # Plot catchments
    catchments_gdf.plot(ax=ax, 
                     edgecolor='darkblue', 
                     facecolor='lightblue', 
                     alpha=0.4, 
                     linewidth=0.6,
                     label='Catchments')
    
    # Plot reservoir gauges
    res_gdf.plot(ax=ax, 
                color='red', 
                markersize=MARKER_SIZE,
                marker='o',
                alpha=0.7,
                label='Reservoir Gauges',
                edgecolor='darkred',
                linewidth=0.5,
                zorder=5)
    
    # Plot non-reservoir gauges
    nonres_gdf.plot(ax=ax, 
                   color='green', 
                   markersize=MARKER_SIZE,
                   marker='^',
                   alpha=0.7,
                   label='Non-Reservoir Gauges',
                   edgecolor='darkgreen',
                   linewidth=0.5,
                   zorder=5)
    
    ax.set_title('CAMELS GB Catchments with Gauge Locations', 
                 fontsize=18, 
                 fontweight='bold',
                 pad=20)
    ax.set_xlabel('Easting (m)', fontsize=12)
    ax.set_ylabel('Northing (m)', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=12, framealpha=0.9)
    
    plt.tight_layout()
    plt.show()


def load_and_filter_catchments(shapefile_dir):
    """Load catchment shapefiles and filter out subsets"""
    shapefile_dir = Path(shapefile_dir)
    
    # Find shapefiles
    shp_files = list(shapefile_dir.glob("*.shp"))
    if not shp_files:
        shp_files = list(shapefile_dir.glob("**/*.shp"))
    
    if not shp_files:
        raise FileNotFoundError(f"No shapefiles found in {shapefile_dir}")
    
    # Read shapefile
    print(f"Reading: {shp_files[0]}")
    gdf = gpd.read_file(shp_files[0])
    print(f"Original number of catchments: {len(gdf)}")
    
    # Filter out subset catchments
    print("Filtering subset catchments...")
    keep_indices = []
    
    for i, geom_i in enumerate(gdf.geometry):
        is_subset = False
        for j, geom_j in enumerate(gdf.geometry):
            if i != j and geom_i.within(geom_j):
                is_subset = True
                break
        if not is_subset:
            keep_indices.append(i)
    
    catchments_gdf = gdf.iloc[keep_indices].copy()
    print(f"Filtered number of catchments: {len(catchments_gdf)}")
    print(f"Removed {len(gdf) - len(catchments_gdf)} subset catchments\n")
    
    return catchments_gdf

def plot_catchments_only(catchments_gdf, output_file="camelsgb_catchments_filtered.png"):
    """Plot catchments only"""
    fig, ax = plt.subplots(figsize=(14, 12))
    
    catchments_gdf.plot(ax=ax, 
                      edgecolor='darkblue', 
                      facecolor='lightblue', 
                      alpha=0.6, 
                      linewidth=0.8)
    
    ax.set_title('CAMELS GB Catchment Boundaries (Subsets Removed)', 
                 fontsize=18, 
                 fontweight='bold',
                 pad=20)
    ax.set_xlabel('Easting (m)', fontsize=12)
    ax.set_ylabel('Northing (m)', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.show()


def create_geographic_split(
    pairs_df, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, random_state=42
):
    """
    Create geographically separated train/val/test splits
    """
    n_pairs = len(pairs_df)
    target_train = int(train_ratio * n_pairs)
    target_val = int(val_ratio * n_pairs)
    target_test = n_pairs - target_train - target_val

    coords = pairs_df[["reservoir_lat", "reservoir_lon"]].values

    # Try different numbers of clusters and select best
    best_split = None
    best_score = float("inf")

    for n_clusters in range(3, 8):
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=50)
        labels = kmeans.fit_predict(coords)

        # Group clusters to match target sizes
        cluster_sizes = pd.Series(labels).value_counts().to_dict()

        # Try to assign clusters to splits
        all_clusters = list(cluster_sizes.keys())

        # Try all possible ways to partition clusters into 3 groups
        best_partition = None
        min_diff = float("inf")

        for train_clusters in combinations(all_clusters, len(all_clusters) // 2 + 1):
            train_size = sum(cluster_sizes[c] for c in train_clusters)
            remaining = [c for c in all_clusters if c not in train_clusters]

            for val_clusters in combinations(remaining, len(remaining) // 2):
                val_size = sum(cluster_sizes[c] for c in val_clusters)
                test_clusters = [c for c in remaining if c not in val_clusters]
                test_size = sum(cluster_sizes[c] for c in test_clusters)

                # Calculate how close to target
                diff = (
                    abs(train_size - target_train)
                    + abs(val_size - target_val)
                    + abs(test_size - target_test)
                )

                if diff < min_diff:
                    min_diff = diff
                    best_partition = (train_clusters, val_clusters, test_clusters)

        if min_diff < best_score:
            best_score = min_diff
            best_split = (labels, best_partition)

    # Apply best split
    labels, (train_clusters, val_clusters, test_clusters) = best_split

    pairs_df["split"] = "train"
    pairs_df.loc[[i for i, l in enumerate(labels) if l in val_clusters], "split"] = (
        "val"
    )
    pairs_df.loc[[i for i, l in enumerate(labels) if l in test_clusters], "split"] = (
        "test"
    )
    return pairs_df
