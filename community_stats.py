import os
import time
import argparse
from typing import List, Set, Dict
import numpy as np
import pandas as pd
import networkx as nx

from config import get_args


def read_graph(graph_file: str) -> nx.Graph:
    G = nx.Graph()
    with open(graph_file, 'r') as file:
        edges = []
        for line in file:
            parts = line.strip().split()
            if len(parts) >= 2:
                node1, node2 = int(parts[0]), int(parts[1])
                edges.append((node1, node2))
        G.add_edges_from(edges)
    return G


def read_communities(community_file: str) -> List[Set[int]]:
    communities = []
    with open(community_file, 'r') as file:
        for line in file:
            nodes = [int(x) for x in line.strip().split() if x]
            if nodes:
                communities.append(set(nodes))
    return communities


def compute_community_size(community: Set[int]) -> int:
    return len(community)


def compute_community_density(g: nx.Graph, community: Set[int]) -> float:
    if len(community) <= 1:
        return 0.0
    
    sub = g.subgraph(community)
    m = sub.number_of_edges()
    n = sub.number_of_nodes()
    
    if n <= 1:
        return 0.0
    
    return 2.0 * m / (n * (n - 1))


def community_separability(g: nx.Graph, community: Set[int]) -> float:
    if not community:
        return 0.0
    
    com_set = community
    
    subgraph = g.subgraph(com_set)
    internal_edges = subgraph.number_of_edges()
    
    external_edges = 0
    for v in com_set:
        if v not in g:
            continue
        for nei in g.neighbors(v):
            if nei not in com_set:
                external_edges += 1
    
    if external_edges == 0:
        return 999999.0 if internal_edges > 0 else 0.0
    
    if internal_edges == 0:
        return 0.0
    
    return internal_edges / external_edges


def analyze_dataset_communities(dataset_name: str, data_path: str) -> pd.DataFrame:
    print(f"\n{'='*80}")
    print(f"Analyzing dataset: {dataset_name}")
    print(f"{'='*80}")
    
    dataset_path = os.path.join(data_path, dataset_name)
    graph_file = os.path.join(dataset_path, "graph")
    community_file = os.path.join(dataset_path, "community")
    
    if not os.path.exists(graph_file):
        return pd.DataFrame()
    
    if not os.path.exists(community_file):
        return pd.DataFrame()
    
    t0 = time.time()
    g = read_graph(graph_file)
    t1 = time.time()
    t0 = time.time()
    communities = read_communities(community_file)
    num_comms = len(communities)
    t1 = time.time()
    
    if num_comms == 0:
        print(f"[{dataset_name}] No communities found!")
        return pd.DataFrame()
    
    t0 = time.time()
    
    results = []
    for idx, comm in enumerate(communities):
        size = compute_community_size(comm)
        density = compute_community_density(g, comm)
        
        results.append({
            'dataset': dataset_name,
            'community_id': idx,
            'size': size,
            'density': density
        })
    
    t1 = time.time()
    df = pd.DataFrame(results)
    
    print(f"\n[{dataset_name}] Summary Statistics:")
    print(f"  Total communities: {len(df)}")
    print(f"  Size - Min: {df['size'].min()}, Max: {df['size'].max()}, Mean: {df['size'].mean():.2f}, Median: {df['size'].median():.2f}")
    print(f"  Density - Min: {df['density'].min():.4f}, Max: {df['density'].max():.4f}, Mean: {df['density'].mean():.4f}, Median: {df['density'].median():.4f}")
    
    return df


def main():
    args = get_args()
    data_path = args.data_path
    datasets = []
    datasets.append(args.dataset)
    output_dir = os.path.join(args.result_path, "community_stats")
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = []
    for dataset in sorted(datasets):
        try:
            df = analyze_dataset_communities(dataset, data_path)
            
            if not df.empty:
                output_file = os.path.join(output_dir, f"{dataset}_community_stats.csv")
                df.to_csv(output_file, index=False)
                print(f"[{dataset}] Results saved to: {output_file}")
                
                all_results.append(df)
            else:
                print(f"[{dataset}] No results to save.")
                
        except Exception as e:
            print(f"[{dataset}] ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        combined_file = os.path.join(output_dir, "all_datasets_community_stats.csv")
        combined_df.to_csv(combined_file, index=False)
        print(f"\n{'='*80}")
        print(f"All results combined and saved to: {combined_file}")
        print(f"Total communities analyzed: {len(combined_df)}")
        print(f"{'='*80}")
        
        print("\nOverall Statistics by Dataset:")
        print("-" * 80)
        summary = combined_df.groupby('dataset').agg({
            'size': ['count', 'mean', 'median', 'min', 'max'],
            'density': ['mean', 'median', 'min', 'max']
        }).round(4)
        print(summary)
    else:
        print("\nNo results to combine.")


if __name__ == "__main__":
    main()

