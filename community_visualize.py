import os
import argparse
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import networkx as nx

from config import get_args

class CommunityLandscapeVisualizer: 
    def __init__(
        self,
        figsize: Tuple[int, int] = (16, 12),
        dpi: int = 150,
        target_fill_rate: float = 0.35,
        min_alpha: float = 0.15,
        max_alpha: float = 0.90,
        edge_width: float = 0.15,
        cmap: str = 'RdYlBu_r'
    ):
        self.figsize = figsize
        self.dpi = dpi
        self.target_fill_rate = target_fill_rate
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.edge_width = edge_width
        self.cmap = cmap
    
    def normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        def min_max_normalize(values):
            max_val = values.max()
            if max_val < 1e-10:
                return np.ones(len(values))
            return values / max_val
        
        df['size_norm'] = df.groupby('dataset')['size'].transform(min_max_normalize)
        
        return df
    
    def compute_uniform_circle_layout(
        self,
        n_communities: int,
        seed: int = 0
    ) -> np.ndarray:
        np.random.seed(seed)
        theta = np.random.uniform(0, 2 * np.pi, n_communities)
        radius = np.sqrt(np.random.uniform(0, 1, n_communities))
        
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        initial_pos = np.column_stack([x, y])
        G = nx.Graph()
        pos_dict = {}
        fixed_nodes = []
        
        for i in range(n_communities):
            G.add_node(i)
            pos_dict[i] = initial_pos[i]
            
            anchor_i = i + n_communities
            G.add_node(anchor_i)
            pos_dict[anchor_i] = initial_pos[i] 
            fixed_nodes.append(anchor_i) 
            G.add_edge(i, anchor_i, weight=0.8)
        k = 1.0 / np.sqrt(n_communities)
        optimized_pos = nx.spring_layout(
            G,
            pos=pos_dict,
            fixed=fixed_nodes,
            k=k,
            iterations=60,    
            seed=seed,
            scale=None  
        )
        positions = np.array([optimized_pos[i] for i in range(n_communities)])
        
        return positions
    def compute_auto_scale(
        self,
        positions: np.ndarray,
        sizes_norm: np.ndarray
    ) -> float:
        x_range = positions[:, 0].max() - positions[:, 0].min()
        y_range = positions[:, 1].max() - positions[:, 1].min()
        canvas_area = x_range * y_range
        
        total_bubble_area = np.sum(np.pi * np.power(sizes_norm, 2/5))
        
        current_fill = total_bubble_area / canvas_area if canvas_area > 0 else 1.0
        scale = np.sqrt(self.target_fill_rate / current_fill) if current_fill > 0 else 1.0
        
        return scale
    
    def create_landscape(
        self,
        df_dataset: pd.DataFrame,
        dataset_name: str,
        output_path_base: str
    ):
        n_communities = len(df_dataset)
        print(f"[{dataset_name}] Visualizing {n_communities} communities...")
        
        df_sorted = df_dataset.sort_values('size', ascending=False).reset_index(drop=True)
        
        sizes_norm = df_sorted['size_norm'].values
        density = df_sorted['density'].values
        
        positions = self.compute_uniform_circle_layout(n_communities)
        
        scale = self.compute_auto_scale(positions, sizes_norm)
        
        radii = np.power(sizes_norm, 1/5) * scale
        
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        ax.set_aspect('equal')
        ax.axis('off')
        
        circles = [
            mpatches.Circle((positions[i, 0], positions[i, 1]), radii[i])
            for i in range(n_communities)
        ]
        
        collection = PatchCollection(circles, match_original=False)
        
        norm = plt.Normalize(vmin=0.0, vmax=1.0)
        
        collection.set_cmap(self.cmap)
        collection.set_norm(norm)
        collection.set_array(density)
        
        collection.set_edgecolors('white')
        collection.set_linewidths(self.edge_width)
        
        ax.add_collection(collection)
        
        x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
        y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
        
        max_radius = radii.max()
        padding = max_radius * 1.5
        
        ax.set_xlim(x_min - padding, x_max + padding)
        ax.set_ylim(y_min - padding, y_max + padding)
        
        plt.tight_layout()
        png_path = f"{output_path_base}.png"
        pdf_path = f"{output_path_base}.pdf"
        svg_path = f"{output_path_base}.svg"
        plt.savefig(png_path, dpi=self.dpi, bbox_inches='tight')
        plt.savefig(pdf_path, bbox_inches='tight')
        plt.savefig(svg_path, bbox_inches='tight')
        plt.close()
        
        print(f"[{dataset_name}] Saved to: {png_path}, {pdf_path}, and {svg_path}")
    
    def create_colorbar(self, output_path_base: str):
        fig = plt.figure(figsize=(2, 20))
        cax = fig.add_axes([0.1, 0.05, 0.15, 0.9])
        norm = plt.Normalize(vmin=0.0, vmax=1.0)
        cbar = plt.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=self.cmap),
            cax=cax,
            orientation='vertical'
        )
        cbar.set_label('Internal density', fontsize=40)
        cbar.ax.tick_params(labelsize=40)
        pdf_path = f"{output_path_base}.pdf"
        svg_path = f"{output_path_base}.svg"
        plt.savefig(pdf_path, bbox_inches='tight')
        plt.savefig(svg_path, bbox_inches='tight')
        plt.close()
        
        print(f"Colorbar saved to: {pdf_path} and {svg_path}")
        
        fig2, ax2 = plt.subplots(figsize=(0.5, 10))
        ax2.axis('off')
        
        norm2 = plt.Normalize(vmin=0.0, vmax=1.0)
        cbar2 = plt.colorbar(
            plt.cm.ScalarMappable(norm=norm2, cmap=self.cmap),
            ax=ax2,
            orientation='vertical',
            fraction=0.8
        )
        cbar2.set_label('Internal density', fontsize=40)
        cbar2.ax.tick_params(labelsize=40)
        
        pdf_path2 = f"{output_path_base}_2.pdf"
        svg_path2 = f"{output_path_base}_2.svg"
        plt.savefig(pdf_path2, bbox_inches='tight')
        plt.savefig(svg_path2, bbox_inches='tight')
        plt.close()


def visualize_dataset(
    dataset_name: str,
    input_dir: str,
    output_dir: str,
    figsize: Tuple[int, int] = (16, 12),
    dpi: int = 150
):
    os.makedirs(output_dir, exist_ok=True)
    
    input_path = Path(input_dir)
    csv_file = input_path / f"{dataset_name}_community_stats.csv"
    
    if not csv_file.exists():
        print(f"CSV file not found: {csv_file}")
        return
    
    print(f"Visualizing dataset: {dataset_name}")
    print("=" * 70)
    
    visualizer = CommunityLandscapeVisualizer(figsize=figsize, dpi=dpi)
    
    print(f"\nProcessing: {dataset_name}")
    print("-" * 70)
    
    df = pd.read_csv(csv_file)
    
    df_norm = visualizer.normalize_features(df)
    
    output_path_base = os.path.join(output_dir, dataset_name)
    visualizer.create_landscape(df_norm, dataset_name, output_path_base)
    
    colorbar_path_base = os.path.join(output_dir, 'colorbar')
    visualizer.create_colorbar(colorbar_path_base)
    
    print("\n" + "=" * 70)
    print(f"Visualization completed! Output saved to: {output_dir}")


if __name__ == '__main__':
    args = get_args()
    input_dir = os.path.join(args.result_path, "community_stats")
    output_dir = os.path.join(args.result_path, "community_landscape")
    visualize_dataset(
        dataset_name=args.dataset,
        input_dir=input_dir,
        output_dir=output_dir
    )

