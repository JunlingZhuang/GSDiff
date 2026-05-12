import os

import imageio
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import wandb


MSD_NODE_NAMES = [
    "Bedroom",
    "Living",
    "Kitchen",
    "Dining",
    "Corridor",
    "Stairs",
    "Storage",
    "Bathroom",
    "Balcony",
]

MSD_NODE_COLORS = [
    "#8da0cb",
    "#47b39c",
    "#f1c27d",
    "#fdae6b",
    "#fdd0a2",
    "#72246c",
    "#ffd92f",
    "#bdbdbd",
    "#a6d854",
]

MSD_EDGE_COLORS = {
    "wall": "#334155",
    "passage": "#64748b",
    "door": "#c2410c",
    "entrance": "#d97706",
}


class MSDSamplingMetrics(nn.Module):
    """Lightweight metrics for phase-1 MSD unconditional graph generation."""

    def __init__(self, datamodule):
        super().__init__()
        self.train_graphs = self.loader_to_nx(datamodule.train_dataloader())
        self.val_graphs = self.loader_to_nx(datamodule.val_dataloader())
        self.test_graphs = self.loader_to_nx(datamodule.test_dataloader())

    def loader_to_nx(self, loader):
        graphs = []
        for batch in loader:
            for data in batch.to_data_list():
                graph = nx.Graph()
                node_classes = torch.argmax(data.x, dim=-1).tolist()
                for idx, node_class in enumerate(node_classes):
                    graph.add_node(idx, attr=int(node_class))

                edge_index = data.edge_index.t().tolist()
                edge_types = torch.argmax(data.edge_attr, dim=-1).tolist()
                for (src, dst), edge_type in zip(edge_index, edge_types):
                    if src < dst and edge_type > 0:
                        graph.add_edge(int(src), int(dst), edge_type=int(edge_type))
                graphs.append(graph)
        return graphs

    def forward(self, generated_graphs, name, current_epoch, val_counter, local_rank, test=False):
        reference_graphs = self.test_graphs if test else self.val_graphs
        generated_nx = [self.sample_to_nx(sample) for sample in generated_graphs]
        metrics = self.compute_metrics(reference_graphs, generated_nx)

        if local_rank == 0:
            print("MSD sampling statistics", metrics)
        if wandb.run:
            wandb.log(metrics, commit=False)

    @staticmethod
    def sample_to_nx(sample):
        node_types, edge_types = sample
        node_types = node_types.detach().cpu().long()
        edge_types = edge_types.detach().cpu().long()

        graph = nx.Graph()
        for idx, node_type in enumerate(node_types.tolist()):
            graph.add_node(idx, attr=int(node_type))

        for i in range(edge_types.shape[0]):
            for j in range(i + 1, edge_types.shape[1]):
                edge_type = int(edge_types[i, j])
                if edge_type > 0:
                    graph.add_edge(i, j, edge_type=edge_type)
        return graph

    @staticmethod
    def compute_metrics(reference_graphs, generated_graphs):
        def safe_mean(values):
            return float(np.mean(values)) if values else 0.0

        valid_generated = [graph for graph in generated_graphs if graph.number_of_nodes() > 0]
        return {
            "sampling/ref_avg_nodes": safe_mean([graph.number_of_nodes() for graph in reference_graphs]),
            "sampling/ref_avg_edges": safe_mean([graph.number_of_edges() for graph in reference_graphs]),
            "sampling/gen_avg_nodes": safe_mean([graph.number_of_nodes() for graph in generated_graphs]),
            "sampling/gen_avg_edges": safe_mean([graph.number_of_edges() for graph in generated_graphs]),
            "sampling/gen_connected_frac": safe_mean([
                nx.is_connected(graph) for graph in valid_generated
            ]),
            "sampling/gen_has_edge_frac": safe_mean([
                graph.number_of_edges() > 0 for graph in generated_graphs
            ]),
        }

    def reset(self):
        pass


class MSDVisualization:
    def __init__(self, dataset_infos=None):
        self.dataset_infos = dataset_infos
        self.node_names = getattr(dataset_infos, "node_decoder", MSD_NODE_NAMES) if dataset_infos else MSD_NODE_NAMES
        self.edge_names = getattr(dataset_infos, "edge_decoder", ["none", "passage", "door", "entrance"])
        self.edge_colors = {
            idx: MSD_EDGE_COLORS.get(name, "#64748b")
            for idx, name in enumerate(self.edge_names)
            if idx > 0
        }

    def sample_to_nx(self, node_list, adjacency_matrix):
        graph = nx.Graph()
        for idx, node_type in enumerate(node_list):
            graph.add_node(idx, attr=int(node_type))

        rows, cols = np.where(adjacency_matrix >= 1)
        for src, dst in zip(rows.tolist(), cols.tolist()):
            if src < dst:
                graph.add_edge(src, dst, edge_type=int(adjacency_matrix[src, dst]))
        return graph

    def visualize(self, path, graphs, num_graphs_to_visualize, log="graph"):
        os.makedirs(path, exist_ok=True)
        num_graphs_to_visualize = min(num_graphs_to_visualize, len(graphs))
        for idx in range(num_graphs_to_visualize):
            node_types, edge_types = graphs[idx]
            graph = self.sample_to_nx(node_types.numpy(), edge_types.numpy())
            file_path = os.path.join(path, f"msd_graph_{idx}.png")
            self.draw_graph(graph, file_path)
            if wandb.run and log is not None:
                wandb.log({log: wandb.Image(file_path)}, commit=False)

    def visualize_chain(self, path, nodes_list, adjacency_matrix):
        os.makedirs(path, exist_ok=True)
        final_graph = self.sample_to_nx(nodes_list[-1], adjacency_matrix[-1])
        pos = nx.spring_layout(final_graph, seed=0) if final_graph.number_of_nodes() else {}

        frame_paths = []
        for frame in range(nodes_list.shape[0]):
            graph = self.sample_to_nx(nodes_list[frame], adjacency_matrix[frame])
            file_path = os.path.join(path, f"frame_{frame}.png")
            self.draw_graph(graph, file_path, pos=pos)
            frame_paths.append(file_path)

        if frame_paths:
            imgs = [imageio.imread(file_path) for file_path in frame_paths]
            gif_path = os.path.join(os.path.dirname(path), f"{os.path.basename(path)}.gif")
            imgs.extend([imgs[-1]] * 10)
            imageio.mimsave(gif_path, imgs, subrectangles=True, duration=20)
            if wandb.run:
                wandb.log({"chain": wandb.Video(gif_path, fps=5, format="gif")}, commit=False)

    def draw_graph(self, graph, file_path, pos=None):
        plt.figure(figsize=(6.0, 5.4))
        if graph.number_of_nodes() == 0:
            plt.axis("off")
            plt.savefig(file_path, bbox_inches="tight")
            plt.close("all")
            return

        if pos is None:
            pos = nx.spring_layout(graph, seed=0, k=0.9, iterations=100)

        node_colors = [
            MSD_NODE_COLORS[graph.nodes[node].get("attr", 0) % len(MSD_NODE_COLORS)]
            for node in graph.nodes()
        ]
        edge_colors = [
            self.edge_colors.get(graph.edges[edge].get("edge_type", 1), "#64748b")
            for edge in graph.edges()
        ]
        labels = {
            node: self.node_names[graph.nodes[node].get("attr", 0) % len(self.node_names)]
            for node in graph.nodes()
        }

        nx.draw_networkx_nodes(
            graph,
            pos,
            node_color=node_colors,
            node_size=820,
            edgecolors="black",
            linewidths=0.8,
        )
        nx.draw_networkx_edges(graph, pos, edge_color=edge_colors, width=1.5)
        nx.draw_networkx_labels(graph, pos, labels=labels, font_size=6)

        plt.axis("off")
        plt.tight_layout()
        plt.savefig(file_path, dpi=140)
        plt.close("all")
