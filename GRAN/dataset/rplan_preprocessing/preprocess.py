"""Standalone RPLAN raw PNG -> GRAN training data pipeline.

Reads 4-channel RPLAN PNGs directly (no dependency on other projects'
preprocessing) and produces GRAN-compatible graph data with:
  - node types (room category, 7 classes)
  - adjacency matrix with edge types (0=no-adj, 1=wall, 2=door)
  - optional centroids

Channel layout (verified against https://github.com/zzilch/RPLAN-Toolbox):
    Channel 0: boundary   (255 = front door marker)
    Channel 1: category   (room types 0-12, walls 14/16, doors 15/17)
    Channel 2: instance   (per-room unique IDs)
    Channel 3: inside     (interior mask)

Interior doors are EXPLICITLY marked with pixel value 17 in the category
channel — no heuristic gap detection needed.

Output formats:
  1. NetworkX pickle  (data/rplan_graphs.p)      — direct GRAN loading
  2. TU-dataset files (data/RPLAN/*)             — graph_load_batch compatible
  3. Per-sample .npy  (data/rplan_processed/*)   — preserves edge types

Usage:
    cd GRAN
    uv run python dataset/rplan_preprocessing/preprocess.py \\
        --raw_dir D:/Github/GSDiff/datasets/rplandata/Data/floorplan_dataset \\
        --out_dir data/ \\
        --max_samples 100
"""
import argparse
import os
import pickle
from collections import defaultdict
from glob import glob

import cv2
import numpy as np
import networkx as nx
from scipy import ndimage
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# RPLAN category ID -> GRAN class ID (7 classes)
# GRAN classes: 0=Living, 1=Bedroom, 2=Bathroom, 3=Kitchen, 4=Balcony,
#               5=Storage, 6=External
# RPLAN IDs verified against https://github.com/zzilch/RPLAN-Toolbox
RPLAN_TO_GRAN_CLASS = {
    0: 0,   # LivingRoom -> Living
    1: 1,   # MasterRoom -> Bedroom
    2: 3,   # Kitchen
    3: 2,   # Bathroom
    4: 0,   # DiningRoom -> Living
    5: 1,   # ChildRoom -> Bedroom
    6: 1,   # StudyRoom -> Bedroom
    7: 1,   # SecondRoom -> Bedroom
    8: 1,   # GuestRoom -> Bedroom
    9: 4,   # Balcony
    10: 0,  # Entrance -> Living
    11: 5,  # Storage
    12: 5,  # Wall-in -> Storage
}

ROOM_VALUES = tuple(range(13))   # 0-12 are rooms
WALL_VALUES = (14, 16)           # ExteriorWall, InteriorWall
INTERIOR_DOOR = 17
FRONT_DOOR = 15

EDGE_NONE = 0
EDGE_WALL = 1
EDGE_DOOR = 2

# Number of pixels to dilate each room mask when detecting adjacency.
# 2-3 is enough because walls in RPLAN are usually 1-2 pixels thick.
ADJACENCY_DILATE_PX = 3

# Minimum number of interior-door pixels between two rooms for the edge to
# be classified as a door. Interior doors in RPLAN are typically 4-8 pixels.
DOOR_PIXEL_MIN = 2


def load_rplan_png(path):
    """Return (boundary, category, instance, inside) as uint8 arrays."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None or img.ndim != 3 or img.shape[2] < 4:
        return None
    return img[:, :, 0], img[:, :, 1], img[:, :, 2], img[:, :, 3]


def extract_rooms(category, instance):
    """Identify individual rooms as connected components.

    Rooms are defined by pixels in category with values 0-12. We split them
    further by (category, instance) pair and by connected component to be
    robust against any instance ID re-use.

    Returns a list of dicts: {'mask', 'raw_category', 'instance', 'centroid', 'area'}
    """
    rooms = []
    room_mask = np.isin(category, ROOM_VALUES)
    if not room_mask.any():
        return rooms

    pairs = set()
    ys, xs = np.nonzero(room_mask)
    for y, x in zip(ys, xs):
        pairs.add((int(category[y, x]), int(instance[y, x])))

    for cat_val, inst_val in pairs:
        mask = (category == cat_val) & (instance == inst_val)
        if not mask.any():
            continue

        labeled, num_cc = ndimage.label(mask)
        for cc_idx in range(1, num_cc + 1):
            cc_mask = labeled == cc_idx
            area = int(cc_mask.sum())
            if area < 20:
                continue
            ys2, xs2 = np.nonzero(cc_mask)
            cy, cx = float(ys2.mean()), float(xs2.mean())
            rooms.append({
                'mask': cc_mask,
                'raw_category': cat_val,
                'instance': inst_val,
                'centroid': (cx, cy),
                'area': area,
            })
    return rooms


def compute_adjacency_with_types(rooms, category):
    """For each pair of rooms, decide no-adj / wall / door.

    Uses RPLAN's explicit interior door labels (category value 17):
      1. Dilate each room mask to find neighbors
      2. For each pair with overlapping dilations:
         - Count interior-door pixels (category == 17) in the intersection
         - If >= DOOR_PIXEL_MIN door pixels -> EDGE_DOOR
         - Otherwise -> EDGE_WALL
    """
    n = len(rooms)
    adj = np.zeros((n, n), dtype=np.int64)
    if n < 2:
        return adj

    dilated = [
        ndimage.binary_dilation(
            r['mask'], iterations=ADJACENCY_DILATE_PX,
        )
        for r in rooms
    ]

    interior_door_mask = category == INTERIOR_DOOR

    for i in range(n):
        for j in range(i + 1, n):
            interface = dilated[i] & dilated[j]
            if not interface.any():
                continue

            door_hits = int((interface & interior_door_mask).sum())
            if door_hits >= DOOR_PIXEL_MIN:
                edge_type = EDGE_DOOR
            else:
                edge_type = EDGE_WALL

            adj[i, j] = edge_type
            adj[j, i] = edge_type

    return adj


def png_to_graph_dict(png_path):
    loaded = load_rplan_png(png_path)
    if loaded is None:
        return None
    boundary, category, instance, inside = loaded

    rooms = extract_rooms(category, instance)
    if len(rooms) < 2:
        return None

    adj_typed = compute_adjacency_with_types(rooms, category)

    raw_categories = np.array(
        [r['raw_category'] for r in rooms], dtype=np.int64,
    )
    gran_classes = np.array(
        [RPLAN_TO_GRAN_CLASS.get(int(c), 0) for c in raw_categories],
        dtype=np.int64,
    )
    centroids = np.array([r['centroid'] for r in rooms], dtype=np.float32)
    areas = np.array([r['area'] for r in rooms], dtype=np.int64)

    return {
        'filename': os.path.basename(png_path),
        'num_rooms': len(rooms),
        'raw_categories': raw_categories,
        'gran_classes': gran_classes,
        'centroids': centroids,
        'areas': areas,
        'adjacency_typed': adj_typed,
    }


def dict_to_nx_graph(d):
    """Convert to networkx Graph for GRAN. Edge type saved as edge attribute."""
    G = nx.Graph()
    n = d['num_rooms']
    for i in range(n):
        G.add_node(
            i,
            attr=int(d['gran_classes'][i]),
            raw_category=int(d['raw_categories'][i]),
            centroid=d['centroids'][i].tolist(),
            area=int(d['areas'][i]),
        )

    adj = d['adjacency_typed']
    for i in range(n):
        for j in range(i + 1, n):
            et = int(adj[i, j])
            if et == EDGE_NONE:
                continue
            G.add_edge(i, j, edge_type=et)

    if G.number_of_edges() > 0 and not nx.is_connected(G):
        largest = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest).copy()
        G = nx.convert_node_labels_to_integers(G)

    return G


GRAN_CLASS_NAMES = ['Living', 'Bedroom', 'Bathroom', 'Kitchen',
                    'Balcony', 'Storage', 'External']
GRAN_CLASS_COLORS = {
    0: '#EE4D4D',  # Living - red
    1: '#C67FFF',  # Bedroom - purple
    2: '#5EBADA',  # Bathroom - cyan
    3: '#FFB84D',  # Kitchen - orange
    4: '#6BDF6B',  # Balcony - green
    5: '#B5896B',  # Storage - brown
    6: '#808080',  # External - gray
}


def _draw_bubble_diagram(ax, G, category_shape, title,
                         background=None, alpha_bg=0.0,
                         show_area=False, show_edge_labels=False):
    """Draw a bubble diagram: circles sized by area, colored by type, with edges.

    Args:
        show_area: if True, annotate each node with its pixel area below the label
        show_edge_labels: if True, write "wall"/"door" at the midpoint of each edge
    """
    h, w = category_shape
    if background is not None:
        ax.imshow(background, cmap='gray', alpha=alpha_bg)

    areas = np.array([G.nodes[n].get('area', 100) for n in G.nodes()])
    max_area = max(float(areas.max()), 1.0)
    min_r, max_r = 12.0, 28.0

    pos = {n: G.nodes[n]['centroid'] for n in G.nodes()}

    for u, v, data in G.edges(data=True):
        cx1, cy1 = pos[u]
        cx2, cy2 = pos[v]
        et = data.get('edge_type', 1)
        if et == 2:
            color, lw, ls, label = '#E53935', 3.0, '-', 'door'
        else:
            color, lw, ls, label = '#2E7D32', 2.2, '--', 'wall'
        ax.plot([cx1, cx2], [cy1, cy2],
                color=color, linewidth=lw, linestyle=ls,
                solid_capstyle='round', zorder=2)

        if show_edge_labels:
            mx, my = (cx1 + cx2) / 2.0, (cy1 + cy2) / 2.0
            ax.text(mx, my, label,
                    fontsize=6, ha='center', va='center',
                    color=color, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.15',
                              facecolor='white', edgecolor=color,
                              linewidth=0.8, alpha=0.9),
                    zorder=5)

    for node in G.nodes():
        cx, cy = pos[node]
        cls = G.nodes[node]['attr']
        color = GRAN_CLASS_COLORS.get(cls, '#888888')
        area = G.nodes[node].get('area', 100)
        radius = min_r + (max_r - min_r) * np.sqrt(area / max_area)
        circ = plt.Circle(
            (cx, cy), radius,
            facecolor=color, edgecolor='black',
            linewidth=1.5, zorder=3, alpha=0.92,
        )
        ax.add_patch(circ)
        label = GRAN_CLASS_NAMES[cls][:3]
        ax.text(cx, cy, label, ha='center', va='center',
                fontsize=8, fontweight='bold', color='black',
                zorder=4)
        if show_area:
            ax.text(cx, cy + radius + 4, f'{area}px',
                    ha='center', va='top', fontsize=6, color='#333',
                    zorder=4)

    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)


def _draw_doors_panel(ax, category):
    """Show only wall + door pixels with clear color coding."""
    structure = np.zeros((*category.shape, 3), dtype=np.uint8)
    structure[:] = 255   # white background

    wall_mask = np.isin(category, list(WALL_VALUES))
    int_door_mask = category == INTERIOR_DOOR
    front_door_mask = category == FRONT_DOOR

    structure[wall_mask] = [120, 120, 120]        # gray walls
    structure[int_door_mask] = [229, 57, 53]      # red interior doors
    structure[front_door_mask] = [30, 136, 229]   # blue front door

    ax.imshow(structure)

    n_int_doors = int(int_door_mask.sum())
    n_front = int(front_door_mask.sum())
    ax.set_title(
        f'Doors (red=interior:{n_int_doors}px, '
        f'blue=front:{n_front}px)'
    )
    ax.set_xticks([])
    ax.set_yticks([])

    legend_items = [
        Patch(facecolor=(120/255, 120/255, 120/255), label='Wall'),
        Patch(facecolor=(229/255, 57/255, 53/255), label='Interior door'),
        Patch(facecolor=(30/255, 136/255, 229/255), label='Front door'),
    ]
    ax.legend(handles=legend_items, loc='lower right', fontsize=6,
              framealpha=0.9)


def visualize_sample(png_path, d, G, out_path):
    """6-panel visualization:
        1. Raw PNG category
        2. Doors panel (walls + interior/front doors)
        3. Bubble diagram on PNG
        4. Bubble diagram with area + edge labels
        5. Abstract graph with edge labels
        6. (legend included in bottom)
    """
    loaded = load_rplan_png(png_path)
    if loaded is None:
        return
    _, category, _, _ = loaded

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    axes[0, 0].imshow(category, cmap='tab20', vmin=0, vmax=17)
    axes[0, 0].set_title('Raw PNG (category channel)')
    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])

    _draw_doors_panel(axes[0, 1], category)

    _draw_bubble_diagram(
        axes[0, 2], G, category.shape,
        title='Bubble diagram on PNG',
        background=category, alpha_bg=0.45,
    )

    _draw_bubble_diagram(
        axes[1, 0], G, category.shape,
        title='Bubble diagram (with area)',
        background=None,
        show_area=True,
    )

    _draw_bubble_diagram(
        axes[1, 1], G, category.shape,
        title='Bubble diagram (with edge labels)',
        background=None,
        show_edge_labels=True,
    )

    pos = {n: G.nodes[n]['centroid'] for n in G.nodes()}
    node_colors = [GRAN_CLASS_COLORS.get(G.nodes[n]['attr'], '#888888')
                   for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, ax=axes[1, 2], node_color=node_colors,
                           node_size=600, edgecolors='black', linewidths=1.5)
    for u, v in G.edges():
        et = G[u][v].get('edge_type', 1)
        color = '#E53935' if et == 2 else '#2E7D32'
        style = '-' if et == 2 else '--'
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], ax=axes[1, 2],
                               edge_color=color, style=style, width=2.5)
    node_labels = {
        n: f"{GRAN_CLASS_NAMES[G.nodes[n]['attr']][:3]}\n{G.nodes[n].get('area', 0)}px"
        for n in G.nodes()
    }
    nx.draw_networkx_labels(G, pos, node_labels, ax=axes[1, 2], font_size=6)
    edge_labels = {
        (u, v): 'door' if G[u][v].get('edge_type', 1) == 2 else 'wall'
        for u, v in G.edges()
    }
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels, ax=axes[1, 2],
        font_size=5, font_color='black',
        bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                  edgecolor='none', alpha=0.85),
    )
    axes[1, 2].set_title('Abstract graph (full labels)')
    axes[1, 2].set_xlim(0, category.shape[1])
    axes[1, 2].set_ylim(category.shape[0], 0)
    axes[1, 2].set_aspect('equal')
    axes[1, 2].set_xticks([])
    axes[1, 2].set_yticks([])

    room_legend = [
        Patch(facecolor=GRAN_CLASS_COLORS[i], label=GRAN_CLASS_NAMES[i])
        for i in range(6)
    ]
    edge_legend = [
        plt.Line2D([0], [0], color='#2E7D32', linestyle='--', linewidth=2,
                   label='wall (adjacent, no door)'),
        plt.Line2D([0], [0], color='#E53935', linestyle='-', linewidth=2.5,
                   label='door (interior door present)'),
    ]
    fig.legend(handles=room_legend + edge_legend,
               loc='lower center', ncol=8, fontsize=8, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.01))

    plt.suptitle(
        f'{os.path.basename(png_path)}  '
        f'({G.number_of_nodes()} rooms, '
        f'{sum(1 for _,_,d_ in G.edges(data=True) if d_.get("edge_type")==2)} doors, '
        f'{sum(1 for _,_,d_ in G.edges(data=True) if d_.get("edge_type")==1)} walls)',
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close()


def save_tu_format(graphs, out_dir, name='RPLAN'):
    os.makedirs(out_dir, exist_ok=True)

    edge_list = []
    node_labels = []
    node_attrs = []
    graph_indicators = []
    graph_labels = []

    node_offset = 0
    for g_idx, G in enumerate(graphs):
        n = G.number_of_nodes()
        graph_labels.append(n)
        for node in range(n):
            node_labels.append(G.nodes[node].get('attr', 0))
            graph_indicators.append(g_idx + 1)
            centroid = G.nodes[node].get('centroid', [0.0, 0.0])
            node_attrs.append(centroid)
        for u, v in G.edges():
            edge_list.append((u + node_offset + 1, v + node_offset + 1))
            edge_list.append((v + node_offset + 1, u + node_offset + 1))
        node_offset += n

    with open(os.path.join(out_dir, f'{name}_A.txt'), 'w') as f:
        for src, dst in edge_list:
            f.write(f'{src}, {dst}\n')
    with open(os.path.join(out_dir, f'{name}_node_labels.txt'), 'w') as f:
        for label in node_labels:
            f.write(f'{label}\n')
    with open(os.path.join(out_dir, f'{name}_graph_indicator.txt'), 'w') as f:
        for ind in graph_indicators:
            f.write(f'{ind}\n')
    with open(os.path.join(out_dir, f'{name}_graph_labels.txt'), 'w') as f:
        for label in graph_labels:
            f.write(f'{label}\n')
    with open(os.path.join(out_dir, f'{name}_node_attributes.txt'), 'w') as f:
        for attr in node_attrs:
            f.write(', '.join(f'{x:.6f}' for x in attr) + '\n')

    print(f'TU-format written: {out_dir}')
    print(f'  nodes={len(node_labels)} edges(undirected)={len(edge_list)//2} '
          f'graphs={len(graph_labels)}')


def print_statistics(graphs, dicts):
    num_nodes = [G.number_of_nodes() for G in graphs]
    num_edges = [G.number_of_edges() for G in graphs]
    print(f'\nTotal graphs: {len(graphs)}')
    print(f'Nodes: min={min(num_nodes)} max={max(num_nodes)} '
          f'mean={np.mean(num_nodes):.1f}')
    print(f'Edges: min={min(num_edges)} max={max(num_edges)} '
          f'mean={np.mean(num_edges):.1f}')

    attr_counts = defaultdict(int)
    for G in graphs:
        for n in G.nodes():
            attr_counts[G.nodes[n]['attr']] += 1
    total_attr = sum(attr_counts.values())
    print('\nRoom type distribution (after mapping to 7 GRAN classes):')
    names = ['Living', 'Bedroom', 'Bathroom', 'Kitchen',
             'Balcony', 'Storage', 'External']
    for k in sorted(attr_counts.keys()):
        pct = 100 * attr_counts[k] / total_attr
        label = names[k] if k < len(names) else f'type{k}'
        print(f'  {label} ({k}): {attr_counts[k]} ({pct:.1f}%)')

    raw_counts = defaultdict(int)
    for d in dicts:
        for c in d['raw_categories']:
            raw_counts[int(c)] += 1
    print('\nRaw RPLAN category distribution:')
    raw_names = {
        0: 'LivingRoom', 1: 'MasterRoom', 2: 'Kitchen', 3: 'Bathroom',
        4: 'DiningRoom', 5: 'ChildRoom', 6: 'StudyRoom', 7: 'SecondRoom',
        8: 'GuestRoom', 9: 'Balcony', 10: 'Entrance', 11: 'Storage',
        12: 'Wall-in',
    }
    total_raw = sum(raw_counts.values())
    for k in sorted(raw_counts.keys()):
        name = raw_names.get(k, f'id{k}')
        pct = 100 * raw_counts[k] / total_raw
        print(f'  {k:2d} {name:12s}: {raw_counts[k]} ({pct:.1f}%)')

    edge_type_counts = defaultdict(int)
    for d in dicts:
        adj = d['adjacency_typed']
        n = adj.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                edge_type_counts[int(adj[i, j])] += 1
    total_pairs = sum(edge_type_counts.values())
    print('\nEdge type distribution (all pairs):')
    print(f'  none (0): {edge_type_counts[0]} '
          f'({100*edge_type_counts[0]/total_pairs:.1f}%)')
    print(f'  wall (1): {edge_type_counts[1]} '
          f'({100*edge_type_counts[1]/total_pairs:.1f}%)')
    print(f'  door (2): {edge_type_counts[2]} '
          f'({100*edge_type_counts[2]/total_pairs:.1f}%)')
    nonzero = edge_type_counts[1] + edge_type_counts[2]
    if nonzero > 0:
        print(f'  (of adjacent pairs: '
              f'wall={100*edge_type_counts[1]/nonzero:.1f}%, '
              f'door={100*edge_type_counts[2]/nonzero:.1f}%)')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_dir', type=str, default='data/rplan/raw',
                        help='Directory containing raw RPLAN *.png files')
    parser.add_argument('--out_dir', type=str, default='data/rplan',
                        help='Base output directory for rplan data')
    parser.add_argument('--max_samples', type=int, default=-1,
                        help='Limit number of samples (-1 = all)')
    parser.add_argument('--min_nodes', type=int, default=3)
    parser.add_argument('--max_nodes', type=int, default=20)
    parser.add_argument('--save_per_sample_npy', action='store_true')
    parser.add_argument('--visualize', type=int, default=0,
                        help='Save visualizations for the first N processed samples')
    args = parser.parse_args()

    per_sample_dir = os.path.join(args.out_dir, 'processed')
    vis_dir = os.path.join(args.out_dir, 'vis')
    tu_dir = os.path.join(args.out_dir, 'tu')
    pickle_path = os.path.join(args.out_dir, 'graphs.p')

    png_files = sorted(glob(os.path.join(args.raw_dir, '*.png')))
    if not png_files:
        print(f'No PNGs found in {args.raw_dir}')
        return
    if args.max_samples > 0:
        png_files = png_files[:args.max_samples]
    print(f'Found {len(png_files)} PNG files')

    os.makedirs(args.out_dir, exist_ok=True)
    if args.save_per_sample_npy:
        os.makedirs(per_sample_dir, exist_ok=True)
    if args.visualize > 0:
        os.makedirs(vis_dir, exist_ok=True)

    all_graphs = []
    all_dicts = []
    all_source_paths = []
    skipped = 0

    for path in tqdm(png_files, desc='Processing'):
        d = png_to_graph_dict(path)
        if d is None:
            skipped += 1
            continue
        if d['num_rooms'] < args.min_nodes or d['num_rooms'] > args.max_nodes:
            skipped += 1
            continue

        G = dict_to_nx_graph(d)
        if G.number_of_nodes() < args.min_nodes:
            skipped += 1
            continue

        all_graphs.append(G)
        all_dicts.append(d)
        all_source_paths.append(path)

        if args.save_per_sample_npy:
            stem = os.path.splitext(os.path.basename(path))[0]
            np.save(
                os.path.join(per_sample_dir, f'{stem}.npy'),
                d, allow_pickle=True,
            )

        if args.visualize > 0 and len(all_graphs) <= args.visualize:
            stem = os.path.splitext(os.path.basename(path))[0]
            vis_path = os.path.join(
                vis_dir, f'sample_{len(all_graphs)-1:03d}_{stem}.png'
            )
            try:
                visualize_sample(path, d, G, vis_path)
            except Exception as e:
                print(f'  visualization error for {stem}: {e}')

    print(f'\nProcessed: {len(all_graphs)}, skipped: {skipped}')
    if not all_graphs:
        return

    print_statistics(all_graphs, all_dicts)

    save_tu_format(all_graphs, tu_dir, name='RPLAN')

    with open(pickle_path, 'wb') as f:
        pickle.dump(all_graphs, f)
    print(f'\nPickle saved: {pickle_path}')


if __name__ == '__main__':
    main()
