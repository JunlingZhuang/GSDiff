"""High-level generation pipelines for unconstrained, topology and boundary modes."""

import base64
import io
import logging
import random

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from app.config import RESOLUTION, MAX_NODES
from app.services.diffusion import reverse_diffusion
from app.services.model_manager import manager
from app.services.rendering import render_floorplan

logger = logging.getLogger(__name__)

from gsdiff.utils import (
    inverse_normalize_and_remove_padding_100_4testing,
    get_near_corners,
    merge_array_elements,
    edges_remove_padding,
    edges_to_coordinates,
    get_cycle_basis_and_semantic_3_semansimplified,
)


def _merge_nearby_corners(corners, semantics):
    merge_components = get_near_corners(corners, merge_threshold=RESOLUTION * 0.01)
    corners_flat = corners.reshape(-1, 2)
    semantics_flat = semantics.reshape(-1, 7)
    full_idx = []
    rand_idx = []
    for idx_set in merge_components:
        full_idx.extend(list(idx_set))
        rand_idx.append(random.choice(list(idx_set)))
    merged_corners = merge_array_elements(corners_flat, full_idx, rand_idx)
    merged_semantics = merge_array_elements(semantics_flat, full_idx, rand_idx)
    return merged_corners[None, :, :], merged_semantics[None, :, :]


def _run_edge_model(edge_model, corners_sample, semantics_sample, device, *extra_args):
    """Run Stage 2 edge prediction. Pass extra_args for conditioned models."""
    n_nodes = corners_sample.shape[1]
    corners_padded = torch.zeros((1, MAX_NODES, 2), dtype=torch.float64, device=device)
    corners_norm = (torch.tensor(corners_sample, dtype=torch.float64, device=device) - RESOLUTION // 2) / (RESOLUTION // 2)
    corners_padded[:, :n_nodes, :] = corners_norm

    semantics_padded = torch.zeros((1, MAX_NODES, 7), dtype=torch.float64, device=device)
    semantics_padded[:, :n_nodes, :] = torch.tensor(semantics_sample, dtype=torch.float64, device=device)

    attn = torch.zeros((1, MAX_NODES, MAX_NODES), dtype=torch.bool, device=device)
    attn[:, :n_nodes, :n_nodes] = True
    padding_mask = torch.zeros((1, MAX_NODES, 1), dtype=torch.uint8, device=device)
    padding_mask[:, :n_nodes, :] = 1

    logger.info(f"Edge model: {n_nodes} nodes, {len(extra_args)} extra args")
    output_edges, _, _ = edge_model(corners_padded, attn, padding_mask, semantics_padded, *extra_args)
    output_edges = F.softmax(output_edges, dim=2)
    output_edges = torch.argmax(output_edges, dim=2)
    output_edges = F.one_hot(output_edges, num_classes=2)

    return output_edges, n_nodes


def _align_polygons(simple_cycles):
    cleaned = [[v[:2] for v in poly] for poly in simple_cycles]
    align_threshold = round(RESOLUTION * 0.01)

    for x_left in range(0, RESOLUTION - align_threshold):
        x_right = x_left + align_threshold
        edges_inbond = []
        for cp in cleaned:
            for pi, p in enumerate(cp):
                if pi < len(cp) - 1:
                    e = (p, cp[pi + 1])
                    if x_left <= e[0][0] <= x_right and x_left <= e[1][0] <= x_right:
                        edges_inbond.append(e)
        if not edges_inbond:
            continue
        G = nx.Graph()
        for eb in edges_inbond:
            G.add_node(eb)
        for e1 in edges_inbond:
            for e2 in edges_inbond:
                if e1 != e2 and set(e1) & set(e2):
                    G.add_edge(e1, e2)
        for comp in nx.connected_components(G):
            comp_edges = list(comp)
            verts = [p for e in comp_edges for p in e]
            x_bar = round(sum(e[0][0] + e[1][0] for e in comp_edges) / (2 * len(comp_edges)))
            cleaned = [[(x_bar, v[1]) if v in verts else v for v in poly] for poly in cleaned]

    for y_top in range(0, RESOLUTION - align_threshold):
        y_bot = y_top + align_threshold
        edges_inbond = []
        for cp in cleaned:
            for pi, p in enumerate(cp):
                if pi < len(cp) - 1:
                    e = (p, cp[pi + 1])
                    if y_top <= e[0][1] <= y_bot and y_top <= e[1][1] <= y_bot:
                        edges_inbond.append(e)
        if not edges_inbond:
            continue
        G = nx.Graph()
        for eb in edges_inbond:
            G.add_node(eb)
        for e1 in edges_inbond:
            for e2 in edges_inbond:
                if e1 != e2 and set(e1) & set(e2):
                    G.add_edge(e1, e2)
        for comp in nx.connected_components(G):
            comp_edges = list(comp)
            verts = [p for e in comp_edges for p in e]
            y_bar = round(sum(e[0][1] + e[1][1] for e in comp_edges) / (2 * len(comp_edges)))
            cleaned = [[(v[0], y_bar) if v in verts else v for v in poly] for poly in cleaned]

    return cleaned


def _graph_to_image(corners, semantics, edges_tensor, n_nodes):
    edges_list = edges_remove_padding([edges_tensor], [n_nodes])
    edges_np = edges_list[0]

    sem_transform = semantics.copy()
    sem_indices = np.indices(sem_transform.shape)[-1]
    sem_transform = np.where(sem_transform == 1, sem_indices, 99999)

    points = [tuple(p) for p in np.concatenate((corners, sem_transform), axis=-1).tolist()[0]]
    n = len(points)
    edge_matrix = edges_np[0, :, 1].reshape(n, n)
    edge_flat = np.triu(edge_matrix).reshape(-1)
    edge_coords = edges_to_coordinates(edge_flat, points)

    _, cycles, cycle_sems = get_cycle_basis_and_semantic_3_semansimplified(points, edge_coords)

    if not cycles:
        return Image.new("RGB", (RESOLUTION, RESOLUTION), (255, 255, 255)), 0

    aligned = _align_polygons(cycles)
    img = render_floorplan(aligned, cycle_sems)
    return img, len(cycle_sems)


@torch.no_grad()
def generate_unconstrained() -> tuple[Image.Image, int]:
    logger.info("Starting unconstrained generation...")
    device = manager.device
    global_attn = torch.ones(1, MAX_NODES, MAX_NODES, dtype=torch.bool, device=device)
    x_0 = reverse_diffusion(manager.unconst_node, global_attn, device)
    logger.info("Stage 1 complete")

    corners_list = [x_0[0, :, :2][None, :, :]]
    semantics_list = [x_0[0, :, 2:9][None, :, :]]
    padding_list = [x_0[0, :, 9:10].view(-1)]

    corners_inv, semantics_inv = inverse_normalize_and_remove_padding_100_4testing(
        corners_list, semantics_list, padding_list, resolution=RESOLUTION)

    corners_merged, semantics_merged = _merge_nearby_corners(corners_inv[0], semantics_inv[0])

    edges_out, n_nodes = _run_edge_model(manager.unconst_edge, corners_merged, semantics_merged, device)
    return _graph_to_image(corners_merged, semantics_merged, edges_out, n_nodes)


@torch.no_grad()
def generate_topology(room_types: list[int], adjacency: list[list[int]]) -> Image.Image:
    device = manager.device
    n_rooms = len(room_types)

    semantics_onehot = np.zeros((1, 8, 7), dtype=np.float32)
    for i, rt in enumerate(room_types):
        semantics_onehot[0, i, rt] = 1.0
    semantics_tensor = torch.tensor(semantics_onehot, device=device)

    adj = np.zeros((1, 8, 8), dtype=np.float32)
    for i in range(n_rooms):
        for j in range(n_rooms):
            adj[0, i, j] = adjacency[i][j]
    adj_tensor = torch.tensor(adj, dtype=torch.bool, device=device)

    padding_mask = torch.zeros(1, 8, 1, device=device)
    padding_mask[0, :n_rooms, 0] = 1.0

    encoder = manager.topo_encoder
    bb_embedding = encoder.semantics_embedding(semantics_tensor.float())
    for layer in encoder.transformer_layers:
        bb_embedding = layer(bb_embedding, adj_tensor)
    bb_embedding = bb_embedding * padding_mask

    global_attn = torch.ones(1, MAX_NODES, MAX_NODES, dtype=torch.bool, device=device)
    x_0 = reverse_diffusion(manager.topo_node, global_attn, device,
                            condition_embedding=bb_embedding, condition_mask=padding_mask)

    corners_list = [x_0[0, :, :2][None, :, :]]
    semantics_list = [x_0[0, :, 2:9][None, :, :]]
    padding_list = [x_0[0, :, 9:10].view(-1)]

    corners_inv, semantics_inv = inverse_normalize_and_remove_padding_100_4testing(
        corners_list, semantics_list, padding_list, resolution=RESOLUTION)

    corners_merged, semantics_merged = _merge_nearby_corners(corners_inv[0], semantics_inv[0])
    logger.info(f"Topology: merged to {corners_merged.shape[1]} corners")
    edges_out, n_nodes = _run_edge_model(
        manager.topo_edge, corners_merged, semantics_merged, device,
        bb_embedding, padding_mask)

    img, _ = _graph_to_image(corners_merged, semantics_merged, edges_out, n_nodes)
    return img


def _encode_boundary_image(boundary_b64: str, encoder, device):
    """Decode a base64 PNG boundary image and run it through the CNN encoder.

    The encoder expects (batch, 3, 256, 256) float input.
    Returns the feat_16 feature map of shape (1, 1024, 16, 16).
    """
    # Strip data-URI prefix if present
    if "," in boundary_b64:
        boundary_b64 = boundary_b64.split(",", 1)[1]
    raw = base64.b64decode(boundary_b64)
    pil_img = Image.open(io.BytesIO(raw)).convert("RGB").resize((256, 256))
    # Normalize to [-1, 1] matching training: (pixel - 128) / 128
    img_np = (np.array(pil_img, dtype=np.float32) - 128.0) / 128.0
    # (H, W, 3) -> (1, 3, H, W)
    img_tensor = torch.tensor(img_np, dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0)

    # Run through encoder layers to extract feat_16 (matching prerunningCNN.py)
    e1 = encoder.Conv1(img_tensor)
    e2 = encoder.Maxpool1(e1)
    e2 = encoder.Conv2(e2) + encoder.shortcut2(e2)
    e3 = encoder.Maxpool2(e2)
    e3 = encoder.Conv3(e3) + encoder.shortcut3(e3)
    e4 = encoder.Maxpool3(e3)
    e4 = encoder.Conv4(e4) + encoder.shortcut4(e4)
    e5 = encoder.Maxpool4(e4)
    e5 = encoder.Conv5(e5) + encoder.shortcut5(e5)
    # e5 shape: (1, 1024, 16, 16)
    return e5


def _run_boundary_edge_model(edge_model, corners_sample, semantics_sample, feat_16, device):
    """Run the boundary-constrained edge model (BoundEdgeModel).

    Same as _run_edge_model but passes feat_16 as an additional argument.
    """
    n_nodes = corners_sample.shape[1]
    corners_padded = torch.zeros((1, MAX_NODES, 2), dtype=torch.float64, device=device)
    corners_norm = (torch.tensor(corners_sample, dtype=torch.float64, device=device) - RESOLUTION // 2) / (RESOLUTION // 2)
    corners_padded[:, :n_nodes, :] = corners_norm

    semantics_padded = torch.zeros((1, MAX_NODES, 7), dtype=torch.float64, device=device)
    semantics_padded[:, :n_nodes, :] = torch.tensor(semantics_sample, dtype=torch.float64, device=device)

    attn = torch.zeros((1, MAX_NODES, MAX_NODES), dtype=torch.bool, device=device)
    attn[:, :n_nodes, :n_nodes] = True
    padding_mask = torch.zeros((1, MAX_NODES, 1), dtype=torch.uint8, device=device)
    padding_mask[:, :n_nodes, :] = 1

    output_edges, _, _ = edge_model(corners_padded, attn, padding_mask, semantics_padded, feat_16)
    output_edges = F.softmax(output_edges, dim=2)
    output_edges = torch.argmax(output_edges, dim=2)
    output_edges = F.one_hot(output_edges, num_classes=2)

    return output_edges, n_nodes


@torch.no_grad()
def generate_boundary(boundary_b64: str) -> Image.Image:
    """Generate a floorplan conditioned on a boundary outline image."""
    device = manager.device

    # Step 1: Encode the boundary image to get feat_16
    feat_16 = _encode_boundary_image(boundary_b64, manager.boun_encoder, device)

    # Step 2: Reverse diffusion with boundary-conditioned node model
    global_attn = torch.ones(1, MAX_NODES, MAX_NODES, dtype=torch.bool, device=device)
    x_0 = reverse_diffusion(manager.boun_node, global_attn, device, boundary_feat=feat_16)

    # Step 3: Extract corners and semantics
    corners_list = [x_0[0, :, :2][None, :, :]]
    semantics_list = [x_0[0, :, 2:9][None, :, :]]
    padding_list = [x_0[0, :, 9:10].view(-1)]

    corners_inv, semantics_inv = inverse_normalize_and_remove_padding_100_4testing(
        corners_list, semantics_list, padding_list, resolution=RESOLUTION)

    # Step 4: Merge nearby corners
    corners_merged, semantics_merged = _merge_nearby_corners(corners_inv[0], semantics_inv[0])

    # Step 5: Edge prediction with boundary-conditioned edge model
    edges_out, n_nodes = _run_boundary_edge_model(
        manager.boun_edge, corners_merged, semantics_merged, feat_16, device)

    # Step 6: Graph to image
    img, _ = _graph_to_image(corners_merged, semantics_merged, edges_out, n_nodes)
    return img
