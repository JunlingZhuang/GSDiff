"""Reusable graph-completion evaluation helpers.

Keep this separate from `test_graph_completion.py` so the test script stays a
thin command-line driver. The next-node and autoregressive evaluators also
reuse these metrics and visualization utilities.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch

from test_graph_generation import (
    graph_to_record,
    label_at,
    resolve_checkpoint,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def cfg_get(section, key, default):
    if section is None:
        return default
    return section.get(key, default)


def resolve_completion_checkpoint(cfg, completion_cfg):
    checkpoint = cfg_get(completion_cfg, "checkpoint", None) or cfg.test.checkpoint
    if checkpoint:
        path = Path(checkpoint)
        return path if path.is_absolute() else PROJECT_ROOT / path
    return resolve_checkpoint(cfg)


def data_to_dense_indices(data):
    n = int(data.x.size(0))
    x_idx = torch.argmax(data.x, dim=-1).long()
    e_idx = torch.zeros((n, n), dtype=torch.long)
    if data.edge_index.numel() > 0:
        edge_types = torch.argmax(data.edge_attr, dim=-1).long()
        for edge_id in range(data.edge_index.size(1)):
            src = int(data.edge_index[0, edge_id])
            dst = int(data.edge_index[1, edge_id])
            e_idx[src, dst] = int(edge_types[edge_id])
    return x_idx, e_idx


def choose_known_nodes(n, known_ratio, rng):
    if n <= 1:
        return np.arange(n, dtype=np.int64)
    k = int(round(float(known_ratio) * n))
    k = max(1, min(n - 1, k))
    return np.sort(rng.choice(n, size=k, replace=False))


def make_completion_tensors(x_idx, e_idx, known_nodes):
    n = int(x_idx.numel())
    known = torch.zeros(n, dtype=torch.bool)
    known[torch.as_tensor(known_nodes, dtype=torch.long)] = True

    anchor_e = known.unsqueeze(0) & known.unsqueeze(1)
    return {
        "X_idx": x_idx.unsqueeze(0),
        "E_idx": e_idx.unsqueeze(0),
        "node_mask": torch.ones((1, n), dtype=torch.bool),
        "anchor_X": known.unsqueeze(0),
        "anchor_E": anchor_e.unsqueeze(0),
    }


def repeat_completion_tensors(tensors, batch_size):
    return {
        key: value.repeat((batch_size,) + (1,) * (value.dim() - 1))
        for key, value in tensors.items()
    }


def completion_graph_condition(model, x_idx, known_nodes, e_idx=None, condition_mode="oracle"):
    if getattr(model.dataset_info, "graph_condition_dim", 0) == 0:
        return None
    n = int(x_idx.numel())
    known = torch.zeros((1, n), dtype=torch.bool)
    known[0, torch.as_tensor(known_nodes, dtype=torch.long)] = True
    node_mask = torch.ones((1, n), dtype=torch.bool)
    is_app_condition = str(condition_mode).lower() == "app"
    return model._build_graph_condition(
        node_mask=node_mask.to(model.device),
        known_nodes=known.to(model.device),
        target_nodes=None,
        X_idx=x_idx.unsqueeze(0).to(model.device) if not is_app_condition else None,
        E_idx=e_idx.unsqueeze(0).to(model.device) if e_idx is not None else None,
        room_inventory_available=not is_app_condition,
        target_density_available=not is_app_condition,
        edge_hist_available=True,
    )


def repeat_graph_condition(graph_condition, batch_size):
    if graph_condition is None:
        return None
    return graph_condition.repeat((batch_size, 1))


def graph_tensor_stats(sample):
    _, e_idx = sample
    n = int(e_idx.size(0))
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if int(e_idx[i, j]) > 0:
                graph.add_edge(i, j)
    edge_count = graph.number_of_edges()
    components = list(nx.connected_components(graph))
    isolated = sum(1 for _, degree in graph.degree() if degree == 0)
    return {
        "edge_count": edge_count,
        "connected": nx.is_connected(graph) if n > 0 else True,
        "num_components": len(components),
        "isolated": isolated,
    }


def score_completion_candidate(
    sample,
    *,
    expected_degree,
    connected_weight=4.0,
    edge_count_weight=1.0,
    isolated_weight=0.25,
):
    """Graph-only rerank score; lower is better and does not inspect GT."""
    stats = graph_tensor_stats(sample)
    n = int(sample[0].numel())
    expected_edges = float(expected_degree) * n / 2.0
    edge_scale = max(1.0, expected_edges)
    edge_term = abs(stats["edge_count"] - expected_edges) / edge_scale
    component_term = max(0, stats["num_components"] - 1)
    isolated_term = stats["isolated"] / max(1, n)
    return (
        edge_count_weight * edge_term
        + connected_weight * component_term
        + isolated_weight * isolated_term
    )


def select_reranked_completion(samples, rerank_cfg):
    expected_degree = float(rerank_cfg.get("expected_degree", 4.2))
    best_sample = None
    best_score = None
    candidate_records = []
    for index, sample in enumerate(samples):
        score = score_completion_candidate(
            sample,
            expected_degree=expected_degree,
            connected_weight=float(rerank_cfg.get("connected_weight", 4.0)),
            edge_count_weight=float(rerank_cfg.get("edge_count_weight", 1.0)),
            isolated_weight=float(rerank_cfg.get("isolated_weight", 0.25)),
        )
        stats = graph_tensor_stats(sample)
        candidate_records.append({"index": index, "score": score, **stats})
        if best_score is None or score < best_score:
            best_score = score
            best_sample = sample
    return best_sample, {"best_score": best_score, "candidates": candidate_records}


def repair_connectivity_sample(sample, known_nodes, edge_type=1):
    """Add deterministic bridge edges between disconnected components."""
    x_idx, e_idx = sample
    repaired_e = e_idx.clone().long()
    n = int(x_idx.numel())
    if n <= 1:
        return [x_idx.clone().long(), repaired_e]

    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if int(repaired_e[i, j]) > 0:
                graph.add_edge(i, j)
    components = [sorted(component) for component in nx.connected_components(graph)]
    if len(components) <= 1:
        return [x_idx.clone().long(), repaired_e]

    known_set = set(int(node) for node in known_nodes)
    main_component = max(components, key=lambda comp: (len(set(comp) & known_set), len(comp)))
    for component in components:
        if component == main_component:
            continue
        source = min(component)
        target = sorted(main_component)[0]
        repaired_e[source, target] = int(edge_type)
        repaired_e[target, source] = int(edge_type)
        main_component = sorted(set(main_component) | set(component))
    return [x_idx.clone().long(), repaired_e]


def partial_graph_from_dense(x_idx, e_idx, known_nodes, spec):
    known_set = {int(node) for node in known_nodes}
    graph = nx.Graph()
    for node in sorted(known_set):
        attr = int(x_idx[node])
        graph.add_node(node, attr=attr, room_type=label_at(spec.node_labels, attr))

    for i in sorted(known_set):
        for j in sorted(known_set):
            if i >= j:
                continue
            edge_type = int(e_idx[i, j])
            if edge_type > 0:
                graph.add_edge(
                    i,
                    j,
                    edge_type=edge_type,
                    edge_label=label_at(spec.edge_labels, edge_type),
                )
    return graph


def tensor_accuracy(pred, target, mask):
    total = int(mask.sum().item())
    if total == 0:
        return None, 0, 0
    correct = int((pred[mask] == target[mask]).sum().item())
    return correct / total, correct, total


def safe_divide(numerator, denominator):
    return numerator / denominator if denominator else None


def precision_recall_f1(tp, fp, fn):
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and (precision + recall) > 0
        else None
    )
    return precision, recall, f1


def class_counts(pred, target, mask, classes):
    counts = {}
    pred_values = pred[mask].long()
    target_values = target[mask].long()
    for cls in classes:
        cls = int(cls)
        tp = int(((pred_values == cls) & (target_values == cls)).sum().item())
        fp = int(((pred_values == cls) & (target_values != cls)).sum().item())
        fn = int(((pred_values != cls) & (target_values == cls)).sum().item())
        support = int((target_values == cls).sum().item())
        counts[str(cls)] = {"tp": tp, "fp": fp, "fn": fn, "support": support}
    return counts


def f1_from_counts(counts):
    scores = []
    weighted_sum = 0.0
    weighted_total = 0
    for item in counts.values():
        precision, recall, f1 = precision_recall_f1(item["tp"], item["fp"], item["fn"])
        item["precision"] = precision
        item["recall"] = recall
        item["f1"] = f1
        if f1 is not None and item["support"] > 0:
            scores.append(f1)
            weighted_sum += f1 * item["support"]
            weighted_total += item["support"]
    return {
        "macro_f1": float(np.mean(scores)) if scores else None,
        "weighted_f1": weighted_sum / weighted_total if weighted_total else None,
        "per_class": counts,
    }


def evaluate_completion(completed_sample, x_true, e_true, known_nodes):
    x_pred, e_pred = completed_sample
    n = int(x_true.numel())
    known = torch.zeros(n, dtype=torch.bool)
    known[torch.as_tensor(known_nodes, dtype=torch.long)] = True

    unknown_nodes = ~known
    node_acc, node_correct, node_total = tensor_accuracy(x_pred.long(), x_true.long(), unknown_nodes)

    upper = torch.triu(torch.ones((n, n), dtype=torch.bool), diagonal=1)
    anchor_edges = known.unsqueeze(0) & known.unsqueeze(1)
    masked_edges = upper & (~anchor_edges)
    edge_acc, edge_correct, edge_total = tensor_accuracy(e_pred.long(), e_true.long(), masked_edges)

    present_masked_edges = masked_edges & (e_true.long() > 0)
    edge_present_acc, edge_present_correct, edge_present_total = tensor_accuracy(
        e_pred.long(),
        e_true.long(),
        present_masked_edges,
    )

    pred_edges = e_pred.long()[masked_edges]
    true_edges = e_true.long()[masked_edges]
    pred_present = pred_edges > 0
    true_present = true_edges > 0
    presence_tp = int((pred_present & true_present).sum().item())
    presence_fp = int((pred_present & (~true_present)).sum().item())
    presence_fn = int(((~pred_present) & true_present).sum().item())
    presence_tn = int(((~pred_present) & (~true_present)).sum().item())
    presence_precision, presence_recall, presence_f1 = precision_recall_f1(
        presence_tp,
        presence_fp,
        presence_fn,
    )

    typed_tp = int(((pred_edges == true_edges) & true_present).sum().item())
    typed_fp = int((pred_present & (pred_edges != true_edges)).sum().item())
    typed_fn = int((true_present & (pred_edges != true_edges)).sum().item())
    typed_precision, typed_recall, typed_f1 = precision_recall_f1(typed_tp, typed_fp, typed_fn)

    pred_degree = (e_pred.long() > 0).sum(dim=1).float()
    true_degree = (e_true.long() > 0).sum(dim=1).float()
    degree_mae = float(torch.mean(torch.abs(pred_degree - true_degree)).item()) if n else None
    pred_edge_count = int(torch.triu(e_pred.long() > 0, diagonal=1).sum().item())
    true_edge_count = int(torch.triu(e_true.long() > 0, diagonal=1).sum().item())
    possible_edges = n * (n - 1) // 2

    node_classes = torch.unique(x_true.long()[unknown_nodes]).detach().cpu().tolist()
    edge_classes = [int(value) for value in torch.unique(true_edges[true_present]).detach().cpu().tolist()]

    return {
        "node_unknown_accuracy": node_acc,
        "node_unknown_correct": node_correct,
        "node_unknown_total": node_total,
        "edge_masked_accuracy_all": edge_acc,
        "edge_masked_correct": edge_correct,
        "edge_masked_total": edge_total,
        "edge_masked_accuracy_present": edge_present_acc,
        "edge_masked_present_correct": edge_present_correct,
        "edge_masked_present_total": edge_present_total,
        "edge_presence_tp": presence_tp,
        "edge_presence_fp": presence_fp,
        "edge_presence_fn": presence_fn,
        "edge_presence_tn": presence_tn,
        "edge_presence_precision": presence_precision,
        "edge_presence_recall": presence_recall,
        "edge_presence_f1": presence_f1,
        "typed_edge_tp": typed_tp,
        "typed_edge_fp": typed_fp,
        "typed_edge_fn": typed_fn,
        "typed_edge_precision": typed_precision,
        "typed_edge_recall": typed_recall,
        "typed_edge_f1": typed_f1,
        "edge_count_error": pred_edge_count - true_edge_count,
        "edge_count_abs_error": abs(pred_edge_count - true_edge_count),
        "edge_density_abs_error": abs(pred_edge_count - true_edge_count) / possible_edges
        if possible_edges else None,
        "degree_mae": degree_mae,
        "normalized_edit_proxy": (
            (node_total - node_correct) + (edge_total - edge_correct)
        ) / (node_total + edge_total) if (node_total + edge_total) else None,
        "node_type_counts": class_counts(x_pred.long(), x_true.long(), unknown_nodes, node_classes),
        "edge_type_present_counts": class_counts(
            e_pred.long(),
            e_true.long(),
            present_masked_edges,
            edge_classes,
        ),
    }


def draw_graph_with_positions(ax, graph, title, spec, pos):
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    if graph.number_of_nodes() == 0:
        return

    graph_pos = {node: pos[node] for node in graph.nodes() if node in pos}
    node_colors = [
        spec.node_colors[graph.nodes[node].get("attr", 0) % len(spec.node_colors)]
        for node in graph.nodes()
    ]
    labels = {
        node: label_at(spec.node_labels, graph.nodes[node].get("attr", 0))
        for node in graph.nodes()
    }
    edge_colors = [
        spec.edge_colors.get(graph.edges[edge].get("edge_type", 1), "#64748b")
        for edge in graph.edges()
    ]
    nx.draw_networkx_nodes(
        graph,
        graph_pos,
        ax=ax,
        node_color=node_colors,
        node_size=520,
        edgecolors="black",
        linewidths=0.8,
    )
    nx.draw_networkx_edges(graph, graph_pos, ax=ax, edge_color=edge_colors, width=1.4)
    nx.draw_networkx_labels(graph, graph_pos, labels=labels, ax=ax, font_size=6)


def save_completion_grid(cases, path, max_cases, spec):
    cases = cases[:max_cases]
    if not cases:
        return
    fig, axes = plt.subplots(len(cases), 3, figsize=(12.0, 3.6 * len(cases)))
    axes = np.array(axes).reshape(len(cases), 3)

    for row, case in enumerate(cases):
        reference = case["reference_graph"]
        completed = case["completed_graph"]
        partial = case["partial_graph"]
        pos = nx.spring_layout(reference, seed=0, k=0.9, iterations=100)
        prefix = (
            f"case {case['index']} | known {case['num_known_nodes']}/{case['num_nodes']} "
            f"({case['known_ratio']:.2f})"
        )
        draw_graph_with_positions(axes[row, 0], partial, f"{prefix}\npartial input", spec, pos)
        draw_graph_with_positions(axes[row, 1], completed, "model completion", spec, pos)
        draw_graph_with_positions(axes[row, 2], reference, "reference graph", spec, pos)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def aggregate_case_metrics(case_metrics):
    totals = {
        "node_unknown_correct": 0,
        "node_unknown_total": 0,
        "edge_masked_correct": 0,
        "edge_masked_total": 0,
        "edge_masked_present_correct": 0,
        "edge_masked_present_total": 0,
        "edge_presence_tp": 0,
        "edge_presence_fp": 0,
        "edge_presence_fn": 0,
        "edge_presence_tn": 0,
        "typed_edge_tp": 0,
        "typed_edge_fp": 0,
        "typed_edge_fn": 0,
    }
    scalar_lists = {
        "edge_count_error": [],
        "edge_count_abs_error": [],
        "edge_density_abs_error": [],
        "degree_mae": [],
        "normalized_edit_proxy": [],
    }
    node_type_counts = {}
    edge_type_present_counts = {}
    for metrics in case_metrics:
        for key in totals:
            totals[key] += int(metrics[key])
        for key in scalar_lists:
            if metrics[key] is not None:
                scalar_lists[key].append(float(metrics[key]))
        for target, source in [
            (node_type_counts, metrics["node_type_counts"]),
            (edge_type_present_counts, metrics["edge_type_present_counts"]),
        ]:
            for cls, values in source.items():
                if cls not in target:
                    target[cls] = {"tp": 0, "fp": 0, "fn": 0, "support": 0}
                for count_key in ["tp", "fp", "fn", "support"]:
                    target[cls][count_key] += int(values[count_key])

    edge_presence_precision, edge_presence_recall, edge_presence_f1 = precision_recall_f1(
        totals["edge_presence_tp"],
        totals["edge_presence_fp"],
        totals["edge_presence_fn"],
    )
    typed_edge_precision, typed_edge_recall, typed_edge_f1 = precision_recall_f1(
        totals["typed_edge_tp"],
        totals["typed_edge_fp"],
        totals["typed_edge_fn"],
    )

    return {
        "node_unknown_accuracy": (
            totals["node_unknown_correct"] / totals["node_unknown_total"]
            if totals["node_unknown_total"] else None
        ),
        "edge_masked_accuracy_all": (
            totals["edge_masked_correct"] / totals["edge_masked_total"]
            if totals["edge_masked_total"] else None
        ),
        "edge_masked_accuracy_present": (
            totals["edge_masked_present_correct"] / totals["edge_masked_present_total"]
            if totals["edge_masked_present_total"] else None
        ),
        "edge_presence_precision": edge_presence_precision,
        "edge_presence_recall": edge_presence_recall,
        "edge_presence_f1": edge_presence_f1,
        "typed_edge_precision": typed_edge_precision,
        "typed_edge_recall": typed_edge_recall,
        "typed_edge_f1": typed_edge_f1,
        "mean_edge_count_error": float(np.mean(scalar_lists["edge_count_error"]))
        if scalar_lists["edge_count_error"] else None,
        "mean_edge_count_abs_error": float(np.mean(scalar_lists["edge_count_abs_error"]))
        if scalar_lists["edge_count_abs_error"] else None,
        "mean_edge_density_abs_error": float(np.mean(scalar_lists["edge_density_abs_error"]))
        if scalar_lists["edge_density_abs_error"] else None,
        "mean_degree_mae": float(np.mean(scalar_lists["degree_mae"]))
        if scalar_lists["degree_mae"] else None,
        "mean_normalized_edit_proxy": float(np.mean(scalar_lists["normalized_edit_proxy"]))
        if scalar_lists["normalized_edit_proxy"] else None,
        "node_type_f1": f1_from_counts(node_type_counts),
        "edge_type_present_f1": f1_from_counts(edge_type_present_counts),
        **totals,
    }


def vector_from_histograms(left, right):
    keys = sorted(set(left) | set(right))
    p = np.array([float(left.get(key, 0.0)) for key in keys], dtype=np.float64)
    q = np.array([float(right.get(key, 0.0)) for key in keys], dtype=np.float64)
    return p, q


def js_divergence(left, right):
    p, q = vector_from_histograms(left, right)
    p_sum = p.sum()
    q_sum = q.sum()
    if p_sum <= 0 or q_sum <= 0:
        return None
    p = p / p_sum
    q = q / q_sum
    m = 0.5 * (p + q)
    eps = 1e-12
    kl_pm = np.sum(np.where(p > 0, p * np.log((p + eps) / (m + eps)), 0.0))
    kl_qm = np.sum(np.where(q > 0, q * np.log((q + eps) / (m + eps)), 0.0))
    return float(0.5 * (kl_pm + kl_qm))


def kl_divergence(left, right):
    p, q = vector_from_histograms(left, right)
    p_sum = p.sum()
    q_sum = q.sum()
    if p_sum <= 0 or q_sum <= 0:
        return None
    p = p / p_sum
    q = q / q_sum
    eps = 1e-12
    return float(np.sum(np.where(p > 0, p * np.log((p + eps) / (q + eps)), 0.0)))


def total_variation(left, right):
    p, q = vector_from_histograms(left, right)
    p_sum = p.sum()
    q_sum = q.sum()
    if p_sum <= 0 or q_sum <= 0:
        return None
    p = p / p_sum
    q = q / q_sum
    return float(0.5 * np.abs(p - q).sum())


def distribution_similarity(completed_distribution, reference_distribution):
    fields = [
        "graph_size_hist",
        "edge_count_hist",
        "degree_hist",
        "node_type_dist",
        "edge_type_dist",
    ]
    return {
        field: {
            "js_divergence": js_divergence(
                completed_distribution.get(field, {}),
                reference_distribution.get(field, {}),
            ),
            "kl_completed_to_reference": kl_divergence(
                completed_distribution.get(field, {}),
                reference_distribution.get(field, {}),
            ),
            "kl_reference_to_completed": kl_divergence(
                reference_distribution.get(field, {}),
                completed_distribution.get(field, {}),
            ),
            "total_variation": total_variation(
                completed_distribution.get(field, {}),
                reference_distribution.get(field, {}),
            ),
        }
        for field in fields
    }
