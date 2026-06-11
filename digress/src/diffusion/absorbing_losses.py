"""Loss functions for absorbing-state graph diffusion."""

import torch
import torch.nn.functional as F


def _node_class_weights(model, pred_X):
    """Return normalized class weights for rare room types, or None if disabled."""
    mode = str(model.cfg.model.get("node_class_balance", "none")).lower()
    if mode in {"none", "false", "0", "off"}:
        return None

    counts = getattr(model.dataset_info, "node_types", None)
    if counts is None:
        return None

    counts = counts[: model.base_Xdim_output].to(device=pred_X.device, dtype=pred_X.dtype)
    counts = counts.clamp(min=1.0)
    freq = counts / counts.sum().clamp(min=1.0)

    if mode in {"inverse", "inv"}:
        weights = 1.0 / freq.clamp(min=1e-6)
    else:
        # inverse_sqrt is less aggressive than full inverse-frequency weighting.
        weights = 1.0 / torch.sqrt(freq.clamp(min=1e-6))

    weights = weights / weights.mean().clamp(min=1e-6)
    max_weight = float(model.cfg.model.get("node_class_weight_max", 4.0))
    return weights.clamp(max=max_weight)


def _known_to_hidden_bridge_mask(model, noisy_data, true_E_idx, node_mask):
    """Mark real masked edges that connect visible partial nodes to hidden nodes.

    Full completion failed mainly by producing usable edge counts without enough
    reliable attachment to the observed partial graph. This mask lets the loss
    up-weight exactly those known-to-hidden real edges without changing the data
    schema or the model architecture.
    """
    bridge_weight = float(model.cfg.model.get("bridge_edge_loss_weight", 1.0))
    if bridge_weight <= 1.0:
        return None

    if bool(model.cfg.model.get("bridge_edges_only_full_completion", True)):
        if noisy_data.get("mask_strategy") != "full_completion":
            return None

    X_t_idx = noisy_data["X_t"].argmax(dim=-1)
    known_nodes = (X_t_idx != model.mask_idx_X) & node_mask
    hidden_nodes = (X_t_idx == model.mask_idx_X) & node_mask

    known_hidden = known_nodes.unsqueeze(2) & hidden_nodes.unsqueeze(1)
    hidden_known = hidden_nodes.unsqueeze(2) & known_nodes.unsqueeze(1)
    return (known_hidden | hidden_known) & (true_E_idx > 0)


def _weighted_mean(values, weights):
    denom = weights.sum().clamp(min=1e-6)
    return (values * weights).sum() / denom


def masked_ce_loss(model, pred, noisy_data, true_X, true_E, node_mask):
    """Cross entropy only where the noisy graph currently contains [MASK].

    `model` is passed in so this helper can stay stateless while still reading
    config values and absorbing dimensions from the LightningModule.
    """
    X_t_idx = noisy_data["X_t"].argmax(dim=-1)
    E_t_idx = noisy_data["E_t"].argmax(dim=-1)
    true_X_idx = true_X[..., :model.base_Xdim_output].argmax(dim=-1)
    true_E_idx = true_E[..., :model.base_Edim_output].argmax(dim=-1)

    x_positions = (X_t_idx == model.mask_idx_X) & node_mask
    n = node_mask.size(1)
    diagonal = torch.eye(n, device=node_mask.device, dtype=torch.bool).unsqueeze(0)
    edge_slots = node_mask.unsqueeze(1) & node_mask.unsqueeze(2) & ~diagonal
    e_positions = (E_t_idx == model.mask_idx_E) & edge_slots

    # The network has an output logit for [MASK] because the residual
    # GraphTransformer expects equal input/output dimensions. We exclude that
    # last logit from CE targets so the model learns real classes.
    pred_X = pred.X[..., :model.base_Xdim_output]
    pred_E = pred.E[..., :model.base_Edim_output]

    zero = pred.X.sum() * 0.0
    node_weights = _node_class_weights(model, pred_X)
    loss_X = (
        F.cross_entropy(pred_X[x_positions], true_X_idx[x_positions], weight=node_weights)
        if x_positions.any()
        else zero
    )
    loss_E = zero
    loss_E_presence = zero
    loss_E_type = zero
    loss_E_density = zero
    loss_E_degree = zero
    if e_positions.any():
        edge_loss_mode = model.cfg.model.get("edge_loss_mode", "weighted_ce")
        if edge_loss_mode == "split_presence_type":
            true_edges = true_E_idx[e_positions]
            pred_edges = pred_E[e_positions]
            present_target = (true_edges > 0).float()
            presence_logits = torch.logsumexp(pred_edges[:, 1:], dim=-1) - pred_edges[:, 0]

            bridge_mask = _known_to_hidden_bridge_mask(model, noisy_data, true_E_idx, node_mask)
            bridge_weights = None
            if bridge_mask is not None:
                bridge_weight = float(model.cfg.model.get("bridge_edge_loss_weight", 1.0))
                bridge_weights = torch.ones_like(present_target)
                bridge_weights = torch.where(
                    bridge_mask[e_positions],
                    torch.full_like(bridge_weights, bridge_weight),
                    bridge_weights,
                )

            loss_E_presence = F.binary_cross_entropy_with_logits(
                presence_logits,
                present_target,
                weight=bridge_weights,
            )
            true_present = true_edges > 0
            if true_present.any():
                type_losses = F.cross_entropy(
                    pred_edges[true_present, 1:],
                    true_edges[true_present] - 1,
                    reduction="none",
                )
                if bridge_weights is not None:
                    loss_E_type = _weighted_mean(type_losses, bridge_weights[true_present])
                else:
                    loss_E_type = type_losses.mean()

            density_weight = float(model.cfg.model.get("edge_density_loss_weight", 0.0))
            degree_weight = float(model.cfg.model.get("edge_degree_loss_weight", 0.0))
            if density_weight > 0 or degree_weight > 0:
                present_prob = F.softmax(pred_E, dim=-1)[..., 1:].sum(dim=-1)
            if density_weight > 0:
                pred_counts = (present_prob * e_positions.float()).sum(dim=(1, 2))
                true_counts = ((true_E_idx > 0) & e_positions).float().sum(dim=(1, 2))
                denom = e_positions.float().sum(dim=(1, 2)).clamp(min=1.0)
                loss_E_density = torch.mean(torch.abs(pred_counts - true_counts) / denom)
            if degree_weight > 0:
                pred_degree = (present_prob * e_positions.float()).sum(dim=2)
                true_degree = (((true_E_idx > 0) & e_positions).float()).sum(dim=2)
                degree_denom = e_positions.float().sum(dim=2).clamp(min=1.0)
                loss_E_degree = torch.mean(torch.abs(pred_degree - true_degree) / degree_denom)
            loss_E = (
                float(model.cfg.model.get("edge_presence_loss_weight", 1.0)) * loss_E_presence
                + float(model.cfg.model.get("edge_type_loss_weight", 1.0)) * loss_E_type
                + float(model.cfg.model.get("edge_density_loss_weight", 0.0)) * loss_E_density
                + float(model.cfg.model.get("edge_degree_loss_weight", 0.0)) * loss_E_degree
            )
        elif float(model.cfg.model.get("edge_present_loss_weight", 1.0)) == 1.0:
            loss_E = F.cross_entropy(pred_E[e_positions], true_E_idx[e_positions])
        else:
            edge_weights = torch.ones(model.base_Edim_output, device=pred_E.device, dtype=pred_E.dtype)
            edge_weights[1:] = float(model.cfg.model.get("edge_present_loss_weight", 1.0))
            loss_E = F.cross_entropy(
                pred_E[e_positions],
                true_E_idx[e_positions],
                weight=edge_weights,
            )

    total = loss_X + float(model.cfg.model.lambda_train[0]) * loss_E
    return total, {
        "loss": total.detach(),
        "x_ce": loss_X.detach(),
        "e_ce": loss_E.detach(),
        "e_presence_ce": loss_E_presence.detach(),
        "e_type_ce": loss_E_type.detach(),
        "e_density": loss_E_density.detach(),
        "e_degree": loss_E_degree.detach(),
        "x_masked": x_positions.float().sum().detach(),
        "e_masked": e_positions.float().sum().detach(),
    }
