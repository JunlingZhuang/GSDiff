import torch
import torch.nn.functional as F


def is_absorbing_transition(cfg) -> bool:
    """Return True when the Hydra config requests the absorbing D3PM path."""
    return str(cfg.model.get("transition", "")).lower() == "absorbing"


def prepare_absorbing_dataset_infos(dataset_infos):
    """Expand DiGress input/output dimensions with one explicit [MASK] class.

    The MSD/RPLAN datasets store only real semantic classes. Absorbing D3PM
    needs one extra class that means "unknown token to be predicted". We add it
    at model construction time instead of rewriting the preprocessed dataset,
    so the existing vanilla DiGress configs/checkpoints remain compatible.
    """
    if getattr(dataset_infos, "absorbing_dims_prepared", False):
        return dataset_infos

    if dataset_infos.input_dims is None or dataset_infos.output_dims is None:
        raise ValueError("Call compute_input_output_dims before preparing absorbing dimensions.")

    dataset_infos.base_Xdim_output = int(dataset_infos.output_dims["X"])
    dataset_infos.base_Edim_output = int(dataset_infos.output_dims["E"])
    dataset_infos.mask_idx_X = dataset_infos.base_Xdim_output
    dataset_infos.mask_idx_E = dataset_infos.base_Edim_output

    dataset_infos.input_dims["X"] += 1
    dataset_infos.input_dims["E"] += 1
    dataset_infos.output_dims["X"] += 1
    dataset_infos.output_dims["E"] += 1

    # Keep human-readable labels aligned with the expanded tensors. The [MASK]
    # label is internal and should not be produced in final samples.
    if hasattr(dataset_infos, "node_decoder") and "[MASK]" not in dataset_infos.node_decoder:
        dataset_infos.node_decoder.append("[MASK]")
    if hasattr(dataset_infos, "edge_decoder") and "[MASK]" not in dataset_infos.edge_decoder:
        dataset_infos.edge_decoder.append("[MASK]")

    dataset_infos.absorbing_dims_prepared = True
    return dataset_infos


def pad_clean_features_for_absorbing(X: torch.Tensor, E: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Append a zero [MASK] column to clean one-hot node/edge features.

    Clean training data should never contain the [MASK] class. The extra zero
    column only makes shapes match the absorbing transition matrix and model
    input/output dimensions.
    """
    return F.pad(X, (0, 1), value=0.0), F.pad(E, (0, 1), value=0.0)


def strip_mask_class_from_sample(sample: list[torch.Tensor], mask_idx_X: int, mask_idx_E: int) -> list[torch.Tensor]:
    """Replace any accidental final [MASK] predictions with safe semantic defaults.

    A well-behaved sampler should not leave [MASK] tokens after the final
    unmasking step. This guard prevents downstream visualization/API code from
    receiving out-of-vocabulary labels if a checkpoint is still poorly trained.
    Node fallback is class 0; edge fallback is "none" class 0.
    """
    node_types, edge_types = sample
    node_types = node_types.clone()
    edge_types = edge_types.clone()
    node_types[node_types == mask_idx_X] = 0
    edge_types[edge_types == mask_idx_E] = 0
    return [node_types, edge_types]
