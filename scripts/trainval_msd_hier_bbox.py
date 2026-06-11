"""Train a bbox smoke model for MSD hierarchical GSDiff.

This script is intentionally separate from the original GSDiff training
scripts. It validates the hierarchical MSD coarse/local data path with a small
edge-aware transformer before adding full corner/polygon diffusion.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.msd_hier import EDGE_TYPES, MSDHierCoarseDataset, MSDHierLocalDataset
from gsdiff.msd_hier.bbox_model import HierBBoxModel, masked_smooth_l1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MSD hierarchical bbox smoke training.")
    parser.add_argument("--task", choices=["coarse", "local"], default="coarse")
    parser.add_argument("--root", default="datasets/msd_hier_v6")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-groups", type=int, default=64)
    parser.add_argument("--max-rooms", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--val-every", type=int, default=200)
    parser.add_argument("--val-batches", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def make_dataset(task: str, root: str, split: str, max_groups: int, max_rooms: int):
    if task == "coarse":
        return MSDHierCoarseDataset(root=root, split=split, max_groups=max_groups)
    return MSDHierLocalDataset(root=root, split=split, max_rooms=max_rooms)


def cycle_loader(loader):
    while True:
        for batch in loader:
            yield batch


def move_batch(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device, max_batches: int, amp: bool) -> float:
    model.eval()
    losses = []
    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        batch = move_batch(batch, device)
        with autocast(enabled=amp):
            pred = model(
                batch["node_features"].float(),
                batch["edge_types"].long(),
                batch["edge_present"].float(),
                batch["node_mask"].float(),
            )
            loss = masked_smooth_l1(pred, batch["target_bboxes"].float(), batch["node_mask"].float())
        losses.append(float(loss.detach().cpu()))
    model.train()
    return sum(losses) / max(len(losses), 1)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")

    out_dir = Path(args.out_dir or f"outputs/msd_hier_bbox_{args.task}")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = make_dataset(args.task, args.root, "train", args.max_groups, args.max_rooms)
    val_dataset = make_dataset(args.task, args.root, "val", args.max_groups, args.max_rooms)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    example = train_dataset[0]
    input_dim = int(example["node_features"].shape[-1])
    model = HierBBoxModel(
        input_dim=input_dim,
        num_edge_types=len(EDGE_TYPES),
        d_model=args.d_model,
        n_layers=args.layers,
        n_heads=args.heads,
        dropout=args.dropout,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=use_amp)
    train_iter = cycle_loader(train_loader)

    run_config = vars(args).copy()
    run_config.update(
        {
            "device": str(device),
            "amp_effective": use_amp,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "input_dim": input_dim,
            "num_edge_types": len(EDGE_TYPES),
            "params": sum(p.numel() for p in model.parameters()),
        }
    )
    with (out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    print(json.dumps(run_config, indent=2))

    best_val = float("inf")
    metrics = []
    model.train()
    for step in range(1, args.steps + 1):
        batch = move_batch(next(train_iter), device)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_amp):
            pred = model(
                batch["node_features"].float(),
                batch["edge_types"].long(),
                batch["edge_present"].float(),
                batch["node_mask"].float(),
            )
            loss = masked_smooth_l1(pred, batch["target_bboxes"].float(), batch["node_mask"].float())
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        record = {"step": step, "train_loss": float(loss.detach().cpu())}
        if step % args.val_every == 0 or step == args.steps:
            val_loss = evaluate(model, val_loader, device, args.val_batches, use_amp)
            record["val_loss"] = val_loss
            if val_loss < best_val:
                best_val = val_loss
                torch.save({"model": model.state_dict(), "config": run_config, "step": step, "val_loss": val_loss}, out_dir / "best.pt")
        metrics.append(record)

        if step % args.log_every == 0 or "val_loss" in record or step == 1:
            print(json.dumps(record), flush=True)

        if args.save_every > 0 and step % args.save_every == 0:
            torch.save({"model": model.state_dict(), "config": run_config, "step": step}, out_dir / f"step_{step:06d}.pt")

    torch.save({"model": model.state_dict(), "config": run_config, "step": args.steps}, out_dir / "last.pt")
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    # Avoid sparse CUDA allocator fragmentation on repeated short smoke runs.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    main()
