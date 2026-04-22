"""Runner for GRANv2 training and testing.

Extends :class:`runner.gran_runner.GranRunner` to:
  1. Handle the ``(edge_loss, attr_loss)`` tuple returned by
     :meth:`model.gran_v2.GRANv2.forward` during training. The two scalars
     are combined via ``total_loss = edge_loss + lambda_attr * attr_loss``
     before ``.backward()``.
  2. Pass ``node_attrs`` through the data-loader pipeline so the model
     eventually sees attribute labels when computing ``attr_loss``.
  3. During test, unpack the ``(A_list, attr_list)`` tuple returned by
     sampling instead of just ``A_list``.

Only :meth:`train` and :meth:`test` are overridden; all graph loading,
train/dev/test splitting, and PMF bookkeeping is inherited from the parent.

Shape glossary:
    B = batch_size
    C = num_canonical_order
    N = max_num_nodes
    H = hidden_dim
    A = num_attr_classes
"""
from __future__ import (division, print_function)

import os
import copy
import time
import pickle
from collections import defaultdict

import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
from tqdm import tqdm

from model import *
from dataset import *
from utils.logger import get_logger
from utils.train_helper import snapshot, load_model
from utils.data_parallel import DataParallel
from utils.dist_helper import compute_mmd, gaussian_emd
from utils.eval_helper import eval_acc_lobster_graph
from utils.vis_helper import draw_graph_list, draw_graph_list_separate
from runner.gran_runner import GranRunner, get_graph, evaluate

logger = get_logger('exp_logger')
__all__ = ['GranRunnerV2']


class GranRunnerV2(GranRunner):
    """Runner supporting joint edge + attribute training for GRANv2.

    Differences from :class:`GranRunner`:
      * Training loss is ``edge_loss + lambda_attr * attr_loss``.
      * ``node_attrs`` is threaded through the data pipeline so the model
        can (once its input wiring is hooked up) train the attribute head.
      * The sampling path returns ``(A_list, attr_list)`` rather than just
        ``A_list``; we unpack and only use ``A_list`` for graph-level
        MMD evaluation, but keep ``attr_list`` around for downstream use.

    Args:
        config: an EasyDict-style config. Accesses ``config.train.lambda_attr``
            (default 1.0) for the attribute-loss weight.
    """

    def __init__(self, config):
        """Initialize from an EasyDict config (same schema as GranRunner).

        v2-specific field:
            config.train.lambda_attr -- weight of the attribute CE loss in
                                         total_loss = edge_loss + lambda * attr_loss.
                                         Defaults to 1.0 if unset.

        [v2] Fix parent's split bug: original GranRunner sets
            graphs_dev = graphs[:num_dev]
        which is a SUBSET of graphs_train, not a held-out validation set.
        We re-slice into three DISJOINT partitions:
            train: [0,               num_train)
            dev:   [num_train,       num_train + num_dev)
            test:  [num_train + num_dev, num_graphs)
        """
        super().__init__(config)
        # lambda_attr: weight of the node-attribute CrossEntropy loss in the
        # joint objective. Configured via config.train.lambda_attr;
        # default 1.0 so it matches the edge loss by default.
        self.lambda_attr = getattr(config.train, 'lambda_attr', 1.0)

        # [v2] Separate train / val SummaryWriters.
        # Writing to train/ and val/ subdirs with SHARED tag names
        # (edge_loss, attr_loss, total_loss) makes TensorBoard overlay both
        # series onto the same chart. The parent's self.writer still exists
        # (it points at the run root) so per-iteration iter-level scalars
        # continue to work unchanged.
        import os as _os
        from tensorboardX import SummaryWriter as _SW
        self.train_writer = _SW(log_dir=_os.path.join(config.save_dir, 'train'))
        self.val_writer = _SW(log_dir=_os.path.join(config.save_dir, 'val'))

        # [v2] optional cap on how many graphs to actually use, before
        # splitting. Makes epochs faster for quick experiments without
        # re-running the preprocessor. `train_ratio` / `dev_ratio` apply as
        # usual within this capped subset.
        #   dataset.total_graphs = 10000  -> use first 10k graphs; train = 9k, dev = 500
        #   dataset.total_graphs unset/0  -> use all graphs (current behavior)
        total_cap = getattr(config.dataset, 'total_graphs', 0) or 0
        if total_cap > 0 and total_cap < len(self.graphs):
            logger.info(
                "capping dataset: {} -> {} graphs (dataset.total_graphs)".format(
                    len(self.graphs), total_cap))
            self.graphs = self.graphs[:total_cap]
            self.num_graphs = len(self.graphs)
            # Rebuild the node-count PMF that GRANv2 sampling uses.
            import numpy as _np
            pmf = _np.bincount([len(g.nodes) for g in self.graphs])[1:]
            if pmf.sum() > 0:
                self.num_nodes_pmf_train = pmf / pmf.sum()
                self.max_num_nodes = len(pmf)

        # [v2] re-slice to disjoint train/dev/test
        num_train = int(self.num_graphs * self.train_ratio)
        num_dev = int(self.num_graphs * self.dev_ratio)
        self.graphs_train = self.graphs[:num_train]
        self.graphs_dev = self.graphs[num_train:num_train + num_dev]
        self.graphs_test = self.graphs[num_train + num_dev:]
        self.num_train = len(self.graphs_train)
        self.num_dev = len(self.graphs_dev)
        self.num_test_gt = len(self.graphs_test)
        logger.info(
            "v2 disjoint split: train={} / dev={} / test={}".format(
                self.num_train, self.num_dev, self.num_test_gt))

    # ======================================================================
    # Training
    # ======================================================================
    def train(self):
        """Joint edge + attribute training loop.

        Mirrors :meth:`GranRunner.train` but:
          * threads ``node_attrs`` (shape ``(B, C, N)`` int64) into the
            per-GPU data dict so downstream model code can wire up attr loss;
          * unpacks ``model(...)`` output as ``(edge_loss, attr_loss)``
            when it is a tuple (GRANv2), falling back to scalar ``edge_loss``
            for the original GRANMixtureBernoulli;
          * uses ``next(train_iterator)`` (Py3) instead of the parent's
            ``.next()`` (Py2) call.

        Returns:
            int: ``1`` on successful completion (matches parent signature).
        """
        # [v2] create data loader (same API; GRANDataV2 adds node_attrs)
        train_dataset = eval(self.dataset_conf.loader_name)(
            self.config, self.graphs_train, tag='train')
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.train_conf.batch_size,
            shuffle=self.train_conf.shuffle,
            num_workers=self.train_conf.num_workers,
            collate_fn=train_dataset.collate_fn,
            drop_last=False)

        # [v2] validation loader -- uses the held-out dev graphs. Only used
        # every `train.valid_epoch` epochs; forward-only (no backward / no
        # precompute overwrite so it doesn't clobber train shards).
        val_loader = None
        if len(self.graphs_dev) > 0:
            # Build a dev-tagged config that does NOT re-run precompute.
            val_dataset = eval(self.dataset_conf.loader_name)(
                self.config, self.graphs_dev, tag='dev')
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=self.train_conf.batch_size,
                shuffle=False,
                num_workers=self.train_conf.num_workers,
                collate_fn=val_dataset.collate_fn,
                drop_last=False)

        # [v2] pick up GRANv2 via config.model.name
        model = eval(self.model_conf.name)(self.config)

        # [v2 plan A] Auto-compute inverse-frequency attr class weights from
        # the training set when config.model.attr_class_weight == 'auto'.
        # Must happen BEFORE DataParallel wrap so we can call the setter on
        # the raw GRANv2 instance.
        if getattr(self.model_conf, 'attr_class_weight', None) == 'auto':
            weights = self._compute_auto_attr_class_weights()
            model.set_attr_class_weights(weights)
            logger.info(
                "auto attr_class_weight: %s (clamped to [%.2f, %.2f])" % (
                    ["%.4f" % w for w in weights.tolist()],
                    float(weights.min()), float(weights.max())))

        if self.use_gpu:
            model = DataParallel(model, device_ids=self.gpus).to(self.device)

        # ---- optimizer (unchanged) -----------------------------------------
        params = filter(lambda p: p.requires_grad, model.parameters())
        if self.train_conf.optimizer == 'SGD':
            optimizer = optim.SGD(
                params,
                lr=self.train_conf.lr,
                momentum=self.train_conf.momentum,
                weight_decay=self.train_conf.wd)
        elif self.train_conf.optimizer == 'Adam':
            optimizer = optim.Adam(
                params, lr=self.train_conf.lr, weight_decay=self.train_conf.wd)
        else:
            raise ValueError("Non-supported optimizer!")

        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=self.train_conf.lr_decay_epoch,
            gamma=self.train_conf.lr_decay)

        optimizer.zero_grad()

        # ---- resume (unchanged) --------------------------------------------
        resume_epoch = 0
        if self.train_conf.is_resume:
            model_file = os.path.join(self.train_conf.resume_dir,
                                      self.train_conf.resume_model)
            load_model(
                model.module if self.use_gpu else model,
                model_file,
                self.device,
                optimizer=optimizer,
                scheduler=lr_scheduler)
            resume_epoch = self.train_conf.resume_epoch

        # ---- training loop --------------------------------------------------
        # [v2] Progress reporting style: one tqdm bar for epochs, one summary
        # line per epoch with mean edge/attr/total losses. Per-iteration logs
        # still go to TensorBoard for detailed inspection but are NOT spammed
        # to the console.
        iter_count = 0
        results = defaultdict(list)
        total_epochs = self.train_conf.max_epoch - resume_epoch

        # [v2] track best model by val loss; saved to model_best.pth and
        # overwritten each time val_total hits a new minimum.
        best_val_total = float('inf')
        best_val_epoch = -1
        best_model_path = os.path.join(self.config.save_dir, 'model_best.pth')

        epoch_bar = tqdm(
            range(resume_epoch, self.train_conf.max_epoch),
            total=total_epochs,
            desc='train',
            unit='epoch',
        )

        for epoch in epoch_bar:
            model.train()
            lr_scheduler.step()
            train_iterator = train_loader.__iter__()

            # Per-epoch running sums so we can log a single summary line.
            epoch_edge_sum = 0.0
            epoch_attr_sum = 0.0
            epoch_iters = 0

            n_inner = len(train_loader) // self.num_gpus
            for inner_iter in range(n_inner):
                optimizer.zero_grad()

                batch_data = []
                if self.use_gpu:
                    for _ in self.gpus:
                        # [v2] Py3: use next(iter) instead of iter.next()
                        data = next(train_iterator)
                        batch_data.append(data)
                        iter_count += 1

                avg_edge_loss = .0      # [v2] track edge / attr losses separately
                avg_attr_loss = .0
                for ff in range(self.dataset_conf.num_fwd_pass):
                    batch_fwd = []

                    if self.use_gpu:
                        for dd, gpu_id in enumerate(self.gpus):
                            data = {}
                            data['adj'] = batch_data[dd][ff]['adj'].pin_memory().to(gpu_id, non_blocking=True)
                            data['edges'] = batch_data[dd][ff]['edges'].pin_memory().to(gpu_id, non_blocking=True)
                            data['node_idx_gnn'] = batch_data[dd][ff]['node_idx_gnn'].pin_memory().to(gpu_id, non_blocking=True)
                            data['node_idx_feat'] = batch_data[dd][ff]['node_idx_feat'].pin_memory().to(gpu_id, non_blocking=True)
                            data['label'] = batch_data[dd][ff]['label'].pin_memory().to(gpu_id, non_blocking=True)
                            data['att_idx'] = batch_data[dd][ff]['att_idx'].pin_memory().to(gpu_id, non_blocking=True)
                            data['subgraph_idx'] = batch_data[dd][ff]['subgraph_idx'].pin_memory().to(gpu_id, non_blocking=True)
                            data['subgraph_idx_base'] = batch_data[dd][ff]['subgraph_idx_base'].pin_memory().to(gpu_id, non_blocking=True)
                            # [v2] thread node_attrs through when provided by the loader
                            if 'node_attrs' in batch_data[dd][ff]:
                                data['node_attrs'] = batch_data[dd][ff]['node_attrs'].pin_memory().to(gpu_id, non_blocking=True)
                            # [planc] thread per-state-row attr labels for the
                            # attr-conditioned edge head.
                            if 'subgraph_node_attrs' in batch_data[dd][ff]:
                                data['subgraph_node_attrs'] = batch_data[dd][ff]['subgraph_node_attrs'].pin_memory().to(gpu_id, non_blocking=True)
                            batch_fwd.append((data,))

                    if batch_fwd:
                        result = model(*batch_fwd)
                        # [v2] GRANv2.forward returns (edge_loss, attr_loss);
                        # the original GRANMixtureBernoulli returns a scalar.
                        # Handle both.
                        if isinstance(result, tuple) and len(result) == 2:
                            edge_loss = result[0].mean()
                            attr_loss = result[1].mean()
                        else:
                            edge_loss = result.mean()
                            attr_loss = edge_loss.new_zeros(())  # 0-dim tensor
                        # [v2] joint objective
                        total_loss = edge_loss + self.lambda_attr * attr_loss

                        avg_edge_loss = avg_edge_loss + edge_loss
                        avg_attr_loss = avg_attr_loss + attr_loss

                        total_loss.backward()

                optimizer.step()
                avg_edge_loss = avg_edge_loss / float(self.dataset_conf.num_fwd_pass)
                avg_attr_loss = avg_attr_loss / float(self.dataset_conf.num_fwd_pass)

                # [v2] record losses to TensorBoard every iter (no console spam)
                edge_loss_val = float(avg_edge_loss.data.cpu().numpy()) \
                    if torch.is_tensor(avg_edge_loss) else float(avg_edge_loss)
                attr_loss_val = float(avg_attr_loss.data.cpu().numpy()) \
                    if torch.is_tensor(avg_attr_loss) else float(avg_attr_loss)
                total_loss_val = edge_loss_val + self.lambda_attr * attr_loss_val

                self.writer.add_scalar('train_loss', total_loss_val, iter_count)
                self.writer.add_scalar('edge_loss', edge_loss_val, iter_count)
                self.writer.add_scalar('attr_loss', attr_loss_val, iter_count)
                results['train_loss'] += [total_loss_val]
                results['edge_loss'] += [edge_loss_val]
                results['attr_loss'] += [attr_loss_val]
                results['train_step'] += [iter_count]

                # [v2] update epoch-level running sums
                epoch_edge_sum += edge_loss_val
                epoch_attr_sum += attr_loss_val
                epoch_iters += 1

                # [v2] live-update tqdm postfix with the RUNNING EPOCH MEAN
                # (same convention as HuggingFace Trainer / Keras). Showing
                # the per-batch instantaneous value makes the bar look noisy
                # even when training is healthy. The running mean is smooth,
                # has no lag, and matches the final epoch-summary INFO line.
                running_edge = epoch_edge_sum / epoch_iters
                running_attr = epoch_attr_sum / epoch_iters
                running_total = running_edge + self.lambda_attr * running_attr
                epoch_bar.set_postfix(
                    edge=f'{running_edge:.4f}',
                    attr=f'{running_attr:.4f}',
                    total=f'{running_total:.4f}',
                )

            # [v2] one summary line per epoch, ML-style
            val_edge = val_attr = val_total = None
            if epoch_iters > 0:
                mean_edge = epoch_edge_sum / epoch_iters
                mean_attr = epoch_attr_sum / epoch_iters
                mean_total = mean_edge + self.lambda_attr * mean_attr
                # [v2] write to the TRAIN sub-writer using shared tag names so
                # TensorBoard overlays train + val on one chart.
                self.train_writer.add_scalar('edge_loss', mean_edge, epoch + 1)
                self.train_writer.add_scalar('attr_loss', mean_attr, epoch + 1)
                self.train_writer.add_scalar('total_loss', mean_total, epoch + 1)

                # [v2] validation step every `valid_epoch` epochs
                if (val_loader is not None
                        and (epoch + 1) % self.train_conf.valid_epoch == 0):
                    val_edge, val_attr, val_total = self._validate(
                        model, val_loader)
                    # SAME tag names -> appear as second line on the same chart
                    self.val_writer.add_scalar('edge_loss', val_edge, epoch + 1)
                    self.val_writer.add_scalar('attr_loss', val_attr, epoch + 1)
                    self.val_writer.add_scalar('total_loss', val_total, epoch + 1)
                    results['val_edge_loss'] += [val_edge]
                    results['val_attr_loss'] += [val_attr]
                    results['val_total_loss'] += [val_total]
                    results['val_epoch'] += [epoch + 1]

                    # [v2] save best model so far (by val_total)
                    if val_total < best_val_total:
                        best_val_total = val_total
                        best_val_epoch = epoch + 1
                        torch.save(
                            {
                                'model': (model.module if self.use_gpu
                                          else model).state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'scheduler': lr_scheduler.state_dict(),
                                'step': epoch + 1,
                                'val_edge': val_edge,
                                'val_attr': val_attr,
                                'val_total': val_total,
                            },
                            best_model_path,
                        )
                        logger.info(
                            "** new best val_total={:.4f} @ epoch {:04d}, "
                            "saved to {}".format(
                                val_total, epoch + 1, 'model_best.pth'))

                # Per-epoch summary. One line normally, two aligned lines when
                # we also just computed validation losses.
                cur_lr = optimizer.param_groups[0]['lr']
                if val_edge is not None:
                    logger.info(
                        "ep {ep:04d}/{total_ep:04d}  lr={lr:.2e}  "
                        "train: edge={te:.4f}  attr={ta:.4f}  total={tt:.4f}\n"
                        "                        "
                        "  val: edge={ve:.4f}  attr={va:.4f}  total={vt:.4f}".format(
                            ep=epoch + 1, total_ep=self.train_conf.max_epoch,
                            lr=cur_lr,
                            te=mean_edge, ta=mean_attr, tt=mean_total,
                            ve=val_edge, va=val_attr, vt=val_total))
                else:
                    logger.info(
                        "ep {ep:04d}/{total_ep:04d}  lr={lr:.2e}  "
                        "train: edge={te:.4f}  attr={ta:.4f}  total={tt:.4f}".format(
                            ep=epoch + 1, total_ep=self.train_conf.max_epoch,
                            lr=cur_lr,
                            te=mean_edge, ta=mean_attr, tt=mean_total))

            # snapshot model (same signature as parent's call)
            if (epoch + 1) % self.train_conf.snapshot_epoch == 0:
                logger.info("saving snapshot @ epoch {:04d}".format(epoch + 1))
                snapshot(
                    model.module if self.use_gpu else model,
                    optimizer,
                    self.config,
                    epoch + 1,
                    scheduler=lr_scheduler)

        epoch_bar.close()

        pickle.dump(results, open(os.path.join(self.config.save_dir, 'train_stats.p'), 'wb'))
        self.writer.close()
        self.train_writer.close()
        self.val_writer.close()

        # [v2] Emit structured training history as JSON and CSV for easy
        # machine parsing (by dashboards, downstream analysis tools, or LLMs
        # reading the experiment directory). `train_stats.p` is still saved
        # above for backward compatibility.
        self._dump_training_history(results, best_val_total, best_val_epoch)

        # [v2] final summary
        if best_val_epoch > 0:
            logger.info(
                "training done. best val_total={:.4f} @ epoch {:04d} "
                "(saved as model_best.pth)".format(
                    best_val_total, best_val_epoch))
        else:
            logger.info("training done. (no validation recorded)")

        return 1

    # ======================================================================
    # Attribute class weight auto-computation (plan A)
    # ======================================================================
    def _compute_auto_attr_class_weights(self):
        """Compute inverse-frequency per-class weights from ``graphs_train``.

        Formula (sklearn "balanced"-style):
            w[c] = total / (A * max(count[c], 1))

        Weights are then clamped to ``[0.1, 10.0]`` to prevent a rare class
        from dominating the gradient — e.g. a class with 0.1% prevalence
        would otherwise get weight ~1000. Empirically cap=10 is a reasonable
        upper bound for class imbalance studies and matches focal-loss
        practice.

        Returns a float32 tensor of shape ``(num_attr_classes,)``.
        """
        A = self.config.model.num_attr_classes
        counts = torch.zeros(A, dtype=torch.float64)
        for g in self.graphs_train:
            for _, node_data in g.nodes(data=True):
                # Node attrs are stored at G.nodes[n]['attr']; fall back to 0
                # to mirror gran_data_v2's `.get('attr', 0)` behavior.
                attr = int(node_data.get('attr', 0))
                if 0 <= attr < A:
                    counts[attr] += 1
        total = counts.sum()
        if total == 0:
            # No attrs found: uniform fallback.
            return torch.ones(A, dtype=torch.float32)
        weights = total / (A * counts.clamp(min=1.0))
        weights = weights.clamp(min=0.1, max=10.0)
        return weights.float()

    # ======================================================================
    # Structured logging helpers
    # ======================================================================
    def _dump_training_history(self, results, best_val_total, best_val_epoch):
        """Emit training history as JSON and CSV.

        JSON structure:
          {
            "run_id": str,
            "config_summary": { ...key hyperparameters },
            "best": { "val_total": float, "epoch": int },
            "per_iter": [{ "step": int, "edge_loss": f, "attr_loss": f, ... }],
            "per_epoch_val": [{ "epoch": int, "val_edge": f, ... }]
          }
        """
        import csv
        import json

        save_dir = self.config.save_dir

        # Condensed config for quick lookup by readers
        config_summary = {
            'run_id': getattr(self.config, 'run_id', None),
            'dataset_name': self.config.dataset.name,
            'total_graphs': getattr(self.config.dataset, 'total_graphs', 0),
            'train_ratio': self.config.dataset.train_ratio,
            'dev_ratio': self.config.dataset.dev_ratio,
            'node_order': self.config.dataset.node_order,
            'model_name': self.config.model.name,
            'hidden_dim': self.config.model.hidden_dim,
            'num_GNN_layers': self.config.model.num_GNN_layers,
            'num_mix_component': self.config.model.num_mix_component,
            'num_attr_classes': self.config.model.num_attr_classes,
            'num_canonical_order': self.config.model.num_canonical_order,
            'use_gatv2': getattr(self.config.model, 'use_gatv2', False),
            'use_degree_feature': getattr(
                self.config.model, 'use_degree_feature', False),
            'attr_temperature': getattr(
                self.config.model, 'attr_temperature', 1.0),
            'lr': self.config.train.lr,
            'batch_size': self.config.train.batch_size,
            'max_epoch': self.config.train.max_epoch,
            'lambda_attr': getattr(self.config.train, 'lambda_attr', 1.0),
            'lr_decay_epoch': list(self.config.train.lr_decay_epoch),
            'lr_decay': self.config.train.lr_decay,
        }

        # Per-iteration series (aligned lists: train_step, edge_loss, ...)
        per_iter = []
        steps = results.get('train_step', [])
        edge_losses = results.get('edge_loss', [])
        attr_losses = results.get('attr_loss', [])
        total_losses = results.get('train_loss', [])
        n_iters = min(len(steps), len(edge_losses), len(attr_losses),
                      len(total_losses))
        for i in range(n_iters):
            per_iter.append({
                'step': int(steps[i]),
                'edge_loss': float(edge_losses[i]),
                'attr_loss': float(attr_losses[i]),
                'total_loss': float(total_losses[i]),
            })

        # Per-epoch val points
        val_epochs = results.get('val_epoch', [])
        val_edges = results.get('val_edge_loss', [])
        val_attrs = results.get('val_attr_loss', [])
        val_totals = results.get('val_total_loss', [])
        per_epoch_val = []
        for i in range(min(len(val_epochs), len(val_edges))):
            per_epoch_val.append({
                'epoch': int(val_epochs[i]),
                'val_edge_loss': float(val_edges[i]),
                'val_attr_loss': float(val_attrs[i]),
                'val_total_loss': float(val_totals[i]),
            })

        history = {
            'run_id': config_summary['run_id'],
            'config_summary': config_summary,
            'best': {
                'val_total': float(best_val_total)
                             if best_val_total != float('inf') else None,
                'epoch': int(best_val_epoch) if best_val_epoch > 0 else None,
            },
            'per_iter': per_iter,
            'per_epoch_val': per_epoch_val,
        }

        # JSON (full structure)
        with open(os.path.join(save_dir, 'training_history.json'),
                  'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        # CSV for iter-level (for quick plotting with pandas/excel)
        with open(os.path.join(save_dir, 'training_history_iter.csv'),
                  'w', encoding='utf-8', newline='') as f:
            w = csv.DictWriter(
                f, fieldnames=['step', 'edge_loss', 'attr_loss', 'total_loss'])
            w.writeheader()
            w.writerows(per_iter)

        # CSV for val-level
        with open(os.path.join(save_dir, 'training_history_val.csv'),
                  'w', encoding='utf-8', newline='') as f:
            w = csv.DictWriter(
                f, fieldnames=['epoch', 'val_edge_loss', 'val_attr_loss',
                               'val_total_loss'])
            w.writeheader()
            w.writerows(per_epoch_val)

        logger.info(
            'saved structured training history: training_history.json + '
            'training_history_iter.csv + training_history_val.csv')

    def _dump_test_results(self, graphs_gen, attr_lists, gen_run_time,
                           mmd_dev, mmd_test):
        """Write machine-readable test results to the run dir.

        Produces:
          * test_results.json — full config summary + MMD + stats + class histogram
          * test_mmd.csv     — flat table (set, metric, value) for quick plotting
          * test_attrs.csv   — per-graph node attribute sequence for inspection

        Everything lands in ``self.config.save_dir`` which, in test mode,
        is the checkpoint's original run dir (so each snapshot has its
        own ``test_results.json`` alongside it).
        """
        import csv
        import json
        from collections import Counter

        save_dir = self.config.save_dir

        # --- dataset + config summary ------------------------------------
        config_summary = {
            'run_id': getattr(self.config, 'run_id', None),
            'test_model_dir': self.config.test.test_model_dir,
            'test_model_name': self.config.test.test_model_name,
            'num_test_gen': self.config.test.num_test_gen,
            'dataset_name': self.config.dataset.name,
            'total_graphs': getattr(self.config.dataset, 'total_graphs', 0),
            'train_ratio': self.config.dataset.train_ratio,
            'dev_ratio': self.config.dataset.dev_ratio,
            'node_order': self.config.dataset.node_order,
            'num_canonical_order': self.config.model.num_canonical_order,
            'use_gatv2': getattr(self.config.model, 'use_gatv2', False),
            'use_degree_feature': getattr(
                self.config.model, 'use_degree_feature', False),
            'hidden_dim': self.config.model.hidden_dim,
            'num_GNN_layers': self.config.model.num_GNN_layers,
        }

        # --- per-generated-graph stats -----------------------------------
        gen_stats = []
        attr_counter = Counter()
        for idx, G in enumerate(graphs_gen):
            n = G.number_of_nodes()
            e = G.number_of_edges()
            attrs = None
            if attr_lists is not None and idx < len(attr_lists):
                attrs = [int(a) for a in attr_lists[idx][:n]]
                for a in attrs:
                    attr_counter[a] += 1
            gen_stats.append({
                'idx': idx,
                'num_nodes': int(n),
                'num_edges': int(e),
                'attrs': attrs,
            })

        total_attrs = sum(attr_counter.values())
        attr_histogram = {
            str(cls): {
                'count': int(cnt),
                'ratio': cnt / total_attrs if total_attrs else 0.0,
            }
            for cls, cnt in sorted(attr_counter.items())
        }

        # --- reference class histogram (for comparison) ------------------
        ref_counter = Counter()
        ref_graphs = (self.graphs_test if len(self.graphs_test) > 0
                      else self.graphs_train)
        for G in ref_graphs:
            for n in G.nodes():
                ref_counter[int(G.nodes[n].get('attr', 0))] += 1
        ref_total = sum(ref_counter.values())
        ref_histogram = {
            str(cls): {
                'count': int(cnt),
                'ratio': cnt / ref_total if ref_total else 0.0,
            }
            for cls, cnt in sorted(ref_counter.items())
        }

        # --- graph size distribution (gen vs ref) ------------------------
        gen_sizes = Counter(int(G.number_of_nodes()) for G in graphs_gen)
        ref_sizes = Counter(int(G.number_of_nodes()) for G in ref_graphs)

        def _size_hist(counter):
            total = sum(counter.values())
            return {str(k): {'count': int(v),
                             'ratio': v / total if total else 0.0}
                    for k, v in sorted(counter.items())}

        # --- attr-aware metrics (endpoint pair KL, living-count, per-class degree)
        from utils.attr_metrics import (
            endpoint_attr_pair_counts,
            kl_divergence,
            living_count_per_graph,
            per_class_degree_stats,
        )

        num_attr_classes = int(self.config.model.num_attr_classes)
        ref_attrs = [
            [int(G.nodes[n].get('attr', 0)) for n in sorted(G.nodes())]
            for G in ref_graphs
        ]
        gen_attrs_only = [s['attrs'] for s in gen_stats]

        gen_pair_counts = endpoint_attr_pair_counts(
            graphs_gen, gen_attrs_only, num_attr_classes)
        ref_pair_counts = endpoint_attr_pair_counts(
            ref_graphs, ref_attrs, num_attr_classes)
        pair_kl = kl_divergence(gen_pair_counts, ref_pair_counts)

        def _pair_hist(counts):
            total = sum(counts.values())
            return {
                '%d-%d' % pair: {
                    'count': int(cnt),
                    'ratio': cnt / total if total else 0.0,
                }
                for pair, cnt in sorted(counts.items())
            }

        def _count_hist(counter):
            total = sum(counter.values())
            return {
                str(k): {
                    'count': int(v),
                    'ratio': v / total if total else 0.0,
                }
                for k, v in sorted(counter.items())
            }

        gen_living = living_count_per_graph(
            graphs_gen, gen_attrs_only, target_class=0)
        ref_living = living_count_per_graph(
            ref_graphs, ref_attrs, target_class=0)

        gen_degree_by_class = per_class_degree_stats(
            graphs_gen, gen_attrs_only, num_attr_classes)
        ref_degree_by_class = per_class_degree_stats(
            ref_graphs, ref_attrs, num_attr_classes)

        attr_aware = {
            'endpoint_attr_pair_kl': float(pair_kl),
            'gen_endpoint_pair_histogram': _pair_hist(gen_pair_counts),
            'ref_endpoint_pair_histogram': _pair_hist(ref_pair_counts),
            'gen_living_count_histogram': _count_hist(gen_living),
            'ref_living_count_histogram': _count_hist(ref_living),
            'gen_per_class_degree_stats': {
                str(c): gen_degree_by_class[c]
                for c in range(num_attr_classes)
            },
            'ref_per_class_degree_stats': {
                str(c): ref_degree_by_class[c]
                for c in range(num_attr_classes)
            },
        }

        # --- assemble full report ----------------------------------------
        report = {
            'config_summary': config_summary,
            'num_generated': len(graphs_gen),
            'avg_gen_time_per_batch_sec': (
                float(np.mean(gen_run_time))
                if len(gen_run_time) > 0 else None),
            'mmd': {
                'dev': {k: (float(v) if v == v else None) for k, v in mmd_dev.items()},
                'test': {k: (float(v) if v == v else None) for k, v in mmd_test.items()},
            },
            'attr_aware_metrics': attr_aware,
            'gen_graph_size_histogram': _size_hist(gen_sizes),
            'ref_graph_size_histogram': _size_hist(ref_sizes),
            'gen_attr_histogram': attr_histogram,
            'ref_attr_histogram': ref_histogram,
        }

        # Per-graph stats written to separate file because it can be large
        with open(os.path.join(save_dir, 'test_results.json'),
                  'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # Flat CSV for MMD: row per (set, metric). Also includes
        # attr-aware scalars on the 'test' set so a single file captures
        # every directly-comparable number.
        with open(os.path.join(save_dir, 'test_mmd.csv'),
                  'w', encoding='utf-8', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['set', 'metric', 'value'])
            w.writeheader()
            for set_name, mmd_dict in (('dev', mmd_dev), ('test', mmd_test)):
                for metric, value in mmd_dict.items():
                    w.writerow({
                        'set': set_name,
                        'metric': metric,
                        'value': (float(value) if value == value else None),
                    })
            w.writerow({
                'set': 'test',
                'metric': 'endpoint_attr_pair_kl',
                'value': float(pair_kl),
            })
            w.writerow({
                'set': 'test',
                'metric': 'living_count_kl',
                'value': float(kl_divergence(gen_living, ref_living)),
            })

        # Per-graph attr sequence (one row per generated graph)
        with open(os.path.join(save_dir, 'test_attrs.csv'),
                  'w', encoding='utf-8', newline='') as f:
            w = csv.DictWriter(
                f, fieldnames=['idx', 'num_nodes', 'num_edges',
                               'attrs_comma_separated'])
            w.writeheader()
            for s in gen_stats:
                w.writerow({
                    'idx': s['idx'],
                    'num_nodes': s['num_nodes'],
                    'num_edges': s['num_edges'],
                    'attrs_comma_separated': (
                        ','.join(str(a) for a in (s['attrs'] or []))
                    ),
                })

        # [v2] save the actual generated graphs in two forms:
        #   1. gen_graphs.json       — human- / AI-readable: edge list + attrs
        #   2. gen_graphs.p          — pickle of networkx.Graph list (for
        #                              downstream consumers like GSDiff)
        gen_graphs_json = []
        for idx, G in enumerate(graphs_gen):
            edges = [[int(u), int(v)] for u, v in G.edges()]
            attrs = None
            if attr_lists is not None and idx < len(attr_lists):
                attrs = [int(a) for a in attr_lists[idx][:G.number_of_nodes()]]
            gen_graphs_json.append({
                'idx': idx,
                'num_nodes': int(G.number_of_nodes()),
                'num_edges': int(G.number_of_edges()),
                'edges': edges,
                'attrs': attrs,
            })

        with open(os.path.join(save_dir, 'gen_graphs.json'),
                  'w', encoding='utf-8') as f:
            json.dump({'graphs': gen_graphs_json}, f,
                      indent=2, ensure_ascii=False)

        # Also save as pickle for downstream use (e.g. feeding into GSDiff
        # or re-loading with networkx without parsing JSON).
        with open(os.path.join(save_dir, 'gen_graphs.p'), 'wb') as f:
            pickle.dump(graphs_gen, f)

        logger.info(
            'saved structured test results: test_results.json + test_mmd.csv + '
            'test_attrs.csv + gen_graphs.json + gen_graphs.p')

    # ======================================================================
    # Validation helper
    # ======================================================================
    def _validate(self, model, val_loader):
        """Run forward-only loss computation on the dev set.

        Args:
            model:      the (possibly DataParallel-wrapped) GRANv2 model
            val_loader: DataLoader over the dev-tagged GRANDataV2

        Returns:
            (val_edge, val_attr, val_total) mean losses as plain Python floats.
            val_total = val_edge + lambda_attr * val_attr.
        """
        model.eval()
        sum_edge = 0.0
        sum_attr = 0.0
        n_batches = 0

        with torch.no_grad():
            val_iter = val_loader.__iter__()
            n_inner = len(val_loader) // self.num_gpus
            for _ in range(n_inner):
                batch_data = []
                if self.use_gpu:
                    for _ in self.gpus:
                        batch_data.append(next(val_iter))
                for ff in range(self.dataset_conf.num_fwd_pass):
                    batch_fwd = []
                    if self.use_gpu:
                        for dd, gpu_id in enumerate(self.gpus):
                            data = {}
                            data['adj'] = batch_data[dd][ff]['adj'].to(gpu_id, non_blocking=True)
                            data['edges'] = batch_data[dd][ff]['edges'].to(gpu_id, non_blocking=True)
                            data['node_idx_gnn'] = batch_data[dd][ff]['node_idx_gnn'].to(gpu_id, non_blocking=True)
                            data['node_idx_feat'] = batch_data[dd][ff]['node_idx_feat'].to(gpu_id, non_blocking=True)
                            data['label'] = batch_data[dd][ff]['label'].to(gpu_id, non_blocking=True)
                            data['att_idx'] = batch_data[dd][ff]['att_idx'].to(gpu_id, non_blocking=True)
                            data['subgraph_idx'] = batch_data[dd][ff]['subgraph_idx'].to(gpu_id, non_blocking=True)
                            data['subgraph_idx_base'] = batch_data[dd][ff]['subgraph_idx_base'].to(gpu_id, non_blocking=True)
                            if 'node_attrs' in batch_data[dd][ff]:
                                data['node_attrs'] = batch_data[dd][ff]['node_attrs'].to(gpu_id, non_blocking=True)
                            # [planc] thread per-state-row attr labels for the
                            # attr-conditioned edge head.
                            if 'subgraph_node_attrs' in batch_data[dd][ff]:
                                data['subgraph_node_attrs'] = batch_data[dd][ff]['subgraph_node_attrs'].to(gpu_id, non_blocking=True)
                            batch_fwd.append((data,))

                    if batch_fwd:
                        result = model(*batch_fwd)
                        if isinstance(result, tuple) and len(result) == 2:
                            edge_loss = result[0].mean()
                            attr_loss = result[1].mean()
                        else:
                            edge_loss = result.mean()
                            attr_loss = edge_loss.new_zeros(())
                        sum_edge += float(edge_loss.data.cpu().numpy())
                        sum_attr += float(attr_loss.data.cpu().numpy())
                        n_batches += 1

        model.train()
        if n_batches == 0:
            return 0.0, 0.0, 0.0
        mean_edge = sum_edge / n_batches
        mean_attr = sum_attr / n_batches
        mean_total = mean_edge + self.lambda_attr * mean_attr
        return mean_edge, mean_attr, mean_total

    # ======================================================================
    # Visualization helper
    # ======================================================================
    # Match the GRAN room class palette used in the preprocessor so
    # training visualizations and generated visualizations are directly
    # comparable.
    _ATTR_NAMES = ('Living', 'Bedroom', 'Bathroom', 'Kitchen',
                   'Balcony', 'Storage', 'External')
    _ATTR_COLORS = {
        0: '#EE4D4D',  # Living
        1: '#C67FFF',  # Bedroom
        2: '#5EBADA',  # Bathroom
        3: '#FFB84D',  # Kitchen
        4: '#6BDF6B',  # Balcony
        5: '#B5896B',  # Storage
        6: '#808080',  # External
    }

    def _draw_grid(self, graphs, out_path, title='', ncols=5):
        """Render a grid of graphs in one PNG.

        Each cell uses _ATTR_COLORS to colour nodes by their 'attr' class.
        There is NO pairing with any other grid — each graph is independent.

        Args:
            graphs:   iterable of networkx.Graph
            out_path: where to save the PNG
            title:    figure-level title
            ncols:    number of columns in the grid (rows inferred)
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch

        graphs = list(graphs)
        n = len(graphs)
        if n == 0:
            return
        nrows = int(np.ceil(n / ncols))

        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(ncols * 2.6, nrows * 2.6))
        # Always make axes iterable as flat list
        if nrows == 1 and ncols == 1:
            axes = [axes]
        else:
            axes = np.asarray(axes).reshape(-1)

        present = set()

        for idx, ax in enumerate(axes):
            if idx >= n:
                ax.axis('off')
                continue

            G = graphs[idx]
            if G.number_of_nodes() == 0:
                ax.text(0.5, 0.5, '(empty)', ha='center', va='center')
                ax.set_title(f'#{idx}', fontsize=8)
                ax.set_xticks([]); ax.set_yticks([])
                continue

            pos = nx.spring_layout(G, seed=42, k=0.8)

            for u, v in G.edges():
                x1, y1 = pos[u]; x2, y2 = pos[v]
                ax.plot([x1, x2], [y1, y2], color='#888',
                        linewidth=1.1, zorder=1)
            for nd in G.nodes():
                cls = int(G.nodes[nd].get('attr', 0))
                present.add(cls)
                color = self._ATTR_COLORS.get(cls, '#888')
                x, y = pos[nd]
                ax.add_patch(plt.Circle((x, y), 0.14, facecolor=color,
                                        edgecolor='black', linewidth=0.8,
                                        zorder=2, alpha=0.95))
                # keep per-node label only when graph is small enough
                if G.number_of_nodes() <= 10:
                    label = self._ATTR_NAMES[cls][:3] if 0 <= cls < len(self._ATTR_NAMES) else '?'
                    ax.text(x, y, label, ha='center', va='center',
                            fontsize=6, fontweight='bold', zorder=3)

            ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2)
            ax.set_aspect('equal')
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f'#{idx} ({G.number_of_nodes()}n {G.number_of_edges()}e)',
                         fontsize=8)

        handles = [Patch(facecolor=self._ATTR_COLORS.get(c, '#888'),
                         label=self._ATTR_NAMES[c] if c < len(self._ATTR_NAMES) else f'cls{c}')
                   for c in sorted(present)]
        if handles:
            fig.legend(handles=handles, loc='lower center',
                       ncol=len(handles), fontsize=9,
                       bbox_to_anchor=(0.5, -0.02))

        fig.suptitle(title, fontsize=12, y=0.995)
        plt.tight_layout(rect=(0, 0.03, 1, 0.98))
        plt.savefig(out_path, dpi=110, bbox_inches='tight')
        plt.close(fig)

    def _draw_attr_bubble(self, G, out_path, title=''):
        """Render a single graph as a bubble diagram colored by node 'attr'.

        Nodes without an 'attr' entry default to class 0 (gray).
        Edges are plotted as plain lines.

        Args:
            G:          networkx.Graph; nodes may have 'attr' (int) attribute
            out_path:   where to save the PNG
            title:      figure title
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch

        if G.number_of_nodes() == 0:
            return

        # spring layout for a clean "bubble" look -- reproducible via nx seed
        pos = nx.spring_layout(G, seed=42, k=0.8)

        fig, ax = plt.subplots(figsize=(5, 5))

        # edges
        for u, v in G.edges():
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            ax.plot([x1, x2], [y1, y2],
                    color='#888', linewidth=1.5, zorder=1)

        # nodes as colored circles with the class name inside
        for n in G.nodes():
            cls = int(G.nodes[n].get('attr', 0))
            color = self._ATTR_COLORS.get(cls, '#888')
            label = self._ATTR_NAMES[cls][:3] if 0 <= cls < len(self._ATTR_NAMES) else '?'
            x, y = pos[n]
            circ = plt.Circle((x, y), 0.12,
                              facecolor=color, edgecolor='black',
                              linewidth=1.2, zorder=2, alpha=0.95)
            ax.add_patch(circ)
            ax.text(x, y, label,
                    ha='center', va='center',
                    fontsize=8, fontweight='bold', zorder=3)

        # legend (only the classes actually present in this graph)
        present = sorted({int(G.nodes[n].get('attr', 0)) for n in G.nodes()})
        handles = [Patch(facecolor=self._ATTR_COLORS.get(c, '#888'),
                         label=self._ATTR_NAMES[c] if c < len(self._ATTR_NAMES) else f'cls{c}')
                   for c in present]
        ax.legend(handles=handles, loc='upper right', fontsize=7, framealpha=0.9)

        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'{title} | {G.number_of_nodes()} nodes, {G.number_of_edges()} edges',
                     fontsize=10)

        plt.tight_layout()
        plt.savefig(out_path, dpi=110, bbox_inches='tight')
        plt.close(fig)

    # ======================================================================
    # Testing
    # ======================================================================
    def test(self):
        """Generate graphs + attributes with GRANv2 and run MMD evaluation.

        Mirrors :meth:`GranRunner.test` but unpacks the
        ``(A_list, attr_list)`` tuple returned by
        :meth:`GRANv2.forward` in sampling mode.

        Returns:
            tuple of MMD scores (same layout as parent), with an extra
            ``attr_lists`` element at the end when attribute sampling
            succeeds. For the lobster case we still keep the lobster
            accuracy as the final element for backward compatibility.
        """
        self.config.save_dir = self.test_conf.test_model_dir

        # ---- Erdos-Renyi baseline branch (no attrs to unpack) --------------
        if self.config.test.is_test_ER:
            p_ER = sum([aa.number_of_edges() for aa in self.graphs_train]) / \
                sum([aa.number_of_nodes() ** 2 for aa in self.graphs_train])
            graphs_gen = [
                nx.fast_gnp_random_graph(self.max_num_nodes, p_ER, seed=ii)
                for ii in range(self.num_test_gen)]
            attr_lists = None
        else:
            # ---- load model ---------------------------------------------
            model = eval(self.model_conf.name)(self.config)
            model_file = os.path.join(self.config.save_dir,
                                      self.test_conf.test_model_name)
            load_model(model, model_file, self.device)

            if self.use_gpu:
                model = nn.DataParallel(model, device_ids=self.gpus).to(self.device)

            model.eval()

            # ---- generate graphs (+ attrs) -----------------------------
            A_pred = []
            attr_pred = []
            num_nodes_pred = []
            num_test_batch = int(np.ceil(self.num_test_gen / self.test_conf.batch_size))

            # Progress is reported in graphs (not batches) so the user can see
            # real throughput when num_test_gen is large.
            gen_bar = tqdm(total=self.num_test_gen, desc='generating',
                           unit='graph')
            gen_run_time = []
            for ii in range(num_test_batch):
                with torch.no_grad():
                    start_time = time.time()
                    input_dict = {}
                    input_dict['is_sampling'] = True
                    input_dict['batch_size'] = self.test_conf.batch_size
                    input_dict['num_nodes_pmf'] = self.num_nodes_pmf_train
                    result = model(input_dict)
                    # [v2] GRANv2 sampling returns (A_list, attr_list);
                    # GRANMixtureBernoulli returns just A_list.
                    if isinstance(result, tuple) and len(result) == 2:
                        A_tmp, attr_tmp = result
                    else:
                        A_tmp, attr_tmp = result, None
                    gen_run_time += [time.time() - start_time]
                    A_pred += [aa.data.cpu().numpy() for aa in A_tmp]
                    num_nodes_pred += [aa.shape[0] for aa in A_tmp]
                    if attr_tmp is not None:
                        attr_pred += [aa.data.cpu().numpy() for aa in attr_tmp]
                    gen_bar.update(len(A_tmp))
            gen_bar.close()

            logger.info('Average test time per mini-batch = {}'.format(
                np.mean(gen_run_time)))

            graphs_gen = [get_graph(aa) for aa in A_pred]
            # [v2] keep attr_lists as np.ndarray list aligned with graphs_gen
            attr_lists = attr_pred if len(attr_pred) > 0 else None

        # ---- visualization (v2: two grids, gen and reference) -------------
        # No one-to-one correspondence between gen and reference in
        # unconditional generation, so we render two separate overview grids
        # (not paired). Reference comes from the TEST set (unseen by model).
        if self.is_vis:
            vis_dir = os.path.join(self.config.save_dir, 'vis')
            os.makedirs(vis_dir, exist_ok=True)

            # Attach generated attr labels onto the networkx graphs for plotting
            for ii, gg in enumerate(graphs_gen[:self.num_vis]):
                n = gg.number_of_nodes()
                if attr_lists is not None and ii < len(attr_lists):
                    attrs_i = attr_lists[ii][:n]
                    for nd in gg.nodes():
                        if nd < len(attrs_i):
                            gg.nodes[nd]['attr'] = int(attrs_i[nd])

            ref_graphs = (self.graphs_test
                          if len(self.graphs_test) > 0
                          else self.graphs_train)
            n_vis = min(self.num_vis, len(graphs_gen), len(ref_graphs))

            gen_grid_path = os.path.join(vis_dir, 'gen_grid.png')
            ref_grid_path = os.path.join(vis_dir, 'ref_grid.png')

            try:
                self._draw_grid(
                    graphs_gen[:n_vis], gen_grid_path,
                    title='Generated samples (from model)',
                    ncols=self.vis_num_row)
                self._draw_grid(
                    ref_graphs[:n_vis], ref_grid_path,
                    title='Real samples (from held-out test set)',
                    ncols=self.vis_num_row)
                logger.info(
                    f'saved gen_grid.png + ref_grid.png -> {vis_dir}')
            except Exception as e:
                logger.warning(f'visualization failed: {e}')

        # ---- evaluation ---------------------------------------------------
        # Each MMD metric is wrapped individually so one bad metric (e.g.
        # orbit_stats_all with no orca extension) does not kill the rest.
        # The error is logged with full traceback via logger.exception.
        if self.config.dataset.name in ['lobster']:
            acc = eval_acc_lobster_graph(graphs_gen)
            logger.info('Validity accuracy of generated graphs = {}'.format(acc))

        num_nodes_gen = [len(aa) for aa in graphs_gen]

        def _safe_mmd(name, fn):
            """Run an MMD computation; log before + after so users see
            progress for the (slow) pair-wise MMD stages."""
            import time as _time
            logger.info(f'  [MMD] computing "{name}"...')
            t0 = _time.time()
            try:
                val = fn()
                logger.info(f'  [MMD] "{name}" = {val:.4f} '
                            f'(took {_time.time() - t0:.1f}s)')
                return val
            except Exception:
                logger.exception(f'  [MMD] "{name}" failed:')
                return float('nan')

        def _safe_evaluate(tag, ref_graphs, gen_graphs):
            """Safe variant of evaluate(); returns (deg, cluster, orbit, spec),
            each value NaN if the underlying computation raised."""
            from utils.eval_helper import (degree_stats, clustering_stats,
                                           orbit_stats_all, spectral_stats)
            logger.info(f'[MMD] stage: {tag} '
                        f'(ref {len(ref_graphs)} vs gen {len(gen_graphs)})')
            deg = _safe_mmd(f'{tag}/degree',
                            lambda: degree_stats(ref_graphs, gen_graphs))
            clust = _safe_mmd(f'{tag}/clustering',
                              lambda: clustering_stats(ref_graphs, gen_graphs))
            orb = _safe_mmd(f'{tag}/4-orbits (needs orca)',
                            lambda: orbit_stats_all(ref_graphs, gen_graphs))
            spec = _safe_mmd(f'{tag}/spectral',
                             lambda: spectral_stats(ref_graphs, gen_graphs))
            return deg, clust, orb, spec

        # Skip DEV MMD entirely when dev set is empty (dev_ratio=0) — users
        # often set dev_ratio=0 and care only about the test-set report.
        if len(self.graphs_dev) > 0:
            num_nodes_dev = [len(gg.nodes) for gg in self.graphs_dev]
            mmd_degree_dev, mmd_clustering_dev, mmd_4orbits_dev, mmd_spectral_dev = \
                _safe_evaluate('DEV', self.graphs_dev, graphs_gen)
            mmd_num_nodes_dev = _safe_mmd(
                'DEV/#nodes',
                lambda: compute_mmd(
                    [np.bincount(num_nodes_dev)], [np.bincount(num_nodes_gen)],
                    kernel=gaussian_emd))
        else:
            logger.info('DEV set is empty; skipping DEV MMD')
            mmd_degree_dev = mmd_clustering_dev = float('nan')
            mmd_4orbits_dev = mmd_spectral_dev = mmd_num_nodes_dev = float('nan')

        num_nodes_test = [len(gg.nodes) for gg in self.graphs_test]
        mmd_degree_test, mmd_clustering_test, mmd_4orbits_test, mmd_spectral_test = \
            _safe_evaluate('TEST', self.graphs_test, graphs_gen)
        mmd_num_nodes_test = _safe_mmd(
            'TEST/#nodes',
            lambda: compute_mmd(
                [np.bincount(num_nodes_test)], [np.bincount(num_nodes_gen)],
                kernel=gaussian_emd))

        logger.info(
            "MMD vs DEV  | #nodes={:.4f} | degree={:.4f} | "
            "clustering={:.4f} | 4orbits={:.4f} | spectral={:.4f}".format(
                mmd_num_nodes_dev, mmd_degree_dev, mmd_clustering_dev,
                mmd_4orbits_dev, mmd_spectral_dev))
        logger.info(
            "MMD vs TEST | #nodes={:.4f} | degree={:.4f} | "
            "clustering={:.4f} | 4orbits={:.4f} | spectral={:.4f}".format(
                mmd_num_nodes_test, mmd_degree_test, mmd_clustering_test,
                mmd_4orbits_test, mmd_spectral_test))

        # [v2] Structured test report for machine consumption. Written to
        # the SAME run dir where the loaded checkpoint lives, so every
        # checkpoint has its matching test_results.json right next to it.
        self._dump_test_results(
            graphs_gen=graphs_gen, attr_lists=attr_lists,
            gen_run_time=gen_run_time,
            mmd_dev={
                '#nodes': mmd_num_nodes_dev,
                'degree': mmd_degree_dev,
                'clustering': mmd_clustering_dev,
                '4orbits': mmd_4orbits_dev,
                'spectral': mmd_spectral_dev,
            },
            mmd_test={
                '#nodes': mmd_num_nodes_test,
                'degree': mmd_degree_test,
                'clustering': mmd_clustering_test,
                '4orbits': mmd_4orbits_test,
                'spectral': mmd_spectral_test,
            },
        )

        if self.config.dataset.name in ['lobster']:
            return (mmd_degree_dev, mmd_clustering_dev, mmd_4orbits_dev,
                    mmd_spectral_dev, mmd_degree_test, mmd_clustering_test,
                    mmd_4orbits_test, mmd_spectral_test, acc)
        else:
            return (mmd_degree_dev, mmd_clustering_dev, mmd_4orbits_dev,
                    mmd_spectral_dev, mmd_degree_test, mmd_clustering_test,
                    mmd_4orbits_test, mmd_spectral_test)
