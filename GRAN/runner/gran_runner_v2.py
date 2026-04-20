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
        """
        super().__init__(config)
        # lambda_attr: weight of the node-attribute CrossEntropy loss in the
        # joint objective. Configured via config.train.lambda_attr;
        # default 1.0 so it matches the edge loss by default.
        self.lambda_attr = getattr(config.train, 'lambda_attr', 1.0)

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

        # [v2] pick up GRANv2 via config.model.name
        model = eval(self.model_conf.name)(self.config)

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

                # [v2] live-update tqdm postfix (no extra log lines)
                epoch_bar.set_postfix(
                    edge=f'{edge_loss_val:.4f}',
                    attr=f'{attr_loss_val:.4f}',
                    total=f'{total_loss_val:.4f}',
                )

            # [v2] one summary line per epoch, ML-style
            if epoch_iters > 0:
                mean_edge = epoch_edge_sum / epoch_iters
                mean_attr = epoch_attr_sum / epoch_iters
                mean_total = mean_edge + self.lambda_attr * mean_attr
                self.writer.add_scalar('epoch/edge_loss', mean_edge, epoch + 1)
                self.writer.add_scalar('epoch/attr_loss', mean_attr, epoch + 1)
                self.writer.add_scalar('epoch/total_loss', mean_total, epoch + 1)
                logger.info(
                    "epoch {:04d}/{:04d} | iters {} | "
                    "edge {:.4f} | attr {:.4f} | total {:.4f} | "
                    "lr {:.2e}".format(
                        epoch + 1, self.train_conf.max_epoch, epoch_iters,
                        mean_edge, mean_attr, mean_total,
                        optimizer.param_groups[0]['lr']))

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

        return 1

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

            gen_run_time = []
            for ii in tqdm(range(num_test_batch)):
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

            logger.info('Average test time per mini-batch = {}'.format(
                np.mean(gen_run_time)))

            graphs_gen = [get_graph(aa) for aa in A_pred]
            # [v2] keep attr_lists as np.ndarray list aligned with graphs_gen
            attr_lists = attr_pred if len(attr_pred) > 0 else None

        # ---- visualization (unchanged from parent) -------------------------
        if self.is_vis:
            num_col = self.vis_num_row
            num_row = int(np.ceil(self.num_vis / num_col))
            test_epoch = self.test_conf.test_model_name
            test_epoch = test_epoch[test_epoch.rfind('_') + 1:test_epoch.find('.pth')]
            save_name = os.path.join(
                self.config.save_dir,
                '{}_gen_graphs_epoch_{}_block_{}_stride_{}.png'.format(
                    self.config.test.test_model_name[:-4], test_epoch,
                    self.block_size, self.stride))

            graphs_pred_vis = [copy.deepcopy(gg) for gg in graphs_gen[:self.num_vis]]

            if self.better_vis:
                for gg in graphs_pred_vis:
                    gg.remove_nodes_from(list(nx.isolates(gg)))

            vis_graphs = []
            for gg in graphs_pred_vis:
                CGs = [gg.subgraph(c) for c in nx.connected_components(gg)]
                CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)
                vis_graphs += [CGs[0]]

            if self.is_single_plot:
                draw_graph_list(vis_graphs, num_row, num_col,
                                fname=save_name, layout='spring')
            else:
                draw_graph_list_separate(vis_graphs, fname=save_name[:-4],
                                         is_single=True, layout='spring')

            save_name = os.path.join(self.config.save_dir, 'train_graphs.png')

            if self.is_single_plot:
                draw_graph_list(self.graphs_train[:self.num_vis], num_row, num_col,
                                fname=save_name, layout='spring')
            else:
                draw_graph_list_separate(self.graphs_train[:self.num_vis],
                                         fname=save_name[:-4], is_single=True,
                                         layout='spring')

        # ---- evaluation (unchanged from parent) ----------------------------
        if self.config.dataset.name in ['lobster']:
            acc = eval_acc_lobster_graph(graphs_gen)
            logger.info('Validity accuracy of generated graphs = {}'.format(acc))

        num_nodes_gen = [len(aa) for aa in graphs_gen]

        num_nodes_dev = [len(gg.nodes) for gg in self.graphs_dev]
        mmd_degree_dev, mmd_clustering_dev, mmd_4orbits_dev, mmd_spectral_dev = evaluate(
            self.graphs_dev, graphs_gen, degree_only=False)
        mmd_num_nodes_dev = compute_mmd(
            [np.bincount(num_nodes_dev)], [np.bincount(num_nodes_gen)],
            kernel=gaussian_emd)

        num_nodes_test = [len(gg.nodes) for gg in self.graphs_test]
        mmd_degree_test, mmd_clustering_test, mmd_4orbits_test, mmd_spectral_test = evaluate(
            self.graphs_test, graphs_gen, degree_only=False)
        mmd_num_nodes_test = compute_mmd(
            [np.bincount(num_nodes_test)], [np.bincount(num_nodes_gen)],
            kernel=gaussian_emd)

        logger.info(
            "Validation MMD scores of #nodes/degree/clustering/4orbits/spectral are = {}/{}/{}/{}/{}".format(
                mmd_num_nodes_dev, mmd_degree_dev, mmd_clustering_dev,
                mmd_4orbits_dev, mmd_spectral_dev))
        logger.info(
            "Test MMD scores of #nodes/degree/clustering/4orbits/spectral are = {}/{}/{}/{}/{}".format(
                mmd_num_nodes_test, mmd_degree_test, mmd_clustering_test,
                mmd_4orbits_test, mmd_spectral_test))

        if self.config.dataset.name in ['lobster']:
            return (mmd_degree_dev, mmd_clustering_dev, mmd_4orbits_dev,
                    mmd_spectral_dev, mmd_degree_test, mmd_clustering_test,
                    mmd_4orbits_test, mmd_spectral_test, acc)
        else:
            return (mmd_degree_dev, mmd_clustering_dev, mmd_4orbits_dev,
                    mmd_spectral_dev, mmd_degree_test, mmd_clustering_test,
                    mmd_4orbits_test, mmd_spectral_test)
