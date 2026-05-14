import math
import os
import time

import torch
import torch.nn.functional as F

from src import utils
from diffusion import diffusion_utils
from diffusion.absorbing_utils import pad_clean_features_for_absorbing, strip_mask_class_from_sample
from diffusion_model_discrete import DiscreteDenoisingDiffusion


class AbsorbingDenoisingDiffusion(DiscreteDenoisingDiffusion):
    """DiGress backbone trained as an absorbing-state masked graph model.

    This class intentionally lives beside the original `DiscreteDenoisingDiffusion`
    instead of replacing it. Existing DiGress configs keep using the marginal
    transition. Only configs with `model.transition=absorbing` use this class.

    Mental model for ML beginners:
      - Clean graph: every node and edge has a real class.
      - Noisy graph: some node/edge classes are replaced by the internal [MASK].
      - Training target: predict the original class only at those [MASK] slots.
      - Sampling: start from all [MASK] and progressively fill in confident slots.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_idx_X = self.Xdim_output - 1
        self.mask_idx_E = self.Edim_output - 1
        self.base_Xdim_output = self.mask_idx_X
        self.base_Edim_output = self.mask_idx_E
        self._reset_absorbing_epoch_stats("train")
        self._reset_absorbing_epoch_stats("val")
        self._reset_absorbing_epoch_stats("test")

    def _reset_absorbing_epoch_stats(self, split: str) -> None:
        setattr(self, f"_{split}_absorbing_stats", {
            "loss": 0.0,
            "x_ce": 0.0,
            "e_ce": 0.0,
            "x_masked": 0.0,
            "e_masked": 0.0,
            "batches": 0.0,
        })

    def _accumulate_absorbing_stats(self, split: str, stats: dict[str, torch.Tensor | float]) -> None:
        bucket = getattr(self, f"_{split}_absorbing_stats")
        for key in ["loss", "x_ce", "e_ce", "x_masked", "e_masked"]:
            value = stats[key]
            bucket[key] += float(value.detach().cpu()) if isinstance(value, torch.Tensor) else float(value)
        bucket["batches"] += 1.0

    @staticmethod
    def _average_absorbing_stats(bucket: dict[str, float]) -> dict[str, float]:
        denom = max(1.0, bucket["batches"])
        return {
            key: bucket[key] / denom
            for key in ["loss", "x_ce", "e_ce", "x_masked", "e_masked"]
        }

    def _clean_to_absorbing_space(self, X: torch.Tensor, E: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if X.size(-1) == self.Xdim_output and E.size(-1) == self.Edim_output:
            return X, E
        return pad_clean_features_for_absorbing(X, E)

    def apply_noise(self, X, E, y, node_mask):
        """Mask clean graph positions according to the absorbing schedule.

        We sample t in [1, T]. At small t only a few positions are masked; at
        large t almost everything is masked. This gives one model experience
        with unconditional generation, partial completion, and node-type
        prediction style inputs.
        """
        t_int = torch.randint(1, self.T + 1, size=(X.size(0), 1), device=X.device).float()
        t_float = t_int / self.T
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)
        beta_t = self.noise_schedule(t_normalized=t_float)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=(t_int - 1) / self.T)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, device=self.device)
        probX = X @ Qtb.X
        probE = E @ Qtb.E.unsqueeze(1)
        sampled_t = diffusion_utils.sample_discrete_features(probX=probX, probE=probE, node_mask=node_mask)

        X_t = F.one_hot(sampled_t.X, num_classes=self.Xdim_output).float()
        E_t = F.one_hot(sampled_t.E, num_classes=self.Edim_output).float()
        z_t = utils.PlaceHolder(X=X_t, E=E_t, y=y).type_as(X_t).mask(node_mask)

        return {
            "t_int": t_int,
            "t": t_float,
            "beta_t": beta_t,
            "alpha_s_bar": alpha_s_bar,
            "alpha_t_bar": alpha_t_bar,
            "X_t": z_t.X,
            "E_t": z_t.E,
            "y_t": z_t.y,
            "node_mask": node_mask,
            # ExtraFeatures uses this to avoid treating [MASK] as a real edge
            # when computing cycles/spectral features.
            "mask_idx_E": self.mask_idx_E,
        }

    def _masked_ce_loss(self, pred, noisy_data, true_X, true_E, node_mask):
        """Cross entropy only where the noisy graph currently contains [MASK]."""
        X_t_idx = noisy_data["X_t"].argmax(dim=-1)
        E_t_idx = noisy_data["E_t"].argmax(dim=-1)
        true_X_idx = true_X[..., :self.base_Xdim_output].argmax(dim=-1)
        true_E_idx = true_E[..., :self.base_Edim_output].argmax(dim=-1)

        x_positions = (X_t_idx == self.mask_idx_X) & node_mask
        n = node_mask.size(1)
        diagonal = torch.eye(n, device=node_mask.device, dtype=torch.bool).unsqueeze(0)
        edge_slots = node_mask.unsqueeze(1) & node_mask.unsqueeze(2) & ~diagonal
        e_positions = (E_t_idx == self.mask_idx_E) & edge_slots

        # The network has an output logit for [MASK] because the residual
        # GraphTransformer expects equal input/output dimensions. We exclude
        # that last logit from the CE target so the model learns real classes.
        pred_X = pred.X[..., :self.base_Xdim_output]
        pred_E = pred.E[..., :self.base_Edim_output]

        zero = pred.X.sum() * 0.0
        loss_X = (
            F.cross_entropy(pred_X[x_positions], true_X_idx[x_positions])
            if x_positions.any()
            else zero
        )
        loss_E = (
            F.cross_entropy(pred_E[e_positions], true_E_idx[e_positions])
            if e_positions.any()
            else zero
        )
        total = loss_X + float(self.cfg.model.lambda_train[0]) * loss_E
        return total, {
            "loss": total.detach(),
            "x_ce": loss_X.detach(),
            "e_ce": loss_E.detach(),
            "x_masked": x_positions.float().sum().detach(),
            "e_masked": e_positions.float().sum().detach(),
        }

    def _step(self, data, split: str):
        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask)
        X, E = self._clean_to_absorbing_space(dense_data.X, dense_data.E)
        noisy_data = self.apply_noise(X, E, data.y, node_mask)
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)
        loss, stats = self._masked_ce_loss(pred, noisy_data, X, E, node_mask)
        self._accumulate_absorbing_stats(split, stats)
        self.log(f"{split}/absorbing_masked_loss", loss, sync_dist=True)
        return {"loss": loss}

    def training_step(self, data, i):
        return self._step(data, "train")

    def validation_step(self, data, i):
        return self._step(data, "val")

    def test_step(self, data, i):
        return self._step(data, "test")

    def on_train_epoch_start(self) -> None:
        self.print(f"Starting train epoch {self.current_epoch + 1}/{self.cfg.train.n_epochs}...")
        self.start_epoch_time = time.time()
        self._reset_absorbing_epoch_stats("train")

    def on_train_epoch_end(self) -> None:
        display_epoch = self.current_epoch + 1
        stats = self._average_absorbing_stats(self._train_absorbing_stats)
        self.log("train/epoch_weighted_loss", stats["loss"], sync_dist=True)
        self.print(
            f"Epoch {display_epoch}: train_masked_loss={stats['loss']:.3f} "
            f"(masked_node_CE={stats['x_ce']:.3f}, "
            f"masked_edge_CE={stats['e_ce']:.3f}, "
            f"avg_masked_nodes={stats['x_masked']:.1f}, "
            f"avg_masked_edges={stats['e_masked']:.1f}) "
            f"-- {time.time() - self.start_epoch_time:.1f}s"
        )
        self._append_history({
            "epoch": display_epoch,
            "global_step": self.global_step,
            "split": "train",
            "train_weighted": stats["loss"],
            "train_x_ce": stats["x_ce"],
            "train_e_ce": stats["e_ce"],
            "train_y_ce": "",
            "train_seconds": time.time() - self.start_epoch_time,
        })

    def on_validation_epoch_start(self) -> None:
        self._reset_absorbing_epoch_stats("val")
        if self.sampling_metrics is not None:
            self.sampling_metrics.reset()

    def on_validation_epoch_end(self) -> None:
        display_epoch = self.current_epoch + 1
        stats = self._average_absorbing_stats(self._val_absorbing_stats)
        val_loss = stats["loss"]

        # Keep the legacy monitor name so ModelCheckpoint still saves "best".
        # For absorbing runs this value is masked validation CE, not VLB NLL.
        self.log("val/absorbing_masked_loss", val_loss, sync_dist=True)
        self.log("val/epoch_NLL", val_loss, sync_dist=True)
        self.log("val_epoch_NLL", val_loss, sync_dist=True)
        self.log("display_epoch", float(display_epoch), sync_dist=True)

        if val_loss < self.best_val_nll:
            self.best_val_nll = val_loss
        self.print(
            f"Epoch {display_epoch}: val_masked_loss={val_loss:.4f} "
            f"(masked_node_CE={stats['x_ce']:.3f}, "
            f"masked_edge_CE={stats['e_ce']:.3f}, "
            f"best_val_masked_loss={self.best_val_nll:.4f})\n"
        )
        self._append_history({
            "epoch": display_epoch,
            "global_step": self.global_step,
            "split": "val",
            "val_nll": val_loss,
            "val_x_kl": stats["x_ce"],
            "val_e_kl": stats["e_ce"],
            "best_val_nll": self.best_val_nll,
        })

        self.val_counter += 1
        if self.sampling_metrics is None or self.val_counter % self.cfg.general.sample_every_val != 0:
            return

        start = time.time()
        samples_left = self.cfg.general.samples_to_generate
        samples = []
        ident = 0
        while samples_left > 0:
            batch_size = min(samples_left, 2 * self.cfg.train.batch_size)
            samples.extend(self.sample_batch(
                batch_id=ident,
                batch_size=batch_size,
                keep_chain=0,
                number_chain_steps=0,
                save_final=min(self.cfg.general.samples_to_save, batch_size),
                num_nodes=None,
            ))
            ident += batch_size
            samples_left -= batch_size
        self.print("Computing sampling metrics...")
        self.sampling_metrics.forward(samples, self.name, self.current_epoch, val_counter=-1, test=False,
                                      local_rank=self.local_rank)
        self.print(f"Done. Sampling took {time.time() - start:.2f} seconds\n")

    def on_test_epoch_start(self) -> None:
        self.print("Starting absorbing test...")
        self._reset_absorbing_epoch_stats("test")

    def on_test_epoch_end(self) -> None:
        stats = self._average_absorbing_stats(self._test_absorbing_stats)
        self.log("test/absorbing_masked_loss", stats["loss"], sync_dist=True)
        self.print(
            f"Test absorbing masked loss={stats['loss']:.4f} "
            f"(node_CE={stats['x_ce']:.3f}, edge_CE={stats['e_ce']:.3f})"
        )

    def _num_to_unmask(self, remaining: int, step: int, total_steps: int) -> int:
        """Return how many masked positions to fill on this MaskGIT-style step."""
        if remaining <= 0:
            return 0
        steps_left = max(1, total_steps - step)
        if steps_left == 1:
            return remaining
        return max(1, math.ceil(remaining / steps_left))

    def _maskgit_unmask_step(self, X, E, pred, node_mask, step: int, total_steps: int):
        X_idx = X.argmax(dim=-1)
        E_idx = E.argmax(dim=-1)
        bs, n = X_idx.shape

        prob_X = F.softmax(pred.X[..., :self.base_Xdim_output], dim=-1)
        edge_logits = pred.E[..., :self.base_Edim_output].clone()
        none_bias = float(self.cfg.model.get("edge_none_logit_bias", 0.0))
        if none_bias != 0.0:
            # V1 generated graphs that were too dense. A positive bias makes
            # the sampler more conservative by increasing the logit of edge
            # class 0 ("none") without changing the trained checkpoint.
            edge_logits[..., 0] = edge_logits[..., 0] + none_bias
        prob_E = F.softmax(edge_logits, dim=-1)
        sample_X = prob_X.reshape(-1, self.base_Xdim_output).multinomial(1).reshape(bs, n)
        sample_E = prob_E.reshape(-1, self.base_Edim_output).multinomial(1).reshape(bs, n, n)
        conf_X = prob_X.max(dim=-1).values
        conf_E = prob_E.max(dim=-1).values

        for b in range(bs):
            node_candidates = torch.nonzero((X_idx[b] == self.mask_idx_X) & node_mask[b], as_tuple=False).flatten()
            k_nodes = self._num_to_unmask(int(node_candidates.numel()), step, total_steps)
            if k_nodes > 0:
                selected = node_candidates[torch.topk(conf_X[b, node_candidates], k=k_nodes).indices]
                X_idx[b, selected] = sample_X[b, selected]

            valid_edges = node_mask[b].unsqueeze(0) & node_mask[b].unsqueeze(1)
            edge_candidates = torch.nonzero(
                torch.triu((E_idx[b] == self.mask_idx_E) & valid_edges, diagonal=1),
                as_tuple=False,
            )
            k_edges = self._num_to_unmask(int(edge_candidates.size(0)), step, total_steps)
            if k_edges > 0:
                candidate_scores = conf_E[b, edge_candidates[:, 0], edge_candidates[:, 1]]
                selected = edge_candidates[torch.topk(candidate_scores, k=k_edges).indices]
                src, dst = selected[:, 0], selected[:, 1]
                E_idx[b, src, dst] = sample_E[b, src, dst]
                E_idx[b, dst, src] = sample_E[b, src, dst]

        X_next = F.one_hot(X_idx, num_classes=self.Xdim_output).float()
        E_next = F.one_hot(E_idx, num_classes=self.Edim_output).float()
        return utils.PlaceHolder(X=X_next, E=E_next, y=torch.zeros(bs, 0, device=X.device)).mask(node_mask)

    @torch.no_grad()
    def sample_batch(self, batch_id: int, batch_size: int, keep_chain: int, number_chain_steps: int,
                     save_final: int, num_nodes=None):
        """Unconditional absorbing sampler: all valid nodes/edges start as [MASK]."""
        if num_nodes is None:
            n_nodes = self.node_dist.sample_n(batch_size, self.device)
        elif type(num_nodes) == int:
            n_nodes = num_nodes * torch.ones(batch_size, device=self.device, dtype=torch.int)
        else:
            n_nodes = num_nodes.to(self.device)

        n_max = torch.max(n_nodes).item()
        arange = torch.arange(n_max, device=self.device).unsqueeze(0).expand(batch_size, -1)
        node_mask = arange < n_nodes.unsqueeze(1)

        X_idx = torch.full((batch_size, n_max), self.mask_idx_X, device=self.device, dtype=torch.long)
        E_idx = torch.full((batch_size, n_max, n_max), self.mask_idx_E, device=self.device, dtype=torch.long)
        diagonal = torch.eye(n_max, device=self.device, dtype=torch.bool).unsqueeze(0)
        E_idx[diagonal.expand(batch_size, -1, -1)] = 0

        state = utils.PlaceHolder(
            X=F.one_hot(X_idx, num_classes=self.Xdim_output).float(),
            E=F.one_hot(E_idx, num_classes=self.Edim_output).float(),
            y=torch.zeros(batch_size, 0, device=self.device),
        ).mask(node_mask)

        steps = int(self.cfg.model.get("maskgit_steps", 16))
        steps = max(1, steps)
        for step in range(steps):
            t_value = 1.0 - (step / steps)
            t = torch.full((batch_size, 1), t_value, device=self.device)
            noisy_data = {
                "X_t": state.X,
                "E_t": state.E,
                "y_t": state.y,
                "t": t,
                "node_mask": node_mask,
                "mask_idx_E": self.mask_idx_E,
            }
            extra_data = self.compute_extra_data(noisy_data)
            pred = self.forward(noisy_data, extra_data, node_mask)
            state = self._maskgit_unmask_step(state.X, state.E, pred, node_mask, step, steps)

        final = state.mask(node_mask, collapse=True)
        samples = []
        for i in range(batch_size):
            n = int(n_nodes[i].item())
            sample = [final.X[i, :n].cpu(), final.E[i, :n, :n].cpu()]
            samples.append(strip_mask_class_from_sample(sample, self.mask_idx_X, self.mask_idx_E))

        if self.visualization_tools is not None and save_final > 0:
            result_path = os.path.join(os.getcwd(), f"graphs/{self.name}/epoch{self.current_epoch}_b{batch_id}/")
            self.visualization_tools.visualize(result_path, samples, save_final)

        return samples
