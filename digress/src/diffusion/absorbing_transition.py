import torch

import src.utils as utils


class AbsorbingGraphTransition:
    """Absorbing-state transition matrix for categorical graph diffusion.

    Vanilla DiGress corrupts each node/edge by moving it toward the dataset
    marginal distribution. Absorbing D3PM instead moves every class toward one
    special [MASK] class. Once a token becomes [MASK] in the forward process,
    it stays unknown; the neural network learns to reconstruct the original
    room/edge type from the partially observed graph.
    """

    def __init__(
        self,
        x_classes: int,
        e_classes: int,
        y_classes: int,
        x_mask_index: int | None = None,
        e_mask_index: int | None = None,
    ):
        self.X_classes = int(x_classes)
        self.E_classes = int(e_classes)
        self.y_classes = int(y_classes)
        self.x_mask_index = self.X_classes - 1 if x_mask_index is None else int(x_mask_index)
        self.e_mask_index = self.E_classes - 1 if e_mask_index is None else int(e_mask_index)

    def _absorbing_matrix(self, beta_or_one_minus_alpha_bar: torch.Tensor, classes: int, mask_index: int, device):
        """Build Q = keep_prob * I + mask_prob * absorbing_rows.

        `beta_or_one_minus_alpha_bar` is beta_t for one-step Qt, or
        (1 - alpha_bar_t) for cumulative Qt_bar. Shape is (batch, 1).
        """
        mask_prob = beta_or_one_minus_alpha_bar.to(device).unsqueeze(1)
        eye = torch.eye(classes, device=device).unsqueeze(0)
        absorbing_rows = torch.zeros(1, classes, classes, device=device)
        absorbing_rows[:, :, mask_index] = 1.0
        return mask_prob * absorbing_rows + (1.0 - mask_prob) * eye

    def _y_matrix(self, beta_or_one_minus_alpha_bar: torch.Tensor, device):
        # MSD/RPLAN use an empty global y vector. Keep this branch for API
        # compatibility with DiGress's transition interface.
        if self.y_classes == 0:
            return torch.zeros((beta_or_one_minus_alpha_bar.shape[0], 0, 0), device=device)
        beta = beta_or_one_minus_alpha_bar.to(device).unsqueeze(1)
        uniform = torch.ones(1, self.y_classes, self.y_classes, device=device) / self.y_classes
        eye = torch.eye(self.y_classes, device=device).unsqueeze(0)
        return beta * uniform + (1.0 - beta) * eye

    def get_Qt(self, beta_t, device):
        """Return one-step transition matrices from t-1 to t."""
        return utils.PlaceHolder(
            X=self._absorbing_matrix(beta_t, self.X_classes, self.x_mask_index, device),
            E=self._absorbing_matrix(beta_t, self.E_classes, self.e_mask_index, device),
            y=self._y_matrix(beta_t, device),
        )

    def get_Qt_bar(self, alpha_bar_t, device):
        """Return cumulative transition matrices from clean data to step t."""
        mask_prob = 1.0 - alpha_bar_t
        return utils.PlaceHolder(
            X=self._absorbing_matrix(mask_prob, self.X_classes, self.x_mask_index, device),
            E=self._absorbing_matrix(mask_prob, self.E_classes, self.e_mask_index, device),
            y=self._y_matrix(mask_prob, device),
        )
