"""Reverse diffusion loop extracted from scripts/test_main.py."""

import torch

from app.config import (DIFFUSION_STEPS, MAX_NODES, SQRT_RECIP_ALPHAS_CUMPROD,
                         SQRT_RECIPM1_ALPHAS_CUMPROD, POSTERIOR_VARIANCE,
                         POSTERIOR_MEAN_COEF1, POSTERIOR_MEAN_COEF2)


@torch.no_grad()
def reverse_diffusion(
    model,
    global_attn_matrix: torch.Tensor,
    device: str,
    condition_embedding: torch.Tensor | None = None,
    condition_mask: torch.Tensor | None = None,
    boundary_feat: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run full 1000-step reverse diffusion. Returns (1, 53, 10) tensor."""
    sqrt_recip = torch.tensor(SQRT_RECIP_ALPHAS_CUMPROD, device=device)
    sqrt_recipm1 = torch.tensor(SQRT_RECIPM1_ALPHAS_CUMPROD, device=device)
    post_var = torch.tensor(POSTERIOR_VARIANCE, device=device)
    post_coef1 = torch.tensor(POSTERIOR_MEAN_COEF1, device=device)
    post_coef2 = torch.tensor(POSTERIOR_MEAN_COEF2, device=device)

    x_t = torch.randn(1, MAX_NODES, 10, device=device, dtype=torch.float64)

    for step in range(DIFFUSION_STEPS - 1, -1, -1):
        t = torch.tensor([step], device=device)

        if boundary_feat is not None:
            out1, out2 = model(x_t, global_attn_matrix, t, boundary_feat)
        elif condition_embedding is not None:
            out1, out2 = model(x_t, global_attn_matrix, t, condition_embedding, condition_mask)
        else:
            out1, out2 = model(x_t, global_attn_matrix, t)
        noise_pred = torch.cat((out1, out2), dim=2)

        pred_x0 = (sqrt_recip[t][:, None, None] * x_t
                    - sqrt_recipm1[t][:, None, None] * noise_pred)
        pred_x0[:, :, 0:2] = torch.clamp(pred_x0[:, :, 0:2], -1, 1)
        pred_x0[:, :, 2:9] = (pred_x0[:, :, 2:9] >= 0.5).float()
        pred_x0[:, :, 9:10] = (pred_x0[:, :, 9:10] >= 0.75).float()

        mean = (post_coef1[t][:, None, None] * pred_x0
                + post_coef2[t][:, None, None] * x_t)

        if step > 0:
            x_t = mean + torch.sqrt(post_var[t][:, None, None]) * torch.randn_like(x_t)
        else:
            x_t = mean

    return x_t
