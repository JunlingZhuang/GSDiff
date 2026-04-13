import math
import numpy as np
import torch
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[3])  # D:\Github\GSDiff
SCRIPTS_DIR = str(Path(PROJECT_ROOT) / "scripts")
OUTPUTS_DIR = str(Path(PROJECT_ROOT) / "scripts" / "outputs")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DIFFUSION_STEPS = 1000
RESOLUTION = 512
MAX_NODES = 53

MODEL_PATHS = {
    "unconst_node": str(Path(OUTPUTS_DIR) / "unconst-node-ddpm" / "model1000000.pt"),
    "unconst_edge": str(Path(OUTPUTS_DIR) / "unconst-edge" / "model_stage2_best_061000.pt"),
    "topo_encoder": str(Path(OUTPUTS_DIR) / "topo-ae" / "model_stage0_best_006000.pt"),
    "topo_node": str(Path(OUTPUTS_DIR) / "topo-node-ddpm" / "model1000000.pt"),
    "topo_edge": str(Path(OUTPUTS_DIR) / "topo-edge" / "model_stage2_best_076000.pt"),
    "boun_encoder": str(Path(OUTPUTS_DIR) / "boun-cnn-ae" / "model_stage0_best_006700.pt"),
    "boun_node": str(Path(OUTPUTS_DIR) / "boun-node-ddpm" / "model1000000.pt"),
    "boun_edge": str(Path(OUTPUTS_DIR) / "boun-edge" / "model_stage2_best_065000.pt"),
}

# Cosine beta schedule (from scripts/test_main.py)
alpha_bar = lambda t: math.cos(t / 1.0 * math.pi / 2) ** 2
betas = []
for i in range(DIFFUSION_STEPS):
    t1 = i / DIFFUSION_STEPS
    t2 = (i + 1) / DIFFUSION_STEPS
    betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), 0.999))
BETAS = np.array(betas, dtype=np.float64)
ALPHAS = 1.0 - BETAS
ALPHAS_CUMPROD = np.cumprod(ALPHAS)
ALPHAS_CUMPROD_PREV = np.append(1.0, ALPHAS_CUMPROD[:-1])
SQRT_RECIP_ALPHAS_CUMPROD = np.sqrt(1.0 / ALPHAS_CUMPROD)
SQRT_RECIPM1_ALPHAS_CUMPROD = np.sqrt(1.0 / ALPHAS_CUMPROD - 1)
POSTERIOR_VARIANCE = BETAS * (1.0 - ALPHAS_CUMPROD_PREV) / (1.0 - ALPHAS_CUMPROD)
POSTERIOR_MEAN_COEF1 = BETAS * np.sqrt(ALPHAS_CUMPROD_PREV) / (1.0 - ALPHAS_CUMPROD)
POSTERIOR_MEAN_COEF2 = (1.0 - ALPHAS_CUMPROD_PREV) * np.sqrt(ALPHAS) / (1.0 - ALPHAS_CUMPROD)

# Grouped for convenient import by diffusion.py
DIFFUSION_PARAMS = {
    "sqrt_recip_alphas_cumprod": SQRT_RECIP_ALPHAS_CUMPROD,
    "sqrt_recipm1_alphas_cumprod": SQRT_RECIPM1_ALPHAS_CUMPROD,
    "posterior_variance": POSTERIOR_VARIANCE,
    "posterior_mean_coef1": POSTERIOR_MEAN_COEF1,
    "posterior_mean_coef2": POSTERIOR_MEAN_COEF2,
}

# Room colors for rendering
ROOM_COLORS = {
    0: (244, 241, 222), 1: (234, 182, 159), 2: (107, 112, 92),
    3: (224, 122, 95), 4: (95, 121, 123), 5: (242, 204, 143), 6: (0, 0, 0)
}
