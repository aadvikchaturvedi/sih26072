"""T4 inference budget (< 2 s per domain). Runs only with -m gpu on a CUDA machine (e.g. Colab)."""

import time

import pytest
import torch

from nowcast_ml.config import load_config
from nowcast_ml.data.transforms import fit_norm_stats
from nowcast_ml.inference.core import ModelRunner
from nowcast_ml.models import NowcastModel


@pytest.mark.gpu
@pytest.mark.parametrize("size", [256, 512])
def test_full_size_model_under_2s_on_gpu(configs_dir, size):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    import numpy as np

    cfg = load_config(configs_dir / "train" / "finetune_india.yaml")
    C = len(cfg.data.channels)
    model = NowcastModel(cfg.model, C).cuda().eval()
    x = np.random.default_rng(0).normal(size=(1, 7, C, size, size)).astype("float32")
    avail = np.ones_like(x, dtype=bool)
    runner = ModelRunner(
        model, fit_norm_stats([x[0]], cfg.data.channels), cfg.data.channels, torch.device("cuda")
    )
    runner.forecast_batch(x, avail)  # warm-up (cuDNN autotune, allocations)
    torch.cuda.synchronize()
    t = time.perf_counter()
    runner.forecast_batch(x, avail)
    torch.cuda.synchronize()
    ms = 1000 * (time.perf_counter() - t)
    print(f"{torch.cuda.get_device_name()} {size}x{size}: {ms:.0f} ms")
    assert ms < 2000
