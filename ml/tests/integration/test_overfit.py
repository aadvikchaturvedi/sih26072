"""A tiny model must overfit one synthetic batch (loss drops > 90%)."""

import torch
from torch.utils.data import default_collate

from nowcast_ml.training.datamodule import NowcastDataModule
from nowcast_ml.training.lit_module import NowcastLitModule
from nowcast_ml.utils.seed import seed_everything


def test_overfit_one_batch(tiny_cfg):
    seed_everything(0)
    cfg = tiny_cfg.model_copy(deep=True)
    cfg.train.modality_dropout.p_radar = 0.0
    cfg.train.modality_dropout.p_channel = 0.0
    dm = NowcastDataModule(cfg)
    dm.setup()
    batch = default_collate([dm.sources.dataset("train")[i] for i in (0, 3)])
    lit = NowcastLitModule(cfg, dm.norm_stats)
    opt = torch.optim.Adam(lit.parameters(), lr=3e-3)
    lit.train()
    losses = []
    for _ in range(250):
        refl, ltg = lit(lit.prepare_batch(batch, train=True))
        loss = lit.compute_losses(batch, refl, ltg)["loss"]
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.1 * losses[0], (losses[0], losses[-1])
