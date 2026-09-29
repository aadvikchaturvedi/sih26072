"""Training stage ``refiner``: diffusion residual refiner on top of a frozen nowcast model."""

from __future__ import annotations

import math

import torch

from nowcast_ml.config import Config
from nowcast_ml.data import channels as ch
from nowcast_ml.data.transforms import NormStats, pad_to_multiple
from nowcast_ml.models.refiner.diffusion import DiffusionRefiner
from nowcast_ml.training.lit_module import NowcastLitModule


class RefinerLitModule(NowcastLitModule):
    """The base model (backbone + heads) is frozen in eval mode; only ``self.refiner`` trains."""

    def __init__(self, cfg: Config, norm_stats: NormStats):
        super().__init__(cfg, norm_stats)
        self.refiner = DiffusionRefiner(
            cfg.data.t_out, self.model.in_channels, cfg.model.refiner_params
        )
        self.model.requires_grad_(False)

    def _apply_trainable(self) -> None:  # base model always frozen in this stage
        if hasattr(self, "refiner"):
            self.model.requires_grad_(False)

    def on_train_epoch_start(self) -> None:
        self.model.eval()

    def _factor(self) -> int:
        return math.lcm(self.model.spatial_factor, self.refiner.spatial_factor)

    def _step(self, batch, train: bool) -> torch.Tensor:
        xin = self.prepare_batch(batch, train=train)
        H, W = xin.shape[-2:]
        xin = pad_to_multiple(xin, self._factor())
        with torch.no_grad():
            det, _ = self.model(xin)
        y = self.norm_stats.normalize_channel(ch.TARGET_CHANNEL, batch["y_refl"].float())
        valid = batch["y_refl_valid"].bool()
        f = self._factor()
        ph, pw = (-H) % f, (-W) % f
        if ph or pw:  # padded border carries no target
            y = torch.nn.functional.pad(y, (0, pw, 0, ph))
            valid = torch.nn.functional.pad(valid, (0, pw, 0, ph))
        return self.refiner.loss(xin, det.float(), y, valid)

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, train=True)
        self.log("train/loss", loss, prog_bar=True, batch_size=batch["x"].shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        g = torch.Generator(device="cpu").manual_seed(
            batch_idx
        )  # fixed noise -> comparable val loss
        state = torch.random.get_rng_state()
        torch.manual_seed(int(torch.randint(0, 2**31 - 1, (1,), generator=g)))
        loss = self._step(batch, train=False)
        torch.random.set_rng_state(state)
        self.log("val/loss", loss, prog_bar=True, batch_size=batch["x"].shape[0])

    def configure_optimizers(self):
        t = self.cfg.train
        opt = torch.optim.AdamW(self.refiner.parameters(), lr=t.lr, weight_decay=t.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(1, t.max_epochs), eta_min=t.lr * 0.01
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}
