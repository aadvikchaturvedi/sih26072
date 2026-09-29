"""LightningModule: multi-task loss, modality dropout, staged freezing."""

from __future__ import annotations

import lightning as L
import torch

from nowcast_ml.config import Config
from nowcast_ml.data import channels as ch
from nowcast_ml.data.transforms import ModalityDropout, NormStats, drop_radar, model_input
from nowcast_ml.models.losses import BinaryFocalLoss, IntensityWeightedMSE
from nowcast_ml.models.nowcast_model import NowcastModel


class NowcastLitModule(L.LightningModule):
    def __init__(self, cfg: Config, norm_stats: NormStats):
        super().__init__()
        norm_stats.check_channels(cfg.data.channels)
        self.cfg = cfg
        self.norm_stats = norm_stats
        self.channels = list(cfg.data.channels)
        self.model = NowcastModel(
            cfg.model,
            n_channels=len(self.channels),
            t_in=cfg.data.t_in,
            t_out=cfg.data.t_out,
            n_ltg=len(cfg.data.lightning_leads_min),
        )
        md = cfg.train.modality_dropout
        self.modality_dropout = ModalityDropout(self.channels, md.p_radar, md.p_channel)
        self.refl_loss = IntensityWeightedMSE(
            cfg.loss.intensity_thresholds_dbz, cfg.loss.intensity_weights
        )
        self.ltg_loss = BinaryFocalLoss(cfg.loss.focal_alpha, cfg.loss.focal_gamma)
        self.save_hyperparameters(
            {"config": cfg.model_dump(mode="json"), "norm_stats": norm_stats.to_dict()}
        )
        self._apply_trainable()

    # ------------------------------------------------------------------ freezing
    def backbone_frozen(self) -> bool:
        t = self.cfg.train
        if t.trainable == "lightning_head":
            return True
        return self.current_epoch < t.freeze_backbone_epochs

    def _apply_trainable(self) -> None:
        frozen = self.backbone_frozen()
        for p in self.model.backbone_parameters():
            p.requires_grad_(not frozen)
        for p in self.model.lightning_head_parameters():
            p.requires_grad_(True)

    def on_train_epoch_start(self) -> None:
        self._apply_trainable()
        if self.backbone_frozen():
            # keep BatchNorm statistics of frozen parts fixed
            self.model.backbone.eval()
            self.model.refl_head.eval()

    # ------------------------------------------------------------------ forward
    def prepare_batch(self, batch: dict, train: bool, satellite_only: bool = False):
        x = batch["x"].float()
        avail = batch["avail"].bool()
        if train:
            avail = self.modality_dropout(avail)
        if satellite_only:
            avail = drop_radar(avail, self.channels)
        xin = model_input(self.norm_stats.normalize(x), avail)
        return xin

    def forward(self, xin: torch.Tensor):
        return self.model(xin)

    def compute_losses(self, batch: dict, refl: torch.Tensor, ltg_logits: torch.Tensor) -> dict:
        y_dbz = batch["y_refl"].float()
        y_norm = self.norm_stats.normalize_channel(ch.TARGET_CHANNEL, y_dbz)
        refl_l = self.refl_loss(refl.float(), y_norm, y_dbz, batch["y_refl_valid"])
        out = {"refl": refl_l}
        total = self.cfg.loss.refl_weight * refl_l
        if self.model.ltg_enabled:
            ltg_l = self.ltg_loss(ltg_logits.float(), batch["y_ltg"].float(), batch["y_ltg_valid"])
            out["ltg"] = ltg_l
            total = total + self.cfg.loss.ltg_weight * ltg_l
        out["loss"] = total
        return out

    def training_step(self, batch, batch_idx):
        refl, ltg = self(self.prepare_batch(batch, train=True))
        losses = self.compute_losses(batch, refl, ltg)
        bs = batch["x"].shape[0]
        self.log_dict({f"train/{k}": v for k, v in losses.items()}, prog_bar=True, batch_size=bs)
        return losses["loss"]

    def validation_step(self, batch, batch_idx):
        bs = batch["x"].shape[0]
        refl, ltg = self(self.prepare_batch(batch, train=False))
        losses = self.compute_losses(batch, refl, ltg)
        self.log_dict({f"val/{k}": v for k, v in losses.items()}, prog_bar=True, batch_size=bs)
        # quick skill signal: CSI at 35 dBZ over all leads
        pred = self.norm_stats.denormalize_channel(ch.TARGET_CHANNEL, refl.float())
        y, v = batch["y_refl"].float(), batch["y_refl_valid"].bool()
        f, o = (pred >= 35) & v, (torch.nan_to_num(y) >= 35) & v
        hits, fa, miss = (f & o).sum(), (f & ~o).sum(), (~f & o).sum()
        self.log("val/csi35", hits / (hits + fa + miss).clamp_min(1), batch_size=bs)

    # ------------------------------------------------------------------ optim
    def configure_optimizers(self):
        t = self.cfg.train
        head = self.model.lightning_head_parameters()
        backbone = self.model.backbone_parameters()
        groups = [{"params": head, "lr": t.lr}]
        if t.trainable == "all":
            bb_lr = t.lr * (t.unfreeze_lr_factor if t.freeze_backbone_epochs > 0 else 1.0)
            groups.append({"params": backbone, "lr": bb_lr})
        opt = torch.optim.AdamW(groups, weight_decay=t.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(1, t.max_epochs), eta_min=t.lr * 0.01
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}

    # ------------------------------------------------------------------ checkpoint
    @classmethod
    def from_checkpoint(cls, path: str, map_location="cpu") -> NowcastLitModule:
        from nowcast_ml.config import Config

        ck = torch.load(path, map_location=map_location, weights_only=False)
        hp = ck["hyper_parameters"]
        m = cls(Config.model_validate(hp["config"]), NormStats.from_dict(hp["norm_stats"]))
        m.load_state_dict(ck["state_dict"])
        return m
