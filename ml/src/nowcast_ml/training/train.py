"""``nowcast-train``: train one stage from a YAML config.

    nowcast-train -c configs/train/pretrain_sevir.yaml
    nowcast-train -c configs/train/smoke.yaml train.max_epochs=1 data.batch_size=2

Positional ``key=value`` arguments override config entries.

Normalization: stage ``lightning_head`` reuses the stats of the ``init_from``
model (the frozen backbone expects them); other stages fit stats on the train
split unless ``data.norm_stats`` points to a file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from nowcast_ml.config import Config, dump_config, load_config
from nowcast_ml.data.transforms import NormStats
from nowcast_ml.training.datamodule import NowcastDataModule, save_splits
from nowcast_ml.training.lit_module import NowcastLitModule
from nowcast_ml.utils.device import lightning_accelerator
from nowcast_ml.utils.logging import get_logger
from nowcast_ml.utils.seed import seed_everything

log = get_logger(__name__)


def _resolve_precision(cfg: Config, accelerator: str) -> str:
    p = cfg.train.precision
    if accelerator != "gpu" and p in ("16-mixed", "16"):
        log.warning("precision %s needs CUDA; using 32-bit on %s", p, accelerator)
        return "32"
    return p


def load_init(path: str) -> tuple[dict, NormStats, list[str]]:
    """Weights, norm stats and channels from a registry artifact dir or a Lightning .ckpt."""
    p = Path(path)
    if p.is_dir() or p.name == "latest":
        from nowcast_ml.inference.registry import load_artifact

        art = load_artifact(p)
        return art.state_dict, art.norm_stats, art.channels
    ck = torch.load(p, map_location="cpu", weights_only=False)
    hp = ck["hyper_parameters"]
    sd = {
        k.removeprefix("model."): v for k, v in ck["state_dict"].items() if k.startswith("model.")
    }
    return sd, NormStats.from_dict(hp["norm_stats"]), list(hp["config"]["data"]["channels"])


def run(cfg: Config) -> dict:
    """Train one stage. Returns paths of the outputs (checkpoint, artifact dir if saved)."""
    seed_everything(cfg.train.seed)
    out = Path(cfg.train.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    init = load_init(cfg.train.init_from) if cfg.train.init_from else None
    if init is not None and init[2] != list(cfg.data.channels):
        raise ValueError(
            f"init_from channels {init[2]} differ from config channels {cfg.data.channels}"
        )
    stats = init[1] if (init is not None and cfg.train.stage == "lightning_head") else None

    dm = NowcastDataModule(cfg, norm_stats=stats)
    dm.setup("fit")
    save_splits(dm.splits, out / "splits.json")
    dm.norm_stats.save(out / "norm_stats.json")
    (out / "config.yaml").write_text(dump_config(cfg))

    lit = NowcastLitModule(cfg, dm.norm_stats)
    if init is not None:
        missing, unexpected = lit.model.load_state_dict(init[0], strict=False)
        if unexpected or (missing and cfg.train.stage != "pretrain"):
            log.warning("init_from: missing=%s unexpected=%s", missing, unexpected)
        log.info("initialized weights from %s", cfg.train.init_from)

    accelerator = lightning_accelerator(cfg.train.accelerator)
    loggers = [CSVLogger(str(out), name="logs")]
    if cfg.train.wandb.enabled:
        from lightning.pytorch.loggers import WandbLogger

        loggers.append(
            WandbLogger(
                project=cfg.train.wandb.project, entity=cfg.train.wandb.entity, save_dir=str(out)
            )
        )
    ckpt = ModelCheckpoint(
        dirpath=out / "checkpoints",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        filename="best",
    )
    trainer = L.Trainer(
        max_epochs=cfg.train.max_epochs,
        accelerator=accelerator,
        devices=cfg.train.devices,
        precision=_resolve_precision(cfg, accelerator),
        gradient_clip_val=cfg.train.gradient_clip_val,
        accumulate_grad_batches=cfg.train.accumulate_grad_batches,
        limit_train_batches=cfg.train.limit_train_batches,
        limit_val_batches=cfg.train.limit_val_batches,
        overfit_batches=cfg.train.overfit_batches,
        log_every_n_steps=cfg.train.log_every_n_steps,
        callbacks=[ckpt, LearningRateMonitor(logging_interval="epoch")],
        logger=loggers,
        default_root_dir=str(out),
        enable_model_summary=False,
        deterministic=False,
    )
    trainer.fit(lit, datamodule=dm)

    best = ckpt.best_model_path or ckpt.last_model_path
    result = {"checkpoint": best, "output_dir": str(out)}
    if best:
        lit = NowcastLitModule.from_checkpoint(best)
    result["val_metrics"] = {
        k: float(v) for k, v in trainer.callback_metrics.items() if k.startswith("val/")
    }
    if cfg.registry.save:
        from nowcast_ml.inference.registry import save_artifact

        art_dir = save_artifact(
            lit.model,
            cfg,
            dm.norm_stats,
            root=cfg.registry.root,
            splits=dm.splits,
            train_metrics=result["val_metrics"],
        )
        result["artifact"] = str(art_dir)
        log.info("saved artifact %s", art_dir)
    return result


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="nowcast-train",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("-c", "--config", required=True, help="YAML under configs/train/")
    ap.add_argument("overrides", nargs="*", help="key.path=value overrides")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    res = run(cfg)
    print(res.get("artifact") or res["checkpoint"])


if __name__ == "__main__":
    main()
