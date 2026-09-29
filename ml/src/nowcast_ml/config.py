"""Typed configuration.

YAML files are merged with OmegaConf and validated into pydantic models.
A top-level config (usually under ``configs/train/``) may contain a
``defaults`` mapping that pulls group files from the configs root::

    defaults:
      data: synthetic              # -> configs/data/synthetic.yaml  (into cfg.data)
      model: [simvp, lightning_head]  # merged in order              (into cfg.model)
      eval: default

Keys in the file itself override the defaults, and ``key.path=value`` CLI
overrides are applied last.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from nowcast_ml.data import channels as ch


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


# ----------------------------------------------------------------------------- data
class SyntheticParams(_Base):
    n_events: int = 24
    size: int = 64
    n_frames: int = 30
    n_storms: tuple[int, int] = (1, 4)
    speed_px: tuple[float, float] = (0.5, 2.5)
    radar_missing_prob: float = 0.0
    seed: int = 0


class SevirParams(_Base):
    root: str = "data/sevir"
    catalog: str = "CATALOG.csv"
    size: int = 128
    frame_stride: int = 2  # SEVIR is 5-min; stride 2 -> 10-min steps
    max_events: int | None = None


class SplitConfig(_Base):
    train: float = 0.7
    val: float = 0.15
    test: float = 0.15
    seed: int = 0
    # Optional explicit assignment {"train": [...], "val": [...], "test": [...]}
    explicit: dict[str, list[str]] | None = None

    @model_validator(mode="after")
    def _fractions(self) -> SplitConfig:
        s = self.train + self.val + self.test
        if abs(s - 1.0) > 1e-6:
            raise ValueError(f"split fractions must sum to 1, got {s}")
        return self


class DataConfig(_Base):
    source: Literal["synthetic", "india", "sevir"] = "synthetic"
    channels: list[str] = Field(default_factory=lambda: list(ch.ALL_CHANNELS))
    t_in: int = 7
    t_out: int = 12
    step_minutes: int = 10
    grid_spacing_km: float = 2.0
    lightning_radius_km: float = 10.0
    lightning_leads_min: list[int] = Field(default_factory=lambda: [30, 60])
    events_dir: str | None = None
    sample_stride: int = 1
    crop: int | None = None
    flips: bool = True
    batch_size: int = 8
    num_workers: int = 0
    norm_stats: str | None = None
    max_norm_samples: int = 64
    splits: SplitConfig = Field(default_factory=SplitConfig)
    synthetic: SyntheticParams = Field(default_factory=SyntheticParams)
    sevir: SevirParams = Field(default_factory=SevirParams)

    @field_validator("channels")
    @classmethod
    def _known_channels(cls, v: list[str]) -> list[str]:
        problems = ch.validate_channel_list(v)
        if problems:
            raise ValueError("; ".join(problems))
        if ch.TARGET_CHANNEL not in v:
            raise ValueError(f"channel list must include target channel {ch.TARGET_CHANNEL!r}")
        return v

    @field_validator("lightning_leads_min")
    @classmethod
    def _leads(cls, v: list[int]) -> list[int]:
        if sorted(v) != v or any(x % 10 for x in v):
            raise ValueError("lightning_leads_min must be sorted multiples of 10")
        return v

    @property
    def lead_minutes(self) -> list[int]:
        return [self.step_minutes * (i + 1) for i in range(self.t_out)]


# ---------------------------------------------------------------------------- model
class LightningHeadConfig(_Base):
    enabled: bool = True
    hidden: int = 32


class RefinerConfig(_Base):
    """Diffusion residual refiner (models/refiner/diffusion.py)."""

    base_channels: int = 32
    channel_mults: list[int] = Field(default_factory=lambda: [1, 2, 2])
    timesteps: int = 1000
    sample_steps: int = 20
    residual_scale: float = 1.0  # residuals are divided by this before diffusion


class ModelConfig(_Base):
    name: str = "nowcast"
    backbone: str = "simvp"
    hid_s: int = 32
    hid_t: int = 128
    n_s: int = 4
    n_t: int = 4
    spatio_kernel_enc: int = 3
    spatio_kernel_dec: int = 3
    mlp_ratio: float = 4.0
    drop: float = 0.0
    drop_path: float = 0.0
    lightning_head: LightningHeadConfig = Field(default_factory=LightningHeadConfig)
    refiner: str | None = None
    refiner_params: RefinerConfig = Field(default_factory=RefinerConfig)


class LossConfig(_Base):
    refl_weight: float = 1.0
    ltg_weight: float = 1.0
    intensity_thresholds_dbz: list[float] = Field(default_factory=lambda: [20.0, 35.0])
    intensity_weights: list[float] = Field(default_factory=lambda: [2.0, 5.0])
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0

    @model_validator(mode="after")
    def _same_len(self) -> LossConfig:
        if len(self.intensity_thresholds_dbz) != len(self.intensity_weights):
            raise ValueError("intensity_thresholds_dbz and intensity_weights differ in length")
        return self


# ---------------------------------------------------------------------------- train
class ModalityDropoutConfig(_Base):
    p_radar: float = 0.3
    p_channel: float = 0.0  # independent per-channel dropout for non-radar channels


class WandbConfig(_Base):
    enabled: bool = False
    project: str = "sih26072-nowcast"
    entity: str | None = None


class TrainConfig(_Base):
    stage: Literal["pretrain", "finetune", "lightning_head", "refiner"] = "pretrain"
    max_epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 0.05
    unfreeze_lr_factor: float = 0.1
    freeze_backbone_epochs: int = 0
    trainable: Literal["all", "lightning_head"] = "all"
    init_from: str | None = None
    precision: str = "16-mixed"
    accelerator: str = "auto"
    devices: int = 1
    gradient_clip_val: float = 1.0
    accumulate_grad_batches: int = 1
    limit_train_batches: float | int = 1.0
    limit_val_batches: float | int = 1.0
    overfit_batches: float | int = 0
    log_every_n_steps: int = 10
    seed: int = 42
    output_dir: str = "runs/train"
    modality_dropout: ModalityDropoutConfig = Field(default_factory=ModalityDropoutConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)


class RegistryConfig(_Base):
    root: str = "artifacts/models"
    save: bool = True


# ------------------------------------------------------------------ eval/inference
class EvalConfig(_Base):
    thresholds_dbz: list[float] = Field(default_factory=lambda: [20.0, 35.0, 45.0])
    fss_threshold_dbz: float = 35.0
    fss_scales_px: list[int] = Field(default_factory=lambda: [1, 5, 11])
    reliability_bins: int = 10
    steps_members: int = 20
    ensemble_members: int = (
        10  # refiner members for the "model_ensemble" row (if the model has one)
    )
    sample_stride: int = 3
    max_samples: int | None = None
    case_study_index: int = 0


class InferenceConfig(_Base):
    first_flash_threshold: float = 0.5
    satellite_only_radar_coverage: float = 0.05
    pad_multiple: int = 4


class Config(_Base):
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    loss: LossConfig = Field(default_factory=LossConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    registry: RegistryConfig = Field(default_factory=RegistryConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)


# --------------------------------------------------------------------------- loader
_GROUPS = ("data", "model", "loss", "train", "registry", "eval", "inference")


def _configs_root(path: Path) -> Path:
    # configs/train/foo.yaml -> configs/
    return path.parent.parent


def _load_defaults(root: Path, defaults: DictConfig) -> DictConfig:
    merged = OmegaConf.create({})
    for group, names in defaults.items():
        if group not in _GROUPS:
            raise ValueError(f"unknown config group in defaults: {group!r}")
        if isinstance(names, str):
            names = [names]
        for name in names:
            f = root / str(group) / f"{name}.yaml"
            if not f.exists():
                raise FileNotFoundError(f"defaults entry {group}: {name} -> {f} not found")
            merged = OmegaConf.merge(merged, {group: OmegaConf.load(f)})
    return merged


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    """Load a YAML config (with ``defaults``) plus dotlist overrides into a :class:`Config`."""
    base = OmegaConf.create({})
    if path is not None:
        path = Path(path)
        raw = OmegaConf.load(path)
        defaults = raw.pop("defaults", None)
        if defaults is not None:
            base = _load_defaults(_configs_root(path), defaults)
        base = OmegaConf.merge(base, raw)
    if overrides:
        base = OmegaConf.merge(base, OmegaConf.from_dotlist(list(overrides)))
    container = OmegaConf.to_container(base, resolve=True)
    return Config.model_validate(container or {})


def config_from_yaml_text(text: str) -> Config:
    return Config.model_validate(OmegaConf.to_container(OmegaConf.create(text), resolve=True))


def dump_config(cfg: Config) -> str:
    return OmegaConf.to_yaml(OmegaConf.create(cfg.model_dump(mode="json")))
