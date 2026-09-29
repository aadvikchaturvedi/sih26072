import pytest
from pydantic import ValidationError

from nowcast_ml.config import Config, config_from_yaml_text, dump_config, load_config


def test_all_train_configs_load(configs_dir):
    for f in sorted((configs_dir / "train").glob("*.yaml")):
        cfg = load_config(f)
        assert isinstance(cfg, Config), f


def test_defaults_merge_and_overrides(configs_dir):
    cfg = load_config(
        configs_dir / "train" / "smoke.yaml", ["train.max_epochs=7", "data.batch_size=2"]
    )
    assert cfg.data.source == "synthetic"
    assert cfg.model.hid_s == 16  # file overrides model/simvp.yaml
    assert cfg.model.lightning_head.enabled  # from model/lightning_head.yaml
    assert cfg.train.max_epochs == 7 and cfg.data.batch_size == 2


def test_sevir_config_uses_full_channel_layout(configs_dir):
    cfg = load_config(configs_dir / "train" / "pretrain_sevir.yaml")
    india = load_config(configs_dir / "train" / "finetune_india.yaml")
    assert cfg.data.channels == india.data.channels


def test_unknown_channel_rejected():
    with pytest.raises(ValidationError, match="unknown channel"):
        Config.model_validate({"data": {"channels": ["maxz", "bogus"]}})


def test_unknown_key_rejected():
    with pytest.raises(ValidationError):
        Config.model_validate({"train": {"lrr": 1}})


def test_dump_roundtrip():
    cfg = Config()
    assert config_from_yaml_text(dump_config(cfg)) == cfg
