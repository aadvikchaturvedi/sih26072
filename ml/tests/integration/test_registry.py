import json

import pytest
import torch

from nowcast_ml.inference.errors import ModelLoadError
from nowcast_ml.inference.registry import (
    load_artifact,
    refresh_manifest,
    resolve_version_dir,
    save_artifact,
)


def test_roundtrip_and_latest(artifact_root, tmp_path):
    art = load_artifact(artifact_root / "latest")
    assert art.version == (artifact_root / "latest").read_text().strip()
    for f in (
        "model.pt",
        "config.yaml",
        "channels.json",
        "norm_stats.json",
        "metrics.json",
        "model_card.md",
        "manifest.json",
        "splits.json",
    ):
        assert art.file(f).exists(), f
    assert resolve_version_dir(artifact_root) == art.path  # name dir resolves via latest
    # save the loaded weights again -> identical tensors after reload
    from nowcast_ml.models import NowcastModel

    cfg = art.config
    m = NowcastModel(cfg.model, len(art.channels), cfg.data.t_in, cfg.data.t_out)
    m.load_state_dict(art.state_dict)
    d2 = save_artifact(m, cfg, art.norm_stats, root=tmp_path, splits=art.splits)
    art2 = load_artifact(tmp_path / "nowcast" / "latest")
    assert art2.path == d2
    for k, v in art.state_dict.items():
        assert torch.equal(v, art2.state_dict[k])


def test_model_card_mentions_synthetic(artifact_root):
    card = (resolve_version_dir(artifact_root) / "model_card.md").read_text()
    assert "SYNTHETIC" in card and "Known limits" in card


def test_tampered_file_fails_loudly(artifact_copy):
    d = resolve_version_dir(artifact_copy)
    with open(d / "model.pt", "ab") as f:
        f.write(b"x")
    with pytest.raises(ModelLoadError, match="manifest hash"):
        load_artifact(d)


def test_channel_mismatch_fails_loudly(artifact_copy):
    d = resolve_version_dir(artifact_copy)
    info = json.loads((d / "channels.json").read_text())
    info["channels"] = list(reversed(info["channels"]))
    (d / "channels.json").write_text(json.dumps(info))
    refresh_manifest(d)
    with pytest.raises(ModelLoadError, match="does not match config channels"):
        load_artifact(d)


def test_norm_mismatch_fails_loudly(artifact_copy):
    d = resolve_version_dir(artifact_copy)
    ns = json.loads((d / "norm_stats.json").read_text())
    for k in ("channels", "mean", "std", "log1p"):
        ns[k] = ns[k][:-1]
    (d / "norm_stats.json").write_text(json.dumps(ns))
    refresh_manifest(d)
    with pytest.raises(ModelLoadError, match="normalization mismatch"):
        load_artifact(d)


def test_missing_file_and_bad_path(artifact_copy, tmp_path):
    d = resolve_version_dir(artifact_copy)
    (d / "norm_stats.json").unlink()
    with pytest.raises(ModelLoadError, match="missing required"):
        load_artifact(d)
    with pytest.raises(ModelLoadError, match="not found"):
        load_artifact(tmp_path / "nope")


def test_weight_shape_mismatch_fails(artifact_copy):
    from nowcast_ml.inference import Predictor

    d = resolve_version_dir(artifact_copy)
    sd = torch.load(d / "model.pt", weights_only=True)
    k = next(iter(sd))
    sd[k] = torch.zeros(1)
    torch.save(sd, d / "model.pt")
    refresh_manifest(d)
    with pytest.raises(ModelLoadError, match="weights do not match"):
        Predictor.load(d, device="cpu")
