"""Documentation examples must run as written."""

import re
import shutil

from nowcast_ml.data import channels as ch

from ..conftest import ML_ROOT


def _example(path, name: str) -> str:
    text = (ML_ROOT / path).read_text()
    m = re.search(rf"<!-- example:{name} -->\s*```python\n(.*?)```", text, re.S)
    assert m, f"example {name!r} not found in {path}"
    return m.group(1)


def test_readme_backend_example_runs(artifact_root, tmp_path, monkeypatch, capsys):
    from nowcast_ml.data.synthetic import write_events

    write_events(tmp_path / "data" / "events", 1, seed=0, size=32, n_frames=14)
    shutil.copytree(artifact_root, tmp_path / "artifacts" / "models" / "nowcast")
    monkeypatch.chdir(tmp_path)
    ns: dict = {}
    exec(compile(_example("README.md", "backend"), "README.md", "exec"), ns)  # noqa: S102
    assert ns["forecast"].attrs["mode"] == "full"
    assert ns["refl_60"].dims == ("y", "x")
    assert ns["baseline"].attrs["model_name"] == "baseline_extrapolation"
    assert "ms" in capsys.readouterr().out


def test_data_contract_producer_example_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    exec(compile(_example("docs/data_contract.md", "producer"), "data_contract.md", "exec"), {})  # noqa: S102
    assert (tmp_path / "20260512_example.zarr").is_dir()


def test_data_contract_lists_every_channel():
    text = (ML_ROOT / "docs" / "data_contract.md").read_text()
    for c in ch.CHANNELS.values():
        row = re.search(rf"^\| `{c.name}` \| (\w+) \| ([^|]+) \|", text, re.M)
        assert row, f"channel {c.name} missing from data_contract.md"
        assert row.group(1) == c.group and row.group(2).strip() == c.unit


def test_colab_notebook_references_exist():
    import json
    import tomllib

    nb = json.loads((ML_ROOT / "notebooks" / "colab_train.ipynb").read_text())
    src = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    for cfg in re.findall(r"configs/[\w/]+\.yaml", src):
        assert (ML_ROOT / cfg).exists(), cfg
    scripts = tomllib.loads((ML_ROOT / "pyproject.toml").read_text())["project"]["scripts"]
    for cli in set(re.findall(r"!(nowcast-[\w-]+)", src)):
        assert cli in scripts, cli
    for s in re.findall(r"scripts/\w+\.py", src):
        assert (ML_ROOT / s).exists(), s
