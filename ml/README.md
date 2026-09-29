# nowcast_ml

Machine-learning package for **thunderstorm and lightning nowcasting** (IMD, SIH 2026 PS 26072):

- **Reflectivity nowcast**: column-max reflectivity (dBZ), 10…120 min ahead in 10-min steps, on a 2 km grid.
- **Lightning probability**: P(at least 1 flash within 10 km) in the next 30 and 60 min, isotonic-calibrated.
- **First-flash flag**: high probability where there was no lightning in the past 30 min.
- **Degraded mode**: one checkpoint serves both full (radar + satellite + lightning + NWP) and
  **satellite-only** inputs (trained with radar modality dropout).
- **Ensemble members** (optional): a diffusion residual refiner samples sharp, reproducible members
  around the deterministic forecast (`predict(..., n_members=N)`).
- **Baselines** with the same output schema: persistence, Lucas–Kanade extrapolation, and a 20-member pysteps STEPS ensemble.

Only the ML part lives here: no web backend, frontend, warnings/CAP, cell tracking or raw file parsers.
The backend imports the package and calls one class, `Predictor`.

> **Status of results.** Everything in this repository has been run on **synthetic data only** (no SEVIR
> or Indian data, and no GPU, were available when it was built). Scores in generated reports carry a
> "SYNTHETIC DATA" banner and say nothing about real skill.

## Setup

Requires Python 3.11.

```bash
cd ml
uv venv --python 3.11 .venv          # or: python3.11 -m venv .venv
make install                         # pip install -e ".[dev]" (+ macOS OpenMP workaround, see below)
make test                            # offline, CPU, ~30 s
make lint
```

Plain pip works too: `pip install -e "ml/[dev]"`. The extras are `onnx` (export), `wandb` (logging),
`sevir` (S3 download) and `dev` (tests, ruff, onnx).

**macOS note:** `pysteps` compiles with OpenMP. Apple clang needs `brew install libomp`, and
`make install` routes the build through `scripts/cc_openmp.sh`. It also clears the macOS "hidden"
flag that can make Python skip the editable-install `.pth` file. Linux/Colab need neither.

## Data layout

```
ml/data/events/<event_id>.zarr     Indian events (input contract, see docs/data_contract.md)
ml/data/sevir/CATALOG.csv + *.h5   SEVIR subset (scripts/download_sevir_subset.py)
ml/artifacts/models/<name>/<ver>/  model registry (gitignored)
ml/reports/<run>/                  evaluation outputs (gitignored)
```

The input format is specified for the data team in **[docs/data_contract.md](docs/data_contract.md)**.
Check stores with `nowcast-validate data/events/`.

## Quick start on synthetic data (no downloads)

```bash
python scripts/make_synthetic_events.py --out data/events --n-events 12 --size 64 --frames 30
nowcast-train -c configs/train/smoke.yaml data.events_dir=data/events       # -> artifacts/models/nowcast/<ver>
nowcast-calibrate --model artifacts/models/nowcast/latest --split val
nowcast-eval      --model artifacts/models/nowcast/latest --split test --baselines all --out reports/smoke
nowcast-export    --model artifacts/models/nowcast/latest
nowcast-predict   --model artifacts/models/nowcast/latest --event data/events/synth_00000.zarr \
                  --t0 2026-05-12T10:50Z --out forecast.zarr
```

`make e2e` runs the same chain under `runs/e2e/` (about 30 s on a laptop CPU). `make e2e-ensemble`
adds the refiner stage and an ensemble forecast.

## Training stages

Each stage is one command with its own config. `key=value` arguments override any config entry.

| Stage | Command | What it does |
|---|---|---|
| (a) pretrain | `nowcast-train -c configs/train/pretrain_sevir.yaml` | SEVIR (VIL→maxz approx., IR, GLM), 128², 7→12 frames at 10 min |
| (b) fine-tune | `nowcast-train -c configs/train/finetune_india.yaml train.init_from=<stage a artifact>` | India events; backbone frozen for `freeze_backbone_epochs`, then unfrozen at lower LR |
| (c) lightning head | `nowcast-train -c configs/train/lightning_head.yaml train.init_from=<stage b artifact>` | only the lightning head trains (focal loss); reuses stage-b normalization |
| (d) calibration | `nowcast-calibrate --model <stage c artifact> --split val` | isotonic curve per (mode, lead) written into the artifact |
| (e) ensemble, optional | `nowcast-train -c configs/train/refiner.yaml train.init_from=<stage d artifact>` | diffusion residual refiner; base model frozen, its calibrator carried over |

Mixed precision (`16-mixed`) is used on CUDA and falls back to fp32 on CPU/MPS. Batch size, gradient
accumulation and workers are config keys (`data.batch_size`, `train.accumulate_grad_batches`,
`data.num_workers`). W&B logging is off by default: `train.wandb.enabled=true`.
Splits are always **by event** and recorded in each artifact's `splits.json`.

SEVIR subset: `pip install -e ".[sevir]" && python scripts/download_sevir_subset.py --out data/sevir --n-events 200`.
Colab: `notebooks/colab_train.ipynb` wraps the same CLIs (install, mount Drive, stages a–d, eval, export).

## Evaluation

```bash
nowcast-eval --model artifacts/models/nowcast/latest --events data/events --split test --baselines all --out reports/run1
```

Persistence, extrapolation, STEPS, the model (full), the model (satellite-only, radar forced
unavailable) and, if the model has a refiner, the model ensemble are all scored by the same code. Outputs in `reports/run1/`:

- `metrics.json` and `skill.md`: CSI/POD/FAR at 20/35/45 dBZ per lead, FSS, Brier/BSS vs climatology
  and vs "lightning persists", reliability, ROC-AUC, first-flash hit rate / FAR / median lead time,
  a power-spectrum sharpness ratio, and CRPS for every row (ensemble CRPS for STEPS and the model
  ensemble, MAE for deterministic rows), with spread/skill for ensembles.
- `csi_vs_lead.png`, `reliability.png`, `case_study.png`.

With a registry artifact, the metrics are also stored in the artifact's `metrics.json`
and `model_card.md` (`--no-artifact-update` turns this off). Options such as `eval.sample_stride=1`
and `eval.max_samples=500` are config overrides.

## Model artifacts

```
artifacts/models/nowcast/
  latest                       -> text file with the current version
  v20260929-135731-a313011d/
    model.pt  model.ts  model.onnx  refiner.pt*  config.yaml  channels.json  norm_stats.json
    calibrator.pkl  splits.json  metrics.json  model_card.md  manifest.json  export_report.json
```

`manifest.json` holds a SHA-256 for every file. Loading fails with `ModelLoadError` if a file is
missing or was modified, or if channels, normalization, calibrator and weights disagree.
`nowcast-export` writes TorchScript and ONNX (opset 17, dynamic H/W) and checks both against
eager PyTorch (atol 1e-4) at two spatial sizes. (`*` `refiner.pt` exists only for models trained with
stage e; the refiner runs eagerly and is not exported.)

## How the backend calls this

Load once at startup, then call `predict` for each new analysis time. `Predictor` has no mutable state
after `load`, imports no web framework, and is safe to call repeatedly.

<!-- example:backend -->
```python
import xarray as xr

from nowcast_ml.inference import InputContractError, ModelLoadError, Predictor

try:
    predictor = Predictor.load("artifacts/models/nowcast/latest", device="auto")  # load once
except ModelLoadError as err:
    raise SystemExit(f"cannot start nowcasting: {err}")

# Latest >= 7 frames (10-min, UTC) in the input-contract format; missing channels are allowed.
inputs = xr.open_zarr("data/events/synth_00000.zarr").isel(time=slice(0, 12))
t0 = inputs["time"].values[-1]

try:
    forecast = predictor.predict(inputs, t0)
except InputContractError as err:
    print("bad input:", err.problems)       # list of human-readable problems
    raise

print(forecast.attrs["mode"], forecast.attrs["t0"], f"{forecast.attrs['inference_ms']:.0f} ms")
refl_60 = forecast["reflectivity"].sel(lead=60)          # dBZ, (y, x)
p_ltg_30 = forecast["lightning_prob_30"]                 # calibrated probability, (y, x)
new_cells = forecast["first_flash"]                      # bool, (y, x)

# Ensemble members, only for models trained with the refiner stage (predictor.info.has_ensemble):
if predictor.info.has_ensemble:
    ens = predictor.predict(inputs, t0, n_members=10, seed=0)   # same seed -> same members
    members = ens["reflectivity_members"]                        # (member, lead, y, x) dBZ

# Same schema from a reference method, e.g. for side-by-side display:
baseline = predictor.predict_baseline(inputs, t0, kind="extrapolation")
```

### Output contract (`inference/schema.py`)

| Variable | Dims | Meaning |
|---|---|---|
| `reflectivity` | `(lead, y, x)` float32 | forecast MAX-Z in dBZ, `lead` = 10…120 min |
| `lightning_prob_30`, `lightning_prob_60` | `(y, x)` float32 | calibrated P(≥1 flash within 10 km) in the next 30 / 60 min |
| `first_flash` | `(y, x)` bool | `lightning_prob_30 ≥ threshold` and no flash within 10 km in the past 30 min |
| `reflectivity_members` | `(member, lead, y, x)` | optional ensemble: STEPS baseline, or the model with `n_members > 0` (needs the refiner) |

Coordinates: `lead`, `valid_time(lead)`, `lat(y, x)`, `lon(y, x)`. Attributes: `model_name`,
`model_version`, `t0` (ISO-8601 UTC with `Z`), `mode` (`full` | `satellite_only`),
`missing_channels`, `inference_ms`, `output_contract_version`, `first_flash_threshold`.

### Errors

All errors derive from `nowcast_ml.inference.NowcastError`. You can import them without torch from
`nowcast_ml.inference.errors`.

| Exception | When |
|---|---|
| `InputContractError` | inputs break the contract, `t0` is not a frame, or fewer than 7 frames end at `t0` (`.problems` lists everything) |
| `ModelLoadError` | artifact missing, modified, or inconsistent |
| `NowcastError` | e.g. `n_members > 0` on a model without the refiner |

CLI mirror: `nowcast-predict --model … --event path.zarr --t0 2026-05-12T10:30Z --out forecast.zarr`
(`--t0 auto` means the last frame, `--baseline extrapolation` runs a baseline, `--members 10` adds
ensemble members). It exits with code 2 on input errors, 3 on model-load errors and 4 on other errors.

### Performance (measured)

Full-size model from `configs/model/simvp.yaml` (10.1 M parameters), one forward pass, random weights:

| Device | 256×256 (512 km) | 512×512 (1024 km) |
|---|---|---|
| CPU, Apple M4 Pro | 607 ms | 1735 ms |
| MPS, Apple M4 Pro | 76 ms | 284 ms |
| Colab T4 | not measured yet: run `pytest -m gpu tests/integration/test_gpu_benchmark.py` | |

Ensemble members cost extra: the full-size refiner (`configs/model/refiner.yaml`, 2.5 M parameters,
20 DDIM steps) took 1.6 s per member on CPU and 0.37 s per member on MPS at 256×256 (10 members:
34 s CPU, 3.9 s MPS). Use a GPU for ensembles; the deterministic path is unaffected (`n_members=0`).

## Tests

`make test` runs 125+ unit and integration tests offline on CPU (about 40 s), all on synthetic data.
Among them: schema validator good/bad fixtures, hand-computed metric values, blob-advection
direction for the baselines, a one-batch overfit (>90% loss drop), registry round-trip and tamper
detection, TorchScript/ONNX parity, `Predictor` schema checks in both modes, a refiner that must learn a known
residual and keep a bimodal spread, CLI smoke tests, and
execution of this README's backend example and the data-contract producer example.
`make test-all` also runs the `gpu` and `sevir` markers, which skip themselves without CUDA or data.

## Known limits

- No real-data skill numbers exist yet. The SEVIR loader was tested against a fake file with the
  published SEVIR layout. `tests/unit/test_sevir.py::test_real_sevir_lightning_colocated_with_vil`
  checks the orientation assumptions once real files are downloaded.
- SEVIR VIL→dBZ is an approximate inversion. India fine-tuning is expected to correct the intensity.
- The deterministic model blurs with lead time. Refiner members are sharper, but whether their spread
  is well calibrated can only be judged on real data (see CRPS and spread/skill in the report).
- STEPS at convective scales (~5–10 km cells) loses skill quickly on small domains, which is the
  expected behaviour of its noise model.
