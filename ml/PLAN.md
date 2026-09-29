# nowcast_ml: build plan (SIH 2026, PS 26072)

This package covers the ML part only: data contract and datasets, baselines, SimVP model, training, calibration, evaluation, and a `Predictor` the backend calls. It does not include a backend, frontend, warnings/CAP, cell tracking, or raw radar/satellite parsers.

## 0. Environment facts (checked, not assumed)

| Item | Status on this machine | Consequence |
|---|---|---|
| Python 3.11 | Not installed (3.13 and 3.14 are). `uv` is available. | Create `ml/.venv` with `uv venv --python 3.11`. uv downloads a standalone 3.11. Nothing global changes. |
| GPU | None (Apple M4 Pro, 24 GB, no CUDA) | All tests and smoke runs use CPU or MPS on synthetic data. T4 timing and real skill numbers can't be measured here. Reports will say so and won't show made-up numbers. |
| SEVIR | Not downloaded | SEVIR code is tested only against a synthetic HDF5 file with the real SEVIR layout. Tests that need real files are marked `@pytest.mark.sevir` and skipped. |
| git | `sih2/` is **not a git repository** | Per-milestone commits need a repo. See open question Q1. |

Version pins: I'll resolve these with `uv pip compile` against Python 3.11 in M0 and write exact `==` pins into `pyproject.toml`, with torch 2.x and lightning 2.x. I won't guess version numbers before resolving them.

## 1. Directory tree

It follows the prompt. **Deviations and additions are marked ★**, with reasons in §4.

```
ml/
  pyproject.toml              # package nowcast_ml, pinned deps, extras [dev,wandb,onnx], console scripts
  Makefile                    # ★ make install / test / test-all / lint / format / e2e
  .gitignore                  # ★ artifacts/, reports/, .venv/, data/, *.zarr
  README.md
  PLAN.md
  docs/
    data_contract.md          # for the data team (docs live inside ml/, see §4)
  scripts/
    download_sevir_subset.py  # boto3/aws-cli anonymous S3 fetch of N VIL/IR/GLM files + catalog
    make_synthetic_events.py  # ★ writes contract-valid synthetic event Zarrs (used by e2e + README)
  configs/
    data/   sevir.yaml  india.yaml  synthetic.yaml
    model/  simvp.yaml  lightning_head.yaml
    train/  pretrain_sevir.yaml  finetune_india.yaml  lightning_head.yaml  smoke.yaml ★
    eval/   default.yaml
  src/nowcast_ml/
    __init__.py               # __version__, re-exports Predictor lazily
    config.py                 # OmegaConf load + defaults-list merge + CLI overrides -> pydantic models
    data/
      schema.py               # INPUT CONTRACT: constants, validate_event(), validate_dataset()
      channels.py             # CHANNELS registry, groups, units, valid ranges, derived-channel recipes
      sevir.py                # SEVIR HDF5+catalog -> (x, mask, targets) in canonical layout
      zarr_events.py          # India event Zarr -> sliding-window samples
      synthetic.py            # moving/growing/decaying storm blobs + correlated sat/lightning/NWP fields
      transforms.py           # NormStats fit/save/load, ModalityDropout, RandomCrop, flips
      labels.py               # lightning labels (disk max-pool over space, any() over lead window)
      splits.py               # split by event id (deterministic hash or explicit list), leak check
    baselines/
      base.py                 # ★ common Forecaster protocol (numpy in -> ForecastArrays out)
      persistence.py
      extrapolation.py        # pysteps LK optical flow + semi-Lagrangian
      steps_ensemble.py       # pysteps STEPS, 20 members (configurable)
    models/
      simvp.py                # Encoder / Translator (gSTA blocks) / Decoder
      backbones.py            # ★ tiny name->class registry so config can swap backbones
      heads.py                # ReflectivityHead (12 frames), LightningHead (2 maps: +30, +60)
      nowcast_model.py        # NowcastModel: builds input = [x_norm*avail, avail_mask], runs backbone+heads
      losses.py               # IntensityWeightedMSE, BinaryFocalLoss, MultiTaskLoss
      refiner/
        base.py               # Refiner protocol: sample(cond, det_pred, n) -> members | None
        null.py               # NullRefiner (default; returns None)
        diffusion.py          # M9 only; stub raising NotImplementedError until then
    training/
      lit_module.py           # LightningModule: multi-task weights, freeze/unfreeze schedule, metrics
      datamodule.py           # LightningDataModule over sevir|india|synthetic
      callbacks.py            # ★ BackboneFreezeUnfreeze, SaveNormStats
      train.py                # nowcast-train
    calibration/
      isotonic.py             # per-lead IsotonicCalibrator fit/apply/save/load (pickle + sklearn version check)
      fit_cli.py              # ★ nowcast-calibrate (stage d as its own command)
    evaluation/
      metrics.py
      forecasters.py          # ★ adapters: Predictor(full), Predictor(sat-only), baselines -> one interface
      evaluate.py             # nowcast-eval
      report.py               # skill.md + PNGs
    inference/
      errors.py               # ★ NowcastError, InputContractError, ModelLoadError
      schema.py               # OUTPUT CONTRACT: build_forecast_dataset(), validate_forecast()
      predictor.py            # Predictor
      export.py               # nowcast-export
      registry.py             # save_artifact / resolve(latest) / load_artifact / model card
      predict_cli.py          # nowcast-predict
    utils/  seed.py  io.py  logging.py  device.py
  artifacts/                  # gitignored
  reports/                    # gitignored
  tests/
    conftest.py               # synthetic fixtures (tiny grids, tiny models); registers gpu/sevir/slow markers
    unit/  test_channels.py test_schema.py test_transforms.py test_labels.py test_losses.py
           test_model_shapes.py test_metrics.py test_baselines.py test_splits.py test_config.py ...
    integration/ test_overfit.py test_registry.py test_export_parity.py test_predictor.py test_cli.py
  notebooks/colab_train.ipynb
```

## 2. Key design decisions

### 2.1 Tensor layout (the same for SEVIR, India, and synthetic)
Every dataset yields a dict:
- `x`: `(T_in=7, C, H, W)` float32 in physical units, over the model's **canonical channel list** (fixed per artifact and stored in `channels.json`). Channels the source doesn't have are NaN at load time.
- `avail`: `(T_in, C, H, W)` bool. It is built from the Zarr per-group `missing` mask, channel presence, and NaNs. Missing values are never zero-filled in the data. Values are set to 0 only **inside the model input, after normalization**, and the model is always given `avail` alongside them.
- `y_refl`: `(12, H, W)` maxz dBZ targets, plus `y_refl_valid` mask. The loss ignores pixels with no radar truth.
- `y_ltg`: `(2, H, W)` binary lightning labels (+30, +60), plus `y_ltg_valid`.

Model input = `concat(x_norm * avail, avail)`, so there are 2C channels per frame. I use **per-channel** masks, not per-group, because the model has to accept *any subset* of channels. SEVIR, for example, has no `cappi3km`. Per-channel masks handle both whole-group dropouts and partial ones. The prompt calls this the "missing-mask channel", and here it is spatial, so it also marks radar coverage holes.

### 2.2 Input contract (event Zarr)
- `x(time, channel, y, x)` float32
- `missing(time, group, y, x)` uint8, with `group ∈ {radar, satellite, lightning, nwp}`. 1 means missing.
- Coordinates: `time` (UTC, datetime64, strict 10-min spacing, gaps allowed only as NaT-free breaks that the validator reports), `channel` (str), `lat(y,x)`, `lon(y,x)`
- attrs: `event_id`, `grid_spacing_km=2.0`, `crs`, `contract_version`
- Optional `lightning_points` table: `flash_time(flash)`, `flash_lat(flash)`, `flash_lon(flash)`

`validate_event(path) -> list[str]` checks all of the above plus unit sanity ranges from `channels.py` (for example, BT in 150–340 K and maxz in −10…80 dBZ). `docs/data_contract.md` documents the contract with a worked xarray example that writes a valid file.

### 2.3 SEVIR mapping
SEVIR has VIL (384², 1 km), IR 10.7 µm and 6.9 µm (192², 2 km), and GLM flash points, all at 5-min steps over 4 h. I **subsample to 10 min**, so 7→12 frames = 70 min history and 120 min lead, matching India. Grids are resampled to 128². Mapping:
- `vil → maxz`: digital VIL is converted to kg m⁻² and then to an equivalent dBZ through the inverse Greene–Clark VIL–Z relation. **This is an approximation.** Pretraining learns motion and growth. India fine-tuning corrects the intensity calibration. I'll state this in the model card.
- `ir107 → tir1_bt`, `ir069 → wv_bt`, and derived `tir1_cooling` and `tir1_minus_wv`
- GLM `lght` points → `flash_density` and lightning labels
- All other channels are unavailable, so `avail=0`.

### 2.4 Labels
A pixel is positive if any flash falls within 10 km during `(t0, t0+L]`, for L = 30 and 60. The implementation takes `any()` over the lead frames, then a binary dilation with a disk of radius `10 km / grid_spacing_km` px. That's 5 px on the India 2 km grid, and the radius is computed from the grid spacing on SEVIR's 128² grid. Flash points are used when available, otherwise `flash_density > 0`. `first_flash` uses the same disk dilation over the past 30 min to define "no flash in the past 30 min".

### 2.5 Model
- **Backbone:** SimVP v2 (gSTA translator), fully convolutional. H and W must be divisible by 4 (2× downsampling ×2), and the Predictor reflect-pads and then crops. Chosen with `model.backbone: simvp` through `backbones.py`, and the interface is `forward(x) -> (dec_feats, latent)`.
- **Reflectivity head:** 1×1 conv on decoder features → 12 frames, predicted in normalized dBZ.
- **Lightning head:** a small conv decoder on the shared translator latent → 2 logits, plus the latest `flash_density`/`avail` as a skip input.
- **Losses:** intensity-weighted MSE with w(dBZ) = 1, then ×k₁ ≥ 20, ×k₂ ≥ 35 (configurable, default 1/2/5), masked by validity. Binary focal loss (α, γ from config). Multi-task weights come from the train config.
- **Modality dropout:** during training, with p=0.3, set `avail[radar]=0` for the whole sample. Occasional per-channel NWP/satellite dropout is available behind a flag.
- **Mode:** `satellite_only` when every radar channel is unavailable over more than 95% of the domain for the last input frame (threshold in config). Otherwise `full`.

### 2.6 Training stages (each is its own config and command)
a. `nowcast-train -c configs/train/pretrain_sevir.yaml`: reflectivity loss, plus lightning loss if GLM is present
b. `finetune_india.yaml`: `init_from` the stage-a checkpoint. Backbone frozen for `freeze_epochs`, then unfrozen with a lower LR
c. `lightning_head.yaml`: freeze everything except the lightning head, focal loss only
d. `nowcast-calibrate --model <ckpt|artifact> --split val`: per-lead isotonic fit, written to `calibrator.pkl`

Defaults are Colab-T4 friendly: `precision: 16-mixed` (a T4 has no bf16), with configurable `batch_size`, `accumulate_grad_batches`, and `num_workers`. `device.py` picks `auto` as cuda → mps → cpu. W&B is off by default (`logger.wandb.enabled: false`) and falls back to CSV logging.

### 2.7 Predictor (backend contract)
```python
Predictor.load(path, device="auto") -> Predictor
Predictor.predict(inputs: xr.Dataset, t0: datetime, *, n_members: int = 0) -> xr.Dataset
Predictor.predict_baseline(inputs, t0, kind: Literal["persistence","extrapolation","steps"]) -> xr.Dataset
Predictor.info -> ModelInfo  # name, version, channels, grid, leads (read-only)
```
- It selects the last 7 frames ≤ t0 and raises `InputContractError` if fewer than 7 are contiguous. It reorders channels to the artifact's list, marks absent ones unavailable, and reports unknown extra channels in a warning attr instead of failing.
- Load fails loudly with `ModelLoadError` if a file is missing, the channel list differs from `norm_stats`, the config hash doesn't match, or the `contract_version` is incompatible.
- It is stateless after load. It runs under `torch.inference_mode()`, has no globals, and does no web imports.
- `n_members > 0` goes through the Refiner, which gives `reflectivity_members`. With `NullRefiner` it raises a clear error until M9. The API doesn't change.
- Baseline lightning fields: persistence uses dilated past-30-min flashes. Extrapolation and STEPS advect that field with the same motion field. STEPS also fills `reflectivity_members`.

### 2.8 Registry
Each version lives in `artifacts/models/<name>/<version>/`, where version = `vYYYYMMDD-HHMMSS-<shorthash>`. The files are `model.pt`, `model.ts`, `model.onnx` (optional), `config.yaml`, `channels.json`, `norm_stats.json`, `calibrator.pkl`, `metrics.json`, `model_card.md`, and `manifest.json` (★ SHA-256 of each file plus library versions). `artifacts/models/<name>/latest` is a text file holding the version string, and `load("…/latest")` resolves it.

### 2.9 Evaluation
`nowcast-eval --model <artifact> --events <split|dir> --baselines all --out reports/<run>` runs every forecaster through the same adapter and the same metric code. Rows: persistence, extrapolation, STEPS, model (full), and model (satellite-only, with radar availability forced to 0). Outputs are `skill.md`, `metrics.json`, `csi_vs_lead.png`, `reliability.png`, `roc.png`, `psd_sharpness.png`, and `case_study.png`. Metrics are hand-checked in unit tests against tiny arrays computed by hand. When the eval runs on synthetic data, the report header says **"SYNTHETIC DATA — not indicative of real skill"**.

## 3. Milestones (each: implement → `make test` → fix → commit)

| # | Deliverable | Tests added |
|---|---|---|
| M0 | uv venv (py3.11), pyproject with pins, Makefile, ruff config, `config.py`, `synthetic.py`, utils, pytest markers | config merge/override/validation, synthetic generator determinism and motion |
| M1 | channels, input schema + validator, zarr_events, sevir (with a fake-HDF5 fixture), transforms, labels, splits, `download_sevir_subset.py`, `make_synthetic_events.py` | good/bad Zarr fixtures, norm round-trip, dropout masks, hand-built label case, no event leakage |
| M2 | output schema, persistence, extrapolation (LK+SL), STEPS | blob advected in the correct direction, output validates, STEPS member count |
| M3 | SimVP, heads, losses, NowcastModel, LitModule, DataModule, `nowcast-train` | shapes (full / radar-missing), loss weighting, focal behaviour, 1-batch overfit (>90% drop) |
| M4 | metrics, forecaster adapters, `nowcast-eval`, report | metrics vs hand values, eval CLI smoke, report files exist |
| M5 | modality dropout wired into training, satellite-only eval row, mode detection | mode detection, dropout rate statistics |
| M6 | lightning-head stage, isotonic calibrator, `nowcast-calibrate` | calibrator monotone + save/load, Brier improves on a miscalibrated synthetic case |
| M7 | registry, Predictor, TorchScript/ONNX export + parity, `nowcast-predict`, model card | registry round-trip + mismatch errors, parity atol 1e-4, predictor schema both modes, CLI smokes |
| M8 | Colab notebook, README (with a backend example that is executed in a test), `docs/data_contract.md` | README example test (extracted and run) |
| M9 (opt.) | diffusion residual refiner behind the Refiner interface | members shape, deterministic seed |

`make e2e` runs make_synthetic_events → train (smoke) → calibrate → eval → export → predict from a clean checkout. The runtime budget for `make test` is under 3 min on CPU, with tiny grids (32–64²) and tiny widths (hidden 16–32).

## 4. Deviations and additions, with reasons
- **`docs/` and `scripts/` live under `ml/`**, because nothing may be created outside `ml/`.
- **`Makefile`, `.gitignore`** are needed for `make test` and `make lint`, and to keep artifacts and reports out of git.
- **`inference/errors.py`** keeps typed exceptions in one importable place so the backend can catch them without loading torch-heavy modules.
- **`baselines/base.py` and `evaluation/forecasters.py`** give one forecaster protocol, so eval scores everything with identical code.
- **`models/backbones.py`** makes the backbone swappable by config name.
- **`training/callbacks.py`** holds the freeze/unfreeze schedule, keeping lit_module small.
- **`calibration/fit_cli.py` (`nowcast-calibrate`)** makes stage (d) an explicit command. The prompt lists four stages but only train/eval/export/predict CLIs.
- **`configs/train/smoke.yaml`, `scripts/make_synthetic_events.py`** are required for the synthetic end-to-end path in the definition of done.
- **`manifest.json` in artifacts** stores hashes and library versions, which support the "fail loudly on mismatch" requirement.

## 5. Risks and limits I'll state rather than hide
- There's no GPU here, so the **<2 s T4 inference target can't be verified locally**. I'll measure and report CPU/MPS timing, and add a `@pytest.mark.gpu` benchmark test for Colab.
- SEVIR VIL→dBZ is an approximation (§2.3).
- STEPS with 20 members on large domains is slow on CPU. Tests use 3 members on 64² grids.
- ONNX export of gSTA (depthwise large-kernel conv, GELU) should work at opset 17. If it fails, `model.onnx` is skipped and the export report records the reason, as the prompt allows ("if export succeeds").
- Every skill number produced here comes from synthetic data and will be labeled that way.

## 6. Open questions for approval
- **Q1: git.** `sih2/` isn't a repo. I propose running `git init` at `sih2/` (repo root) and committing only `ml/`. The `.git/` directory is repo metadata, not code, so this doesn't break "no code outside ml/". The alternative is `git init` inside `ml/`.
- **Q2: Python.** Is it OK to create `ml/.venv` with a uv-managed Python 3.11? The alternative is running on the installed 3.13, but some pinned deps (for example pysteps wheels) are less certain there.
- **Q3: Channel masking granularity.** I'm using per-channel availability masks (§2.1), a superset of per-group masks. Is that OK?
