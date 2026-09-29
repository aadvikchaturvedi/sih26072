#!/usr/bin/env bash
# Full SEVIR training pipeline (all stages that SEVIR data supports), resumable per stage.
#
#   scripts/train_sevir_pipeline.sh            # from ml/, after `make install` + pip install -e ".[sevir]"
#   N_EVENTS=600 PRETRAIN_EPOCHS=30 scripts/train_sevir_pipeline.sh
#
# Stages: download -> convert -> (a) pretrain -> (c) lightning head -> (d) calibrate
#         -> evaluate + export -> (e) diffusion refiner -> evaluate ensemble.
# Each finished stage writes runs/sevir/<stage>.done (artifact path inside), so a rerun skips it.
set -euo pipefail
cd "$(dirname "$0")/.."

N_EVENTS=${N_EVENTS:-600}
PRETRAIN_EPOCHS=${PRETRAIN_EPOCHS:-30}
LTG_EPOCHS=${LTG_EPOCHS:-8}
REFINER_EPOCHS=${REFINER_EPOCHS:-15}
REG=${REG:-artifacts/models}
BIN=.venv/bin
OUT=runs/sevir
mkdir -p "$OUT"

stage() { echo; echo "=== $(date '+%F %T') $*"; }
done_file() { echo "$OUT/$1.done"; }
is_done() { [ -f "$(done_file "$1")" ]; }
artifact_of() { cat "$(done_file "$1")"; }

if ! is_done download; then
  stage "download $N_EVENTS SEVIR events"
  $BIN/python scripts/download_sevir_subset.py --out data/sevir_raw --n-events "$N_EVENTS" --workers 8
  touch "$(done_file download)"
fi

if ! is_done convert; then
  stage "convert to contract Zarr (3 km, 128x128, north-up, exact LAEA geolocation)"
  $BIN/python scripts/sevir_to_zarr.py --raw data/sevir_raw --out data/sevir_events
  $BIN/nowcast-validate data/sevir_events --grid-spacing-km 3.0 --no-values | tail -1
  touch "$(done_file convert)"
fi

if ! is_done pretrain; then
  stage "(a) pretrain on SEVIR, $PRETRAIN_EPOCHS epochs"
  $BIN/nowcast-train -c configs/train/pretrain_sevir.yaml train.max_epochs="$PRETRAIN_EPOCHS" \
    registry.root="$REG" | tail -1 > "$(done_file pretrain)"
fi
echo "pretrained: $(artifact_of pretrain)"

if ! is_done ltg_head; then
  stage "(c) lightning head on GLM labels, $LTG_EPOCHS epochs"
  $BIN/nowcast-train -c configs/train/sevir_lightning_head.yaml train.init_from="$(artifact_of pretrain)" \
    train.max_epochs="$LTG_EPOCHS" registry.root="$REG" | tail -1 > "$(done_file ltg_head)"
fi
MODEL=$(artifact_of ltg_head)
echo "lightning head: $MODEL"

if ! is_done calibrate; then
  stage "(d) isotonic calibration on the validation split"
  $BIN/nowcast-calibrate --model "$MODEL" --split val
  touch "$(done_file calibrate)"
fi

if ! is_done eval; then
  stage "evaluate on the test split vs persistence / extrapolation / STEPS"
  $BIN/nowcast-eval --model "$MODEL" --split test --baselines all --out reports/sevir_test
  touch "$(done_file eval)"
fi

if ! is_done export; then
  stage "export TorchScript + ONNX with parity check"
  $BIN/nowcast-export --model "$MODEL" --size 128
  touch "$(done_file export)"
fi

if ! is_done refiner; then
  stage "(e) diffusion refiner, $REFINER_EPOCHS epochs"
  $BIN/nowcast-train -c configs/train/sevir_refiner.yaml train.init_from="$MODEL" \
    train.max_epochs="$REFINER_EPOCHS" registry.root="$REG" | tail -1 > "$(done_file refiner)"
fi
ENS=$(artifact_of refiner)
echo "ensemble model: $ENS"

if ! is_done eval_ensemble; then
  stage "evaluate the ensemble model on the test split"
  $BIN/nowcast-eval --model "$ENS" --split test --baselines steps --no-satellite-only \
    --out reports/sevir_test_ensemble
  $BIN/nowcast-export --model "$ENS" --size 128
  touch "$(done_file eval_ensemble)"
fi

stage "done"
echo "deterministic model (calibrated, exported): $MODEL"
echo "ensemble model (same base + refiner):       $ENS"
echo "reports: reports/sevir_test/skill.md, reports/sevir_test_ensemble/skill.md"
