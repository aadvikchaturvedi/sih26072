PY ?= .venv/bin/python
PIP ?= uv pip

.PHONY: venv install test test-all lint format e2e clean

venv:
	uv venv --python 3.11 .venv

# macOS: pysteps needs OpenMP (brew install libomp); scripts/cc_openmp.sh maps -fopenmp for Apple clang.
# Python skips .pth files carrying the macOS "hidden" flag, which breaks the editable install; clear it.
install:
	@if [ "$$(uname)" = "Darwin" ]; then \
		CC=$(CURDIR)/scripts/cc_openmp.sh LDSHARED="$(CURDIR)/scripts/cc_openmp.sh -bundle -undefined dynamic_lookup" \
		$(PIP) install --python $(PY) -e ".[dev]" && chflags nohidden $$($(PY) -c "import site;print(site.getsitepackages()[0])")/*.pth; \
	else $(PIP) install --python $(PY) -e ".[dev]"; fi

test:
	$(PY) -m pytest

test-all:
	$(PY) -m pytest -m ""

lint:
	$(PY) -m ruff check src tests scripts
	$(PY) -m ruff format --check src tests scripts

format:
	$(PY) -m ruff format src tests scripts
	$(PY) -m ruff check --fix src tests scripts

# Full synthetic pipeline from a clean checkout: data -> train -> calibrate -> eval -> export -> predict
E2E := runs/e2e
e2e:
	$(PY) scripts/make_synthetic_events.py --out $(E2E)/events --n-events 12 --size 64 --frames 30
	.venv/bin/nowcast-train -c configs/train/smoke.yaml data.events_dir=$(E2E)/events train.output_dir=$(E2E)/train registry.root=$(E2E)/artifacts
	.venv/bin/nowcast-calibrate --model $(E2E)/artifacts/nowcast/latest --events $(E2E)/events --split val
	.venv/bin/nowcast-eval --model $(E2E)/artifacts/nowcast/latest --events $(E2E)/events --split test --baselines all --out $(E2E)/reports
	.venv/bin/nowcast-export --model $(E2E)/artifacts/nowcast/latest
	.venv/bin/nowcast-predict --model $(E2E)/artifacts/nowcast/latest --event $$(ls -d $(E2E)/events/*.zarr | head -1) --t0 auto --out $(E2E)/forecast.zarr

clean:
	rm -rf runs reports .pytest_cache .ruff_cache
