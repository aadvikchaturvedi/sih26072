# The backend shares the ML package's virtualenv (it imports nowcast_ml).
PY ?= ../ml/.venv/bin/python
PIP ?= uv pip

.PHONY: install test test-all lint format serve demo

# macOS can re-apply the "hidden" flag that makes Python skip the editable-install .pth (see
# ml/Makefile), so src/ is also added through the venv's sitecustomize.py. `make install` in
# ../ml rewrites that file: run this target again afterwards.
install:
	$(PIP) install --python $(PY) -e ".[dev]"
	@if [ "$$(uname)" = "Darwin" ]; then SP=$$($(PY) -c "import site;print(site.getsitepackages()[0])") && \
		chflags nohidden $$SP/*.pth && \
		{ grep -qs "$(CURDIR)/src" $$SP/sitecustomize.py || \
		printf 'import sys\nif "%s" not in sys.path:\n    sys.path.insert(0, "%s")\n' "$(CURDIR)/src" "$(CURDIR)/src" >> $$SP/sitecustomize.py; }; fi

test:
	$(PY) -m pytest

test-all:
	$(PY) -m pytest -m ""

lint:
	$(PY) -m ruff check src tests
	$(PY) -m ruff format --check src tests

format:
	$(PY) -m ruff format src tests
	$(PY) -m ruff check --fix src tests

serve:
	$(PY) -m nowcast_backend.cli serve

# Replays one archived event through the full chain (ingest -> forecast -> cells -> warnings),
# one frame every 5 s. Defaults use the outputs of `make e2e` in ../ml.
MODEL ?= ../ml/runs/e2e/artifacts/nowcast/latest
EVENT ?= ../ml/runs/e2e/events/synth_00000.zarr
demo:
	NOWCAST_MODEL_PATH=$(MODEL) NOWCAST_SOURCE=replay NOWCAST_SOURCE_PATH=$(EVENT) \
	NOWCAST_CLOCK=data NOWCAST_POLL_SECONDS=5 NOWCAST_DATA_DIR=var/demo $(PY) -m nowcast_backend.cli serve
