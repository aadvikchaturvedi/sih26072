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

# The whole backend on dummy data (first run trains a small model on synthetic storms, ~2 min).
demo:
	$(PY) -m nowcast_backend.cli demo
