# Project 13 — TTS pipeline orchestration.
# Modules live under src/ and import each other as top-level packages
# (e.g. `from core.logger import logger`), so every run sets PYTHONPATH=src.

# Use the project venv's Python once `make install` has created it, else the system one.
PYTHON ?= $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)
RUN    := PYTHONPATH=src $(PYTHON)

# Overridable knobs:  make generate CKPT=models/finetuned   make eval MOCK=--mock
CKPT  ?= pretrained
LABEL ?= model
MOCK  ?=

.PHONY: help install data prepare train generate eval all smoke lint format clean

help:
	@echo "Targets:"
	@echo "  install   create .venv and install requirements + dev tools"
	@echo "  data      download LJSpeech (~2.6GB)"
	@echo "  prepare   build the test manifest + reference audio"
	@echo "  train     fine-tune SpeechT5 (currently a MOCK)"
	@echo "  generate  synthesize the test set      [CKPT=, MOCK=--mock]"
	@echo "  eval      score generated audio         [LABEL=, MOCK=--mock]"
	@echo "  all       end-to-end on MOCKS (offline) — proves the wiring"
	@echo "  smoke     run the pytest smoke test"
	@echo "  lint      ruff check       format  ruff format       clean  remove outputs"

install:
	$(PYTHON) -m venv .venv
	./.venv/bin/pip install -r requirements.txt -r requirements-dev.txt

data:
	$(RUN) -m data.download

prepare:
	$(RUN) -m data.prepare

train:
	$(RUN) -m model.train

generate:
	$(RUN) -m model.synthesize --checkpoint $(CKPT) $(MOCK)

eval:
	$(RUN) -m evaluation.evaluate --label $(LABEL) $(MOCK)

# End-to-end on mocks: prepare -> train -> generate -> evaluate, fully offline.
all: train
	$(RUN) -m data.prepare --mock
	$(RUN) -m model.synthesize --mock
	$(RUN) -m evaluation.evaluate --mock --label mock

smoke:
	$(RUN) -m pytest

lint:
	ruff check src tests

format:
	ruff format src tests

clean:
	rm -rf audio/generated data/test_manifest.jsonl data/reference models/finetuned
