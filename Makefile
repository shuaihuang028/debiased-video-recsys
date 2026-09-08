PYTHON ?= .venv/bin/python

.PHONY: setup test lint download-data validate-data split-data build-features build-random-features eval-popularity eval-retrieval prepare-item-features build-training-examples train-two-tower encode-two-tower eval-two-tower train-ranker train-multitask-rankers eval-randomized train-debiased-ranker train-random-adaptation prepare-serving-snapshot serve serve-redis benchmark

setup:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check src tests scripts

download-data:
	bash scripts/download_kuairand_1k.sh

validate-data:
	$(PYTHON) -m short_video_recsys.cli validate-data --config configs/data.yaml

split-data:
	$(PYTHON) -m short_video_recsys.cli split-data --config configs/data.yaml

build-features:
	$(PYTHON) -m short_video_recsys.cli build-features --config configs/data.yaml

build-random-features:
	$(PYTHON) -m short_video_recsys.cli build-random-features --config configs/data.yaml

eval-popularity:
	$(PYTHON) -m short_video_recsys.cli eval-popularity --config configs/retrieval_baselines.yaml

eval-retrieval:
	$(PYTHON) -m short_video_recsys.cli eval-retrieval --config configs/retrieval_baselines.yaml

prepare-item-features:
	$(PYTHON) -m short_video_recsys.cli prepare-item-features --config configs/two_tower.yaml

build-training-examples:
	$(PYTHON) -m short_video_recsys.cli build-training-examples --config configs/two_tower.yaml

train-two-tower:
	$(PYTHON) -m short_video_recsys.cli train-two-tower --config configs/two_tower.yaml

encode-two-tower:
	$(PYTHON) -m short_video_recsys.cli encode-two-tower --config configs/two_tower.yaml

eval-two-tower: encode-two-tower
	$(PYTHON) -m short_video_recsys.cli eval-two-tower --config configs/two_tower.yaml

train-ranker:
	$(PYTHON) -m short_video_recsys.cli train-ranker --config configs/ranker.yaml

train-multitask-rankers:
	$(PYTHON) -m short_video_recsys.cli train-multitask-rankers --config configs/multitask.yaml

eval-randomized:
	$(PYTHON) -m short_video_recsys.cli eval-randomized --config configs/debias.yaml

train-debiased-ranker:
	$(PYTHON) -m short_video_recsys.cli train-debiased-ranker --config configs/debias.yaml

train-random-adaptation:
	$(PYTHON) -m short_video_recsys.cli train-random-adaptation --config configs/debias.yaml

prepare-serving-snapshot:
	$(PYTHON) -m short_video_recsys.cli prepare-serving-snapshot --config configs/serving.yaml

serve:
	$(PYTHON) -m short_video_recsys.cli serve --config configs/serving.yaml

serve-redis:
	RECSYS_FEATURE_STORE=redis $(PYTHON) -m short_video_recsys.cli serve --config configs/serving.yaml

benchmark:
	$(PYTHON) scripts/benchmark_serving.py --config configs/serving.yaml
