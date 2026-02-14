.PHONY: install test lint migrate dev-api dev-worker synth train modal-train modal-train-elliptic
install:
	pip install -e ".[ml]"
test:
	ruff check apps ml tests && pytest tests apps/api/tests apps/worker/tests ml/tests -v
lint:
	ruff check apps ml tests
migrate:
	supabase db push || true
dev-api:
	cd apps/api && uvicorn api.main:app --reload
dev-worker:
	cd apps/worker && python -m worker.main
synth:
	python scripts/synthetic_scenarios.py --output data/synthetic_events.json
train:
	cd ml && python train.py --data-dir data/synthetic

# Modal remote training (requires: pip install modal, modal setup)
modal-train:
	modal run ml/modal_train.py -- --config ml/configs/hgt_baseline.yaml --data-dir data/synthetic
modal-train-elliptic:
	modal run ml/modal_train_elliptic.py -- --dataset elliptic --model fraud_gt_style --data-dir data/elliptic --output runs/elliptic
