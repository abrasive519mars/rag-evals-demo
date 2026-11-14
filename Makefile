.PHONY: install lint test test-all eval serve

install:
	pip install -e ".[dev,build]"

lint:
	ruff check .

test:
	pytest -m "not slow and not integration" -q

test-all:
	pytest -q

eval:
	python -m ragdemo.evals.run --write-readme

serve:
	uvicorn ragdemo.api:app --host 0.0.0.0 --port 7860
