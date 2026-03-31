.PHONY: install test lint build run train clean

install:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check src tests

build:
	docker build -t churn-prediction:latest .

run:
	docker-compose up

train:
	python -m src.ml.train --output-dir models/v1

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; \
	find . -name "*.pyc" -delete 2>/dev/null; \
	rm -rf .pytest_cache .coverage htmlcov dist build; \
	echo "Cleaned."
