.PHONY: format lint test type check sample

format:
	black .
	ruff check --fix .

lint:
	ruff check .

type:
	mypy src

test:
	pytest -q --cov=src

check: format lint type test

sample:
	python -m belladonna --help
