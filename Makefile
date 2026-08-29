.PHONY: help test lint fmt run-web run-demo build-exe clean

help:
	@echo "Targets:"
	@echo "  make test       - run pytest"
	@echo "  make run-web    - start the FastAPI dev server"
	@echo "  make run-demo   - run the CLI demo (needs API key)"
	@echo "  make build-exe  - build standalone .exe (Windows, needs PyInstaller)"
	@echo "  make clean      - remove generated files"

test:
	python -m pytest tests/ -v

run-web:
	uvicorn app.web:app --reload --port 8077

run-demo:
	python -m app.cli demo

run-ab:
	python -m eval.run_ab

build-exe:
	python build_exe.py

clean:
	rm -rf dist/ build/ *.spec
	rm -rf data/memory.db data/memory.db-shm data/memory.db-wal
	rm -rf __pycache__ .pytest_cache .ruff_cache
	find . -name "*.pyc" -delete
