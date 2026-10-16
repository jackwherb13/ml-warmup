.PHONY: setup test run clean

setup:
	uv sync

test:
	uv run pytest

run:
	uv run main.py

data:
	uv run python src/download.py

clean:
	uv run python -c "import shutil; [shutil.rmtree(d, ignore_errors=True) for d in ('out', 'data')]"