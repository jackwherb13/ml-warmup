.PHONY: setup test run data plot clean

setup:
	uv sync

test: data
	uv run pytest

run:
	uv run main.py

data:
	uv run python src/download.py

plot: data
	uv run python src/plot_samples.py

clean:
	uv run python -c "import shutil; [shutil.rmtree(d, ignore_errors=True) for d in ('out', 'data')]"