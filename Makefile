deps:
	uv sync --extra dev

lint:
	uv run --active ruff check imgio/*.py

test: lint
	uv run --active pytest -v

install:
	rm -rf dist || true
	uv build
	uv pip uninstall --quiet imgio
	uv pip install --quiet dist/*.whl
	unzip -v dist/*.whl
	@python3 -c 'import imgio; print(f"Installed imgio version {imgio.__version__}.")'

release:
	@echo "Publishing is handled by GitHub Actions with PyPI Trusted Publishing."
	@echo "Create a GitHub Release to trigger the publish workflow."

.PHONY: deps lint test install release
