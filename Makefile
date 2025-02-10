deps:
	uv pip install --quiet hatch pytest

lint:
	ruff check --output-format=full imgio/*.py

clean:
	hatch clean

install: deps lint clean
	hatch build
	uv pip uninstall --quiet imgio
	uv pip install --quiet dist/*.whl
	unzip -v dist/*.whl
	@python3 -c 'import imgio; print(f"Installed imgio version {imgio.__version__}.")'

qinstall:  # quick & quiet install; wheel only
	@hatch build -t wheel
	@uv pip uninstall --quiet imsize
	@uv pip install --quiet dist/*.whl || true

release: install
	uv pip install setuptools wheel twine
	make install
	twine upload dist/*
