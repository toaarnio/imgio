deps:
	pip install --quiet hatch pytest

lint:
	ruff check --output-format=full imgio/*.py

clean:
	hatch clean

install: deps lint clean
	hatch build
	pip3 uninstall --quiet --yes imgio
	pip3 install --quiet dist/*.whl
	unzip -v dist/*.whl
	@python3 -c 'import imgio; print(f"Installed imgio version {imgio.__version__}.")'

release: install
	pip3 install --user setuptools wheel twine
	make install
	twine upload dist/*
