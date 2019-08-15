install:
	pip3 uninstall --yes imgio || true
	rm -rf build dist imgio.egg-info || true
	python3 setup.py sdist bdist_wheel
	pip3 install --user dist/*.whl
	@python3 -c 'import imgio; print(f"Installed imgio version {imgio.__version__}.")'

release:
	pip3 install --user setuptools wheel twine
	make install
	twine upload dist/*
