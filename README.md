# imgio

[![Build Status](https://travis-ci.com/toaarnio/imgio.svg?branch=master)](https://travis-ci.com/github/toaarnio/imgio)

Easy image file reading &amp; writing for Python. Tested on Python 3.8.

Supports PGM / PPM / PNM / PFM / PNG / BMP / JPG / INSP / TIFF, plus EXR / BMP / RAW in read-only mode.

**Installing on Linux:**
```
pip install imgio
```

**Installing on Windows:**
```
pip install pipwin
pipwin install imread
pip install imgio
```

**Building & installing from source:**
```
rm -rf build/ dist/
python setup.py build sdist test
pip uninstall imgio
pip install --user dist/*.tar.gz
```

**Releasing to PyPI:**
```
pip install --user --upgrade setuptools wheel twine
python setup.py sdist bdist_wheel
twine upload dist/*
```
