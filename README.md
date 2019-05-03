# imgio

[![Build Status](https://travis-ci.org/toaarnio/imgio.svg?branch=master)](https://travis-ci.org/toaarnio/imgio)

Easy image file reading &amp; writing for Python. Tested on Python 2.7 and 3.5+.

Supports PGM / PPM / PNM / PFM / PNG / JPG / INSP, plus headerless Bayer RAW in read-only mode.

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
