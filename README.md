# imgio

[![Build Status](https://travis-ci.org/toaarnio/imgio.svg?branch=master)](https://travis-ci.org/toaarnio/imgio)

Easy image file reading &amp; writing for Python. Tested on Python 2.7 and 3.5+.

Supports PGM / PPM / PNM / PFM / PNG / JPG / INSP, plus headerless Bayer RAW in read-only mode.

**Installing from PyPI:**
```
pip3 install imgio
```

**Building & installing from source:**
```
python3 setup.py build sdist test
pip3 uninstall imgio
pip3 install --user dist/*.tar.gz
```

**Releasing to PyPI:**
```
pip3 install --user --upgrade setuptools wheel twine
python3 setup.py sdist bdist_wheel
twine upload dist/*
```
