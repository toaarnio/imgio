# imgio

[![Build Status](https://travis-ci.com/toaarnio/imgio.svg?branch=master)](https://travis-ci.com/github/toaarnio/imgio)

Easy image file reading &amp; writing for Python. Tested on Python 3.8.

Supports PGM / PPM / PNM / PFM / PNG / BMP / JPG / INSP / TIFF, plus EXR / BMP / RAW in read-only mode.

**Installing on Linux:**
```
sudo apt install libopenexr-dev
pip install imgio
```

**Installing on Windows:**
```
pip install pipwin
pipwin install imread
pipwin install pyexr
pip install imgio
```

**Building & installing from source:**
```
git clone <this-repository>
cd <this-repository>
make install
```

**Releasing to PyPI:**
```
make release
```
