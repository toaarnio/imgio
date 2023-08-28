# imgio

[![Build Status](https://travis-ci.com/toaarnio/imgio.svg?branch=master)](https://travis-ci.com/github/toaarnio/imgio)

Easy image file reading &amp; writing for Python. Tested on Python 3.8+.

Supported file formats (read/write):

```
   .pnm
   .pgm
   .ppm
   .pfm
   .png
   .jpg
   .jpeg
   .bmp
   .tif
   .tiff
   .insp
   .npy
   .exr
   .raw
   .hdr
```

**Installing on Linux:**
```
sudo apt install libopenexr-dev
pip install imgio
```

**Installing on Windows:**
```
pip install pipwin
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
