"""
Easy image file reading & writing. Supports PGM/PPM/PNM/PFM/PNG/JPG/TIFF/EXR/RAW.

Example:
  image, maxval = imgio.imread("foo.png")
  imgio.imwrite("foo.ppm", image, maxval)

https://github.com/toaarnio/imgio
"""

from .imgio import *

__version__ = "0.7.0"
__all__ = ["imread", "imwrite", "selftest"]
