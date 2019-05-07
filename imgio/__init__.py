"""
Easy image file reading & writing. Supports PGM/PPM/PFM/PNG/JPG.

Example:
  image, maxval = imgio.imread("foo.png")
  imgio.imwrite("foo.ppm", image, maxval)

https://github.com/toaarnio/imgio
"""

from .imgio import *

__version__ = "0.4.4"
__all__ = ["imread", "imwrite", "selftest"]
