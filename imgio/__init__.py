"""
Easy image file reading & writing. Supports PGM/PPM/PFM/PNG/JPG.

Example:
  image, maxval = imgio.imread("foo.png")
  imgio.imwrite("foo.ppm", image, maxval)

https://github.com/toaarnio/imgio
"""

from .imgio import *

__version__ = "0.5.1"
__all__ = ["imread", "imwrite", "selftest"]
