"""
Easy image file reading & writing.

Example:
  image, maxval = imgio.imread("foo.png")
  imgio.imwrite("foo.ppm", image, maxval)

https://github.com/toaarnio/imgio
"""

from .imgio import *

__version__ = "1.4.1"
__all__ = ["imread", "imread_f64", "imread_f32", "imread_f16", "imwrite", "selftest"]
