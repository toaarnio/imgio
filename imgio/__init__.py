"""
Easy image file reading & writing.

Example:
  image, maxval = imgio.imread("foo.png")
  imgio.imwrite("foo.ppm", image, maxval)

https://github.com/toaarnio/imgio
"""

from .version import __version__

from .imgio import *

__all__ = ["__version__", "imread", "imread_f64", "imread_f32", "imread_f16", "imwrite", "selftest"]
