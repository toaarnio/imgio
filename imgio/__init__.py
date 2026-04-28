"""
Easy image file reading & writing.

Example:
  image, maxval = imgio.imread("foo.png")
  imgio.imwrite("foo.ppm", image, maxval)

https://github.com/toaarnio/imgio
"""
from .version import __version__

from .imgio import imread
from .imgio import imread_f16
from .imgio import imread_f32
from .imgio import imread_f64
from .imgio import imwrite
from .imgio import rawread
from .imgio import ImageIOError
from .imgio import RO_FORMATS
from .imgio import RW_FORMATS


__all__ = ["RO_FORMATS",
           "RW_FORMATS",
           "ImageIOError",
           "__version__",
           "imread",
           "imread_f16",
           "imread_f32",
           "imread_f64",
           "imwrite",
           "rawread"]
