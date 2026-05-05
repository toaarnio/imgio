"""
Easy image file reading & writing.

Example:
  image, maxval = imgio.imread("foo.png")
  imgio.imwrite("foo.ppm", image, maxval)
"""

import os                         # standard library
import sys                        # standard library
from pathlib import Path          # standard library
from pathlib import PurePath      # standard library

import numpy as np                # pip install numpy
import imageio                    # pip install imageio
import imageio.v3 as iio          # pip install imageio

from . import pnm                 # local import: pnm.py
from . import pfm                 # local import: pfm.py
from . import raw                 # local import: raw.py

try:
    import pyexr                  # pip install pyexr + apt install libopenexr-dev
except ModuleNotFoundError:
    print("imgio: OpenEXR (.exr) support disabled. To enable:")
    print("  sudo apt install libopenexr-dev")
    print("  pip install pyexr")
    print()
    pyexr = None

def _init_freeimage():
    if sys.platform == "darwin":
        print("imgio: FreeImage disabled on macOS, using Pillow instead. Some file formats may not work.")
        print()
        return False
    try:
        imageio.plugins.freeimage.download()  # required for 16-bit PNG
    except OSError:
        print("imgio: FreeImage is malfunctioning, using Pillow instead. Some file formats may not work.")
        print()
        return False
    else:
        return True


freeimage = _init_freeimage()


######################################################################################
#
#  P U B L I C   A P I
#
######################################################################################


RW_FORMATS = [".pnm", ".pgm", ".ppm", ".pfm", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".insp", ".npy", ".raw", ".exr", ".hdr"]
RO_FORMATS = RW_FORMATS + [".mipi"]


def imread(filespec: str | Path,
           verbose: bool = False) -> [np.ndarray, float]:
    """
    Reads the given image file from disk and returns it as a NumPy array.
    Grayscale images are returned as 2D arrays of shape H x W, color images
    as 3D arrays of shape H x W x 3.

    :param filespec: file to load
    :param verbose: True to print file properties on the console
    :return frame: the image as a 2D/3D array of dtype uint8/16 or float16/32
    :return maxval: the nominal maximum representable value of the frame
    """
    prefix = "Failed to read %s: "%(repr(filespec))
    ispath = lambda p: isinstance(p, (PurePath, str, bytes))
    _enforce(ispath(filespec), "filespec must be a Path or string, was %s (%s)."%(type(filespec), repr(filespec)), prefix)
    _enforce(isinstance(verbose, bool), "verbose must be True or False, was %s (%s)."%(type(verbose), repr(verbose)), prefix)
    filespec = Path(filespec)
    filename = filespec.name  # path/to/image.pgm => image.pgm
    basename, extension = os.path.splitext(filename)  # image.pgm => ("image", ".pgm")
    _enforce(len(basename) > 0, "filename `%s` must have at least 1 character + extension."%(filename), prefix)
    _enforce(extension.lower() in RO_FORMATS, "unrecognized file extension `%s`."%(extension), prefix)
    filetype = extension.lower()
    if filetype in [".raw", ".mipi"]:
        raise ImageIOError("use .rawread() instead of .imread() to load sensor raw files", prefix)
    if filetype == ".npy":
        frame = _reraise(lambda: _read_npy(filespec, verbose), prefix)
        scale = 1.0
        return frame, scale
    elif filetype == ".pfm":
        frame, scale = _reraise(lambda: pfm.read(filespec, verbose), prefix)
        return frame, scale
    elif filetype == ".exr":
        _enforce(pyexr is not None, "OpenEXR support not installed", prefix)
        frame = _reraise(lambda: _read_exr(filespec, verbose), prefix)
        scale = 1.0
        return frame, scale
    elif filetype in [".pnm", ".pgm", ".ppm"]:
        frame, maxval = _reraise(lambda: pnm.read(filespec, verbose), prefix)
        return frame, maxval
    elif filetype == ".hdr":
        frame = _reraise(lambda: iio.imread(filespec, plugin="HDR-FI"), prefix)
        maxval = np.max(frame)
    elif filetype in [".jpg", ".jpeg", ".insp"]:
        frame = _reraise(lambda: iio.imread(filespec, plugin="JPEG-FI" if freeimage else "pillow"), prefix)
        maxval = np.iinfo(frame.dtype).max
    elif filetype in [".tiff", ".tif"]:
        frame = _reraise(lambda: iio.imread(filespec, plugin="TIFF-FI" if freeimage else "pillow"), prefix)
        maxval = np.iinfo(frame.dtype).max
    elif filetype in [".png"]:
        frame = _reraise(lambda: iio.imread(filespec, plugin="PNG-FI" if freeimage else "pillow"), prefix)
        maxval = np.iinfo(frame.dtype).max
    elif filetype in [".bmp"]:
        frame = _reraise(lambda: iio.imread(filespec, plugin="BMP-FI" if freeimage else "pillow"), prefix)
        maxval = np.iinfo(frame.dtype).max
    else:
        raise ImageIOError("unrecognized file type `%s`."%(filetype), prefix)
    h, w = frame.shape[:2]
    c = frame.shape[2] if frame.ndim > 2 else 1
    _print(verbose, "Reading file %s (w=%d, h=%d, c=%d, maxval=%d)"%(filespec, w, h, c, maxval))
    return frame, maxval


def rawread(filespec: str | Path,
            width: int,
            height: int,
            bpp: int,
            stride: int | None = None,
            packing: str | None = None,
            header_size: int | None = None,
            verbose: bool = False) -> [np.ndarray, int]:
    """
    Reads the given sensor raw file from disk and unpacks it into a uint16 array.

    :param filespec: file to load
    :param width: width of the frame in pixels
    :param height: height of the frame in pixels
    :param bpp: bits per pixel; must be in [10, 12, 14, 16]
    :param stride: row length in bytes; can be greater than width * bpp / 8
    :param header_size: number of header bytes to skip (not decoded)
    :param packing: bit packing mode; must be unpacked|plain|mipi|None
    :param verbose: True to print file properties on the console
    :return frame: the raw frame as a (H, W) array of dtype uint16
    :return maxval: the nominal maximum representable value of the frame
    """
    prefix = "Failed to read %s: "%(repr(filespec))
    _enforce(isinstance(bpp, int) and bpp in [10, 12, 14, 16], "bpp must be in [10, 12, 14, 16]; was %s"%(repr(bpp)), prefix)
    _enforce(isinstance(width, int) and width % 2 == 0, "width must be an integer multiple of 2; was %s"%(repr(width)), prefix)
    _enforce(isinstance(height, int) and height >= 1, "height must be an integer and >= 1; was %s"%(repr(height)), prefix)
    _enforce(stride is None or stride >= width, "stride must be None or >= width; was %s"%(repr(stride)), prefix)
    _enforce(packing == "unpacked" or bpp in [10, 12], f"{bpp}-bit packed RAW reading is not supported", prefix)

    data = _reraise(lambda: np.fromfile(filespec, dtype=np.uint8), prefix)
    header_size = header_size or 0
    is_packed = data.size < width * height * 2 + header_size

    _enforce(not (packing == "unpacked" and is_packed), f"not enough bytes for {width} x {height} pixels as unpacked raw", prefix)

    if packing is None:
        packing = "plain" if is_packed else "unpacked"

    if stride is None:
        if packing in ["mipi", "plain"]:
            # round up to nearest 16-byte boundary
            stride = int(np.ceil(width * bpp / 8 / 16) * 16)
        else:  # unpacked
            # no need to round up, any alignment is ok
            stride = width * 2

    nbytes = stride * height + header_size
    bytedepth = 2 if packing == "unpacked" else bpp / 8
    _enforce(data.size >= nbytes, f"expected at least {stride} * {height} + {header_size} = {nbytes} bytes, got {data.size}", prefix)

    data = data[header_size:nbytes]  # trim header & footer bytes
    data = data.reshape(height, stride)
    data = data[:, :int(width * bytedepth)]  # trim right edge padding
    data = data.ravel()
    frame = _reraise(lambda: raw.decode(data, bpp, packing), prefix)
    frame = frame.reshape(-1, width)  # allow extra rows at bottom
    frame = frame[:height, :width]  # trim bottom edge padding
    maxval = 2 ** bpp - 1
    h, w = frame.shape[:2]
    _print(verbose, "Reading raw file %s (w=%d, h=%d, maxval=%d)"%(filespec, w, h, maxval))
    return frame, maxval


def imread_f64(filespec, **kwargs):
    """
    Reads the given image file from disk and returns it as a float64 array,
    normalized by maxval. This is a shorthand for imread() followed by type
    conversion and division by maxval.
    """
    img, maxval = imread(filespec, **kwargs)
    img = img.astype(np.float64) / maxval
    return img


def imread_f32(filespec, **kwargs):
    """
    Reads the given image file from disk and returns it as a float32 array,
    normalized by maxval. This is a shorthand for imread() followed by type
    conversion and division by maxval.
    """
    img, maxval = imread(filespec, **kwargs)
    img = img.astype(np.float32) / maxval
    return img


def imread_f16(filespec, **kwargs):
    """
    Reads the given image file from disk and returns it as a float16 array,
    normalized by maxval. This is a shorthand for imread() followed by type
    conversion and division by maxval.
    """
    img, maxval = imread(filespec, **kwargs)
    img = img.astype(np.float32) / maxval
    img = img.astype(np.float16)
    return img


def imwrite(filespec, image, maxval=255, packed=False, verbose=False):
    """
    Writes the given image to the given file, returns nothing. Grayscale images
    are expected to be provided as NumPy arrays with shape H x W, color images
    with shape H x W x C. Metadata, alpha channels, etc. are not supported.
    """
    ispath = lambda p: isinstance(p, (PurePath, str, bytes))
    prefix = "Failed to write %s: "%(repr(filespec))
    _enforce(ispath(filespec), "filespec must be a Path or string, was %s (%s)."%(type(filespec), repr(filespec)), prefix)
    _enforce(isinstance(image, np.ndarray), "image must be a NumPy ndarray; was %s."%(type(image)), prefix)
    _enforce(image.dtype.char in "BHefd", "image.dtype must be uint{8,16}, or float{16,32,64}; was %s"%(image.dtype), prefix)
    _enforce(image.size >= 1, "image must have at least one pixel; had none.", prefix)
    _enforce(isinstance(maxval, (float, int)), "maxval must be an integer or a float; was %s."%(repr(maxval)), prefix)
    _enforce(isinstance(maxval, int) or image.dtype.char in "efd", "maxval must be an integer in [1, 65535]; was %s."%(repr(maxval)), prefix)
    _enforce(1 <= maxval <= 65535 or image.dtype.char in "efd", "maxval must be an integer in [1, 65535]; was %s."%(repr(maxval)), prefix)
    _enforce(isinstance(verbose, bool), "verbose must be True or False, was %s (%s)."%(type(verbose), repr(verbose)), prefix)
    _disallow(image.ndim not in [2, 3], "image.shape must be (m, n) or (m, n, c); was %s."%(str(image.shape)), prefix)
    _disallow(maxval > 255 and image.dtype == np.uint8, "maxval (%d) and image.dtype (%s) are inconsistent."%(maxval, image.dtype), prefix)
    _disallow(maxval <= 255 and image.dtype == np.uint16, "maxval (%d) and image.dtype (%s) are inconsistent."%(maxval, image.dtype), prefix)
    filespec = Path(filespec)
    filename = os.path.basename(filespec)  # path/to/image.pgm => image.pgm
    basename, extension = os.path.splitext(filename)  # image.pgm => ("image", ".pgm")
    _enforce(len(basename) > 0, "filename `%s` must have at least 1 character + extension."%(filename), prefix)
    _enforce(extension.lower() in RW_FORMATS, "unrecognized or unsupported file extension `%s`."%(extension), prefix)
    filetype = extension.lower()
    if filetype == ".raw":
        _enforce(packed is False, "packed Bayer RAW images are not yet supported.", prefix)
        _enforce(image.ndim == 2, "image.shape must be (m, n) for a Bayer RAW; was %s."%(str(image.shape)), prefix)
        _reraise(lambda: raw.write(filespec, image, maxval, packed, verbose), prefix)
    elif filetype == ".pfm":
        _disallow(image.ndim == 3 and image.shape[2] != 3, "image.shape must be (m, n) or (m, n, 3); was %s."%(str(image.shape)), prefix)
        _enforce(image.dtype.char in "efd", "image.dtype must be float{16,32,64} for PFM; was %s"%(image.dtype), prefix)
        _enforce(maxval >= 0.0, "maxval (scale) must be non-negative; was %s."%(repr(maxval)), prefix)
        _reraise(lambda: pfm.write(filespec, image, maxval, little_endian=True, verbose=verbose), prefix)
    elif filetype == ".exr":
        _enforce(image.dtype.char in "efd", "image.dtype must be float{16,32,64} for EXR; was %s"%(image.dtype), prefix)
        _enforce(pyexr is not None, "OpenEXR support not installed", prefix)
        _reraise(lambda: _write_exr(filespec, image, verbose), prefix)
    elif filetype == ".npy":
        _reraise(lambda: _write_npy(filespec, image, verbose), prefix)
    elif filetype == ".hdr":
        _enforce(image.dtype.char in "f", "image.dtype must be float32 for HDR; was %s"%(image.dtype), prefix)
        _enforce(np.min(image) >= 0.0, "negative colors cannot be stored in HDR format; use EXR/PFM/NPY instead", prefix)
        _reraise(lambda: iio.imwrite(filespec, image, plugin="HDR-FI"), prefix)
    elif filetype in [".pnm", ".pgm", ".ppm"]:
        _reraise(lambda: pnm.write(filespec, image, maxval, verbose), prefix)
    elif filetype in [".png", ".tif", ".tiff", ".jpg", ".jpeg", ".insp", ".bmp"]:
        _disallow(image.ndim == 3 and image.shape[2] != 3, "image.shape must be (m, n) or (m, n, 3); was %s."%(str(image.shape)), prefix)
        _disallow(maxval not in [255, 65535], "maxval must be 255 or 65535 for JPEG/PNG/BMP/TIFF; was %d."%(maxval), prefix)
        if filetype in [".jpg", ".jpeg", ".insp"]:
            _disallow(maxval != 255, "maxval must be 255 for a JPEG; was %d."%(maxval), prefix)
            _reraise(lambda: iio.imwrite(filespec, image, plugin="pillow", extension=".jpg", quality=95), prefix)
        if filetype in [".tiff", ".tif"]:
            _reraise(lambda: iio.imwrite(filespec, image, plugin="TIFF-FI" if freeimage else "pillow"), prefix)
        if filetype in [".png"]:
            _reraise(lambda: iio.imwrite(filespec, image, plugin="PNG-FI" if freeimage else "pillow", compression=1), prefix)
        if filetype in [".bmp"]:
            _reraise(lambda: iio.imwrite(filespec, image, plugin="BMP-FI" if freeimage else "pillow"), prefix)
        h, w = image.shape[:2]
        c = image.shape[2] if image.ndim > 2 else 1
        _print(verbose, "Writing file %s (w=%d, h=%d, c=%d, maxval=%d)"%(filespec, w, h, c, maxval))
    else:
        raise ImageIOError("unrecognized file type `%s`."%(filetype), prefix)


class ImageIOError(RuntimeError):
    """
    A custom exception raised in all error conditions.
    """
    def __init__(self, msg, prefix=""):
        RuntimeError.__init__(self, "%s%s"%(prefix, msg))


######################################################################################
#
#  I N T E R N A L   F U N C T I O N S
#
######################################################################################


def _enforce(expression, error_message_if_false, prefix=""):
    if not expression:
        raise ImageIOError("%s"%(error_message_if_false), prefix)

def _disallow(expression, error_message_if_true, prefix=""):
    if expression:
        raise ImageIOError("%s"%(error_message_if_true), prefix)

def _reraise(func, prefix=""):
    try:
        return func()
    except Exception as e:
        raise ImageIOError("%s"%(repr(sys.exc_info()[1])), prefix) from e

def _print(verbose, *args, **kwargs):
    if verbose:
        print(*args, **kwargs)

def _read_exr(filespec, verbose=False):
    exr = pyexr.open(str(filespec))
    precision = list(exr.channel_precision.values())[0]  # noqa: RUF015
    data = exr.get(precision=precision)
    must_squeeze = (data.ndim > 2 and data.shape[2] == 1)
    data = data.squeeze(axis=2) if must_squeeze else data
    w, h, ch, dt = exr.width, exr.height, len(exr.channels), data.dtype
    _print(verbose, "Reading OpenEXR file %s (w=%d, h=%d, c=%d, %s)"%(filespec, w, h, ch, dt))
    return data

def _write_exr(filespec, image, verbose=False):
    h, w = image.shape[:2]
    ch = image.shape[2] if image.ndim == 3 else 1
    dt = pyexr.HALF if image.dtype == np.float16 else pyexr.FLOAT
    channels = [f"ch{idx:02d}" for idx in range(ch)] if ch >= 5 else None
    pyexr.write(str(filespec), image, precision=dt, channel_names=channels)
    _print(verbose, "Writing OpenEXR file %s (w=%d, h=%d, c=%d, %s)"%(filespec, w, h, ch, image.dtype))

def _read_npy(filespec, verbose=False):
    data = np.load(filespec)
    _enforce(data.ndim in [2, 3], "NumPy file %s image has unsupported shape %s"%(filespec, str(data.shape)))
    h, w = data.shape[:2]
    ch = data.shape[2] if data.ndim == 3 else 1
    _print(verbose, "Reading NumPy file %s (w=%d, h=%d, c=%d, %s)"%(filespec, w, h, ch, data.dtype))
    return data

def _write_npy(filespec, image, verbose=False):
    _enforce(image.ndim in [2, 3], "image.shape must be or (m, n) or (m, n, c) for .npy; was %s."%(str(image.shape)))
    h, w = image.shape[:2]
    ch = image.shape[2] if image.ndim == 3 else 1
    _print(verbose, "Writing NumPy file %s (w=%d, h=%d, c=%d, %s)"%(filespec, w, h, ch, image.dtype))
    np.save(filespec, image)
