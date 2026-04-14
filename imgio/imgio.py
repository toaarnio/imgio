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

try:
    import pyexr                  # pip install pyexr + apt install libopenexr-dev
except ModuleNotFoundError:
    print("imgio: OpenEXR (.exr) support disabled. To enable:")
    print("  sudo apt install libopenexr-dev")
    print("  pip install pyexr")
    print()
    pyexr = None

try:
    imageio.plugins.freeimage.download()  # required for 16-bit PNG
    freeimage = True
except OSError:
    print("imgio: FreeImage is malfunctioning, using Pillow instead. Some file formats may not work.")
    print()
    freeimage = False

try:
    # package mode
    from imgio import pnm         # local import: pnm.py
    from imgio import pfm         # local import: pfm.py
except ImportError:
    # stand-alone mode
    import pnm                    # local import: pnm.py
    import pfm                    # local import: pfm.py


######################################################################################
#
#  P U B L I C   A P I
#
######################################################################################


RW_FORMATS = [".pnm", ".pgm", ".ppm", ".pfm", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".insp", ".npy", ".raw", ".exr", ".hdr"]
RO_FORMATS = RW_FORMATS


def imread(filespec, width=None, height=None, bpp=None, raw_header_size=None, verbose=False):
    """
    Reads the given image file from disk and returns it as a NumPy array.
    Grayscale images are returned as 2D arrays of shape H x W, color images
    as 3D arrays of shape H x W x 3.
    """
    ImageIOError.error_message_prefix = "Failed to read %s: "%(repr(filespec))
    ispath = lambda p: isinstance(p, (PurePath, str, bytes))
    _enforce(ispath(filespec), "filespec must be a Path or string, was %s (%s)."%(type(filespec), repr(filespec)))
    _enforce(isinstance(verbose, bool), "verbose must be True or False, was %s (%s)."%(type(verbose), repr(verbose)))
    filespec = Path(filespec)
    filename = filespec.name  # path/to/image.pgm => image.pgm
    basename, extension = os.path.splitext(filename)  # image.pgm => ("image", ".pgm")
    _enforce(len(basename) > 0, "filename `%s` must have at least 1 character + extension."%(filename))
    _enforce(extension.lower() in RO_FORMATS, "unrecognized file extension `%s`."%(extension))
    filetype = extension.lower()
    if filetype == ".raw":
        _enforce(isinstance(bpp, int) and 1 <= bpp <= 16, "bpp must be an integer in [1, 16]; was %s"%(repr(bpp)))
        _enforce(isinstance(width, int) and width >= 1, "width must be an integer >= 1; was %s"%(repr(width)))
        _enforce(isinstance(height, int) and height >= 1, "height must be an integer >= 1; was %s"%(repr(height)))
        frame, maxval = _reraise(lambda: _read_raw(filespec, width, height, bpp, raw_header_size, verbose=verbose))
        return frame, maxval
    elif filetype == ".npy":
        frame = _reraise(lambda: _read_npy(filespec, verbose))
        scale = 1.0
        return frame, scale
    elif filetype == ".pfm":
        frame, scale = _reraise(lambda: pfm.read(filespec, verbose))
        return frame, scale
    elif filetype == ".exr":
        _enforce(pyexr is not None, "OpenEXR support not installed")
        frame = _reraise(lambda: _read_exr(filespec, verbose))
        scale = 1.0
        return frame, scale
    elif filetype in [".pnm", ".pgm", ".ppm"]:
        frame, maxval = _reraise(lambda: pnm.read(filespec, verbose))
        return frame, maxval
    elif filetype == ".hdr":
        frame = _reraise(lambda: iio.imread(filespec, plugin="HDR-FI"))
        maxval = np.max(frame)
    elif filetype in [".jpg", ".jpeg", ".insp"]:
        frame = _reraise(lambda: iio.imread(filespec, plugin="JPEG-FI" if freeimage else "pillow"))
        maxval = np.iinfo(frame.dtype).max
    elif filetype in [".tiff", ".tif"]:
        frame = _reraise(lambda: iio.imread(filespec, plugin="TIFF-FI" if freeimage else "pillow"))
        maxval = np.iinfo(frame.dtype).max
    elif filetype in [".png"]:
        frame = _reraise(lambda: iio.imread(filespec, plugin="PNG-FI" if freeimage else "pillow"))
        maxval = np.iinfo(frame.dtype).max
    elif filetype in [".bmp"]:
        frame = _reraise(lambda: iio.imread(filespec, plugin="BMP-FI" if freeimage else "pillow"))
        maxval = np.iinfo(frame.dtype).max
    else:
        raise ImageIOError("unrecognized file type `%s`."%(filetype))
    h, w = frame.shape[:2]
    c = frame.shape[2] if frame.ndim > 2 else 1
    _print(verbose, "Reading file %s (w=%d, h=%d, c=%d, maxval=%d)"%(filespec, w, h, c, maxval))
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
    ImageIOError.error_message_prefix = "Failed to write %s: "%(repr(filespec))
    _enforce(ispath(filespec), "filespec must be a Path or string, was %s (%s)."%(type(filespec), repr(filespec)))
    _enforce(isinstance(image, np.ndarray), "image must be a NumPy ndarray; was %s."%(type(image)))
    _enforce(image.dtype.char in "BHefd", "image.dtype must be uint{8,16}, or float{16,32,64}; was %s"%(image.dtype))
    _enforce(image.size >= 1, "image must have at least one pixel; had none.")
    _enforce(isinstance(maxval, (float, int)), "maxval must be an integer or a float; was %s."%(repr(maxval)))
    _enforce(isinstance(maxval, int) or image.dtype.char in "efd", "maxval must be an integer in [1, 65535]; was %s."%(repr(maxval)))
    _enforce(1 <= maxval <= 65535 or image.dtype.char in "efd", "maxval must be an integer in [1, 65535]; was %s."%(repr(maxval)))
    _enforce(isinstance(verbose, bool), "verbose must be True or False, was %s (%s)."%(type(verbose), repr(verbose)))
    _disallow(image.ndim not in [2, 3], "image.shape must be (m, n) or (m, n, c); was %s."%(str(image.shape)))
    _disallow(maxval > 255 and image.dtype == np.uint8, "maxval (%d) and image.dtype (%s) are inconsistent."%(maxval, image.dtype))
    _disallow(maxval <= 255 and image.dtype == np.uint16, "maxval (%d) and image.dtype (%s) are inconsistent."%(maxval, image.dtype))
    filespec = Path(filespec)
    filename = os.path.basename(filespec)  # path/to/image.pgm => image.pgm
    basename, extension = os.path.splitext(filename)  # image.pgm => ("image", ".pgm")
    _enforce(len(basename) > 0, "filename `%s` must have at least 1 character + extension."%(filename))
    _enforce(extension.lower() in RW_FORMATS, "unrecognized or unsupported file extension `%s`."%(extension))
    filetype = extension.lower()
    if filetype == ".raw":
        _enforce(packed is False, "packed Bayer RAW images are not yet supported.")
        _enforce(image.ndim == 2, "image.shape must be (m, n) for a Bayer RAW; was %s."%(str(image.shape)))
        _reraise(lambda: _write_raw(filespec, image, maxval, packed, verbose))
    elif filetype == ".pfm":
        _disallow(image.ndim == 3 and image.shape[2] != 3, "image.shape must be (m, n) or (m, n, 3); was %s."%(str(image.shape)))
        _enforce(image.dtype.char in "efd", "image.dtype must be float{16,32,64} for PFM; was %s"%(image.dtype))
        _enforce(maxval >= 0.0, "maxval (scale) must be non-negative; was %s."%(repr(maxval)))
        _reraise(lambda: pfm.write(filespec, image, maxval, little_endian=True, verbose=verbose))
    elif filetype == ".exr":
        _enforce(image.dtype.char in "efd", "image.dtype must be float{16,32,64} for EXR; was %s"%(image.dtype))
        _enforce(pyexr is not None, "OpenEXR support not installed")
        _reraise(lambda: _write_exr(filespec, image, verbose))
    elif filetype == ".npy":
        _reraise(lambda: _write_npy(filespec, image, verbose))
    elif filetype == ".hdr":
        _enforce(image.dtype.char in "f", "image.dtype must be float32 for HDR; was %s"%(image.dtype))
        _enforce(np.min(image) >= 0.0, "negative colors cannot be stored in HDR format; use EXR/PFM/NPY instead")
        _reraise(lambda: iio.imwrite(filespec, image, plugin="HDR-FI"))
    elif filetype in [".pnm", ".pgm", ".ppm"]:
        _reraise(lambda: pnm.write(filespec, image, maxval, verbose))
    elif filetype in [".png", ".tif", ".tiff", ".jpg", ".jpeg", ".insp", ".bmp"]:
        _disallow(image.ndim == 3 and image.shape[2] != 3, "image.shape must be (m, n) or (m, n, 3); was %s."%(str(image.shape)))
        _disallow(maxval not in [255, 65535], "maxval must be 255 or 65535 for JPEG/PNG/BMP/TIFF; was %d."%(maxval))
        if filetype in [".jpg", ".jpeg", ".insp"]:
            _disallow(maxval != 255, "maxval must be 255 for a JPEG; was %d."%(maxval))
            _reraise(lambda: iio.imwrite(filespec, image, plugin="pillow", extension=".jpg", quality=95))
        if filetype in [".tiff", ".tif"]:
            _reraise(lambda: iio.imwrite(filespec, image, plugin="TIFF-FI" if freeimage else "pillow"))
        if filetype in [".png"]:
            _reraise(lambda: iio.imwrite(filespec, image, plugin="PNG-FI" if freeimage else "pillow", compression=1))
        if filetype in [".bmp"]:
            _reraise(lambda: iio.imwrite(filespec, image, plugin="BMP-FI" if freeimage else "pillow"))
        h, w = image.shape[:2]
        c = image.shape[2] if image.ndim > 2 else 1
        _print(verbose, "Writing file %s (w=%d, h=%d, c=%d, maxval=%d)"%(filespec, w, h, c, maxval))
    else:
        raise ImageIOError("unrecognized file type `%s`."%(filetype))


class ImageIOError(RuntimeError):
    """
    A custom exception raised in all error conditions.
    """
    error_message_prefix = ""
    def __init__(self, msg):
        RuntimeError.__init__(self, "%s%s"%(self.error_message_prefix, msg))


######################################################################################
#
#  I N T E R N A L   F U N C T I O N S
#
######################################################################################


def _enforce(expression, error_message_if_false):
    if not expression:
        raise ImageIOError("%s"%(error_message_if_false))

def _disallow(expression, error_message_if_true):
    if expression:
        raise ImageIOError("%s"%(error_message_if_true))

def _reraise(func):
    try:
        return func()
    except Exception as e:
        raise ImageIOError("%s"%(repr(sys.exc_info()[1]))) from e

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

def _read_raw(filespec, width, height, bpp, header_size=None, verbose=False):
    # Warning: hardcoded endianness (x86)
    with open(filespec, "rb") as infile:
        buf = infile.read()
        shape = (height, width)
        maxval = 2 ** bpp - 1
        wordsize = 2 if bpp > 8 else 1
        packed = len(buf) < (width * height * wordsize)
        if not packed:
            if header_size is None:
                header_size = len(buf) - (width * height * wordsize)
            fs, w, h, hdrsz = filespec, width, height, header_size
            _print(verbose, "Reading raw Bayer file %s (w=%d, h=%d, maxval=%d, header=%d)"%(fs, w, h, maxval, hdrsz))
            dtype = "<u2" if bpp > 8 else np.uint8
            pixels = np.frombuffer(buf, dtype, count=width * height, offset=hdrsz)
            pixels = pixels.reshape(shape).astype(np.uint8 if bpp <= 8 else np.uint16)
        else:
            if bpp != 10:
                raise ImageIOError(f"{bpp}-bit packed RAW reading not implemented yet!")
            nbytes = width * height * bpp // 8
            if header_size is None:
                header_size = len(buf) - nbytes
            fs, w, h, hdrsz = filespec, width, height, header_size
            _print(verbose, "Reading packed raw Bayer file %s (w=%d, h=%d, maxval=%d, header=%d)"%(fs, w, h, maxval, hdrsz))
            data = np.frombuffer(buf, dtype=np.uint8, count=nbytes, offset=hdrsz)
            pixels = _read_uint10(data, lsb_first=True)
            pixels = pixels.reshape(height, width)
        return pixels, maxval

def _read_uint10(data, lsb_first):
    # 5 bytes contain 4 10-bit pixels (5 * 8 == 4 * 10 == 40)
    b1, b2, b3, b4, b5 = data.astype(np.uint16).reshape(-1, 5).T
    if lsb_first:
        # byte0: a7 a6 a5 a4 a3 a2 a1 a0
        # byte1: b5 b4 b3 b2 b1 b0 a9 a8
        # byte2: c3 c2 c1 c0 b9 b8 b7 b6
        # byte3: d1 d0 c9 c8 c7 c6 c5 c4
        # byte4: d9 d8 d7 d6 d5 d4 d3 d2
        o1 = ((b2 % 4) << 8) + b1
        o2 = ((b3 % 16) << 6) + (b2 >> 2)
        o3 = ((b4 % 64) << 4) + (b3 >> 4)
        o4 = (b5 << 2) + (b4 >> 6)
    else:
        o1 = (b1 << 2) + (b2 >> 6)
        o2 = ((b2 % 64) << 4) + (b3 >> 4)
        o3 = ((b3 % 16) << 6) + (b4 >> 2)
        o4 = ((b4 % 4) << 8) + b5
    unpacked = np.c_[o1, o2, o3, o4].ravel()
    return unpacked

def _write_raw(filespec, image, _maxval, _pack=False, _verbose=False):
    # Warning: hardcoded endianness (x86)
    with open(filespec, "wb") as outfile:
        image = image.copy(order="C")  # ensure x86 byte order
        outfile.write(image)
