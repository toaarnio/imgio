#!/usr/bin/python3 -B

"""
Easy image file reading & writing.

Example:
  image, maxval = imgio.imread("foo.png")
  imgio.imwrite("foo.ppm", image, maxval)
"""

import os                         # standard library
import sys                        # standard library
import unittest                   # standard library

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
    # package mode
    from imgio import pnm         # local import: pnm.py
    from imgio import pfm         # local import: pfm.py
except ImportError:
    # stand-alone mode
    import pnm                    # local import: pnm.py
    import pfm                    # local import: pfm.py


imageio.plugins.freeimage.download()  # required for 16-bit PNG


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
    _enforce(isinstance(filespec, str), "filespec must be a string, was %s (%s)."%(type(filespec), repr(filespec)))
    _enforce(isinstance(verbose, bool), "verbose must be True or False, was %s (%s)."%(type(verbose), repr(verbose)))
    filename = os.path.basename(filespec)             # "path/image.pgm" => "image.pgm"
    basename, extension = os.path.splitext(filename)  # "image.pgm" => ("image", ".pgm")
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
        frame, maxval = _reraise(lambda: _read_npy(filespec, verbose))
        return frame, maxval
    elif filetype == ".pfm":
        frame, scale = _reraise(lambda: pfm.read(filespec, verbose))
        return frame, scale
    elif filetype == ".exr":
        _enforce(pyexr is not None, "OpenEXR support not installed")
        frame, maxval = _reraise(lambda: _read_exr(filespec, verbose))
        return frame, maxval
    elif filetype in [".pnm", ".pgm", ".ppm"]:
        frame, maxval = _reraise(lambda: pnm.read(filespec, verbose))
        return frame, maxval
    elif filetype == ".hdr":
        frame = _reraise(lambda: iio.imread(filespec, plugin="HDR-FI"))
        maxval = np.max(frame)
    elif filetype in [".jpg", ".jpeg", ".insp"]:
        frame = _reraise(lambda: iio.imread(filespec, plugin="JPEG-FI"))
        maxval = np.iinfo(frame.dtype).max
    elif filetype in [".tiff", ".tif"]:
        frame = _reraise(lambda: iio.imread(filespec, plugin="TIFF-FI"))
        maxval = np.iinfo(frame.dtype).max
    elif filetype in [".png"]:
        frame = _reraise(lambda: iio.imread(filespec, plugin="PNG-FI"))
        maxval = np.iinfo(frame.dtype).max
    elif filetype in [".bmp"]:
        frame = _reraise(lambda: iio.imread(filespec, plugin="BMP-FI"))
        maxval = np.iinfo(frame.dtype).max
    else:
        raise ImageIOError("unrecognized file type `%s`."%(filetype))
    h, w = frame.shape[:2]
    c = frame.shape[2] if frame.ndim > 2 else 1
    _print(verbose, "Reading file %s (w=%d, h=%d, c=%d, maxval=%d)"%(filespec, w, h, c, maxval))
    return frame, maxval

def imwrite(filespec, image, maxval=255, packed=False, verbose=False):
    """
    Writes the given image to the given file, returns nothing. Grayscale images
    are expected to be provided as NumPy arrays with shape H x W, color images
    with shape H x W x C. Metadata, alpha channels, etc. are not supported.
    """
    ImageIOError.error_message_prefix = "Failed to write %s: "%(repr(filespec))
    _enforce(isinstance(filespec, str), "filespec must be a string, was %s (%s)."%(type(filespec), repr(filespec)))
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
    filename = os.path.basename(filespec)            # "path/image.pgm" => "image.pgm"
    basename, extension = os.path.splitext(filename)  # "image.pgm" => ("image", ".pgm")
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
            _reraise(lambda: iio.imwrite(filespec, image, plugin="TIFF-FI"))
        if filetype in [".png"]:
            _reraise(lambda: iio.imwrite(filespec, image, plugin="PNG-FI"))
        if filetype in [".bmp"]:
            _reraise(lambda: iio.imwrite(filespec, image, plugin="BMP-FI"))
        h, w = image.shape[:2]
        c = image.shape[2] if image.ndim > 2 else 1
        _print(verbose, "Writing file %s (w=%d, h=%d, c=%d, maxval=%d)"%(filespec, w, h, c, maxval))
    else:
        raise ImageIOError("unrecognized file type `%s`."%(filetype))

def selftest():
    """
    Runs the full suite of unit tests that comes bundled with the package.
    """
    print("--" * 35)
    suite = unittest.TestLoader().loadTestsFromTestCase(_TestImgIo)
    unittest.TextTestRunner(verbosity=0, failfast=True).run(suite)

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
    except Exception as e:  # noqa: BLE001 [blind-except]
        raise ImageIOError("%s"%(repr(sys.exc_info()[1]))) from e

def _print(verbose, *args, **kwargs):
    if verbose:
        print(*args, **kwargs)

def _read_exr(filespec, verbose=False):
    exr = pyexr.open(filespec)
    precision = list(exr.channel_precision.values())[0]  # noqa: RUF015
    data = exr.get(precision=precision)
    maxval = np.max(data)
    must_squeeze = (data.ndim > 2 and data.shape[2] == 1)
    data = data.squeeze(axis=2) if must_squeeze else data
    w, h, ch, dt = exr.width, exr.height, len(exr.channels), data.dtype
    _print(verbose, "Reading OpenEXR file %s (w=%d, h=%d, c=%d, %s)"%(filespec, w, h, ch, dt))
    return data, maxval

def _write_exr(filespec, image, verbose=False):
    h, w = image.shape[:2]
    ch = image.shape[2] if image.ndim == 3 else 1
    dt = pyexr.HALF if image.dtype == np.float16 else pyexr.FLOAT
    channels = [f"ch{idx:02d}" for idx in range(ch)] if ch >= 5 else None
    pyexr.write(filespec, image, precision=dt, channel_names=channels)
    _print(verbose, "Writing OpenEXR file %s (w=%d, h=%d, c=%d, %s)"%(filespec, w, h, ch, image.dtype))

def _read_npy(filespec, verbose=False):
    data = np.load(filespec)
    _enforce(data.ndim in [2, 3], "NumPy file %s image has unsupported shape %s"%(filespec, str(data.shape)))
    maxval = np.max(data)
    h, w = data.shape[:2]
    ch = data.shape[2] if data.ndim == 3 else 1
    _print(verbose, "Reading NumPy file %s (w=%d, h=%d, c=%d, %s)"%(filespec, w, h, ch, data.dtype))
    return data, maxval

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
        image = image.copy(order='C')  # ensure x86 byte order
        outfile.write(image)


######################################################################################
#
#  U N I T   T E S T S
#
######################################################################################


class _TestImgIo(unittest.TestCase):

    TEST_SHAPES_1 = ((1, 1), (7, 11))
    TEST_SHAPES_3 = ((1, 1, 3), (7, 11, 3), (123, 321, 3))
    TEST_SHAPES_N = ((1, 1, 2), (1, 1, 9), (9, 13, 31))

    TEST_SHAPES_ALL = TEST_SHAPES_1 + TEST_SHAPES_3 + TEST_SHAPES_N

    def assertRaisesRegex(self, expected_exception, expected_regex, *args, **kwargs):  # noqa: N802 [invalid-function-name]
        """
        Checks that the correct type of exception is raised, and that the exception
        message matches the given regular expression. Also prints out the message
        for visual inspection.
        """
        try:  # check the type of exception first
            unittest.TestCase.assertRaises(self, expected_exception, *args, **kwargs)
        except Exception as e:  # noqa: BLE001 [blind-except]
            raised_name = sys.exc_info()[0].__name__
            expected_name = expected_exception.__name___
            errstr = "Expected %s with a message matching '%s', got %s."%(expected_name, expected_regex, raised_name)
            print("   FAIL: %s"%(errstr))
            raise AssertionError(errstr) from e
        try:  # then check the exception message
            assertRaisesRegex = getattr(unittest.TestCase, "assertRaisesRegex", unittest.TestCase.assertRaisesRegex)
            assertRaisesRegex(self, expected_exception, expected_regex, *args, **kwargs)  # python 2 vs. 3 compatibility
            try:  # print the exception message also when everything is OK
                func = args[0]
                args = args[1:]
                func(*args, **kwargs)
            except expected_exception:
                print("   PASS: %s"%(sys.exc_info()[1]))
        except AssertionError as e:
            print("   FAIL: %s"%(e))
            raise

    def test_exceptions(self):  # noqa: PLR0915 [too-many-statements]
        print("Testing exception handling...")
        shape = (7, 11, 3)
        pixels = np.random.random(shape).astype(np.float32)
        pixels8b = (pixels * 255).astype(np.uint8)
        pixels16b = (pixels * 65535).astype(np.uint16)
        imwrite("validimage.ppm", pixels8b, 255)
        imwrite("imgio.test.ppm", pixels8b, 255)
        imwrite("imgio.test.png", pixels8b, 255)
        imwrite("imgio.test.jpg", pixels8b, 255)
        os.rename("imgio.test.ppm", "invalidformat.pfm")
        os.rename("imgio.test.png", "invalidformat.jpg")
        os.rename("imgio.test.jpg", "invalidformat.ppm")
        self.assertRaisesRegex(ImageIOError, "^Failed to read", imread, None)
        self.assertRaisesRegex(ImageIOError, "^Failed to read", imread, 0xdeadbeef)
        self.assertRaisesRegex(ImageIOError, "^Failed to read", imread, "")
        self.assertRaisesRegex(ImageIOError, "^Failed to read", imread, ".ppm")
        self.assertRaisesRegex(ImageIOError, "^Failed to read", imread, "invalidfilename")
        self.assertRaisesRegex(ImageIOError, "^Failed to read", imread, "nonexisting/")
        self.assertRaisesRegex(ImageIOError, "^Failed to read", imread, "nonexisting/.ppm")
        self.assertRaisesRegex(ImageIOError, "^Failed to read", imread, "nonexisting/invalidfilename")
        self.assertRaisesRegex(ImageIOError, "^Failed to read", imread, "nonexisting.jpg")
        self.assertRaisesRegex(ImageIOError, "^Failed to read", imread, "nonexisting.png")
        self.assertRaisesRegex(ImageIOError, "^Failed to read", imread, "nonexisting.ppm")
        self.assertRaisesRegex(ImageIOError, "^Failed to read", imread, "nonexisting.pgm")
        self.assertRaisesRegex(ImageIOError, "^Failed to read", imread, "nonexisting.pfm")
        self.assertRaisesRegex(ImageIOError, "^Failed to read", imread, "nonexisting.exr")
        self.assertRaisesRegex(ImageIOError, "^Failed to read", imread, "nonexisting.bmp")
        self.assertRaisesRegex(ImageIOError, "^Failed to read", imread, "invalidformat.pfm")
        self.assertRaisesRegex(ImageIOError, "^Failed to read", imread, "invalidformat.jpg")
        self.assertRaisesRegex(ImageIOError, "^Failed to read", imread, "invalidformat.ppm")
        self.assertRaisesRegex(ImageIOError, "^Failed to read.*verbose", imread, "validimage.ppm", verbose="foo")
        self.assertRaisesRegex(ImageIOError, "^Failed to write", imwrite, "invalidfilename", pixels8b, 255)
        self.assertRaisesRegex(ImageIOError, "^Failed to write", imwrite, "invaliddepth.ppm", pixels16b, 255)
        self.assertRaisesRegex(ImageIOError, "^Failed to write", imwrite, "invaliddepth.png", pixels16b, 255)
        self.assertRaisesRegex(ImageIOError, "^Failed to write", imwrite, "invaliddepth.png", pixels8b, 254)
        self.assertRaisesRegex(ImageIOError, "^Failed to write", imwrite, "invaliddepth.jpg", pixels8b, 254)
        self.assertRaisesRegex(ImageIOError, "^Failed to write", imwrite, "invaliddepth.png", pixels16b, 1023)
        self.assertRaisesRegex(ImageIOError, "^Failed to write", imwrite, "invaliddepth.ppm", pixels8b, 1023)
        self.assertRaisesRegex(ImageIOError, "^Failed to write", imwrite, None, pixels8b, 255)
        self.assertRaisesRegex(ImageIOError, "^Failed to write", imwrite, 0xdeadbeef, pixels8b, 255)
        self.assertRaisesRegex(ImageIOError, "^Failed to write", imwrite, "", pixels8b, 255)
        self.assertRaisesRegex(ImageIOError, "^Failed to write", imwrite, ".ppm", pixels8b, 255)
        self.assertRaisesRegex(ImageIOError, "^Failed to write", imwrite, "nonexisting/.ppm", pixels8b, 255)
        self.assertRaisesRegex(ImageIOError, "^Failed to write", imwrite, "nonexisting/foo.ppm", pixels8b, 255)
        self.assertRaisesRegex(ImageIOError, "^Failed to write", imwrite, "imgio.test.ppm", None)
        self.assertRaisesRegex(ImageIOError, "^Failed to write", imwrite, "imgio.test.ppm", np.zeros((7, 0), np.uint8))
        self.assertRaisesRegex(ImageIOError, "^Failed to write", imwrite, "imgio.test.ppm", np.zeros((7,), np.uint8))
        self.assertRaisesRegex(ImageIOError, "^Failed to write", imwrite, "imgio.test.ppm", np.zeros((7, 7, 1), np.uint8))
        self.assertRaisesRegex(ImageIOError, "^Failed to write", imwrite, "imgio.test.ppm", np.zeros((7, 7, 2), np.uint8))
        self.assertRaisesRegex(ImageIOError, "^Failed to write", imwrite, "imgio.test.ppm", np.zeros((7, 7, 4), np.uint8))
        self.assertRaisesRegex(ImageIOError, "^Failed to write", imwrite, "imgio.test.ppm", np.zeros((7, 7, 3, 1), np.uint8))
        self.assertRaisesRegex(ImageIOError, "^Failed to write", imwrite, "imgio.test.ppm", pixels.astype(bool))
        self.assertRaisesRegex(ImageIOError, "^Failed to write", imwrite, "imgio.test.ppm", pixels.astype(np.float16))
        self.assertRaisesRegex(ImageIOError, "^Failed to write", imwrite, "imgio.test.ppm", pixels.astype(np.float64))
        self.assertRaisesRegex(ImageIOError, "^Failed to write", imwrite, "imgio.test.ppm", pixels.astype('>f4'))
        self.assertRaisesRegex(ImageIOError, "^Failed to write", imwrite, "imgio.test.pfm", pixels, -1.0)
        self.assertRaisesRegex(ImageIOError, "^Failed to write", imwrite, "imgio.test.pfm", pixels, "255")
        self.assertRaisesRegex(ImageIOError, "^Failed to write.*shape", imwrite, "imgio.test.raw", pixels8b, 255)
        self.assertRaisesRegex(ImageIOError, "^Failed to write.*supported", imwrite, "imgio.test.raw", pixels8b, 255, packed=True)
        self.assertRaisesRegex(ImageIOError, "^Failed to write.*verbose", imwrite, "imgio.test.ppm", pixels8b, 255, verbose=0)
        os.remove("invalidformat.pfm")
        os.remove("invalidformat.jpg")
        os.remove("invalidformat.ppm")
        os.remove("validimage.ppm")

    def test_png(self):
        for shape in self.TEST_SHAPES_1 + self.TEST_SHAPES_3:
            for bpp in [8, 16]:
                maxval = 2**bpp - 1
                tempfile = "imgio.test%db.png"%(bpp)
                print("Testing PNG reading & writing in %d-bit mode, shape=%s..."%(bpp, repr(shape)))
                dtype = np.uint8 if bpp <= 8 else np.uint16
                pixels = np.random.random(shape)
                pixels = (pixels * maxval).astype(dtype)
                imwrite(tempfile, pixels, maxval, verbose=False)
                result, resmaxval = imread(tempfile, verbose=False)
                self.assertEqual(resmaxval, maxval)
                self.assertEqual(result.dtype, dtype)
                self.assertEqual(result.shape, pixels.shape)
                np.testing.assert_allclose(result, pixels)
                os.remove(tempfile)

    def test_pnm(self):
        for shape in self.TEST_SHAPES_1 + self.TEST_SHAPES_3:
            for bpp in [1, 5, 7, 8, 10, 12, 15, 16]:
                maxval = 2**bpp - 1
                tempfile = "imgio.test%db.pnm"%(bpp)
                print("Testing PGM/PPM reading & writing in %d-bit mode, shape=%s..."%(bpp, repr(shape)))
                dtype = np.uint8 if bpp <= 8 else np.uint16
                pixels = np.random.random(shape)
                pixels = (pixels * maxval).astype(dtype)
                imwrite(tempfile, pixels, maxval, verbose=False)
                result, resmaxval = imread(tempfile, verbose=False)
                self.assertEqual(resmaxval, maxval)
                self.assertEqual(result.dtype, dtype)
                self.assertEqual(result.shape, pixels.shape)
                np.testing.assert_allclose(result, pixels)
                os.remove(tempfile)

    def test_tiff(self):
        for shape in self.TEST_SHAPES_1 + self.TEST_SHAPES_3:
            for bpp in [8, 16]:
                maxval = 2**bpp - 1
                tempfile = "imgio.test%db.tif"%(bpp)
                print("Testing TIFF reading & writing in %d-bit mode, shape=%s..."%(bpp, repr(shape)))
                dtype = np.uint8 if bpp <= 8 else np.uint16
                pixels = np.random.random(shape)
                pixels = (pixels * maxval).astype(dtype)
                imwrite(tempfile, pixels, maxval, verbose=False)
                result, resmaxval = imread(tempfile, verbose=False)
                self.assertEqual(resmaxval, maxval)
                self.assertEqual(result.dtype, dtype)
                self.assertEqual(result.shape, pixels.shape)
                np.testing.assert_allclose(result, pixels)
                os.remove(tempfile)

    def test_jpg(self):
        for shape in self.TEST_SHAPES_1 + self.TEST_SHAPES_3:
            maxval = 255
            pixels = np.ones(shape)
            pixels = (pixels * 127).astype(np.uint8)
            for extension in ["jpg", "insp"]:
                tempfile = "imgio.test." + extension
                print("Testing %s reading & writing, shape=%s..."%(extension.upper(), repr(shape)))
                imwrite(tempfile, pixels, maxval, verbose=False)
                result, resmaxval = imread(tempfile, verbose=False)
                self.assertEqual(resmaxval, maxval)
                self.assertEqual(result.dtype, np.uint8)
                self.assertEqual(result.shape, pixels.shape)
                np.testing.assert_allclose(result, pixels)
                os.remove(tempfile)

    def test_pfm(self):
        for shape in self.TEST_SHAPES_1 + self.TEST_SHAPES_3:
            bpp = 32
            scale = 3.141
            tempfile = "imgio.test.pfm"
            print("Testing PFM reading & writing in %d-bit mode, shape=%s..."%(bpp, repr(shape)))
            pixels = np.random.random(shape)    # float64 pixels
            pixels = pixels.astype(np.float32)  # convert to float32
            imwrite(tempfile, pixels, maxval=scale, verbose=False)
            result, resscale = imread(tempfile, verbose=False)
            pixels = pixels[..., 0] if pixels.ndim == 3 and shape[-1] == 1 else pixels
            self.assertEqual(resscale, scale)
            self.assertEqual(result.dtype, np.float32)
            self.assertEqual(result.shape, pixels.shape)
            np.testing.assert_allclose(result, pixels)
            os.remove(tempfile)

    def test_npy(self):
        for dt in ["float16", "float32"]:
            for shape in list(self.TEST_SHAPES_ALL) + [(1, 1, 1), (7, 11, 1)]:
                scale = 3.141
                tempfile = "imgio.test.npy"
                print("Testing NPY reading & writing in %s mode, shape=%s..."%(dt, repr(shape)))
                pixels = np.random.random(shape) * 100000  # float64
                inf_mask = pixels >= np.finfo(dt).max
                pixels[inf_mask] = np.inf
                pixels = pixels.astype(dt)  # convert to float16/32
                imwrite(tempfile, pixels, maxval=scale, verbose=False)
                result, resscale = imread(tempfile, verbose=False)
                self.assertEqual(result.dtype, dt)
                self.assertEqual(result.shape, pixels.shape)
                np.testing.assert_allclose(result, pixels)
                os.remove(tempfile)

    def test_exr(self):
        for dt in ["float16", "float32"]:
            for shape in self.TEST_SHAPES_1 + self.TEST_SHAPES_3:
                scale = 3.141
                tempfile = "imgio.test.exr"
                print("Testing EXR reading & writing in %s mode, shape=%s..."%(dt, repr(shape)))
                pixels = np.random.random(shape) * 100000  # float64
                inf_mask = pixels >= np.finfo(dt).max
                pixels[inf_mask] = np.inf
                pixels = pixels.astype(dt)  # convert to float16/32
                imwrite(tempfile, pixels, maxval=scale, verbose=False)
                result, resscale = imread(tempfile, verbose=False)
                self.assertEqual(result.dtype, dt)
                self.assertEqual(result.shape, pixels.shape)
                np.testing.assert_allclose(result, pixels)
                os.remove(tempfile)

    def test_hdr(self):
        for dt in ["float16"]:
            for shape in self.TEST_SHAPES_3:
                tempfile = "imgio.test.hdr"
                print("Testing HDR reading & writing in %s mode, shape=%s..."%(dt, repr(shape)))
                pixels = np.random.random(shape) * 100000  # float64
                inf_mask = pixels >= np.finfo(dt).max
                pixels[inf_mask] = np.finfo(dt).max  # replace infs with maxval
                pixels = pixels.astype(dt)  # convert to float16/32
                pixels = pixels.astype(np.float32)
                imwrite(tempfile, pixels, verbose=True)
                result, resscale = imread(tempfile, verbose=True)
                self.assertEqual(result.dtype, np.float32)
                self.assertEqual(result.shape, pixels.shape)
                # maximum absolute error of RGBE encoding at float16 range is ~256
                np.testing.assert_allclose(result, pixels, atol=256)
                os.remove(tempfile)

    def test_exif(self):
        print("Testing EXIF orientation handling...")
        thispath = os.path.dirname(os.path.abspath(__file__))
        for orientation in ["landscape", "portrait"]:
            reffile = "%s_1.jpg"%(orientation)
            filespec = os.path.join(thispath, "test-images", reffile)
            refimg, refmax = imread(filespec)
            for idx in range(2, 9):
                filename = "%s_%d.jpg"%(orientation, idx)
                print("  %s image, orientation value = %d"%(orientation, idx))
                filespec = os.path.join(thispath, "test-images", filename)
                testimg, testmax = imread(filespec, verbose=False)
                epsdiff = np.isclose(refimg, testimg, atol=5.0, rtol=0.2)
                self.assertEqual(refmax, testmax)
                self.assertEqual(refimg.shape, testimg.shape)
                self.assertGreater(np.sum(epsdiff), 0.5 * epsdiff.size)

    def test_exr_read(self):
        print("Testing EXR reading...")
        thispath = os.path.dirname(os.path.abspath(__file__))
        filespec = os.path.join(thispath, "test-images", "GrayRampsDiagonal.exr")
        img, maxval = imread(filespec)
        self.assertEqual(img.shape, (800, 800))
        #self.assertEqual(img.dtype, np.float16)
        self.assertEqual(maxval, np.max(img))

    def test_raw(self):
        for packed in [False]:
            for shape in self.TEST_SHAPES_1:
                for bpp in [1, 5, 7, 8, 10, 12, 13, 16]:
                    maxval = 2**bpp - 1
                    tempfile = "imgio.test%db.raw"%(bpp)
                    dtype = np.uint8 if bpp <= 8 else np.uint16
                    packstr = "packed" if packed else "padded to %d bits"%(np.dtype(dtype).itemsize * 8)
                    print("Testing RAW reading & writing in %d-bit mode (%s), shape=%s..."%(bpp, packstr, repr(shape)))
                    pixels = np.random.random(shape)
                    pixels = (pixels * maxval).astype(dtype)
                    imwrite(tempfile, pixels, maxval, packed=packed)
                    result, resmaxval = imread(tempfile, width=shape[1], height=shape[0], bpp=bpp)
                    self.assertEqual(resmaxval, maxval)
                    self.assertEqual(result.dtype, dtype)
                    self.assertEqual(result.shape, shape)
                    self.assertEqual(result.tolist(), pixels.tolist())
                    os.remove(tempfile)

    def test_packed_raw(self):
        bpp = 10
        maxval = 2**bpp - 1
        pixels = np.random.random(4)
        pixels = (pixels * maxval).astype(np.uint16)
        packed = np.zeros(5, dtype=np.uint8)
        packed[0] |= (pixels[0] & 0x00ff)  # byte0 / pixel0: 8 lsb
        packed[1] |= (pixels[0] & 0x0300) >> 8  # byte1 / pixel0: 2 msb
        packed[1] |= (pixels[1] & 0x003f) << 2  # byte1 / pixel1: 6 lsb
        packed[2] |= (pixels[1] & 0x03c0) >> 6  # byte2 / pixel1: 4 msb
        packed[2] |= (pixels[2] & 0x000f) << 4  # byte2 / pixel2: 4 lsb
        packed[3] |= (pixels[2] & 0x03f0) >> 4  # byte3 / pixel2: 6 msb
        packed[3] |= (pixels[3] & 0x0003) << 6  # byte3 / pixel3: 2 lsb
        packed[4] |= (pixels[3] & 0x03fc) >> 2  # byte4 / pixel3: 8 msb
        tempfile = "imgio.test%db.raw"%(bpp)
        packed.tofile(tempfile)
        result, resmaxval = imread(tempfile, width=2, height=2, bpp=bpp)
        self.assertEqual(resmaxval, maxval)
        self.assertEqual(result.shape, (2, 2))
        np.testing.assert_equal(result.flatten(), pixels)

    def test_raw_header(self):
        bpp = 12
        shape = (28, 60)
        maxval = 2**bpp - 1
        tempfile = "imgio.test%db.raw"%(bpp)
        print("Testing RAW header reading & writing...")
        header = np.arange(17).astype(np.uint16)
        footer = np.arange(19).astype(np.uint16)
        pixels = np.random.random(shape)
        pixels = (pixels * maxval).astype(np.uint16)
        data = np.hstack((header, pixels.flatten(), footer)).reshape(1, -1)
        imwrite(tempfile, data, maxval)
        result, resmaxval = imread(tempfile, width=shape[1], height=shape[0], bpp=bpp, raw_header_size=17 * 2)
        self.assertEqual(resmaxval, maxval)
        self.assertEqual(result.dtype, np.uint16)
        self.assertEqual(result.shape, shape)
        np.testing.assert_allclose(result, pixels)
        os.remove(tempfile)

    def test_allcaps(self):
        print("Testing Windows-style all-caps filenames...")
        maxval = 255
        dtype = np.uint8
        for ext in [".pnm", ".ppm", ".jpg", ".jpeg"]:
            tempfile = "imgio.test%s"%(ext)
            capsfile = "imgio.test%s"%(ext.upper())
            shape = (7, 11, 3)
            pixels = np.ones(shape)
            pixels = (pixels * maxval).astype(dtype)
            imwrite(tempfile, pixels, maxval, verbose=True)
            os.rename(tempfile, capsfile)
            result, resmaxval = imread(capsfile, verbose=True)
            self.assertEqual(resmaxval, maxval)
            self.assertEqual(result.shape, shape)
            np.testing.assert_allclose(result, pixels)
            os.remove(capsfile)

    def test_verbose(self):
        print("Testing verbose mode...")
        for dt in ["uint8", "uint16"]:
            maxval = np.iinfo(dt).max
            for shape in [(7, 11), (9, 13, 3)]:
                for ext in [".pnm", ".jpg", ".png"]:
                    if dt == "uint8" or ext != ".jpg":
                        tempfile = "imgio.test%s"%(ext)
                        pixels = np.random.random(shape)
                        pixels = (pixels * maxval).astype(dt)
                        imwrite(tempfile, pixels, maxval, verbose=True)
                        result, resmaxval = imread(tempfile, verbose=True)
                        self.assertEqual(resmaxval, maxval)
                        self.assertEqual(result.shape, shape)
                        os.remove(tempfile)


if __name__ == "__main__":
    selftest()
