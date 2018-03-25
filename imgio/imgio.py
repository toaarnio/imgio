#!/usr/bin/python3 -B

"""
Easy image file reading & writing. Supports PGM/PPM/PFM/PNG/JPG.

Example:
  image, maxval = imgio.imread("foo.png")
  imgio.imwrite("foo.ppm", image, maxval)
"""

from __future__ import absolute_import
from __future__ import print_function as __print

import os                         # standard library
import sys                        # standard library
import unittest                   # standard library

import numpy as np                # pip install numpy
import imread as _imread          # pip install imread

try:
    # package mode
    from imgio import pnm         # local import: pnm.py
    from imgio import pfm         # local import: pfm.py
except ImportError:
    # stand-alone mode
    import pnm                    # local import: pnm.py
    import pfm                    # local import: pfm.py

# pylint: disable=no-member

######################################################################################
#
#  P U B L I C   A P I
#
######################################################################################

def imread(filespec, width=None, height=None, bpp=None, verbose=False):
    # type: (str, int, int, int, bool) -> Tuple[np.ndarray, Union[int, float]]
    """
    Reads the given image file from disk and returns it as a NumPy array.
    Grayscale images are returned as 2D arrays of shape H x W, color images
    as 3D arrays of shape H x W x 3.
    """
    ImageIOError.error_message_prefix = "Failed to read %s: "%(repr(filespec))
    filename = os.path.basename(filespec)            # "path/image.pgm" => "image.pgm"
    basename, filetype = os.path.splitext(filename)  # "image.pgm" => ("image", ".pgm")
    _enforce(len(basename) > 1, "filename `%s` must have at least 1 character + extension."%(filename))
    _enforce(len(filetype) > 3, "filename `%s` must have at least 1 character + extension."%(filename))
    _enforce(isinstance(verbose, bool), "verbose must be True or False, was %s (%s)."%(type(verbose), repr(verbose)))
    if filetype in [".raw", ".RAW"]:
        _enforce(isinstance(bpp, int) and 1 <= bpp <= 16, "bpp must be an integer in [1, 16]; was %s"%(repr(bpp)))
        _enforce(isinstance(width, int) and width >= 1, "width must be an integer >= 1; was %s"%(repr(width)))
        _enforce(isinstance(height, int) and height >= 1, "height must be an integer >= 1; was %s"%(repr(height)))
        frame, maxval = _reraise(lambda: _read_raw(filespec, width, height, bpp, verbose=verbose))
        return frame, maxval
    elif filetype in [".pfm", ".PFM"]:
        frame, scale = _reraise(lambda: pfm.read(filespec, verbose))
        return frame, scale
    elif filetype in [".pnm", ".pgm", ".ppm", ".PNM", ".PGM", ".PPM"]:
        frame, maxval = _reraise(lambda: pnm.read(filespec, verbose))
        return frame, maxval
    elif filetype in [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]:
        _print(verbose, "Reading file %s "%(filespec), end='')
        frame = _reraise(lambda: _imread.imread(filespec))
        maxval = 255 if frame.dtype == np.uint8 else 65535
        h, w = frame.shape[:2]
        c = frame.shape[2] if frame.ndim > 2 else 1
        _print(verbose, "(w=%d, h=%d, c=%d, maxval=%d)"%(w, h, c, maxval))
        must_squeeze = (frame.ndim > 2 and frame.shape[2] == 1)
        frame = frame.squeeze(axis=2) if must_squeeze else frame
        return frame, maxval
    else:
        raise ImageIOError("unrecognized file type `%s`."%(filetype))

def imwrite(filespec, image, maxval=255, packed=False, verbose=False):
    # type: (str, np.ndarray, Union[int, float], bool, bool) -> None
    """
    Writes the given image to the given file. Grayscale images are expected
    to be provided as NumPy arrays with shape H x W, color images with shape
    H x W x 3. Metadata, alpha channels, etc. are not supported.
    """
    ImageIOError.error_message_prefix = "Failed to write %s: "%(repr(filespec))
    _enforce(isinstance(filespec, str), "filespec must be a string, was %s (%s)."%(type(filespec), repr(filespec)))
    _enforce(isinstance(image, np.ndarray), "image must be a NumPy ndarray; was %s."%(type(image)))
    _enforce(image.dtype in [np.uint8, np.uint16, np.float32], "image.dtype must be uint8, uint16, or float32; was %s"%(image.dtype))
    _enforce(image.size >= 1, "image must have at least one pixel; had none.")
    _enforce(isinstance(maxval, (float, int)), "maxval must be an integer or a float; was %s."%(repr(maxval)))
    _enforce(isinstance(maxval, int) or image.dtype == np.float32, "maxval must be an integer in [1, 65535]; was %s."%(repr(maxval)))
    _enforce(1 <= maxval <= 65535 or image.dtype == np.float32, "maxval must be an integer in [1, 65535]; was %s."%(repr(maxval)))
    _enforce(isinstance(verbose, bool), "verbose must be True or False, was %s (%s)."%(type(verbose), repr(verbose)))
    _disallow(image.ndim not in [2, 3], "image.shape must be (m, n) or (m, n, 3); was %s."%(str(image.shape)))
    _disallow(image.ndim == 3 and image.shape[2] != 3, "image.shape must be (m, n) or (m, n, 3); was %s."%(str(image.shape)))
    _disallow(maxval > 255 and image.dtype == np.uint8, "maxval (%d) and image.dtype (%s) are inconsistent."%(maxval, image.dtype))
    _disallow(maxval <= 255 and image.dtype == np.uint16, "maxval (%d) and image.dtype (%s) are inconsistent."%(maxval, image.dtype))
    filename = os.path.basename(filespec)            # "path/image.pgm" => "image.pgm"
    basename, filetype = os.path.splitext(filename)  # "image.pgm" => ("image", ".pgm")
    _enforce(len(basename) > 0, "filename `%s` must have at least 1 character + extension."%(filename))
    _enforce(len(filetype) > 3, "filename `%s` must have at least 1 character + extension."%(filename))
    if filetype in [".raw", ".RAW"]:
        _enforce(packed is False, "packed Bayer RAW images are not yet supported.")
        _enforce(image.ndim == 2, "image.shape must be (m, n) for a Bayer RAW; was %s."%(str(image.shape)))
        _reraise(lambda: _write_raw(filespec, image, maxval, packed, verbose))
    elif filetype == ".pfm":
        _enforce(maxval >= 0.0, "maxval (scale) must be non-negative; was %s."%(repr(maxval)))
        _reraise(lambda: pfm.write(filespec, image, maxval, verbose))
    elif filetype in [".pnm", ".pgm", ".ppm"]:
        _reraise(lambda: pnm.write(filespec, image, maxval, verbose))
    elif filetype in [".png", ".jpg", ".jpeg"]:
        _disallow(filetype in [".jpg", ".jpeg"] and maxval != 255, "maxval must be 255 for a JPEG; was %d."%(maxval))
        _disallow(filetype == ".png" and maxval not in [255, 65535], "maxval must be 255 or 65535 for a PNG; was %d."%(maxval))
        _print(verbose, "Writing file %s "%(filespec), end='')
        _reraise(lambda: _imread.imsave(filespec, image))
        h, w = image.shape[:2]
        c = image.shape[2] if image.ndim > 2 else 1
        _print(verbose, "(w=%d, h=%d, c=%d, maxval=%d)"%(w, h, c, maxval))
    else:
        raise ImageIOError("unrecognized file type `%s`."%(filetype))

def selftest():
    """
    Runs the full suite of unit tests that comes bundled with the package.
    """
    print("--" * 35)
    suite = unittest.TestLoader().loadTestsFromTestCase(_TestImgIo)
    unittest.TextTestRunner(verbosity=0).run(suite)

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

# pylint: disable=missing-docstring

def _enforce(expression, error_message_if_false):
    if not expression:
        raise ImageIOError("%s"%(error_message_if_false))

def _disallow(expression, error_message_if_true):
    if expression:
        raise ImageIOError("%s"%(error_message_if_true))

def _reraise(func):
    try:
        return func()
    except Exception:
        raise ImageIOError("%s"%(repr(sys.exc_info()[1])))

def _print(verbose, *args, **kwargs):
    if verbose:
        print(*args, **kwargs)

def _read_raw(filespec, width, height, bpp, verbose=False):
    # Warning: hardcoded endianness (x86)
    with open(filespec, "rb") as infile:
        buf = infile.read()
        shape = (height, width)
        maxval = 2 ** bpp - 1
        wordsize = 2 if bpp > 8 else 1
        packed = len(buf) < (width * height * wordsize)
        _print(verbose, "Reading raw Bayer file %s "%(filespec), end='')
        _print(verbose, "(w=%d, h=%d, maxval=%d, packed=%r)"%(width, height, maxval, packed))
        if not packed:
            dtype = "<u2" if bpp > 8 else np.uint8
            pixels = np.frombuffer(buf, dtype, count=width * height, offset=0)
            pixels = pixels.reshape(shape).astype(np.uint8 if bpp <= 8 else np.uint16)
        else:
            # TODO: unpack!
            raise ImageIOError("Packed RAW reading not implemented yet!")
        return pixels, maxval

def _write_raw(filespec, image, maxval, pack=False, verbose=False):
    # Warning: hardcoded endianness (x86)
    bpp = int(np.log2(maxval + 1))
    with open(filespec, "wb") as outfile:
        image = image.copy(order='C')  # ensure x86 byte order
        outfile.write(image)

######################################################################################
#
#  U N I T   T E S T S
#
######################################################################################

class _TestImgIo(unittest.TestCase):

    def assertRaisesRegexp(self, exc_class, regex, func, *args, **kwargs):  # pylint: disable=invalid-name
        """
        Checks that the correct type of exception is raised, and that the exception
        message matches the given regular expression. Also prints out the message
        for visual inspection.
        """
        try:  # check the type of exception first
            unittest.TestCase.assertRaises(self, exc_class, func, *args, **kwargs)
        except Exception as e:
            raised_type = sys.exc_info()[0]
            errstr = "Expected %s with a message matching '%s', got %s."%(exc_class.__name__, regex, raised_type.__name__)
            print("   FAIL: %s"%(errstr))
            raise AssertionError(errstr)
        try:  # then check the exception message
            assertRaisesRegex = getattr(unittest.TestCase, "assertRaisesRegex", unittest.TestCase.assertRaisesRegexp)
            assertRaisesRegex(self, exc_class, regex, func, *args, **kwargs)  # python 2 vs. 3 compatibility hack
            try:  # print the exception message also when everything is OK
                func(*args, **kwargs)
            except exc_class:
                print("   PASS: %s"%(sys.exc_info()[1]))
        except AssertionError as e:
            print("   FAIL: %s"%(e))
            raise

    def test_exceptions(self):
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
        self.assertRaisesRegexp(ImageIOError, "^Failed to read", imread, "nonexisting.jpg")
        self.assertRaisesRegexp(ImageIOError, "^Failed to read", imread, "nonexisting.png")
        self.assertRaisesRegexp(ImageIOError, "^Failed to read", imread, "nonexisting.ppm")
        self.assertRaisesRegexp(ImageIOError, "^Failed to read", imread, "nonexisting.pgm")
        self.assertRaisesRegexp(ImageIOError, "^Failed to read", imread, "nonexisting.pfm")
        self.assertRaisesRegexp(ImageIOError, "^Failed to read", imread, "invalidformat.pfm")
        self.assertRaisesRegexp(ImageIOError, "^Failed to read", imread, "invalidformat.jpg")
        self.assertRaisesRegexp(ImageIOError, "^Failed to read", imread, "invalidformat.ppm")
        self.assertRaisesRegexp(ImageIOError, "^Failed to read", imread, "invalidtype.bmp")
        self.assertRaisesRegexp(ImageIOError, "^Failed to read", imread, "invalidtype.bmp")
        self.assertRaisesRegexp(ImageIOError, "^Failed to read.*verbose", imread, "validimage.ppm", verbose="foo")
        self.assertRaisesRegexp(ImageIOError, "^Failed to write", imwrite, "invalidtype.bmp", pixels8b, 255)
        self.assertRaisesRegexp(ImageIOError, "^Failed to write", imwrite, "invaliddepth.ppm", pixels16b, 255)
        self.assertRaisesRegexp(ImageIOError, "^Failed to write", imwrite, "invaliddepth.png", pixels16b, 255)
        self.assertRaisesRegexp(ImageIOError, "^Failed to write", imwrite, "invaliddepth.png", pixels8b, 254)
        self.assertRaisesRegexp(ImageIOError, "^Failed to write", imwrite, "invaliddepth.jpg", pixels8b, 254)
        self.assertRaisesRegexp(ImageIOError, "^Failed to write", imwrite, "invaliddepth.png", pixels16b, 1023)
        self.assertRaisesRegexp(ImageIOError, "^Failed to write", imwrite, "invaliddepth.ppm", pixels8b, 1023)
        self.assertRaisesRegexp(ImageIOError, "^Failed to write", imwrite, None, pixels8b, 255)
        self.assertRaisesRegexp(ImageIOError, "^Failed to write", imwrite, 0xdeadbeef, pixels8b, 255)
        self.assertRaisesRegexp(ImageIOError, "^Failed to write", imwrite, "", pixels8b, 255)
        self.assertRaisesRegexp(ImageIOError, "^Failed to write", imwrite, ".ppm", pixels8b, 255)
        self.assertRaisesRegexp(ImageIOError, "^Failed to write", imwrite, "nonexisting/.ppm", pixels8b, 255)
        self.assertRaisesRegexp(ImageIOError, "^Failed to write", imwrite, "nonexisting/foo.ppm", pixels8b, 255)
        self.assertRaisesRegexp(ImageIOError, "^Failed to write", imwrite, "imgio.test.ppm", None)
        self.assertRaisesRegexp(ImageIOError, "^Failed to write", imwrite, "imgio.test.ppm", np.zeros((7, 0), np.uint8))
        self.assertRaisesRegexp(ImageIOError, "^Failed to write", imwrite, "imgio.test.ppm", np.zeros((7,), np.uint8))
        self.assertRaisesRegexp(ImageIOError, "^Failed to write", imwrite, "imgio.test.ppm", np.zeros((7, 7, 1), np.uint8))
        self.assertRaisesRegexp(ImageIOError, "^Failed to write", imwrite, "imgio.test.ppm", np.zeros((7, 7, 2), np.uint8))
        self.assertRaisesRegexp(ImageIOError, "^Failed to write", imwrite, "imgio.test.ppm", np.zeros((7, 7, 4), np.uint8))
        self.assertRaisesRegexp(ImageIOError, "^Failed to write", imwrite, "imgio.test.ppm", np.zeros((7, 7, 3, 1), np.uint8))
        self.assertRaisesRegexp(ImageIOError, "^Failed to write", imwrite, "imgio.test.ppm", pixels.astype(np.bool))
        self.assertRaisesRegexp(ImageIOError, "^Failed to write", imwrite, "imgio.test.ppm", pixels.astype(np.float16))
        self.assertRaisesRegexp(ImageIOError, "^Failed to write", imwrite, "imgio.test.ppm", pixels.astype(np.float64))
        self.assertRaisesRegexp(ImageIOError, "^Failed to write", imwrite, "imgio.test.ppm", pixels.astype('>f4'))
        self.assertRaisesRegexp(ImageIOError, "^Failed to write", imwrite, "imgio.test.pfm", pixels, -1.0)
        self.assertRaisesRegexp(ImageIOError, "^Failed to write", imwrite, "imgio.test.pfm", pixels, "255")
        self.assertRaisesRegexp(ImageIOError, "^Failed to write.*shape", imwrite, "imgio.test.raw", pixels8b, 255)
        self.assertRaisesRegexp(ImageIOError, "^Failed to write.*supported", imwrite, "imgio.test.raw", pixels8b, 255, packed=True)
        self.assertRaisesRegexp(ImageIOError, "^Failed to write.*verbose", imwrite, "imgio.test.ppm", pixels8b, 255, verbose=0)
        os.remove("invalidformat.pfm")
        os.remove("invalidformat.jpg")
        os.remove("invalidformat.ppm")
        os.remove("validimage.ppm")

    def test_png(self):
        for shape in [(1, 1), (1, 1, 3), (7, 11), (9, 13, 3), (123, 321, 3)]:
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
                self.assertEqual(result.shape, shape)
                self.assertEqual(result.tolist(), pixels.tolist())
                os.remove(tempfile)

    def test_pnm(self):
        for shape in [(1, 1), (1, 1, 3), (7, 11), (9, 13, 3), (123, 321, 3)]:
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
                self.assertEqual(result.shape, shape)
                self.assertEqual(result.tolist(), pixels.tolist())
                os.remove(tempfile)

    def test_jpg(self):
        for shape in [(1, 1), (1, 1, 3), (7, 11), (9, 13, 3), (123, 321, 3)]:
            maxval = 255
            tempfile = "imgio.test.jpg"
            print("Testing JPG reading & writing...")
            pixels = np.ones(shape)
            pixels = (pixels * 127).astype(np.uint8)
            imwrite(tempfile, pixels, maxval, verbose=False)
            result, resmaxval = imread(tempfile, verbose=False)
            self.assertEqual(resmaxval, maxval)
            self.assertEqual(result.dtype, np.uint8)
            self.assertEqual(result.shape, shape)
            self.assertEqual(result.tolist(), pixels.tolist())
            os.remove(tempfile)

    def test_pfm(self):
        for shape in [(1, 1), (1, 1, 3), (7, 11), (9, 13, 3), (123, 321, 3)]:
            bpp = 32
            scale = 3.141
            tempfile = "imgio.test.pfm"
            print("Testing PFM reading & writing in %d-bit mode, shape=%s..."%(bpp, repr(shape)))
            pixels = np.random.random(shape)    # float64 pixels
            pixels = pixels.astype(np.float32)  # convert to float32
            imwrite(tempfile, pixels, maxval=scale, verbose=False)
            result, resscale = imread(tempfile, verbose=False)
            self.assertEqual(resscale, scale)
            self.assertEqual(result.dtype, np.float32)
            self.assertEqual(result.shape, shape)
            self.assertEqual(result.shape, shape)
            os.remove(tempfile)

    def test_raw(self):
        for packed in [False]:
            for shape in [(1, 1), (7, 11)]:
                for bpp in [1, 5, 7, 8, 10, 12, 13, 16]:
                    maxval = 2**bpp - 1
                    tempfile = "imgio.test%db.raw"%(bpp)
                    dtype = np.uint8 if bpp <= 8 else np.uint16
                    packstr = "packed" if packed else "padded to %d bits"%(np.dtype(dtype).itemsize * 8)
                    print("Testing RAW reading & writing in %d-bit mode (%s), shape=%s..."%(bpp, packstr, repr(shape)))
                    pixels = np.random.random(shape)
                    pixels = (pixels * maxval).astype(dtype)
                    imwrite(tempfile, pixels, maxval, packed=packed)
                    result, resmaxval = imread(tempfile, shape[1], shape[0], bpp)
                    self.assertEqual(resmaxval, maxval)
                    self.assertEqual(result.dtype, dtype)
                    self.assertEqual(result.shape, shape)
                    self.assertEqual(result.tolist(), pixels.tolist())
                    os.remove(tempfile)

    def test_allcaps(self):
        print("Testing Windows-style all-caps filenames...")
        maxval = 255
        dtype = np.uint8
        for ext in [".pnm", ".pfm", ".ppm", ".jpg", ".jpeg"]:
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
            self.assertEqual(result.tolist(), pixels.tolist())
            os.remove(capsfile)

    def test_verbose(self):
        print("Testing verbose mode...")
        for shape in [(7, 11), (9, 13, 3)]:
            for ext in [".pnm", ".jpg", ".pfm", ".png"]:
                maxval = 255
                tempfile = "imgio.test%s"%(ext)
                pixels = np.random.random(shape)
                pixels = (pixels * maxval).astype(np.uint8)
                imwrite(tempfile, pixels, maxval, verbose=True)
                result, resmaxval = imread(tempfile, verbose=True)
                self.assertEqual(resmaxval, maxval)
                self.assertEqual(result.shape, shape)
                os.remove(tempfile)


if __name__ == "__main__":
    selftest()
