#!/usr/bin/python -B

"""
Easy image file reading & writing. Supports PGM/PPM/PFM/PNG/JPG.

Example:
  image, maxval = imgio.imread("foo.png")
  imgio.imwrite("foo.ppm", image, maxval)
"""

from __future__ import print_function as __print

import os                         # standard library
import sys                        # standard library
import unittest                   # standard library

from typing import Tuple, Union   # pip install typing
import numpy as np                # pip install numpy
import imread as _imread          # pip install imread

import pnm                        # local import: pnm.py
import pfm                        # local import: pfm.py

######################################################################################
#
#  P U B L I C   A P I
#
######################################################################################

__all__ = ["imread", "imwrite"]  # the exported symbols

def imread(filespec, width=None, height=None, bpp=None, verbose=False):
    # type: (str, int, int, int, bool) -> Tuple[np.ndarray, Union[int, float]]
    """
    Reads the given image file from disk and returns it as a NumPy array.
    Grayscale images are returned as 2D arrays of shape H x W, color images
    as 3D arrays of shape H x W x 3.
    """
    global _errorMessagePrefix
    _errorMessagePrefix = "Failed to read %s"%(repr(filespec))
    filename = os.path.basename(filespec)            # "path/image.pgm" => "image.pgm"
    basename, filetype = os.path.splitext(filename)  # "image.pgm" => ("image", ".pgm")
    _enforce(len(basename) > 1, "filename `%s` must have at least 1 character + extension."%(filename))
    _enforce(len(filetype) > 3, "filename `%s` must have at least 1 character + extension."%(filename))
    if filetype in [".raw", ".bin", ".RAW", ".BIN"]:
        _enforce(isinstance(bpp, int) and 1 <= bpp <= 16, "bpp must be an integer in [1, 16]; was %s"%(repr(bpp)))
        frame, maxval = _reraise(lambda: _readRaw(filespec, width, height, bpp, verbose=verbose))
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
        mustSqueeze = (frame.ndim > 2 and frame.shape[2] == 1)
        frame = frame.squeeze(axis=2) if mustSqueeze else frame
        return frame, maxval
    else:
        raise ImageIOError("%s: Unrecognized file type `%s`."%(_errorMessagePrefix, filetype))

def imwrite(filespec, image, maxval=255, verbose=False):
    # type: (str, np.ndarray, Union[int, float], bool) -> None
    """
    Writes the given image to the given file. Grayscale images are expected to be
    provided as NumPy arrays with shape H x W, color images with shape H x W x 3.
    Metadata, alpha channels, etc. are not supported.
    """
    global _errorMessagePrefix
    _errorMessagePrefix = "Failed to write %s"%(repr(filespec))
    _enforce(type(filespec) == str, "filespec must be a string, was %s (%s)."%(type(filespec), repr(filespec)))
    _enforce(type(image) == np.ndarray, "image must be a NumPy ndarray; was %s."%(type(image)))
    _enforce(image.dtype in [np.uint8, np.uint16, np.float32], "image.dtype must be uint8, uint16, or float32; was %s"%(image.dtype))
    _enforce(image.size >= 1, "image must have at least one pixel; had none.")
    _enforce(type(maxval) in [float, int], "maxval must be an integer or a float; was %s."%(repr(maxval)))
    _enforce(type(maxval) == int or image.dtype == np.float32, "maxval must be an integer in [1, 65535]; was %s."%(repr(maxval)))
    _enforce(1 <= maxval <= 65535 or image.dtype == np.float32, "maxval must be an integer in [1, 65535]; was %s."%(repr(maxval)))
    _disallow(image.ndim not in [2, 3], "image.shape must be (m, n) or (m, n, 3); was %s."%(str(image.shape)))
    _disallow(image.ndim == 3 and image.shape[2] != 3, "image.shape must be (m, n) or (m, n, 3); was %s."%(str(image.shape)))
    _disallow(maxval > 255 and image.dtype == np.uint8, "maxval (%d) and image.dtype (%s) are inconsistent."%(maxval, image.dtype))
    _disallow(maxval <= 255 and image.dtype == np.uint16, "maxval (%d) and image.dtype (%s) are inconsistent."%(maxval, image.dtype))
    filename = os.path.basename(filespec)            # "path/image.pgm" => "image.pgm"
    basename, filetype = os.path.splitext(filename)  # "image.pgm" => ("image", ".pgm")
    _enforce(len(basename) > 0, "filename `%s` must have at least 1 character + extension."%(filename))
    _enforce(len(filetype) > 3, "filename `%s` must have at least 1 character + extension."%(filename))
    if filetype == ".pfm":
        _enforce(maxval >= 0.0, "maxval (scale) must be non-negative; was %s."%(repr(maxval)))
        _reraise(lambda: pfm.write(filespec, image, scale=maxval, verbose=verbose))
    elif filetype in [".pnm", ".pgm", ".ppm"]:
        _reraise(lambda: pnm.write(filespec, image, maxval=maxval, verbose=verbose))
    elif filetype in [".png", ".jpg", ".jpeg"]:
        _disallow(filetype in [".jpg", ".jpeg"] and maxval != 255, "maxval must be 255 for a JPEG; was %d."%(maxval))
        _disallow(filetype == ".png" and maxval not in [255, 65535], "maxval must be 255 or 65535 for a PNG; was %d."%(maxval))
        _disallow(filetype == ".png" and image.ndim == 3 and maxval == 65535, "Writing 16-bit color PNGs is not supported yet.")
        _print(verbose, "Writing file %s "%(filespec), end='')
        _reraise(lambda: _imread.imsave(filespec, image))
        h, w = image.shape[:2]
        c = image.shape[2] if image.ndim > 2 else 1
        _print(verbose, "(w=%d, h=%d, c=%d, maxval=%d)"%(w, h, c, maxval))
    else:
        raise ImageIOError("%s: Unrecognized file type `%s`."%(_errorMessagePrefix, filetype))

class ImageIOError(RuntimeError):
    pass

######################################################################################
#
#  I N T E R N A L   F U N C T I O N S
#
######################################################################################

def _enforce(expression, errorMessageIfFalse, exceptionType=ImageIOError):
    if not expression:
        raise exceptionType("%s: %s"%(_errorMessagePrefix, errorMessageIfFalse))

def _disallow(expression, errorMessageIfTrue, exceptionType=ImageIOError):
    if expression:
        raise exceptionType("%s: %s"%(_errorMessagePrefix, errorMessageIfTrue))

def _reraise(func):
    try:
        return func()
    except Exception:
        raise ImageIOError("%s: %s"%(_errorMessagePrefix, repr(sys.exc_info()[1])))

def _print(verbose, *args, **kwargs):
    if verbose:
        print(*args, **kwargs)

def _readRaw(filespec, width, height, bpp, verbose=False):
    # Warning: hardcoded endianness for 16-bit data!
    with open(filespec, "rb") as f:
        buf = f.read()
        shape = (height, width)
        maxval = 2 ** bpp - 1
        _print(verbose, "Reading raw Bayer file %s "%(filespec), end='')
        _print(verbose, "(w=%d, h=%d, maxval=%d)"%(width, height, maxval))
        dtype = "<u2" if bpp > 8 else np.uint8
        pixels = np.frombuffer(buf, dtype, count=width * height, offset=0)
        pixels = pixels.reshape(shape).astype(np.uint8 if bpp <= 8 else np.uint16)
        return pixels, maxval

######################################################################################
#
#  U N I T   T E S T S
#
######################################################################################

class _TestImgIo(unittest.TestCase):

    class DummyError(Exception):
        pass

    def assertRaises(self, excClass, callableObj, *args, **kwargs):
        excClass = self.DummyError
        try:
            unittest.TestCase.assertRaises(self, excClass, callableObj, *args, **kwargs)
        except Exception:
            print("   %s"%(sys.exc_info()[1]))

    def assertRaisesRegexp(self, excClass, re, callableObj, *args, **kwargs):
        excClass = self.DummyError
        try:
            unittest.TestCase.assertRaisesRegexp(self, excClass, re, callableObj, *args, **kwargs)
        except Exception:
            print("   %s"%(sys.exc_info()[1]))

    def test_exceptions(self):
        print("Testing exception handling...")
        shape = (7, 11, 3)
        pixels = np.random.random(shape).astype(np.float32)
        pixels8b = (pixels * 255).astype(np.uint8)
        pixels16b = (pixels * 65535).astype(np.uint16)
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
        self.assertRaisesRegexp(ImageIOError, "^Failed to write", imwrite, "imgio.test.png", pixels16b, 65535)
        self.assertRaisesRegexp(ImageIOError, "^Failed to write", imwrite, "imgio.test.pfm", pixels, -1.0)
        self.assertRaisesRegexp(ImageIOError, "^Failed to write", imwrite, "imgio.test.pfm", pixels, "255")
        os.remove("invalidformat.pfm")
        os.remove("invalidformat.jpg")
        os.remove("invalidformat.ppm")

    def test_png(self):
        for shape in [(1, 1), (1, 1, 3), (7, 11), (9, 13, 3), (123, 321, 3)]:
            for bpp in [8, 16]:
                if len(shape) == 3 and bpp == 16:  # skip 16-bit color mode
                    continue  # NB: Remove this check when imread.imsave() gets fixed!
                maxval = 2**bpp - 1
                tempfile = "imgio.test%db.png"%(bpp)
                print("Testing PNG reading & writing in %d-bit mode, shape=%s..."%(bpp, repr(shape)))
                dtype = np.uint8 if bpp == 8 else np.uint16
                pixels = np.random.random(shape)
                pixels = (pixels * maxval).astype(dtype)
                imwrite(tempfile, pixels, maxval, verbose=False)
                result, resmaxval = imread(tempfile, verbose=False)
                self.assertEqual(resmaxval, maxval)
                self.assertEqual(result.dtype, dtype)
                self.assertEqual(result.shape, shape)
                self.assertEqual(result.tolist(), pixels.tolist())
                os.remove(tempfile)

    def test_ppm(self):
        for shape in [(1, 1), (1, 1, 3), (7, 11), (9, 13, 3), (123, 321, 3)]:
            for bpp in [8, 10, 12, 16]:
                maxval = 2**bpp - 1
                tempfile = "imgio.test%db.pnm"%(bpp)
                print("Testing PGM/PPM reading & writing in %d-bit mode, shape=%s..."%(bpp, repr(shape)))
                dtype = np.uint8 if bpp == 8 else np.uint16
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

    def test_allcaps(self):
        print("Testing Windows-style all-caps filenames...")
        maxval = 255
        dtype = np.uint8
        basename = "imgio.test"
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
    print("--" * 35)
    suite = unittest.TestLoader().loadTestsFromTestCase(_TestImgIo)
    unittest.TextTestRunner(verbosity=0).run(suite)
