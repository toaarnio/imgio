#!/usr/bin/python -B

"""
Easy image file reading & writing. Supports PGM/PPM/PFM/PNG/JPG.

Example:
  image, maxval = imgio.imread("foo.png")
  imgio.imwrite("foo.ppm", image, maxval)
"""

from __future__ import print_function as _print

import sys, os
import numpy as np
import imread as _imread   # pip install imread
import pnm                 # pnm.py
import pfm                 # pfm.py

######################################################################################
#
#  P U B L I C   A P I
#
######################################################################################

def imread(filespec, verbose=False):
    """
    Reads the given image file from disk and returns it as a NumPy array.
    Grayscale images are returned as 2D arrays of shape H x W, color images
    as 3D arrays of shape H x W x 3.
    """
    filename = os.path.basename(filespec)            # "path/image.pgm" => "image.pgm"
    basename, filetype = os.path.splitext(filename)  # "image.pgm" => ("image", ".pgm")
    _enforce(len(basename) > 1, "filename `%s` must have at least 1 character + extension."%(filename))
    _enforce(len(filetype) > 3, "filename `%s` must have at least 1 character + extension."%(filename))
    if filetype in [".pnm", ".pgm", ".ppm", ".pfm"]:
        reader = {".pnm":pnm, ".pgm":pnm, ".ppm":pnm, ".pfm":pfm}[filetype]
        backup, reader.VERBOSE = reader.VERBOSE, verbose
        frame, maxval = _reraise(lambda: reader.read(filespec))
        reader.VERBOSE = backup
        return frame, maxval
    elif filetype in [".png", ".jpg", ".jpeg"]:
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
        raise ImageIOError("Failed to read %s: Unrecognized file type `%s`."%(filespec, filetype))


def imwrite(filespec, image, maxval=255, verbose=False):
    """
    Writes the given image to the given file. Grayscale images are expected to be
    provided as NumPy arrays with shape H x W, color images with shape H x W x 3.
    Metadata, alpha channels, etc. are not supported.
    """
    ## preconditions
    _enforce(type(filespec) == str, "filespec must be a string, was %s (%r)."%(type(filespec), filespec))
    _enforce(type(image) == np.ndarray, "image must be a NumPy ndarray; was %r."%(type(image)))
    _enforce(image.dtype in [np.uint8, np.uint16, np.float32], "image.dtype must be uint8, uint16, or float32; was %s"%(image.dtype))
    _enforce(image.ndim in [2, 3], "image must have either 2 or 3 dimensions; had %d."%(image.ndim))
    _enforce(image.size >= 1, "image must have at least one pixel; had none.")
    _enforce(type(maxval) == int or image.dtype == np.float32, "maxval must be an integer in [1, 65535]; was %r."%(maxval))
    _enforce(1 <= maxval <= 65535 or image.dtype == np.float32, "maxval must be an integer in [1, 65535]; was %r."%(maxval))
    _disallow(image.ndim == 3 and image.shape[2] != 3, "Color images must have exactly 3 channels.")
    _disallow(maxval > 255 and image.dtype == np.uint8, "maxval (%d) and image.dtype (%s) are inconsistent."%(maxval, image.dtype))
    _disallow(maxval <= 255 and image.dtype == np.uint16, "maxval (%d) and image.dtype (%s) are inconsistent."%(maxval, image.dtype))
    ## implementation
    filename = os.path.basename(filespec)            # "path/image.pgm" => "image.pgm"
    basename, filetype = os.path.splitext(filename)  # "image.pgm" => ("image", ".pgm")
    _enforce(len(basename) > 0, "filename `%s` must have at least 1 character + extension."%(filename))
    _enforce(len(filetype) > 3, "filename `%s` must have at least 1 character + extension."%(filename))
    if filetype in [".pnm", ".pgm", ".ppm", ".pfm"]:
        writer = {".pnm":pnm, ".pgm":pnm, ".ppm":pnm, ".pfm":pfm}[filetype]
        backup, writer.VERBOSE = writer.VERBOSE, verbose
        _reraise(lambda: writer.write(filespec, image, maxval))
        writer.VERBOSE = backup
    elif filetype in [".png", ".jpg", ".jpeg"]:
        _disallow(filetype in [".jpg", ".jpeg"] and maxval != 255, "maxval must be 255 for a JPEG; was %d."%(maxval))
        _disallow(filetype == ".png" and maxval not in [255, 65535], "maxval must be 255 or 65535 for a PNG; was %d."%(maxval))
        _disallow(image.ndim == 3 and maxval == 65535, "Writing 16-bit color PNGs is not supported yet.")
        _print(verbose, "Writing file %s "%(filespec), end='')
        _reraise(lambda: _imread.imsave(filespec, image))
        h, w = image.shape[:2]
        c = image.shape[2] if image.ndim > 2 else 1
        _print(verbose, "(w=%d, h=%d, c=%d, maxval=%d)"%(w, h, c, maxval))
    else:
        raise ImageIOError("Failed to write %s: Unrecognized file type `%s`."%(filespec, filetype))

class ImageIOError(RuntimeError):
    pass

######################################################################################
#
#  I N T E R N A L   F U N C T I O N S
#
######################################################################################

__all__ = ["imread", "imwrite"]

def _enforce(expression, errorMessageIfFalse, exceptionType=ImageIOError):
    if not expression:
        raise exceptionType(errorMessageIfFalse)

def _disallow(expression, errorMessageIfTrue, exceptionType=ImageIOError):
    if expression:
        raise exceptionType(errorMessageIfTrue)

def _reraise(func):
    try:
        return func()
    except Exception as e:
        raise ImageIOError(sys.exc_info()[1])

def _print(verbose, *args, **kwargs):
    if verbose:
        print(*args, **kwargs)

######################################################################################
#
#  U N I T   T E S T S
#
######################################################################################

import unittest

class _TestImgIo(unittest.TestCase):

   def assertRaises(self, excClass, callableObj, *args, **kwargs):
       class DummyError(Exception): pass
       excClass = DummyError
       try:
           unittest.TestCase.assertRaises(self, excClass, callableObj, *args, **kwargs)
       except:
           print("   %r"%(sys.exc_info()[1]))

   def test_exceptions(self):
       print("Testing exception handling...")
       shape = (7,11,3)
       pixels = np.random.random(shape)
       pixels8b = (pixels * 255).astype(np.uint8)
       pixels16b = (pixels * 65535).astype(np.uint16)
       imwrite("imgio.test.ppm", pixels8b, 255)
       imwrite("imgio.test.png", pixels8b, 255)
       imwrite("imgio.test.jpg", pixels8b, 255)
       os.rename("imgio.test.ppm", "invalidformat.pfm")
       os.rename("imgio.test.png", "invalidformat.jpg")
       os.rename("imgio.test.jpg", "invalidformat.ppm")
       self.assertRaises(ImageIOError, imread, "nonexisting.jpg")
       self.assertRaises(ImageIOError, imread, "nonexisting.png")
       self.assertRaises(ImageIOError, imread, "nonexisting.ppm")
       self.assertRaises(ImageIOError, imread, "nonexisting.pgm")
       self.assertRaises(ImageIOError, imread, "nonexisting.pfm")
       self.assertRaises(ImageIOError, imread, "invalidformat.pfm")
       self.assertRaises(ImageIOError, imread, "invalidformat.jpg")
       self.assertRaises(ImageIOError, imread, "invalidformat.ppm")
       self.assertRaises(ImageIOError, imread, "invalidtype.bmp")
       self.assertRaises(ImageIOError, imwrite, "invalidtype.bmp", pixels8b, 255)
       self.assertRaises(ImageIOError, imwrite, "invaliddepth.ppm", pixels16b, 255)
       self.assertRaises(ImageIOError, imwrite, "invaliddepth.png", pixels16b, 255)
       self.assertRaises(ImageIOError, imwrite, "invaliddepth.png", pixels8b, 254)
       self.assertRaises(ImageIOError, imwrite, "invaliddepth.jpg", pixels8b, 254)
       self.assertRaises(ImageIOError, imwrite, "invaliddepth.png", pixels16b, 1023)
       self.assertRaises(ImageIOError, imwrite, "invaliddepth.ppm", pixels8b, 1023)
       self.assertRaises(ImageIOError, imwrite, "", pixels8b, 255)
       self.assertRaises(ImageIOError, imwrite, ".ppm", pixels8b, 255)
       self.assertRaises(ImageIOError, imwrite, "imgio.test.ppm", pixels.astype(np.bool))
       self.assertRaises(ImageIOError, imwrite, "imgio.test.ppm", pixels.astype(np.float16))
       self.assertRaises(ImageIOError, imwrite, "imgio.test.png", pixels16b, 65535)
       os.remove("invalidformat.pfm")
       os.remove("invalidformat.jpg")
       os.remove("invalidformat.ppm")

   def test_png(self):
       for shape in [(1,1), (1,1,3), (7,11), (9,13,3), (123,321,3)]:
           for bpp in [8, 16]:
               if len(shape) == 3 and bpp == 16:  # skip 16-bit color mode
                   continue  # NB: Remove this check when imread.imsave() gets fixed!
               maxval = 2**bpp - 1
               tempfile = "imgio.test%db.png"%(bpp)
               print("Testing PNG reading & writing in %d-bit mode, shape=%r..."%(bpp, shape))
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
       for shape in [(1,1), (1,1,3), (7,11), (9,13,3), (123,321,3)]:
           for bpp in [8, 10, 12, 16]:
               maxval = 2**bpp - 1
               tempfile = "imgio.test%db.pnm"%(bpp)
               print("Testing PGM/PPM reading & writing in %d-bit mode, shape=%r..."%(bpp, shape))
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
       for shape in [(1,1), (1,1,3), (7,11), (9,13,3), (123,321,3)]:
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
       for shape in [(1,1), (1,1,3), (7,11), (9,13,3), (123,321,3)]:
           bpp = 32
           scale = 3.141
           tempfile = "imgio.test.pfm"
           print("Testing PFM reading & writing in %d-bit mode, shape=%r..."%(bpp, shape))
           pixels = np.random.random(shape)    # float64 pixels
           pixels = pixels.astype(np.float32)  # convert to float32
           imwrite(tempfile, pixels, maxval=scale, verbose=False)
           result, resscale = imread(tempfile, verbose=False)
           self.assertEqual(resscale, scale)
           self.assertEqual(result.dtype, np.float32)
           self.assertEqual(result.shape, shape)
           self.assertEqual(result.shape, shape)
           os.remove(tempfile)

   def test_verbose(self):
       print("Testing verbose mode...")
       for shape in [(7,11), (9,13,3)]:
           for ext in [".pnm", ".jpg", ".pfm", ".png"]:
               maxval = 255
               tempfile = "imgio.test%s"%(ext)
               pixels = np.random.random(shape)
               pixels = (pixels * maxval).astype(np.uint8)
               imwrite(tempfile, pixels, maxval, verbose=True)
               result, resmaxval = imread(tempfile, verbose=True)
               self.assertEqual(result.shape, shape)
               os.remove(tempfile)


if __name__ == "__main__":
    print("--" * 35)
    suite = unittest.TestLoader().loadTestsFromTestCase(_TestImgIo)
    unittest.TextTestRunner(verbosity=0).run(suite)
