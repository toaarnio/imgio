#!/usr/bin/python -B

"""
A collection of utility functions for processing Bayer raw and demosaiced RGB
image data using NumPy.
"""

from __future__ import print_function

import os
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
    basename = os.path.basename(filespec)     # "path/image.pgm" => "image.pgm"
    filetype = os.path.splitext(basename)[1]  # "image.pgm" => ".pgm"
    if filetype in [".pgm", ".ppm", ".pfm"]:
        reader = {".pgm":pnm, ".ppm":pnm, ".pfm":pfm}[filetype]
        backup, reader.VERBOSE = reader.VERBOSE, verbose
        frame, maxval = reader.read(filespec)
        reader.VERBOSE = backup
    elif filetype in [".png", ".jpg"]:
        if verbose:
            print("Reading file %s "%(filespec), end='')
        frame = _imread.imread(filespec)
        maxval = 255 if frame.dtype == np.uint8 else 65535
        h, w = frame.shape[:2]
        c = frame.shape[2] if frame.ndim > 2 else 1
        if verbose:
            print("(w=%d, h=%d, c=%d, maxval=%d)"%(w, h, c, maxval))
    else:
        raise RuntimeError("Failed to read %s: Unrecognized file type."%(basename))
    return frame, maxval

def imwrite(filespec, frame, maxval=None, verbose=False):
    """
    Writes the given frame to the given file. Grayscale images are expected to be
    provided as NumPy arrays with shape H x W, color images with shape H x W x 3.
    """
    basename = os.path.basename(filespec)     # "path/image.pgm" => "image.pgm"
    filetype = os.path.splitext(basename)[1]  # "image.pgm" => ".pgm"
    if filetype in [".pgm", ".ppm", ".pfm"]:
        writer = {".pgm":pnm, ".ppm":pnm, ".pfm":pfm}[filetype]
        backup, writer.VERBOSE = writer.VERBOSE, verbose
        writer.write(filespec, frame, maxval)
        writer.VERBOSE = backup
    elif filetype in [".png", ".jpg"]:
        if maxval is not None and maxval != 255:
            raise RuntimeError("PNG/JPG writing currently works on 8-bit data only.")
        if verbose:
            print("Writing file %s "%(filespec), end='')
        frame = frame.astype(np.uint8)
        _imread.imsave(filespec, frame)
        h, w = frame.shape[:2]
        c = frame.shape[2] if frame.ndim > 2 else 1
        if verbose:
            print("(w=%d, h=%d, c=%d, maxval=255)"%(w, h, c))
    else:
        raise RuntimeError("Failed to write %s: Unrecognized file type."%(basename))

######################################################################################
#
#  U N I T   T E S T S
#
######################################################################################

import sys, unittest

class _TestImgIo(unittest.TestCase):

   def test_png(self):
       for bpp in [8]:  # NB: Add 16-bit when imread gets fixed
           maxval = 2**bpp - 1
           print("Testing PNG reading & writing in %d-bit mode..."%(bpp))
           tempfile = "rawutils.test%db.png"%(bpp)
           dtype = np.uint8 if bpp == 8 else np.uint16
           pixels = np.random.random((7, 11, 3))  # 11 x 7 pixels
           pixels = (pixels * maxval).astype(np.uint16)
           imwrite(tempfile, pixels, maxval, verbose=False)
           result, resmaxval = imread(tempfile, verbose=False)
           self.assertEqual(resmaxval, maxval)
           self.assertEqual(result.dtype, dtype)
           self.assertEqual(result.shape, (7, 11, 3))
           self.assertEqual(pixels.tolist(), result.tolist())
           os.remove(tempfile)

   def test_ppm(self):
       for bpp in [8, 10, 12, 16]:
           maxval = 2**bpp - 1
           print("Testing PPM reading & writing in %d-bit mode..."%(bpp))
           tempfile = "rawutils.test%db.ppm"%(bpp)
           dtype = np.uint8 if bpp == 8 else np.uint16
           pixels = np.random.random((7, 11, 3))  # 11 x 7 pixels
           pixels = (pixels * maxval).astype(np.uint16)
           imwrite(tempfile, pixels, maxval, verbose=False)
           result, resmaxval = imread(tempfile, verbose=False)
           self.assertEqual(resmaxval, maxval)
           self.assertEqual(result.dtype, dtype)
           self.assertEqual(result.shape, (7, 11, 3))
           self.assertEqual(pixels.tolist(), result.tolist())
           os.remove(tempfile)

   def test_pgm(self):
       for bpp in [8, 10, 12, 16]:
           maxval = 2**bpp - 1
           print("Testing PGM reading & writing in %d-bit mode..."%(bpp))
           tempfile = "rawutils.test%db.pgm"%(bpp)
           dtype = np.uint8 if bpp == 8 else np.uint16
           pixels = np.random.random((7, 11))  # 11 x 7 pixels
           pixels = (pixels * maxval).astype(np.uint16)
           imwrite(tempfile, pixels, maxval, verbose=False)
           result, resmaxval = imread(tempfile, verbose=False)
           self.assertEqual(resmaxval, maxval)
           self.assertEqual(result.dtype, dtype)
           self.assertEqual(result.shape, (7, 11))
           self.assertEqual(pixels.tolist(), result.tolist())
           os.remove(tempfile)

   def test_jpg(self):
       print("Testing JPG reading & writing...")
       tempfile = "rawutils.test.jpg"
       rgb = np.ones((7, 11, 3))  # 11 x 7 pixels, constant values
       rgb = (rgb * 127).astype(np.uint8)
       imwrite(tempfile, rgb, maxval=255, verbose=False)
       result, maxval = imread(tempfile, verbose=False)
       self.assertEqual(result.dtype, np.uint8)
       self.assertEqual(result.shape, (7, 11, 3))
       self.assertEqual(rgb[..., 0].tolist(), result[..., 0].tolist())
       self.assertEqual(rgb[..., 1].tolist(), result[..., 1].tolist())
       self.assertEqual(rgb[..., 2].tolist(), result[..., 2].tolist())
       os.remove(tempfile)

   def test_pfm(self):
       print("Testing PFM reading & writing...")
       expected = np.random.random((7, 11, 3))  # 11 x 7 float64 pixels
       expected = expected.astype(np.float32)   # convert to float32
       imwrite("pfmtest_le.pfm", expected, maxval=1e6)
       imwrite("pfmtest_be.pfm", expected, maxval=None)
       result_le, maxval_le = imread("pfmtest_le.pfm")
       result_be, maxval_be = imread("pfmtest_be.pfm")
       self.assertEqual(np.max(expected), maxval_le)
       self.assertEqual(np.max(expected), maxval_be)
       self.assertEqual(result_le.dtype, np.float32)
       self.assertEqual(result_be.dtype, np.float32)
       self.assertEqual(result_le.shape, (7, 11, 3))
       self.assertEqual(result_be.shape, (7, 11, 3))
       self.assertEqual(expected[..., 0].tolist(), result_le[..., 0].tolist())
       self.assertEqual(expected[..., 1].tolist(), result_le[..., 1].tolist())
       self.assertEqual(expected[..., 2].tolist(), result_le[..., 2].tolist())
       self.assertEqual(expected[..., 0].tolist(), result_be[..., 0].tolist())
       self.assertEqual(expected[..., 1].tolist(), result_be[..., 1].tolist())
       self.assertEqual(expected[..., 2].tolist(), result_be[..., 2].tolist())
       os.remove("pfmtest_le.pfm")
       os.remove("pfmtest_be.pfm")

   def test_png_16bit(self):  # NB: This is currently failing!
       print("Testing PNG reading & writing in 16-bit mode...")
       tempfile = "rawutils.test16b.png"
       rgb = np.random.random((5, 13, 3))  # 13 x 5 pixels
       rgb = (rgb * 65535).astype(np.uint16)
       imwrite(tempfile, rgb, maxval=65535, verbose=False)
       result, maxval = imread(tempfile, verbose=False)
       self.assertEqual(maxval, 65535)
       self.assertEqual(result.dtype, np.uint16)
       self.assertEqual(result.shape, (5, 13, 3))
       self.assertEqual(rgb[..., 0].tolist(), result[..., 0].tolist())
       self.assertEqual(rgb[..., 1].tolist(), result[..., 1].tolist())
       self.assertEqual(rgb[..., 2].tolist(), result[..., 2].tolist())
       os.remove(tempfile)


if __name__ == "__main__":
    print("--" * 35)
    suite = unittest.TestLoader().loadTestsFromTestCase(_TestImgIo)
    unittest.TextTestRunner(verbosity=0).run(suite)
