#!/usr/bin/python -B

"""
Easy image file reading & writing. Supports PGM/PPM/PFM/PNG/JPG.

Example:
  image, maxval = imgio.imread("foo.png")
  imgio.imwrite("foo.ppm", image, maxval)
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
    if filetype in [".pnm", ".pgm", ".ppm", ".pfm"]:
        reader = {".pnm":pnm, ".pgm":pnm, ".ppm":pnm, ".pfm":pfm}[filetype]
        backup, reader.VERBOSE = reader.VERBOSE, verbose
        frame, maxval = reader.read(filespec)
        reader.VERBOSE = backup
        return frame, maxval
    elif filetype in [".png", ".jpg", ".jpeg"]:
        if verbose:
            print("Reading file %s "%(filespec), end='')
        frame = _imread.imread(filespec)
        maxval = 255 if frame.dtype == np.uint8 else 65535
        h, w = frame.shape[:2]
        c = frame.shape[2] if frame.ndim > 2 else 1
        if verbose:
            print("(w=%d, h=%d, c=%d, maxval=%d)"%(w, h, c, maxval))
        mustSqueeze = (frame.ndim > 2 and frame.shape[2] == 1)
        frame = frame.squeeze(axis=2) if mustSqueeze else frame
        return frame, maxval
    else:
        raise RuntimeError("Failed to read %s: Unrecognized file type."%(basename))


def imwrite(filespec, frame, maxval=None, verbose=False):
    """
    Writes the given frame to the given file. Grayscale images are expected to be
    provided as NumPy arrays with shape H x W, color images with shape H x W x 3.
    """
    basename = os.path.basename(filespec)     # "path/image.pgm" => "image.pgm"
    filetype = os.path.splitext(basename)[1]  # "image.pgm" => ".pgm"
    if filetype in [".pnm", ".pgm", ".ppm", ".pfm"]:
        writer = {".pnm":pnm, ".pgm":pnm, ".ppm":pnm, ".pfm":pfm}[filetype]
        backup, writer.VERBOSE = writer.VERBOSE, verbose
        writer.write(filespec, frame, maxval)
        writer.VERBOSE = backup
    elif filetype in [".png", ".jpg", ".jpeg"]:
        if maxval is not None and maxval != 255:
            raise NotImplementedError("PNG/JPG writing currently works on 8-bit data only.")
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

import unittest

class _TestImgIo(unittest.TestCase):

   def test_exceptions(self):
       print("Testing exception handling...")
       maxval = 255
       shape = (7,11,3)
       pixels = np.random.random(shape)
       pixels8b = (pixels * maxval).astype(np.uint8)
       imwrite("imgio.test.ppm", pixels8b, maxval)
       imwrite("imgio.test.png", pixels8b, maxval)
       imwrite("imgio.test.jpg", pixels8b, maxval)
       os.rename("imgio.test.ppm", "invalidformat.pfm")
       os.rename("imgio.test.png", "invalidformat.jpg")
       os.rename("imgio.test.jpg", "invalidformat.ppm")
       self.assertRaises(EnvironmentError, imread, "nonexisting.jpg")
       self.assertRaises(EnvironmentError, imread, "nonexisting.png")
       self.assertRaises(EnvironmentError, imread, "nonexisting.ppm")
       self.assertRaises(EnvironmentError, imread, "nonexisting.pgm")
       self.assertRaises(EnvironmentError, imread, "nonexisting.pfm")
       self.assertRaises(RuntimeError, imread, "invalidformat.pfm")
       self.assertRaises(RuntimeError, imread, "invalidformat.jpg")
       self.assertRaises(RuntimeError, imread, "invalidformat.ppm")
       self.assertRaises(RuntimeError, imread, "unknowntype.bmp")
       self.assertRaises(RuntimeError, imwrite, "unknowntype.bmp", pixels)
       self.assertRaises(NotImplementedError, imwrite, "imgio.test.png", pixels8b, maxval=1023)
       os.remove("invalidformat.pfm")
       os.remove("invalidformat.jpg")
       os.remove("invalidformat.ppm")

   def test_png(self):
       for shape in [(1,1), (1,1,3), (7,11), (9,13,3), (123,321,3)]:
           for bpp in [8]:  # NB: Add 16-bit when imread.imsave() gets fixed
               maxval = 2**bpp - 1
               tempfile = "imgio.test%db.png"%(bpp)
               print("Testing PNG reading & writing in %d-bit mode, shape=%r..."%(bpp, shape))
               dtype = np.uint8 if bpp == 8 else np.uint16
               pixels = np.random.random(shape)
               pixels = (pixels * maxval).astype(np.uint16)
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
               pixels = (pixels * maxval).astype(np.uint16)
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

   def test_png_16bit(self):  # NB: Remove this when imread.imsave() gets fixed
       print("Testing PNG reading & writing in 16-bit mode...")
       tempfile = "imgio.test16b.png"
       rgb = np.random.random((5, 13, 3))
       rgb = (rgb * 65535).astype(np.uint16)
       imwrite(tempfile, rgb, maxval=65535, verbose=False)


if __name__ == "__main__":
    print("--" * 35)
    suite = unittest.TestLoader().loadTestsFromTestCase(_TestImgIo)
    unittest.TextTestRunner(verbosity=0).run(suite)
