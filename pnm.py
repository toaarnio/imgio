#!/usr/bin/python -B

"""
Utility functions for reading and writing RGB and grayscale PGM/PPM files.
"""

from __future__ import print_function

import sys, os, re
import numpy as np

######################################################################################
#
#  P U B L I C   A P I
#
######################################################################################

VERBOSE = True

def read(filename):
    """
    Reads in a PGM/PPM file by the given name and returns its contents in a new numpy
    ndarray with 8/16-bit elements. Also returns the maximum representable value of a
    pixel (typically 255, 1023, 4095, or 65535).
    """
    with open(filename, "rb") as f:
        buf = f.read()
        regexPnmHeader = b"(^(P[56])\s+(\d+)\s+(\d+)\s+(\d+)\s)"
        match = re.search(regexPnmHeader, buf)
        if match is not None:
            header, typestr, width, height, maxval = match.groups()
            width, height, maxval = int(width), int(height), int(maxval)
            numch = 3 if typestr == b"P6" else 1
            shape = (height, width, numch) if typestr == b"P6" else (height, width)
            if VERBOSE:
                print("Reading file %s "%(filename), end='')
                print("(w=%d, h=%d, c=%d, maxval=%d)"%(width, height, numch, maxval))
            dtype = ">u2" if maxval > 255 else np.uint8
            pixels = np.frombuffer(buf, dtype, count=width*height*numch, offset=len(header))
            pixels = pixels.reshape(shape).astype(np.uint8 if maxval <= 255 else np.uint16)
            return pixels, maxval
        else:
            raise RuntimeError("File %s is not a valid PGM/PPM file."%(filename))

def write(filename, pixelArray, maxval=1023):
    """
    Writes the contents of the given 8/16-bit numpy ndarray into a PGM/PPM file by the
    given name.
    """
    width = pixelArray.shape[1]
    height = pixelArray.shape[0]
    numch = pixelArray.shape[2] if pixelArray.ndim == 3 else 1
    if maxval > 255:
        pixels = pixelArray.astype(np.uint16)
        pixels = pixels.byteswap()
    else:
        pixels = pixelArray.astype(np.uint8)
    if VERBOSE:
        print("Writing file %s "%(filename), end='')
        print("(w=%d, h=%d, c=%d, maxval=%d)"%(width, height, numch, maxval))
    with open(filename, "wb") as f:
        typestr = "P6" if numch == 3 else "P5"
        f.write(("%s %d %d %d\n"%(typestr, width, height, maxval)).encode("utf-8"))
        f.write(pixels)
