#!/usr/bin/python -B

"""
Utility functions for reading and writing RGB and grayscale PFM files.
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
    Reads in a PFM file by the given name and returns its contents in a new
    numpy ndarray with float32 elements. The maximum (absolute) pixel value
    is also returned, mainly for consistency with the pnm (pgm/ppm) module.
    Both 1-channel and 3-channel images are supported, as well as both byte
    orders.
    """
    if os.path.isfile(filename):
        with open(filename, 'rb') as f:
            if VERBOSE:
                print("Reading %s"%(filename))
            buf = f.read()
            f32 = parse(buf)
            return f32, np.max(np.abs(f32))
    else:
        raise IOError("File %s does not exist."%(filename))

def write(filename, pixelArray, maxval=None, littleEndian=True):
    """
    Writes the contents of the given float32 ndarray into a 1- or 3-channel
    PFM file by the given name. Both little-endian and big-endian files are
    supported. The shape of the given array must be (h, w, c), where c is
    either 1 or 3. The 'maxval' argument is included only for compatibility
    with the pnm module; it has currently no effect.
    """
    pfmByteArray = generate(pixelArray, littleEndian)
    with open(filename, 'wb') as f:
        if VERBOSE:
            print("Writing %s"%(filename))
        f.write(pfmByteArray)

def parse(pfmByteArray):
    """
    Converts the given byte array, representing the contents of a PFM file, into
    a 1-channel or 3-channel numpy ndarray with float32 elements.
    """
    header, typestr, width, height, scale = re.search(
        b"(^(P[Ff])\s+(\d+)\s+(\d+)\s+([+-]?\d+(?:\.\d+)?)\s)", pfmByteArray).groups()
    width, height, scale = int(width), int(height), float(scale)
    numchannels = 3 if typestr == b"PF" else 1
    dtype = "<f" if scale < 0.0 else ">f"
    if VERBOSE:
        print("Parsing PFM data (w=%d, h=%d, c=%d, byteorder='%s')"%(width, height, numchannels, dtype[0]))
    f32 = np.frombuffer(pfmByteArray, dtype=dtype, count=width*height*numchannels, offset=len(header))
    f32 = f32.reshape((height, width) if numchannels == 1 else (height, width, 3))
    return f32

def generate(pixelArray, littleEndian=True):
    """
    Converts the given float32 ndarray into an immutable byte array representing
    the contents of a PFM file. The byte array can be written to disk as-is. Both
    1-channel and 3-channel images are supported, and the pixels can be written
    in little-endian or big-endian order. The shape of the given array must be
    either (h, w), representing grayscale data, or (h, w, c), where c is either
    1 or 3, for grayscale and color data, respectively.
    """
    assert pixelArray.ndim in [2, 3], "pixel array must not have ndim == %d"%(pixelArray.ndim)
    assert pixelArray.ndim == 2 or pixelArray.shape[2] in [1, 3]
    numchannels = 1 if pixelArray.ndim == 2 else pixelArray.shape[2]
    typestr = "PF" if numchannels == 3 else "Pf"
    width = pixelArray.shape[1]
    height = pixelArray.shape[0]
    f32 = pixelArray.astype(np.float32)
    if littleEndian:
        byteorder = "<"
        scale = -1.0
        f32bs = f32
    else:
        byteorder = ">"
        scale = 1.0
        f32bs = f32.byteswap()
    if VERBOSE:
        print("Generating PFM data (w=%d, h=%d, c=%d, byteorder='%s')"%(width, height, numchannels, byteorder))
    pfmByteArray = bytearray("%s %d %d %.1f\n"%(typestr, width, height, scale), 'utf-8')
    pfmByteArray.extend(f32bs.flatten())
    return bytes(pfmByteArray)
