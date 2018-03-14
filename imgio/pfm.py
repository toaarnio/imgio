"""
Utility functions for reading and writing RGB and grayscale PFM files.
"""

from __future__ import print_function as __print

import re
import numpy as np

######################################################################################
#
#  P U B L I C   A P I
#
######################################################################################

def read(filename, verbose=False):
    """
    Reads in a PFM file by the given name and returns its contents in a new
    numpy ndarray with float32 elements. The absolute value of the 'scale
    factor' attribute is also returned. Both 1-channel and 3-channel images
    are supported, as well as both byte orders.
    """
    with open(filename, 'rb') as f:
        if verbose:
            print("Reading file %s "%(filename), end='')
        buf = f.read()
        parsed = parse(buf, verbose)
        if parsed is not None:
            pixels, scale = parsed
            return pixels, scale
        else:
            raise RuntimeError("File %s is not a valid PFM file."%(filename))

def write(filename, pixels, scale=1.0, little_endian=True, verbose=False):
    """
    Writes the contents of the given float32 ndarray into a 1- or 3-channel
    PFM file by the given name. Both little-endian and big-endian files are
    supported. The shape of the given array must be (h, w, c), where c is
    either 1 or 3.
    """
    with open(filename, 'wb') as f:
        if verbose:
            print("Writing file %s "%(filename), end='')
        pfm_bytearray = generate(pixels, scale, little_endian, verbose)
        f.write(pfm_bytearray)

def parse(pfm_bytearray, verbose=False):
    """
    Converts the given byte array, representing the contents of a PFM file, into
    a 1-channel or 3-channel numpy ndarray with float32 elements. Returns a tuple
    of (pixels, scale), where 'pixels' is the array and 'scale' is the absolute
    value of the scale factor attribute extracted from the header. Returns None
    if the file cannot be parsed.
    """
    regex_pfm_header = b"(^(P[Ff])\\s+(\\d+)\\s+(\\d+)\\s+([+-]?\\d+(?:\\.\\d+)?)\\s)"
    match = re.search(regex_pfm_header, pfm_bytearray)
    if match is not None:
        header, typestr, width, height, scale = match.groups()
        width, height, scale = int(width), int(height), float(scale)
        numchannels = 3 if typestr == b"PF" else 1
        dtype = "<f" if scale < 0.0 else ">f"
        scale = abs(scale)
        if verbose:
            print("(w=%d, h=%d, c=%d, scale=%.3f, byteorder='%s')"%(width, height, numchannels, scale, dtype[0]))
        f32 = np.frombuffer(pfm_bytearray, dtype=dtype, count=width * height * numchannels, offset=len(header))
        f32 = f32.reshape((height, width) if numchannels == 1 else (height, width, 3))
        f32 = f32.astype(np.float32)  # pylint: disable=no-member
        return f32, scale
    return None

def generate(pixels, scale=1.0, little_endian=True, verbose=False):
    """
    Converts the given float32 ndarray into an immutable byte array representing
    the contents of a PFM file. The byte array can be written to disk as-is. Both
    1-channel and 3-channel images are supported, and the pixels can be written
    in little-endian or big-endian order. The shape of the given array must be
    either (h, w), representing grayscale data, or (h, w, c), where c is either
    1 or 3, for grayscale and color data, respectively.
    """
    assert pixels.ndim in [2, 3], "pixel array must not have ndim == %d"%(pixels.ndim)
    assert pixels.ndim == 2 or pixels.shape[2] in [1, 3]
    numchannels = 1 if pixels.ndim == 2 else pixels.shape[2]
    typestr = "PF" if numchannels == 3 else "Pf"
    width = pixels.shape[1]
    height = pixels.shape[0]
    f32 = pixels.astype(np.float32)  # pylint: disable=no-member
    if little_endian:
        byteorder = "<"
        scale = -scale
        f32bs = f32
    else:
        byteorder = ">"
        scale = scale
        f32bs = f32.byteswap()
    if verbose:
        print("(w=%d, h=%d, c=%d, scale=%.3f, byteorder='%s')"%(width, height, numchannels, abs(scale), byteorder))
    pfm_bytearray = bytearray("%s %d %d %.3f\n"%(typestr, width, height, scale), 'utf-8')
    pfm_bytearray.extend(f32bs.flatten())
    return bytes(pfm_bytearray)
