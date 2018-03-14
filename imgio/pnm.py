"""
Utility functions for reading and writing RGB and grayscale PGM/PPM files.
"""

from __future__ import print_function as __print

import re
import numpy as np

# pylint: disable=no-member
# pylint: disable=too-many-locals

######################################################################################
#
#  P U B L I C   A P I
#
######################################################################################

def read(filespec, verbose=False):
    """
    Reads in a PGM/PPM file by the given name and returns its contents in a new numpy
    ndarray with 8/16-bit elements. Also returns the maximum representable value of a
    pixel (typically 255, 1023, 4095, or 65535).
    """
    valid_extensions = [".pnm", ".ppm", ".pgm", ".PNM", ".PPM", ".PGM"]
    __enforce(isinstance(filespec, str) and len(filespec) >= 5, "filespec must be a string of length >= 5, was %r."%(filespec))
    __enforce(filespec[-4:] in valid_extensions, "file extension must be .pnm, .ppm, or .pgm; was %s."%(filespec[-4:]))
    with open(filespec, "rb") as f:
        buf = f.read()
        regex_pnm_header = b"(^(P[56])\\s+(\\d+)\\s+(\\d+)\\s+(\\d+)\\s)"
        match = re.search(regex_pnm_header, buf)
        if match is not None:
            header, typestr, width, height, maxval = match.groups()
            width, height, maxval = int(width), int(height), int(maxval)
            numch = 3 if typestr == b"P6" else 1
            shape = (height, width, numch) if typestr == b"P6" else (height, width)
            if verbose:
                print("Reading file %s "%(filespec), end='')
                print("(w=%d, h=%d, c=%d, maxval=%d)"%(width, height, numch, maxval))
            dtype = ">u2" if maxval > 255 else np.uint8
            pixels = np.frombuffer(buf, dtype, count=width * height * numch, offset=len(header))
            pixels = pixels.reshape(shape).astype(np.uint8 if maxval <= 255 else np.uint16)
            return pixels, maxval
        else:
            raise RuntimeError("File %s is not a valid PGM/PPM file."%(filespec))

def write(filespec, image, maxval, verbose=False):
    """
    Writes the contents of the given 8/16-bit numpy ndarray into a PGM/PPM file by the
    given name. The dtype of image must be consistent with maxval, i.e., np.uint8 for
    maxval <= 255, and np.uint16 otherwise.
    """
    color_extensions = [".pnm", ".ppm", ".PNM", ".PPM"]
    gray_extensions = [".pnm", ".pgm", ".PNM", ".PGM"]
    __enforce(isinstance(filespec, str) and len(filespec) >= 5, "filespec must be a string of length >= 5, was %r."%(filespec))
    __enforce(isinstance(image, np.ndarray), "image must be a NumPy ndarray; was %r."%(type(image)))
    __enforce(image.dtype in [np.uint8, np.uint16], "image.dtype must be uint8 or uint16; was %s."%(image.dtype))
    __enforce(image.ndim in [2, 3], "image must have either 2 or 3 dimensions; had %d."%(image.ndim))
    __enforce(image.size >= 1, "image must have at least one pixel; had none.")
    __enforce(isinstance(maxval, int) and 1 <= maxval <= 65535, "maxval must be an integer in [1, 65535]; was %r."%(maxval))
    __enforce(filespec[-4:] in color_extensions or image.ndim == 2, "file extension must be .pnm or .ppm; was %s."%(filespec[-4:]))
    __enforce(filespec[-4:] in gray_extensions or image.ndim == 3, "file extension must be .pnm or .pgm; was %s."%(filespec[-4:]))
    __disallow(image.ndim == 3 and image.shape[2] != 3, "color images must have exactly 3 channels")
    __disallow(maxval > 255 and image.dtype == np.uint8, "maxval (%d) and image.dtype (%s) are inconsistent."%(maxval, image.dtype))
    __disallow(maxval <= 255 and image.dtype == np.uint16, "maxval (%d) and image.dtype (%s) are inconsistent."%(maxval, image.dtype))
    height, width = image.shape[:2]
    numch = 3 if image.ndim == 3 else 1
    image = image.byteswap() if maxval > 255 else image
    if verbose:
        print("Writing file %s "%(filespec), end='')
        print("(w=%d, h=%d, c=%d, maxval=%d)"%(width, height, numch, maxval))
    with open(filespec, "wb") as f:
        typestr = "P6" if numch == 3 else "P5"
        f.write(("%s %d %d %d\n"%(typestr, width, height, maxval)).encode("utf-8"))
        f.write(image)

######################################################################################
#
#  I N T E R N A L   F U N C T I O N S
#
######################################################################################

def __enforce(expression, error_msg_if_false, exception_type=RuntimeError):
    if not expression:
        raise exception_type(error_msg_if_false)

def __disallow(expression, error_msg_if_true, exception_type=RuntimeError):
    if expression:
        raise exception_type(error_msg_if_true)
