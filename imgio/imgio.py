#!/usr/bin/python3 -B

"""
Easy image file reading & writing. Supports PGM/PPM/PNM/PFM/PNG/BMP/JPG/TIFF/EXR/RAW.

Example:
  image, maxval = imgio.imread("foo.png")
  imgio.imwrite("foo.ppm", image, maxval)
"""

import os                         # standard library
import sys                        # standard library
import unittest                   # standard library

import piexif                     # pip install piexif
import numpy as np                # pip install numpy
import imread as _imread          # pip install imread
import pyexr                      # pip install pyexr

try:
    # package mode
    from imgio import pnm         # local import: pnm.py
    from imgio import pfm         # local import: pfm.py
except ImportError:
    # stand-alone mode
    import pnm                    # local import: pnm.py
    import pfm                    # local import: pfm.py

######################################################################################
#
#  P U B L I C   A P I
#
######################################################################################

RW_FORMATS = [".pnm", ".pgm", ".ppm", ".pfm", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".insp", ".raw"]
RO_FORMATS = RW_FORMATS + [".exr", ".bmp"]

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
    if filetype == ".pfm":
        frame, scale = _reraise(lambda: pfm.read(filespec, verbose))
        return frame, scale
    if filetype == ".exr":
        frame, maxval = _reraise(lambda: _read_exr(filespec, verbose))
        return frame, maxval
    if filetype in [".pnm", ".pgm", ".ppm"]:
        frame, maxval = _reraise(lambda: pnm.read(filespec, verbose))
        return frame, maxval
    if filetype in [".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg", ".insp"]:
        _print(verbose, "Reading file %s "%(filespec), end='')
        formatstr = "jpg" if filetype == ".insp" else filetype[1:]
        frame = _reraise(lambda: _imread.imread(filespec, formatstr=formatstr))
        maxval = 255 if frame.dtype == np.uint8 else 65535
        must_squeeze = (frame.ndim > 2 and frame.shape[2] == 1)
        frame = frame.squeeze(axis=2) if must_squeeze else frame
        frame = _reraise(lambda: _exif_rotate(frame, filespec))
        h, w = frame.shape[:2]
        c = frame.shape[2] if frame.ndim > 2 else 1
        _print(verbose, "(w=%d, h=%d, c=%d, maxval=%d)"%(w, h, c, maxval))
        return frame, maxval
    raise ImageIOError("unrecognized file type `%s`."%(filetype))

def imwrite(filespec, image, maxval=255, packed=False, verbose=False):
    """
    Writes the given image to the given file, returns nothing. Grayscale images
    are expected to be provided as NumPy arrays with shape H x W, color images
    with shape H x W x 3. Metadata, alpha channels, etc. are not supported.
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
    _disallow(image.ndim not in [2, 3], "image.shape must be (m, n) or (m, n, 3); was %s."%(str(image.shape)))
    _disallow(image.ndim == 3 and image.shape[2] != 3, "image.shape must be (m, n) or (m, n, 3); was %s."%(str(image.shape)))
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
        _enforce(maxval >= 0.0, "maxval (scale) must be non-negative; was %s."%(repr(maxval)))
        _reraise(lambda: pfm.write(filespec, image, maxval, little_endian=True, verbose=verbose))
    elif filetype in [".pnm", ".pgm", ".ppm"]:
        _reraise(lambda: pnm.write(filespec, image, maxval, verbose))
    elif filetype in [".png", ".tif", ".tiff", ".jpg", ".jpeg", ".insp"]:
        _disallow(filetype in [".jpg", ".jpeg"] and maxval != 255, "maxval must be 255 for a JPEG; was %d."%(maxval))
        _disallow(filetype == ".png" and maxval not in [255, 65535], "maxval must be 255 or 65535 for a PNG; was %d."%(maxval))
        _print(verbose, "Writing file %s "%(filespec), end='')
        formatstr = "jpg" if filetype == ".insp" else filetype[1:]
        _reraise(lambda: _imread.imsave(filespec, image, formatstr=formatstr))
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
    except Exception as e:
        raise ImageIOError("%s"%(repr(sys.exc_info()[1]))) from e

def _print(verbose, *args, **kwargs):
    if verbose:
        print(*args, **kwargs)

def _exif_rotate(img, filespec):
    try:
        orientation = 1
        exif_dict = piexif.load(filespec).pop("0th")
        exif_orientation = exif_dict.get(piexif.ImageIFD.Orientation)
        orientation = 1 if exif_orientation is None else exif_orientation
    except piexif.InvalidImageDataError:
        pass
    exif_to_rot90 = {1: 0, 2: 0, 3: 2, 4: 0, 5: 1, 6: 3, 7: 3, 8: 1}
    if orientation in [2, 5, 7]:
        img = np.fliplr(img)
    if orientation in [4]:
        img = np.flipud(img)
    if orientation in exif_to_rot90:
        rot90_ccw_steps = exif_to_rot90[orientation]
        img = np.rot90(img, rot90_ccw_steps)  # 0/90/180/270 CCW
    return img

def _read_exr(filespec, verbose=True):
    exr = pyexr.open(filespec)
    data = exr.get()
    maxval = np.max(data)
    _print(verbose, "Reading OpenEXR file %s "%(filespec), end='')
    _print(verbose, "(w=%d, h=%d, c=%d, %s)"%(exr.width, exr.height, len(exr.channels), data.dtype))
    return data, maxval

def _read_raw(filespec, width, height, bpp, header_size=None, verbose=False):
    # Warning: hardcoded endianness (x86)
    with open(filespec, "rb") as infile:
        buf = infile.read()
        shape = (height, width)
        maxval = 2 ** bpp - 1
        wordsize = 2 if bpp > 8 else 1
        packed = len(buf) < (width * height * wordsize)
        if header_size is None:
            header_size = len(buf) - (width * height * wordsize)
        _print(verbose, "Reading raw Bayer file %s "%(filespec), end='')
        _print(verbose, "(w=%d, h=%d, maxval=%d, header=%d, packed=%r)"%(width, height, maxval, header_size, packed))
        if not packed:
            dtype = "<u2" if bpp > 8 else np.uint8
            pixels = np.frombuffer(buf, dtype, count=width * height, offset=header_size)
            pixels = pixels.reshape(shape).astype(np.uint8 if bpp <= 8 else np.uint16)
        else:
            # TODO: packed raw support
            raise ImageIOError("Packed RAW reading not implemented yet!")
        return pixels, maxval

def _write_raw(filespec, image, _maxval, _pack=False, _verbose=False):
    # TODO: packed raw support
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

    def assertRaisesRegexp(self, expected_exception, expected_regex, *args, **kwargs):
        """
        Checks that the correct type of exception is raised, and that the exception
        message matches the given regular expression. Also prints out the message
        for visual inspection.
        """
        try:  # check the type of exception first
            unittest.TestCase.assertRaises(self, expected_exception, *args, **kwargs)
        except Exception as e:
            raised_name = sys.exc_info()[0].__name__
            expected_name = expected_exception.__name___
            errstr = "Expected %s with a message matching '%s', got %s."%(expected_name, expected_regex, raised_name)
            print("   FAIL: %s"%(errstr))
            raise AssertionError(errstr) from e
        try:  # then check the exception message
            assertRaisesRegex = getattr(unittest.TestCase, "assertRaisesRegex", unittest.TestCase.assertRaisesRegexp)
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

    def test_exceptions(self):
        # pylint: disable=too-many-statements
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
        self.assertRaisesRegexp(ImageIOError, "^Failed to read", imread, None)
        self.assertRaisesRegexp(ImageIOError, "^Failed to read", imread, 0xdeadbeef)
        self.assertRaisesRegexp(ImageIOError, "^Failed to read", imread, "")
        self.assertRaisesRegexp(ImageIOError, "^Failed to read", imread, ".ppm")
        self.assertRaisesRegexp(ImageIOError, "^Failed to read", imread, "invalidfilename")
        self.assertRaisesRegexp(ImageIOError, "^Failed to read", imread, "nonexisting/")
        self.assertRaisesRegexp(ImageIOError, "^Failed to read", imread, "nonexisting/.ppm")
        self.assertRaisesRegexp(ImageIOError, "^Failed to read", imread, "nonexisting/invalidfilename")
        self.assertRaisesRegexp(ImageIOError, "^Failed to read", imread, "nonexisting.jpg")
        self.assertRaisesRegexp(ImageIOError, "^Failed to read", imread, "nonexisting.png")
        self.assertRaisesRegexp(ImageIOError, "^Failed to read", imread, "nonexisting.ppm")
        self.assertRaisesRegexp(ImageIOError, "^Failed to read", imread, "nonexisting.pgm")
        self.assertRaisesRegexp(ImageIOError, "^Failed to read", imread, "nonexisting.pfm")
        self.assertRaisesRegexp(ImageIOError, "^Failed to read", imread, "nonexisting.bmp")
        self.assertRaisesRegexp(ImageIOError, "^Failed to read", imread, "invalidformat.pfm")
        self.assertRaisesRegexp(ImageIOError, "^Failed to read", imread, "invalidformat.jpg")
        self.assertRaisesRegexp(ImageIOError, "^Failed to read", imread, "invalidformat.ppm")
        self.assertRaisesRegexp(ImageIOError, "^Failed to read.*verbose", imread, "validimage.ppm", verbose="foo")
        self.assertRaisesRegexp(ImageIOError, "^Failed to write", imwrite, "invalidfilename", pixels8b, 255)
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
        self.assertRaisesRegexp(ImageIOError, "^Failed to write", imwrite, "imgio.test.ppm", pixels.astype(bool))
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

    def test_tiff(self):
        for shape in [(1, 1), (1, 1, 3), (7, 11), (9, 13, 3), (123, 321, 3)]:
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
                self.assertEqual(result.shape, shape)
                self.assertEqual(result.tolist(), pixels.tolist())
                os.remove(tempfile)

    def test_jpg(self):
        for shape in [(1, 1), (1, 1, 3), (7, 11), (9, 13, 3), (123, 321, 3)]:
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
            self.assertTrue(np.allclose(result, pixels))
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

    def test_exr(self):
        print("Testing EXR reading...")
        thispath = os.path.dirname(os.path.abspath(__file__))
        filespec = os.path.join(thispath, "test-images", "GrayRampsDiagonal.exr")
        img, maxval = imread(filespec)
        self.assertEqual(img.shape, (800, 800, 1))
        self.assertEqual(img.dtype, np.float32)
        self.assertEqual(maxval, np.max(img))

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
                    result, resmaxval = imread(tempfile, width=shape[1], height=shape[0], bpp=bpp)
                    self.assertEqual(resmaxval, maxval)
                    self.assertEqual(result.dtype, dtype)
                    self.assertEqual(result.shape, shape)
                    self.assertEqual(result.tolist(), pixels.tolist())
                    os.remove(tempfile)

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
