import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

import imgio
from imgio import imread, imread_f16, imread_f32, imread_f64, imwrite, rawread, ImageIOError
from imgio.imgio import freeimage

IMAGES_DIR = Path(__file__).parent / "images"


class TestImgIo(unittest.TestCase):

    TEST_SHAPES_1 = ((1, 1), (7, 11))
    TEST_SHAPES_3 = ((1, 1, 3), (7, 11, 3), (123, 321, 3))
    TEST_SHAPES_N = ((1, 1, 2), (1, 1, 9), (9, 13, 31))
    TEST_SHAPES_RAW = ((4, 8), (32, 36))

    TEST_SHAPES_ALL = TEST_SHAPES_1 + TEST_SHAPES_3 + TEST_SHAPES_N + TEST_SHAPES_RAW

    def setUp(self):
        self._original_dir = os.getcwd()
        self._tmpdir = tempfile.TemporaryDirectory()
        os.chdir(self._tmpdir.name)

    def tearDown(self):
        os.chdir(self._original_dir)
        self._tmpdir.cleanup()

    def assertRaisesRegex(self, expected_exception, expected_regex, *args, **kwargs):  # noqa: N802 [invalid-function-name]
        """
        Checks that the correct type of exception is raised, and that the exception
        message matches the given regular expression. Also prints out the message
        for visual inspection.
        """
        try:  # check the type of exception first
            unittest.TestCase.assertRaises(self, expected_exception, *args, **kwargs)
        except Exception as e:
            raised_name = sys.exc_info()[0].__name__
            expected_name = expected_exception.__name__
            errstr = "Expected %s with a message matching '%s', got %s."%(expected_name, expected_regex, raised_name)
            print("   FAIL: %s"%(errstr))
            raise AssertionError(errstr) from e
        try:  # then check the exception message
            assertRaisesRegex = getattr(unittest.TestCase, "assertRaisesRegex", unittest.TestCase.assertRaisesRegex)
            assertRaisesRegex(self, expected_exception, expected_regex, *args, **kwargs)
            try:  # print the exception message also when everything is OK
                func = args[0]
                args = args[1:]
                func(*args, **kwargs)
            except expected_exception:
                print("   PASS: %s"%(sys.exc_info()[1]))
        except AssertionError as e:
            print("   FAIL: %s"%(e))
            raise

    def test_exceptions(self):  # noqa: PLR0915 [too-many-statements]
        print("Testing exception handling...")
        shape = (7, 11, 3)
        pixels = np.random.random(shape).astype(np.float32)
        pixels8b = (pixels * 255).astype(np.uint8)
        pixels16b = (pixels * 65535).astype(np.uint16)
        imwrite("validimage.ppm", pixels8b, 255)
        imwrite("imgio.test.ppm", pixels8b, 255)
        imwrite("imgio.test.png", pixels8b, 255)
        imwrite("imgio.test.npy", pixels8b, 255)
        os.rename("imgio.test.ppm", "invalidformat.pfm")
        os.rename("imgio.test.npy", "invalidformat.jpg")
        os.rename("imgio.test.png", "invalidformat.ppm")
        self.assertRaisesRegex(ImageIOError, "^Failed to read", imread, None)
        self.assertRaisesRegex(ImageIOError, "^Failed to read", imread, 0xdeadbeef)
        self.assertRaisesRegex(ImageIOError, "^Failed to read", imread, "")
        self.assertRaisesRegex(ImageIOError, "^Failed to read", imread, ".ppm")
        self.assertRaisesRegex(ImageIOError, "^Failed to read", imread, "invalidfilename")
        self.assertRaisesRegex(ImageIOError, "^Failed to read", imread, "nonexisting/")
        self.assertRaisesRegex(ImageIOError, "^Failed to read", imread, "nonexisting/.ppm")
        self.assertRaisesRegex(ImageIOError, "^Failed to read", imread, "nonexisting/invalidfilename")
        self.assertRaisesRegex(ImageIOError, "^Failed to read", imread, "nonexisting.jpg")
        self.assertRaisesRegex(ImageIOError, "^Failed to read", imread, "nonexisting.png")
        self.assertRaisesRegex(ImageIOError, "^Failed to read", imread, "nonexisting.ppm")
        self.assertRaisesRegex(ImageIOError, "^Failed to read", imread, "nonexisting.pgm")
        self.assertRaisesRegex(ImageIOError, "^Failed to read", imread, "nonexisting.pfm")
        self.assertRaisesRegex(ImageIOError, "^Failed to read", imread, "nonexisting.exr")
        self.assertRaisesRegex(ImageIOError, "^Failed to read", imread, "nonexisting.bmp")
        self.assertRaisesRegex(ImageIOError, "^Failed to read", imread, "invalidformat.pfm")
        self.assertRaisesRegex(ImageIOError, "^Failed to read", imread, "invalidformat.jpg")
        self.assertRaisesRegex(ImageIOError, "^Failed to read", imread, "invalidformat.ppm")
        self.assertRaisesRegex(ImageIOError, "^Failed to read.*verbose", imread, "validimage.ppm", verbose="foo")
        self.assertRaisesRegex(ImageIOError, "^Failed to write", imwrite, "invalidfilename", pixels8b, 255)
        self.assertRaisesRegex(ImageIOError, "^Failed to write", imwrite, "invaliddepth.ppm", pixels16b, 255)
        self.assertRaisesRegex(ImageIOError, "^Failed to write", imwrite, "invaliddepth.png", pixels16b, 255)
        self.assertRaisesRegex(ImageIOError, "^Failed to write", imwrite, "invaliddepth.png", pixels8b, 254)
        self.assertRaisesRegex(ImageIOError, "^Failed to write", imwrite, "invaliddepth.jpg", pixels8b, 254)
        self.assertRaisesRegex(ImageIOError, "^Failed to write", imwrite, "invaliddepth.png", pixels16b, 1023)
        self.assertRaisesRegex(ImageIOError, "^Failed to write", imwrite, "invaliddepth.ppm", pixels8b, 1023)
        self.assertRaisesRegex(ImageIOError, "^Failed to write", imwrite, None, pixels8b, 255)
        self.assertRaisesRegex(ImageIOError, "^Failed to write", imwrite, 0xdeadbeef, pixels8b, 255)
        self.assertRaisesRegex(ImageIOError, "^Failed to write", imwrite, "", pixels8b, 255)
        self.assertRaisesRegex(ImageIOError, "^Failed to write", imwrite, ".ppm", pixels8b, 255)
        self.assertRaisesRegex(ImageIOError, "^Failed to write", imwrite, "nonexisting/.ppm", pixels8b, 255)
        self.assertRaisesRegex(ImageIOError, "^Failed to write", imwrite, "nonexisting/foo.ppm", pixels8b, 255)
        self.assertRaisesRegex(ImageIOError, "^Failed to write", imwrite, "imgio.test.ppm", None)
        self.assertRaisesRegex(ImageIOError, "^Failed to write", imwrite, "imgio.test.ppm", np.zeros((7, 0), np.uint8))
        self.assertRaisesRegex(ImageIOError, "^Failed to write", imwrite, "imgio.test.ppm", np.zeros((7,), np.uint8))
        self.assertRaisesRegex(ImageIOError, "^Failed to write", imwrite, "imgio.test.ppm", np.zeros((7, 7, 1), np.uint8))
        self.assertRaisesRegex(ImageIOError, "^Failed to write", imwrite, "imgio.test.ppm", np.zeros((7, 7, 2), np.uint8))
        self.assertRaisesRegex(ImageIOError, "^Failed to write", imwrite, "imgio.test.ppm", np.zeros((7, 7, 4), np.uint8))
        self.assertRaisesRegex(ImageIOError, "^Failed to write", imwrite, "imgio.test.ppm", np.zeros((7, 7, 3, 1), np.uint8))
        self.assertRaisesRegex(ImageIOError, "^Failed to write", imwrite, "imgio.test.ppm", pixels.astype(bool))
        self.assertRaisesRegex(ImageIOError, "^Failed to write", imwrite, "imgio.test.ppm", pixels.astype(np.float16))
        self.assertRaisesRegex(ImageIOError, "^Failed to write", imwrite, "imgio.test.ppm", pixels.astype(np.float64))
        self.assertRaisesRegex(ImageIOError, "^Failed to write", imwrite, "imgio.test.ppm", pixels.astype(">f4"))
        self.assertRaisesRegex(ImageIOError, "^Failed to write", imwrite, "imgio.test.pfm", pixels, -1.0)
        self.assertRaisesRegex(ImageIOError, "^Failed to write", imwrite, "imgio.test.pfm", pixels, "255")
        self.assertRaisesRegex(ImageIOError, "^Failed to write.*shape", imwrite, "imgio.test.raw", pixels8b, 255)
        self.assertRaisesRegex(ImageIOError, "^Failed to write.*supported", imwrite, "imgio.test.raw", pixels8b, 255, packed=True)
        self.assertRaisesRegex(ImageIOError, "^Failed to write.*verbose", imwrite, "imgio.test.ppm", pixels8b, 255, verbose=0)

    def test_png(self):
        for shape in self.TEST_SHAPES_1 + self.TEST_SHAPES_3:
            for maxval, dtype in [(255, np.uint8), (65535, np.uint16)]:
                if freeimage or dtype == np.uint8:
                    tempfile = Path("imgio.test.png")
                    print("Testing PNG reading & writing, shape=%s, maxval=%d..."%(repr(shape), maxval))
                    pixels = np.random.random(shape)
                    pixels = (pixels * maxval).astype(dtype)
                    imwrite(tempfile, pixels, maxval)
                    result, resmaxval = imread(tempfile)
                    self.assertEqual(resmaxval, maxval)
                    self.assertEqual(result.dtype, dtype)
                    self.assertEqual(result.shape, pixels.shape)
                    result_f64 = imread_f64(tempfile)
                    result_f32 = imread_f32(tempfile)
                    result_f16 = imread_f16(tempfile)
                    self.assertEqual(result_f64.dtype, np.float64)
                    self.assertEqual(result_f32.dtype, np.float32)
                    self.assertEqual(result_f16.dtype, np.float16)
                    os.remove(tempfile)

    def test_pnm(self):
        for shape in self.TEST_SHAPES_1 + self.TEST_SHAPES_3:
            for maxval, dtype in [(255, np.uint8), (65535, np.uint16)]:
                tempfile = Path("imgio.test.ppm") if len(shape) == 3 else Path("imgio.test.pgm")
                print("Testing PNM reading & writing, shape=%s, maxval=%d..."%(repr(shape), maxval))
                pixels = np.random.random(shape)
                pixels = (pixels * maxval).astype(dtype)
                imwrite(tempfile, pixels, maxval)
                result, resmaxval = imread(tempfile)
                self.assertEqual(resmaxval, maxval)
                self.assertEqual(result.dtype, dtype)
                self.assertEqual(result.shape, pixels.shape)
                result_f64 = imread_f64(tempfile)
                result_f32 = imread_f32(tempfile)
                result_f16 = imread_f16(tempfile)
                self.assertEqual(result_f64.dtype, np.float64)
                self.assertEqual(result_f32.dtype, np.float32)
                self.assertEqual(result_f16.dtype, np.float16)
                os.remove(tempfile)

    def test_tiff(self):
        for shape in self.TEST_SHAPES_1 + self.TEST_SHAPES_3:
            for maxval, dtype in [(255, np.uint8), (65535, np.uint16)]:
                tempfile = Path("imgio.test.tiff")
                print("Testing TIFF reading & writing, shape=%s, maxval=%d..."%(repr(shape), maxval))
                pixels = np.random.random(shape)
                pixels = (pixels * maxval).astype(dtype)
                imwrite(tempfile, pixels, maxval)
                result, resmaxval = imread(tempfile)
                self.assertEqual(resmaxval, maxval)
                self.assertEqual(result.dtype, dtype)
                self.assertEqual(result.shape, pixels.shape)
                result_f64 = imread_f64(tempfile)
                result_f32 = imread_f32(tempfile)
                result_f16 = imread_f16(tempfile)
                self.assertEqual(result_f64.dtype, np.float64)
                self.assertEqual(result_f32.dtype, np.float32)
                self.assertEqual(result_f16.dtype, np.float16)
                os.remove(tempfile)

    def test_jpg(self):
        for shape in self.TEST_SHAPES_1 + self.TEST_SHAPES_3:
            for maxval, dtype in [(255, np.uint8)]:
                tempfile = Path("imgio.test.jpg")
                print("Testing JPEG reading & writing, shape=%s, maxval=%d..."%(repr(shape), maxval))
                pixels = np.random.random(shape)
                pixels = (pixels * maxval).astype(dtype)
                imwrite(tempfile, pixels, maxval)
                result, resmaxval = imread(tempfile)
                self.assertEqual(resmaxval, maxval)
                self.assertEqual(result.dtype, np.uint8)
                self.assertEqual(result.shape, pixels.shape)
                result_f64 = imread_f64(tempfile)
                result_f32 = imread_f32(tempfile)
                result_f16 = imread_f16(tempfile)
                self.assertEqual(result_f64.dtype, np.float64)
                self.assertEqual(result_f32.dtype, np.float32)
                self.assertEqual(result_f16.dtype, np.float16)
                os.remove(tempfile)

    def test_pfm(self):
        for shape in self.TEST_SHAPES_1 + self.TEST_SHAPES_3:
            tempfile = Path("imgio.test.pfm")
            print("Testing PFM reading & writing, shape=%s..."%(repr(shape)))
            pixels = np.random.random(shape).astype(np.float32)
            scale = 1.0
            imwrite(tempfile, pixels, scale)
            result, resscale = imread(tempfile)
            self.assertEqual(resscale, scale)
            self.assertEqual(result.dtype, np.float32)
            self.assertEqual(result.shape, pixels.shape)
            result_f64 = imread_f64(tempfile)
            result_f32 = imread_f32(tempfile)
            result_f16 = imread_f16(tempfile)
            self.assertEqual(result_f64.dtype, np.float64)
            self.assertEqual(result_f32.dtype, np.float32)
            self.assertEqual(result_f16.dtype, np.float16)
            os.remove(tempfile)

    def test_npy(self):
        for dt in [np.uint8, np.uint16, np.float16, np.float32, np.float64]:
            for shape in list(self.TEST_SHAPES_ALL) + [(1, 1, 1), (7, 11, 1)]:
                tempfile = Path("imgio.test.npy")
                print("Testing NPY reading & writing, shape=%s, dtype=%s..."%(repr(shape), dt))
                pixels = np.random.random(shape)
                pixels = pixels.astype(dt)
                maxval = np.iinfo(dt).max if np.issubdtype(dt, np.integer) else 1.0
                imwrite(tempfile, pixels, maxval)
                result, _ = imread(tempfile)
                self.assertEqual(result.dtype, dt)
                self.assertEqual(result.shape, pixels.shape)
                os.remove(tempfile)

    def test_exr(self):
        for dt in [np.float16, np.float32]:
            for shape in self.TEST_SHAPES_1 + self.TEST_SHAPES_3:
                tempfile = Path("imgio.test.exr")
                print("Testing EXR reading & writing, shape=%s, dtype=%s..."%(repr(shape), dt))
                pixels = np.random.random(shape).astype(dt)
                imwrite(tempfile, pixels, 1.0)
                result, resscale = imread(tempfile)
                self.assertEqual(result.dtype, dt)
                self.assertEqual(result.shape, pixels.shape)
                self.assertEqual(resscale, 1.0)
                result_f64 = imread_f64(tempfile)
                result_f32 = imread_f32(tempfile)
                result_f16 = imread_f16(tempfile)
                self.assertEqual(result_f64.dtype, np.float64)
                self.assertEqual(result_f32.dtype, np.float32)
                self.assertEqual(result_f16.dtype, np.float16)
                os.remove(tempfile)

    def test_hdr(self):
        for dt in [np.float16, np.float32]:
            for shape in self.TEST_SHAPES_3:
                tempfile = Path("imgio.test.hdr")
                print("Testing HDR reading & writing, shape=%s, dtype=%s..."%(repr(shape), dt))
                pixels = np.random.random(shape)
                inf_mask = pixels >= np.finfo(dt).max
                pixels[inf_mask] = np.finfo(dt).max  # replace infs with maxval
                pixels = pixels.astype(dt)  # convert to float16/32
                pixels = pixels.astype(np.float32)
                imwrite(tempfile, pixels, verbose=True)
                result, _resscale = imread(tempfile, verbose=True)
                self.assertEqual(result.dtype, np.float32)
                self.assertEqual(result.shape, pixels.shape)
                # maximum absolute error of RGBE encoding at float16 range is ~256
                np.testing.assert_allclose(result, pixels, atol=256)
                os.remove(tempfile)

    def test_exif(self):
        print("Testing EXIF orientation handling...")
        for orientation in ["landscape", "portrait"]:
            reffile = "%s_1.jpg"%(orientation)
            filespec = str(IMAGES_DIR / reffile)
            refimg, refmax = imread(filespec)
            for idx in range(2, 9):
                filename = "%s_%d.jpg"%(orientation, idx)
                print("  %s image, orientation value = %d"%(orientation, idx))
                filespec = str(IMAGES_DIR / filename)
                testimg, testmax = imread(filespec, verbose=False)
                epsdiff = np.isclose(refimg, testimg, atol=5.0, rtol=0.2)
                self.assertEqual(refmax, testmax)
                self.assertEqual(refimg.shape, testimg.shape)
                self.assertGreater(np.sum(epsdiff), 0.5 * epsdiff.size)

    def test_exr_read(self):
        print("Testing EXR reading...")
        filespec = str(IMAGES_DIR / "GrayRampsDiagonal.exr")
        img, scale = imread(filespec)
        self.assertEqual(img.shape, (800, 800))
        self.assertEqual(img.dtype, np.float16)
        self.assertEqual(scale, 1.0)

    def test_unpacked_raw(self):
        for packed in [False]:
            for shape in self.TEST_SHAPES_RAW:
                for bpp in [10, 12]:
                    maxval = 2**bpp - 1
                    tempfile = Path("imgio.test%db.raw"%(bpp))
                    dtype = np.uint8 if bpp <= 8 else np.uint16
                    packstr = "packed" if packed else "padded to %d bits"%(np.dtype(dtype).itemsize * 8)
                    print("Testing RAW reading & writing in %d-bit mode (%s), shape=%s..."%(bpp, packstr, repr(shape)))
                    pixels = np.random.random(shape)
                    pixels = (pixels * maxval).astype(dtype)
                    imwrite(tempfile, pixels, maxval, packed=packed)
                    result, resmaxval = rawread(tempfile, shape[1], shape[0], bpp)
                    self.assertEqual(resmaxval, maxval)
                    self.assertEqual(result.dtype, dtype)
                    self.assertEqual(result.shape, shape)
                    self.assertEqual(result.tolist(), pixels.tolist())
                    os.remove(tempfile)

    def test_packed_raw(self):
        bpp = 10
        maxval = 2**bpp - 1
        pixels = np.random.random(4)
        pixels = (pixels * maxval).astype(np.uint16)
        packed = np.zeros(5, dtype=np.uint8)
        packed[0] |= (pixels[0] & 0x00ff)  # byte0 / pixel0: 8 lsb
        packed[1] |= (pixels[0] & 0x0300) >> 8  # byte1 / pixel0: 2 msb
        packed[1] |= (pixels[1] & 0x003f) << 2  # byte1 / pixel1: 6 lsb
        packed[2] |= (pixels[1] & 0x03c0) >> 6  # byte2 / pixel1: 4 msb
        packed[2] |= (pixels[2] & 0x000f) << 4  # byte2 / pixel2: 4 lsb
        packed[3] |= (pixels[2] & 0x03f0) >> 4  # byte3 / pixel2: 6 msb
        packed[3] |= (pixels[3] & 0x0003) << 6  # byte3 / pixel3: 2 lsb
        packed[4] |= (pixels[3] & 0x03fc) >> 2  # byte4 / pixel3: 8 msb
        packed = np.tile(packed, 3)  # 3 * 4 = 12 pixels => 15 bytes
        packed = np.r_[packed, 0]  # pad to 16 bytes
        pixels = np.tile(pixels, 3)  # 3 * 4 = 12 pixels
        tempfile = Path("imgio.test%db.raw"%(bpp))
        packed.tofile(tempfile)
        result, resmaxval = rawread(tempfile, width=12, height=1, bpp=bpp)
        self.assertEqual(resmaxval, maxval)
        self.assertEqual(result.shape, (1, 12))
        np.testing.assert_equal(result.flatten(), pixels)

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
        expected_bytes = (28 * 60 + 17 + 19) * 2
        self.assertEqual(data.nbytes, expected_bytes)
        self.assertEqual(os.path.getsize(tempfile), expected_bytes)
        result, resmaxval = rawread(tempfile, width=shape[1], height=shape[0], bpp=bpp, header_size=17 * 2)
        self.assertEqual(resmaxval, maxval)
        self.assertEqual(result.dtype, np.uint16)
        self.assertEqual(result.shape, shape)
        np.testing.assert_allclose(result, pixels)
        os.remove(tempfile)

    def test_allcaps(self):
        print("Testing Windows-style all-caps filenames...")
        maxval = 255
        dtype = np.uint8
        for ext in [".pnm", ".ppm", ".jpg", ".jpeg"]:
            tempfile = Path("imgio.test%s"%(ext))
            capsfile = Path("imgio.test%s"%(ext.upper()))
            shape = (7, 11, 3)
            pixels = np.ones(shape)
            pixels = (pixels * maxval).astype(dtype)
            imwrite(tempfile, pixels, maxval, verbose=True)
            os.rename(tempfile, capsfile)
            result, resmaxval = imread(capsfile, verbose=True)
            self.assertEqual(resmaxval, maxval)
            self.assertEqual(result.shape, shape)
            np.testing.assert_allclose(result, pixels)
            os.remove(capsfile)

    def test_verbose(self):
        print("Testing verbose mode...")
        for dt in ["uint8", "uint16"] if freeimage else ["uint8"]:
            maxval = np.iinfo(dt).max
            for shape in [(7, 11), (9, 13, 3)]:
                for ext in [".pnm", ".jpg", ".png"]:
                    if dt == "uint8" or ext != ".jpg":
                        tempfile = Path("imgio.test%s"%(ext))
                        pixels = np.random.random(shape)
                        pixels = (pixels * maxval).astype(dt)
                        imwrite(tempfile, pixels, maxval, verbose=True)
                        result, resmaxval = imread(tempfile, verbose=True)
                        self.assertEqual(resmaxval, maxval)
                        self.assertEqual(result.shape, shape)
                        os.remove(tempfile)


# Allow running as: python test/test_imgio.py
if __name__ == "__main__":
    unittest.main()
