import unittest
import numpy
import speckle

class TestWrapping(unittest.TestCase):

    def setUp(self):
        self.shape = (256, 256)
        self.r, self.R = 0, self.shape[0]/2
        self.center = (self.shape[0]/2, self.shape[1]/2)

    def test_unwrap_wrap(self):
        orig = speckle.shape.annulus(self.shape, (self.r, self.R), self.center, AA=False)
        unw = speckle.wrapping.unwrap(orig, (self.r, self.R, self.center))
        wrap = speckle.wrapping.wrap(unw, (self.r, self.R))

        speckle.io.writefits('orig.fits', orig, overwrite=True)
        speckle.io.writefits('wrap.fits', wrap, overwrite=True)
        self.assertEqual(orig.shape, wrap.shape)

        self.assertTrue(numpy.array_equal(orig, wrap))

if __name__ == '__main__':
    unittest.main()
