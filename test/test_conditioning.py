import unittest
import numpy
import speckle
sc = speckle.conditioning

class TestConditioning(unittest.TestCase):

    def setUp(self):
        self.shape = (256, 256)
        self.center = (self.shape[0]/2, self.shape[1]/2)
        self.data = numpy.random.random((self.shape[0]*self.shape[1])).reshape(self.shape)

    #def test_remove_dust(self):
    #    pass

    #def test_subtract_background(self):
    #    pass

    def test_remove_hot_pixels(self):
        # Try to set some number of bad pixels and replace them
        offset = 5
        data = self.data.copy() + offset
        nbad = 10
        peakval = 10000
#        badcoords = []
        for i in range(nbad):
            y,x = numpy.random.randint(self.shape[0], size=2)
            data[y,x] = peakval
#            badcoords.append((y,x))

        removed = sc.remove_hot_pixels(data, threshold = 5)
        self.assertAlmostEqual((data - removed).sum(), nbad*peakval, delta=offset*nbad)

    def test_align(self):
        rollx, rolly = numpy.random.randint(0, self.shape[0]/2, 2)
        rolled = numpy.roll(numpy.roll(self.data, rollx, axis=1), rolly, axis=0)

        # Check we get the same coordinates back
        coords = sc.align_frames(rolled, self.data, return_type='coordinates')
#        print "coords", coords, (rolly, rollx), coords[0]
        self.assertEqual(tuple(coords[0]), (-rolly, -rollx))

        # Roll an array and check that align_frames rolls us back
        rolled = numpy.roll(numpy.roll(self.data, rollx, axis=1), rolly, axis=0)
        res = sc.align_frames(rolled, self.data)
        self.assertTrue(numpy.array_equal(res, self.data))

        # check to make sure that the input array is unchanged. align_frames modifies the array in-place.
        rolled = numpy.roll(numpy.roll(self.data, rollx, axis=1), rolly, axis=0)
        rs = rolled.copy()
        res = sc.align_frames(rolled, self.data)
        self.assertTrue(numpy.array_equal(rs, rolled))

    def test_match_counts(self):
        s, d1, d2 = numpy.random.random(3)
        s *= 10
        d1 *= 1000
        d2 *= 1000
        region = speckle.shape.square(self.shape, 50)

        # check 1 parameter fit
        matched = sc.match_counts(self.data, self.data/s, region, nparam=1)
        self.assertTrue(numpy.allclose(matched, self.data))

        # check 2 parameter fit
        matched = sc.match_counts(self.data, (self.data+d1)/s, region, nparam=2)
        self.assertTrue(numpy.allclose(matched, self.data, atol=1e-4))

        # check 3 parameter fit
        matched = sc.match_counts(self.data, (self.data+d1)/s + d2, region, nparam=3)
        self.assertTrue(numpy.allclose(matched, self.data))

    def test_find_center(self):
        # Put two circles at xc+w, xc-w, and yc+w, yc-w to simulate centrosymmety.  Use this to align and test the result.
        (yc, xc) = numpy.random.randint(self.shape[0]/4, 3*self.shape[0]/4, 2)
        #print (yc, xc)
        (yw, xw) = numpy.random.randint(self.shape[0]/4, size=2)
        c1 = speckle.shape.circle(self.shape, 14, (yc-yw, xc-xw))
        c2 = speckle.shape.circle(self.shape, 14, (yc+yw, xc+xw))
        outc = sc.find_center(c1+c2+self.data)
        print (yc, xc), outc
        # the find circle routine can be off by a pixel every now and again. Let it go.
        self.assertAlmostEqual(yc, outc[0], delta=1)
        self.assertAlmostEqual(xc, outc[1], delta=1)

    def test_merge(self):
        # create a copy of the data and scale it down, then try to merge it.
        no_bb = self.data/100.
        with_bb = self.data
        r, R = 30, 45
        fill_reg = speckle.shape.circle(self.shape, r)
        fit_reg = speckle.shape.annulus(self.shape, (r, R))
        merged = sc.merge(with_bb, no_bb, fill_reg, fit_reg, width=0)
        self.assertTrue(numpy.allclose(merged, self.data))

if __name__ == '__main__':
    unittest.main()
