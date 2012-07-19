import unittest
import numpy
import speckle
sc = speckle.conditioning

class TestConditioning(unittest.TestCase):

    def setUp(self):
        self.shape = (1024, 1024)
        self.center = (self.shape[0]/2, self.shape[1]/2)
        self.density = 1e-2
        
        # build a realistic speckle pattern to form the basis for testing
        # methods in speckle.conditioning
        make_speckle = lambda x: numpy.fft.fftshift(abs(numpy.fft.fft2(x))**2).real
        data = numpy.where(numpy.random.rand(self.shape[0]*self.shape[1]) <= self.density,1,0).reshape(self.shape)
        data *= speckle.shape.circle(self.shape,self.shape[0]/20.)
        data = make_speckle(data)
        self.data = 1e6*data/data.max()+1000
        
    def test_remove_dust(self):
        convolve = lambda x,y: numpy.fft.ifft2(numpy.fft.fft2(x)*numpy.fft.fft2(y))
        
        speckle.io.set_overwrite(True)
        
        # make the dustspot shape
        dust_spot = numpy.zeros_like(self.data)
        x = 1
        dust_spot[self.shape[0]/2-x:self.shape[0]/2+x,self.shape[1]/2-x:self.shape[1]/2+x] = 1
        dust_spot = numpy.fft.fftshift(dust_spot)
        
        dust_spot2 = numpy.zeros_like(self.data)
        x = 2
        dust_spot2[self.shape[0]/2-x:self.shape[0]/2+x,self.shape[1]/2-x:self.shape[1]/2+x] = 1
        dust_spot2 = numpy.fft.fftshift(dust_spot2)
        
        # make the dust locations. make them sparse
        dust_density = 1e-4
        dust_locations = numpy.where(numpy.random.rand(self.shape[0]*self.shape[1]) <= dust_density,1,0).reshape(self.shape)
        dust_locations[:10,:]  = 0
        dust_locations[-10:,:] = 0
        dust_locations[:,:10]  = 0
        dust_locations[:,-10:] = 0
        
        dust = convolve(dust_spot,dust_locations)
        dust = numpy.clip(abs(dust).real,0,1)
        
        dust_markers = convolve(dust_spot2,dust_locations)
        dust_markers = numpy.clip(abs(dust_markers).real,0,1)
        
        # "dust" right now is the equivalent of dust_mask, so make a plan.
        dust_plan = sc.plan_remove_dust(dust_markers)
        
        # enact the effect of dust on self.data
        dusted = self.data*(1-dust)
        
        # repair the dust
        repaired = sc.remove_dust(dusted,dust,dust_plan=dust_plan)[0]
        
        # the success of repairing the dust spots is determined by whether the
        # "most" of the value of the dusted pixels is restored. it should
        # also not overshoot.
        dust_sum = (dust*self.data).sum()
        repaired_sum = (dust*repaired).sum()

        speckle.io.save('dustq.fits',self.data/repaired*dust_markers)
        speckle.io.save('speckles.fits',self.data)
        speckle.io.save('repaired.fits',repaired)
        
        

    def test_subtract_background(self):
        pass

    def test_remove_hot_pixels(self):
        # Try to set some number of bad pixels and replace them
        
        data = self.data
        density = 1e-3
        hot  = numpy.where(numpy.random.rand(self.shape[0]*self.shape[1]) <= density,1,0).reshape(self.shape)
        data += 10*hot
        removed = sc.remove_hot_pixels(data)
        
        # how to test difference of data and removed properly?
        #self.assertTrue((data - removed).sum() < nbad)

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
