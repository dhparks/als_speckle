import unittest
import numpy
import speckle
sa = speckle.averaging

class TestAveraging(unittest.TestCase):
    
    def setUp(self):
        self.shape = (256, 256)
        self.center = (self.shape[0]/2, self.shape[1]/2)
        
        self.radius = 20
        self.circle = speckle.shape.circle(self.shape, self.radius, self.center, AA=False)
        
        self.rdim = (self.radius*4, self.radius*2)
        self.rect = speckle.shape.rect(self.shape, self.rdim, self.center)
    
    def test_smth_with_rect(self):
        boxsize = 10
        # check to make sure that when we smooth a constant array we get the same result
        scale = numpy.random.randint(0, 10000, 1)[0]
        res = sa.smooth_with_rectangle(numpy.ones(self.shape)*scale, boxsize)
        self.assertEqual(res.max(), scale)
        self.assertEqual(res.min(), scale)
    
    def test_smth_with_circle(self):
        # check to make sure that when we smooth a constant array we get the same result
        scale = numpy.random.randint(0, 10000, 1)[0]
        res = sa.smooth_with_circle(numpy.ones(self.shape)*scale, self.radius)
        self.assertEqual(res.max(), scale)
        self.assertEqual(res.min(), scale)
    
    def test_smth_with_gauss(self):
        # check to make sure that when we smooth a constant array we get the same result
        scale = numpy.random.randint(0, 10000, 1)[0]
        res = sa.smooth_with_gaussian(numpy.ones(self.shape)*scale, self.radius)
        self.assertAlmostEqual(res.max(), scale)
        self.assertAlmostEqual(res.min(), scale)
    
    def test_calc_average(self):
        arr = numpy.random.random((self.shape[0]*self.shape[1])).reshape(self.shape)
        vals = sa.calculate_average(arr)
        # average
        self.assertAlmostEqual(vals[0], 0.5, places=2)
        # standard deviation
        self.assertAlmostEqual(vals[1], numpy.sqrt((1-0)**2/12.), places=2)
        # number of pixels
        self.assertEqual(vals[2], self.shape[0]*self.shape[1])

if __name__ == '__main__':
    unittest.main()
