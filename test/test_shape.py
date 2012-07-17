import unittest
import numpy
import speckle
ss = speckle.shape

class TestShape(unittest.TestCase):

    def setUp(self):
        self.shape = (256, 256)
        self.center = (self.shape[0]/2, self.shape[1]/2)
        self.radius = 100

    def test_radial(self):
        rad = ss.radial(self.shape)
        self.assertEqual(rad.max(), numpy.sqrt(self.center[0]**2 + self.center[1]**2))
        self.assertEqual(rad.min(), 0)
        self.assertEqual(rad.shape, self.shape)

    def test_angular(self):
        pass

    def test_square(self):
        length = 14
        sq = ss.square(self.shape, length)
        self.assertEqual(sq.sum(), length**2)

        # check to see if it correctly draws a square if parts of the square are off the array
        sq = ss.square(self.shape, length, (4, 4))
        self.assertEqual(sq.sum(), (4+length/2)**2)

    def test_circle(self):
        # test circle area
        circ = ss.circle(self.shape, self.radius, AA=False)
        self.assertAlmostEqual(circ.sum(), numpy.pi*self.radius**2, delta = circ.sum()*0.001)

        # test circles on the edge (by area)
        circ = ss.circle(self.shape, self.radius, center=(0,0))
        self.assertAlmostEqual(circ.sum(), numpy.pi/4*self.radius**2, delta = circ.sum()*0.1)

        # test drawing a circle outside the edge.  Shouldn't throw an assert
        circ = ss.circle(self.shape, self.radius, center=(-10,-10))

        # need to figure out how to test the AA flag

    def test_annulus(self):
        R, r = 120, 50
        annulus = ss.annulus(self.shape, (r, R), AA=False)
        self.assertAlmostEqual(annulus.sum(), numpy.pi*abs(r**2-R**2), delta=annulus.sum()*0.001)

        annulus = ss.annulus(self.shape, (r, R), (0, 0), AA=False)
        self.assertAlmostEqual(annulus.sum(), numpy.pi/4.*abs(r**2-R**2), delta=annulus.sum()*0.01)

    def test_ellipse(self):
        pass

    def test_gaussian(self):
        pass

    def test_lorentzian(self):
        pass

if __name__ == '__main__':
    unittest.main()
