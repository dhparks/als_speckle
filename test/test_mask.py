import unittest
import numpy
import speckle

class TestMasking(unittest.TestCase):

    def setUp(self):
#        print "setup"
        self.shape = (256, 256)
        self.center = (self.shape[0]/2, self.shape[1]/2)

        self.radius = 20
        self.circle = speckle.shape.circle(self.shape, self.radius, self.center, AA=False)

        self.rdim = (self.radius*4, self.radius*2)
        self.rect = speckle.shape.rect(self.shape, self.rdim, self.center)

    def test_bb(self):
        # Check how well we do with a box
        box = numpy.zeros(self.shape)
        ymin, ymax = 18, 90
        xmin, xmax = 45, 63
        box[ymin:ymax, xmin:xmax] = 1
        bb = speckle.masking.bounding_box(box)
        self.assertEqual(tuple(bb), (ymin, ymax, xmin, xmax))

        # Check how well we do with a box (with padding)
        pad = 14
        bb = speckle.masking.bounding_box(box, pad=pad)
        self.assertEqual(tuple(bb), (ymin-pad, ymax+pad, xmin-pad, xmax+pad))

        # Get bounds right for a circle
        bb = speckle.masking.bounding_box(self.circle)
        sc = self.center
        sr = self.radius
        self.assertTrue(tuple(bb), (sc[0]-sr, sc[0]+sr, sc[1]-sr, sc[1]+sr))

        # correctly get bounds for a circle with some padding
        pad = 0
        bb = speckle.masking.bounding_box(self.circle, pad = pad)
        sp = sr + pad
        self.assertEqual(tuple(bb), (sc[0]-sp, sc[0]+sp, sc[1]-sp, sc[1]+sp))

        # check rectangular region with 
        bb = speckle.masking.bounding_box(self.rect)
        srd = self.rdim
        self.assertTrue(numpy.array_equal(bb, numpy.array([sc[0]-srd[0],sc[0]+srd[0],sc[1]-srd[1],sc[1]+srd[1]])))

    def test_apply_shrink_mask(self):
        # ----- Circle -----
        asm = speckle.masking.apply_shrink_mask(self.circle, self.circle)
        # dimension
        self.assertEqual(asm.shape, (self.radius*2-1, self.radius*2-1))
        # sum
        self.assertEqual(asm.sum(), self.circle.sum())

        # ----- Rectangle -----
        asm = speckle.masking.apply_shrink_mask(self.rect, self.rect)
        # dimension
        self.assertEqual(asm.shape, self.rdim)
        # sum
        self.assertEqual(asm.sum(), self.rect.sum())

    def test_take_masked_pixels(self):
        # check to see if we have the same number of pixels in both (circle)
        tmp = speckle.masking.take_masked_pixels(self.circle, self.circle)
        self.assertEqual(len(tmp), self.circle.sum())

        # check to see if we have the same number of pixels in both (rect)
        tmp = speckle.masking.take_masked_pixels(self.rect, self.rect)
        self.assertEqual(len(tmp), self.rect.sum())

        # check to see if we have the same number of pixels in both (circle and rect)
        tmp = speckle.masking.take_masked_pixels(self.rect, self.circle)
        self.assertEqual(len(tmp), (self.circle*self.rect).sum())

if __name__ == '__main__':
    unittest.main()
