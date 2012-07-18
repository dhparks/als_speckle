import unittest
import numpy
import speckle
sm = speckle.masking

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
        bb = sm.bounding_box(box)
        self.assertEqual(tuple(bb), (ymin, ymax, xmin, xmax))

        # Check how well we do with a box (with padding)
        pad = 14
        bb = sm.bounding_box(box, pad=pad)
        self.assertEqual(tuple(bb), (ymin-pad, ymax+pad, xmin-pad, xmax+pad))

    def test_apply_shrink_mask(self):
        # ----- Circle -----
        asm = sm.apply_shrink_mask(self.circle, self.circle)
        # dimension
        self.assertEqual(asm.shape, (self.radius*2-1, self.radius*2-1))
        # sum
        self.assertEqual(asm.sum(), self.circle.sum())

        # ----- Rectangle -----
        asm = sm.apply_shrink_mask(self.rect, self.rect)
        # dimension
        self.assertEqual(asm.shape, self.rdim)
        # sum
        self.assertEqual(asm.sum(), self.rect.sum())

    def test_take_masked_pixels(self):
        # check to see if we have the same number of pixels in both (circle)
        tmp = sm.take_masked_pixels(self.circle, self.circle)
        self.assertEqual(len(tmp), self.circle.sum())

        # check to see if we have the same number of pixels in both (rect)
        tmp = sm.take_masked_pixels(self.rect, self.rect)
        self.assertEqual(len(tmp), self.rect.sum())

        # check to see if we have the same number of pixels in both (circle and rect)
        tmp = sm.take_masked_pixels(self.rect, self.circle)
        self.assertEqual(len(tmp), (self.circle*self.rect).sum())

if __name__ == '__main__':
    unittest.main()
