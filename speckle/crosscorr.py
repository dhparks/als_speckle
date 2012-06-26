""" Functions that calculate the image cross-correlation between two images. Implements the sptaial memory algorithms.

Author: Keoki Seu (KASeu@lbl.gov)
"""
import numpy as np

from . import averaging
from . import shape

def point_memory( imgA, imgB, darks=None, qacfs=None, qacfdarks=None, mask=None, flatten_width=30, removeCCBkg=True, pickPeakMethod="integrate", intermediates=False):
    """Calculates the cross-correlation coefficient between two image pairs imgA
        and imgB. This calculates the value rho where:
        rho = \frac{\sum(CC(imgA, imgB))}{\sqrt{AC(A) AC(B)}}.

    Darks, quasi-ACFs, and qACF darks can also be provided, as well as a mask to
    calculate the memory for one region.  If quasi-ACFs are provided, then it
    calculates:
        rho = \frac{\sum(CC(imgA,imgB))}{\sqrt{qACF(A) qACF(B)}}.

    If the option is on, the images are preprocessed first by dividing by a
    gaussian of fwhm flatten_width before the ACs and CCs are calculated.  Then
    (if the option is on), a spline is used to remove the AC/CC background
    before the peak is summed (or the max value is used).

    arguments:
        imgA - First image to calculate.  Must be a two dimensional array.
        imgB - First image to calculate.  Must be a two dimensional array.
        darks - A tuple of dark images (darkA, darkB). These are subtracted
            from the image.
        qacfs - A tuple of quasi-ACF images (qacfA, qacfB). If provided, these
            are used in the normalization.
        mask - Masked region to calculate over.  If not provided, the memory is
            calculated over the entire array region.
        removeCCBkg - This option will create a spline fit to remove the
            non-speckled background in the AC and CC.  Defaults to True.
        flatten_width - Width used for flattening the image before calculating
            AC/CCs. This value should be larger than the pixel size and defaults
            to 30 px. If False, flattening does not happen. 
        peakPickMethod - method to pick the peak.  Can be either "max" or
            "integrate".  "integrate" sums over a circle of radius 7. Defaults
            to "integrate".
        intermediates - Weather or not the program should return intermediate
            arrays. If this is true, it returns a dictionary of intermediate
            calculations, such as ACFs, qACFs, spline fits..etc.  The
            calculations are numbered in the order that they are calculated.

    returns:
        CC-coefficient - the correlation coeffiecient.  Typically this is a
        number between [0,1]. If intermediates is set, a dictonary of
        intermediate arrays is also returned.
    """
    FLATTEN_WIDTH = 30 # radius used for flattening the speckle pattern before ACs are calculated.
    PEAK_INTEGRATE_RADIUS = 8 # radius of the circle that we integrate the speckle peak over.
    # function to make sure we have a list of two numpy arrays of the same size.
    check_pairs = lambda val: True if (val is not None and type(val) in (list, tuple) and len(val) == 2 and isinstance(val[0], np.ndarray) and isinstance(val[1], np.ndarray) and val[0].shape == val[1].shape) else False

    assert check_pairs((imgA, imgB)), "imgA and imgB must be arrays with the same shape"
    assert imgA.ndim == 2, "arrays must be two-dimensional"
    assert type(flatten_width) in (bool, float, int), "flatten_width must be a number or bool"
    assert type(removeCCBkg) == bool, "removeCCBkg must be a boolean"
    assert type(intermediates) == bool, "intermediates must be a boolean"
    assert pickPeakMethod in ("integrate", "max"), "pickPeakMethod must be 'integrate' or 'max'"

    if intermediates:
        intermediate_arrays = {}

    # helper function for storing intermediate arrays 
    def store(key, val):
        if intermediates: intermediate_arrays[key] = val

    if isinstance(flatten_width, bool):
        if flatten_width == True:
            flatten_width = FLATTEN_WIDTH
    else:
        flatten_width == float(flatten_width)

    if mask is not None and isinstance(mask, np.ndarray):
        if mask.shape == imgA.shape:
            havemask = True
        else:
            havemask = False

    haveqACFs = check_pairs(qacfs)
    if haveqACFs:
        if check_pairs(qacfdarks):
            if qacfs[0].shape == qacfdarks[0].shape:
                qacfA = qacfs[0] - qacfdarks[0]
                qacfB = qacfs[1] - qacfdarks[1]
            else:
                haveqACFs = False
        else:
            qacfA = qacfs[0]
            qacfB = qacfs[1]

    if haveqACFs:
        store('00-A_qACF_orig', qacfA)
        store('00-B_qACF_orig', qacfB)
    store('00-A_orig', imgA)
    store('00-B_orig', imgB)
    store('00-mask', mask)

    if check_pairs(darks):
        if imgA.shape == darks[0].shape:
            imgA -= darks[0]
            imgB -= darks[1]

    if flatten_width:
        filterString = "Flattening"
    else:
        filterString = "Not flattening"

    if removeCCBkg:
        bkgString = ""
    else:
        bkgString = "not "

    if pickPeakMethod == "integrate":
        peakString = "integrating over a circle of radius %d" % PEAK_INTEGRATE_RADIUS
    elif pickPeakMethod == "max":
        peakString = "picking maximum value"

    print("%s before CC/AC, %sremoving backgrounds, %s for AC/CC peak." % (filterString, bkgString, peakString))

    if flatten_width:
        flatA = averaging.smooth_with_gaussian(imgA, flatten_width)
        flatB = averaging.smooth_with_gaussian(imgB, flatten_width)
        imgA = imgA/flatA
        imgB = imgB/flatB
        store('01-A_flatten', imgA)
        store('01-B_flatten', imgB)
        store('01-A_flatten_smoothed', flatA)
        store('01-B_flatten_smoothed', flatB)
        if haveqACFs:
            flatqacfA = averaging.smooth_with_gaussian(qacfA, flatten_width)
            flatqacfB = averaging.smooth_with_gaussian(qacfB, flatten_width)
            qacfA = qacfA/flatqacfA
            qacfB = qacfB/flatqacfB
            store('01-A_qACF_flatten', qacfA)
            store('01-B_qACF_flatten', qacfB)
            store('01-A_qACF_flatten_smoothed', flatqacfA)
            store('01-B_qACF_flatten_smoothed', flatqacfB)


    if havemask:
        imgA = apply_shrink_mask(imgA, mask)
        imgB = apply_shrink_mask(imgB, mask)
        store('03-A_flatten_masked', imgA)
        store('03-B_flatten_masked', imgB)       
        if haveqACFs:
            qacfA = apply_shrink_mask(qacfA, mask)
            qacfB = apply_shrink_mask(qacfB, mask)
            store('03-A_qACF_flatten_masked', qacfA)
            store('03-B_qACF_flatten_masked', qacfB)       
        # this was in the old code. not sure why?
        #sumpx = region.sum()
        #imgA = imgA/(imgA.sum()/sumpx)
        #imgB = imgB/(imgB.sum()/sumpx)
        #qacfA = qacfA/(qacfA.sum()/sumpx)
        #qacfB = qacfB/(qacfB.sum()/sumpx)

    ys, xs = imgA.shape
    cc = crosscorr(np.real(imgA), np.real(imgB))
    store("04-cc", cc)
    if haveqACFs:
        autoA = crosscorr(imgA, qacfA)
        autoB = crosscorr(imgB, qacfB)
        store("04-A_qACF", autoA)
        store("04-B_qACF", autoB)
    else:
        autoA = autocorr(imgA)
        autoB = autocorr(imgB)
        store("04-A_ac", autoA)
        store("04-B_ac", autoB)

    autoA = np.abs(autoA)
    autoB = np.abs(autoB)

    if removeCCBkg:
        # make sure number of control points is even so one does not put a control point on the speckle peak.
        # xshape/ctlpoints should be > 2*(xsize you integrate peak over)
        make_even = lambda v: v % 2 == 1 and v - 1 or v
        xctlpts = make_even(int(np.floor(xs/(PEAK_INTEGRATE_RADIUS*2)-1)))
        yctlpts = make_even(int(np.floor(ys/(PEAK_INTEGRATE_RADIUS*2)-1)))

        spl_cc = averaging.smooth_with_spline(cc, xctlpts, yctlpts)
        spl_autoA = averaging.smooth_with_spline(autoA, xctlpts, yctlpts)
        spl_autoB = averaging.smooth_with_spline(autoB, xctlpts, yctlpts)
        cc = cc - spl_cc
        autoA = autoA - spl_autoA
        autoB = autoB - spl_autoB
        store("05-cc_remCCbkg", cc)
        store("05-A_ac_remCCbkg", autoA)
        store("05-B_ac_remCCbkg", autoB)
        store("05-cc_spline", spl_cc)
        store("05-A_ac_spline", spl_autoA)
        store("05-B_ac_spline", spl_autoB)

    if pickPeakMethod == "integrate":
        circ = shape.circle(cc.shape, PEAK_INTEGRATE_RADIUS)
        CCval = (cc*circ).sum()
        ACval = np.sqrt((autoA*circ).sum() * (autoB*circ).sum())
        store("06-cc_summed_region", cc*circ)
        store("06-A_ac_summed_region", autoA*circ)
        store("06-B_ac_summed_region", autoB*circ)
    else: # pickPeakMethod == "max"
        CCval = cc.max()
        ACval = np.sqrt(autoA.max()*autoB.max())
    
    ccval = np.real(CCval/ACval)
    if intermediates:
        return ccval, intermediate_arrays
    else:
        return ccval

def apply_shrink_mask(img, mask):
    """ Applys a mask and shrinks the image to just the size of the mask.

    arguments:
        img - img to mask.  Must be array of 2 or 3 dimensions.
        mask - mask to use.  Must be two dimensional.

    returns:
        img - shrunk image with a mask applied
    """
    assert isinstance(img, np.ndarray), "img must be an array"
    assert isinstance(mask, np.ndarray), "mask must be an array"
    assert img.shape == mask.shape, "img and mask must be the same shape"
    assert img.ndim in (2, 3), "image must be two or three dimensional"
    
    def apply_shrink(img, mask):
        result = img*mask
        (ys, xs) = img.shape
        aw = np.argwhere(result)
        (ystart, xstart) = aw.min(0)
        (ystop, xstop) = aw.max(0) + 1

        if xstart == xstop:
            # we have a single row, try to increase x in both directions
            if xstart != 0:
                xstart -= 1
            if xstop != xs:
                xstop += 1

        if ystart == ystop:
            # we have a single column, try to increase y in both directions
            if ystart != 0:
                ystart -= 1
            if ystop != xs:
                ystop += 1

        return img[ystart:ystop, xstart:xstop]

    if img.ndim == 3:
        res = np.zeros_like(img)
        for i in range(img.shape[0]):
            res[i] = apply_shrink(img[i], mask)
        return res
    else:
        return apply_shrink(img, mask)

def crosscorr(imgA, imgB):
    """ Calculates the cross correlation of the function. Returns the
        complex-valued cross-correlation of imgA and imgB.

    arguments:
        imgA - two-dimensional image
        imgB - two-dimensional image

    returns:
        cc(imgA, imgB) - cross correlation of imgA with imgB
    """
    assert imgA.shape == imgB.shape, "images not the same size"
    assert imgA.ndim == 2, "images must be two-dimensional"

    (ysize, xsize) = imgA.shape
    
    # This makes the FFT work for datatypes of '>f4'.  We occasionally see data that is in this format.
    if imgA.dtype.byteorder == '>':
        imgA = imgA.astype(imgA.dtype.name)
    if imgB.dtype.byteorder == '>':
        imgB = imgB.astype(imgB.dtype.name)     

    # detect if we're doing an AC and save a FFT.
    if np.array_equal(imgA, imgB):
        fftA = np.fft.fft2(imgA)
        cc = np.fft.ifft2(fftA*np.conjugate(fftA))
    else:
        cc = np.fft.ifft2(np.fft.fft2(imgA)*np.conjugate(np.fft.fft2(imgB)))

    return np.fft.fftshift(cc)


def autocorr(img):
    """ Calculates the autocorrelation of an image.

    arguments:
        img - two-dimensional image

    returns:
        ac(img) - autocorrelation of input image
    """
    return crosscorr(img, img)

def fftconvolve(imgA, imgB):
    """ Calculates the convoluton of two input images.

    arguments:
        imgA - 1st image to convolve.
        imgB - 2nd image to convolve.  imgA and imgB must be the same shape.

    returns:
        Conv(imgA, imgB).  Shape is same as inputs.
    """

    assert isinstance(imgA, np.ndarray) and isinstance(imgB, np.ndarray), "Images must be arrays"
    assert imgA.shape == imgB.shape, "Images must be the same shape."
    assert imgA.ndim == 2, "Images must be two-dimensional."

    (ysize, xsize) = imgA.shape
    result = np.fft.ifft2(np.fft.fft2(imgA) * np.fft.fft2(imgB))
    result = np.roll(result, int(ysize/2), axis=0)
    result = np.roll(result, int(xsize/2), axis=1)
    return result
