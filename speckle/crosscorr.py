""" Functions that calculate the image cross-correlation between two images. Implements the spatial memory algorithms.

Author: Keoki Seu (KASeu@lbl.gov)
"""
import numpy as np

from . import averaging, shape, wrapping, masking

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

    havemask = False
    if mask is not None and isinstance(mask, np.ndarray):
        if mask.shape == imgA.shape:
            havemask = True

    haveDarks = False
    if check_pairs(darks):
        if imgA.shape == darks[0].shape:
            imgA -= darks[0]
            imgB -= darks[1]
            haveDarks = True

    haveqACFs = check_pairs(qacfs)
    haveqACFdarks = check_pairs(qacfdarks)

    if haveqACFs:
        if haveqACFdarks:
            if qacfs[0].shape == qacfdarks[0].shape:
                qacfA = qacfs[0] - qacfdarks[0]
                qacfB = qacfs[1] - qacfdarks[1]
            else:
                haveqACFdarks = False
                qacfA = qacfs[0]
                qacfB = qacfs[1]
        else:
            qacfA = qacfs[0]
            qacfB = qacfs[1]

    if haveqACFs:
        store('00-A_qACF_orig', qacfA)
        store('00-B_qACF_orig', qacfB)
    store('00-A_orig', imgA)
    store('00-B_orig', imgB)
    store('00-mask', mask)


    file_string = "reading (img, "
    if haveDarks:
        file_string += "dark, "
    if haveqACFs:
        file_string += "qACF, "
    if haveqACFdarks:
        file_string += "qACFdarks, "
    if havemask:
        file_string += "mask, "
    file_string = file_string[0:-2] + ") -> "

    if flatten_width:
        flat_string = "flattening (fwhm %d px) -> " % flatten_width
    else:
        flat_string = ""

    if removeCCBkg:
        removeCC_string = "removing backgrounds -> "
    else:
        removeCC_string = ""

    if pickPeakMethod == "integrate":
        peakString = "integrating speckle peak over circle of R=%d px-> " % PEAK_INTEGRATE_RADIUS
    else:
        peakString = "picking maximum value of peak ->"


    print(file_string + flat_string + "CC/AC -> " + removeCC_string + peakString + "returning result")

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
        imgA = masking.apply_shrink_mask(imgA, mask)
        imgB = masking.apply_shrink_mask(imgB, mask)
        store('03-A_flatten_masked', imgA)
        store('03-B_flatten_masked', imgB)       
        if haveqACFs:
            qacfA = masking.apply_shrink_mask(qacfA, mask)
            qacfB = masking.apply_shrink_mask(qacfB, mask)
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

def crosscorr(imgA, imgB, axes=(0,1), already_fft=(), shift=True):
    """ Calculates the cross correlation of the function. Returns the
        complex-valued cross-correlation of imgA and imgB. Note: it is always
        the Fourier transfrom of imgB which is conjugated.

    arguments:
        imgA - two-dimensional image
        imgB - two-dimensional image
        axes - tuple of dimensions along which to calculate the autocorrelation.
            Defaults to both (0,1).
        already_fft - (optional) tuple listing which (if any) inputs have
            already been ffted.
        shift - Flag to determine if the image should be rolled to the center.
            Defaults to True.

    returns:
        cc(imgA, imgB) - cross correlation of imgA with imgB
    """
    
    assert isinstance(imgA,np.ndarray) and isinstance(imgB,np.ndarray), "input must be arrays"
    assert imgA.shape == imgB.shape, "images not the same size"
    assert imgA.ndim == 2, "images must be two-dimensional"
    assert isinstance(already_fft,(list,tuple,int)), "already_fft must be list or tuple"
    if isinstance(already_fft, int):
        already_fft = [already_fft]
    assert all([e in (0, 1, -1, -2) for e in already_fft]), "unrecognized already_fft values"
    assert isinstance(shift, (bool, int)), "shift must be bool or int"

    (ysize, xsize) = imgA.shape

    # This makes the FFT work for datatypes of '>f4'.  We occasionally see data that is in this format.
    if imgA.dtype.byteorder == '>':
        imgA = imgA.astype(imgA.dtype.name)
    if imgB.dtype.byteorder == '>':
        imgB = imgB.astype(imgB.dtype.name)
        
    if np.array_equal(imgA,imgB): # AC condition
        if already_fft != ():
            fftA = fftB = imgA
        else:
            fftA = fftB = np.fft.fft2(imgA, axes=axes)
    else:
        # compute forward ffts accounting for pre-computed ffts
        if 0 in already_fft:
            fftA = imgA
        else:
            fftA = np.fft.fft2(imgA,axes=axes)

        if 1 in already_fft:
            fftB = imgB
        else:
            fftB = np.fft.fft2(imgB,axes=axes)

    cc = np.fft.ifft2(fftA*np.conjugate(fftB),axes=axes)

    if shift:
        return np.fft.fftshift(cc, axes=axes)
    else:
        return cc
    
def autocorr(img,axes=(0,1)):
    """ Calculates the autocorrelation of an image.

    arguments:
        img - two-dimensional image
        axes - tuple of dimensions along which to calculate the autocorrelation

    returns:
        ac(img) - autocorrelation of input image
    """
    return crosscorr(img, img, axes=axes)
    
def pairwise_covariances(data,save_memory=False):
    """ Given 3d data, cross-correlates every frame with every other frame to generate
    all the normalized pairwise covariances (ie, "similarity" or "rho" in Pierce etc).
    This is useful for determining which frames to sum in the presence of drift or slow
    dynamics. The pairwise calculation between A and B is computed as:
    
    cross_correlation(A,B).max/sqrt(autocorrelation(A).max*autocorrelation(B).max)
    
    Requires:
        data - 3d ndarray
        
    Optional:
        save_memory: if True, DFTs of frames of data are precomputed which massively improves
        speed at the expense of memory. Default is False.
        
    Returns:
        a 2d ndarray of normalized covariances values where returned[row,col] is the
        cross-correlation max of frames data[row] and data[col].
    """
        
    assert isinstance(data, np.ndarray) and data.ndim == 3, "data must be 3d ndarray"

    frames = data.shape[0]
    dfts = np.zeros_like(data).astype('complex')
    ACs = np.zeros(frames,float)
    
    covar = lambda c,a,b: c/np.sqrt(a*b)

    # precompute the dfts and autocorrelation-maxes for speed
    for n in range(frames):
        dft = np.fft.fft2(data[n].astype('float'))
        if not save_memory: dfts[n] = dft
        ACs[n] = abs(crosscorr(dft,dft,already_fft=(0,1))).max()
          
    # calculate the pair-wise normalized covariances  
    covars = np.zeros((frames,frames),float)
        
    for j in range(frames):
        ac = ACs[j]
        for k in range(frames-j):
            k += j
            bc = ACs[k]
            if save_memory: corr = crosscorr(data[j],data[k])
            else:
                corr = crosscorr(dfts[j],dfts[k],already_fft=(0,1))
                fill = covar(abs(corr).max(),ac,bc)
            covars[j,k] = fill
            covars[k,j] = fill
            
    return covars

def alignment_coordinates(obj, ref, already_fft=()):
    """ Computes the roll coordinates to align imgA and imgB. The returned
    values r0 and r1 are such the following numpy command will align obj to ref.

    aligned_to_ref = np.roll(np.roll(obj,r0,axis=0),r1,axis=1)
    
    arguments:
        obj - image (numpy array) which will be aligned to ref
        ref - reference image (numpy array) to which obj will be aligned
        already_fft - (optional) tuple listing which (if any) inputs have
            already been ffted

    returns:
        coords - (r0,r1) which describe how to align obj to ref using np.roll
    """
    
    assert isinstance(ref,np.ndarray) and ref.ndim==2, "ref must be 2d array"
    assert isinstance(obj,np.ndarray) and obj.ndim==2, "obj must be 2d array"
    assert ref.shape == obj.shape, "ref and obj must have same same"
    
    rows,cols = ref.shape
    
    # compute the cross correlation and find the location of the max
    corr = crosscorr(obj,ref,already_fft=already_fft)
    cc_max = abs(corr).argmax()
    max_row,max_col = cc_max/cols,cc_max%cols
    
    # do modulo arithmetic to account for the fftshift in crosscorr and the
    # cyclic boundary conditions of the fft
    max_row += -rows/2
    max_col += -cols/2
    if max_row > rows/2: max_row += -rows
    if max_col > cols/2: max_col += -cols
    
    return -max_row, -max_col

def rot_crosscorr(imgA, imgB, center, RRange):
    """ Unwinds an image an calculates the rotational cross-correlation function,
               rotCCF = <I(R,theta), J(R,theta+dtheta)>_theta.
        where I and J are images of the same dimension and <>_theta means summed
        over theta.

        arguments:
            imgA - first image to correlate.
            imgB - second image to correlate.  Must be the same size as imgA.
            center - center coordinates in the format (yc, xc).
            RRange - Radial range to calculate the crosscorr. The  format is
            (Rmin, Rmax, Rstep). Typical values would be (100, 800, 15).
        returns:
            CCF results - a (m, n+1) array of CCF results, where n=number of
            regions determined by RRange, and m is the number of angles
            determined by angleRange.  There is an extra element along x for
            the angles column.
    """
    assert isinstance(imgA, np.ndarray) and isinstance(imgB, np.ndarray), "imgA and imgB must be arrays"
    assert imgA.shape == imgB.shape, "Images must be the same shape"

    # function to check of is tuple or list and length 2 and all elem are integer
    is_tl_2_int = lambda i, l: True if isinstance(i, (tuple, list, set)) and len(i) == l and all([isinstance(b, int) for b in i]) else False
    assert is_tl_2_int(center, 2), "center must be length 2 list/tuple of ints"
    assert is_tl_2_int(RRange, 3), "RRange must be length 3 list/tuple of ints"

    (yc, xc) = center
    Rstart, Rstop, Rstep = RRange
    
    maxR = _maxRadius(imgA, center)
    if maxR < Rstop:
        Rstop = maxR

    imgA = imgA[yc-Rstop:yc+Rstop, xc-Rstop:xc+Rstop]
    imgB = imgB[yc-Rstop:yc+Rstop, xc-Rstop:xc+Rstop]
    # xc and yc get moved once we crop the image.
    xc = yc =Rstop
    (ys, xs) = imgA.shape

    plan = wrapping.unwrap_plan(Rstart, Rstop, (yc,xc))
    unwA = wrapping.unwrap(imgA, plan)
    unwB = wrapping.unwrap(imgB, plan)

    cc = abs(crosscorr(unwA, unwB, axes=(1,)))
    
    unwy, unwx = unwA.shape
    Rvals = range(Rstart, Rstop, Rstep)
    avg_cc = np.zeros( (len(Rvals), unwx))
    for i, R in enumerate(Rvals):
        avg_cc[i] = cc[R:R+Rstep].sum(axis=0)
    
    return avg_cc

def rot_memory( imgA, imgB, center, darks=None, qacfs=None, qacfdarks=None, Rvals=25, flatten_width=30, removeCCBkg=True, pickPeakMethod="integrate", intermediates=False):
    """Calculates the rotational cross-correlation coefficient between two image
    pairs imgA and imgB. This calculates the value rho_q where:
        rho_q = \frac{\sum(CC(A_q, B_q))}{\sqrt{AC(A_q) AC(B_q)}},
    and A_q (B_q) are images that are sliced radially in q.  The 3-tuple RRange
    determines the (Rstart, Rstop, Rstep) values that the image is sliced.
    Darks, quasi-ACFs, and qACF darks can also be provided. If quasi-ACFs are
    provided, then it calculates:
        rho = \frac{\sum(CC(imgA,imgB))}{\sqrt{qACF(A) qACF(B)}}.

    The images are preprocessed first by dividing by a gaussian of fwhm
    flatten_width (if it's nonzero) before the images are unwound and ACs and
    CCs are calculated.  Then (if the option is on), a spline is used to remove
    the AC/CC background before the peak is summed (or the max value is used).

    arguments:
        imgA - First image to calculate.  Must be a two dimensional array.
        imgB - First image to calculate.  Must be a two dimensional array.
        center - the center of the image.  Must be a tuple of (row, col) center.
        darks - A tuple of dark images (darkA, darkB). These are subtracted
            from the image. Defaults to None.
        qacfs - A tuple of quasi-ACF images (qacfA, qacfB). If provided, these
            are used in the normalization. Defaults to None.
        qacfs - A tuple of quasi-ACF dark images (qacfAdark, qacfBdark). If
            provided, these are subtracted from the dark. Defaults to None.
        removeCCBkg - This option will create a spline fit to remove the
            background after the AC and CC are calculated.  Defaults to True.
        Rvals - Radial steps for the calculation.  If this is a single value,
            then it is taken to be the step size, in pixels.  If this is a
            tuple then it specifies the (Rmin, Rmax, Rstep) pixel values where
            the correlation should be calculated. The default is 25 (steps of
            25 px each).
        flatten_width - Width used for flattening the image before unwinding and
            calculating AC/CCs. This value should be larger than the pixel size
            and defaults to 30 px. If False, flattening does not happen. 
        peakPickMethod - method to pick the peak.  Can be either "max" or
            "integrate".  "integrate" sums over an area of 16 'qspace pixels'.
            The term 'qspace pixels' is used because the amount of pixels used
            after unwinding gets smaller as R increases, but it always tries to
            sweep out 16 'qspace pixels'. Defaults to "integrate".
        intermediates - Weather or not the program should return intermediate
            arrays. If this is True, it returns a dictionary of intermediate
            calculations, such as ACFs, qACFs, spline fits..etc.  The
            calculations are numbered in the order that they are calculated. The
            default is False

    returns:
        RRanges/CC coefficients - a (2xN) array of (R, correlation-coef) values.
            Typically correlation-coef is a number between [0,1].
        intermediates - if intermediates=True, a dictonary of intermediate
            arrays is also returned.
    """
    FLATTEN_WIDTH = 30 # radius used for flattening the speckle pattern before ACs are calculated.
    PEAK_INTEGRATE_RADIUS = 8 # radius of the circle that we integrate the speckle peak over.
    import scipy.ndimage

    # function to make sure we have a list of two numpy arrays of the same size.
    check_pairs = lambda val: True if (val is not None and type(val) in (list, tuple) and len(val) == 2 and isinstance(val[0], np.ndarray) and isinstance(val[1], np.ndarray) and val[0].shape == val[1].shape) else False

    assert check_pairs((imgA, imgB)), "imgA and imgB must be arrays with the same shape"
    assert imgA.ndim == 2, "arrays must be two-dimensional"
    assert type(flatten_width) in (bool, float, int), "flatten_width must be a number or bool"
    assert type(removeCCBkg) == bool, "removeCCBkg must be a boolean"
    assert type(intermediates) == bool, "intermediates must be a boolean"
    assert pickPeakMethod in ("integrate", "max"), "pickPeakMethod must be 'integrate' or 'max'"
    assert isinstance(center, (tuple, set, list)) and len(center) == 2, "center must be 2-tuple/list/set"

    # Set up the radial ranges. If we see one element, then we go from [0:as_large_as_possible]. If we set a size(3) tuple, then we set those as the ranges.
    (ys, xs) = imgA.shape
    (yc, xc) = center
    if type(Rvals) == int:
        R = min( (xc, yc, xs-xc, ys-yc) )
        r = 0
        Rstep = Rvals
    else:
        assert isinstance(Rvals, (tuple, set, list)) and len(Rvals) == 3, "Rvals must be length-3 tuple or a single integer."
        r, R, Rstep = Rvals
        if R < r:
            r, R = R, r

    assert Rstep < R-r, "Rstep (%d) must be less than R-r (%d-%d=%d)" % (Rstep, R, r, R-r)
    RRange = r, R, Rstep

    def unw_cc(A, B, RRange):
        """ Helper function to calculate the cross correlation of two unwound
        arrays. This is similar to rot_CCF but avoids the unwrapping part since
        it's done already.
        """
        cc = abs(crosscorr(A, B, axes=(1,)))
    
        unwy, unwx = A.shape
        r, R, Rstep = RRange
        Rvals = range(0, R-r, Rstep)
        avg_cc = np.zeros( (len(Rvals), unwx))
        for i, R in enumerate(Rvals):
            avg_cc[i] = cc[R:R+Rstep].sum(axis=0)
        return avg_cc

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

    haveDarks = False
    if check_pairs(darks):
        if imgA.shape == darks[0].shape:
            imgA -= darks[0]
            imgB -= darks[1]
            haveDarks = True

    haveqACFs = check_pairs(qacfs)
    haveqACFdarks = check_pairs(qacfdarks)

    if haveqACFs:
        if haveqACFdarks:
            if qacfs[0].shape == qacfdarks[0].shape:
                qacfA = qacfs[0] - qacfdarks[0]
                qacfB = qacfs[1] - qacfdarks[1]
            else:
                haveqACFdarks = False
                qacfA = qacfs[0]
                qacfB = qacfs[1]
        else:
            qacfA = qacfs[0]
            qacfB = qacfs[1]

    file_string = "reading (img, "
    if haveDarks:
        file_string += "dark, "
    if haveqACFs:
        file_string += "qACF, "
    if haveqACFdarks:
        file_string += "qACFdarks, "
    file_string = file_string[0:-2] + ") -> "

    if flatten_width:
        flat_string = "flattening (fwhm %d px) -> " % flatten_width
    else:
        flat_string = ""

    unw_string = "unwinding with center (%d, %d) from (%d, %d) px -> " % (xc, yc, r, R)

    if removeCCBkg:
        removeCC_string = "removing backgrounds -> "
    else:
        removeCC_string = ""

    if pickPeakMethod == "integrate":
        peakString = "integrating speckle peak over %d px -> " % (PEAK_INTEGRATE_RADIUS*2)
    else:
        peakString = "picking maximum value of peak ->"

    print(file_string + flat_string + unw_string + "CC/AC and binning in steps of %d px -> "  % Rstep + removeCC_string + peakString + "returning result")

    if flatten_width:
        flatA = averaging.smooth_with_gaussian(imgA, flatten_width)
        flatB = averaging.smooth_with_gaussian(imgB, flatten_width)
        store('01-A_flatten_smoothed', flatA)
        store('01-B_flatten_smoothed', flatB)

        imgA = imgA/flatA
        imgB = imgB/flatB
        store('01-A_flatten', imgA)
        store('01-B_flatten', imgB)
        if haveqACFs:
            flatqacfA = averaging.smooth_with_gaussian(qacfA, flatten_width)
            flatqacfB = averaging.smooth_with_gaussian(qacfB, flatten_width)
            store('01-A_qACF_flatten_smoothed', flatqacfA)
            store('01-B_qACF_flatten_smoothed', flatqacfB)

            qacfA = qacfA/flatqacfA
            qacfB = qacfB/flatqacfB
            store('01-A_qACF_flatten', qacfA)
            store('01-B_qACF_flatten', qacfB)

    plan = wrapping.unwrap_plan(r, R, center)
    unwA = wrapping.unwrap(imgA.astype('float'), plan)
    unwB = wrapping.unwrap(imgB.astype('float'), plan)
    store('02-A-unwrapped', unwA)
    store('02-B-unwrapped', unwB)

    rad_per_pix = 2*np.pi/len(unwA[0]) # this the num of radians in 1 pixel of the unwound array. It's needed for determining the num. ctl. pts. for removeCCBkg

    cc = unw_cc(unwA, unwB, RRange)
    store("04-cc", cc)
    if haveqACFs:
        unwqA = wrapping.unwrap(qacfA.astype('float'), plan)
        unwqB = wrapping.unwrap(qacfB.astype('float'), plan)
        store('02-A_qACF_unwrapped', unwqA)
        store('02-B_qACF_unwrapped', unwqB)

        autoA = unw_cc(unwA, unwqA, RRange)
        autoB = unw_cc(unwB, unwqB, RRange)
        store("04-A_qACF", autoA)
        store("04-B_qACF", autoB)
    else:
        autoA = unw_cc(unwA, unwA, RRange)
        autoB = unw_cc(unwB, unwB, RRange)
        store("04-A_ac", autoA)
        store("04-B_ac", autoB)

    def linfit(img, nx, order=3):
        """fit an lineplot to a spline.

        parameters:
            line - 1d linescan.
            nx - number of control points in x
            order - order of splines. Defaults to 3.

        returns:
            spline - the linear fitted image, of the same dimension as img.
        """
        j = complex(0,1)
        xs = img.shape[0]
    
        # reduce image down to (nx,) shape
        x = np.mgrid[0:xs-1:nx*j]
    
        reduced = scipy.ndimage.map_coordinates(img, np.array([x]), order=order)
    
        # blow image back up to img.shape
        rxs = reduced.shape[0]
        outx = np.mgrid[0:rxs-1:xs*j]
        spl = scipy.ndimage.map_coordinates(reduced, np.array([outx]), order=order)
    
        return spl

    def spl_lines(data):
        # make_even sets the number of control points to an even value so there is never a control point on the speckle peak.
        make_even = lambda v: v % 2 == 1 and v - 1 or v
        spl = np.zeros_like(data)
        mask_npix =  calc_radial_maskpx()
        ys, xs = data.shape
        for i, mask_px in enumerate(mask_npix):
            nx = make_even(int(np.floor(xs/mask_px)))
            spl[i] = linfit(data[i], nx)
        return spl

    def calc_radial_maskpx():
        """ Returns a (1,m) set of # px in mask
        """
        Ravg = np.arange(r, R, Rstep, dtype=float) + Rstep/2.0
        return np.floor(2 * PEAK_INTEGRATE_RADIUS / (Ravg * rad_per_pix))


    if removeCCBkg:
        spl_cc= spl_lines(cc)
        spl_autoA = spl_lines(autoA)
        spl_autoB = spl_lines(autoB)
        store("05-cc_spline", spl_cc)
        store("05-A_ac_spline", spl_autoA)
        store("05-B_ac_spline", spl_autoB)

        cc = cc - spl_cc
        autoA = autoA - spl_autoA
        autoB = autoB - spl_autoB
        store("05-cc_remCCbkg", cc)
        store("05-A_ac_remCCbkg", autoA)
        store("05-B_ac_remCCbkg", autoB)

    if pickPeakMethod == "integrate":
        mask = np.zeros_like(cc)
        mask_npix = calc_radial_maskpx()
        (ys, xs) = cc.shape
        xmid = xs/2
        for i in range(ys):
            halfmask = mask_npix[i]/2
            mask[i, xmid-halfmask:xmid+halfmask] = 1

        CCval = (cc*mask).sum(axis=1)
        ACval = np.sqrt((autoA*mask).sum(axis=1) * (autoB*mask).sum(axis=1))

        store("06-cc_summed_region", cc*mask)
        store("06-A_ac_summed_region", autoA*mask)
        store("06-B_ac_summed_region", autoB*mask)
    else: # pickPeakMethod == "max"
        CCval = cc.max()
        ACval = np.sqrt(autoA.max()*autoB.max())
    
    Ravg = np.arange(r, R, Rstep, dtype=float) + Rstep/2.0
    ccval = np.real(CCval/ACval)

    R_ccvals = np.vstack((Ravg, ccval)).swapaxes(0,1)    
    if intermediates:
        return R_ccvals, intermediate_arrays
    else:
        return R_ccvals

def _maxRadius(img, center):
    """ Returns the maximum radius of which the angular cross correlation
        can be calculated.
    """
    ys, xs = img.shape
    yc, xc = center
    return min(ys, xs, ys-yc, xs-xc)
