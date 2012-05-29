import numpy as np

#from crosscorr import fftconvolve
from . import shape
from . import averaging

def _convert_to_3d(img):
    """ get image shape and reshape image to three dimensional (if necessary).
    Most of the g2 functions require a 3d image
    arguments:
        img - image to check shape
    returns:
        (fr, ys, xs) - a tuple of the image shape.
    """
    assert type(img) == np.ndarray, "image is not an array"
    dim = img.ndim
    assert dim in (1,2,3), "image is not 1d, 2d, or 3d"
    if dim == 3:
        (fr, ys, xs) = img.shape
        return img, fr, ys, xs
    elif dim == 2:
        fr = 1
        (ys, xs) = img.shape
    else:
        fr =1
        ys = 1
        (xs,) = img.shape

    return img.reshape((fr, ys, xs)), fr, ys, xs

def shift_img_up(img):
    """Check to make sure all the values in the image are > 0.  If there are
    pixels < 0, it shifts the whole image so that all of the pixels are > 0. 
    inputs:
        img - image to test
    returns:
        img - image shifted up so that all px > 0
    """
    assert type(img) == np.ndarray, "image is not an array"
    if img.min() < 0:
        print("adjusting all pixels by %1.2f so that they are > 0" % (1.001*img.min()))
        return img - img.min()*1.1
    elif img.min() == 0:
        print("adjusting all pixels by 1 so that they are > 0")
        return img + 1        
    else:
        return img

def normalize_intensity(img, method="maxI"):
    """Normalize each frame of an image.
    arguments:
        img - img to normalize.  Must be 3d.
        method - method of normalization, either by maximum intensity (maxI) or summed intensity (sumI).  Note, sumI is the summed intensity in each frame, not the entire 3d image.
    returns:
        img - normalized image.    
    """
    assert type(img) == np.ndarray, "image is not an array"

    if img.ndim != 3:
        return img

    (fr,ys,xs) = img.shape

    if method == "sumI":
        print("Normalizing each frame by total frame intensity")
        for f in range(fr):
            img[f] = img[f]/img[f].sum()
    else: # assume maxI
        print("Normalizing each frame by maximum frame intensity")
        for f in range(fr):
            img[f] = img[f]/img[f].max()

    return img

def g2_symm_norm(img, numtau, qAvg = ("circle", 10)):
    """ calculate correlation function g_2 with a symmetric normalization.
    The symmetric normalization is defined as
    g_2 (q, tau) = < I(q,t) * I(q, t+tau) >_t / (<I(q,t)>_q,left * <I(q,t)>_q,right)
    where <>_t (<>_q) is averaging over variable t (q). The left (right) average time over frames 1:numtau (fr-numtau:fr).
    arguments:
        img - 3d array
        numtau - tau to correlate
        qAvg - a list of the averaging type and the size. The possible types and the size parameter are:
            "square" - the parameter size is size dimension of the square, in pixels
            "circle" - the size is the radius of the circle, in pixels
            "gaussian" - size is the FWHM of the gaussian, in pixels
    returns:
        g2 - correlation function for numtau values
    """

    """ 4/25 - same numerical for  qAvg=("none") and qAvg("box") results as
        g2Coefficient(testdata, numtau, boxsize=0, normtype='symm-noqavg')
        and
        g2Coefficient(testdata, numtau, boxsize=10, normtype='symm')
    slightly slower
    """
    import sys

    assert type(qAvg) in (list, tuple) and len(qAvg) == 2, "unknown size for qAvg"
    avgType, avgSize = qAvg
    assert avgType in _averagingFunctions.keys(), "unknown averaging type %s" % qAvgType

    img, fr, ys, xs = _convert_to_3d(img)

    tauvals = _numtauToTauvals(numtau)

    ntaus = len(tauvals)
    result = np.zeros((ntaus, ys, xs), dtype=float)

    IQ = _averagingFunctions[avgType](img, avgSize)

    i = 0 # we need I because tauvals may not be in order, linear, or start at 0.
    for t in tauvals:
        result[i] = g2_numerator(img, t)/(_time_average(IQ, 0, fr-t)*_time_average(IQ, t, fr))
        i += 1
        sys.stdout.write("\rtau %d (%d/%d)" % (t, i, ntaus))
    print("")
    return result

def g2_symm_borthwick_norm(img, numtau, qAvg = ("circle", 10)):
    """ calculate correlation function g_2 with Matt Borthwick's symmetric normalization.
    His symmetric normalization is defined as
    g_2 (q, tau) = < I(q,t) * I(q, t+tau) >_t / (<I(q,t)>_q,left * <I(q,t)>_q,right) * <I(q, t)>_q,t^2/<I(q, t)>_t^2
    where <>_t (<>_q) is averaging over variable t (q). The left (right) average time over frames 1:numtau (fr-numtau:fr).
    This varies slightly from the symmetric normalization as it added a term indended to correct for long-range trends in intensity.
    arguments:
        img - 3d array
        numtau - tau to correlate
        qAvg - a list of the averaging type and the size. The possible types and the size parameter are:
            "square" - the parameter size is size dimension of the square, in pixels
            "circle" - the size is the radius of the circle, in pixels
            "gaussian" - size is the FWHM of the gaussian, in pixels
    returns:
        g2 - correlation function for numtau values
    """
    import sys
    assert type(qAvg) in (list, tuple) and len(qAvg) == 2, "unknown size for qAvg"
    avgType, avgSize = qAvg
    assert avgType in _averagingFunctions.keys(), "unknown averaging type %s" % qAvgType

    img, fr, ys, xs = _convert_to_3d(img)

    tauvals = _numtauToTauvals(numtau)

    ntaus = len(tauvals)
    result = np.zeros((ntaus, ys, xs), dtype=float)

    IQ = _averagingFunctions[avgType](img, avgSize)
    IQT = _time_average(IQ)
    IT = _time_average(img)

    IQTsqIQsq = IQT*IQT/(IT*IT)
    i = 0 # we need I because tauvals may not be in order, linear, or start at 0.
    for t in tauvals:
        result[i] = IQTsqIQsq * g2_numerator(img, t)/(_time_average(IQ, 0, fr-t)*_time_average(IQ, t, fr))
        i += 1
        sys.stdout.write("\rtau %d (%d/%d)" % (t, i, ntaus))
    print("")
    return result
    
def g2_standard_norm(img, numtau, qAvg = ("circle", 10)):
    """ calculate correlation function g_2 with a standard normalization.
    The standard normalization is defined as
    g_2 (q, tau) = < I(q,t) * I(q, t+tau) / (<I(q,t)>_q * <I(q,t+tau)>_q) >_t * <I(q,t)>_q,t^2 / <I(q,t)>_t^2
    where <>_t (<>_q) is averaging over variable t (q).
    arguments:
        img - 3d array
        numtau - tau to correlate
        qAvg - a list of the averaging type and the size. The possible types and the size parameter are:
            "square" - the parameter size is size dimension of the square, in pixels
            "circle" - the size is the radius of the circle, in pixels
            "gaussian" - size is the FWHM of the gaussian, in pixels
    returns:
        g2 - correlation function for numtau values
    """
    assert type(qAvg) in (list, tuple) and len(qAvg) == 2, "unknown size for qAvg"
    avgType, avgSize = qAvg
    assert avgType in _averagingFunctions.keys(), "unknown averaging type %s" % qAvgType
    assert avgType != "none", "'none' averaging not supported for g2StandardNorm()."

    img, fr, ys, xs = _convert_to_3d(img)

    if ys < avgSize or xs < avgSize:
        print("warning: size for averaging %d is larger than array size, changing to %d" % min(xs,ys))
        avgSize = min(ys,xs)

    # in standard normalization defined by Borthwick, the img array is modified by the q-averaged intensity then used to calculate g2.
    IT = _time_average(img)
    IQ = _averagingFunctions[avgType](img, avgSize)
    IQT = _time_average(IQ)
    return g2_no_norm(img/IQ, numtau)*IQT*IQT/(IT*IT)
    
def g2_plain_norm(img, numtau): # formerly sujoy norm
    """ calculate correlation function g_2 with a 'plain' normalization.
    The 'plain' normalization is defined as
    g_2 (q, tau) = < I(q,t) * I(q, t+tau) >_t / <I(q,t)>_t^2
    where <>_t (<>_q) is averaging over variable t (q).
    arguments:
        img - 3d array
        numtau - tau to correlate
        qAvg - a list of the averaging type and the size. The possible types and the size parameter are:
            "square" - the parameter size is size dimension of the square, in pixels
            "circle" - the size is the radius of the circle, in pixels
            "gaussian" - size is the FWHM of the gaussian, in pixels
    returns:
        g2 - correlation function for numtau values
    """

    IT = _time_average(img)
    return g2_no_norm(img, numtau)/(IT*IT)

def g2_no_norm(img, numtau):
    """ calculate correlation function g_2 numerator without normalization.
    The is defined as
    g_2 (q, tau) = < I(q,t) * I(q, t+tau) >_t
    where <>_t is averaging over variable t.
    arguments:
        img - 3d array
        numtau - tau to correlate
    returns:
        g2 - correlation function for numtau values
    """
    import sys
    img, fr, ys, xs = _convert_to_3d(img)

    tauvals = _numtauToTauvals(numtau)

    ntaus = len(tauvals)
    numerator = np.zeros((len(tauvals), ys, xs), dtype=float)
    i = 0 # we need I because tauvals may not be in order, linear, or start at 0.
    for t in tauvals:
        numerator[i] = g2_numerator(img, t)
        i += 1
        sys.stdout.write("\rtau %d (%d/%d)" % (t, i, ntaus))
    print("")
    return numerator

def _numtauToTauvals(numtau, maxtau=0):
    """ program to convert numtau to a list of values to iterate over.
    arguments:
        numtau - tau's to correlate.  If it is a single number, it generates a list from (0, numtau), if it's a list or tuple of size 2, it will do the range between those two taus.
        maxtau - Maximum possible tau value.  If it exists, then the function checks to see if all taus are smaller than this value.
    returns:
        tauvals - list of taus to correlate.
    """
    assert type(numtau) in (int, list, tuple), "_numtauToTauvals(): numtau is not recognizeable."
    if type(numtau) == int:
        tauvals = range(numtau)
    elif type(numtau) in (list, tuple) and len(numtau) == 2:
        tst, tend = numtau
        tauvals = range(tst, tend)
    else:
        # We must have a list of tauvals
        tauvals = list(numtau)

    assert np.array(tauvals).min() >= 0, "All taus must be larger non-negative. taus: (%s)" % repr(tauvals)
    if maxtau > 0:
        assert np.array(tauvals).max() < maxtau, "All taus must be smaller than %d. taus: (%s)" % (maxtau, repr(tauvals))

    return tauvals

def g2_numerator(img, onetau):
    """ Calculates <I(t) I(t + tau)>_t for a single value of tau.
    arguments:
        img - data to caclculate g2.  Must be 3d.
        onetau - a single tau value.  Must be integer and less than the total frames in img.
    returns:
        numerator - g2 numerator calcualted for one tau value.
    """
    assert img.ndim == 3, "g2_numerator: Must be a three dimensional image."
    (fr, ys, xs) = img.shape
    numerator = np.zeros((ys,xs), dtype='float')
    for f in range(fr-onetau):
        numerator += img[f]*img[f+onetau]
    numerator = numerator/(fr-onetau)
    # numerator = np.average(img[0:fr-onetau] * img[onetau:fr], axis=0) # This is slower than the above implementaion.
    return numerator

def _noAverage(img, size):
    return img

def _time_average(img, start='', stop=''):
    if start == '' and stop == '':
        return np.average(img,axis=0)
    else:
        assert type(start) == int and type(stop) == int, "start and stop must be integers"
        return np.average(img[start:stop],axis=0)

# This is a dictionary of averaging functions.  They all take (img, size) as arguments.
_averagingFunctions = {
    "none": _noAverage,
    "circle" : averaging.smooth_with_circle,
    "square" : averaging.smooth_with_rectangle,
    "rectangle": averaging.smooth_with_rectangle,
    "gaussian" : averaging.smooth_with_gaussian,
}
