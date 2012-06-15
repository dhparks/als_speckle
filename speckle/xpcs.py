"""Library for XPCS and photon correlation experiments.  The library calculates
the photon correlation function for a set of frames represented as a 3d numpy
array.  The library will also calculate the correlation function for a single-
photon timestamping detector.

Author: Keoki Seu (kaseu@lbl.gov)

"""
import numpy as np
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

def g2_numerator(img, onetau):
    """ Calculates <I(t) I(t + tau)>_t for a single value of tau.
    arguments:
        img - data to caclculate g2.  Must be 3d.
        onetau - a single tau value.  Must be integer and less than the total frames in img.
    returns:
        numerator - g2 numerator calculated for one tau value.
    """
    assert img.ndim == 3, "g2_numerator: Must be a three dimensional image."
    (fr, ys, xs) = img.shape
    numerator = np.zeros((ys,xs), dtype='float')
    for f in range(fr-onetau):
        numerator += img[f]*img[f+onetau]
    numerator = numerator/(fr-onetau)
    # numerator = np.average(img[0:fr-onetau] * img[onetau:fr], axis=0) # This is slower than the above implementaion.
    return numerator

def g2_numerator_fft(img,taus=None):
    """ Calculates <I(t) I(t + tau)>_t for specified values of tau via fft autocorrelation method. For large N this may be significantly faster than the g2_numerator function, but for small N or a small number of tau the g2_numerator function may be faster.
    
    If data sets are very large this function may lead to malloc errors, necessitating fallback to g2_numerator or more sophisticated handling of the dataset.
    
    arguments:
        img - data to caclculate g2.  Must be 3d.
        taus - iterable set of tau values where you want g2. all taus are evaluated by the fft so a limited set of taus does not provide a speed up.
    returns:
        numerator - g2 numerator calculated for requested tau values.
    """
    assert img.ndim == 3, "g2_numerator: Must be a three dimensional image."
    (fr, ys, xs) = img.shape

    IDFT = numpy.fft.ifftn
    DFT = numpy.fft.fftn

    img2       = np.zeros((2*fr,ys,xs),float)
    img2[0:fr] = img
    numerator  = abs(IDFT(abs(DFT(img2,axes=(0,)))**2,axes=(0,)))[0:fr]
    for f in range(fr): numerator[f] *= 1./(fr-f)
    
    if taus != None: return numerator[taus]
    else: return numerator

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

#
######################## Single photon correlations ########################
#

# maximum pixel size for single photon detector
SP_MAX_SIZE = 4096

def sp_bin_by_space_and_time(data, frameTime, xybin=8, counterTime=40.0e-9):
    """Takes a dataset collected by the SSI photon counting fast camera and bins
    the data in time and xy.

    arguments:
        data - data to bin.  This is an (N, 4) array where N is the number of elements.
        frameTime - Amout of time between bins.  Measured in seconds
        xybin - amount to bin in x and y.  defaults to 8, which is a 512x512 output.
        counterTime - clock time of the camera. Anton Tremsin says its 40 ns.
    returns:
        binnedData - a 3-dimension array (frames, y, x) of the binned data.
    """
    assert isinstance(data, np.ndarray), "data must be an ndarray"
    assert data.shape[1] == 4, "Second dimension must be 4."
    assert np.isreal(frameTime), "frameTime (%s) must be real" % repr(frameTime)
    assert np.isreal(counterTime), "counterTime (%s) must be real" % repr(counterTime)
    # TODO: This could be accelerated if written for a GPU

    firsttime = data[:,3].min()
    lasttime = data[:, 3].max()
    datalen = len(data[:, 0])
    
    nbins = int(np.ceil(float(lasttime - firsttime)*counterTime/frameTime))
    xybinnedDim = int(np.ceil(SP_MAX_SIZE/xybin))

    binnedData = np.zeros((nbins, xybinnedDim, xybinnedDim), dtype='int')
    bin_edges = np.linspace(firsttime, lasttime, nbins+1)
    # slightly increase the last bin.  For some reason the last bin is slightly too small, and the photons on the edges are not included
    bin_edges[-1] = bin_edges[-1]*1.01

    # If the frameTime is larger than the total acq time, then sum up everything
    if frameTime >= (lasttime-firsttime)*counterTime:
        histfn = lambda a,b: (len(a), b)
    else:
        # this is the default; bin data using histogram
        histfn = np.histogram

    for i in range(xybinnedDim):
        x = data[:, 0]
        y = data[:, 1]
        xc = np.argwhere( (i*xybin <= x) & (x < (i+1)*xybin) )
        idx_to_delete = np.array([])
        for j in range(xybinnedDim):
            yc = np.argwhere( (j*xybin <= y) & (y < (j+1)*xybin) )
            isect, a, b = intersect(xc, yc, assume_unique=True)
            if len(isect) != 0:
                binned, bin_edges = histfn(data[isect,3]+0.5, bin_edges)
                binnedData[:, i, j] = binned
            idx_to_delete = np.append(idx_to_delete, isect)
        data = np.delete(data, idx_to_delete, axis=0)
#        print i, j, len(idx_to_delete)

    if binnedData.sum() != datalen:
        print "warning: # of binned data photons (%d) do not match with dataset (%d)" % (binnedData.sum(), datalen)

    return binnedData

def sp_bin_by_time(data, frameTime, counterTime=40.0e-9):
    """Takes a stream of photon incidence times and bins the data in time.

    arguments:
        data - Data to bin.  This is a 1 dimensional array of incidence times.
        frameTime - Amount of time for each bin.  Measured in seconds
        counterTime - Clock time of the camera. Anton Tremsin says its 40 ns.
    returns:
        binnedData - a 3-dimension array (frames, y, x) of the binned data.
        also returns edges?? check
    """
    assert isinstance(data, np.ndarray), "Data must be an ndarray."
    assert data.ndim == 1, "Data must be 1-dimensional."
    assert np.isreal(frameTime), "frameTime (%s) must be real" % repr(frameTime)
    assert np.isreal(counterTime), "counterTime (%s) must be real" % repr(counterTime)

    firsttime = data.min()
    lasttime = data.max()

    nbins = int(np.ceil(float(lasttime - firsttime)*counterTime/frameTime))

    bin_edges = np.linspace(firsttime, lasttime, nbins)
    bin_edges[-1] += 1

    return np.histogram(data+0.5, bin_edges)[0]

def sp_sum_bin_all(data, xybin=4, counterTime=40.0e-9):
    """ Sum and bin all of the data from the single photon detector into one frame.

    arguments:
        data - a (N, 4) sized array of the data generated by the single photon detector.
    returns:
        img - a 2d image of the entire dataset.
    """
    assert isinstance(data, np.ndarray), "Data must be an ndarray."
    assert data.shape[1] == 4, "Second dimension must be 4."

    firsttime = data[:, 3].min()
    lasttime = data[:,3].max()

    return sp_bin_by_space_and_time(data, lasttime*counterTime, xybin=xybin, counterTime=counterTime)

def intersect(a, b, assume_unique=False):
    """ implements the matlab intersect() function. Details of the function are
    here: http://www.mathworks.com/help/techdoc/ref/intersect.html

    arguments:
        a - 1st array to test
        b - 2nd array to test
        assume_unique - If True, assumes that both input arrays are unique, which speeds up the calculation.  The default is False.
    returns:
        res - values common to both a and b.
        ia - indicies of a such that res = a[ia]
        ib - indicies of b such that res = b[ib]
    """
    res = np.intersect1d(a, b, assume_unique=False)
    ai = np.nonzero(np.in1d(a, res, assume_unique=False))
    bi = np.nonzero(np.in1d(b, res, assume_unique=False))
    return res, ai, bi

def sp_autocorrelation(data, p=30, m=2):
    """Implements the time-to-tag correlation algorithm outlined by Wahl et al. in Opt. Exp. 11 3583

    arguments:
        data - A sorted (1xN) array of incidence times.
        p - number of linear points. Defaults to 30.
        m - factor that the time is changed for each correlator. Defaults to 2.
    returns:
        corr - A (1 x bins) autocorrelation of data.
        corrtime - A (1 x bins ) list of time ranges.
    """
    return sp_crosscorrelation(data, data, p, m)

def sp_crosscorrelation(d1, d2, p=30, m=2):
    """Implements the time-to-tag correlation algorithm outlined by Wahl et al. in Opt. Exp. 11 3583

    arguments:
        d1 - A sorted (1xN) array of incidence times for the 1st signal.
        d2 - A sorted (1xN) array of incidence times for the 2nd signal.
        p - number of linear points. Defaults to 30.
        m - factor that the time is changed for each correlator. Defaults to 2.
    returns:
        corr - A (1 x bins) crosscorrelation between (d1, d2).
        corrtime - A (1 x bins ) list of time ranges.

    """
    import sys
    assert isinstance(d1, np.ndarray) and isinstance(d2, np.ndarray), "d1 and d2 must 1-dim be arrays"

    d1 = d1.astype('int64')
    d1 = d1[d1.argsort()]
    d2 = d2.astype('int64')
    d2 = d2[d2.argsort()]

    tmax = max(d1.max() - d1.min(), d2.max() - d2.min())
    Ncorr = np.ceil(np.log(tmax/float(p) + 1)/np.log(float(m)) - 1).astype('int') + 1

    corrtime = np.zeros(Ncorr*p)
    corr = np.zeros_like(corrtime)
    w1 = np.ones_like(d1)
    w2 = np.ones_like(d2)

    delta = 1
    shift = 0
    for i in range(Ncorr):
        sys.stdout.write("\rcorrelator %d/%d" % (i+1, Ncorr))
        sys.stdout.flush()
        for j in range(p):
            shift = shift + delta
            lag = np.floor(shift/delta).astype('int')
            (isect, ai, bi) = intersect(d1, d2 + lag, assume_unique=True)
            corr[i*p+j] = np.dot(w1[ai], w2[bi]) / float(delta)
            corrtime[i*p+j] = shift
            
        delta = delta*m
        
        d1, w1 = _half_data(d1, m, w1)
        d2, w2 = _half_data(d2, m, w2)

    print("")

    # Normalize. Another part of normalization is dividing by corr[i*p+j]/delta (done above)
    for i in range(Ncorr*p):
        corr[i] = corr[i]*tmax/(tmax - corrtime[i])

    return corr, corrtime

def _sort_data(data, col=3):
    """ Helper function to sort data by increasing photon incidence value along column col.
    arguments:
        data - data to process.  Must be two dimensional
    returns:
        sorted data by increasing photon incidence value
    """
    assert isinstance(data, np.ndarray) and data.ndim == 2, "data must be a 2d array"
    return data[data[:, col].argsort(),:]

def _list_duplicates(seq):
    """ 
    helper function used by _half_data() to get a list of duplicate entries in an iteratable
    from http://stackoverflow.com/questions/5419204/index-of-duplicates-items-in-a-python-list
    """
    from collections import defaultdict

    tally = defaultdict(list)
    for i,item in enumerate(seq):
        tally[item].append(i)
    return ((key,locs) for key,locs in tally.items() if len(locs)>1)

def _half_data(data, m, w):
    """ Helper function that halves the data by dividing the incidence times in data by m and finding duplicates.  The w matrix is augmented depending on the number of duplicates found.  The algorithm is found in Opt. Exp. 11 3583.
    """
    import sys
    halvedData = np.floor(data/float(m)).astype('int64')
    wsumbefore = w.sum()
    if (halvedData - np.roll(halvedData, 1) == 0).any():
        listDupIterator = _list_duplicates(halvedData)
        valsToRemove = []
        for val, keys in sorted(listDupIterator):
            valsToRemove.append(keys[1:])
            w[keys[0]] += w[keys[1:]].sum()
        w = np.delete(w, valsToRemove)
        data = np.delete(halvedData, valsToRemove)
        sys.stdout.write("\r\t\t  | removing %6d duplicates." % (len(valsToRemove)))
        sys.stdout.flush()
        if wsumbefore != w.sum():
            print("WARNING: values in weighted array do not match after eliminating duplicates")
    else:
        data = halvedData

    return data, w
