"""Library for XPCS and photon correlation experiments.  The library calculates
the photon correlation function for a set of frames represented as a 3d numpy
array.  The library will also calculate the correlation function for a single-
photon timestamping detector.

Authors: Keoki Seu (kaseu@lbl.gov), Daniel Parks (dhparks@lbl.gov)

"""
import numpy as np
import sys
from . import averaging
import math

# keep a record of gpu fft plans
fftplans = {}

def _convert_to_3d(img):
    """ get image shape and reshape image to three dimensional (if necessary).
    Most of the g2 functions require a 3d image.

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
        ys = 1
        (fr, xs) = img.shape
    else:
        xs = 1
        ys = 1
        (fr,) = img.shape

    return img.reshape((fr, ys, xs)), fr, ys, xs

def shift_img_up(img):
    """Check to make sure all the values in the image are > 0.  If there are
    pixels < 0, it shifts the whole image so that all of the pixels are > 0. 

    arguments:
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
        method - method of normalization, either by maximum intensity (maxI) or
            summed intensity (sumI).  Note, sumI is the summed intensity in each
            frame, not the entire 3d image.

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

def g2_symm_norm(img, numtau, qAvg = ("circle", 10), fft=False):
    """ calculate correlation function g_2 with a symmetric normalization.

    The symmetric normalization is defined as
    
    g_2 (q, tau) = < I(q,t) * I(q, t+tau) >_t / (<I(q,t)>_q,left * <I(q,t)>_q,right)
    
    where <>_t (<>_q) is averaging over variable t (q). The left (right) average
    time over frames 1:numtau (fr-numtau:fr).

    arguments:
        img - 3d array
        numtau - tau to correlate
        qAvg - a list of the averaging type and the size. The possible types and
            the size parameter are:
            "square" - the parameter size is dimension of the square, in pixels
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
    assert avgType in _averagingFunctions.keys(), "unknown averaging type %s" % avgType

    img, fr, ys, xs = _convert_to_3d(img)

    tauvals = _numtauToTauvals(numtau, maxtau=fr)

    ntaus = len(tauvals)
    result = np.zeros((ntaus, ys, xs), dtype=float)

    IQ = _averagingFunctions[avgType](img, avgSize)
    
    numerator = _g2_numerator(img,tauvals,fft=fft)

    for i, tau in enumerate(tauvals):
        result[i] = numerator[i]/(_time_average(IQ, 0, fr-tau)*_time_average(IQ, tau, fr))
        sys.stdout.write("\rtau %d (%d/%d)" % (tau, i, ntaus))
        sys.stdout.flush()
    print("")
    return result

def g2(data,numtau=None,norm="plain",qAvg=("circle",10),gpu_info=None,silent=True):
    
    """ Calculate correlation function g_2. A variety of normalizations can be
    selected through the optional kwarg "norm".
    
    arguments:
        data - 3d array. g2 correlates along time axis (axis 0)
        numtau - tau to correlate. If a single number, all frame spacings
            from (0, numtau) will be calculated. If a list or tuple of length 2,
            calculate g2 within the range between those two values.
            If not supplied, assume the range is (0,nframes/2).
        norm - the normalization specification. See below for details.
        qAvg - some norms require an average over the q coordinate. This requires
            a 2-tuple of (shape,size). The possible types and the size parameter
            are:
            "square" - the parameter size is dimension of the square, in pixels
            "circle" - the size is the radius of the circle, in pixels
            "gaussian" - size is the FWHM of the gaussian, in pixels
            Internally, this average is calculated as a convolution of data with
            the kernel specified here.
        gpu_info - for fast calculation of the g2 function, a gpu context
            can be passed to this function. This requires the gpu context
            to be created outside of this function's invocation, typically
            though speckle.gpu.init(). If gpu_info is supplied, fft methods are
            automatically used and the value of the fft argument is overridden.

    returns:
        g2 - correlation function for numtau values 
    
    The following g_2(q,tau) normalizations and their fomulas are available [<>_t (<>_q) 
    indicates averaging over variable t (q)]:
    
        1. "borthwick" from Matthew Borthwick's thesis (pg 120):
        < I(q,t) * I(q, t+tau) >_t / (<I(q,t)>_q,left * <I(q,t)>_q,right) * <I(q, t)>_q,t^2/<I(q, t)>_t^2
    
        2. "none" no normalization is applied. NOT RECOMMENDED.
        < I(q,t) * I(q, t+tau) >_t 
    
        3. "plain" this is the canonical g2 function:
        < I(q,t) * I(q, t+tau) >_t / <I(q,t)>_t^2

        4. "standard"
        < I(q,t) * I(q, t+tau) / (<I(q,t)>_q * <I(q,t+tau)>_q) >_t * <I(q,t)>_q,t^2 / <I(q,t)>_t^2
    
        5. "symmetric" The left (right) average time over frames 1:numtau (fr-numtau:fr).
        < I(q,t) * I(q, t+tau) >_t / (<I(q,t)>_q,left * <I(q,t)>_q,right)

    """ 
    
    import sys
    
    # check the norm specification
    if norm == None: norm = "plain"
    assert norm in ("borthwick","none","plain","standard","symmetric")
    
    # prep the data. make 3d and convert numtau to a sequence "tauvals"
    data, fr, ys, xs = _convert_to_3d(data)
    if numtau == None: numtau = int(round(fr/2))
    tauvals = _numtauToTauvals(numtau, maxtau=fr)
    
    # depending on the normalization type, prep the normalizing factors
    if norm in ("symmetric","standard","borthwick"):
        
        # check that the averaging kernel is properly specified
        assert isinstance(qAvg,(list,tuple)) and len(qAvg) == 2, "unknown size for qAvg"
        avgType, avgSize = qAvg
        assert avgType in _averagingFunctions.keys(), "unknown averaging type %s" % avgType
        if ys < avgSize or xs < avgSize:
            print("warning: size for averaging %d is larger than array size, changing to %d" % (avgSize, min(xs,ys)))
            avgSize = min(ys,xs)

        sys.stdout.write("calculating q-average (may take a while)\n")
        IQ  = _averagingFunctions[avgType](data, avgSize)
        
    if norm in ("standard","plain","borthwick"):
        IT  = _time_average(data)
        IT2 = IT*IT
        
    if norm in ("standard","borthwick"):
        IQT  = _time_average(IQ)
        IQT2 = IQT*IQT
        
    if norm == "standard": data /= IQ # dont understand this but whatever
        
    # now compute the numerator of the g2 function; autocorrelate along t axis.
    # if the dataset is large, computing the fft in a single pass may be
    # inefficient as it no longer fits in memory. for this reason, the data
    # is broken into 8x8 pixel tiles and correlated in batches.

    import time
    if not silent: print "g2 numerator"
    numerator = _g2_numerator(data,batch_size=bs,gpu_info=gpu_info)[tauvals]

    # normalize the numerator. depending on the norm method different values are calculated.
    if not silent: print "g2 normalizing"
    if norm   == "none":     pass # so just the numerator is returned
    elif norm == "plain": numerator /= IT2
    elif norm == "standard": numerator *= IQT2/IT2
    elif norm == "symmetric":
        for i, tau in enumerate(tauvals):
            numerator[i] /= (_time_average(IQ, 0, fr-tau)*_time_average(IQ, tau, fr))
            sys.stdout.write("\rtau %d (%d/%d)" % (tau, i, ntaus))
            sys.stdout.flush()
        sys.stdout.write('\n')
    elif norm == "borthwick":
        numerator *= IQT2/IT2
        for i, tau in enumerate(tauvals):
            numerator[i] /= (_time_average(IQ, 0, fr-tau)*_time_average(IQ, tau, fr))
            sys.stdout.write("\rtau %d (%d/%d)" % (tau, i, ntaus))
            sys.stdout.flush()
        sys.stdout.write('\n')

    return numerator

def _g2_numerator(data,batch_size=64,gpu_info=None):
    
    # check incoming
    assert data.ndim == 3, "g2_numerator: Must be a three dimensional image."
    (fr, ys, xs) = data.shape

    def _prep_cpu():

        if batches > 0: tmp1 = np.zeros((batch_size,2*fr),np.float32)
        else: tmp1 = 0
            
        if rem > 0: tmp2 = np.zeros((rem,2*fr),np.float32)
        else: tmp2 = 0

        return tmp1, tmp2
            
    def _prep_gpu():
        
        # try loading the necessary gpu libraries
        try:
            import gpu
            import string
            import pyopencl as cl
            import pyopencl.array as cla
            from pyfft.cl import Plan
        except ImportError:
            print "couldnt load gpu libraries, falling back to cpu xpcs"
            _g2_numerator(data,tauvals,batch_size=batch_size,gpu_info=None)
        
        # check gpu_info
        assert gpu.valid, "gpu_info in xpcs.g2 improperly specified"
        context, device, queue, platform = gpu_info
    
        # if the number of frames is not a power of two, find the next power of
        # 2 larger than twice the number of frames.
        p2 = ((fr & (fr - 1)) == 0)
        if p2:     L = int(2*p2)
        if not p2: L = int(2**(math.floor(math.log(2*fr,2))+1))
        
        # build the 1 kernel necessary for the correlation
        kp   = string.join(gpu.__file__.split('/')[:-1],'/')+'/kernels/' # kernel path
        abs1 = gpu.build_kernel_file(context, device, kp+'common_abs_f2_f2.cl')
        abs2 = gpu.build_kernel_file(context, device, kp+'common_abs2_f2_f2.cl')
        
        # if the plan for this size does not exist, create the plan
        global fftplans
        if L not in fftplans.keys():
            fftplan     = Plan((L,),queue=queue)
            fftplans[L] = fftplan
        else:
            fftplan = fftplans[L]
            
        # allocate memory for the correlations
        if batches > 0:
            gpu1 = cla.empty(queue, (batch_size, L), np.complex64)
            tmp1 = np.zeros((batch_size,L),np.complex64)
        else:
            gpu1 = None
            tmp1 = None
            
        if rem > 0:
            gpu2 = cla.empty(queue, (rem, L), np.complex64)
            tmp2 = np.zeros((rem,L),np.complex64)
        else:
            gpu2 = None
            tmp2 = None
            
        return tmp1, tmp2, gpu1, gpu2, L, abs2, abs1, fftplan

    def _calc(data_in,target):
        
        def _gpu_correlation(target,rows):
            fftp.execute(target.data,batch=rows,wait_for_finish=True)
            abs2.execute(gpu_info[2],(int(rows*L),),target.data,target.data)
            fftp.execute(target.data,batch=rows,wait_for_finish=True,inverse=True)
            abs1.execute(gpu_info[2],(int(rows*L),),target.data,target.data)
        
        ds = data_in.shape
        
        if gpu_info == None:
            if ds[0] == batch_size:
                tmp1[:,:fr] = data_in
                run_on      = tmp1
            else:
                tmp2[:,:fr] = data_in
                run_on      = tmp2
            numerator = np.abs(IDFT(np.abs(DFT(run_on,axes=(1,)))**2,axes=(1,)))[:,:fr]
        
        if gpu_info != None:
            
            # move the batch to the gpu
            if ds[0] == batch_size:
                tmp1[:,:fr] = data_in
                target.set(tmp1,queue=gpu_info[2])
            else:
                tmp2[:,:fr] = data_in
                target.set(tmp2,queue=gpu_info[2])
                
            # correlate and remove from gpu
            _gpu_correlation(target,ds[0])
            numerator = (target.get().real)[:,:fr]
            
        return numerator[:,:fr] # not normalized

    if gpu_info != None: batch_size = 1024
    cpu_data     = data.reshape(fr,ys*xs).transpose()
    output       = np.zeros((ys*xs,fr),np.float32)
    batches, rem = (ys*xs)/batch_size, (ys*xs)%batch_size
    
    # make plans, allocate memory, etc
    if gpu_info == None:
        IDFT = np.fft.ifftn
        DFT  = np.fft.fftn
        tmp1, tmp2 = _prep_cpu()
    if gpu_info != None:
        tmp1, tmp2, gpu1, gpu2, L, abs2, abs1, fftp = _prep_gpu()
        
    # run the correlations in batches for improved speed
    for batch in range(batches):
        if gpu_info == None: target = None
        if gpu_info != None: target = gpu1
        g2batch = _calc(cpu_data[batch*batch_size:(batch+1)*batch_size],target)
        output[batch*batch_size:(batch+1)*batch_size] = g2batch.real
        
    if rem > 0:
        if gpu_info == None: target = None
        if gpu_info != None: target = gpu2
        g2batch = _calc(cpu_data[-rem:],target)
        output[-rem:] = g2batch[:rem].real
        
    # normalize, reshape, and return data
    output *= 1./(fr-np.arange(fr))
    return output.transpose().reshape((fr,ys,xs))

# opt_xx_tile are global variables which are set once by _tune_tile_size
opt_fft_tile = None
opt_sm_tile  = None
def _tune_tile_size(data):
    
    global opt_fft_tile
    global opt_sm_tile
    
    import time
    print "tuning"
    
    times     = []
    test_data = np.random.rand(nf,260,260)
    print test_data.shape
    tauvals   = np.arange(nf/2)
    
    if fft and opt_fft_tile != None:
        return opt_fft_tile
    if fft and opt_fft_tile == None:

        tiles = (4,8,16,32,64,128)
        for ts in tiles:
            t0 = time.time()
            test_data = np.random.rand(nf,260,260)
            out = _g2_numerator(test_data,tauvals,tile_size=ts)
            times.append(time.time()-t0)
            
        print times

        #exit()
        #return opt_fft_tile
    
    if not fft: return 128

def g2_symm_borthwick_norm(img, numtau, qAvg = ("circle", 10), fft=False):
    """ calculate correlation function g_2 with Matt Borthwick's symmetric
    normalization. His symmetric normalization is pg #120 of this thesis and
    defined as

    g_2 (q, tau) = < I(q,t) * I(q, t+tau) >_t / (<I(q,t)>_q,left * <I(q,t)>_q,right) * <I(q, t)>_q,t^2/<I(q, t)>_t^2

    where <>_t (<>_q) is averaging over variable t (q). The left (right) average
    time over frames 1:numtau (fr-numtau:fr). This varies slightly from the
    symmetric normalization as it added a term indended to correct for
    long-range trends in intensity.

    arguments:
        img - 3d array
        numtau - tau to correlate
        qAvg - a list of the averaging type and the size. The possible types and
            the size parameter are:
            "square" - the parameter size is dimension of the square, in pixels
            "circle" - the size is the radius of the circle, in pixels
            "gaussian" - size is the FWHM of the gaussian, in pixels

    returns:
        g2 - correlation function for numtau values
    """
    
    import sys
    assert type(qAvg) in (list, tuple) and len(qAvg) == 2, "unknown size for qAvg"
    avgType, avgSize = qAvg
    assert avgType in _averagingFunctions.keys(), "unknown averaging type %s" % avgType

    img, fr, ys, xs = _convert_to_3d(img)

    tauvals = _numtauToTauvals(numtau, maxtau=fr)

    ntaus = len(tauvals)
    result = np.zeros((ntaus, ys, xs), dtype=float)

    IQ = _averagingFunctions[avgType](img, avgSize)
    IQT = _time_average(IQ)
    IT = _time_average(img)
    
    # calculate the numerator
    numerator = _g2_numerator(img,tauvals,fft=fft)

    # normalize the numerator
    IQTsqIQsq = IQT*IQT/(IT*IT)
    for i, tau in enumerate(tauvals):
        result[i] = IQTsqIQsq * numerator[i]/(_time_average(IQ, 0, fr-tau)*_time_average(IQ, tau, fr))
        sys.stdout.write("\rtau %d (%d/%d)" % (tau, i, ntaus))
        sys.stdout.flush()
    print("")
    return result
    
def g2_standard_norm(img, numtau, qAvg = ("circle", 10), fft=False):
    """ calculate correlation function g_2 with a standard normalization. The
    standard normalization is defined as

    g_2 (q, tau) = < I(q,t) * I(q, t+tau) / (<I(q,t)>_q * <I(q,t+tau)>_q) >_t * <I(q,t)>_q,t^2 / <I(q,t)>_t^2

    where <>_t (<>_q) is averaging over variable t (q).

    arguments:
        img - 3d array
        numtau - tau to correlate
        qAvg - a list of the averaging type and the size. The possible types and
            the size parameter are:
            "square" - the parameter size is dimension of the square, in pixels
            "circle" - the size is the radius of the circle, in pixels
            "gaussian" - size is the FWHM of the gaussian, in pixels

    returns:
        g2 - correlation function for numtau values
    """
    assert type(qAvg) in (list, tuple) and len(qAvg) == 2, "unknown size for qAvg"
    avgType, avgSize = qAvg
    assert avgType in _averagingFunctions.keys(), "unknown averaging type %s" % avgType
    assert avgType != "none", "'none' averaging not supported for g2StandardNorm()."

    img, fr, ys, xs = _convert_to_3d(img)

    if ys < avgSize or xs < avgSize:
        print("warning: size for averaging %d is larger than array size, changing to %d" % (avgSize, min(xs,ys)))
        avgSize = min(ys,xs)

    # in standard normalization defined by Borthwick, the img array is modified by the q-averaged intensity then used to calculate g2.
    IT  = _time_average(img)
    IQ  = _averagingFunctions[avgType](img, avgSize)
    IQT = _time_average(IQ)
    
    numerator = _g2_numerator(img/IQ,tauvals,fft=fft)
    
    return numerator*IQT*IQT/(IT*IT)
    
def g2_plain_norm(img, numtau, fft=False): # formerly sujoy norm
    """ calculate correlation function g_2 with a 'plain' normalization. The
    'plain' normalization is defined as

    g_2 (q, tau) = < I(q,t) * I(q, t+tau) >_t / <I(q,t)>_t^2

    where <>_t (<>_q) is averaging over variable t (q).

    arguments:
        img - 3d array
        numtau - tau to correlate

    returns:
        g2 - correlation function for numtau values
    """

    IT = _time_average(img)
    return _g2_numerator(img,tauvals,fft=fft)/(IT*IT)

def g2_no_norm(img, numtau, fft=False):
    """ calculate correlation function g_2 numerator without normalization. The
    no normalization defined as

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

    tauvals = _numtauToTauvals(numtau, maxtau=fr)

    ntaus = len(tauvals)
    numerator = np.zeros((len(tauvals), ys, xs), dtype=float)
    
    numerator = _g2_numerator(img,tauvals,fft=fft)
    
    sys.stdout.write("\rtau %d (%d/%d)" % (tau, i, ntaus))
    sys.stdout.flush()
    for i, tau in enumerate(tauvals):
        numerator[i] = g2_numerator(img, tau)
        sys.stdout.write("\rtau %d (%d/%d)" % (tau, i, ntaus))
        sys.stdout.flush()
    print("")
    return numerator

def g2_numerator_fft(img, taus=None):
    """ Calculates <I(t) I(t + tau)>_t for specified values of tau via fft
    autocorrelation method. For large N this may be significantly faster than
    the g2_numerator function, but for small N or a small number of tau the
    g2_numerator function may be faster.
    
    If data sets are very large this function may lead to malloc errors,
    necessitating fallback to g2_numerator or more sophisticated handling of the
    dataset.
    
    arguments:
        img - data to caclculate g2.  Must be 3d.
        taus - iterable set of tau values where you want g2.
            All taus are evaluated by the fft so a limited set of taus does not
            provide a speed up.

    returns:
        numerator - g2 numerator calculated for requested tau values.
    """
    assert img.ndim == 3, "g2_numerator: Must be a three dimensional image."
    (fr, ys, xs) = img.shape
    
    IDFT = np.fft.ifftn
    DFT = np.fft.fftn

    # fft is cyclic so the data must be zero-padded to prevent the data
    # from wrapping around inappropriately.
    img2       = np.zeros((2*fr,ys,xs),float)
    img2[0:fr] = img
    numerator  = abs(IDFT(abs(DFT(img2,axes=(0,)))**2,axes=(0,)))[0:fr]
    for f in range(fr): numerator[f] *= 1./(fr-f)
    
    if taus != None: return numerator[taus]
    else: return numerator

def _numtauToTauvals(numtau, maxtau=0):
    """ program to convert numtau to a list of values to iterate over.

    arguments:
        numtau - tau's to correlate.  If it is a single number, it generates a
            list from (0, numtau), if it's a list or tuple of size 2, it will do
            the range between those two taus.
        maxtau - Maximum possible tau value.  If it exists, then the function
            checks to see if all taus are smaller than this value.

    returns:
        tauvals - list of taus to correlate.
    """
    assert type(numtau) in (int, list, tuple), "_numtauToTauvals(): numtau is not recognizeable."
    assert type(maxtau) == int and maxtau >= 0, "_numtauToTauvals(): maxtau must be non-negative int."

    if type(numtau) == int:
        tauvals = range(numtau)
    elif type(numtau) in (list, tuple) and len(numtau) == 2:
        tst, tend = numtau
        assert type(tst) == int and tst >= 0, "_numtauToTauvals(): t_start must be non-negative int."
        assert type(tend) == int and tend >= 0, "_numtauToTauvals(): t_end must be non-negative int."
        
        tauvals = range(min(tst, tend), max(tst, tend))
    else:
        # We must have a list of tauvals
        tauvals = list(numtau)

    tarray = np.array(tauvals)
    if (tarray < 0).any():
        print "_numtauToTauvals(): Found values < 0. Removing"
        tarray = tarray[tarray < 0]
    if maxtau > 0:
        if tarray.max() > maxtau:
            print "_numtauToTauvals(): Found values > maxtau (%d). Removing" % maxtau
            tarray = tarray[tarray <= maxtau]

    return list(tarray)

def _noAverage(img, size):
    return img

def _time_average(img, start='', stop=''):
    if start == '' and stop == '':
        return np.average(img,axis=0)
    else:
        intlist = (int, np.int, np.int8, np.int16, np.int32, np.int64)
        assert type(start) in intlist and type(stop) in intlist, "start and stop must be integers. Got start (%s), stop (%s)" % (type(start), type(stop))
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
        data - data to bin.  This is an (N, 4) array where N is the number of
            elements.
        frameTime - Amout of time between bins.  Measured in seconds
        xybin - amount to bin in x and y.  defaults to 8, which is a 512x512
            output.
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

    if binnedData.sum() != datalen:
        print "warning: # of binned data photons (%d) do not match original dataset (%d)" % (binnedData.sum(), datalen)

    return binnedData

def sp_bin_by_time(data, frameTime, counterTime=40.0e-9):
    """Takes a stream of photon incidence times and bins the data in time.

    arguments:
        data - Data to bin.  This is a 1 dimensional array of incidence times.
        frameTime - Amount of time for each bin.  Measured in seconds
        counterTime - Clock time of the camera. Anton Tremsin says its 40 ns.

    returns:
        binnedData - a 3-dimension array (frames, y, x) of the binned data.
        binEdges - edges of the bins.
    """
    assert isinstance(data, np.ndarray), "Data must be an ndarray."
    assert data.ndim == 1, "Data must be 1-dimensional."
    assert np.isreal(frameTime), "frameTime (%s) must be real" % repr(frameTime)
    assert np.isreal(counterTime), "counterTime (%s) must be real" % repr(counterTime)

    sorteddata = data[data.argsort()]

    firsttime = sorteddata.min()
    lasttime = sorteddata.max()
    nbins = int(np.ceil(float(lasttime - firsttime)*counterTime/frameTime))
    return np.histogram(sorteddata, nbins, (0, lasttime))

def sp_sum_bin_all(data, xybin=4):
    """ Sum and bin all of the data from the single photon detector into one
    frame.

    arguments:
        data - a (N, 4) sized array of the data generated by the single photon
            detector.

    returns:
        img - a 2d image of the entire dataset.
    """
    assert isinstance(data, np.ndarray), "Data must be an ndarray."
    assert data.shape[1] == 4, "Second dimension must be 4."
    assert type(xybin) == int, "xybin must be integer."

    # This algorithm is significantly faster than calling sp_bin_by_time() with large times.
    xybins = SP_MAX_SIZE // xybin
    binned = np.zeros((xybins, xybins))

    for x, y, g, t in data:
        binned[y//xybin,x//xybin] += 1

    return binned

def intersect(a, b, assume_unique=False):
    """ implements the matlab intersect() function. Details of the function are
    here: http://www.mathworks.com/help/techdoc/ref/intersect.html

    arguments:
        a - 1st array to test
        b - 2nd array to test
        assume_unique - If True, assumes that both input arrays are unique,
            which speeds up the calculation.  The default is False.

    returns:
        res - values common to both a and b.
        ia - indicies of a such that res = a[ia]
        ib - indicies of b such that res = b[ib]
    """
    res = np.intersect1d(a, b, assume_unique=assume_unique)
    ai = np.nonzero(np.in1d(a, res, assume_unique=assume_unique))
    bi = np.nonzero(np.in1d(b, res, assume_unique=assume_unique))
    return res, ai, bi

def sp_autocorrelation_range(data, xr, yr, p=30, m=2):
    """Implements the time-to-tag correlation algorithm outlined by Wahl et al.
        in Opt. Exp. 11 3583. A (Nx4) array must be provided so that the program
        can figure out what photons occur within the xr and yr.

    arguments:
        data - a 4 col, N row list of photon incidence times and locations.
        xr - range in x.  Must be a 2-tuple or list.
        xr - range in y.  Must be a 2-tuple or list.
        p - number of linear points. Defaults to 30.
        m - factor that the time is changed for each correlator. Defaults to 2.    
            The number of correlators is approx. log(max(t_delta)/p)/log(m) + 1.

    returns:
        corr - A (1 x bins) autocorrelation of data.
        corrtime - A (1 x bins ) list of time ranges.        
    """
    return sp_autocorrelation(sp_select_ROI(data, xr, yr)[:,3], p, m)

def sp_select_ROI(data, xr, yr):
    """ Selects a region defined by xr = (xmin, xmax), yr = (ymin, ymax) and
        returns the data that is between these regions. A (Nx4) array must be
        provided so that the program can figure out what photons occur within
        the xr and yr.

        arguemnts:
            data - a 4 col, N row list of photon incidence times and locations.
            xr - tuple of (xmin, xmax)
            yr - tuple of (ymin, ymax)

        returns:
            subset of data that is within xr and yr.
    """
    assert isinstance(data, np.ndarray), "Data must be an ndarray."
    assert data.shape[1] == 4, "Second dimension must be 4."
    assert type(xr) in (list, tuple) and len(xr) == 2, "xr must be len 2 tuple/list."
    assert type(yr) in (list, tuple) and len(yr) == 2, "yr must be len 2 tuple/list."
    assert type(xr[0]) == int and type(xr[1]) == int, "xr must be integers."
    assert type(yr[0]) == int and type(yr[1]) == int, "yr must be integers."

    reg = (xr[0] <= data[:,0]) & (data[:,0] <= xr[1] ) & (yr[0] <= data[:,1]) & (data[:,1] <= yr[1])

    idx = [i for i in range(len(reg)) if reg[i]]
    print("sp_select_ROI: found %d elements" % len(idx))
    return data[idx, :]

def sp_autocorrelation(data, p=30, m=2, removeZeros=False):
    """Implements the time-to-tag correlation algorithm outlined by Wahl et al.
    in Opt. Exp. 11 3583

    arguments:
        data - A sorted (1xN) array of incidence times.
        p - number of linear points. Defaults to 30.
        m - factor that the time is changed for each correlator. Defaults to 2.
            The number of correlators is approx. log(max(t_delta)/p)/log(m) + 1.
        removeZeros - Remove zeros from the correlation function. Zeros in the
            correlation function mean that there aren't any photon pairs with a
            given delay time and don't make any physical sense. Defaults to
            False (leave them in the result).

    returns:
        corr - A (1 x bins) autocorrelation of data.
        corrtime - A (1 x bins ) list of time ranges.
    """
    return sp_crosscorrelation(data, data, p, m, removeZeros)

def sp_crosscorrelation(d1, d2, p=30, m=2, removeZeros=False):
    """Implements the time-to-tag correlation algorithm outlined by Wahl et al.
    in Opt. Exp. 11 3583

    arguments:
        d1 - A sorted (1xN) array of incidence times for the 1st signal.
        d2 - A sorted (1xN) array of incidence times for the 2nd signal.
        p - number of linear points. Defaults to 30.
        m - factor that the time is changed for each correlator. Defaults to 2.
            The number of correlators is approx. log(max(t_delta)/p)/log(m) + 1.
        removeZeros - Remove zeros from the correlation function. Zeros in the
            correlation function mean that there aren't any photon pairs with a
            given delay time and don't make any physical sense. Defaults to
            False (leave them in the result).

    returns:
        corr - A (1 x bins) crosscorrelation between (d1, d2).
        corrtime - A (1 x bins ) list of time ranges.
    """
    import sys
    assert isinstance(d1, np.ndarray) and isinstance(d2, np.ndarray), "d1 and d2 must 1-dim be arrays"
    assert d1.ndim == d2.ndim == 1, "d1 and d2 must be the same dimension"
    assert len(d1) == len(set(d1)), "d1 has duplicate entries"
    assert len(d2) == len(set(d2)), "d2 has duplicate entries"
    assert isinstance(removeZeros, (bool, int)), "removeZeros must be boolean or int"

    sortd1 = d1[d1.argsort()].astype('int64')
    sortd2 = d2[d2.argsort()].astype('int64')

    tmax = max(sortd1.max() - sortd1.min(), sortd2.max() - sortd2.min())
    # Ncorr does always gives 1 if sortd1 and sortd2 are incidence times (in s) rather than clock counters. Need to fix.
    Ncorr = np.ceil(np.log(tmax/float(p) + 1)/np.log(float(m)) - 1).astype('int') + 1

    corrtime = np.zeros(Ncorr*p)
    corr = np.zeros_like(corrtime)
    w1 = np.ones_like(sortd1)
    w2 = np.ones_like(sortd2)

    delta = 1
    shift = 0
    for i in range(Ncorr):
        sys.stdout.write("\rcorrelator %d/%d" % (i+1, Ncorr))
        sys.stdout.flush()
        for j in range(p):
            shift = shift + delta
            lag = np.floor(shift/delta).astype('int')
            (isect, ai, bi) = intersect(sortd1, sortd2 + lag, assume_unique=True)
            corr[i*p+j] = np.dot(w1[ai], w2[bi]) / float(delta)
            corrtime[i*p+j] = shift
            
        delta = delta*m
        
        sortd1, w1 = _half_data(sortd1, m, w1)
        sortd2, w2 = _half_data(sortd2, m, w2)

    print("")

    # Normalize. Another part of normalization is dividing by corr[i*p+j]/delta (done above)
    for i in range(Ncorr*p):
        corr[i] = corr[i]*tmax/(tmax - corrtime[i])

    if removeZeros:
        val = corr != 0
        return corr[val], corrtime[val]
    else:
        return corr, corrtime

def _sort_data(data, col=3):
    """Helper function to sort data by increasing photon incidence value along
    column col.

    arguments:
        data - data to process.  Must be two dimensional

    returns:
        sorted data by increasing photon incidence value
    """
    assert isinstance(data, np.ndarray) and data.ndim == 2, "data must be a 2d array"
    return data[data[:, col].argsort(),:]

def _list_duplicates(seq):
    """Helper function used by _half_data() to get a list of duplicate entries
    in an iteratable. From http://stackoverflow.com/questions/5419204/index-of-duplicates-items-in-a-python-list .
    """
    from collections import defaultdict

    tally = defaultdict(list)
    for i,item in enumerate(seq):
        tally[item].append(i)
    return ((key,locs) for key,locs in tally.items() if len(locs)>1)

def _half_data(data, m, w):
    """Helper function that halves the data by dividing the incidence times in
    data by m and finding duplicates.  The w matrix is augmented depending on
    the number of duplicates found.  The algorithm is found in Opt. Exp. 11 
    3583.
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
