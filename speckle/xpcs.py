"""Library for XPCS and photon correlation experiments.  The library calculates
the photon correlation function for a set of frames represented as a 3d numpy
array.  The library will also calculate the correlation function for a single-
photon timestamping detector.

Authors: Keoki Seu (kaseu@lbl.gov), Daniel Parks (dhparks@lbl.gov)

"""
import numpy as np
from . import averaging
import math
import time

try:
    import pyfftw
    #pyfftw.interfaces.cache.enable()
    HAVE_FFTW = True
except ImportError:
    HAVE_FFTW = False
    
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
    assert dim in (1, 2, 3), "image is not 1d, 2d, or 3d"
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
        print("adjusting pixels by %1.2f so all are > 0" % (1.001*img.min()))
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
            summed intensity (sumI).  Note, sumI is the summed intensity in
            each frame, not the entire 3d image.

    returns:
        img - normalized image.    
    """
    assert type(img) == np.ndarray, "image is not an array"

    if img.ndim != 3:
        return img

    fr = img.shape[0]

    if method == "sumI":
        print("Normalizing each frame by total frame intensity")
        for f in range(fr):
            img[f] = img[f]/img[f].sum()
    else: # assume maxI
        print("Normalizing each frame by maximum frame intensity")
        for f in range(fr):
            img[f] = img[f]/img[f].max()

    return img

def g2(data, numtau=None, norm="plain", qAvg=("circle", 10), gpu_info=None,
       silent=True):
    
    """ Calculate correlation function g_2. A variety of normalizations can be
    selected through the optional kwarg "norm".
    
    arguments:
        data - 3d array. g2 correlates along time axis (axis 0)
        numtau - tau to correlate. If a single number, all frame spacings
            from (0, numtau) will be calculated. If a list or tuple of length
            2, calculate g2 within the range between those two values.
            If not supplied, assume the range is (0, nframes/2).
        norm - the normalization specification. See below for details.
        qAvg - some norms require an average over the q coordinate. This
            requires a 2-tuple of (shape,size). The possible types and the size
            parameter are:
            "square" - the parameter size is dimension of the square, in pixels
            "circle" - the size is the radius of the circle, in pixels
            "gaussian" - size is the FWHM of the gaussian, in pixels
            Internally, this average is calculated as a convolution of data
            with the kernel specified here.
        gpu_info - for fast calculation of the g2 function, a gpu context
            can be passed to this function. This requires the gpu context
            to be created outside of this function's invocation, typically
            though speckle.gpu.init().

    returns:
        g2 - correlation function for numtau values 
    
    The following g_2(q,tau) normalizations (denominators only!) and their
    fomulae are available [<>_t (<>_q) indicates averaging over
    variable t (q)]:
    
        1. "borthwick" from Matthew Borthwick's thesis (pg 120):
        (<I(q,t)>_q,left * <I(q,t)>_q,right) * <I(q, t)>_q,t^2/<I(q, t)>_t^2
    
        2. "none" no normalization is applied. NOT RECOMMENDED.
        1 
    
        3. "plain" this is the canonical g2 function:
        <I(q,t)>_t^2

        4. "standard"
        (<I(q,t)>_q * <I(q,t+tau)>_q) >_t * <I(q,t)>_q,t^2 / <I(q,t)>_t^2
    
        5. "symmetric" The left (right) average time over
        frames 1:ntau (fr-ntau:fr).
        (<I(q,t)>_q,left * <I(q,t)>_q,right)

    """ 
    
    import sys
    
    def _normalizers():
        """ Depending on the normalization prep the normalizing factors"""
        
        IQ, IT, IT2, IQT, IQT2 = None, None, None, None, None
        
        if norm in ("symmetric", "standard", "borthwick"):
            
            # check that the averaging kernel is properly specified
            assert isinstance(qAvg, (list, tuple)) and len(qAvg) == 2,\
            "unknown size for qAvg"
            
            avg_type, avg_size = qAvg
            
            assert avg_type in _averagingFunctions.keys(),\
            "unknown averaging type %s"%avg_type
            
            if ys < avg_size or xs < avg_size:
                w = "warning: size for averaging %d is larger \
                    than array size, changing to %d"
                print(w%(avg_size, min(xs, ys)))
                avg_size = min(ys, xs)
    
            sys.stdout.write("calculating q-average (may take a while)\n")
            IQ = _averagingFunctions[avg_type](data, avg_size)
    
        if norm in ("standard", "plain", "borthwick"):
            IT = _time_average(data)
            IT2 = np.reciprocal(IT*IT)
            
        if norm in ("standard", "borthwick"):
            IQT = _time_average(IQ)
            IQT2 = np.reciprocal(IQT*IQT)
            
        return IQ, IT2, IQT2
    
    # check the norm specification
    if norm == None:
        norm = "plain"
    assert norm in ("borthwick", "none", "plain", "standard", "symmetric")
    
    # prep the data. make 3d and convert numtau to a sequence "tauvals"
    data, fr, ys, xs = _convert_to_3d(data)
    if numtau == None:
        numtau = int(round(fr/2))
    tauvals = _numtauToTauvals(numtau, maxtau=fr)
    
    IQ, IT2, IQT2 = _normalizers()

    if norm == "standard":
        data /= IQ

    # now compute the numerator of the g2 function; autocorrelate along t axis.
    # if the dataset is large, computing the fft in a single pass may be
    # inefficient as it no longer fits in memory. for this reason, the data
    # is broken into 8x8 pixel tiles and correlated in batches.
    if not silent:
        print "g2 numerator"
    numerator = _g2_numerator(data, batch_size=64, gpu_info=gpu_info)[tauvals]

    # normalize the numerator. depending on the norm method, different
    # values are calculated.
    if not silent:
        print "g2 normalizing"
    if norm == "none":
        pass # so just the numerator is returned
    elif norm == "plain":
        numerator *= IT2
    elif norm == "standard":
        numerator *= IQT2*IT2
    elif norm == "symmetric":
        for i, tau in enumerate(tauvals):
            denom = _time_average(IQ, 0, fr-tau)*_time_average(IQ, tau, fr)
            numerator[i] /= denom
            sys.stdout.write("\rtau %d (%d/%d)" % (tau, i, numtau))
            sys.stdout.flush()
        sys.stdout.write('\n')
    elif norm == "borthwick":
        numerator *= IQT2/IT2
        for i, tau in enumerate(tauvals):
            denom = _time_average(IQ, 0, fr-tau)*_time_average(IQ, tau, fr)
            numerator[i] /= denom
            sys.stdout.write("\rtau %d (%d/%d)" % (tau, i, numtau))
            sys.stdout.flush()
        sys.stdout.write('\n')

    return np.nan_to_num(numerator)

def _g2_numerator(data, batch_size=64, gpu_info=None):
    
    """ Heavy-lifting function which calculates the numerator of the g2
    function by FFT method. """
    
    # check incoming
    assert data.ndim == 3, "g2_numerator: Must be a three dimensional image."
    (fr, ys, xs) = data.shape

    def _prep_cpu():
        """ Set up dependencies for CPU calculations """
        cpu_d = {}
        if batches > 0:
            cpu_d['tmp1'] = np.zeros((batch_size, 2*fr), np.float32)
        if rem > 0:
            cpu_d['tmp2'] = np.zeros((rem, 2*fr), np.float32)
        if HAVE_FFTW:
            pyfftw.interfaces.cache.enable()
            cpu_d['DFT'] = pyfftw.interfaces.numpy_fft.fft2
            cpu_d['IDFT'] = pyfftw.interfaces.numpy_fft.ifft2
        else:
            cpu_d['DFT'] = np.fft.ifftn
            cpu_d['IDFT'] = np.fft.fftn
        return cpu_d
            
    def _prep_gpu():
        """ Set up dependencies for GPU calculations """
        
        # try loading the necessary gpu libraries
        try:
            import gpu
            import string
            import pyopencl as cl
            import pyopencl.array as cla
            from pyfft.cl import Plan
        except ImportError:
            print "couldnt load gpu libraries, falling back to cpu xpcs"
            _g2_numerator(data, batch_size=batch_size, gpu_info=None)
            
        gpu_d = {}
        
        # check gpu_info
        assert gpu.valid, "gpu_info in xpcs.g2 improperly specified"
        gpu_d['c'] = gpu_info[0]
        gpu_d['d'] = gpu_info[1]
        gpu_d['q'] = gpu_info[2]
    
        # if the number of frames is not a power of two, find the next power of
        # 2 larger than twice the number of frames.
        p2 = ((fr & (fr - 1)) == 0)
        if p2:
            L = int(2*p2)
        if not p2:
            L = int(2**(math.floor(math.log(2*fr, 2))+1))
        gpu_d['L'] = np.int32(L)
        
        build = lambda f: gpu.build_kernel_file(gpu_d['c'], gpu_d['d'], kp+f)
        
        # build the kernels necessary for the correlation
        kp = string.join(gpu.__file__.split('/')[:-1], '/')+'/kernels/'
        gpu_d['abs1'] = build('common_abs_f2_f2.cl')
        gpu_d['abs2'] = build('common_abs2_f2_f2.cl')
        gpu_d['cpy1'] = build('xpcs_embed_f_f2.cl')
        gpu_d['cpy2'] = build('xpcs_pull_f2_f.cl')
        gpu_d['setz'] = build('common_set_zero_f2.cl')
        gpu_d['norm'] = build('xpcs_fr_norm.cl')
        gpu_d['fftp'] = Plan((L,), queue=gpu_d['q'])
        
        # allocate memory for the correlations
        if batches > 0:
            gpu_d['bf'] = cla.empty(gpu_d['q'], (batch_size, fr), np.float32)
            gpu_d['bc'] = cla.empty(gpu_d['q'], (batch_size, L), np.complex64)
            gpu_d['t1'] = np.zeros((batch_size, fr), np.float32)
            
        if rem > 0:
            gpu_d['rf'] = cla.empty(gpu_d['q'], (rem, fr), np.float32)
            gpu_d['rc'] = cla.empty(gpu_d['q'], (rem, L), np.complex64)
            gpu_d['t2'] = np.zeros((rem, fr), np.float32)
            
        return gpu_d

    def _calc_g2(data_in, flt, cpx):
        
        ds = data_in.shape
        
        def _gpu_set(data_in):
            """ Move data to GPU """
            flt.set(data_in.astype(np.float32), queue=gpu_d['q'])
            
        def _cpu_set(data_in):
            """ Set arrays """
            if ds[0] == batch_size:
                cpu_d['tmp1'][:, :fr] = data_in
                return cpu_d['tmp1']
            else:
                cpu_d['tmp2'][:, :fr] = data_in
                return cpu_d['tmp2']
        
        def _gpu_correlate(flt, cpx, rows):
            """ Do the FFT correlation on the GPU"""
            # six steps: 1. move to cpx after zeroing 2. fft
            # 3. abs**2 4. ifft 5. abs 6. pull data, cast to float
            # in the future, combine abs1 and cpy2. however this
            # will be a very small speedup...
            pxls = int(rows*gpu_d['L'])

            gpuq = gpu_d['q']
            gpul = gpu_d['L']
            gpu_d['setz'].execute(gpuq, (pxls,), None, cpx.data)
            gpu_d['cpy1'].execute(gpuq, (rows, fr), None, flt.data, cpx.data, gpul)
            gpu_d['fftp'].execute(cpx.data, batch=rows)
            gpu_d['abs2'].execute(gpuq, (pxls,), None, cpx.data, cpx.data)
            gpu_d['fftp'].execute(cpx.data, batch=rows, inverse=True)
            gpu_d['abs1'].execute(gpuq, (pxls,), None, cpx.data, cpx.data)
            gpu_d['cpy2'].execute(gpuq, (rows, fr), None, cpx.data, flt.data, gpul)
            f = flt.get()
            
            return f
            
        def _cpu_correlate(run_on):
            """ Do the FFT correlation on the CPU"""
            tmp = cpu_d['DFT'](run_on, axes=(1,))
            tmp = np.abs(tmp)**2
            tmp = cpu_d['IDFT'](tmp, axes=(1,))
            tmp = np.abs(tmp)
            return tmp[:, :fr].real
        
        # move data to pre-allocated memory. correlate.
        if gpu_info == None:
            run_on = _cpu_set(data_in)
            numerator = _cpu_correlate(run_on)
        
        if gpu_info != None:
            _gpu_set(data_in)
            numerator = _gpu_correlate(flt, cpx, ds[0])
            
        return numerator

    if gpu_info != None:
        batch_size = 2048

    cpu_data = np.ascontiguousarray(data.reshape(fr, ys*xs).transpose())
    output = np.zeros((ys*xs, fr), np.float32)
    batches, rem = (ys*xs)/batch_size, (ys*xs)%batch_size
    
    # question for future: if someone passes an enormous dataset (like say 2GB)
    # to this function, how does the above reshaping handle it? When the data
    # is opened, it creates a mem-map... so does the actual float data get
    # loaded to memory in the above or is something gnarly done with the
    # representation returned by pyfits?
    
    # make plans, allocate memory, etc
    if gpu_info == None:
        cpu_d = _prep_cpu()
    if gpu_info != None:
        gpu_d = _prep_gpu()
        
    # run the correlations in batches for improved speed. first, make a list of
    # jobs, then iterate over the jobs list. in theory, this could also be
    # used to dispatch jobs to multiple compute devices. each job is specified
    # by a start row, stop row, and for gpu calculations two memory pointers.
    if gpu_info == None:
        flt, cpx, fltr, cpxr = None, None, None, None
    if gpu_info != None:
        flt, cpx = gpu_d.get('bf'), gpu_d.get('bc')
        fltr, cpxr = gpu_d.get('rf'), gpu_d.get('rc')
    
    jobs = []
    for n in range(batches):
        jobs.append((n*batch_size, (n+1)*batch_size, flt, cpx))
    if  rem > 0:
        jobs.append((-rem, ys*xs, fltr, cpxr))
    
    t0 = time.time()
    for n, job in enumerate(jobs):
        start, stop, flt, cpx = job
        g2batch = _calc_g2(cpu_data[start:stop], flt, cpx)
        output[start:stop] = g2batch
        
    # normalize, reshape, and return data
    output *= 1./(fr-np.arange(fr))
    t1 = time.time()
    
    print "fft executation time %s"%(t1-t0)
    if gpu_info == None and HAVE_FFTW:
        pyfftw.interfaces.cache.disable()
    
    return output.transpose().reshape((fr, ys, xs))

def g2_symm_borthwick_norm(img, numtau, qAvg=("circle", 10), fft=False):
    """
    THIS FUNCTION NOW SIMPLY WRAPS THE MASTER G2.
    KEPT FOR BACKWARD COMPATABILITY

    arguments:
        img - 3d array
        numtau - tau to correlate
        qAvg - a list of the averaging type and the size. The possible types
            and the size parameter are:
            "square" - the parameter size is dimension of the square, in pixels
            "circle" - the size is the radius of the circle, in pixels
            "gaussian" - size is the FWHM of the gaussian, in pixels

    returns:
        g2 - correlation function for numtau values
    """
    
    return g2(img, numtau=numtau, qAvg=qAvg, norm='borthwick')
    
def g2_standard_norm(img, numtau, qAvg=("circle", 10), fft=False):
    
    """
    Wraps g2 with norm='standard'

    arguments:
        img - 3d array
        numtau - tau to correlate
        qAvg - a list of the averaging type and the size. The possible types
            and the size parameter are:
            "square" - the parameter size is dimension of the square, in pixels
            "circle" - the size is the radius of the circle, in pixels
            "gaussian" - size is the FWHM of the gaussian, in pixels

    returns:
        g2 - correlation function for numtau values
    """
    
    return g2(img, numtau=numtau, qAvg=qAvg, norm='standard')

def g2_symm_norm(img, numtau, qAvg=("circle", 10), fft=False):
    """ Wraps g2 with norm='symmetric'

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

    return g2(img, numtau=numtau, qAvg=qAvg, norm='symmetric')
    
def g2_plain_norm(img, numtau, fft=False): # formerly sujoy norm
    """ Now wraps g2 with normalization='plain'

    arguments:
        img - 3d array
        numtau - tau to correlate

    returns:
        g2 - correlation function for numtau values
    """
    return g2(img, numtau=numtau, norm='plain')

def g2_no_norm(img, numtau, fft=False):
    """ Wrap g2() with norm='none' """
    return g2(img, numtau=numtau, norm='none')

def _numtauToTauvals(numtau, maxtau=0):
    """ program to convert numtau to a list of values to iterate over.

    arguments:
        numtau - tau's to correlate.  If it is a single number, it generates a
            list from (0, numtau), if it's a list or tuple of size 2, it will
            do the range between those two taus.
        maxtau - Maximum possible tau value.  If it exists, then the function
            checks to see if all taus are smaller than this value.

    returns:
        tauvals - list of taus to correlate.
    """
    assert type(numtau) in (int, list, tuple), \
    "_numtauToTauvals(): numtau is not recognizeable."
    
    assert type(maxtau) == int and maxtau >= 0, \
    "_numtauToTauvals(): maxtau must be non-negative int."

    if type(numtau) == int:
        tauvals = range(numtau)
    elif type(numtau) in (list, tuple) and len(numtau) == 2:
        tst, tend = numtau
        
        assert type(tst) == int and tst >= 0, \
        "_numtauToTauvals(): t_start must be non-negative int."
        
        assert type(tend) == int and tend >= 0, \
        "_numtauToTauvals(): t_end must be non-negative int."
        
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
            print "_numtauToTauvals(): Found values > maxtau (%d). Removing"%maxtau
            tarray = tarray[tarray <= maxtau]

    return list(tarray)

def _noAverage(img, size):
    return img

def _time_average(img, start='', stop=''):
    if start == '' and stop == '':
        return np.average(img, axis=0)
    else:
        intlist = (int, np.int, np.int8, np.int16, np.int32, np.int64)
        
        assert type(start) in intlist and type(stop) in intlist, \
        "start and stop must be integers. Got start (%s), \
        stop (%s)"%(type(start), type(stop))
        
        return np.average(img[start:stop], axis=0)

# This is a dictionary of averaging functions.
# They all take (img, size) as arguments.
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

def sp_bin_by_space_and_time(data, frame_time, xy_bin=8, counter_time=40.0e-9):
    """Takes a dataset collected by the SSI photon counting fast camera and
    bins the data in time and xy.

    arguments:
        data - data to bin.  This is an (N, 4) array where N is the number of
            elements.
        frame_time - Amout of time between bins.  Measured in seconds
        xy_bin - amount to bin in x and y.  defaults to 8, which is a 512x512
            output.
        counter_time - clock time of the camera. Anton Tremsin says its 40 ns.

    returns:
        binned_data - a 3-dimension array (frames, y, x) of the binned data.
    """
    assert isinstance(data, np.ndarray), "data must be an ndarray"
    assert data.shape[1] == 4, "Second dimension must be 4."
    
    assert np.isreal(frame_time),\
    "frame_time (%s) must be real"%repr(frame_time)
    
    assert np.isreal(counter_time),\
    "counter_time (%s) must be real"%repr(counter_time)
    # TODO: This could be accelerated if written for a GPU

    first_time = data[:, 3].min()
    last_time = data[:, 3].max()
    data_length = len(data[:, 0])
    
    nbins = int(np.ceil(float(last_time - first_time)*counter_time/frame_time))
    xy_binned_dim = int(np.ceil(SP_MAX_SIZE/xy_bin))

    binned_data = np.zeros((nbins, xy_binned_dim, xy_binned_dim), dtype='int')
    bin_edges = np.linspace(first_time, last_time, nbins+1)
    
    # slightly increase the last bin.  For some reason the last bin is slightly
    #too small, and the photons on the edges are not included
    bin_edges[-1] = bin_edges[-1]*1.01

    # If the frame_time is larger than the total acq time,
    # then sum up everything
    if frame_time >= (last_time-first_time)*counter_time:
        histfn = lambda a, b: (len(a), b)
    else:
        # this is the default; bin data using histogram
        histfn = np.histogram

    for i in range(xy_binned_dim):
        x = data[:, 0]
        y = data[:, 1]
        xc = np.argwhere((i*xy_bin <= x) & (x < (i+1)*xy_bin))
        idx_to_delete = np.array([])
        for j in range(xy_binned_dim):
            yc = np.argwhere((j*xy_bin <= y) & (y < (j+1)*xy_bin))
            isect, a, b = intersect(xc, yc, assume_unique=True)
            if len(isect) != 0:
                binned, bin_edges = histfn(data[isect, 3]+0.5, bin_edges)
                binned_data[:, i, j] = binned
            idx_to_delete = np.append(idx_to_delete, isect)
        data = np.delete(data, idx_to_delete, axis=0)

    if binned_data.sum() != data_length:
        print "warning: # of binned data photons (%d) do not match original\
        dataset (%d)" % (binned_data.sum(), data_length)

    return binned_data

def sp_bin_by_time(data, frame_time, counter_time=40.0e-9):
    """Takes a stream of photon incidence times and bins the data in time.

    arguments:
        data - Data to bin.  This is a 1 dimensional array of incidence times.
        frame_time - Amount of time for each bin.  Measured in seconds
        counter_time - Clock time of the camera. Anton Tremsin says its 40 ns.

    returns:
        binned_data - a 3-dimension array (frames, y, x) of the binned data.
        binEdges - edges of the bins.
    """
    assert isinstance(data, np.ndarray), "Data must be an ndarray."
    assert data.ndim == 1, "Data must be 1-dimensional."
    
    assert np.isreal(frame_time), \
    "frame_time (%s) must be real" % repr(frame_time)
    
    assert np.isreal(counter_time), \
    "counter_time (%s) must be real" % repr(counter_time)

    sorted_data = data[data.argsort()]

    first_time = sorted_data.min()
    last_time = sorted_data.max()
    nbins = int(np.ceil(float(last_time - first_time)*counter_time/frame_time))
    return np.histogram(sorted_data, nbins, (0, last_time))

def sp_sum_bin_all(data, xy_bin=4):
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
    assert type(xy_bin) == int, "xy_bin must be integer."

    # This algorithm is significantly faster than calling sp_bin_by_time()
    # with large times.
    xy_bins = SP_MAX_SIZE // xy_bin
    binned = np.zeros((xy_bins, xy_bins))

    for x, y, g, t in data:
        binned[y//xy_bin, x//xy_bin] += 1

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
    return sp_autocorrelation(sp_select_ROI(data, xr, yr)[:, 3], p, m)

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
    
    assert type(xr) in (list, tuple) and len(xr) == 2, \
    "xr must be len 2 tuple/list."
    
    assert type(yr) in (list, tuple) and len(yr) == 2, \
    "yr must be len 2 tuple/list."
    
    assert type(xr[0]) == int and type(xr[1]) == int, "xr must be integers."
    assert type(yr[0]) == int and type(yr[1]) == int, "yr must be integers."

    reg = (xr[0] <= data[:, 0]) & (data[:, 0] <= xr[1])\
    & (yr[0] <= data[:, 1]) & (data[:, 1] <= yr[1])

    idx = [i for i in range(len(reg)) if reg[i]]
    print("sp_select_ROI: found %d elements" % len(idx))
    return data[idx, :]

def sp_autocorrelation(data, p=30, m=2, remove_zeros=False):
    """Implements the time-to-tag correlation algorithm outlined by Wahl et al.
    in Opt. Exp. 11 3583

    arguments:
        data - A sorted (1xN) array of incidence times.
        p - number of linear points. Defaults to 30.
        m - factor that the time is changed for each correlator. Defaults to 2.
            The number of correlators is approx. log(max(t_delta)/p)/log(m) + 1.
        remove_zeros - Remove zeros from the correlation function. Zeros in the
            correlation function mean that there aren't any photon pairs with a
            given delay time and don't make any physical sense. Defaults to
            False (leave them in the result).

    returns:
        corr - A (1 x bins) autocorrelation of data.
        corrtime - A (1 x bins ) list of time ranges.
    """
    return sp_crosscorrelation(data, data, p, m, remove_zeros)

def sp_crosscorrelation(d1, d2, p=30, m=2, remove_zeros=False):
    """Implements the time-to-tag correlation algorithm outlined by Wahl et al.
    in Opt. Exp. 11 3583

    arguments:
        d1 - A sorted (1xN) array of incidence times for the 1st signal.
        d2 - A sorted (1xN) array of incidence times for the 2nd signal.
        p - number of linear points. Defaults to 30.
        m - factor that the time is changed for each correlator. Defaults to 2.
            The number of correlators is approx. log(max(t_delta)/p)/log(m) + 1.
        remove_zeros - Remove zeros from the correlation function. Zeros in the
            correlation function mean that there aren't any photon pairs with a
            given delay time and don't make any physical sense. Defaults to
            False (leave them in the result).

    returns:
        corr - A (1 x bins) crosscorrelation between (d1, d2).
        corrtime - A (1 x bins ) list of time ranges.
    """
    import sys
    
    assert isinstance(d1, np.ndarray) and isinstance(d2, np.ndarray), \
    "d1 and d2 must 1-dim be arrays"
    
    assert d1.ndim == d2.ndim == 1, "d1 and d2 must be the same dimension"
    assert len(d1) == len(set(d1)), "d1 has duplicate entries"
    assert len(d2) == len(set(d2)), "d2 has duplicate entries"
    
    assert isinstance(remove_zeros, (bool, int)),\
    "remove_zeros must be boolean or int"

    sortd1 = d1[d1.argsort()].astype('int64')
    sortd2 = d2[d2.argsort()].astype('int64')

    tmax = max(sortd1.max() - sortd1.min(), sortd2.max() - sortd2.min())
    
    # Ncorr does always gives 1 if sortd1 and sortd2 are incidence
    # times (in s) rather than clock counters. Need to fix.
    Ncorr = np.ceil(np.log(tmax/float(p)+1)/np.log(float(m))-1).astype('int')+1

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
            (isect, ai, bi) = intersect(sortd1, sortd2+lag, assume_unique=True)
            corr[i*p+j] = np.dot(w1[ai], w2[bi]) / float(delta)
            corrtime[i*p+j] = shift
            
        delta = delta*m
        
        sortd1, w1 = _half_data(sortd1, m, w1)
        sortd2, w2 = _half_data(sortd2, m, w2)

    print("")

    # Normalize. Another part of normalization is dividing
    # by corr[i*p+j]/delta (done above)
    for i in range(Ncorr*p):
        corr[i] = corr[i]*tmax/(tmax - corrtime[i])

    if remove_zeros:
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
    assert isinstance(data, np.ndarray) and data.ndim == 2, \
    "data must be a 2d array"
    
    return data[data[:, col].argsort(), :]

def _list_duplicates(seq):
    """Helper function used by _half_data() to get a list of duplicate entries
    in an iteratable.
    From http://stackoverflow.com/questions/5419204/index-of-duplicates-items-in-a-python-list .
    """
    from collections import defaultdict

    tally = defaultdict(list)
    for i, item in enumerate(seq):
        tally[item].append(i)
    return ((key, locs) for key, locs in tally.items() if len(locs) > 1)

def _half_data(data, m, w):
    """Helper function that halves the data by dividing the incidence times in
    data by m and finding duplicates.  The w matrix is augmented depending on
    the number of duplicates found.  The algorithm is found in Opt. Exp. 11 
    3583.
    """
    import sys
    halved_data = np.floor(data/float(m)).astype('int64')
    w_sum_before = w.sum()
    if (halved_data - np.roll(halved_data, 1) == 0).any():
        list_dup_iterator = _list_duplicates(halved_data)
        vals_to_remove = []
        for val, keys in sorted(list_dup_iterator):
            vals_to_remove.append(keys[1:])
            w[keys[0]] += w[keys[1:]].sum()
        w = np.delete(w, vals_to_remove)
        data = np.delete(halved_data, vals_to_remove)
        sys.stdout.write("\r\t\t  | removing %6d duplicates."%len(vals_to_remove))
        sys.stdout.flush()
        if w_sum_before != w.sum():
            print("WARNING: values in weighted array do not match after \
                  eliminating duplicates")
    else:
        data = halved_data

    return data, w
