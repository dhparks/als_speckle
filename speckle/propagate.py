""" A library for simulating the near-field propagation of a complex wavefield.

Author: Daniel Parks (dhparks@lbl.gov)"""

import numpy as np

from . import shape, scattering

try:
    import numexpr
    HAVE_NUMEXPR = True
except ImportError:
    HAVE_NUMEXPR = False
    
try:
    import pyfftw
    DFT = pyfftw.interfaces.numpy_fft.fft2
    IDFT = pyfftw.interfaces.numpy_fft.ifft2
    HAVE_FFTW = True
except ImportError:
    DFT = np.fft.fft2
    IDFT = np.fft.ifft2
    HAVE_FFTW = False

def propagate_one_distance(data, energy_or_wavelength=None, z=None,
                           pixel_pitch=None, phase=None, data_is_fourier=False,
                           band_limit=False):
    
    """ Propagate a wavefield a single distance, by supplying either the energy
    (or wavelength) and the distance or by supplying a pre-calculated quadratic
    phase factor.
    
    Required input:
        data: 2d array (nominally complex, but real valued is ok) describing
            the wavefield.
        
    Optional input:
        energy_or_wavelength: the energy or wavelength of the wavefield.
            If > 1, assumed to be energy in eV. If < 1, assumed to be wavelength
            in meters. Ignored if a quadratic phase factor is supplied.
        
        z: the distance to propagate the wavefield, in meters. Ignored if a
            quadratic phase factor is supplied.
        
        pixel_pitch: the size of the pixels in real space, in meters.
            Ignored if a quadratic phase factor is supplied.
        
        phase: a precalculated quadratic phase factor containing the wavelength,
            distance, etc information already. If supplied, any optional
            arguments to energy_or_wavelength, z, or pixel_pitch are ignored.
            
        data_is_fourier: set to True if the wavefield data has already been
            fourier transformed.
        
    returns: a complex array representing the propagated wavefield.
    """
    
    def _check_types(data, energy_or_wavelength, z, pixel_pitch, phase,
                     data_is_fourier, band_limit):
        
        """ Check types """
        
        assert isinstance(data, np.ndarray), \
        "data must be an array; is %s"%type(data)
        
        assert data.shape[0] == data.shape[1], \
        "data must be square"
        
        data = data.astype(np.complex64)
        
        assert data.ndim == 2, \
        "supplied wavefront data must be 2-dimensional"
        
        assert isinstance(data_is_fourier, bool), \
        "data_is_fourier must be bool"
        
        assert isinstance(phase, (np.ndarray, type(None))), \
        "phase must be array or None"
        
        assert isinstance(band_limit, bool), \
        "band_limit must be bool"

        if phase == None:
            assert isinstance(energy_or_wavelength, (int, float, type(None))), \
            "energy/wavelength must be float or int"
            
            assert isinstance(pixel_pitch, float), \
            "pixel_pitch must be a float saying how big each pixel is in meters"
            
            assert isinstance(z, float), \
            "z must be a float giving how far to propagate in meters"
            
        if phase != None:
            assert isinstance(phase, np.ndarray), \
            "phase must be an array"
            
            assert phase.shape == data.shape, \
            "phase and data must be same shape"
            
    # check types
    _check_types(data, energy_or_wavelength, z, pixel_pitch, phase,
                 data_is_fourier, band_limit)

    # first see if a phase is supplied. if not, make it from the
    # supplied parameters.
    if phase == None:
    
        # convert energy_or_wavelength to wavelength.
        # If < 1 assume it's a wavelength.
        if energy_or_wavelength < 1:
            wavelength = energy_or_wavelength
        else:
            wavelength = scattering.energy_to_wavelength(energy_or_wavelength)*1e-10
    
        # make the phase factor
        N = len(data)
        r = np.fft.fftshift((shape.radial((N, N)))**2)
        phase = np.exp(-1j*np.pi*wavelength*z*r/(pixel_pitch*N)**2)
        
        # calculate the upper limit (nyquist limit on phase factor)
        upper_limit = N*pixel_pitch**2/wavelength
 
    else:
        # phase has been supplied, so check its types for correctness.
        # if phase-generating parameters are supplied they are ignored.
        upper_limit = -1
        
    if data_is_fourier:
        res = data
    if not data_is_fourier:
        res = np.fft.fft2(data)
        
    if z > upper_limit and z > 0 and band_limit:
        res *= np.fft.fftshift(shape.circle((N, N), N/2*upper_limit/z))

    return np.fft.ifft2(res*phase)

def propagate_distances(data, distances, energy_or_wavelength, pixel_pitch,
                        subregion=None, silent=True, band_limit=False,
                        gpu_info=None, im_convert=False):
    
    """ Propagates a complex-valued wavefield through a range of distances
    using the CPU. GPU acceleration is available if properly formed gpu_info is
    supplied, as is returned by gpu.init().
    
    Required input:
        data -- a square numpy array, either float32 or complex64, to propagate
        distances -- an iterable set of distances (in meters!) to propagate
        energy_or_wavelength -- the energy (in eV) or wavelength (in meters) of
            the wavefield. If < 1, assume wavelength is specified.
        pixel_pitch -- the size (in meters) of each pixel in data.
    
    Optional input:
        subregion -- because the number of files can be large, a subregion can
            be specified. If a single number is given, a the subregion will be
            a square with sides of this length co-centered with data.
            If a 2-tuple, the square becomes a rectangle (rows, columns).
            If a 4-tuple, the subregion is [subregion[0]:subregion[1],
            subregion[2]:subregion[3]], so the subregion is NOT cocentered with
            the data.
        silent -- if False, report what distance is currently being calculated.
        gpu_info -- what gets returned by speckle.gpu.init(). If None, the
            calculations are performed on the CPU.
        im_convert -- if True, will convert each of the propagated frames
            to hsv color space and then into PIL objects. If gpu_info is
            supplied, the complex -> hsv -> rgb conversion
    
    Returns:
    
        if im_convert == False:
            a complex-valued 3d ndarray with shape:
            (len(distances),(rows),(cols)); (rows) and (cols) varies with
                the subregion argument.
        
            If the subregion argument wasn't used the output shape is:
            (len(distances),len(data),len(data))
            
        if im_convert == True:
        
            (propagated_array, propagated_pil_images)
            
            propagated_array is the same as in the im_convert=False case.
            propagated_pil_images is a list of images whose shapes are
            the same as the frame in propagated_array.
    """
    
    import sys
    
    iterable_types = (tuple, list, np.ndarray)
    coord_types = (int, float, np.int32, np.float32, np.float64)
    
    def _check_types():
        
        # check types and defaults
        assert isinstance(data, np.ndarray),\
        "data must be an array"
        
        assert data.ndim == 2,\
        "supplied wavefront data must be 2-dimensional"
        
        assert data.shape[0] == data.shape[1],\
        "data must be square"
        
        assert isinstance(distances, iterable_types),\
        "distances must be an iterable set (list or ndarray) \
        of distances given in meters"
        
        # convert to correct datatypes
        d = data.astype(np.complex64)
        N = data.shape[0]
        pp = np.float32(pixel_pitch)
        ew = np.float32(energy_or_wavelength)
        nf = len(distances)
        
        return d, N, pp, ew, nf
        
    def _check_subregion():
        """ Do checking on subregion specification. Subregion can be any of
        the following:
        1. an integer, in which case the subregion is a square cocentered with
            data
        2. a 2 element iterable, in which case the subregion is a rectangle
            cocentered with data
        3. a 4 element iterable, in which case the subregion is a rectangle
            specified by (rmin, rmax, cmin, cmax) """
        
        sr = subregion

        if sr == None:
            sr = N
        assert isinstance(sr, coord_types) or isinstance(sr, iterable_types),\
        "sr type must be integer or iterable"
        
        # return a box with L = sr co-centered with supplied data
        if isinstance(sr, coord_types):
            sr = (int(sr),)
        
        # if we're at this point, sr must be iterable. check its
        # spec then continue
        assert len(sr) in (1, 2, 4), \
        "sr length must be 1, 2, or 4; is %s"%len(sr)
        
        for x in sr:
            assert isinstance(x, coord_types) and x < data.shape[0] and x >= 0,\
            "malformed sr %s"%sr
            
        if len(sr) == 1:
            x = int(sr[0])/2
            h1 = N/2-x
            h2 = N/2+x
            w1 = N/2-x
            w2 = N/2+x

        if len(sr) == 2:
            x, y = int(sr[0])/2, int(sr[1])/2
            h1 = N/2-x
            h2 = N/2+x
            w1 = N/2-y
            w2 = N/2+y
            
        if len(sr) == 4:
            s1, s2, s3, s4 = int(sr[0]), int(sr[1]), int(sr[2]), int(sr[3])
            h1 = min([s1, s2])
            h2 = max([s1, s2])
            w1 = min([s3, s4])
            w2 = max([s3, s4])
            
        return (h1, h2, w1, w2), h2-h1, w2-w1  

    def _prep_gpu():
        """ Set up GPU calculation dependencies """

        # try to import the necessary libraries
        fallback = False
        try:
            import gpu
            import string
            import pyopencl as cl
            import pyopencl.array as cla
            from pyfft.cl import Plan
        except ImportError:
            fallback = True
            
        # check gpu_info
        try:
            assert gpu.valid(gpu_info),\
            "gpu_info in propagate_distances improperly specified"
            
            context, device, queue, platform = gpu_info
        except AssertionError:
            fallback = True
            
        if fallback:
            propagate_distances(data, distances, energy_or_wavelength,
                                pixel_pitch, subregion=subregion,
                                silent=silent, band_limit=band_limit,
                                gpu_info=None, im_convert=im_convert)
    
        # if everything is OK, allocate memory and build kernels
        kp = string.join(gpu.__file__.split('/')[:-1], '/')+'/kernels/'
        build = _build_helper(context, device, kp)
        phase_multiply = build('propagate_phase_multiply.cl')
        copy_to_buffer = build('propagate_copy_to_save_buffer.cl')
        fftplan = Plan((N, N), queue=queue)

        # put the signals onto the gpu along with buffers for the
        # various operations
        rarray = cla.to_device(queue, r.astype(np.float32))
        fourier = cla.to_device(queue, data.astype(np.complex64))
        phase = cla.empty(queue, (N, N), np.complex64)
        back = cla.empty(queue, (N, N), np.complex64)
        store = cla.empty(queue, (nf, rows, cols), np.complex64)
        
        # precompute the fourier transform of data. 
        fftplan.execute(fourier.data, wait_for_finish=True)

        return phase_multiply, copy_to_buffer, fftplan, rarray, fourier,\
               phase, back, store, build
        
    def _build_helper(con, dev, path):
        
        def _hidden(name):
            import gpu
            return gpu.build_kernel_file(con, dev, path+name)
        
        return _hidden
        
    def _prep_cpu():
        
        """ Set up CPU calculation dependencies """
        
        try:
            store = np.zeros((nf, rows, cols), np.complex64)
        except MemoryError:
            print "save buffer was too large. either use a smaller \
            set of distances or a smaller subregion"
            
        f = np.fft.fft2(data)
        return f, store
    
    def _calc(n, z):
        
        """ Calculate the transform for a single distance """
        
        if gpu_info == None:
            
            phase_z = _cpu_phase(z)
            back = IDFT(f*phase_z)
            store[n] = back[sr[0]:sr[1], sr[2]:sr[3]]
            
        if gpu_info != None:
            
            # make the phase factor; multiply the fourier rep; inverse transform
            k_pm.execute(gpu_info[2], (int(N*N),), None, gpu_r.data,
                         np.float32(t*z), gpu_f.data, gpu_phase.data)
            fftplan.execute(data_in=gpu_phase.data, data_out=gpu_back.data,
                            inverse=True, wait_for_finish=True)

            # slice the subregion from the back-propagation and save in store
            k_ctb.execute(gpu_info[2], (rows, cols), None, gpu_store.data,
                          gpu_back.data, np.int32(n), np.int32(N),
                          np.int32(sr[0]), np.int32(sr[2]))
        
    def _im_convert(norm='local'):
        
        """ Image conversion from np array to hsv PIL """

        assert norm in ('global', 'local')

        if gpu_info != None:
            
            context, device, queue, platform = gpu_info
            import gpu, string
            import pyopencl.array as cla
                
            # build the additional kernels
            hsv_convert = build('complex_to_rgb.cl')
            cl_abs = build('common_abs_f2_f.cl')
            
            # allocate new memory (frames, rows, columns, 3); uchar -> uint8?
            new_shape = gpu_store.shape+(3,)
            image_buffer = cla.empty(queue, new_shape, np.uint8)
            debug_buffer = cla.empty(queue, gpu_store.shape, np.float32)
            
            # each frame must be normalized by SOMETHING. if norm=='local',
            # that something is the max value of the abs of each frame. if
            # norm == 'global', that something is the max value of the entire
            # array. in either case, we make a list of maxvals which will
            # correspond to each frame so that we can use the same hsv
            # conversion kernel.
            
            if norm == 'global':
                # calculate the max for the entire array
                abs_store = cla.empty(queue, gpu_store.shape, np.float32)
                mv_gpu = cla.empty(queue, (gpu_store.shape[0],), np.float32)
                cl_abs.execute(queue, (gpu_store.size,), None, gpu_store.data,
                               abs_store.data)
                mv_gpu.fill(np.float32(cla.max(abs_store).get()))

            if norm == 'local':
                # calculate the max for one frame at a time. pull this value
                # and save it in the max_vals list. when finished, move
                # max_vals to the gpu.
                nz = len(distances)
                max_vals = np.zeros(nz, np.float32)
                slice_frame = build('common_slice_frame_f2.cl')
                slice_store = cla.empty(queue, (rows, cols), np.complex64)
                abs_store = cla.empty(queue, (rows, cols), np.float32)
                for n in xrange(nz):
                    slice_frame.execute(queue, (rows, cols), None,
                                        slice_store.data, gpu_store.data,
                                        np.int32(n))
                    cl_abs.execute(queue, (slice_store.size,), None,
                                   slice_store.data, abs_store.data)
                    max_vals[n] = np.float32(cla.max(abs_store).get())
                mv_gpu = cla.to_device(queue, max_vals.astype(np.float32))

            # run the complex -> hsv -> rgb converter on each pixel
            hsv_convert.execute(queue, gpu_store.shape, None, gpu_store.data,
                                image_buffer.data, mv_gpu.data).wait()

            # now convert to pil objects
            import scipy.misc as smp
            image_buffer = image_buffer.get()
            print image_buffer.shape
            images = [smp.toimage(i) for i in image_buffer]
            return gpu_store.get(), images
        
        if gpu_info == None:
            
            # now convert frames to pil objects
            import scipy.misc as smp
            import io
            images = [smp.toimage(io.complex_hsv_image(x)) for x in store]
            return store, images
       
    def _cpu_phase(z):
        if HAVE_NUMEXPR:
            return numexpr.evaluate('exp(-I*z*t*r)')
        else:
            return np.exp(-I*t*z*r)
        
    # check input types
    data, N, pxl_pitch, e_or_w, nf = _check_types()
    
    # turn the subregion spec into usable coordinates
    sr, rows, cols = _check_subregion()

    # convert energy_or_wavelength to wavelength.
    # If < 1 assume it's a wavelength.
    if e_or_w < 1:
        w = e_or_w
    else:
        w = scattering.energy_to_wavelength(e_or_w)*1e-10
    
    # define some useful common quantities
    I = 1j
    t = np.float32(np.pi*w/((pxl_pitch*N)**2))
    r = np.fft.fftshift((shape.radial((N, N)))**2)
    upperlimit = (pxl_pitch*N)**2/(w*N)

    # depending on the compute device, allocate memory etc
    if gpu_info == None:
        f, store = _prep_cpu()
        if HAVE_FFTW:
            pyfftw.interfaces.cache.enable()
        
    if gpu_info != None:
        k_pm, k_ctb, fftplan, gpu_r, gpu_f, gpu_phase, gpu_back,\
        gpu_store, build = _prep_gpu()
  
    # now compute the propagations
    for n, z in enumerate(distances):
        if z > upperlimit:
            w = "propagation distance (%s) exceeds accuracy upperlimit (%s)"
            print w%(z, upperlimit)
        if not silent:
            sys.stdout.write("\rpropagating: %1.2e m (%02d/%02d)"%(z, n+1, nf))
            sys.stdout.flush()
        _calc(n, z)
        
    if gpu_info == None and HAVE_FFTW:
        pyfftw.interfaces.cache.disable()
        
    # if im_convert, return both array data AND converted images.
    # otherwise, just return the array data
    if im_convert:
        store, images = _im_convert()
        return store, images
    else:
        if gpu_info != None:
            store = gpu_store.get()
        return store

def apodize(data_in, sigma=3, threshold=0.01):
    
    """ Apodizes a 2d array to suppress Fresnel ringing from an aperture
    during back-propagation. The algorithm is straightforward: a binary
    image of the aperture (data_in) is convolved with a gaussian with stdevs
    (sigma, sigma). The convolution is then scaled so that it remains 1 at the
    center of data_in and approaches 0 at the edge.
    
    Required arguments:
        data_in: should be a 2d numpy array of the data to be apodized. If
            it is the raw data and not a binary image of the aperture, the
            optionalargument threshold should also be supplied.
        
    Optional arguments:
        sigma: degree of blurring (pixels)
        threshold: the data must be binarized before the apodizer is
            calculated.  This is done by:
            numpy.where(data_in/data_in.max > threshold, 1, 0)
        
    Returns:
        0. The apodizing filter
        1. The apodized data.
        If data_in is a binarized image of the aperture, 0 and 1 should be
        identical.
    """
    
    def _check_types(data_in, sigma, threshold):
        """ Check types """
        
        # check types
        assert isinstance(data_in, np.ndarray), "data must be an array"
        assert data_in.ndim == 2, "data must be 2d"
        
        try:
            sigma = float(sigma)
        except:
            raise ValueError("couldnt cast sigma to float")
        
        try:
            threshold = float(threshold)
        except:
            raise ValueError("couldnt cast threshold to float")
        
        return sigma, threshold
    
    def convolve(x, y):
        """ Helper for convolution  """
        return IDFT(DFT(x)*DFT(y))
    
    sigma, threshold = _check_types(data_in, sigma, threshold)
    
    ad = np.abs(data_in)
    data2 = np.where(ad/ad.max() > threshold, 1, 0)
    
    import math
    sig2 = math.floor(sigma)+1
    
    # ensure that the array is even
    r0, c0 = data2.shape
    if r0%2 == 1:
        data2 = np.vstack([data2, np.zeros((1, data2.shape[1]), data2.dtype)])
    if c0%2 == 1:
        data2 = np.hstack([data2, np.zeros((data2.shape[0], 1), data2.dtype)])
    
    # now pad the array sufficiently to ensure cyclic boundary conditions
    # don't matter (much)
    r_pad = np.zeros((2*sig2, data2.shape[1]), data2.dtype)
    data2 = np.vstack([data2, r_pad])
    data2 = np.vstack([r_pad, data2])
    c_pad = np.zeros((data2.shape[0], 2*sig2), data2.dtype)
    data2 = np.hstack([data2, c_pad])
    data2 = np.hstack([c_pad, data2])
    
    # convolve
    kernel = np.fft.fftshift(shape.gaussian(data2.shape, (sigma, sigma)))
    convolved = np.abs(convolve(kernel, data2))
    convolved *= data2
    
    # rescale
    min_val = convolved[np.nonzero(convolved)].min()
    convolved -= min_val*data2
    convolved /= convolved.max()
    
    # slice out the original, unpadded region
    convolved = convolved[2*sig2:2*sig2+r0, 2*sig2:2*sig2+c0]

    # return
    return convolved, data_in*convolved
 
def acutance(data, method='sobel', exponent=2, normalized=True, mask=None,
             return_type='acutance'):
    
    """ Calculates the acutance of a back propagated wave field. Right now it
    just does the acutance of the magnitude component.
    
    Required input:
        data -- 2d or 3d ndarray object. if 3d, the acutance of each frame will
            be calculated.
    
    Optional input:
        method -- keyword indicating how to compute the discrete derivative.
            options are 'sobel' and 'roll'. Default is 'sobel'. 'roll'
            estimates the derivative by a shift-subtract in each axis.
        exponent -- raise the modulus of the gradient to this value. Default 2.
        normalized -- normalize all the back propagated signals so they have
            the same amount of power. Default is True.
        mask -- you can supply a binary ndarray as mask so that the calculation
            happens only in a certain  region of space. If no mask is supplied
            the calculation is over all space. Default is None.
        return_type -- (default 'acutance'). If 'acutance', return just the
            integrated acutance. If 'derivative', return an array of spatial
            derivatives (for example, to make into images). If 'all' returns
            acutance, derivatives as a tuple of arrays.
     
    Returns:
        a list of acutance values, one for each frame of the supplied data.
    """
   
    def _check_types(data, method, exponent, normalized, mask, return_type):
        """ Check types""" 
        
        assert isinstance(data, np.ndarray) and data.ndim in (2, 3),\
        "data must be 2d or 3d array"
        
        assert method in ('sobel', 'roll'), \
        "acutance derivative method %s unrecognized"%method
        
        assert normalized in (True, False, 1, 0), \
        "noramlized must be boolean evaluable"
        
        assert isinstance(mask, (type(None), np.ndarray)), \
        "mask must be None or ndarray"
        
        assert return_type in ('acutance', 'derivatives', 'all'), \
        "return_type %s not recognized"%(return_type)
        
        try:
            exponent = float(exponent)
        except:
            raise ValueError("couldnt cast expnt to float in acutance")
        
        return exponent
    
    def _calcD(frame):
        """ Calculate the 2d derivative of the frame """

        if method == 'sobel':
            from scipy.ndimage.filters import sobel
            dx = np.abs(sobel(frame, axis=-1))
            dy = np.abs(sobel(frame, axis=0))
                
        if method == 'roll':
            dx = frame-np.roll(frame, 1, axis=0)
            dy = frame-np.roll(frame, 1, axis=1)
    
        gradient = np.sqrt(dx**2+dy**2)**exponent
        
        return gradient*mask

    exponent = _check_types(data, method, exponent, normalized, mask,
                            return_type)
    
    if data.ndim == 2:
        data.shape = (1, data.shape[0], data.shape[1])
    if mask == None:
        mask = np.ones(data.shape[1:], float)

    data = np.abs(data)

    # first, calculate the derivatives. i tried using multiprocessing.pool
    # on this calculation but it turned out that the multiprocessing
    # overhead and the simplicity of the calculation made the total
    # time about twice as much, even using 4 processors (!).
    derivatives = np.zeros_like(data)
    for j in range(data.shape[0]):
        derivatives[j] = _calcD(data[j])

    # bail out early if possible
    if return_type == 'derivatives':
        return derivatives
    
    # now sum the derivatives to get the acutance
    #acutance = np.array([x[1] for x in output])
    acutance = np.sum(np.sum(derivatives, axis=-1), axis=-1)
    
    # normalize if requested
    if normalized:
        #acutance /= np.array([x[2] for x in output])
        acutance /= np.sum(np.sum(data*mask, axis=-1), axis=-1)

    # final returns
    if return_type == 'acutance':
        return acutance
    if return_type == 'all':
        return acutance, derivatives

