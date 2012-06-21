import numpy
DFT = numpy.fft.fft2
IDFT = numpy.fft.ifft2
from numpy.fft import fftshift

from . import shape
#import shape
I = complex(0,1)

def propagate_one_distance(data,energy_or_wavelength=None,z=None,pixel_pitch=None,phase=None,data_is_fourier=False):
    """ Propagate a wavefield a single distance, by supplying either the energy (or wavelength) and the distance
    or by supplying a pre-calculated quadratic phase factor.
    
    Required input:
        data: 2d array (nominally complex, but real valued is ok) describing the wavefield.
        
    Optional input:
        energy_or_wavelength: the energy or wavelength of the wavefield. If > 1, assumed to be energy in eV.
            If < 1, assumed to be wavelength in meters. Ignored if a quadratic phase factor is supplied.
        
        z: the distance to propagate the wavefield, in meters. Ignored if a quadratic phase factor is supplied.
        
        pixel_pitch: the size of the pixels in real space, in meters. Ignored if a quadratic phase factor is supplied.
        
        phase: a precalculated quadratic phase factor containing the wavelength, distance, etc information already.
            If supplied, any optional arguments to energy_or_wavelength, z, or pixel_pitch are ignored.
            
        data_is_fourier: set to True if the wavefield data has already been fourier transformed.
        
    returns: a complex array representing the propagated wavefield.
    """
    
    # check requirements regarding types of data
    assert isinstance(data,numpy.ndarray),                "data must be an array"
    assert data.shape[0] == data.shape[1],                "data must be square"
    assert data.dtype in (float, complex),                "data must be float32 or complex64"
    assert data.ndim == 2,                                "supplied wavefront data must be 2-dimensional"
    assert type(data_is_fourier) == bool,                 "data_is_fourier must be bool"
    assert isinstance(phase,(numpy.ndarray,type(None))),  "phase must be array or None"

    # first see if a phase is supplied. if not, make it from the supplied parameters.
    if phase == None:
    
        assert isinstance(energy_or_wavelength, (int,float,type(None))), "energy/wavelength must be float or int"
        assert isinstance(pixel_pitch, float), "pixel_pitch must be a float saying how big each pixel is in meters"
        assert isinstance(z, float), "z must be a float giving how far to propagate in meters"
    
        # convert energy_or_wavelength to wavelength.  If < 1 assume it's a wavelength.
        if energy_or_wavelength < 1: wavelength = energy_or_wavelength
        else: wavelength = scattering.energy_to_wavelength(energy_or_wavelength)*1e-10
    
        N = len(data)
        r = fftshift((shape.radial((N,N)))**2)
        phase = numpy.exp(-I*numpy.pi*wavelength*z*r/(pixel_pitch*N)**2)
        
    else:
        # phase has been supplied, so check its types for correctness. if phase-generating parameters are supplied they are ignored.
        assert isinstance(phase,numpy.ndarray), "phase must be an array"
        assert phase.shape == data.shape,       "phase and data must be same shape"
        
    if not data_is_fourier: data = DFT(data)
    
    return IDFT(data*phase)

def propagate_distance(data,distances,energy_or_wavelength,pixel_pitch,gpuinfo=None,subarraysize=None):
    """ Propagates a complex-valued wavefield through a range of distances, with the ability to use a GPU
    for faster computation.
    
    Required input:
    data: a square numpy.ndarray, either float32 or complex64, to be propagated
    distances: an iterable set of distances (in meters!) over which the propagated field is calculated
    energy_or_wavelength: the energy (in eV) or wavelength (in meters) of the wavefield. It's assumed that if the number is < 1 it is a wavelength.
    pixel_pitch: the size (in meters) of each pixel in data.
    
    Optional input:
    gpuinfo: pass the bundle of information coming from gpu.init() here to use the GPU.
    subarraysize: because the number of files can be large a subarray cocentered with data can be specified.
    
    Returns: a complex-valued 3d ndarray of shape (len(distances),subarraysize,subarraysize). If the subarraysize argument wasn't used the
    size of the output array is (len(distances),len(data),len(data)). returned[n] is the wavefield propagated to distances[n].
    """
    from . import scattering
    #import scattering
   
    # check types and defaults
    assert isinstance(data,numpy.ndarray),                                   "data must be an array"
    assert data.dtype in (float, complex),                                   "data must be float or complex"
    assert data.ndim == 2,                                                   "supplied wavefront data must be 2-dimensional"
    assert data.shape[0] == data.shape[1],                                   "data must be square"
    assert isinstance(energy_or_wavelength, (float, int)),                   "must supply either wavelength in meters or energy in eV"
    assert isinstance(distances, (list, tuple, numpy.ndarray)),              "distances must be an iterable set (list or ndarray) of distances given in meters"
    assert isinstance(pixel_pitch, (int,float)),                             "pixel_pitch must be a float saying how big each pixel is in meters"
    if subarraysize == None: subarraysize = len(data)
    assert isinstance(subarraysize, int) and subarraysize <= len(data),      "subarray must be int smaller than supplied length of data"
    assert isinstance(gpuinfo,(tuple,type(None))),                           "gpuinfo not recognized"
    if isinstance(gpuinfo,tuple):
        usegpu = True
        import gpu
        assert subarraysize%2 == 0, "subarraylength on gpu should be even number"
        context, device, queue = gpuinfo
    if gpuinfo == None: usegpu = False
    
    # convert energy_or_wavelength to wavelength.  If < 1 assume it's a wavelength.
    if energy_or_wavelength < 1: wavelength = energy_or_wavelength
    else: wavelength = scattering.energy_to_wavelength(energy_or_wavelength)*1e-10
        
    N = len(data)
    
    try: buffer = numpy.zeros((len(distances),subarraysize,subarraysize),'complex64')
    except MemoryError:
        print "save buffer was too large. either use a smaller distances or a smaller subarraylength"
   
    # this is the upper limit of the propagation distance. after this point nysquist sampling
    # limits the reliability of the phase component, although the magnitude component is still useful.
    upperlimit = (pixel_pitch*N)**2/(wavelength*N)

    if not usegpu: 
        # compute the distances of back propagations on the cpu.
        # precompute the fourier signal. define the phase as a lambda function. loop through the distances
        # calling phase and propagate_one_distance. save the region of interest (subarray) to the buffer.
        fourier = DFT(data)
        r = fftshift((shape.radial((N,N)))**2)
        cpu_phase = lambda z: numpy.exp(-I*numpy.pi*wavelength*z*r/(pixel_pitch*N)**2)
        for n,z in enumerate(distances):
            phase_z = cpu_phase(z)
            back = propagate_one_distance(fourier,phase=phase_z,data_is_fourier=True)
            buffer[n] = back[N/2-subarraysize/2:N/2+subarraysize/2,N/2-subarraysize/2:N/2+subarraysize/2]

    if usegpu:
        
        import pyopencl as cl
        import pyopencl.array as cl_array
        import kernels
        from pyopencl.elementwise import ElementwiseKernel
        from pyfft.cl import Plan
        
        # make the fft plan for NxN interleaved (ie, complex64) data
        print N
        fftplan = Plan((N,N),queue=queue)
        print "here"

        # make the kernel which computes the phase factor
        t = numpy.float32(numpy.pi*wavelength/((pixel_pitch*N)**2))
        phase_factor = ElementwiseKernel(context,
            "float *r, "   # r-array
            "float t, "    # combination of factors (pi,lambda,pitch,etc)
            "float z, "    # z, the dependent variable: DISTANCE
            "float2 *out", # output is complex type so float2
            "out[i] = (float2)(native_cos(t*z*r[i]),native_sin(-1*t*z*r[i]))",
            "phase_factor")
        
        complex_multiply = ElementwiseKernel(context,
            "float2 *a, " # in1
            "float2 *b, " # in2
            "float2 *c",  # product
            "c[i] = (float2)(a[i].x*b[i].x-a[i].y*b[i].y, a[i].x*b[i].y+a[i].y*b[i].x)",
            "mult_buffs")
        
        # build the copy kernel from the kernels file
        copy_to_buffer = gpu.build_kernel(context, device, kernels.copy_to_buffer)
        print "done with kernels"
        
        # precompute the fourier signals
        r = fftshift((shape.radial((N,N)))**2)
        f = DFT(data)
        
        # put the signals onto the gpu along with buffers for the various operations
        L = subarraysize
        gpu_rarray  = cl_array.to_device(queue,r.astype(numpy.float32))   # r array
        gpu_fourier = cl_array.to_device(queue,f.astype(numpy.complex64)) # fourier data
        gpu_phase   = cl_array.empty(queue,(N,N),numpy.complex64)         # buffer for phase factor and phase-fourier product
        gpu_back    = cl_array.empty(queue,(N,N),numpy.complex64)         # buffer for propagated wavefield, +z direction
        try:
            gpu_buffer  = cl_array.empty(queue,(len(distances),L,L),numpy.complex64) # allocate propagation buffer
        except:
            print "probably ran out of memory. make subarraysize smaller or have fewer points in the distances"
            exit()

        # now compute the propagations
        for n,z in enumerate(distances):
            # compute on gpu and transfer to buffer.
            phase_factor(gpu_rarray,t,numpy.float32(z),gpu_phase)    # form the phase factor
            complex_multiply(gpu_phase,gpu_fourier,gpu_phase)        # form the phase-data product. store in gpu_phase
            fftplan.execute(gpu_phase.data,data_out=gpu_back.data,   # inverse transform
                            inverse=True,wait_for_finish=True)

            copy_to_buffer.execute(queue,(L,L),                                     # opencl stuff
                                   gpu_buffer.data, gpu_back.data,                  # destination, source
                                   numpy.int32(n), numpy.int32(L), numpy.int32(N))  # frame number, destination frame size,source frame size

        buffer = gpu_buffer.get()

    return buffer
            
def apodize(data,kt=8,threshold=0.01,sigma=5,return_type='data'):
    """ Apodizes a 2d array so that upon back propagation ringing from the aperture is at least somewhat suppressed.
    This steps this function follows are as follows:
        # 1. unwrap the data
        # 2. find the boundary at each angle
        # 3. do a 1d filter along each angle
        # 4. rewrap
        
    This function works best with binary supports where the edge is clearly defined. Antialiased objects will lead to filters
    with some artifacts.

    Required input:
    data: 2d ndarray containing the data to be apodized. This data should be been sliced so that the only object is the target data. For example, a phase reconstruction with a multipartite support should be sliced so that only one component of the support is passed to this function (this may change in the future but object detection can be tricky). data should also be sufficiently symmetric such that at each angle after unwrapping there is only a single solution to the location of the object boundary.
    
    Optional named input:
    kt: strength of apodizer. float or int.
    threshold: float or int value defines the boundary of the data.
    sigma: boundary locations are smoothed to avoid with jagged edges; this sets  the smoothing. float or int.
    return_type: can return the filtered data, just the filter, or intermediates for debugging/inspection
    
    Returns: apodized array
    """
    
    # check types
    assert isinstance(data,numpy.ndarray),    "data must be an array"
    assert data.ndim == 2,                    "data must be 2d"
    assert isinstance(kt,(int,float)),        "kt must be float or int"
    assert isinstance(sigma,(int,float)),     "sigma must be float or int"
    assert isinstance(threshold,(int,float)), "threshold must be float or int"
    
    # import necessary libraries
    import wrapping
    from numpy.fft import fft
    from numpy.fft import ifft
    convolve = lambda x,y: ifft(fft(x)*fft(y))

    # find the center of mass
    rows,cols = numpy.indices(data.shape,float)
    av_row    = int(numpy.sum(data*rows)/numpy.sum(data))
    av_col    = int(numpy.sum(data*cols)/numpy.sum(data))
    
    # determine the maximum unwrapping radius
    R = int(min([av_row,len(data)-av_row,av_col,len(data)-av_col]))
    
    # unwrap the data
    unwrap_plan = wrapping.unwrap_plan(0,R,(av_col,av_row)) # very important to set r = 0!
    unwrapped   = wrapping.unwrap(data,unwrap_plan)
    ux          = unwrapped.shape[1]
    
    # at each column, find the edge of the data by stepping along rows until the object
    # is found. ways to make this faster?
    threshold = float(threshold)
    boundary = numpy.zeros(ux,float)
    for col in range(ux):
        n, temp = 1,unwrapped[:,col]
        while temp[-n] < threshold: n += 1
        boundary[col] = R-n
        
    # smooth the edge values by convolution with a gaussian
    kernel = fftshift(shape.gaussian((ux,),(sigma,),normalization=1.0))
    boundary = abs(convolve(boundary,kernel))
    
    # now that the coordinates have been smoothed, build the filter as a series of 1d filters along the column (angle) axis
    x = numpy.arange(R).astype(float)
    filter = numpy.zeros((R,ux),float)
    for col in range(ux): filter[:,col] = _apodization_f(x,boundary[col],kt)
    
    # rewrap the filter and the data to ensure cocentering and equal sizing
    rplan  = wrapping.wrap_plan(0,R)
    filter = wrapping.wrap(filter,rplan)
    data   = wrapping.wrap(unwrapped,rplan)
    
    # return a filtered version of the data.
    if return_type == 'filter': return filter
    if return_type == 'data':   return filter*data
    if return_type == 'all':    return filter*data,filter,boundary

def acutance(data,method='sobel',exponent=2,normalized=True,mask=None):
    """ Calculates the acutance of a back propagated wave field. Right now it just
    does the acutance of the magnitude component.
    
    Required input:
    data: 2d or 3d ndarray object. if 3d, the acutance of each frame will be calculated
    
    Optional input:
    method -- keyword indicating how to compute the discrete derivative. options are 'sobel' and 'roll'.
            Default is 'sobel'
    exponent -- raise the modulus of the gradient to this value. 2 by default.
    normalized -- normalize all the back propagated signals so they have the same amount of power. Default is True.
    mask -- you can supply a binary ndarray as mask so that the calculation happens only in a certain
        region of space. If no mask is supplied the calculation is over all space.
        Default is None
     
    Returns:
    a list of acutance values, one for each frame of the supplied data.
    
    """
   
    # check types
    assert isinstance(data, numpy.ndarray),               "data must be ndarray"
    assert data.ndim in (2,3),                            "data must be 2d or 3d"
    assert method in ('sobel','roll'),                    "unknown derivative method"
    assert isinstance(exponent,(int, float)),             "exponent must be float or int"
    assert isinstance(normalized, type(True)),            "normalized must be bool"
    assert isinstance(mask, (type(None), numpy.ndarray)), "mask must be None or ndarray"
    
    if data.ndim == 2: data.shape = (1,data.shape[0],data.shape[1])
    if mask == None: mask = numpy.zeros(data.shape[1:],float)

    # calculate the acutance
    acutance_list = []
    for frame in data:
        acutance_list.append(_acutance_calc(abs(frame),method,normalized,mask,exponent))
        
    # return the calculation
    return acutance_list
            
def _acutance_calc(data,method,normalized,mask,exponent):

    if method == 'sobel':
        from scipy.ndimage.filters import sobel
        dx = abs(sobel(data,axis=-1))
        dy = abs(sobel(data,axis=0))
                
    if method == 'roll':
        dx = frame-numpy.roll(data,1,axis=0)
        dy = frame-numpy.roll(data,1,axis=1)

    gradient  = numpy.sqrt(dx**2+dy**2)**exponent
            
    a = numpy.sum(d*mask)
    if normalized: a *= 1./numpy.sum(frame*mask)
    return a
    
def _apodization_f(x,n,kt):
    F = 1./(numpy.exp((x-(n-1.5))/kt)+1) # why n-1.5 instead of just n???
    F += -.5
    F[F < 0] = 0
    F *= 1./F.max()
    return F
    
