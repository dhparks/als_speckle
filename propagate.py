import scipy
from scipy.fftpack import fft2 as DFT
from scipy.fftpack import ifft2 as IDFT
from scipy.fftpack import fftshift
from types import *
import shape, io2

def propagate_single_cpu(data,phase=None,z=None,energy=None,wavelength=None,pixel_pitch=None,data_is_fourier=False):

    # check requirements regarding types of data
    assert type(data) is scipy.ndarray,                                      "data must be an array"
    assert data.dtype in [scipy.dtype('float32'), scipy.dtype('complex64')], "data must be float32 or complex64"
    assert data.ndim == 2,                                                   "supplied wavefront data must be 2-dimensional"
    assert type(data_is_fourier) == bool,                                    "data_is_fourier must be bool"
    
    # first see if a phase is supplied. if not, make it from the supplied parameters.
    if phase == None:
    
        assert (energy != None or wavelength != None),  "must supply either wavelength in meters or energy in eV"
        if energy != None and wavelength != None: print "don't supply both energy and wavelength", exit()
        assert type(pixel_pitch) == FloatType,          "pixel_pitch must be a float saying how big each pixel is in meters"
        assert type(z) == FloatType,                    "z must be a float giving how far to propagate in meters"
    
        N = len(data)
        I = complex(0,1)
        r = fftshift((shape.radial((N,N)))**2)
        if energy != None and wavelength == None: wavelength = 1240e-9/energy
        phase = scipy.exp(-I*scipy.pi*wavelength*z*r/(pixel_pitch*N)**2)
        
    else:
        # phase has been supplied, so check its types for correctness. if phase-generating parameters are supplied they are ignored.
        assert type(phase) == scipy.ndarray, "phase must be an array"
        assert phase.shape == data.shape,    "phase and data must be same shape"
        
    if not data_is_fourier: data = DFT(data)
    
    return IDFT(data*phase)
    
def propagate_spectrum(data,spectrum,energy=None,wavelength=None,pixel_pitch=None,device='CPU',subarraysize=None):

    """ Propagates a complex-valued wavefield through a range of distances.
    
    Required positional input:
    1. data: a square numpy.ndarray, either float32 or complex64, to be propagated
    2. spectrum: an iterable set of distances (in meters!) over which the propagated field is calculated
    
    Required named input:
    energy or wavelength: the energy (in eV) or wavelength (in meters) of the wavefield. Only supply one or the other, not both.
    pixel_pitch: the size (in meters) of each pixel in data.
    
    Optional named input:
    device: 'CPU' or 'GPU'. If GPU, you get much faster computation although data transfers limit the speed-up. Default is CPU.
    subarraysize: because the number of files can be large a subarray cocentered with data can be specified.
    
    Returns: a complex-valued 3d ndarray of shape (len(spectrum),subarraysize,subarraysize). If the subarraysize argument wasn't used the
    size of the output array is (len(spectrum),len(data),len(data)).
    """
    
    # check requirements regarding types of data
    assert type(data) is scipy.ndarray,                                      "data must be an array"
    assert data.dtype in [scipy.dtype('float32'), scipy.dtype('complex64')], "data must be float32 or complex64"
    assert data.ndim == 2,                                                   "supplied wavefront data must be 2-dimensional"
    assert data.shape[0] == data.shape[1],                                   "data must be square"
    assert (energy != None or wavelength != None),                           "must supply either wavelength in meters or energy in eV"
    assert type(spectrum) in [ListType,scipy.ndarray],                       "spectrum must be an iterable set (list or ndarray) of distances given in meters"
    assert type(pixel_pitch) == FloatType,                                   "pixel_pitch must be a float saying how big each pixel is in meters"
    if subarraysize == None: subarraysize = len(data)
    assert type(subarraysize) == int and subarraysize <= len(data),          "subarray must be int smaller than supplied length of data"
    assert device in ['CPU','GPU'],                                          "device not recognized"
    if device == 'GPU': assert subarraysize%2 == 0,                          "subarraylength on gpu should be even number"
    
    # don't supply both energy and wavelength even if they are consistent.
    # there must be a better way to check this
    if energy != None and wavelength != None:
        print "don't supply both energy and wavelength"
        exit()
        
    N = len(data)
    I = complex(0,1)
    try:
        buffer = scipy.zeros((len(spectrum),subarraysize,subarraysize),'complex64')
    except MemoryError:
        print "save buffer was too large. either use a smaller spectrum or a smaller subarraylength"

    # if energy was supplied, convert it to wavelength
    if energy != None and wavelength == None: wavelength = 1240e-9/energy
    
    # this is the upper limit of the propagation distance. after this point nysquist sampling
    # limits the reliability of the phase component, although the magnitude component is still useful.
    upperlimit = (pixel_pitch*N)**2/(wavelength*N)

    if device == 'CPU':
        # compute the spectrum of back propagations on the cpu.
        
        r = fftshift((shape.radial((N,N)))**2)
        cpu_phase = lambda z: scipy.exp(-I*scipy.pi*wavelength*z*r/(pixel_pitch*N)**2)
        
        # precompute the fourier signal    
        fourier = DFT(data)
        
        # propagate each distance in the spectrum. compute the phase factor and call the propagation routine
        for n,z in enumerate(spectrum):
            
            phase_z = cpu_phase(z)
            back = propagate_single_cpu(fourier,phase=phase_z,data_is_fourier=True)
            buffer[n] = back[N/2-subarraysize/2:N/2+subarraysize/2,N/2-subarraysize/2:N/2+subarraysize/2]

    if device == 'GPU':
        
        # do everything within this function. there is no reason to define a propagation_single_gpu as io transfer will
        # obliterate any benefit to gpu FFT
        
        import pyopencl as cl
        import pyopencl.array as cl_array
        from pyopencl.elementwise import ElementwiseKernel
        from pyfft.cl import Plan
        
        L = subarraysize
        
        # initialize the gpu
        platforms = cl.get_platforms()
        devices = platforms[0].get_devices(cl.device_type.GPU)
        context = cl.Context([devices[0]])
        device = devices[0]
        queue = cl.CommandQueue(context)
        
        # make the fft plan for complex64
        fftplan = Plan((N,N),queue=queue)
        
        # this is a bunch of constants which go to the phase_factor kernel
        t = scipy.pi*wavelength/((pixel_pitch*N)**2)
        
        # make the kernel which computes the phase factor
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
        
        # put the buffer-copy kernel in as a docstring rather than build from an external file
        copy_to_buffer = cl.Program(context,
        """__kernel void execute(__global float2 *dst, __global float2 *src, int n, int L, int N)
        {
            int i_dst = get_global_id(0);
            int j_dst = get_global_id(1);
            int x0 = N/2-L/2;
            int y0 = N/2-L/2;
        
            // i_dst and j_dst are the coordinates of the destination. we "simply" need to turn them into 
            // the correct indices to move values from src to dst.
            
            int dst_index = (n*L*L)+(j_dst*L)+i_dst; // (frames)+(rows)+cols
            int src_index = (i_dst+x0)+(j_dst+y0)*N; // (cols)+(rows)
            
            dst[dst_index] = src[src_index];
        }""").build(devices=[device])
        
        # precompute the fourier signals
        r = fftshift((shape.radial((N,N)))**2)
        f = DFT(data)
        
        # put the signals onto the gpu along with buffers for the various operations
        gpu_rarray  = cl_array.to_device(queue,r.astype(scipy.float32))   # r array
        gpu_fourier = cl_array.to_device(queue,f.astype(scipy.complex64)) # fourier data
        gpu_phase   = cl_array.empty(queue,(N,N),scipy.complex64)         # buffer for phase factor
        gpu_product = cl_array.empty(queue,(N,N),scipy.complex64)         # buffer for data-phase product
        gpu_back    = cl_array.empty(queue,(N,N),scipy.complex64)         # buffer for propagated wavefield, +z direction
        try:
            gpu_buffer  = cl_array.empty(queue,(len(spectrum),L,L),scipy.complex64) # allocate propagation buffer
        except:
            print "probably ran out of memory. make subarraysize smaller or have fewer points in the spectrum"
            exit()
        
        # now compute the propagations
        for n,z in enumerate(spectrum):

            # compute on gpu and transfer to buffer.
            phase_factor(gpu_rarray,scipy.float32(t),scipy.float32(z),gpu_phase) # form the phase factor
            complex_multiply(gpu_phase,gpu_fourier,gpu_product)                  # form the phase-data product
            fftplan.execute(gpu_product.data,data_out=gpu_back.data,             # inverse transform
                            inverse=True,wait_for_finish=True)

            copy_to_buffer.execute(queue,(L,L),     # opencl stuff
                                   gpu_buffer.data, # destination
                                   gpu_back.data,   # source
                                   scipy.int32(n),  # frame number
                                   scipy.int32(L),  # destination frame size
                                   scipy.int32(N))  # source frame size

        buffer = gpu_buffer.get()

    return buffer
            
def apodize(data,kt=8,threshold=0.01,sigma=5):
    # 1. unwrap the data
    # 2. find the boundary at each angle
    # 3. do a 1d filter along each angle
    # 4. rewrap
    
    """ Apodizes a 2d array so that upon back propagation ringing from the aperture is at least somewhat suppressed.
    
    Required input:
    data: 2d ndarray containing the data to be apodized. This data should be been sliced so that the only object
    is the target data. For example, a phase reconstruction with a multipartite support should be sliced so that only
    one component of the support is passed to this function (this may change in the future but object detection
    can be tricky). data should also be sufficiently symmetric such that at each angle after unwrapping there is only
    a single solution to the location of the object boundary.
    
    Optional named input:
    kt: strength of apodizer. float or int.
    threshold: float or int value defines the boundary of the data.
    sigma: boundary locations are smoothed to avoid with jagged edges; this sets  the smoothing. float or int.
    
    Returns: apodized array
    """
    
    # check types
    assert type(data) == scipy.ndarray,            "data must be an array"
    assert data.ndim == 2,                         "data must be 2d"
    assert type(kt) in (IntType,FloatType),        "kt must be float or int"
    assert type(threshold) in (IntType,FloatType), "threshold must be float or int"
    if type(threshold) == IntType: print           "strongly recommend a float for threshold"
    assert type(sigma) in (IntType,FloatType),     "sigma must be float or int"
    
    # import necessary libraries
    import wrapping
    from shape import gaussian
    from scipy.fftpack import fft
    from scipy.fftpack import ifft

    # find the center of mass
    rows,cols = scipy.indices(data.shape,float)
    av_row    = int(scipy.sum(data*rows)/scipy.sum(data))
    av_col    = int(scipy.sum(data*cols)/scipy.sum(data))
    
    # determine the maximum unwrapping radius
    R = int(min([av_row,len(data)-av_row,av_col,len(data)-av_col]))
    
    # unwrap the data
    unwrap_plan = wrapping.unwrap_plan(0,R,(av_col,av_row)) # very important to set r = 0!
    unwrapped   = wrapping.unwrap(data,unwrap_plan)
    unwrapped_x = unwrapped.shape[1]
    
    # at each column, find the edge of the data.
    n, n_list = 1, scipy.zeros(unwrapped_x,float)
    for col in range(unwrapped_x):
        while unwrapped[-n,col] < threshold: n += 1
        n_list[col] = R-n
        
    # smooth the edge values by convolution with a gaussian
    kernel = scipy.roll(gaussian((unwrapped_x,),(sigma,),normalization=1.0))
    n_list = abs(ifft(fft(nlist)*fft(kernel)))
    
    # now that the coordinates have been smoothed, build the filter as a series of 1d filters along the column (angle) axis
    x = scipy.arange(R).astype(float)
    filter = scipy.zeros((R,unwrapped_x),float)
    for col in range(unwrapped_x): filter[:,col] = _apodization_f(x,n_list[col],kt)
    
    # rewrap the filter and the data to ensure cocentering and equal sizing
    rplan  = wrapping.wrap_plan(0,R)
    filter = wrapping.wrap(filter,rplan)
    data   = wrapping.wrap(unwrapped,rplan)
    
    # return a filtered version of the data. it would be better if there were some way to cache most of the intermediates
    # (for example, the n_list so that kt could be quickly changed) but I don't know how to do that
    return filter*data

def acutance(data,method='sobel',exponent=2,normalized=True,mask=None):
    # open the results
    # for each frame, calculate the acutance
    # plot the results

    """ Calculates the acutance of a back propagated wave field. Right now it just
    does the acutance of the magnitude component.
    
    Required input:
    1. data: 2d or 3d ndarray object. if 3d, the acutance of each frame will be calculated
    
    Optional input:
    1. method: keyword indicating how to compute the discrete derivative. options are 'sobel',
    'roll', and 'gaussiand'. sobel by default.
    2. exponent: raise the modulus of the gradient to this value. 2 by default.
    3. normalized: normalize all the back propagated signals so they have the same amount of power. True by default.
    4. mask: you can supply a binary ndarray as mask so that the calculation happens only
    in a certain region of space. If no mask is supplied the calculation is over all space.
    
    Returns: a list of acutance values, one for each frame of the supplied data.
    
    """
    
    # check types
    assert type(data) == scipy.ndarray,             "data must be ndarray"
    assert data.ndim in [2,3],                      "data must be 2d or 3d"
    assert method in ['sobel','roll'],              "unknown derivative method"
    assert type(exponent) in [int, float],          "exponent must be float or int"
    assert type(normalized) == Bool,                "normalized must be bool"
    assert type(mask) in [NoneType, scipy.ndarray], "mask must be None or ndarray"
    
    # get everything loaded
    acutance_list = []
    if method == 'sobel':
        from scipy.ndimage.filters import sobel
    if mask == None:
        if data.ndim == 3: mask = scipy.zeros(data[0].shape,float)
        if data.ndim == 2: mask = scipy.zeros(data.shape,float)

    # calculate the acutance
    if data.ndim == 3:
        for frame in data:
            acutance_list.append(_acutance_calc(abs(frame),method,normalized,mask,exponent))
            
    if data.ndim == 2:
        acutance_list.append(_acutance_calc(abs(data),method,normalized,mask,exponent))
        
    return acutance_list
            
def _acutance_calc(data,method,normalized,mask,exponent):
    if method == 'sobel':
        dx = abs(sobel(data,axis=-1))
        dy = abs(sobel(data,axis=0))
                
    if method == 'roll':
        dx = frame-scipy.roll(data,1,axis=0)
        dy = frame-scipy.roll(data,1,axis=1)

    d  = scipy.sqrt(dx**2+dy**2)**exponent
            
    a = scipy.sum(d*mask)
    if normalized: a *= 1./scipy.sum(frame*mask)
    return a
    
def _apodization_f(x,n,kt):
    F = 1./(scipy.exp((x-(n-1.5))/kt)+1) # why n-1.5 instead of just n???
    F += -.5
    F[F < 0] = 0
    F *= 1./F.max()
    return F
    
