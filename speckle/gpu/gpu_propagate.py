from .. import scattering,shape
import numpy,gpu,string

def gpu_propagate_distance(gpuinfo,data,distances,energy_or_wavelength,pixel_pitch,subregion=None,silent=True):
    """ Propagates a complex-valued wavefield through a range of distances using
    the GPU. Technically this can be used to propagate a single distance, but
    the expense of starting the gpu calculation means it is only worthwhile for
    a larger set of values.
    
    Required input:
        gpuinfo: the information tuple returned by gpu.init()
        data: a square numpy.ndarray, either float32 or complex64, to be
            propagated.
        distances: an iterable set of distances (in meters) over which the
            propagated field is calculated.
        energy_or_wavelength: the energy (in eV) or wavelength (in meters) of
            the wavefield. It's assumed that if the number is < 1 it is a
            wavelength.
        pixel_pitch: the size (in meters) of each pixel in data.
    
    Optional input:
        subregion: because the size of the propagated dataset can become very
        large, subregion allows only a portion to be saved. If specified as an
        integer, save a square of size subregion centered at the center of data.
        If a tuple or list, interpret subregion as (rmin, rmax, cmin, cmax);
        consequently must be of length 4.
    
    Returns: a complex-valued 3d ndarray of shape
        (len(distances),subarraysize,subarraysize). If the subarraysize argument
        wasn't used the size of the output array is
        (len(distances),len(data),len(data)). Returned[n] is the wavefield
        propagated to distances[n].
    """
    
    try:
        import pyopencl as cl
        import pyopencl.array as cl_array
        from pyopencl.elementwise import ElementwiseKernel
        from pyfft.cl import Plan
    except ImportError:
        print "necessary gpu libraries not installed"
        exit()
    
    kp = string.join(gpu.__file__.split('/')[:-1],'/')+'/kernels/' # kernel path
        
    # check types and defaults
    assert isinstance(data,numpy.ndarray),                      "data must be an array"
    assert data.dtype in (float, complex),                      "data must be float or complex"
    assert data.ndim == 2,                                      "supplied wavefront data must be 2-dimensional"
    assert data.shape[0] == data.shape[1],                      "data must be square"
    assert isinstance(energy_or_wavelength, (float, int)),      "must supply either wavelength in meters or energy in eV"
    assert isinstance(distances, (list, tuple, numpy.ndarray)), "distances must be an iterable set (list or ndarray) of distances given in meters"
    assert isinstance(pixel_pitch, (int,float)),                "pixel_pitch must be a float saying how big each pixel is in meters"
    
    # do checking on subregion specification
    # subregion can be any of the following:
    # 1. an integer, in which case the subregion is a square cocentered with data
    # 2. a 2 element iterable, in which case the subregion is a rectangle cocentered with data
    # 3. a 4 element iterable, in which case the subregion is a rectangle specified by (rmin, rmax, cmin, cmax)
    iterable_types = (tuple,list,numpy.ndarray)
    coord_types = (int,float,numpy.int32,numpy.float32,numpy.float64)
    N = data.shape[0]
    if subregion == None: subregion = N
    
    assert isinstance(subregion,coord_types) or isinstance(subregion,iterable_types), "subregion type must be integer or iterable"
    if isinstance(subregion,(int,numpy.int32)):
        sr = (N/2-subregion/2,N/2+subregion/2,N/2-subregion/2,N/2+subregion/2)
    if isinstance(subregion,(float,numpy.float32,numpy.float64)):
        subregion = int(subregion)
        sr = (N/2-subregion/2,N/2+subregion/2,N/2-subregion/2,N/2+subregion/2)
    
    if isinstance(subregion, iterable_types):
        assert len(subregion) in (2,4), "subregion length must be 2 or 4, is %s"%len(subregion)
        for x in subregion:
            assert isinstance(x,coord_types), ""
            assert x <= data.shape[0] and x >= 0, "coords in subregion must be between 0 and size of data"
        if len(subregion) == 2:
            h = int(subregion[0])/2
            w = int(subregion[1])/2
            sr = (N/2-h,N/2+h,N/2-w,N/2+w)
        if len(subregion) == 4:
            sr = (int(subregion[0]),int(subregion[1]),int(subregion[2]),int(subregion[3]))

    context,device,queue,platform = gpuinfo
    
    # convert energy_or_wavelength to wavelength.  If < 1 assume it's a wavelength.
    if energy_or_wavelength < 1: wavelength = energy_or_wavelength
    else: wavelength = scattering.energy_to_wavelength(energy_or_wavelength)*1e-10
    
    # check distances against upperlimit
    upperlimit = (pixel_pitch*N)**2/(wavelength*N)
    if numpy.sum(numpy.where(distances > upperlimit,1,0)) > 0:
        print "some distances are greater than the phase-accurate upper limit (%s)"%upperlimit
    
    # make the fft plan for NxN interleaved (ie, complex64) data
    fftplan = Plan((N,N),queue=queue)

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
    copy_to_buffer = gpu.build_kernel_file(context, device, kp+'copy_to_save_buffer.cl')
    
    # preco0mpute the fourier signals
    r = numpy.fft.fftshift((shape.radial((N,N)))**2)
    f = numpy.fft.fft2(data)
    
    # put the signals onto the gpu along with buffers for the various operations
    gpu_rarray  = cl_array.to_device(queue,r.astype(numpy.float32))   # r array
    gpu_fourier = cl_array.to_device(queue,f.astype(numpy.complex64)) # fourier data
    gpu_phase   = cl_array.empty(queue,(N,N),numpy.complex64)         # buffer for phase factor and phase-fourier product
    gpu_back    = cl_array.empty(queue,(N,N),numpy.complex64)         # buffer for propagated wavefield
    gpu_buffer  = cl_array.empty(queue,(len(distances),int(sr[1]-sr[0]),int(sr[3]-sr[2])),numpy.complex64) # allocate propagation buffer

    # now compute the propagations
    rows = int(sr[1]-sr[0])
    cols = int(sr[3]-sr[2])
    for n,z in enumerate(distances):
        
        # compute on gpu and transfer to buffer.
        if not silent: print z
        phase_factor(gpu_rarray,t,numpy.float32(z),gpu_phase)    # form the phase factor
        complex_multiply(gpu_phase,gpu_fourier,gpu_phase)        # form the phase-data product. store in gpu_phase
        fftplan.execute(gpu_phase.data,data_out=gpu_back.data,   # inverse transform
                        inverse=True,wait_for_finish=True)

        copy_to_buffer.execute(queue,(rows,cols),                      # opencl stuff
                               gpu_buffer.data,    gpu_back.data,      # destination, source pointers
                               numpy.int32(n),     numpy.int32(N),     # frame number, source frame size
                               numpy.int32(sr[0]), numpy.int32(sr[2])) # subregion bounding coordinates

    return gpu_buffer.get()