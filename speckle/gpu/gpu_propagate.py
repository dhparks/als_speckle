def gpu_propagate_distance(gpuinfo,data,distances,energy_or_wavelength,pixel_pitch,subarraysize=None):
    
    """ Propagates a complex-valued wavefield through a range of distances using the GPU.
    
    Required input:
        gpuinfo: the information tuple returned by gpu.init()
        data: a square numpy.ndarray, either float32 or complex64, to be propagated
        distances: an iterable set of distances (in meters!) over which the propagated field is calculated
        energy_or_wavelength: the energy (in eV) or wavelength (in meters) of the wavefield. It's assumed that if the number is < 1 it is a wavelength.
        pixel_pitch: the size (in meters) of each pixel in data.
    
    Optional input:
        subarraysize: because the number of files can be large a subarray cocentered with data can be specified.
    
    Returns: a complex-valued 3d ndarray of shape (len(distances),subarraysize,subarraysize). If the subarraysize argument wasn't used the
    size of the output array is (len(distances),len(data),len(data)). returned[n] is the wavefield propagated to distances[n]."""
    
    try:
        import pyopencl as cl
        import pyopencl.array as cl_array
        from pyopencl.elementwise import ElementwiseKernel
        from pyfft.cl import Plan
    except ImportError:
        print "necessary gpu libraries not installed"
        exit()
        
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
    assert subarraysize%2 == 0, "subarraylength on gpu should be even number"
    
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
    copy_to_buffer = gpu.build_kernel_file(context, device, 'kernels/copy_to_save_buffer.cl')
    
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