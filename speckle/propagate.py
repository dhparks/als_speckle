""" A library for simulating the near-field propagation of a complex wavefield.

Author: Daniel Parks (dhparks@lbl.gov)"""

import numpy
DFT = numpy.fft.fft2
IDFT = numpy.fft.ifft2
shift = numpy.fft.fftshift

from . import shape, conditioning
#import shape
I = complex(0,1)

def propagate_one_distance(data,energy_or_wavelength=None,z=None,pixel_pitch=None,phase=None,data_is_fourier=False):
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
        r = shift((shape.radial((N,N)))**2)
        phase = numpy.exp(-I*numpy.pi*wavelength*z*r/(pixel_pitch*N)**2)
        
    else:
        # phase has been supplied, so check its types for correctness. if phase-generating parameters are supplied they are ignored.
        assert isinstance(phase,numpy.ndarray), "phase must be an array"
        assert phase.shape == data.shape,       "phase and data must be same shape"
        
    if not data_is_fourier: data = DFT(data)
    
    return IDFT(data*phase)

def propagate_distance(data,distances,energy_or_wavelength,pixel_pitch,gpuinfo=None,subarraysize=None):
    """ Propagates a complex-valued wavefield through a range of distances
    using the CPU. A GPU-accelerated version is available elsewhere.
    
    Required input:
        data: a square numpy array, either float32 or complex64, to propagate
        distances: an iterable set of distances (in meters!) to propagate
        energy_or_wavelength: the energy (in eV) or wavelength (in meters) of
            the wavefield. If < 1, assume wavelength is specified.
        pixel_pitch: the size (in meters) of each pixel in data.
    
    Optional input:
        subarraysize: because the number of files can be large a subarray
            cocentered with data can be specified.
    
    Returns: a complex-valued 3d ndarray with shape:
        (len(distances),subarraysize,subarraysize)
    
        If the subarraysize argument wasn't used the output shape is:
        (len(distances),len(data),len(data))
    
    returned[n] is the wavefield propagated to distances[n].
    """
    
    from . import scattering
   
    # check types and defaults
    assert isinstance(data,numpy.ndarray),                              "data must be an array"
    assert data.dtype in (float, complex),                              "data must be float or complex"
    assert data.ndim == 2,                                              "supplied wavefront data must be 2-dimensional"
    assert data.shape[0] == data.shape[1],                              "data must be square"
    assert isinstance(energy_or_wavelength, (float, int)),              "must supply either wavelength in meters or energy in eV"
    assert isinstance(distances, (list, tuple, numpy.ndarray)),         "distances must be an iterable set (list or ndarray) of distances given in meters"
    assert isinstance(pixel_pitch, (int,float)),                        "pixel_pitch must be a float saying how big each pixel is in meters"
    if subarraysize == None: subarraysize = len(data)
    assert isinstance(subarraysize, int) and subarraysize <= len(data), "subarray must be int smaller than supplied length of data"

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

    # compute the distances of back propagations on the cpu.
    # precompute the fourier signal. define the phase as a lambda function. loop through the distances
    # calling phase and propagate_one_distance. save the region of interest (subarray) to the buffer.
    fourier = DFT(data)
    r = shift((shape.radial((N,N)))**2)
    cpu_phase = lambda z: numpy.exp(-I*numpy.pi*wavelength*z*r/(pixel_pitch*N)**2)
    for n,z in enumerate(distances):
        if z > upperlimit: print "propagation distance (%s) exceeds accuracy upperlimit (%s)"%(z,upperlimit)
        phase_z = cpu_phase(z)
        back = propagate_one_distance(fourier,phase=phase_z,data_is_fourier=True)
        buffer[n] = back[N/2-subarraysize/2:N/2+subarraysize/2,N/2-subarraysize/2:N/2+subarraysize/2]

    return buffer
            
def apodize(data,kt=.1,threshold=0.01,sigma=5,return_type='data'):
    """ Apodizes a 2d array so that upon back propagation ringing from the
    aperture is at least somewhat suppressed. The steps this function follows
    are as follows:
        # 1. unwrap the data
        # 2. find the boundary at each angle
        # 3. do a 1d filter along each angle
        # 4. rewrap

    Required arguments:
        data: 2d ndarray containing the data to be apodized. This data should
        be been sliced so that the only object is the target data.
    
    Optional arguments:
        kt: strength of apodizer. Default is kt=0.1. kt=1.0 is VERY strong.
        threshold: float or int value defines the boundary of the data.
        sigma: boundary locations are smoothed to avoid with jagged edges;
            this sets  the smoothing. float or int.
        return_type: Determines what can be returned.  This can be 'data'
            (default), 'filter', which returns the filter or 'all', which
            returns three items: (data, filter, boundary)
        
    Returns:
        The return value depends on return_type, but the default is 'data'
            which returns the apodized array. Other options are 'filter' or
            'all'.
    """
    
    # check types
    assert isinstance(data,numpy.ndarray),    "data must be an array"
    assert data.ndim == 2,                    "data must be 2d"
    assert isinstance(kt,(int,float)),        "kt must be float or int"
    assert isinstance(sigma,(int,float)),     "sigma must be float or int"
    assert isinstance(threshold,(int,float)), "threshold must be float or int"
    
    # if the data is complex, operate only on the magnitude component
    was_complex = False
    if numpy.iscomplexobj(data):
        phase = numpy.angle(data)
        data = abs(data)
        was_complex = True
        
    # import necessary libraries
    from . import wrapping
    convolve = lambda x,y: numpy.fft.ifft(numpy.fft.fft(x)*numpy.fft.fft(y))

    # find the center of mass
    N,M       = data.shape
    rows,cols = numpy.indices((N,M),float)
    av_row    = numpy.sum(data*rows)/numpy.sum(data)
    av_col    = numpy.sum(data*cols)/numpy.sum(data)
    
    # determine the maximum unwrapping radius
    R = int(min([av_row,len(data)-av_row,av_col,len(data)-av_col]))
    
    # unwrap the data
    unwrap_plan = wrapping.unwrap_plan(0,R,(av_col,av_row)) # very important to set r = 0!
    unwrapped   = wrapping.unwrap(data,unwrap_plan)
    ux          = unwrapped.shape[1]
    u_der       = unwrapped-numpy.roll(unwrapped,-1,axis=0)
    
    # at each column, find the edge of the data by stepping along rows until the object
    # is found. ways to make this faster?
    threshold = float(threshold)
    boundary = numpy.zeros(ux,float)
    indices = numpy.arange(R-1)
    for col in range(ux):
        u_der_col = u_der[:,col]
        ave = numpy.sum(u_der_col[:-1]*indices)/numpy.sum(u_der_col[:-1])
        boundary[col] = ave
        
    # smooth the edge values by convolution with a gaussian
    kernel = shift(shape.gaussian((ux,),(sigma,),normalization=1.0))
    boundary = abs(convolve(boundary,kernel))-1
    
    # now that the coordinates have been smoothed, build the filter as a series of 1d filters along the column (angle) axis
    x = numpy.outer(numpy.arange(R),numpy.ones(ux)).astype(float)/boundary
    filter = _apodization_f(x,kt)
    
    # rewrap the filter. align the filter to the data
    rplan  = wrapping.wrap_plan(0,R)
    filter = wrapping.wrap(filter,rplan)
    
    e_filter = numpy.zeros_like(data)
    e_filter[0:filter.shape[0],0:filter.shape[1]] = filter
    
    data_mask   = numpy.where(data > 1e-6,1,0)
    filter_mask = numpy.where(e_filter > 1e-6,1,0)
    
    rolls  = lambda d, r0, r1: numpy.roll(numpy.roll(d,r0,axis=0),r1,axis=1)
    coords = conditioning.align_frames(filter_mask,align_to=data_mask,return_type='coordinates')[0]
    filter = rolls(e_filter,coords[0],coords[1])

    if was_complex: data *= numpy.exp(complex(0,1)*phase)
    
    # return a filtered version of the data.
    if return_type == 'filter': return filter
    if return_type == 'data':   return filter*data
    if return_type == 'all':    return filter*data,filter,boundary

def acutance(data,method='sobel',exponent=2,normalized=True,mask=None):
    """ Calculates the acutance of a back propagated wave field. Right now it
    just does the acutance of the magnitude component.
    
    Required input:
        data -- 2d or 3d ndarray object. if 3d, the acutance of each frame will
            be calculated.
    
    Optional input:
        method -- keyword indicating how to compute the discrete derivative.
            options are 'sobel' and 'roll'. Default is 'sobel'. 'roll' estimates
            the derivative by a shift-subtract in each axis.
        exponent -- raise the modulus of the gradient to this value. Default 2.
        normalized -- normalize all the back propagated signals so they have the
            same amount of power. Default is True.
        mask -- you can supply a binary ndarray as mask so that the calculation
            happens only in a certain  region of space. If no mask is supplied
            the calculation is over all space. Default is None.
     
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
    if mask == None: mask = numpy.ones(data.shape[1:],float)

    # calculate the acutance
    acutance_list = []
    for frame in data:
        acutance_list.append(_acutance_calc(abs(frame).real,method,normalized,mask,exponent))
        
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
            
    a = numpy.sum(gradient*mask)
    if normalized: a *= 1./numpy.sum(data*mask)
    return a

def _apodization_f(x,kt):
    F = 1./(numpy.exp((x-1)/kt)+1)
    F += -.5
    F[F < 0] = 0
    m = numpy.max(F,axis=0)
    F *= 1./m
    return F
    
