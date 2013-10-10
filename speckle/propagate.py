""" A library for simulating the near-field propagation of a complex wavefield.

Author: Daniel Parks (dhparks@lbl.gov)"""

import numpy as np

from . import shape, conditioning, scattering, wrapping

def propagate_one_distance(data,energy_or_wavelength=None,z=None,pixel_pitch=None,phase=None,data_is_fourier=False,band_limit=False):
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
    assert isinstance(data,np.ndarray),                "data must be an array"
    assert data.shape[0] == data.shape[1],                "data must be square"
    data = data.astype(np.complex64)
    assert data.ndim == 2,                                "supplied wavefront data must be 2-dimensional"
    assert type(data_is_fourier) == bool,                 "data_is_fourier must be bool"
    assert isinstance(phase,(np.ndarray,type(None))),  "phase must be array or None"
    assert type(band_limit) == bool,                      "band_limit must be bool"

    I = complex(0,1)
    # first see if a phase is supplied. if not, make it from the supplied parameters.
    if phase == None:
    
        assert isinstance(energy_or_wavelength, (int,float,type(None))), "energy/wavelength must be float or int"
        assert isinstance(pixel_pitch, float), "pixel_pitch must be a float saying how big each pixel is in meters"
        assert isinstance(z, float), "z must be a float giving how far to propagate in meters"
    
        # convert energy_or_wavelength to wavelength.  If < 1 assume it's a wavelength.
        if energy_or_wavelength < 1: wavelength = energy_or_wavelength
        else: wavelength = scattering.energy_to_wavelength(energy_or_wavelength)*1e-10
    
        N = len(data)
        r = np.fft.fftshift((shape.radial((N,N)))**2)
        phase = np.exp(-I*np.pi*wavelength*z*r/(pixel_pitch*N)**2)
        
        upper_limit = N*pixel_pitch**2/wavelength # this is the nyquist limit on the far-field quadratic phase factor

                
    else:
        # phase has been supplied, so check its types for correctness. if phase-generating parameters are supplied they are ignored.
        assert isinstance(phase,np.ndarray), "phase must be an array"
        assert phase.shape == data.shape,       "phase and data must be same shape"
        upper_limit = -1
        
    if not data_is_fourier:
        res = np.fft.fft2(data)
    else:
        res = data
        
    if z > upper_limit and z > 0:
        #print "warning! z (%s) exceeds upper limit (%s)"%(z,upper_limit)
        if band_limit: res *= np.fft.fftshift(shape.circle((N,N),N/2*upper_limit/z))

    return np.fft.ifft2(res*phase)

def propagate_distances(data,distances,energy_or_wavelength,pixel_pitch,subregion=None,silent=True,band_limit=False,gpu_info=None,im_convert=False):
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
        subarraysize -- because the number of files can be large a subarray
            cocentered with data can be specified.
        silent -- if False, report what distance is currently being calculated.
        gpu_info -- what gets returned by speckle.gpu.init().
        im_convert -- if True, will convert each of the propagated frames
            to hsv color space and then into PIL objects. 
    
    Returns:
    
        if im_convert == False:
            a complex-valued 3d ndarray with shape:
            (len(distances),subarraysize,subarraysize)
        
            If the subarraysize argument wasn't used the output shape is:
            (len(distances),len(data),len(data))
            
        if im_convert == True:
        
            (propagated_array, propagated_pil_images)
            
            propagated_array is the same as in the im_convert=False case.
            propagated_pil_images is a list of images whose shapes are
            the same as the frame in propagated_array.
    """
    
    import sys
    
    iterable_types = (tuple,list,np.ndarray)
    coord_types = (int,float,np.int32,np.float32,np.float64)
    
    # check types and defaults
    assert isinstance(data,np.ndarray),            "data must be an array"
    data = data.astype(np.complex64)
    assert data.ndim == 2,                         "supplied wavefront data must be 2-dimensional"
    assert data.shape[0] == data.shape[1],         "data must be square"
    assert isinstance(distances, iterable_types), "distances must be an iterable set (list or ndarray) of distances given in meters" 

    N = data.shape[0]
    pixel_pitch          = np.float32(pixel_pitch)
    energy_or_wavelength = np.float32(energy_or_wavelength)
    numfr = len(distances)

    # do checking on subregion specification
    # subregion can be any of the following:
    # 1. an integer, in which case the subregion is a square cocentered with data
    # 2. a 2 element iterable, in which case the subregion is a rectangle cocentered with data
    # 3. a 4 element iterable, in which case the subregion is a rectangle specified by (rmin, rmax, cmin, cmax)
    if subregion == None: subregion = N
    assert isinstance(subregion,coord_types) or isinstance(subregion,iterable_types), "subregion type must be integer or iterable"
    if isinstance(subregion,(int,np.int32)):
        sr = (N/2-subregion/2,N/2+subregion/2,N/2-subregion/2,N/2+subregion/2)
    if isinstance(subregion,(float,np.float32,np.float64)):
        subregion = int(subregion)
        sr = (N/2-subregion/2,N/2+subregion/2,N/2-subregion/2,N/2+subregion/2)
    if isinstance(subregion, iterable_types):
        assert len(subregion) in (1,2,4), "subregion length must be 2 or 4, is %s"%len(subregion)
        for x in subregion:
            assert isinstance(x,coord_types), ""
            assert x <= data.shape[0] and x >= 0, "coords in subregion must be between 0 and size of data"
        if len(subregion) == 1:
            h = int(subregion[0])/2
            w = int(subregion[0])/2
            sr = (N/2-h,N/2+h,N/2-w,N/2+w)
        if len(subregion) == 2:
            h = int(subregion[0])/2
            w = int(subregion[1])/2
            sr = (N/2-h,N/2+h,N/2-w,N/2+w)
        if len(subregion) == 4:
            sr = (int(subregion[0]),int(subregion[1]),int(subregion[2]),int(subregion[3]))
            
    rows = int(sr[1]-sr[0])
    cols = int(sr[3]-sr[2])

    # convert energy_or_wavelength to wavelength.  If < 1 assume it's a wavelength.
    if    energy_or_wavelength < 1: wavelength = energy_or_wavelength
    else: wavelength = scattering.energy_to_wavelength(energy_or_wavelength)*1e-10
    
    # precompute the fourier signals
    r = np.fft.fftshift((shape.radial((N,N)))**2)
    f = np.fft.fft2(data)
    t = np.float32(np.pi*wavelength/((pixel_pitch*N)**2))

    ### first, the gpu codepath
    use_gpu = True
    if gpu_info == None:
        use_gpu = False
    if use_gpu:
        # if a gpu is requested, see if one is available (almost certainly yes since
        # the init information is being passed)
        try:
            import gpu
            import string
            import pyopencl as cl
            import pyopencl.array as cla
            from pyfft.cl import Plan
            use_gpu = True
        except ImportError:
            use_gpu = False
            
    if use_gpu:
        
        context,device,queue,platform = gpu_info

        # make fft plan
        fftplan = Plan((N,N),queue=queue)
        
        # build kernels
        kp = string.join(gpu.__file__.split('/')[:-1],'/')+'/kernels/' # kernel path
        phase_factor   = gpu.build_kernel_file(context, device, kp+'propagate_phase_factor.cl')
        copy_to_buffer = gpu.build_kernel_file(context, device, kp+'propagate_copy_to_save_buffer.cl')
        multiply       = gpu.build_kernel_file(context, device, kp+'common_multiply_f2_f2.cl')

        # put the signals onto the gpu along with buffers for the various operations
        rarray  = cla.to_device(queue,r.astype(np.float32))      # r array
        fourier = cla.to_device(queue,data.astype(np.complex64)) # fourier data
        phase   = cla.empty(queue,(N,N),np.complex64)            # buffer for phase factor and phase-fourier product
        back    = cla.empty(queue,(N,N),np.complex64)            # buffer for propagated wavefield
        store   = cla.empty(queue,(numfr,rows,cols),np.complex64) # allocate propagation buffer
        
        # precompute the fourier transform of data
        fftplan.execute(fourier.data,wait_for_finish=True)
        
        # now compute the propagations
        N = np.int32(N)
        sr0 = np.int32(sr[0])
        sr2 = np.int32(sr[2])

        for n,z in enumerate(distances):
            
            n = np.int32(n)
            
            # compute on gpu and transfer to buffer.
            if not silent:
                sys.stdout.write("\rpropagating: %1.2e m (%02d/%02d)" % (z, n+1, numfr))
                sys.stdout.flush()
            
            # make the phase factor; multiply the fourier rep; inverse transform
            phase_factor.execute(queue,(int(N*N),),rarray.data,np.float32(t*z), phase.data)
            multiply.execute(queue,(int(N*N),),phase.data,fourier.data,phase.data) 
            fftplan.execute(data_in=phase.data,data_out=back.data,inverse=True,wait_for_finish=True)

            # slice the subregion from the back-propagation and save in store
            copy_to_buffer.execute(queue,(rows,cols), store.data, back.data, n, N, sr0, sr2)
            
        if not silent:
            print ""

        if im_convert:
            
            # build the additional kernels
            hsv_convert = gpu.build_kernel_file(context, device, kp+'complex_to_rgb.cl')
            cl_abs      = gpu.build_kernel_file(context, device, kp+'common_abs_f2_f.cl')
            
            # calculate abs of entire buffer. this is so we can get the maxval
            abs_store = cla.empty(queue,store.shape,np.float32)
            cl_abs.execute(queue,(store.size,),store.data,abs_store.data)
            maxval = np.float32(cla.max(abs_store).get())
            
            # allocate new memory (frames, rows, columns, 3); uchar -> uint8?
            new_shape    = store.shape+(3,)
            image_buffer = cla.empty(queue,new_shape,np.uint8)
            debug_buffer = cla.empty(queue,store.shape,np.float32)
            hsv_convert.execute(queue,store.shape,store.data,image_buffer.data,maxval).wait()

            # now convert to pil objects
            import scipy.misc as smp
            images = []
            image_buffer = image_buffer.get()
            for image in image_buffer: images.append(smp.toimage(image))
            return store.get(), images # done
        
        else:
            
            return store.get()

    ### now the cpu fallback
    if not use_gpu:
    
        I = complex(0,1)
        upperlimit = (pixel_pitch*N)**2/(wavelength*N)
        
        try: store = np.zeros((numfr,rows,cols),'complex64')
        except MemoryError:
            print "save buffer was too large. either use a smaller set of distances or a smaller subregion"

        # compute the distances of back propagations on the cpu.
        # precompute the fourier signal. define the phase as a lambda function. loop through the distances
        # calling phase and propagate_one_distance. save the region of interest (subarray) to the buffer.
        cpu_phase = lambda z: np.exp(-I*t*z*r)
        for n,z in enumerate(distances):
            if z > upperlimit: print "propagation distance (%s) exceeds accuracy upperlimit (%s)"%(z,upperlimit)
            if not silent:
                sys.stdout.write("\rpropagating: %1.2e m (%02d/%02d)" % (z, n+1, numfr))
                sys.stdout.flush()
            phase_z = cpu_phase(z)
            back = np.fft.ifft2(f*phase_z)
            store[n] = back[sr[0]:sr[1],sr[2]:sr[3]]
    
        if not silent:
            print ""
    
        if im_convert:
            
            # now convert frames to pil objects
            import scipy.misc as smp
            import io
            images = []
            for frame in store:
                image = io.complex_hsv_image(frame)
                images.append(smp.toimage(image))
                
            return store, images
        
        if not im_convert: return store

def apodize(data_in, sigma=3, threshold = 0.01):
    
    """ Apodizes a 2d array to suppress Fresnel ringing from an aperture
    during back-propagation. The algorithm is straightforward: a binary
    image of the aperture (data_in) is convolved with a gaussian with stdevs
    (sigma, sigma). The convolution is then scaled so that it remains 1 at the
    center of data_in and approaches 0 at the edge.
    
    Required arguments:
        data_in: should be a 2d numpy array of the data to be apodized. If
            it is the raw data and not a binary image of the aperture, the optional
            argument threshold should also be supplied.
        
    Optional arguments:
        sigma: degree of blurring (pixels)
        threshold: the data must be binarized before the apodizer is calculated.
            This is done by numpy.where(data_in/data_in.max > threshold, 1, 0)
        
    Returns:
        0. The apodizing filter
        1. The apodized data.
        If data_in is a binarized image of the aperture, 0 and 1 should be
        identical.
    """
    
    # check types
    assert isinstance(data_in,np.ndarray),    "data must be an array"
    assert data_in.ndim == 2,                 "data must be 2d"
    
    try: sigma = float(sigma)
    except: print "couldnt cast sigma to float"; exit()
    
    try: threshold = float(threshold)
    except: print "couldnt cast threshold to float"; exit()
    
    ad    = np.abs(data_in)
    data2 = np.where(ad/ad.max() > threshold, 1, 0)
    
    import math
    sig2  = math.floor(sigma)+1
    
    # ensure that the array is even
    r0, c0 = data2.shape
    if r0%2 == 1: data2 = np.vstack([data2,np.zeros((1,data2.shape[1]),data2.dtype)])
    if c0%2 == 1: data2 = np.hstack([data2,np.zeros((data2.shape[0],1),data2.dtype)])
    
    # now pad the array sufficiently to ensure cyclic boundary conditions
    # don't matter (much)
    r1, c1 = data2.shape
    r_pad  = np.zeros((2*sig2,data2.shape[1]),data2.dtype)
    data2  = np.vstack([data2,r_pad])
    data2  = np.vstack([r_pad,data2])
    c_pad  = np.zeros((data2.shape[0],2*sig2),data2.dtype)
    data2  = np.hstack([data2,c_pad])
    data2  = np.hstack([c_pad,data2])
    c = lambda x,y: np.fft.ifft2(np.fft.fft2(x)*np.fft.fft2(y))
    
    # convolve
    kernel     = np.fft.fftshift(shape.gaussian(data2.shape,(sigma,sigma)))
    convolved  = np.abs(c(kernel,data2))
    convolved *= data2
    
    # rescale
    min_val    = convolved[np.nonzero(convolved)].min()
    convolved -= min_val*data2
    convolved /= convolved.max()
    
    # slice out the original, unpadded region
    convolved  = convolved[2*sig2:2*sig2+r0,2*sig2:2*sig2+c0]

    # return
    return convolved, data_in*convolved


def apodize_old(data_in,kt=.1,threshold=0.01,sigma=2,return_type='data'):
    """ Apodizes a 2d array so that upon back propagation ringing from the
    aperture is at least somewhat suppressed. The steps this function follows
    are as follows:
        # 1. unwrap the data
        # 2. find the boundary at each angle
        # 3. do a 1d blur along each angle
        # 4. rewrap

    Required arguments:
        data_in: 2d ndarray containing the data to be apodized. This data should
        be been sliced so that the only object is the target data.
    
    Optional arguments:
        kt: strength of apodizer. Default is kt=0.1. kt=1.0 is VERY strong.
        threshold: float or int value defines the boundary of the data.
        sigma: boundary locations are smoothed to avoid with jagged edges;
            this sets  the smoothing. float or int.
        return_type: Determines what can be returned.  This can be 'data'
            (default), 'flter', which returns the flter or 'all', which
            returns three items: (data, flter, boundary)
        
    Returns:
        The return value depends on return_type, but the default is 'data'
            which returns the apodized array. Other options are 'flter' or
            'all'.
    """

    data = np.copy(data_in)
    
    # check types
    assert isinstance(data,np.ndarray),    "data must be an array"
    assert data.ndim == 2,                    "data must be 2d"
    assert isinstance(kt,(int,float)),        "kt must be float or int"
    assert isinstance(sigma,(int,float)),     "sigma must be float or int"
    assert isinstance(threshold,(int,float)), "threshold must be float or int"
    
    # if the data is complex, operate only on the magnitude component
    was_complex = False
    if np.iscomplexobj(data):
        phase = np.angle(data)
        data  = np.abs(data)
        was_complex = True
        
    convolve = lambda x,y: np.fft.ifft(np.fft.fft(x)*np.fft.fft(y))

    # select a subregion of data corresponding to a bounding box; embed
    # it in a padding region of zeros. make sure it is square!
    import masking, math
    bbox    = masking.bounding_box(data)
    sliced  = data[bbox[0]:bbox[1],bbox[2]:bbox[3]]
    
    r, c    = sliced.shape
    L       = max(r,c)
    if L%2 == 1: L += 1
    r2 = (L+8)/2-math.floor(r/2)
    c2 = (L+8)/2-math.floor(c/2)
    
    embed = np.zeros((L+8,L+8),sliced.dtype)
    embed[r2:r2+r,c2:c2+c] = sliced

    # find the center of mass
    N,M       = embed.shape
    rows,cols = np.indices((N,M),float)
    av_row    = np.sum(embed*rows)/np.sum(embed)
    av_col    = np.sum(embed*cols)/np.sum(embed)

    # determine the maximum unwrapping radius
    r, c = embed.shape
    R = int(min([av_row,r-av_row,av_col,c-av_col]))
    
    # unwrap the data
    unwrap_plan = wrapping.unwrap_plan(0,R,(av_col,av_row)) # very important to set r = 0!
    unwrapped   = wrapping.unwrap(embed,unwrap_plan)
    ux          = unwrapped.shape[1]
    u_der       = unwrapped-np.roll(unwrapped,-1,axis=0)
    
    # at each column, find the edge of the data by stepping along rows until the object
    # is found. any ways to make this faster?
    threshold = float(threshold)
    boundary  = np.zeros(ux,float)
    indices   = np.arange(R-1)
    for col in range(ux):
        u_der_col = u_der[:,col]
        ave = np.sum(u_der_col[:-1]*indices)/np.sum(u_der_col[:-1])
        boundary[col] = ave+1

    # smooth the edge values by convolution with a gaussian
    kernel = np.fft.fftshift(shape.gaussian((ux,),(sigma,),normalization=1.0))
    boundary = np.abs(convolve(boundary,kernel))-1
    
    # now that the coordinates have been smoothed, build the flter as a series of 1d flters along the column (angle) axis
    x = np.outer(np.arange(R),np.ones(ux)).astype(float)/boundary
    flter = _apodization_f(x,kt)
    
    # rewrap the flter. align the filter to the data
    rplan  = wrapping.wrap_plan(0,R)
    flter  = wrapping.wrap(flter,rplan)
    
    e_flter = np.zeros_like(embed).astype(np.float32)
    e_flter[0:flter.shape[0],0:flter.shape[1]] = flter
    
    data_mask  = np.where(embed > 1e-6,1,0)
    flter_mask = np.where(e_flter > 1e-6,1,0)
    
    rolls  = lambda d, r0, r1: np.roll(np.roll(d,r0,axis=0),r1,axis=1)
    coords = conditioning.align_frames(flter_mask,align_to=data_mask,return_type='coordinates')[0]
    flter  = rolls(e_flter,coords[0],coords[1])

    if was_complex: data *= np.exp(complex(0,1)*phase)
    
    # return a fltered version of the data.
    if return_type == 'filter': return flter
    if return_type == 'data':   return flter*data
    if return_type == 'all':    return flter*data,flter,boundary
    
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
    assert isinstance(data, np.ndarray),               "data must be ndarray"
    assert data.ndim in (2,3),                            "data must be 2d or 3d"
    assert method in ('sobel','roll'),                    "unknown derivative method"
    assert isinstance(exponent,(int, float)),             "exponent must be float or int"
    assert isinstance(normalized, type(True)),            "normalized must be bool"
    assert isinstance(mask, (type(None), np.ndarray)), "mask must be None or ndarray"
    
    import masking
    if data.ndim == 2: data.shape = (1,data.shape[0],data.shape[1])
    if mask == None: mask = np.ones(data.shape[1:],float)
    bounds = masking.bounding_box(mask)

    # calculate the acutance
    import time
    acutance_list = []
    for n,frame in enumerate(data):
        acutance_list.append(_acutance_calc(np.abs(frame).real,method,normalized,mask,exponent,bounds))
        
    # return the calculation
    return acutance_list
            
def _acutance_calc(data,method,normalized,mask,exponent,bounds=None):
    assert data.ndim == 2, "data is wrong ndim"

    if bounds != None:
        data = data[bounds[0]:bounds[1],bounds[2]:bounds[3]]
        mask = mask[bounds[0]:bounds[1],bounds[2]:bounds[3]]

    if method == 'sobel':
        from scipy.ndimage.filters import sobel
        dx = np.abs(sobel(data,axis=-1))
        dy = np.abs(sobel(data,axis=0))
                
    if method == 'roll':
        dx = data-np.roll(data,1,axis=0)
        dy = data-np.roll(data,1,axis=1)

    gradient = np.sqrt(dx**2+dy**2)**exponent
            
    a = np.sum(gradient*mask)
    if normalized: a *= 1./np.sum(data*mask)
    return a

def _apodization_f(x,kt):
    F = 1./(np.exp((x-1)/kt)+1)
    F += -.5
    F[F < 0] = 0
    m = np.max(F,axis=0)
    F *= 1./m
    return F

