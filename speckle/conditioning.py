""" Helper functions which are intended to beat on raw data until it is suitable
for proper analysis. The goal is to be able to transparently handle experimental
and simulated data through the same analytical functions.

Author: Daniel Parks (dhparks@lbl.gov)
Author: Keoki Seu (kaseu@lbl.gov)
"""
import numpy as np
from . import masking, io, shape, crosscorr

def remove_dust(data, dust_mask, dust_plan=None):
    """ Attempts to remove dust through with grid_data, an interpolation
    routine for irregularly spaced data.
    
    Arguments
        data   - data with dust marks. if 3d, each frame gets dedusted.
        dust_mask - a mask which shows dust locations.
        dust_plan - dust_mask gets turned into dust_plan, which is what is
            actually used to remove the dust. If not supplied, gets created
            and returned with the dedusted data. Creation is expensive, so
            it makes sense to reuse the plan when feasible.
            
    Returned:
        dedusted - data with dust removed
        dust_plan - plan for removing dust
    """

    def _check_types(data_in, dm, dp):
        """ Type check helper """
        e1 = "input must be array; is %s"
        e2 = "input must be array (or None), is %s"
        assert isinstance(data_in, np.ndarray), e1%type(data_in)
        assert isinstance(dm, np.ndarray), e1%type(dm)
        assert isinstance(dp, (type(None), np.ndarray)), e2%type(dp)
        
        assert data_in.ndim in (2, 3), "data_in must be 2d or 3d."
        assert dm.ndim == 2, "dust_mask must be 2d"
        assert dm.shape[0] == dm.shape[1], "need square arrays"
        if isinstance(dp, np.ndarray):
            assert dm.shape == dp.shape, "mask and plan must be same shape"

    try:
        import numexpr
        have_numexpr = True
    except ImportError:
        have_numexpr = False
        
    data_in = np.copy(data)
        
    # check types
    _check_types(data_in, dust_mask, dust_plan)
    
    # check that dust_mask and data_in are commensurate
    was_2d = False
    if data_in.ndim == 2:
        was_2d = True
        data_in.shape = (1,)+tuple(data_in.shape)
    assert data_in.shape[1:] == dust_mask.shape, \
    "data_in, dust_mask shapes differ: %s %s"%(data_in.shape, dust_mask.shape)
    dust_mask = dust_mask.astype('>f4')

    # if the dust_plan doesnt exist, make it.
    if dust_plan == None:
        dust_plan = plan_remove_dust(dust_mask)

    # loop through the frames of data_in, removing the dust from each using the
    # same dust plan. interpolation is handled by grid_data from scipy.
    from scipy.interpolate import griddata
    y_points, x_points = np.nonzero(dust_plan)
    pts = np.array([y_points, x_points])
    
    for n, frame in enumerate(data_in):
        
        interpolated = np.zeros_like(frame)
        
        # pull the values out of frame into a format usable by grid_data
        vals = np.zeros_like(y_points).astype(data_in.dtype)
        for m in range(len(y_points)):
            vals[m] = frame[y_points[m], x_points[m]]

        # figure out what region we will actually interpolate over
        ymin = y_points.min()
        ymax = y_points.max()
        xmin = x_points.min()
        xmax = x_points.max()
        gy, gx = np.mgrid[ymin-1:ymax+1, xmin-1:xmax+1]

        # do the interpolation and put it into interpolated_data. 'linear'
        # method gives best results for test data AND is fastest.
        tmp = griddata(pts.transpose(), vals, (gy, gx), method='linear')
        interpolated[ymin-1:ymax+1, xmin-1:xmax+1] = tmp
        interpolated = np.nan_to_num(interpolated)
        
        if have_numexpr:
            fill = numexpr.evaluate("interpolated*dust_mask+frame*(1-dust_mask)")
        else:
            fill = interpolated*dust_mask+frame*(1-dust_mask)
        
        data_in[n] = fill
        
    if was_2d:
        data_in.shape = data_in.shape[1:]
        
    return data_in, dust_plan
        
def plan_remove_dust(mask):
    
    """ Make a plan for the dust removal by convolutions """
    
    def _expand(x):
        """ Convolution helper """
        f1 = df*kf**x 
        f2 = df*kf**(x-1)
        r1 = np.fft.ifft2(f1).real
        r2 = np.fft.ifft2(f2).real
        return np.clip(r1, 0., 1.01)-np.clip(r2, 0., 1.01)
    
    assert isinstance(mask, np.ndarray), "mask must be array"
    assert mask.ndim == 2, "mask must be 2d"
    assert mask.shape[0] == mask.shape[1], "mask must be square"

    # first, clip the mask to the minimum size needed to bound the region
    # identified as having dust.
    bb = masking.bounding_box(mask, force_to_square=True, pad=10)
    mask2 = mask[bb[0]:bb[1], bb[2]:bb[3]]
    
    # define the expand-by-1 kernel
    kernel = np.zeros(mask2.shape, np.float32)
    kernel[0:2, 0:2] = 1
    kernel[-1:, 0:2] = 1
    kernel[-1:, -1:] = 1
    kernel[0:2, -1:] = 1
     
    # precompute the fourier transforms for the convolutions
    kf = np.fft.fft2(kernel)
    df = np.fft.fft2(mask2)

    # now expand by various amount of pixels
    plan = np.zeros(mask2.shape, np.float32)
    for x in (1, 4, 7, 10):
        plan += _expand(x)
    
    # restore to original size and position.
    plan2 = np.zeros(mask.shape, np.uint8)
    plan2[bb[0]:bb[1], bb[2]:bb[3]] = plan.astype(np.uint8)
    
    return plan2

def subtract_dark(data_in, dark, match_region=20, return_type='data'):
    """Subtract a background file. The data and dark files may have been
    collected under different acquisition parameters so this tries to scale
    the dark file appropriately in the region [:x,:x].

    arguments:
        data - data to subtract.
        dark - dark file. Must be an ndarray or None.  Defaults to None.  In
            this case only the DC component is subtracted.
        x - amount from the data edge that should be used for counts matching.
        return_type - (optional) default is 'data', which retur

    returns:
        data - the background-subtracted data.
        parameters - if return_type == 'all', this functions returns a tuple
            (data, parameters) where data is the background-subtracted data and
            parameters are the fit parameters.
    """

    # check types
    assert isinstance(data_in, np.ndarray), "data must be an array"
    assert data_in.ndim in (2, 3), "data must be 2d or 3d"
    assert isinstance(dark, np.ndarray), "dark must be an array"
    assert dark.ndim == 2, "dark must be 2d"
    assert data_in.shape[-2:] == dark.shape, "dark and data must be same size"

    if isinstance(match_region, int):
        x = match_region
        match_region = np.zeros(dark.shape, np.uint8)
        match_region[:x, :x] = 1
        
    if isinstance(match_region, str):
        assert match_region.split('.')[-1] == '.reg'
        match_region = io.open(match_region, force_reg_size=data_in.shape)
        match_region = match_region.astype(np.bool)
    
    data = np.copy(data_in)
    
    was2d = False
    if data.ndim == 2:
        data.shape = (1,)+data.shape
        was2d = True
    
    for n, frame in enumerate(data):
        scaled_dark, x = match_counts(frame, dark,
                                      region=match_region, return_type='all')
        frame = np.abs(frame-scaled_dark)
        data[n] = frame
        
    if was2d: data = data[0]
    
    if return_type == 'data':
        return data
    if return_type == 'all':
        return (data, x)
    
    return data

def remove_hot_pixels(data_in, iterations=1, threshold=2, gpu_info=None):
    """Uses np.medfilt to define hot pixels as those which exceed a certain
    multiple of the local median and remove them by replacing with the median of
    the nearest neighbors.

    Required:
        data_in - 2d or 3d array from which hot pixels will be removed
        
    Optional:
        iterations - number of iterations to run the smoother. Default is 1.
        threshold - threshold to specify when pixels are hot. When a pixel has
            value greater than threshold*median, it is replaced by the median.
            Default is 2.
        gpu_info - if supplied, the algorithm will run on the gpu. this provides
            a substantial speed boost.

    Returns:
        data - The data wih the hot pixels that meet the threshold removed.
    """

    from scipy.signal import medfilt
    data = np.copy(data_in)
    
    # check types
    assert isinstance(data, np.ndarray), "data must be ndarray"
    assert data.ndim in (2, 3), "data must be 2d or 3d"
    assert isinstance(iterations, int) and iterations > 0, \
    "number of iterations must be integer > 0"
    assert isinstance(threshold, (int, float)), \
    "threshold must be float or int"

    was_2d = False
    if data.ndim == 2:
        was_2d = True
        data.shape = (1, data.shape[0], data.shape[1])
        
    # set up gpu stuff
    if gpu_info != None:

        failed = False
        
        # load libraries
        try:
            import gpu
            import string
            import pyopencl as cl
            import pyopencl.array as cla
            context, device, queue, platform = gpu_info
            
            if not gpu.valid(gpu_info):
                print "malformed gpu info in remove_hot_pixels, using cpu"
                failed = True

        except ImportError:
            print "couldnt load gpu libraries, falling back to cpu"
            failed = True
            
        def _build_helper(name):
            """ Wrap build kernel file """
            return gpu.build_kernel_file(context, device, kp+name)
            
        # build kernels
        try:
            kp = string.join(gpu.__file__.split('/')[:-1], '/')+'/kernels/'
            gpu_median3 = _build_helper('medianfilter3_ex.cl')
            #gpu_median5 = _build_helper('medianfilter5.cl')
            gpu_hotpix = _build_helper('remove_hot_pixels.cl')
            
        except:
            print "error building kernels"
            failed = True
            
        if failed:
            remove_hot_pixels(data_in)
        
        # allocate memory
        gpu_data_hot = cla.empty(queue, data[0].shape, np.float32)
        gpu_data_cold = cla.empty(queue, data[0].shape, np.float32)

    # for each frame in data, run the hot pixel remover the number of times
    # specified by iterations
    for z, frame in enumerate(data):
        
        if gpu_info != None:
            gpu_data_hot.set(frame.astype(np.float32))

        for m in range(iterations):
            
            if gpu_info != None:
                
                gpu_median3.execute(queue, frame.shape, (16, 16),
                                    gpu_data_hot.data, gpu_data_cold.data,
                                    cl.LocalMemory(18*18*4))
                
                gpu_hotpix.execute(queue, (frame.size,), None,
                                   gpu_data_hot.data, gpu_data_cold.data,
                                   np.float32(threshold))
            else:
                median = medfilt(frame)+.1
                frame = np.where(frame/median > threshold, median, frame)
 
        if gpu_info != None:
            frame = gpu_data_hot.get()
        data[z] = frame

    if was_2d:
        data = data[0]
    return data
    
def align_frames(data, align_to=None, region=None, use_mag_only=False, return_type='data'):
    """ Align a set of data frames by FFT/cross-correlation method.

    Inputs:
        data - A 2d or 3d ndarray.

    Optional arguments:
        align_to - A 2d array used as the alignment reference. If None, data
            must be 3d and the first frame of data will be the reference.
            Default is None
        region - A 2d mask which specifies which data to use in the cross
            correlation. If None, all pixels will contribute equally to the
            alignment. Default is None.
        use_mag_only - Align using only the magnitude component of data.
            Default is False.
        return_type - align_frames is called from multiple places, and
            expectations of what is returned vary. Returned can be aligned
            data, aligned and summed data, or just the alignment coordinates;
            keywords for these are 'data', 'sum', and 'coordinates',
            respectively. Default is 'data'.

    Returns:
        result - The result depends on the input return_type. If return_type is
        set to:
            'data' - An aligned array of shape and dtype identical to data, or
            'sum' - the the summed array, or
            'coordinates' - the coordinates that the arrays need to be rolled
                in order to align them. This is a ndarray of dimension (fr, 2).
    """
    # check types
    assert isinstance(data, np.ndarray), \
    "data to align must be an array"
    
    assert isinstance(align_to, (type(None), np.ndarray)), \
    "align_to must be an array or None"
    
    assert isinstance(region, (type(None), np.ndarray)),\
    "region must be an array or None"
    
    assert use_mag_only in (0, 1, True, False), \
    "use_mag_only must be boolean-evaluable"
    
    assert return_type in ('data', 'sum', 'coordinates'), \
    "return_type must be 'data', 'sum', or 'coordinates'; 'data' is default"
    
    if data.ndim == 2:
        assert isinstance(align_to, np.ndarray), \
        "data is 2d; need an explicit alignment reference"

    # define some simple helper functions to improve readability
    rolls = lambda d, r0, r1: np.roll(np.roll(d, r0, axis=0), r1, axis=1)
    def prep(tmp):
        if use_mag_only:
            tmp = np.abs(tmp)
        if region != None:
            tmp = region*tmp
        return tmp

    # cast 2d to 3d so the loops below are simpler
    was_2d = False
    if data.ndim == 2:
        was_2d = True
        data.shape = (1, data.shape[0], data.shape[1])

    frames, rows, cols = data.shape
    
    # check some more assumptions
    if region != None:
        assert region.shape == (rows, cols), \
        "region, data frames must be same shape"

    # set up explicit align_to in case of None
    if align_to == None:
        align_to = data[0]
    else:
        assert align_to.shape == (rows, cols), \
        "align_to, data frames must be same shape"
    
    # for speed, precompute the reference dft
    ref = np.fft.fft2(prep(align_to))
    
    # get the alignment coordinates for each frame in data by the
    # argmax of the cross correlation with the reference
    coordinates = np.zeros((frames, 2), int)
    for n, frame in enumerate(data):
        coordinates[n] = crosscorr.alignment_coordinates(prep(frame),
                                                         ref, already_fft=(1,))
        
    # now return the data according to return_type
    if return_type == 'coordinates':
        if was_2d:
            data.shape = (rows, cols)
        return coordinates
    
    if return_type == 'sum':
        # Create a new array instead of modifying data in-place.
        # Modifying data in-place is a bad idea
        result = np.zeros((rows, cols), dtype=data.dtype)
        for n in range(frames):
            rr, rc = coordinates[n]
            result += rolls(data[n], rr, rc)
            if was_2d:
                data.shape = (rows, cols)
        return result
    
    if return_type == 'data':
        # Create a new array instead of modifying data in-place
        result = np.zeros_like(data)
        for n in range(frames):
            rr, rc = coordinates[n]
            result[n] = rolls(data[n], rr, rc)
        if was_2d:
            data.shape = (rows, cols)
            result = result[0]
        return result

def match_counts(img1, img2, region=None, nparam=3, silent=True, return_type='data'):
    """ Match the counts between two images. There are options to match in a
        region of interest and the number of fitting parameters to be used.

    arguments:
        img1 - first image to match. Must be 2d and the same size as img2.
        img2 - second image to match. Must be 2d and the same size as img1.
        region - mask image that is >=1 in the pixels/areas that are to be used
            for matching.  Defaults to None, which uses the entire image.
        nparam - Number of fitting parameters used to match counts. The fit
            function used depends on nparam:
                1 - min(img1 - s*img2) with one parameter, s
                2 - min(img1 - s*(img2 - d2)), with two parameters, (s, d2)
                3 - min(img1 - s*img2 + (s*d2 - d1)) with (s, d1, d2)
            The default is 3 parameters, which accounts for different
            backgrounds of the two images (d1, d2) and scaling s.

    returns:
        img2 - a scaled img2 such that the counts in region match.
    """
    
    import scipy.optimize
    
    try:
        import numexpr
        have_numexpr = True
    except ImportError:
        have_numexpr = False
        
    def diff3(c, img1, img2):
        """ minimize (I1 - d1) - s(I2-d2)
            = I1 - s*I2 + (s*d2 - d1)
        """
        (s, d1, d2) = c
        if d1 < 0 or d2 < 0:
            return 1e30
        if have_numexpr:
            dif = numexpr.evaluate("(img1 - d1 - s*(img2 - d2))**2")
            dif = numexpr.evaluate("sum(dif)")
        else:
            dif = ((img1 - d1 - s*(img2 - d2))**2).sum()
        return dif

    def diff2(c, img1, img2):
        """ minimize I1 - s(I2-d2)
            = I1 - s*(I2 - d2)
        """
        (s, d2) = c
        if d2 < 0:
            return 1e30
        if have_numexpr:
            dif = numexpr.evaluate("(img1 - s*(img2 - d2))**2")
            dif = numexpr.evaluate("sum(dif)")
        else:
            dif = ((img1 - s*(img2 - d2))**2).sum()
        return dif

    def diff1(c, img1, img2):
        """ minimize I1 - s*I2
            = I1 - s*I2
        """
        (s) = c
        if have_numexpr:
            dif = numexpr.evaluate("(img1 - s*img2)**2")
            dif = numexpr.evaluate("sum(dif)")
        else:
            dif = ((img1 - s*img2)**2).sum()
        return dif

    assert isinstance(img1, np.ndarray) and img1.ndim == 2, \
    "img1 must be a 2d ndarray."
    
    assert isinstance(img2, np.ndarray) and img2.ndim == 2, \
    "img2 must be a 2d ndarray."
    
    assert img1.shape == img2.shape, \
    "(img1, img2) must be the same shape"

    if not (type(nparam) == int and nparam in (1, 2, 3)):
        nparam = 3 

    if type(region) != np.ndarray and region == False:
        region = np.ones_like(img1)
    else:
        # convert region to 1/0 just to be sure
        region = np.where(region >= 1, 1, 0)
    if region.sum() < 1e5:
        have_numexpr = False 

    if region.sum() == 0:
        print("***** match_counts: Region of interest is empty! *****")
        return img2

    d1 = np.average(img1[:, 0])
    d2 = np.average(img2[:, 0])
    try:
        c0guess = (img1-d1)/(img2-d2)
    except RuntimeWarning:
        pass
    
    # remove the infinities and nans.  we get these from img2-d2 =0
    c0guess = np.where(np.isfinite(c0guess), c0guess, 0)

    s = (c0guess*region).sum()/region.sum()

    if nparam == 1:
        c = np.array([s])
        diff = diff1
        paramstr = "s=%1.2f"
        funcstr = "img1-s*img2 (%d parameter)" % nparam
        
    elif nparam == 2:
        c = np.array([s, d2])
        diff = diff2
        paramstr = "s=%1.2f, d2=%1.2f"
        funcstr = "img1-s*(img2-d2) (%d parameters)" % nparam
        
    else: # assume 3 parameter fit
        c = np.array([s, d1, d2])
        diff = diff3
        paramstr = "s=%1.2f, d1=%1.2f, d2=%1.2f"
        funcstr = "(img1-d1)-s*(img2-d2) (%d parameters)" % nparam

    if not silent:
        print("minimizing %s.\nInitial guess: %s." % (funcstr, paramstr % tuple(c)))
    
    # optimize only in region; this saves the time required to multiply and
    # sum and consider all the zeros outside the region
    img1_shrunk = masking.take_masked_pixels(img1, region)
    img2_shrunk = masking.take_masked_pixels(img2, region)

    x = scipy.optimize.fmin(diff, c,
                            args=(img1_shrunk, img2_shrunk), disp=False)

    if not silent:
        print("Final result: %s." % (paramstr % tuple(x)))

    if nparam == 1:
        data_return = x[0]*img2
    elif nparam == 2:
        data_return = x[0]*(img2-x[1])
    else:
        data_return = x[0]*(img2-x[2])+x[1]
        
    if return_type == 'data':
        return data_return
    if return_type == 'all':
        return (data_return, x)

def open_dust_mask(path):
    """ Open a dust mask.

    arguments:
        path - the path to a dust mask

    returns:
        mask - the opened dust mask
    """
    
    assert isinstance(path, str)
    pathsplit = path.split('.')
    assert len(pathsplit) >= 2, "dust mask path has no file extension"
    ext = pathsplit[-1]
    
    assert ext in ('fits', 'png', 'gif', 'bmp'), \
    "dust mask file extension %s not recognized"%ext
        
    if ext == 'fits':
        mask = io.openfits(path).astype('float')
    else:
        mask = np.flipud(io.openimage(path)).astype('float')
        mask = np.where(mask > .1, 1, 0)
    assert mask.ndim == 2, "mask must be 2d"
    return mask

def find_center(data, return_type='coords'):
    """ Tries to find the center of a speckle pattern that has inversion
    symmetry where the natual center (direct beam) has been blocked.  An
    example of this situation is a speckle pattern from labyrinth magnetic
    domains.

    arguments:
        data -- data whose center is to be found
        return_type -- What type of data to return.  If 'data', data is
            returned in human-centered form. If 'coords', the coordinates
            of the center of inversion are returned. Default is 'coords'.

    returns:
        depending on return_type, can return various:
            'coords' (dflt) -- returns center coordinates in (row, col) format.
            'data' -- returns centered data.
    """
    assert isinstance(data, np.ndarray) and data.ndim == 2, \
    "data must be a 2-dimensional ndarrray"
    
    assert return_type in ('data', 'coords'), \
    "return_type must be 'data' or 'coords'."
    
    rolls = lambda d, r0, r1: np.roll(np.roll(d, r0, axis=0), r1, axis=1)
    
    # pass to align_frames both the image to be centered and a version
    # rotated 180 degrees. coordinates to shift to center are 1/2
    # coordinates to align the images
    rotated = np.rot90(np.rot90(data))
    r0, r1 = align_frames(rotated, align_to=data, return_type='coordinates')[0]

    r0 = -r0*0.5
    r1 = -r1*0.5

    if return_type == 'data':
        return rolls(data, int(r0), int(r1))
    if return_type == 'coords':
        return int(data.shape[0]/2.0-r0), int(data.shape[1]/2.0-r1)

def merge(data_to, data_from, fill_region, fit_region=None, width=10, align=False):
    """ Merge together two images (data_to, data_from) into a single
    high-dynamic-range image. A primary use of this function is to stitch a
    pair of images taken in transmission: one with the blocker in, one with the
    the blocker out.
    
    Merging follows two steps: count levels are matched in a selectable region,
    and then counts are smoothly blended between the two images.
    
    It is assumed that the two images have been properly aligned before being
    passed to this function.
    
    arguments:
        data_to -- data will be copied "into" this image. Experimentally,
            this corresponds to the image with the blocker at center.
        data_from -- data will be copied "from" this image. Experimentally,
            this corresponds to the image with the blocker out of the way.
        fill_region -- an array or path to an array or ds9 file. fill_region
            describes which pixels in data_to will be replaced entirely by
            pixels in data_from. Outside the fill region there is a transition
            zone where the output is a weighted average of data_from and
            data_to. Often, fill_region is a binary mask of the blocker.
        fit_region -- (optional) an array or path to an array or ds9 region
            file. fit_region describes where the counts should be compared.
            If fit_region is None, count matching is skipped.
        width -- (optional) Sets a width for the blending-transition region
            outside fill_region. Larger width creates a broader transition
            between data_to and data_from. Numerically, width is the standard
            deviation of a gaussian used to convolve fill_region. 10 by
            default, but for better merging of thin features such as the
            blocker wire a smaller value may be appropriate.  If width <= 0,
            a hard merge is conducted (no blending).
        align -- (optional) If true, attempts to align data_to and
            data_from through a cross-correlation within fit_region. Requires
            fit_region to be specified. Generally, this method has a good
            chance of success only when the misalignment of the dataset is
            minor. For more extreme misalignment, the user must perform a
            coarse alignment prior to merging.
             
    returns:
        an array with data smoothly blended between data_to and data_from.
    """
    # check types
    assert isinstance(data_to, np.ndarray) and data_to.ndim == 2,\
    "data_to must be 2d array"
    
    assert isinstance(data_from, np.ndarray) and data_from.ndim == 2,\
    "data_from must be 2d array"
    
    assert data_to.shape == data_from.shape, \
    "data_to and data_from must be same shape"
    
    assert isinstance(fit_region, (np.ndarray, str, type(None))), \
    "fit_region must be an array or a path to an array"
    
    assert isinstance(fill_region, (np.ndarray, str)), \
    "fill_region must be an array or a path to an array"
    
    assert isinstance(width, (int, float)), \
    "width must be float or int"

    def _make_blender(fill_region, width):
        """ Fast blender generation from fill_region. Helper function for
        merge. Returns a blender array that is 0 in fill_region with a gradual
        transition to 1 outside.  If width <= 0, do a hard merge without
        blending.
        """
        if width <= 0: # Do a hard merge if width <= 0
            return np.where(fill_region, 0, 1)

        convolve = lambda x, y: np.fft.ifft2(np.fft.fft2(x)*np.fft.fft2(y))
        shift = np.fft.fftshift
    
        # to speed up the calculation, only do the convolutions in the pixels
        # closely surrounding fill_region
        r0, r1, c0, c1 = masking.bounding_box(fill_region,
                                              force_to_square=True,
                                              pad=int(5*width))
        bounded = fill_region[r0:r1, c0:c1]
        
        # define convolution kernels.
        grow_kernel = shift(shape.circle(bounded.shape, 2*width, AA=0))
        blur_kernel = shift(shape.gaussian(bounded.shape, (width, width),
                                           normalization=1.0))
    
        # two convolutions make the blender
        expanded = np.clip(np.abs(convolve(bounded, grow_kernel)), 0, 1)
        blurred = 1-np.abs(convolve(expanded, blur_kernel))

        # embed the bounded convolutions inside the correct spot in an array of
        # the correct size to return and set pixels inside fill_region to
        # exactly 0 rathern than some small decimal.
        blender = np.ones(fill_region.shape, np.float32)
        blender[r0:r1, c0:c1] = blurred*(1-bounded)

        return blender
        

    # open merge and fit regions if necessary 
    if isinstance(fill_region, str):
        fill_region = io.open(fill_region, force_reg_size=data_to.shape)
        
    if isinstance(fit_region, str):
        fit_region = io.open(fit_region, force_reg_size=data_to.shape)
        
    # make the blender
    assert fill_region.shape == data_to.shape, \
    "fill_region and data must be same shape"
    
    blender = _make_blender(fill_region, width)
    
    # if requested, align the data
    if align:
        rolls = lambda d, r0, r1: np.roll(np.roll(d, r0, axis=0), r1, axis=1)
        
        # slice the data out of fit_region
        bb = masking.bounding_box(fit_region)
        d1 = data_from[bb[0]:bb[1], bb[2]:bb[3]]
        d2 = data_to[bb[0]:bb[1], bb[2]:bb[3]]
        d3 = np.array([d1, d2])
        
        # align using cross-correlation method
        coords = align_frames(d3, return_type='coordinates')[1]
        data_to = rolls(data_to, coords[0], coords[1])
    
    # scale the data to reconcile acquisition times etc
    if fit_region != None:
        assert fit_region.shape == data_to.shape, \
        "fit_region and data must be same shape"
        scaled_from = match_counts(data_to, data_from, region=fit_region)
    else:
        scaled_from = data_from

    # return the merged data
    return data_to*blender+scaled_from*(1-blender) 

def sort_configurations(data, capture=0.5, louvain=False, gpu_info=None):
    
    """ Sort a similarity matrix into configurations. This is used, for example,
    when collecting CDI or FTH frames and the beam is drifting. In these
    circumstances there will be some variability between frames as the speckle
    pattern shifts configurations. This code sorts the speckle patterns into
    configurations using two different algorithms.
    
    The non-default algorithm, specifed by louvain=True, uses graph-theoretical
    tools and requires two additional python packages which can be found
    with google:
    
        1. networkx
        2. community (implements Louvain partitioning method for networkx)
        
    The louvain partitioner used here is tuned to be very aggressive; it should
    always find at least two configurations.
    
    The louvain and default algorithms should give similar results, but will
    probably disagree in the case of elements which could reasonably go into
    two configurations.
    
    Arguments:
    
        weights - a 2d or 3d numpy array. If 2d, the value of entry (i,j)
            corresponds to some similarity metric between frames i and j of the
            experimental data. Such a metric may be obtained through
            speckle.crosscorr.pairwise_covariances or through your own method.
            If 3d, assume that the frame axis shows individual 2d frames, in
            which case the data is passed to crosscorr.pairwise_covariances
            before configurations are sorted.
            Must be real valued.
            
        louvain - optional, default False. If True, will attempt to interpret
            weights as a weighted network and partition it according to the
            Louvain method of community detection in large graphs.
            
        capture - optional, default 0.5. If louvain = False, this sets the
            degree of improvement in the average pair-correlation coefficient
            sought by the configruation sorter. For example, say the average
            correlation is 0.95. This will try to reach a value of
            0.95+capture*(1-0.95) by increasing the number of configurations.
            
        gpu_info - optional, default None. If:
            1. gpu_info is a valid gpu specification obtained by gpu.init()
            2. data is 3d, meaning it must be correlated.
            Then the correlations will be calculated on the gpu.
            
    Returns:
        a list of lists. Each sub-list represents a configuration,
            and the elements of each list are the frame numbers which
            belong to the configuration."""
        
    def _check_types():
        """ Check types """
        assert isinstance(data, np.ndarray), "weights must be an array"
        assert data.ndim in (2, 3), "weights must be 2d"
        assert data.shape[-2] == data.shape[-1], "weights must be square"
        assert louvain in (True, False, 0, 1), "louvain must be boolean"
        if not louvain:
            try:
                float(capture)
            except TypeError:
                e = "cant cast capture to float in sort_configurations; value is %s"
                raise TypeError(e%capture)
        if data.ndim == 3 and gpu_info != None:
            from . import gpu
            assert gpu.valid(gpu_info)
        
    def _sort(weights, capture):
        
        """ Sort using a top-down reverse agglomeration. Initially there is one
        cluster. The worst-correlated entry wrt to the first element is
        used as the start of the second cluster. Elements are transferred from
        the first to second cluster if they correlate better with the anchor.
        This process is repeated until there are N clusters. """
        
        class Configurations(object):
            
            """ Manages the set of configurations; methods
            for sorting """

            def __init__(self):
                """ Pro forma """
                
                # declare self variables
                self.correlations = None
                self.N = None
                self.configs = None
                self.average_corr = None
                self.average_corr2 = None

            def load(self, data):
                """ Get data into the class; make the first Config """
                    
                data = data.astype(np.float32)
                self.correlations = data
                self.N = data.shape[0]
                    
                # make the initial correlation when the new data is opened.
                self.configs = []
                config0 = Config(0)
                config0.members = np.arange(self.N)
                self.configs.append(config0)
                
                # calculate the initial correlation average
                self._average_correlation()
                    
            def sort_iteration(self):
                """ Execute 1 iteration of the splitting-sort """
                
                # first: find the worst member of any configuration.
                # we define "worst" as having the lowest correlation
                # with the anchor.
                worst = 1
                worst_member = None
                worst_config = None
                
                for n, c in enumerate(self.configs):
                    c.find_worst_corr(self.correlations)
                    if c.worst_corr < worst:
                        worst = c.worst_corr
                        worst_member = c.worst_member
                        worst_config = c
                
                # second: split off the worst member and make a new
                # configuration using that member as the anchor.
                worst_config.remove(worst_member)
                new_config = Config(worst_member)
                worst_corrs = self.correlations[worst_member]
                
                for c in self.configs:
                    take = []
                    for m in c.members:
                        if worst_corrs[m] > self.correlations[c.anchor, m]:
                            take.append(m)
                            
                    # add members to new_config; remove from c
                    new_config.members += take
                    c.remove(take)
                        
                new_config.members.append(new_config.anchor)
                new_config.members = np.array(new_config.members)
                self.configs.append(new_config)
                
                # update the average correlation
                self._average_correlation()
                
            def _average_correlation(self):
                """ Compute the average correlation"""
                total_corr = 0
                for c in self.configs:
                    c.get_corrs(self.correlations)
                    total_corr += c.corrs.sum()
                self.average_corr = total_corr/self.N
                lsc = len(self.configs)
                self.average_corr2 = (total_corr-lsc)/(self.N-lsc)
        
        class Config(object):
            
            """ Holds information about a specific configuration's
            membership """
    
            def __init__(self, anchor):
                """ Pro forma """
                self.anchor = anchor
                self.members = []
                self.corrs = None
                self.worst_corr = None
                self.worst_member = None
                
            def find_worst_corr(self, corrs):
                """ Find the worst correlated member in the config;
                used to seed a new config """
                self.get_corrs(corrs)
                self.worst_corr = self.corrs.min()
                self.worst_member = self.members[self.corrs.argmin()]
                
            def get_corrs(self, corrs):
                """ Get an array of the members """
                acorrs = corrs[self.anchor]
                self.corrs = np.array([acorrs[mbr] for mbr in self.members])
        
            def remove(self, which):
                """ Remove a member from the configuration """
                assert isinstance(which, (int, list))
                m = self.members.tolist()
                if isinstance(which, int):
                    m.remove(which)
                if isinstance(which, list):
                    for w in which:
                        m.remove(w)
                self.members = np.array(m)   
    
        # load the weights into the sorter
        c = Configurations()
        c.load(weights)
        
        # each iteration adds a configuration. add configurations until
        # we reach the goal specified by the capture parameter
        goal = c.average_corr+float(capture)*(1-c.average_corr)
        while c.average_corr < goal:
            c.sort_iteration()
        
        return [x.members.tolist() for x in c.configs]
    
    def _louvain(weights):
        
        try:
            import networkx
            import community
        except ImportError:
            print "couldnt import networkx, community for louvain partitioning"
            print "falling back to default method"
            sort_configurations(weights, capture=capture, louvain=False)

        # open the data, then turn it into a network. data is rescaled to force
        # the sorter to find configurations.
        weights = weights.astype(np.float32)
        weights = (weights-weights.min())/(1-weights.min())
        N = weights.shape[0]
        
        # turn the array into a weighted network
        edges = []
        for i in range(N):
            for j in range(i):
                edge = (i, j, weights[i, j])
                edges.append(edge)
        graph = networkx.Graph()
        graph.add_weighted_edges_from(edges)
        
        # run the cluster detection
        clusters = community.best_partition(graph)
        
        # clusters is a dictionary built like frame:configuration.
        # invert so we have a list of lists, each of which is a configuration
        n = len(set(clusters.values()))
        configs = [[] for m in range(n)]
        for key in clusters.keys():
            configs[clusters[key]].append(key)
        
        return configs

    _check_types()
    
    # correlate if necessary
    if data.ndim == 3:
        from . import crosscorr
        weights = crosscorr.pairwise_covariances(data, gpu_info=gpu_info)
        
    # otherwise, just rename the data
    if data.ndim == 2:
        weights = data
        
    if louvain:
        return _louvain(weights)
    else:
        return _sort(weights, capture)
