""" Helper functions which are intended to beat on raw data until it is suitable
for proper analysis. The goal is to be able to transparently handle experimental
and simulated data through the same analytical functions.

Author: Daniel Parks (dhparks@lbl.gov)
Author: Keoki Seu (kaseu@lbl.gov)
"""
import numpy

def remove_dust(data,dust_mask,dust_plan=None):
    """ Attempts to remove dust and burn marks from CCD images by interpolating
    in regions marked as defective in a plan file.
    
    Requires:
        data - the data from which dust will be removed. 2d or 3d ndarray.
        dust_mask - a binary array describing from which pixels dust must be
            removed.

    Optional input:
        plan - generated from dust_mask; this can only be supplied as an
            argument if remove_dust has been previously run and plan returned
            then as output.

    Returns:
        data - the data array with the dust spots removed.
        plan - plan unique to dust_mask which can be passed again for re-use.
    """

    from scipy.signal import cspline1d, cspline1d_eval

    # check initial types
    assert isinstance(data,numpy.ndarray),       "data must be ndarray"
    assert data.ndim in (2,3),                   "data must be 2d or 3d"
    assert isinstance(dust_mask,numpy.ndarray),  "plan must be ndarray"
          
    if dust_plan == None: dust_plan = plan_remove_dust(dust_mask)

    # because this function accepts both 2d and 3d functions the easiest solution is to upcast 2d arrays
    # to 3d arrays with frame axis of length 1
    was_2d = False
    if data.ndim == 2:
        was_2d = True
        data.shape = (1,data.shape[0],data.shape[1])
    Lz, Lr, Lc = data.shape
    
    for z, frame in enumerate(data):
    
        interpolated_values = numpy.zeros_like(frame)

        for n,entry in enumerate(dust_plan):
    
            # warning to self:
            # this code works and relies intimately on the format of what comes out of remove_dust_plan which i've now forgotten. probably i should
            # change that function to return some sort of dictionary but even then: don't change this unless you reunderstand it! 
    
            # decompose the string into the essential information: which row or column to interpolate over, and the bounds of the fill region
            which,slice,splice_min,spline_max = entry.split(',')
            slice,splice_min,spline_max = int(slice),int(splice_min),int(spline_max)
            
            # we only have to interpolate within the local environment of the pixel, not the whole row or col
            step = spline_max-splice_min
            minsteps = min([5,splice_min/step]) # make sure we don't encounter an IndexError by going > L or < 0
            if which == 'r': maxsteps = min([5,(Lr-spline_max)/step])
            if which == 'c': maxsteps = min([5,(Lc-spline_max)/step])
            
            index_min = splice_min-minsteps*step
            index_max = spline_max+maxsteps*step
            indices = numpy.arange(index_min,index_max,step)
                
            # slice the data according to spline orientation
            if which == 'r': data_slice = frame[:,slice]
            if which == 'c': data_slice = frame[slice,:]
    
            # interpolate
            to_fit = data_slice[indices]
            splined = cspline1d(to_fit)
            interpolated = cspline1d_eval(splined,numpy.arange(index_min,index_max,1),dx=step,x0=indices[0])
    
            # copy the correct data into the 3d array of interpolated spline values
            if which == 'c': interpolated_values[slice,splice_min+1:spline_max] = interpolated[splice_min+1-index_min:spline_max-index_min]
            if which == 'r': interpolated_values[splice_min+1:spline_max,slice] = interpolated[splice_min+1-index_min:spline_max-index_min]
           
        # insert the interpolated data
        data[z] = frame*(1.-dust_mask)+dust_mask*interpolated_values

    if was_2d: data = data[0]
    return data, dust_plan

def plan_remove_dust(mask):
    """ Dust removal has requires two pieces of information: a mask describing
    the dust and a plan of operations for doing the spline interpolation within
    the mask. Only the mask is specified by the user. This function generates
    the second piece of information from the mask.
    
    arguments:
        mask -- a binary mask marking the dust as ones, not-dust as zeros
        
    returns
        plan -- a tuple which is given to remove_dust along with the mask to
            remove the dust from the ccd image"""

    mask = mask.astype('int')
    
    # mark the pixels in mask with unique identifiers
    L = len(mask)
    marked = numpy.zeros(L**2)
    marked[:] = numpy.ravel(mask)[:]
    marked *= (numpy.arange(L**2)+1)
    marked = marked[numpy.nonzero(marked)]-1 # locations of all non-zero pixels
    
    PixelIDStrings = []
    
    for value in marked:

        # record only some information about each pixel pertaining to the
        # eventual spline fits. for example, we dont actually need to know both
        # the exact row and column of each pixel, only the range over which the
        # spline will be evaluated. this eliminates duplicate work by avoiding
        # re-interpolation of the same range of pixels. this information is
        # recorded in idstring
        
        idstring = ''
        
        r,c = value/L,value%L # coordinate of an active pixel in mask
        row,col = mask[r],mask[:,c]
        
        # find out how wide the object is in row and col
        r1,r2,c1,c2 = 0,0,0,0
        while row[c+c1] > 0: c1 += 1
        while row[c+c2] > 0: c2 += -1
        while col[r+r1] > 0: r1 += 1
        while col[r+r2] > 0: r2 += -1
        
        rmax = r+r1
        rmin = r+r2
        cmax = c+c1
        cmin = c+c2
        
        # figure out whether we should interpolate this pixel by row or by column
        if rmax-rmin <= cmax-cmin: idstring += 'r,%.4d,%.4d,%.4d'%(c,rmin,rmax)
        if rmax-rmin > cmax-cmin:  idstring += 'c,%.4d,%.4d,%.4d'%(r,cmin,cmax)
        
        # record the essentials about the pixel
        PixelIDStrings.append(idstring)
    
    # return only the unique ID strings to eliminate redundant interpolations.
    # set() returns unique elements of a list without order preservation which
    # doesn't matter for this purpose
    return tuple(set(PixelIDStrings))

def subtract_background(data, dark=None, x=20, scale=1, abs_val=True):
    """Subtract a background file. The DC component of both files is subtracted
    first. the DC component is calculated by averaging pixels 0:x in both
    directions.

    arguments:
        data - data to subtract.
        dark - dark file. Defaults to None.  In this case only the DC component
            is subtracted.
        x - amount from the data edge that should be used for DC component
            averaging. The dc component is average(data[0:x, 0:x]). Defaults to
            20 pixels.
        scale - amount to scale up the dark image before subtraction. Defaults
            to 1 (no scaling).
        abs_val - Weather the absolute value of the data should be returned.
            Defaults to True.

    returns:
        data - the background-subtracted data.
    """

    # check types
    assert isinstance(data,numpy.ndarray), "data must be ndarray"
    assert data.ndim in (2,3), "data must be 2d or 3d"
    assert isinstance(dark,(type(None),numpy.ndarray)), "dark must be None or ndarray"
    if isinstance(dark,numpy.ndarray):
        assert dark.ndim == 2, "dark must be 2d"
        assert data.shape[-2:] == dark.shape, "data and dark must be same shape"
    assert abs_val in (bool, int), "abs_val must be boolean-evaluable"
    
    # subtract DC component from data
    if data.ndim == 2:
        dc = numpy.average(data[0:x,0:x])
        data = abs(data-dc)
    if data.ndim == 3:
        for n in range(data.shape[0]):
            dc = numpy.average(data[n,0:x,0:x])
            data[n] = abs(data[n]-dc)
        
    # dark subtraction can be broadcast to all frames of data so no need to check ndim
    if dark is not None:
        dark = dark-numpy.average(dark[0:x,0:x])
        data = data-dark*scale

    if abs_val:
        return abs(data)
    else:
        return data

def hot_pixels(data, iterations=1, threshold=2):
    """A basic hot pixel remover which uses numpy.medfilt to define hot pixels
    as those which exceed a certain multiple of the local median.
    
    Required:
        data - 2d or 3d array from which hot pixels will be removed
        
    Optional:
        iterations - number of iterations to run the smoother. Default is 1.
        threshold - threshold to specify when pixels are hot. When a pixel has
            value greater than threshold*median, it is replaced by the median.
            Default is 2.

    Returns:
        data - The data wih the hot pixels that meet the threshold removed.
    """
    #  This is slow for large arrays; this operation would probably benefit a great deal from GPU acceleration.

    from scipy.signal import medfilt
    
    # check types
    assert isinstance(data,numpy.ndarray), "data must be ndarray"
    assert data.ndim in (2,3), "data must be 2d or 3d"
    assert isinstance(iterations,int), "number of iterations must be integer"
    assert isinstance(threshold, (int,float)), "threshold must be float or int"
    
    was_2d = False
    if data.ndim == 2:
        was_2d = True
        data.shape = (1,data.shape[0],data.shape[1])
    for z,frame in enumerate(data):

        for m in range(iterations):
            median = medfilt(frame)+.1
            Q = numpy.where(frame/median > threshold,1,0)
            data[z] = frame*(1-Q)+median*Q

    if was_2d: data = data[0]
    return data
    
def align_frames(data,align_to=None,region=None,use_mag_only=False,return_type='data'):
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
        use_mag_only - Align using only the magnitude component of data. Default
            is False.
        return_type - align_frames is called from multiple places, and
            expectations of what is returned vary. Returned can be aligned data,
            aligned and summed data, or just the alignment coordinates; keywords
            for these are 'data', 'sum', and 'coordinates', respectively.
            Default is 'data'.

    Returns:
        result - The result depends on the input return_type. If return_type is
        set to:
            'data' - An aligned array of shape and dtype identical to data, or
            'sum' - the the summed array, or
            'coordinates' - the coordinates that the arrays need to be rolled
                in order to align them.
    """

    from . import crosscorr
    
    # check types
    assert isinstance(data,numpy.ndarray),                        "data to align must be an array"
    assert isinstance(align_to,(type(None),numpy.ndarray)),       "align_to must be an array or None"
    assert isinstance(region,(type(None),numpy.ndarray)),         "region must be an array or None"
    assert use_mag_only in (0,1,True,False),                      "use_mag_only must be boolean-evaluable"
    assert return_type in ('data','sum','coordinates'),           "return_type must be 'data', 'sum', or 'coordinates'; 'data' is default"
    if data.ndim == 2: assert isinstance(align_to,numpy.ndarray), "data is 2d; need an explicit alignment reference"
#    if data.ndim == 2 and return_type == 'sum': print             "summing 2d data is non-sensical" # not an assert!
    
    # define some simple helper functions to improve readability
    rolls = lambda d, r0, r1: numpy.roll(numpy.roll(d,r0,axis=0),r1,axis=1)
    def prep(x):
        if use_mag_only: x = abs(x)
        if region != None: x = region*x
        return x

    # cast 2d to 3d so the loops below are simpler
    was_2d = False
    if data.ndim == 2:
        was_2d = True
        data.shape = (1,data.shape[0],data.shape[1])
    else:
        was_2d = False
    frames, rows, cols = data.shape
    
    # check some more assumptions
    if region != None:   assert region.shape == (rows,cols),    "region and data frames must be same shape"

    # set up explicit align_to in case of None
    if align_to == None:
        align_to = data[0]
    else:
        assert align_to.shape == (rows,cols),  "align_to and data frames must be same shape"
    
    # for speed, precompute the reference dft
    ref = numpy.fft.fft2(prep(align_to))
    
    # get the alignment coordinates for each frame in data by the argmax of the cross
    # correlation with the reference
    coordinates = numpy.zeros((frames,2),int)
    for n,frame in enumerate(data):
        coordinates[n] = crosscorr.alignment_coordinates(prep(frame),ref,already_fft=(1,))
        
    # now return the data according to return_type
    if return_type == 'coordinates': return coordinates
    # align data frames by rolling
    for n in range(frames):
        rr, rc = coordinates[n]
        data[n] = rolls(data[n],rr,rc)
        
    if return_type == 'data':
        if was_2d: data.shape = (rows,cols)
        return data
    
    if return_type == 'sum': return numpy.sum(data,axis=0) 

def match_counts(img1, img2, region=None, nparam=3):
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
    def diff3(c, img1, img2, region):
        """ minimize (I1 - d1) - s(I2-d2)
            = I1 - s*I2 + (s*d2 - d1)
        """
        (s, d1, d2) = c
        if d1 < 0 or d2 < 0:
            return 1e30
        dif = region*(img1 - d1 - s*(img2 - d2))**2
        return dif.sum()

    def diff2(c, img1, img2, region):
        """ minimize I1 - s(I2-d2)
            = I1 - s*(I2 - d2)
        """
        (s, d2) = c
        if d2 < 0:
            return 1e30
        dif = region*(img1 - s*(img2 - d2))**2
        return dif.sum()

    def diff1(c, img1, img2, region):
        """ minimize I1 - s*I2
            = I1 - s*I2
        """
        (s) = c
        dif = region*(img1 - s*img2)**2
        return dif.sum()

    assert isinstance(img1, numpy.ndarray) and img1.ndim == 2, "img1 must be a 2d ndarray."
    assert isinstance(img2, numpy.ndarray) and img2.ndim == 2, "img2 must be a 2d ndarray."
    assert img1.shape == img2.shape, "(img1, img2) must be the same shape"

    if not (type(nparam) == int and nparam in (1,2,3)):
        nparam = 3 

    if type(region) != numpy.ndarray and region == False:
        region = numpy.ones_like(img1)
    else:
        # convert region to 1/0 just to be sure
        region = numpy.where(region >= 1, 1, 0) 

    if region.sum() == 0:
        print("***** match_counts: Region of interest is empty! *****")
        return img2

    d1 = img1[0,0]
    d2 = img2[0,0]
    c0guess = (img1-d1)/(img2-d2)
    # remove the infinities and nans.  we get these from img2-d2 =0
    c0guess = numpy.where(numpy.isfinite(c0guess), c0guess, 0)

    s = (c0guess*region).sum()/region.sum()

    if nparam == 1:
        c = numpy.array([s])
        diff = diff1
        paramstr = "s=%1.2f"
        funcstr = "img1-s*img2 (%d parameter)" % nparam
    elif nparam == 2:
        c = numpy.array([s, d2])
        diff = diff2
        paramstr = "s=%1.2f, d2=%1.2f"
        funcstr = "img1-s*(img2-d2) (%d parameters)" % nparam
    else: # assume 3 parameter fit
        c = numpy.array([s, d1, d2])
        diff = diff3
        paramstr = "s=%1.2f, d1=%1.2f, d2=%1.2f"
        funcstr = "(img1-d1)-s*(img2-d2) (%d parameters)" % nparam

    print("minimizing %s.\nInitial guess: %s." % (funcstr, paramstr % tuple(c)))

    x = scipy.optimize.fmin(diff, c, args=(img1, img2, region), disp=False)

    print("Final result: %s." % (paramstr % tuple(x)))

    if nparam == 1:
        return x[0]*img2
    elif nparam == 2:
        return x[0]*(img2 - x[1])
    else:
        return x[0]*(img2 - x[2]) + x[1]
