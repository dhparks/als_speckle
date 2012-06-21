# For helper functions which beat on raw experimental data until it is suitable for proper analysis.
# I expect most of this code will be mostly experiment-specific but at least some may be relatively
# universal (ie, alignment or dust-removal).

import numpy
DFT = numpy.fft.fft2
IDFT = numpy.fft.ifft2

def covariance_matrix(data,save_memory=False):
    
    """ Given 3d data, cross-correlates every frame with every other frame. This is useful
    for determining which frames to sum in the presence of drift or slow dynamics.
    
    Requires:
        data - 3d ndarray
        
    Optional:
        save_memory: if True, DFTs of frames of data are precomputed which massively improves
        speed at the expense of memory. Default is False.
        
    Returns:
        a 2d ndarray of normalized covariances max values where returned[row,col] is the
        cross-correlation max of frames data[row] and data[col]"""
        
    assert isinstance(data, numpy.ndarray) and data.ndim == 3, "data must be 3d ndarray"
    
    frames = data.shape[0]
    dfts = numpy.zeros_like(data).astype('complex')
    ACs = numpy.zeros(frames,float)
        
    # precompute the dfts and autocorrelation-maxes
    for n in range(frames):
        dft = DFT(data[n].astype('float'))
        if not save_memory: dfts[n] = dft
        ACs[n]  = abs(IDFT(dft*numpy.conjugate(dft))).max()
          
    # calculate the pair-wise normalized covariances  
    covars = numpy.zeros((frames,frames),float)
        
    for j in range(frames):
        for k in range(frames-j):
            k += j
            ac = ACs[j]
            bc = ACs[k]
            if save_memory:
                d1 = DFT(data[j])
                d2 = numpy.conjugate(DFT(data[k]))
            else:
                d1 = dfts[j]
                d2 = numpy.conjugate(dfts[k])
            cc = abs(IDFT(d1*d2)).max()
            covar = cc/numpy.sqrt(ac*bc)
            covars[j,k] = covar
            covars[k,j] = covar
            
    return covars

def remove_dust(data,dust_mask,dust_plan=None):
    """ Attempts to remove dust and burn marks from CCD images by interpolating in regions marked as
    defective in a plan file.
    
    Requires:
        data - the data from which dust will be removed. ndarray or path to some sort of data
        dust_mask - a binary array describing from which pixels dust must be removed
        
    Optional input:
        plan - generated from dust_mask; this can only be supplied as an argument if remove_dust has
        been previously run and plan returned then as output
        
    Returns:
        0 - fixed data
        1 - plan unique to dust_mask which can be passed again for re-use."""

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

def plan_remove_dust(Mask):
    # find which pixels need to be interpolated based on Mask. based on knowledge of the pixels
    # determine where to interpolate various rows and cols to avoid redundant interpolation.

    Mask = Mask.astype('int')
    
    # mark the pixels in Mask with unique identifiers
    L = len(Mask)
    Marked = numpy.zeros(L**2)
    Marked[:] = numpy.ravel(Mask)[:]
    Marked *= (numpy.arange(L**2)+1)
    Marked = Marked[numpy.nonzero(Marked)]-1 # this is the location of all the non-zero pixels
    
    PixelIDStrings = []
    
    for Value in Marked:

        # record only some information about each pixel pertaining to the eventual spline fits. for
        # example we dont actually need to know both the exact row and column of each pixel, only the
        # range over which the spline will be evaluated. this eliminates duplicate work by avoiding
        # re-interpolation of the same range of pixels.
        # this information is recorded in idstring
        idstring = ''
        
        r,c = Value/L,Value%L # coordinate of an active pixel in Mask
        Row,Col = Mask[r],Mask[:,c]
        
        # now slook in both directions to find out how wide the object is in row in col
        r1,r2,c1,c2 = 0,0,0,0
        while Row[c+c1] > 0: c1 += 1
        while Row[c+c2] > 0: c2 += -1
        while Col[r+r1] > 0: r1 += 1
        while Col[r+r2] > 0: r2 += -1
        
        rmax = r+r1
        rmin = r+r2
        cmax = c+c1
        cmin = c+c2
        
        # figure out whether we should interpolate this pixel by row or by column
        if rmax-rmin <= cmax-cmin: idstring += 'r,%.4d,%.4d,%.4d'%(c,rmin,rmax)
        if rmax-rmin > cmax-cmin:  idstring += 'c,%.4d,%.4d,%.4d'%(r,cmin,cmax)
        
        # record the essentials about the pixel
        PixelIDStrings.append(idstring)
    
    # return only the unique ID strings to eliminate redundant interpolations. set() returns unique
    # elements of a list without order preservation which doesn't matter for this purpose
    return tuple(set(PixelIDStrings))

def subtract_background(data,dark=None,x=20,scale=1):
    """Subtract a background file. The DC component of both files is subtracted first."""

    # check types
    assert isinstance(data,numpy.ndarray), "data must be ndarray"
    assert data.ndim in (2,3), "data must be 2d or 3d"
    assert isinstance(dark,(type(None),numpy.ndarray)), "dark must be None or ndarray"
    if isinstance(dark,numpy.ndarray):
        assert dark.ndim == 2, "dark must be 2d"
        assert data.shape[-2:] == dark.shape, "data and dark must be same shape"
    
    # subtract DC component from data
    if data.ndim == 2:
        dc = float(numpy.sum(data[0:x,0:x])/x**2)
        data = abs(data-dc)
    if data.ndim == 3:
        for n in range(data.shape[0]):
            dc = float(numpy.sum(data[n,0:x,0:x])/x**2)
            data[n] = abs(data[n]-dc)
        
    # dark subtraction can be broadcast to all frames of data so no need to check ndim
    if dark == None:
        pass
    else:
        bgdc = float(numpy.sum(dark[0:x,0:x])/x**2)
        dark = abs(dark-bgdc)
        data = abs(data-dark*scale)
        
    return data

def hot_pixels(data,i=1,t=2):
    """A basic hot pixel remover which uses numpy.medfilt to define hot pixels as those
    which exceed a certain multiple of the local median. Slow for large arrays; this operation
    would probably benefit a great deal from GPU acceleration.
    
    Required:
        data -- 2d or 3d array from which hot pixels will be removed
        
    Optional:
        i -- number of iterations to run the smoother. Default is 1.
        t -- threshold to specify when pixels are hot. When a pixel has
            value greater than t*median, it is replaced by the median.
            Default is 2.
    
    """
    
    from scipy.signal import medfilt
    
    # check types
    assert isinstance(data,numpy.ndarray), "data must be ndarray"
    assert data.ndim in (2,3), "data must be 2d or 3d"
    assert isinstance(i,int), "number of iterations must be integer"
    assert isinstance(t, (int,float)), "threshold must be float or int"
    
    was_2d = False
    if data.ndim == 2:
        was_2d = True
        data.shape = (1,data.shape[0],data.shape[1])
    for z,frame in enumerate(data):

        for m in range(i):
            median = medfilt(frame)+.1
            Q = numpy.where(frame/median > t,1,0)
            data[z] = frame*(1-Q)+median*Q

    if was_2d: data = data[0]
    return data
    
def align_frames(data,align_to=None,region=None,use_mag_only=False,return_type='data'):
    
    """ Align a set of data frames by FFT/cross-correlation method.
    
    Required input:
    data -- A 2d or 3d ndarray. Probably this needs to be real-valued, not complex.
    
    Optional arguments:
        align_to -- A 2d array used as the alignment reference. If None, data must
                    be 3d and the first frame of data will be the reference. Default is None
                
        region -- A 2d mask which specifies which data to use in the cross correlation. If None,
                  all pixels will contribute equally to the alignment. Default is None.
        
        use_mag_only -- Do alignment using only the magnitude component of data. Default is False.
                  
        return_type -- align_frames is called from multiple places, and expectations of what is
                       returned vary. Returned can be aligned data, aligned and summed data, or
                       just the alignment coordinates; keywords for these are 'data', 'sum', and
                       'coordinates', respectively. Default is 'data'.
              
    Returns: an array of shape and dtype identical to data, data summed along the 0 axis, or a
             list of alignment coordinates depending on return_type argument."""
    
    # check types
    assert isinstance(data,numpy.ndarray),                        "data to align must be an array"
    assert isinstance(align_to,(type(None),numpy.ndarray)),       "align_to must be an array or None"
    assert isinstance(region,(type(None),numpy.ndarray)),         "region must be an array or None"
    assert use_mag_only in (0,1,True,False),                      "use_mag_only must be boolean-evaluable"
    assert return_type in ('data','sum','coordinates'),           "return_type must be 'data', 'sum', or 'coordinates'; 'data' is default"
    if data.ndim == 2: assert isinstance(align_to,numpy.ndarray), "data is 2d; need an explicit alignment reference"
    if data.ndim == 2 and return_type == 'sum': print             "summing 2d data is non-sensical" # not an assert!
    
    # define some simple helper functions to improve readability
    if use_mag_only: dft2 = lambda x: DFT(abs(x))
    if not use_mag_only: dft2 = lambda x: DFT(x)
    corr_max = lambda x,y: abs(IDFT(x*y)).argmax() # the incoming frames have both been ffted already
    rolls = lambda d, r0, r1: numpy.roll(numpy.roll(d,r0,axis=0),r1,axis=1)

    # cast 2d to 3d so the loops below are simpler
    was2d = False
    if data.ndim == 2:
        was2d = True
        data.shape = (1,data.shape[0],data.shape[1])
    frames, rows, cols = data.shape
    
    # check some more assumptions
    if region != None:   assert region.shape == (rows,cols),    "region and data frames must be same shape"
    if align_to != None: assert align_to.shape == (rows,cols),  "align_to and data frames must be same shape"

    # set up explicit region and align_to in case of None
    if align_to == None: align_to = data[0]
    if region == None:   region = numpy.ones_like(align_to)
    # for speed, precompute the reference dft
    dft_0 = numpy.conjugate(dft2(align_to*region))
    
    # get the alignment coordinates for each frame in data by the argmax of the cross
    # correlation with the reference
    coordinates = []
    for frame in data:
        dft_n = dft2(frame*region)
        cc_max = corr_max(dft_n,dft_0)
        max_row,max_col = cc_max/cols,cc_max%cols # turn cc_max into 2d coordinates
        if max_row > rows/2: max_row += -rows # modulo arithmetic for cyclic BCs
        if max_col > cols/2: max_col += -cols
        coordinates.append([-max_row,-max_col])
        
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
