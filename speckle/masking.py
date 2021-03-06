""" Basic mask application methods. These are used on masks to hide or remove
data that is not of interest.

"""
import numpy as np

def bounding_box(data, threshold=1e-10, force_to_square=False, pad=0,
                 power_of_2=False):
    
    """ Find the minimally-bounding subarray for data.
    
    arguments:
        data -- array to be bounded
        threshold -- (optional) value above which data is considered boundable
        force_to_square -- (optional) if True, forces the minimally bounding
            subarray to be square in shape. Default is False. This can fail
            to be achieved in certain cases, such as when the input data
            is not square; a guarantee of a square bounding box requires
            access to padding with zeros.
        pad -- (optional) expand the minimally bounding coordinates by this
            amount on each side. Default = 0 for minimally bounding.
        power_of_2 -- (optional) Default False. If True, will try to make the
            sides of the bounding box powers of 2 in length. If the next
            largest power of 2 is larger than the array dimensions, the array
            dimension is returned. If force_to_square = True, the larger power
            of 2, subject to the array size limitation, is returned.
    
    returns:
        bounding coordinates -- a tuple of (row_min, row_max, col_min,
            col_max) which bound data.  These values can be used for slicing
            all of the data out of an array.
    """
    
    def _next_p2(fr):
        """ Finds next-largest power of two"""
        p2 = ((fr & (fr-1)) == 0)
        if p2:
            L = fr
        if not p2:
            L = int(2**(math.floor(math.log(fr, 2))+1))
        return L
    
    def _expand(rmin, rmax, cmin, cmax):
        """ Try to make sides a power of 2 in length """
        
        bb_rows = rmax-rmin
        bb_cols = cmax-cmin

        r_p2 = _next_p2(rmax-rmin)
        c_p2 = _next_p2(cmax-cmin)
        
        if r_p2 >= rows:
            rmin, rmax = 0, rows
        if c_p2 >= cols:
            cmin, cmax = 0, cols
        
        if r_p2 < rows:
            r_diff = r_p2-bb_rows

            if r_diff%2 == 1:
                r_diff_l = math.floor(r_diff/2.)
                r_diff_h = r_diff_l+1
                
            else:
                r_diff_l = r_diff/2
                r_diff_h = r_diff/2
                
            rmin -= r_diff_l
            rmax += r_diff_h
            
        if c_p2 < cols:
            c_diff = c_p2-bb_cols

            if c_diff%2 == 1:
                c_diff_l = math.floor(c_diff/2.)
                c_diff_h = c_diff_l+1
                
            else:
                c_diff_l = c_diff/2
                c_diff_h = c_diff/2
                
            cmin -= c_diff_l
            cmax += c_diff_h
            
        return rmin, rmax, cmin, cmax
    
    def _square(rmin, rmax, cmin, cmax):
        
        """ Try to make the bounding box a square """
        
        delta_r = rmax-rmin
        delta_c = cmax-cmin
        
        if delta_r%2 == 1:
            delta_r += 1
            if rmax < rows:
                rmax += 1
            else:
                rmin += -1
            
        if delta_c%2 == 1:
            delta_c += 1
            if cmax < cols:
                cmax += 1
            else:
                cmin += -1
            
        if delta_r > delta_c:
            average_c = (cmax+cmin)/2
            cmin = average_c-delta_r/2
            cmax = average_c+delta_r/2
            
        if delta_c > delta_r:
            average_r = (rmax+rmin)/2
            rmin = average_r-delta_c/2
            rmax = average_r+delta_c/2
            
        return rmin, rmax, cmin, cmax

    import math
    
    assert isinstance(data, np.ndarray) and data.ndim == 2, \
    "data must be a 2d ndarray"
    
    assert isinstance(pad, int), "data must be an integer"
    
    assert isinstance(force_to_square, bool), "force_to_square must be a bool"
    
    assert isinstance(threshold, (int, float)), \
    "threshold must be int or float"
    
    mask = np.where(data > threshold, 1, 0)
    rows, cols = mask.shape

    aw = np.argwhere(mask)
    rmin, cmin = aw.min(0)
    rmax, cmax = aw.max(0)+1
    
    # pad the boundaries
    rmin = 0 if rmin < pad else rmin - pad
    rmax = rows if rmax + pad > rows else rmax + pad
    cmin = 0 if cmin < pad else cmin - pad
    cmax = cols if cmax + pad > cols else cmax + pad

    # try to expand the box to the next largest power of 2
    if power_of_2:
        rmin, rmax, cmin, cmax = _expand(rmin, rmax, cmin, cmax)
        
    # try to make the bounding box a square; can fail
    if force_to_square:
        rmin, rmax, cmin, cmax = _square(rmin, rmax, cmin, cmax)
            
    # this here is the panic section
    if rmin < 0:
        rmin, rmax = 0, rmax-rmin
    if cmin < 0:
        cmin, cmax = 0, cmax-cmin
        
    return np.array([rmin, rmax, cmin, cmax]).astype('int')

def apply_shrink_mask(img, mask):
    """ Applys a mask and shrinks the image to just the size of the mask.

    arguments:
        img - img to mask.  Must be array of 2 or 3 dimensions.
        mask - mask to use.  Must be two dimensional.

    returns:
        img - shrunk image with a mask applied
    """
    assert isinstance(img, np.ndarray), "img must be an array"
    assert isinstance(mask, np.ndarray), "mask must be an array"
    assert img.shape == mask.shape, "img and mask must be the same shape"
    assert img.ndim in (2, 3), "image must be two or three dimensional"
    
    def _apply_shrink(img, mask):
        """ Helper """
        ystart, ystop, xstart, xstop = bounding_box(mask, threshold=0)
        return (img*mask)[ystart:ystop, xstart:xstop]

    if img.ndim == 3:
        res = np.zeros_like(img)
        for i in range(img.shape[0]):
            res[i] = _apply_shrink(img[i], mask)
        return res
    else:
        return _apply_shrink(img, mask)

def take_masked_pixels(data, mask):
    """ Copy the pixels in data masked by mask into a 1d array.
    The pixels are unordered. This is useful for optimizing subtraction, etc,
    within a masked region (particularly a multipartite mask) because it
    eliminates all the pixels that don't contribute in the mask.
    
    arguments:
        data -- 2d or 3d data from which pixels are taken
        mask -- 2d mask file describing which pixels to take.
        
    returns:
        if data is 2d, a 1d array of selected pixels
        if data is 3d, a 2d array where axis 0 is the frame axis
    """
    
    assert isinstance(data, np.ndarray) and data.ndim in (2, 3), \
    "data must be 2d or 3d array"
    
    if data.ndim == 2:
        assert data.shape == mask.shape, "data and mask must be same shape"
    else: # data.ndim == 3
        assert data[0].shape == mask.shape, "data and mask must be same shape"

    # find which pixels to take (ie, which are not masked)
    assert isinstance(mask, np.ndarray) and mask.ndim == 2, \
    "mask must be 2d array"

    indices = np.nonzero(np.where(mask > 1e-6, 1, 0))

    # return the requested pixels
    if data.ndim == 2:
        return data[indices]
    if data.ndim == 3:
        result = np.zeros((data.shape[0], len(indices[0])))
        for i, fr in enumerate(data):
            result[i] = fr[indices]
        return result

def put_masked_pixels(data, mask):
    """ This function is the functional inverse of take_masked_pixels.
    Whereas that function takes a 2d or 3d array and a corresponding mask and
    returns a 1d list of values from the mask, this function takes the 1d list
    of values and the mask and returns filled 2d or 3d array. If the same mask
    was applied to a sequence of frames, this attempts to undo that operation.
    The reason for this function is you might want to take some pixels to speed
    an operation (such as background subtraction), then put them back in order.
    
    arguments:
        data: 1d or 2d list of values of any type
        mask: 2d or 3d mask saying where those values came from
        
    returns:
        data put back into the shape of mask
    """
    
    assert isinstance(mask, np.ndarray) and mask.ndim in (2, 3), \
    "mask must be 2d or 3d array"
    
    assert isinstance(data, np.ndarray) and data.ndim in (1, 2), \
    "data must be 1d or 2d array"

    if data.ndim == 1:
        nf = 1
    if data.ndim == 2:
        nf = data.shape[0]
    
    ms = mask.shape
    mr = mask.ravel()
    indices = np.nonzero(np.where(mr > 1e-6, 1, 0))[0]

    # slice each frame out of the data, embed in a list of zeros, and reshape
    toreturn = np.zeros((nf,)+ms, data.dtype)
    for n in range(nf):
        tmp = np.zeros(mask.size, data.dtype)
        tmp[indices] = np.copy(data[n, :])
        toreturn[n] = tmp.reshape(ms)
    
    # return the reshaped data
    if nf == 1:
        toreturn = toreturn[0]
    return toreturn
    
def trace_object(data_in, start, detect_val=None, return_type='detected'):
    """ Using a floodfill algorithm (basically, the paintbuck tool in software
    like Photoshop), find all the elements of an array which are connected to
    a specified pixel.
    
    arguments:
        data_in -- 2d array to be flood-filled
        start -- a 2-tuple containing the (row,col) of the starting pixel
        detect_val -- (optional) a 2-tuple containing the (min,max) of what is
            considered to be connected to the start pixel. default is
            (data_in[start], data_in[start]), which requires all the connected
            pixels to be of exactly the same value. if not passing a previously
            thresholded array it is wise to supply a detect_val tuple.
        return_type -- specify the type of data returned
            if 'detected': returns a boolean array of the object
            if 'iterations': returns an integer array showing the algorithm
                iteration at which each pixel was detected. this can be useful
                for calculating the approximate length of an object.
            default is 'detected'
        
    returns:
        a 2d array of with the same shape as data_in. return type depends on
        optional argument return_type
    """
    
    def _neighbors(y, x):
        """ Helper which gives nearest neighbors """
        
        return ((y+1, x-1), (y+1, x), (y+1, x+1), (y, x-1), \
            (y, x+1), (y-1, x-1), (y-1, x), (y-1, x+1))
    
    # check types
    assert isinstance(data_in, np.ndarray) and data_in.ndim == 2, \
    "data must be a 2d array"
    
    assert isinstance(start, tuple) and len(start) == 2,\
    "start must be a 2-tuple"
    
    assert isinstance(start[0], int) and isinstance(start[1], int), \
    "start coordinates must be integer for indices"
    
    assert isinstance(detect_val, (tuple, type(None))), \
    "threshold must be 2-tuple of float or int"
    
    if isinstance(detect_val, tuple):
        tmp1 = len(detect_val) == 2
        tmp2 = isinstance(detect_val[0], (int, float))
        tmp3 = isinstance(detect_val[1], (int, float))
        
        assert tmp1 and tmp2 and tmp3, \
        "threshold must be 2-tuple of float or int"
        
    if return_type not in ('detected', 'iterations'):
        print "return_type %s unrecognized, falling back to \
        'detected' default"%return_type
        
        return_type = 'detected'
    
    detected = np.zeros(data_in.shape, int)
    
    # set up the detect_val bounds and check the value of data_in at start
    start_val = data_in[start]
    if detect_val == None:
        detect_val = (start_val, start_val)
    if detect_val[0] == 0:
        print "warning! lower bound is 0!"
        
    working = np.where(data_in >= detect_val[0], 1, 0)
    working *= np.where(data_in <= detect_val[1], 1, 0)

    if not working[start]:
        print "value of data_in at start is not within detect_val bounds!"
        print "returning the input array with marked start for debugging"
        data_in[start] = 1.1*data_in.max()
        return data_in
    
    N, M = data_in.shape
    mNM = lambda a: (np.mod(a[0]+N, N), np.mod(a[1]+M, M))

    # do floodfill. the basic idea of the algorithm is as follows: for every
    # pixel known to be in edge, look at the 8 nearest neighbors. if they pass
    # the threshold test, add them to edge, and add the primary pixel to the
    # detected object. when edge is exhausted, the algorithm is complete
    def _update(n, ne):
        if working[n] and not detected[n]:
            detected[n] = iteration
            ne.append(n)
    
    iteration = 1
    edge = [start,]
    
    while edge:
        new_edge = []
        for (y, x) in edge:
            for neighbor in _neighbors(y, x):
                try:
                    _update(neighbor, new_edge)
                except IndexError:
                    neighbor = mNM(neighbor)
                    _update(neighbor, new_edge)   
        edge = new_edge
        iteration += 1
        
    if return_type == 'iterations':
        return detected
    if return_type == 'detected':
        return detected.astype(bool)
    
def find_all_objects(data, detect_val=(.95, 1.05), return_type='objects'):
    """ Find all the objects in an array. This is intended for finding all the
    components of a multi-partite support, but could also be used for finding
    all the domains in an image. Works by iteratively applying the trace_object
    function when a new object is encountered.
    
    input:
        data -- array with objects. best if binary or nearly binary.
        detect_val -- (optional) a 2-tuple containing the (min,max) of what is
            considered to be connected to the start pixel. default is
            (.95,1.05), which assumes a nearly binary object. if not passing a
            previously thresholded array it is wise to supply this tuple.
        return_type -- specify the returned output (see below)
            
    returns: depends on return_type
        if return_type = 'objects': get back a 3d array, each array frame has
            an isolated object. this can get VERY LARGE if there are many
            objects!
        if return_type = 'coordinates': get back a list of tuples. each tuple
            is a coordinate in the object (specifically: the first pixel).
            from this information the objects can be found using trace_object.
    """
    
    assert isinstance(data, np.ndarray), "input data must be an array"
    assert data.ndim == 2, "for now data must be 2d"
    
    assert isinstance(detect_val, (tuple, list, np.ndarray, int, float)),\
    "detect_val type (is: %s) not recognized"%(type(detect_val))
    
    assert return_type in ('objects', 'coordinates'),\
    "return_type string (is: %s) not recognized"%(return_type)
    
    objects = [] # make this into an array at return
    starts = []
    
    for r in range(data.shape[0]):
        if data[r].any(): # scan all the pixels until we find a starting point
            for c in range(data.shape[1]):
                
                if data[r, c] >= detect_val[0] and data[r, c] <= detect_val[1]:
                    # found an object! trace it!
                    
                    found = trace_object(data, (r, c), detect_val=detect_val)
                    if return_type == 'objects':
                        objects.append(found.astype(int))
                    if return_type == 'coordinates':
                        starts.append((r, c))
                        
                    # zero it out so we dont find it again
                    data[found == 1] = 0

                    
    if return_type == 'coordinates':
        return starts
    if return_type == 'objects':
        return np.array(objects).astype(int)

