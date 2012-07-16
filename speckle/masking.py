""" Basic mask application methods. These are used on masks to hide or remove
data that is not of interest.

"""
import numpy as np

def bounding_box(data,threshold=1e-10,force_to_square=False,pad=0):
    
    """ Find the minimally-bounding subarray for data.
    
    arguments:
        data -- array to be bounded
        threshold -- (optional) value above which data is considered boundable
        force_to_square -- (optional) if True, forces the minimally bounding
            subarray to be square in shape. Default is False.
        pad -- (optional) expand the minimally bounding coordinates by this
            amount on each side. Default = 0 for minimally bounding.
    
    returns:
        bounding coordinates -- an array of (row_min, row_max, col_min, col_max)
            which bound data
    """
    
    data = np.where(data > threshold, 1, 0)
    rows, cols = data.shape
    
    rmin, rmax, cmin, cmax = 0, 0, 0, 0
    
    for row in range(rows):
        if data[row, :].any():
            rmin = row
            break
            
    for row in range(rows):
        if data[rows-row-1, :].any():
            rmax = rows-row
            break
            
    for col in range(cols):
        if data[:, col].any():
            cmin = col
            break
    
    for col in range(cols):
        if data[:, cols-col-1].any():
            cmax = cols-col
            break
        
    if rmin >= pad: rmin += -pad
    else: rmin = 0
    
    if rows-rmax >= pad: rmax += pad
    else: rmax = rows
    
    if cmin >= pad: cmin += -pad
    else: cmin = 0
    
    if cols-cmax >= pad: cmax += pad
    else: cmax = cols
        
    if force_to_square:
        delta_r = rmax-rmin
        delta_c = cmax-cmin
        
        if delta_r%2 == 1:
            delta_r += 1
            if rmax < rows: rmax += 1
            else: rmin += -1
            
        if delta_c%2 == 1:
            delta_c += 1
            if cmax < cols: cmax += 1
            else: cmin += -1
            
        if delta_r > delta_c:
            average_c = (cmax+cmin)/2
            cmin = average_c-delta_r/2
            cmax = average_c+delta_r/2
            
        if delta_c > delta_r:
            average_r = (rmax+rmin)/2
            rmin = average_r-delta_c/2
            rmax = average_r+delta_c/2
            
        if delta_r == delta_c:
            pass
        
    return np.array([rmin, rmax, cmin, cmax]).astype('int32')

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
    
    def apply_shrink(img, mask):
        result = img*mask
        (ys, xs) = img.shape
        aw = np.argwhere(result)
        (ystart, xstart) = aw.min(0)
        (ystop, xstop) = aw.max(0) + 1

        if xstart == xstop:
            # we have a single row, try to increase x in both directions
            if xstart != 0:
                xstart -= 1
            if xstop != xs:
                xstop += 1

        if ystart == ystop:
            # we have a single column, try to increase y in both directions
            if ystart != 0:
                ystart -= 1
            if ystop != xs:
                ystop += 1

        return img[ystart:ystop, xstart:xstop]

    if img.ndim == 3:
        res = np.zeros_like(img)
        for i in range(img.shape[0]):
            res[i] = apply_shrink(img[i], mask)
        return res
    else:
        return apply_shrink(img, mask)

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
        if data is 2d, a 2d array where axis 0 is the frame axis
    """
    assert isinstance(data, np.ndarray) and data.ndim in (2, 3), "data must be 2d or 3d array"
    assert isinstance(mask, np.ndarray) and mask.ndim == 2, "mask must be 2d array"
    if data.ndim == 2:
        assert data.shape == mask.shape, "data and mask must be same shape"
    else: # data.ndim == 3
        assert data[0].shape == mask.shape, "data and mask must be same shape"

    # find which pixels to take (ie, which are not masked)
    indices = np.nonzero(np.where(mask > 1e-6, 1, 0))

    # return the requested pixels
    if data.ndim == 2:
        return data[indices]
    if data.ndim == 3:
        result = np.zeros((data.shape[0], len(indices[0])))
        for i, fr in enumerate(data):
            result[i] = fr[indices]
        return result
    
def trace_object(data_in, start, detect_val=None, return_type='detected'):
    """ Using a floodfill algorithm (basically, the paintbuck tool in software
    like Photoshop), find all the elements of an array which are connected to
    the a specified pixel.
    
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
    
    # check types
    assert isinstance(data_in,np.ndarray) and data_in.ndim == 2, "data must be a 2d array"
    assert isinstance(start,tuple) and len(start) == 2, "start must be a 2-tuple"
    assert isinstance(start[0],int) and isinstance(start[1],int), "start coordinates must be integer for indices"
    assert isinstance(detect_val, (tuple,type(None))), "threshold must be 2-tuple of float or int"
    if isinstance(detect_val,tuple):
        assert len(detect_val) == 2 and isinstance(detect_val[0],(int,float)) and isinstance(detect_val[1],(int,float)), "threshold must be 2-tuple of float or int"
    if return_type not in ('detected','iterations'):
        print "return_type %s unrecognized, falling back to 'detected' default"%return_type
        return_type = 'detected'
    
    # set up arrays
    detected = np.zeros(data_in.shape,int)
    
    # set up the detect_val bounds and check the value of data_in at start
    start_val = data_in[start]
    if detect_val == None: detect_val = (start_val,start_val)
    working = np.where(data_in >= detect_val[0],1,0)*np.where(data_in <= detect_val[1],1,0)
    if not working[start]:
        print "value of data_in at start is not within detect_val bounds!"
        print "returning the input array with marked start for debugging"
        data_in[start] = 1.1*data_in.max()
        return data_in
    
    # do floodfill. the basic idea of the algorithm is as follows: for every
    # pixel known to be in edge, look at the 8 nearest neighbors. if they pass
    # the threshold test, add them to edge, and add the primary pixel to the
    # detected object. when edge is exhausted, the algorithm is complete
    iteration = 1
    edge = [start]
    while edge:
        new_edge = []
        for (y,x) in edge:
            for neighbor in ((y+1,x-1),(y+1,x),(y+1,x+1),(y,x-1),(y,x+1),(y-1,x-1),(y-1,x),(y-1,x+1)):
                try:
                    if working[neighbor] and not detected[neighbor]:
                        detected[neighbor] = iteration
                        new_edge.append(neighbor)
                except IndexError:
                    pass
        edge = new_edge
        iteration += 1
        
    if return_type == 'iterations': return detected
    if return_type == 'detected': return detected.astype(bool)