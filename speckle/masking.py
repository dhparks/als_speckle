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
    assert isinstance(data, np.ndarray) and data.ndim == 2, "data must be a 2d ndarray"
    assert isinstance(pad, int), "data must be an integer"
    assert isinstance(force_to_square, bool), "force_to_square must be a bool"
    assert isinstance(threshold, (int, float)), "threshold must be int or float"
    data = np.where(data > threshold, 1, 0)
    rows, cols = data.shape
    
    aw = np.argwhere(data)
    (rmin, cmin) = aw.min(0)
    (rmax, cmax) = aw.max(0) + 1

    # pad the boundaries
    rmin = 0 if rmin < pad else rmin - pad
    rmax = rows if rmax + pad > rows else rmax + pad
    cmin = 0 if cmin < pad else cmin - pad
    cmax = cols if cmax + pad > cols else cmax + pad
        
    if force_to_square:
        delta_r = rmax-rmin
        delta_c = cmax-cmin
        goal_width = max(delta_r, delta_c)
        
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
    
    def apply_shrink(img, mask):
        result = img*mask
        ystart, ystop, xstart, xstop = bounding_box(img, threshold=0)
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
