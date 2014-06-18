"""A library for calculating image shapes and functions

Author: Daniel Parks (dparks@uoregon.edu)
Author: Keoki Seu (kaseu@lbl.gov)
"""
import numpy as np

def radial(size, center=None):
    """ Returns azimuthally symmetric distance function in n-dimensions
    
    arguments:
        size: an n-tuple for the array size in format (rows, columns).
              Must be integer.
        center: an n-tuple giving origin of coordinate system (ie, where
                r = 0). Defaults to array center in absence of supplied
                values. Boundary conditions are NOT cyclic.

    returns:
        A n-dimensioanl array where each element is the distance (in pixels)
            from the center.
    """
   
   
    assert isinstance(size, tuple), "size must be a tuple"
    ndims = len(size)
    
    if center != None:
        assert isinstance(center, tuple), \
        "center must be supplied as a tuple"
        
        assert len(center) == ndims, \
        "size and center must be same dimensions"
    else:
        center = np.zeros(ndims, float)
        for d in range(ndims):
            center[d] = size[d]/2.
        
    # evaluate pythagorean theorem in n-dimensions
    indices = np.indices(size, float)

    if ndims == 2:
        return np.hypot(indices[0]-center[0], indices[1]-center[1])
    else:
        r = np.zeros_like(indices[0])
        for d in range(ndims):
            r += (indices[d]-center[d])**2
        return np.sqrt(r)
    
def angular(size, center=None):    
    """ Generate a radially symmetric angle function.
    
    arguments:
        size: the size of the array as a tuple (rows, columns).
        center: the center of the coordinate system as a tuple
            (center_row, center_column)

    returns:
        A 2-dimensional array of the angular values, in radians.
    """

    def _check_types(size, center):
        """ Check types """
        
        assert isinstance(size, (tuple, list)) and len(size) == 2,\
        "size must be a tuple or list of length 2"
        
        try:
            size = (int(size[0]), int(size[1]))
        except:
            raise ValueError("couldnt cast size to integers in shape.angular")
        
        if center == None:
            center = (size[0]/2, size[1]/2)
            
        assert isinstance(center, (tuple, list)) and len(center) == 2,\
        "center must be a tuple or list of length 2"
        
        try:
            center = (float(center[0]), float(center[1]))
        except:
            raise ValueError("couldnt cast center to floats in shape.angular")
        
        return size, center
            
    size, center = _check_types(size, center)

    rows, cols = np.indices(size, float)
    rows += -center[0]
    cols += -center[1]
    return np.angle(cols+1j*rows)
    
def square(size, length, center=None):
    """ Generate a square in a numpy array.
    
    arguments:
        size: the size of the array as a tuple (rows, columns).
        length: length of the square. Must be float or int; is casted to int
        center: the center of the coordinate system as a tuple (center_row,
            center_column)

    returns:
        A 2-dimensional numpy array with a square of length (length) centered at
            (center).
    """
    
    def _check_types(size, length, center):
        """ Check types """
        
        assert isinstance(size, (list, tuple)) and len(size) == 2,\
        "size must be a list or tuple of length 2"
        
        try:
            size = (int(size[0]), int(size[1]))
        except:
            raise ValueError("couldnt cast size to integers in shape.square")
        
        try:
            length = int(length)
        except:
            raise ValueError("couldnt cast length to integer in shape.square")
        
        if center == None:
            center = (size[0]/2, size[1]/2)
            
        assert isinstance(center, (list, tuple)),\
        "center must be a list or tuple of length 2"
        
        try:
            center = (int(center[0]), int(center[1]))
        except:
            raise ValueError("couldnt cast center to integers in shape.square")
        
        return size, length, center
    
    size, length, center = _check_types(size, length, center)
    return rect(size, (length, length), center)
    
def rect(size, lengths, center=None):
    """ Generate a rectangle in a numpy array. If the recangle to be drawn is
        larger than the array, the rectangle is drawn to the edges of the array
        and will be smaller than the specified size.
    
    arguments:
        size: the size of the array as a tuple (rows, columns).
        lengths: a 2-tuple formatted as (rows_length, col_length).
        center: the center of the coordinate system as a tuple
            (center_row, center_column)

    returns:
        A 2-dimensional numpy array with a rectangle of (lengths) centered at
            (center).
    """
    
    def _check_types(size, lengths, center):
        """ Check types """

        assert isinstance(size, (tuple, list)) and len(size) == 2,\
        "size must be a 2-tuple"
        
        try:
            size = (int(size[0]), int(size[1]))
        except:
            raise ValueError('couldnt cast size in shape.rect to integers')
        
        assert size[0] > 0 and size[1] > 0, "size must be > 0"

        assert isinstance(lengths, (tuple, list)) and len(lengths) == 2,\
        "lengths must be tuple"
        
        try:
            lengths = (int(lengths[0]), int(lengths[1]))
        except:
            raise ValueError('couldnt cast lengths in shape.rect to integers')
        
        assert lengths[0] > 0 and lengths[1] > 0, "lengths must be > 0"

        if center == None:
            center = (size[0]/2, size[1]/2)
        assert isinstance(center, (tuple, list)) and len(center) == 2,\
        "center must be a tuple or list of length 2"
        
        try:
            center = (int(center[0]), int(center[1]))
        except:
            raise ValueError('couldnt cast center in shape.rect to integers')
        
        assert center[0] > 0 and center[1] > 0, "center must be > 0"

        return size, lengths, center
    
    size, lengths, center = _check_types(size, lengths, center)

    temp = np.zeros(size, int)
    r_min = center[0]-lengths[0]/2
    r_max = center[0]+lengths[0]/2
    c_min = center[1]-lengths[1]/2
    c_max = center[1]+lengths[1]/2

    # Crop the extremum values if we go outside the array.
    # If this is not done, then the array is not filled.
    warn = False
    if r_min < 0:
        warn, r_min = True, 0
    if c_min < 0:
        warn, c_min = True, 0
    if r_max > size[0]:
        warn, r_min = True, size[0]
    if c_max > size[1]:
        warn, r_max = True, size[1]

    if warn:
        print "rect() warning: parts of the rectangle are outside the array"

    temp[r_min:r_max, c_min:c_max] = 1
    return temp
    
def circle(size, radius, center=None, AA=True):
    """ Generate a circle in a numpy array.
    
    arguments:
        size: a 2-tuple formatted as (rows, columns)
        radius of circle
        center: a 2-tuple of (center_row, center_column) of the circle center
        AA: if AA is True, returns an antialiased circle. AA = False gives a
            hard edge and is marginally faster

    returns:
        a numpy array of size with a circle of radius centered on center.
    """
    
    def _check_types(size, radius, center, AA):
        """ Check types"""
        
        assert isinstance(size, (tuple, list)) and len(size) == 2,\
        "size must be a tuple or list of length 2"
        
        try:
            size = (int(size[0]), int(size[1]))
        except:
            raise ValueError("couldnt cast size in shape.circle to integers")
        
        try:
            radius = float(radius)
        except:
            raise ValueError("couldnt cast radius in shape.circle to float")
        assert radius > 0, "radius must be > 0"
        
        if center == None: 
            center = (size[0]/2, size[1]/2)
        assert isinstance(center, (list, tuple)) and len(center) == 2, \
        "center must be a tuple or list of length 2"
        
        try:
            center = (float(center[0]), float(center[1]))
        except:
            raise ValueError("couldnt cast center in shape.circle to float")
        
        assert isinstance(AA, bool) or AA in (0, 1), \
        "AA value must be boolean-evaluable"

        return size, radius, center, AA        
        
    size, radius, center, AA = _check_types(size, radius, center, AA)

    r = radial(size, center)
    
    if not AA:
        return np.where(r < radius, 1, 0)
    if AA:
        return 1-np.clip(r-radius, 0, 1.) 
    
def annulus(size, radii, center=None, AA=True):
    """ Returns an annulus (ie a ring) in a numpy array.

    arguments:
        size: size of array (rows, columns). tuple entries must be int.
        radii: interior and exterior radius of the annulus as a tuple. The
            order is not important because smaller radius must be the interior.
        center: a 2-tuple of (center_row, center_column) of the annulus center.
        AA: if True, antialiases the annulus.

    returns:
        a numpy array with annuls centered on center of radius (r_in, r_out)    
    """
    
    def _check_types(size, radii, center, AA):
        """ Check types """
        
        assert isinstance(size, (tuple, list)) and len(size) == 2,\
        "size must be a tuple or list of length 2"
        
        try:
            size = (int(size[0]), int(size[1]))
        except:
            raise ValueError("couldnt cast size in shape.circle to integers")
        assert size[0] > 0 and size[1] > 0, "size must be > 0"
        
        assert isinstance(radii, (tuple, list)) and len(radii) == 2,\
        "radii must be a tuple or list of length 2"
        
        try:
            radii = (float(radii[0]), float(radii[1]))
        except:
            raise ValueError("couldnt cast radii in shape.circle to float")
        assert radii[0] > 0 and radii[1] > 0, "radii must be > 0"
        
        if center == None:
            center = (size[0]/2, size[1]/2)
        assert isinstance(center, (list, tuple)) and len(center) == 2,\
        "center must be a tuple or list of length 2"
        
        try:
            center = (float(center[0]), float(center[1]))
        except:
            raise ValueError("couldnt cast center in shape.circle to float")
        assert center[0] > 0 and center[1] > 0, "center must be > 0"
        
        assert isinstance(AA, bool) or AA in (0, 1), \
        "AA value must be boolean-evaluable"

        return size, radii, center, AA        
        
    size, radii, center, AA = _check_types(size, radii, center, AA)

    c1 = circle(size, max(radii), center, AA)
    c2 = circle(size, min(radii), center, AA)
    
    return c1-c2

def ellipse(size, axes, center=None, angle=0, AA=True):
    """ Returns an ellipse in a numpy array.
    
    arguments:
        size: size of array, 2-tuple (rows, columns)
        axes: the length of the axes (vertical "radius", horizontal "radius")
        center: 2-tuple recentering coordinate system to (row, column)
        angle: rotation angle in degrees. this uses the standard rotation
            matrix, but whether that corresponds to clockwise or ccw depends on
            how the y-axis is defined. check rotation direction before using!
            (if saved as .fits, +angle is ccw and -angle is cw)
        AA: if True, anti-aliases the edge of the ellipse

    returns:
        numpy array with an ellipse drawn.
    """
    
    def _check_types(size, axes, center, angle, AA):
        """ Check types """
        
        assert isinstance(size, (list, tuple)) and len(size) == 2,\
        "size must be a list or tuple of length 2"
        
        try:
            size = (int(size[0]), int(size[1]))
        except:
            raise ValueError("couldn't cast size in shape.ellipse to integers")
        assert size[0] > 0 and size[1] > 0, "size must be > 0"
        
        assert isinstance(axes, (list, tuple)) and len(axes) == 2,\
        "axes must be a list or tuple of length 2"
        
        try:
            axes = (float(axes[0]), float(axes[1]))
        except:
            raise ValueError("couldn't cast axes in shape.ellipse to float")
        assert axes[0] > 0 and axes[1] > 0, "axes must be > 0"
        
        if center == None:
            center = (size[0]/2, size[1]/2)
        assert isinstance(center, (list, tuple)) and len(center) == 2,\
        "center must be a tuple or list of length 2"
        
        try:
            center = (float(center[0]), float(center[1]))
        except:
            raise ValueError("couldnt cast center in shape.ellipse to float")
        assert center[0] > 0 and center[1] > 0, "center must be > 0"
        
        try:
            angle = float(angle)
        except:
            raise ValueError("couldnt cast angle in shape.ellipse to float")
        
        assert isinstance(AA, bool) or AA in (0, 1), \
        "AA value must be boolean evaluable"
        
        return size, axes, center, angle, AA
    
    size, axes, center, angle, AA = _check_types(size, axes, center, angle, AA)
    
    # we can do this like a circle by stetching the coordinates along an axis
    rows, cols = _make_indices(size, center, angle)

    ratio = float(axes[1])/float(axes[0])
    if ratio >= 1:
        rows *= float(axes[1])/axes[0]
        radius = axes[1]
    if ratio < 1:
        cols *= 1./ratio
        radius = axes[0]
    
    # now with the stretched coordinate system the evaluation is just like
    # that for a circle
    r = np.hypot(rows, cols)
    if not AA:
        return np.where(r < radius, 1, 0)
    if AA:
        return 1-np.clip(r-radius, 0, 1)
        
def gaussian(size, lengths, center=None, angle=0, normalization=None):
    """ Returns an 1-dimensional or 2-dimensional gausssian. This implements

        f(x) = exp(-(x-x0)^2/(2*sigma^2)
    
    where x0 are the center coordinate(s) and sigma are the length(s).
    
    arguments:
        size: size of array (rows, columns). must be int.
        lengths: stdevs of gaussian (rows, columns). float or int.
        center: Coordinates to place the gaussian.  The center coordinates can
            be larger than the size, in that case the peak will be outside the
            array.
        angle: if it's a 2d gaussian, rotates the gaussian by an angle.  The
            angle is in degrees.
        normalization: Normalize the integrated gaussian to the value
            normalization.

    returns:
        numpy array (1d or 2d) with a gaussian of the given parameters.
    """
    
    def _check_types(size, lengths, center, angle, norm):
        """ Check types """
        
        assert isinstance(size, (list, tuple)) and len(size) in (1, 2),\
        "size must be a 1d or 2d list or tuple"
        
        ts = []
        for entry in size:
            try:
                x = int(entry)
                assert x > 0, "size must be > 0"
                ts.append(x)
            except:
                raise ValueError("couldnt cast size to integer in \
                                 shape.gaussian")
            
        assert isinstance(lengths, (list, tuple)) and len(lengths) in (1, 2),\
        "lengths must be a 1d or 2d list or tuple"
        
        tl = []
        for entry in lengths:
            try:
                tl.append(float(entry))
            except:
                raise ValueError("couldnt cast lengths to float in \
                                 shape.gaussian")
            
        assert len(size) == len(lengths),\
        "size and length must be same len in shape.gaussian"
        
        tc = []
        if center == None:
            for entry in size:
                tc.append(entry/2)
        else:
            for entry in center:
                try:
                    x = float(entry)
                except:
                    raise ValueError("couldnt cast center to float in \
                                     shape.gaussian")
                assert x > 0, "center must be > 0 in shape.gaussian"
                tc.append(x)
                
        if norm != None:
            try:
                norm = float(norm)
            except:
                raise ValueError("couldnt cast normalization\
                                 to float in shape.gaussian")
        
        try:
            angle = float(angle)
        except:
            raise ValueError("couldnt cast angle to float in shape.gaussian")
            
        return ts, tl, tc, angle, norm
    
    # check types
    size, lengths, center, angle, normalization =\
                                                 _check_types(size, lengths,
                                                              center, angle,
                                                              normalization)

    # now build the gaussian. 
    if len(size) == 1:
        x = np.arange(size[0]).astype('float')
        s = lengths[0]
        c = center[0]
        gauss = np.exp(-(x-c)**2/(2*s**2))
        
    if len(size) == 2:
        
        if angle == 0:
            # if angle = 0, we can use separability for speed
            x = np.arange(size[0]).astype('float')
            s = lengths[0]
            c = center[0]
            gaussian1 = np.exp(-(x-c)**2/(2*s**2))
        
            x = np.arange(size[1]).astype('float')
            s = lengths[1]
            c = center[1]
            gaussian2 = np.exp(-(x-c)**2/(2*s**2))

            gauss = np.outer(gaussian1, gaussian2)
        
        if angle != 0:
            # if angle != 0, we have to evaluate the whole array because
            # the function  is no longer separable. this takes about 2x
            # as long as the angle = 0 case
            rows, cols = _make_indices(size, center, angle)
            gauss = np.exp(-rows**2/(2*lengths[0]**2))
            gauss *= np.exp(-cols**2/(2*lengths[1]**2))

    if normalization == None:
        return gauss
    else:
        return gauss*float(normalization)/(np.sum(gauss))

def _make_indices(size, center, angle):
    """Generate array indices for an array of size, centered on center rotated
    by angle.  The rotation is in a clockwise direction.

    arguments:
        size: The (rows, cols) size of the output array
        center: The center of the indexed array.
        angle: Angle to rotate the matrices, in degrees. The angle is
            clockwise.

    returns:
        rows, cols - two numpy arrays with the rows and cols indexed.
    """
    orows, ocols = np.indices(size).astype('float')
        
    orows -= center[0]
    ocols -= center[1]
    
    if angle != 0:
        angle *= np.pi/180.
        rows = orows*np.cos(angle)+ocols*np.sin(angle)
        cols = ocols*np.cos(angle)-orows*np.sin(angle)
        return rows, cols
        
    return orows, ocols

def lorentzian(size, widths, center=None, angle=0, normalization=None):
    """ Returns an 1-dimensional or 2-dimensional lorentzian. This implements

        f(x) = 1/( ((x-x0)/w)^2 + 1 )

    where x0 are the center coordinate(s) and w are the half-width(s) half-max.
    
    arguments:
        size: size of array (rows, columns). must be int.
        lengths: HWHM of lorentzian (rows, columns). float or int.
        center: Coordinates to place the lorentzian.  The center coordinates
            can be larger than the size, in that case the peak will be outside
            the array.
        angle: if it's a 2d lorentzian, rotates the lorentzian by an angle.
            The angle is in degrees.
        normalization: Normalize the integrated lorentzian to the value
            normalization.

    returns:
        numpy array (1d or 2d) with a gaussian of the given parameters.
    """
    
    assert isinstance(size, tuple), "size must be a tuple"
    for d in range(len(size)):
        assert type(size[d]) is int, "size values must be int"
    assert isinstance(widths, tuple), "widths must be a tuple"
    
    assert len(size) == len(widths),\
    "size and widths must be same dimensionality"
    
    for d in range(len(widths)):
        assert isinstance(widths[d], (int, float)),\
        "widths values must be float or int"
    
    if center != None:
        assert isinstance(center, tuple), "center must be supplied as a tuple"
        
        assert len(center) == len(size),\
        "size and center must be same dimensionality"
        
        for d in range(len(center)):
            assert isinstance(center[d], (int, float)),\
            "center values must be float or int"
    else:
        center = np.zeros_like(size)
        for d in range(len(center)):
            center[d] = size[d]/2.

    if normalization is not None:
        assert isinstance(normalization, (float, int)),\
        "normalization must be float or int"

    # now build the lorentzian.
    if len(size) == 1:
        x = np.arange(size[0]).astype('float')
        s = widths[0]
        c = center[0]
        lorentz = 1./(((x-c)/s)**2 + 1.)
        
    if len(size) == 2:
        
        if angle == 0:
            # if angle = 0, we can use separability for speed
            y = ((np.arange(size[0], dtype=np.float32)-center[0])/widths[0])**2
            x = ((np.arange(size[1], dtype=np.float32)-center[1])/widths[1])**2
            denom = 1+np.add.outer(y, x)
            lorentz = 1./denom
        
        if angle != 0:
            # if angle != 0, we have to evaluate the whole array because
            # the function is no longer separable. this takes about 2x
            # as long as the angle = 0 case
            rows, cols = _make_indices(size, center, angle)
            lorentz = 1. / ((rows/widths[0])**2+(cols/widths[1])**2+1)

    if normalization == None:
        return lorentz
    else:
        return lorentz*float(normalization)/lorentz.sum()

def radial_fermi_dirac(size, r, kt, center=None):
    """ Create a radial fermi dirac function, good for use as an apodizer """
    rad = radial(size, center)
    return 1/(1+np.exp((rad-r)/kt))
