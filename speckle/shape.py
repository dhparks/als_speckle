"""A library for calculating image shapes and functions

Author: Daniel Parks (dparks@uoregon.edu)
"""
import numpy

def radial(size,center=None):
    """ Returns azimuthally symmetric distance function in n-dimensions
    
    arguments:
        size: an n-tuple for the array size in format (rows,columns).
              Must be integer.
        center: an n-tuple giving origin of coordinate system (ie, where r = 0).
                Defaults to array center in absence of supplied values.
                Boundary conditions are NOT cyclic.
    returns:
        A n-dimensioanl array where each element is the distance (in pixels) from the center.
   """
    assert isinstance(size,tuple), "size must be a tuple"
    ndims = len(size)
    
    if center != None:
        assert isinstance(center,tuple), "center must be supplied as a tuple"
        assert len(center) == ndims, "size and center must be same dimensionality"
    else:
        center = numpy.zeros(ndims,float)
        for d in range(ndims): center[d] = size[d]/2.
        
    # evaluate pythagorean theorem in n-dimensions
    indices = numpy.indices(size,float)
    r = numpy.zeros_like(indices[0])

    if ndims == 2:
        return numpy.hypot(indices[0]-center[0],indices[1]-center[1])
    else:
        for d in range(ndims): r += (indices[d]-center[d])**2
        return numpy.sqrt(r)
    
def angular(size,center=None):    
    """ Generate a radially symmetric angle function.
    
    arguments:
        size: the size of the array as a tuple (rows,columns).
        center: the center of the coordinate system as a tuple (center_row,center_column)
    returns:
        A 2-dimensional array of the angular values, in radians.
    """

    assert isinstance(size,tuple) and len(size) == 2, "size must be a 2-tuple"
    
    if center == None: center = (size[0]/2,size[1]/2)
    assert isinstance(center,tuple) and len(center) == 2, "center must be a 2-tuple"

    rows,cols = numpy.indices(size,float)
    rows += -center[0]
    cols += -center[1]
    return numpy.angle(cols+complex(0,1)*rows)
    
def square(size,length,center=None):
    """ Generate a square in a numpy array.
    
    arguments:
        size: the size of the array as a tuple (rows,columns).
        length: length of the square.
        center: the center of the coordinate system as a tuple (center_row,center_column)
    returns:
        A 2-dimensional numpy array with a square of length (length) centered at (center).
    """
    return rect(size, (length, length), center)
    
def rect(size,lengths,center=None):
    """ Generate a rectangle in a numpy array.
    
    arguments:
        size: the size of the array as a tuple (rows,columns).
        lengths: a 2-tuple formatted as (rows_length,col_length).
        center: the center of the coordinate system as a tuple (center_row,center_column)
    returns:
        A 2-dimensional numpy array with a rectangle of (lengths) centered at (center).
    """
    if center == None: center = (size[0]/2,size[1]/2)

    assert isinstance(size,tuple) and len(size) == 2, "size must be a 2-tuple"
    assert isinstance(lengths,tuple) and len(lengths) == 2, "lengths must be tuple"
    row_length,col_length = lengths
    assert isinstance(row_length,(float, int)) and isinstance(col_length,(float, int)), "lengths must be float or integer, but will be cast to int"
    
    row_length = int(row_length)
    col_length = int(col_length)
    
    temp = numpy.zeros(size,int)
    temp[center[0]-row_length/2:center[0]+row_length/2,center[1]-col_length/2:center[1]+col_length/2] = 1
    return temp
    
def circle(size,radius,center=None,AA=True):
    """ Generate a circle in a numpy array.
    
    arguments:
        size: a 2-tuple formatted as (rows,columns)
        radius of circle
        center: a 2-tuple of (center_row,center_column) where the circle is centered
        AA: if AA is True, returns an antialiased circle. AA = False gives a jagged edge and is marginally faster
    returns:
        a numpy array of size with a circle of radius centered on center.
    """
    
    # check types
    assert isinstance(size,tuple) and len(size) == 2, "size must be a 2-tuple"
    if center == None: center = (size[0]/2,size[1]/2)
    assert isinstance(center,tuple) and len(center) == 2, "center must be a 2-tuple"
    assert isinstance(radius,(int, float)), "radius must be int or float"
    assert isinstance(AA,bool) or AA in (0,1), "AA value must be boolean evaluable"

    r = radial(size,center)
    
    if not AA: return numpy.where(r < radius,1,0)
    if AA: return 1-numpy.clip(r-radius,0,1.) 
    
def annulus(size,radii,center=None,AA=True):
    """ Returns an annulus (ie a ring) in a numpy array.

    arguments:
        size: size of array (rows,columns). tuple entries must be int.
        radii: interior and exterior radius of the annulus as a tuple. The order is not important because smaller radius must be the interior.
        center: a 2-tuple of (center_row,center_column) where the annulus is centered
        AA: if True, antialiases the annulus.
    returns:
        a numpy array with annuls centered on center of radius (r_in, r_out)    
    """
    assert isinstance(radii,tuple) and isinstance(radii[0],(int,float)) and isinstance(radii[1],(int,float))
    # no need to do other asserts, circle() takes care of it.
    
    return circle(size,max(radii),center,AA)-circle(size,min(radii),center,AA)

def ellipse(size,axes,center=None,angle=0,AA=True):
    """ Returns an ellipse in a numpy array.
    
    arguments:
        size: size of array, 2-tuple (rows,columns)
        axes: (vertical "radius", horizontal "radius")
        center: 2-tuple recentering coordinate system to (row,column)
        angle: rotation angle in degrees. this uses the standard rotation matrix, but whether that corresponds
            to cw or ccw depends on how the y-axis is defined. check rotation direction before using!
            (if saved as .fits, +angle is ccw and -angle is cw)
        AA: if True, anti-aliases the edge of the ellipse
    returns:
        numpy array with an ellipse drawn.
    """
    
    # check types
    assert isinstance(size,tuple) and len(size) == 2, "size must be a 2-tuple"
    assert isinstance(size[0],int) and isinstance(size[1],int), "size values must be int"
    assert isinstance(axes,tuple) and len(axes) == 2, "axes must be a 2-tuple"
    assert isinstance(axes[0],(int, float)) and isinstance(axes[1],(int,float)), "axes values must be float or int"
    if center == None: center = (size[0]/2,size[1]/2)
    assert isinstance(center,tuple) and len(center) == 2, "center must be a 2-tuple"
    assert isinstance(center[0],(int, float)) and isinstance(center[1],(int,float)), "center values must be float or int"
    assert isinstance(angle,(int,float)), "angle must be int or float"
    assert isinstance(AA,bool) or AA in (0,1), "AA value must be boolean evaluable"
    
    # we can do this like a circle by stetching the coordinates along an axis
    rows,cols = _make_indicies(size,center,angle)

    ratio = float(axes[1])/float(axes[0])
    if ratio >= 1:
        rows *= float(axes[1])/axes[0]
        radius = axes[1]
    if ratio < 1:
        cols *= 1./ratio
        radius = axes[0]
    
    # now with the stretched coordinate system the evaluation is just like that for a circle
    r = numpy.hypot(rows,cols)
    if not AA: return numpy.where(r < radius,1,0)
    if AA: return 1-numpy.clip(r-radius,0,1)
        
def gaussian(size,lengths,center=None,angle=0,normalization=None):
    """ Returns an 1-dimensional or 2-dimensional gausssian. This implements

        f(x) = exp(-(x-x0)^2/(2*sigma^2)
    
    where x0 are the center coordinate(s) and sigma are the length(s).
    
    arguments:
        size: size of array (rows,columns). must be int.
        lengths: stdevs of gaussian (rows,columns). float or int.
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
    
    # check types
    assert isinstance(size,tuple), "size must be a tuple"
    for d in range(len(size)): assert type(size[d]) is int, "size values must be int"
    assert isinstance(lengths,tuple), "lengths must be a tuple"
    assert len(size) == len(lengths), "size and lengths must be same dimensionality"
    for d in range(len(lengths)): assert isinstance(lengths[d],(int,float)), "lengths values must be float or int"
    
    if center != None:
        assert isinstance(center,tuple), "center must be supplied as a tuple"
        assert len(center) == len(size), "size and center must be same dimensionality"
        for d in range(len(center)): assert isinstance(center[d],(int,float)), "center values must be float or int"
    else:
        center = numpy.zeros_like(size)
        for d in range(len(center)): center[d] = size[d]/2.

    if normalization is not None:
        assert isinstance(normalization,(float,int)), "normalization must be float or int"

    # now build the gaussian. 
    
    if len(size) == 1:
        x = numpy.arange(size[0]).astype('float')
        s = lengths[0]
        c = center[0]
        gaussian = numpy.exp(-(x-c)**2/(2*s**2))
        
    if len(size) == 2:
        
        if angle == 0:
            # if angle = 0, we can use separability for speed
            x = numpy.arange(size[0]).astype('float')
            s = lengths[0]
            c = center[0]
            gaussian1 = numpy.exp(-(x-c)**2/(2*s**2))
        
            x = numpy.arange(size[1]).astype('float')
            s = lengths[1]
            c = center[1]
            gaussian2 = numpy.exp(-(x-c)**2/(2*s**2))

            gaussian = numpy.outer(gaussian1,gaussian2)
        
        if angle != 0:
            # if angle != 0, we have to evaluate the whole array because the function
            # is no longer separable. this takes about 2x as long as the angle = 0 case
            rows,cols = _make_indicies(size,center,angle)
            gaussian = numpy.exp(-rows**2/(2*lengths[0]**2))*numpy.exp(-cols**2/(2*lengths[1]**2))

    if normalization == None: return gaussian
    else: return gaussian*float(normalization)/(numpy.sum(gaussian))

def _make_indicies(size,center,angle):
    """Generate array indicies for an array of size, centered on center rotated by angle.  The rotation is in a clockwise direction.
    """
    rows,cols = numpy.indices(size).astype('float')
        
    rows -= center[0]
    cols -= center[1]
    
    if angle != 0:
        angle *= numpy.pi/180.
        rows = rows*numpy.cos(angle)+cols*numpy.sin(angle)
        cols = cols*numpy.cos(angle)-rows*numpy.sin(angle)
        
    return rows, cols

def lorentzian(size,widths,center=None,angle=0,normalization=None):
    """ Returns an 1-dimensional or 2-dimensional lorentzian. This implements

        f(x) = 1/( ((x-x0)/w)^2 + 1 )

    where x0 are the center coordinate(s) and w are the half-width(s) half-max.
    
    arguments:
        size: size of array (rows,columns). must be int.
        lengths: HWHM of lorentzian (rows,columns). float or int.
        center: Coordinates to place the lorentzian.  The center coordinates can
            be larger than the size, in that case the peak will be outside the
            array.
        angle: if it's a 2d lorentzian, rotates the lorentzian by an angle.  The
            angle is in degrees.
        normalization: Normalize the integrated lorentzian to the value
            normalization.

    returns:
        numpy array (1d or 2d) with a gaussian of the given parameters.
    """
    assert isinstance(size,tuple), "size must be a tuple"
    for d in range(len(size)): assert type(size[d]) is int, "size values must be int"
    assert isinstance(widths,tuple), "widths must be a tuple"
    assert len(size) == len(widths), "size and widths must be same dimensionality"
    for d in range(len(widths)): assert isinstance(widths[d],(int,float)), "widths values must be float or int"
    
    if center != None:
        assert isinstance(center,tuple), "center must be supplied as a tuple"
        assert len(center) == len(size), "size and center must be same dimensionality"
        for d in range(len(center)): assert isinstance(center[d],(int,float)), "center values must be float or int"
    else:
        center = numpy.zeros_like(size)
        for d in range(len(center)): center[d] = size[d]/2.

    if normalization is not None:
        assert isinstance(normalization,(float,int)), "normalization must be float or int"

    # now build the lorentzian.
    if len(size) == 1:
        x = numpy.arange(size[0]).astype('float')
        s = widths[0]
        c = center[0]
        lorentzian = 1./( ((x-c)/s)**2 + 1. )
        
    if len(size) == 2:
        
        if angle == 0:
            # if angle = 0, we can use separability for speed
            y = ((numpy.arange(size[0],dtype=float) - center[0])/widths[0])**2
            x = ((numpy.arange(size[1],dtype=float) - center[1])/widths[1])**2
            denom = 1 + numpy.add.outer(y,x)
            lorentzian = 1./denom
        
        if angle != 0:
            # if angle != 0, we have to evaluate the whole array because the function
            # is no longer separable. this takes about 2x as long as the angle = 0 case
            rows,cols = _make_indicies(size,center,angle)
            lorentzian = 1. / ( (rows/widths[0])**2 + (cols/widths[1])**2 + 1)

    if normalization == None:
        return lorentzian
    else:
        return lorentzian*float(normalization)/lorentzian.sum()

