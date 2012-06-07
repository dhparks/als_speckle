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
    assert type(size) is tuple, "size must be a tuple"
    
    if center != None:
        assert isinstance(center,tuple), "center must be supplied as a tuple"
        assert len(center) == len(size), "size and center must be same dimensionality"
    else:
        center = numpy.zeros(len(size),float)
        for d in range(len(size)): center[d] = size[d]/2.
        
    # evaluate pythagorean theorem in n-dimensions
    indices = numpy.indices(size,float)
    r = numpy.zeros_like(indices[0])

    if len(size) == 2: return numpy.hypot(indices[0]-center[0],indices[1]-center[1])
    else:
        for d in range(len(size)): r += (indices[d]-center[d])**2
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
    return rect(size, length, legnth, center)
    
def rect(size,row_length,col_length,center=None):
    """ Generate a rectangle in a numpy array.
    
    arguments:
        size: the size of the array as a tuple (rows,columns).
        row_length: row length of the rectangle.
        col_length: col length of the rectangle.
        center: the center of the coordinate system as a tuple (center_row,center_column)
    returns:
        A 2-dimensional numpy array with a rectangle of (row_length, col_length) centered at (center).
    """
    if center == None: center = (size[0]/2,size[1]/2)

    assert isinstance(size,tuple) and len(size) == 2, "size must be a 2-tuple"
    assert type(row_length) in (float, int) and type(col_length) in (float, int), "lengths must be float or integer"
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
    assert type(radius) in (int, float), "radius must be int or float"
    assert type(AA) is bool or AA in (0,1), "AA value must be boolean evaluable"

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
    assert type(radii) is tuple and type(radii[0]) in (int,float) and type(radii[1]) in (int,float), "radius must be a 2-tuple of floats or ints"
    # no need to do other asserts, circle() takes care of it.
    
    return circle(size,max(radii),center,AA)-circle(size,min(radii),center,AA)

def ellipse(size,axes,center=None,AA=True):
    """ Returns an ellipse in a numpy array.
    
    arguments:
        size: size of array, 2-tuple (rows,columns)
        axes: (vertical "radius", horizontal "radius")
        center: 2-tuple recentering coordinate system to (row,column)
        AA: if True, anti-aliases the edge of the ellipse
    returns:
        numpy array with an ellipse drawn.
    """
    
    # check types
    assert isinstance(size,tuple) and len(size) == 2, "size must be a 2-tuple"
    assert isinstance(size[0],int) and isinstance(size[1],int), "size values must be int"
    assert isinstance(axes,tuple) and len(axes) == 2, "axes must be a 2-tuple"
    assert type(axes[0]) in (int, float) and type(axes[1]) in (int,float), "axes values must be float or int"
    if center == None: center = (size[0]/2,size[1]/2)
    assert isinstance(center,tuple) and len(center) == 2, "center must be a 2-tuple"
    assert type(center[0]) in (int,float) and type(center[1]) in (int,float), "center values must be float or int"
    assert type(AA) is bool or AA in (0,1), "AA value must be bool-evaluable"
    
    # we can do this like a circle by stetching the coordinates along an axis
    rows,cols = numpy.indices(size).astype('float')
    rows += -center[0]
    cols += -center[1]
    
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
        
def gaussian(size,lengths,center=None,normalization=None):
    """ Returns an 1-dimensional or 2-dimensional gausssian.
    
    arguments:
        size: size of array (rows,columns). must be int.
        lengths: stdevs of gaussian (rows,columns). float or int.
        center: recenter coordinate system to (row,column).
                Boundary conditions are NOT cyclic.
    returns:
        numpy array (1d or 2d) with a gaussian of the given parameters.
    """
    
    # check types
    assert type(size) is tuple, "size must be a tuple"
    for d in range(len(size)): assert type(size[d]) is int, "size values must be int"
    assert type(lengths) is tuple, "lengths must be a tuple"
    assert len(size) == len(lengths), "size and lengths must be same dimensionality"
    for d in range(len(lengths)): assert type(lengths[d]) in (int,float), "lengths values must be float or int"
    
    if center != None:
        assert type(center) is tuple, "center must be supplied as a tuple"
        assert len(center) == len(size), "size and center must be same dimensionality"
        for d in range(len(center)): assert type(center[d]) in (int,float), "center values must be float or int"
    else:
        center = numpy.zeros_like(size)
        for d in range(len(center)): center[d] = size[d]/2.

    if normalization is not None:
        assert type(normalization) in (float,int), "normalization must be float or int"

    # now build the gaussian.
    
    if len(size) == 1:
        x = numpy.arange(size[0]).astype('float')
        s = lengths[0]
        c = center[0]
        gaussian = numpy.exp(-(x-c)**2/(2*s**2))
        
    if len(size) == 2:
        x = numpy.arange(size[0]).astype('float')
        s = lengths[0]
        c = center[0]
        gaussian1 = numpy.exp(-(x-c)**2/(2*s**2))
        
        x = numpy.arange(size[1]).astype('float')
        s = lengths[1]
        c = center[1]
        gaussian2 = numpy.exp(-(x-c)**2/(2*s**2))
        
        gaussian = numpy.outer(gaussian1,gaussian2)
            
    if normalization == None: return gaussian
    else: return gaussian*float(normalization)/(numpy.sum(gaussian))
