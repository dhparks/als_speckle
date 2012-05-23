#shapes
#	circle/ellipse (anti-aliased)
#	square/rect
#	gaussian
#	annulus
#	radial/angular

import scipy

def radial(size,center=None):
    
    """
    Returns azimuthally symmetric distance function in n-dimensions
    
    Required input:
    size: an n-tuple for the array size in format (rows,columns).
          must be integer.
    
    Optional input:
    center: an n-tuple giving origin of coordinate system (ie, where r = 0).
            Defaults to array center in absence of supplied values.
            Boundary conditions are NOT cyclic.
    """

    # check types
    assert type(size) is tuple, "size must be a tuple"
    
    if center != None:
        assert type(center) is tuple, "center must be supplied as a tuple"
        assert len(center) == len(size), "size and center must be same dimensionality"
    else:
        center = scipy.zeros(len(size),float)
        for d in range(len(size)): center[d] = size[d]/2.
        
    # evaluate pythagorean theorem in n-dimensions
    indices = scipy.indices(size,float)
    r = scipy.zeros_like(indices[0])
    for d in range(len(size)): r += (indices[d]-center[d])**2
    return scipy.sqrt(r)
    
def angular(size,center=None):
    
    """
    Returns radially symmetric angle function by means of atan2 (?)
    
    Required input:
    size: the size of the array as a tuple (rows,columns).
    
    Optional inputs:
    center: the center of the coordinate system as a tuple (center_row,center_column)
    """

    assert type(size) is tuple and len(size) == 2, "size must be given as a 2-tuple"
    
    if center == None: center = (size[0]/2,size[1]/2)
    assert type(center) is tuple and len(center) == 2, "center must be given as a 2-tuple"

    rows,cols = scipy.indices(size,float)
    rows += -center[0]
    cols += -center[1]
    return scipy.angle(cols+complex(0,1)*rows)
    
def square(size,length,center=None):
    
    # check types
    
    assert type(size) is tuple and len(size) == 2, "size must be a 2-tuple"
    assert type(length) is int, "length must be an integer"
    
    if center == None: center = (size[0]/2,size[1]/2)
    assert type(center) is tuple and len(center) == 2, "center must be a 2-tuple"
    
    temp = scipy.zeros(size,int)
    temp[center[0]-l/2:center[0]+l/2,center[1]-l/2:center[1]+l/2] = 1
    return temp
    
def rect(size,row_length,col_length,center=None):
    
    if center == None: center = (size[0]/2,size[1]/2)

    assert type(size) is tuple and len(size) == 2, "size must be a 2-tuple"
    assert type(row_length) is int and type(col_length) is int, "lengths must be integer"
    
    temp = scipy.zeros(size,int)
    temp[center[0]-row_length/2:center[0]+row_length/2,center[1]-col_length/2:center[1]+col_length/2] = 1
    return temp
    
def circle(size,radius,center=None,AA=True):
    
    """
    Returns a circle.
    
    Required input:
    1. size: a 2-tuple formatted as (rows,columns)
    2. radius of circle
    
    Optional input:
    center: a 2-tuple of (center_row,center_column) where the circle is centered
    AA: if AA is True, returns an antialiased circle. AA = False gives a jagged edge and is marginally faster
    """
    
    # check types
    assert type(size) is tuple and len(size) == 2, "size must be a 2-tuple"
    if center == None: center = (size[0]/2,size[1]/2)
    assert type(center) is tuple and len(center) == 2, "center must be a 2-tuple"
    assert type(radius) in (int, float), "radius must be int or float"
    assert type(AA) is bool or AA in (0,1), "AA value must be boolean evaluable"

    r = radial(size,center)
    
    if not AA: return scipy.where(r**2 < radius**2,1,0)
    if AA == 1:
        temp = r-radius
        temp[temp < 0] = 0
        temp[temp > 1] = 1
        return 1-temp
    
def annulus(size,radii,center=None,AA=True):
    """ Returns an annulus (ie a ring)
    
    Required input:
    size: size of array (rows,columns). tuple entries must be int.
    radii: interior and exterior radius of the annulus as a tuple. order is not important because smaller radius must be the interior.
    
    Optional input:
    center: a 2-tuple of (center_row,center_column) where the annulus is centered
    AA: if True, antialiases the annulus.
    
    """

    assert type(size) is tuple and len(size) == 2, "size must be a 2-tuple"
    if center == None: center = (size[0]/2,size[1]/2)
    assert type(center) is tuple and len(center) == 2, "center must be a 2-tuple"
    assert type(radii) is tuple and type(radii[0]) in (int,float) and type(radii[1]) in (int,float), "radius must be a 2-tuple of floats or ints"
    assert type(AA) is bool or AA in (0,1), "AA value must be boolean evaluable"
    
    return circle(size,max(radii),center,AA)-circle(size,min(radii),center,AA)
      
def ellipse(size,axes,center=None,AA=True):
    """ Returns an ellipse
    
    Required input:
    size: size of array, 2-tuple (rows,columns)
    axes: (vertical "radius", horizontal "radius")
    
    Optional input:
    center: 2-tuple recentering coordinate system to (row,column)
    AA: if True, anti-aliases the edge of the ellipse
    """
    
    # check types
    assert type(size) is tuple and len(size) == 2, "size must be a 2-tuple"
    assert type(size[0]) is int and type(size[1]) is int, "size values must be int"
    assert type(axes) is tuple and len(axes) == 2, "axes must be a 2-tuple"
    assert type(axes[0]) in (int, float) and type(axes[1]) in (int,float), "axes values must be float or int"
    if center == None: center = (size[0]/2,size[1]/2)
    assert type(center) is tuple and len(center) == 2, "center must be a 2-tuple"
    assert type(center[0]) in (int,float) and type(center[1]) in (int,float), "center values must be float or int"
    assert type(AA) is bool or AA in (0,1), "AA value must be boolean evaluable"
    
    # we can do this like a circle by stetching the coordinates along an axis
    rows,cols = scipy.indices(size).astype('float')
    rows += -center[0]
    cols += -center[1]
    
    ratio = float(axes[1])/float(axes[0])
    if ratio >= 1:
        rows *= float(axes[1])/axes[0]
        radius = axes[1]
    if ratio <= 1:
        cols *= 1./ratio
        radius = axes[0]
    
    # now with the stretched coordinate system the evaluation is just like that for a circle
    r = scipy.sqrt(rows**2+cols**2)
    if not AA: return scipy.where(r < radius**2,1,0)
    if AA:
        temp = r-radius
        temp[temp < 0] = 0.
        temp[temp > 1] = 1.

        return 1-temp
    
def gaussian(size,lengths,center=None,normalization=None):
    """ Returns an 1-dimensional or 2-dimensional gausssian.
    
    Required input:
    size: size of array (rows,columns). must be int.
    lengths: stdevs of gaussian (rows,columns). float or int.
    
    Optional input:
    center: recenter coordinate system to (row,column).
            Boundary conditions are NOT cyclic.
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
        center = scipy.zeros_like(size)
        for d in range(len(center)): center[d] = size[d]/2.
        
    assert type(normalization) in (None,float,int), "normalization must be float or int"
        
    # now build the gaussian.
    
    if len(size) == 1:
        x = scipy.arange(size[0]).astype('float')
        s = lengths[0]
        c = center[0]
        gaussian = scipy.exp(-(x-c)**2/(2*s**2))
        
    if len(size) == 2:
        x = scipy.arange(size[0]).astype('float')
        s = lengths[0]
        c = center[0]
        gaussian1 = scipy.exp(-(x-c)**2/(2*s**2))
        
        x = scipy.arange(size[1]).astype('float')
        s = lengths[1]
        c = center[1]
        gaussian2 = scipy.exp(-(x-c)**2/(2*s**2))
        
        gaussian = scipy.outer(gaussian1,gaussian2)
            
    if normalization == None: return gaussian
    else: return gaussian*float(normalization)/(scipy.sum(gaussian))
