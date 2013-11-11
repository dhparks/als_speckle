""" A Library for wrapping and unwrapping arrays that have azimuthal symmetry.

Author: Daniel Parks (dparks@uoregon.edu)
"""

import numpy as np

def unwrap_plan(r, R, center, columns=None, modulo=None, target='cpu'):
    """ Make the array which constitutes the unwrap plan. This is then passed
    to the actual unwrap function along with the array to be unwrapped.
    Generating a separate plan results in serious speed improvements if the
    unwrap plan is always the same.
    
    arguments:
        r - the interior radius if the unwrapping annulus. float or int.
        R - the exterior radius of the unwrapping annulus. float or int.
        center - a 2-tuple of integers giving the center of the annulus.
        columns - number of columns in the unwrapped array. By default, the data
            is not resampled after unwrapping and the number of columns is
            2*pi*R. For large images or 3d arrays, this number of columns might
            be too large.
        modulo - if put the coordinates through a modulo operation of this
            value. This is intended for use in unwrapping machine-centered
            speckle without requiring a fftshift. Generally, this should be the
            array size.
        target - (optional) either 'cpu' (default) or 'gpu'

    returns:
        plan - an array object that contains the unwrapping plan. The plan is
            formatted as:
                [[y0,y1...yn,R],
                 [x0,x1...xn,r]]
    """
    assert isinstance(r, (float, int)), "r must be int or float"
    assert isinstance(R, (float, int)), "R must be int or float"
    assert isinstance(center, (list, tuple, set)) and len(center) == 2, "center must be 2-tuple"
    # don't bother checking if center is int, just cast it. Found out that numpy.int64 type is not considered IntType
    #center = [int(i) for i in center]
    
    ur,uR = min([r,R]),max([r,R])
    assert ur >= 0, "inner radius must be >= 0"

    # setup up polar arrays
    if columns == None: ucols = int(2*np.pi*uR)
    else: ucols = columns
    r,c = np.indices((uR-ur,ucols),float)
    phi = np.ones_like(r)*np.arange(ucols)*2*np.pi/ucols
    r += ur
    
    y = r*np.sin(phi)+center[0]
    x = r*np.cos(phi)+center[1]
    
    if modulo != None:
        assert isinstance(modulo,int), "modulo must be int"
        x = np.mod(x+modulo,modulo-1)
        y = np.mod(y+modulo,modulo-1)
        
    if target == 'gpu':
        plan = (y,x)
        
    if target == 'cpu':
        #Plan format:
        # [[y0,y1,y2...R],[x0,x1,x2...r]]
        plan = np.zeros((2,x.size+1),float)
        plan[0,:-1] = np.ravel(y)
        plan[1,:-1] = np.ravel(x)
        plan[0,-1]  = uR
        plan[1,-1]  = ur
        
    return plan

def wrap_plan(r, R):
    """ Generate a plan to rewrap an array from polar coordinates into cartesian
    coordinates. Rewrapped array will be returned with coordinate center at
    the array center.
    
    arguments:
        r - the inner raidus of the data to be wrapped.  int or float
        R - the outer radius of the data to be wrapped. int or float.

    returns:
        plan - a ndarray plan of coordinate wrapping map.  To be used in wrap().
    """
    assert isinstance(R, (int, float)), "R must be float or int"
    assert isinstance(r, (int, float)), "r must be float or int"
    
    R1 = max([r,R])
    R2 = min([r,R])
 
    # setup the cartesian arrays
    yArray,xArray = np.indices((2*R1,2*R1),float)-R1
    RArray = np.sqrt(yArray**2+xArray**2)-R2
    PhiArray = np.mod(np.angle(xArray+complex(0,1)*yArray)+2*np.pi,2*np.pi)
    PhiArray *= 1./PhiArray.max()*int(2*np.pi*R-1)

    # put them in the plan
    Plan = np.zeros((2,len(yArray),len(yArray[0])),float)
    Plan[0] = RArray
    Plan[1] = PhiArray
    
    return Plan

def unwrap(array,plan,interpolation_order=3,modulo=None,columns=None):
    """ Given an array and a plan, unwrap an array into polar coordinates.
    This is a cpu-only function.
    
    arguments:
        array - data to be unwrapped. must be numpy ndarray
        plan or (r, R, center) - Either a description of unwrapping, obtained by
            invoking unwrap_plan(), or the 3-tuple (r, R, center).  The former
            method is useful if you only need to call unwrap() many times with
            the same 3-tuple values.
        interpolation_order - (optional) coordinate transformation interpolation order.
            0 is nearest neighbor.
            1 is linear/bilinear
            3 is cubic/bicubic (default)
            4 and 5 order is also possible

    returns:
        unwrapped - the unwrapped array. The size of this is (R-r,2*pi*R)
    """
    import scipy.ndimage

    assert isinstance(array,np.ndarray),  "input data must be ndarray"
    assert array.ndim in (2,3), "input data must be 2d or 3d"
    assert interpolation_order in range(6), "interpolation order must be 0-5"
    
    if columns != None:
        try: columns = int(columns)
        except TypeError:
            print "error casting columns=%s to int in speckle.wrapping.unwrap"%columns
            exit()
    
    if not isinstance(plan,np.ndarray):
        assert len(plan) == 3, "unwrap plan must be a len 2 list/tuple/set or a ndarray"
        r, R, center = plan
        plan = unwrap_plan(r, R, center, modulo=modulo, columns=columns)
    

    # separate plan into coordinate map and (r,R)
    R,r = plan[:,-1]
    plan = plan[:,:-1]
    rows, cols = R-r, len(plan[0])/(R-r)
    
    # cast dimensions
    was_2d = True
    if array.ndim == 3: was_2d = False
    if array.ndim == 2: array.shape = (1,)+array.shape
    
    l0,l1,l2 = array.shape
    ymax = plan[0].max()
    xmax = plan[1].max()
    
    assert ymax <= l1 and xmax <= l2, "max unwrapped coordinates fall outside array.\ndid you use the correct plan for this array size?"

    # unwrap each frame. reshape, then store in unwrapped
    unwrapped = np.zeros((array.shape[0],int(rows),int(cols)),np.float32)
    for nf in range(array.shape[0]):
        uw = scipy.ndimage.map_coordinates(array[nf],plan,order=interpolation_order)
        uw.shape = (R-r,len(plan[0])/(R-r))
        unwrapped[nf] = uw

    # restore the shape of the array if it was 2d
    if was_2d:
        array.shape = array.shape[1:]
        unwrapped   = unwrapped[0]

    return unwrapped

def wrap(array,plan,interpolation_order=1):
    """ Wraps data from polar coordinates into cartesian coordinates. A plan
    must be supplied. This is basically just a wrapper to scipy.ndimage.map_coordinates
    with all the real work being done in generating the plan. The plan can be a
    2-tuple of the inner and outer radii (r, R).
    
    arguments:
        array - data in polar coordinates (y axis is radial coordinate, x axis
            is angular coordinate)
        plan or (r, R) - Either a description of unwrapping obtained by calling
            wrap_plan(), or a 2-tuple (r, R) of inner/outer radii.
        interpolation_order - coordinate transformation interpolation order.
            0 is nearest neighbor.
            1 is linear/bilinear
            3 is cubic/bicubic (default)
            4 and 5 order is also possible
    
    returns:
        wrapped - A 2d wrapped array according to the plan.
    """
    import scipy.ndimage
    
    assert isinstance(array,np.ndarray),  "input data must be ndarray"
    assert interpolation_order in range(6), "interpolation order must be 0-5"
    if isinstance(plan, np.ndarray):
        pass
    else:
        assert len(plan) == 2, "plan must be len-2 tuple/set/list or ndarray"
        plan = wrap_plan(plan[0], plan[1])

    return scipy.ndimage.map_coordinates(array, plan, order=interpolation_order)

def resize_plan(shape_in,shape_out,target='cpu'):
    
    """ This makes the resizing plan to transform a rectangular array of shape
    shape_in into a rectangular array of shape_out. This uses map_coordinates
    to circumvent the other method of resizing which uses PIL.
    
    arguments:
        shape_in - shape of array to be input. must be length 2.
        shape_out - shape of array to be output. must be length 2.
        target - (optional) either 'cpu' (default) or 'gpu' depending on which device does the calculation"""
    
    # uR-ur gives the number of rows
    # columns_in gives the number of columns in the image being resize
    # columns_out gives the number of columns in the image after resizing
    
    assert isinstance(shape_in,(tuple,list,np.ndarray)), "shape_in must be interable (list, tuple, ndarray)"
    assert isinstance(shape_out,(tuple,list,np.ndarray)), "shape_out must be iterable (list, tuple, ndarray)"
    assert len(shape_in) == 2, "shape_in must be 2d"
    assert len(shape_out) == 2, "shape_out must be 2d"
    
    r_in, c_in = shape_in
    r_out, c_out = shape_out
    
    assert isinstance(r_in,int) and isinstance(c_in,int), "shape_in must be composed of integers"
    assert isinstance(r_out,int) and isinstance(c_out,int), "shape_out must be composed of integers"
    
    cols = np.arange(c_out).astype(np.float32)*c_in/float(c_out)
    rows = np.arange(r_out).astype(np.float32)*r_in/float(r_out)
    
    rows = np.outer(rows,np.ones(c_out,int))
    cols = np.outer(np.ones(r_out,int),cols)

    # resizing on gpu vs on cpu requires a different plan format to use
    # in scipy.nd_image.map_coordinates vs map_coords.cl. This may change in the future.
    if target == 'gpu':
        plan = (rows,cols)
    
    if target == 'cpu':
        plan = np.zeros((2,cols.size+1),float)

        plan[0,:-1] = np.ravel(rows)
        plan[1,:-1] = np.ravel(cols)
        plan[0,-1] = r_out
        plan[1,-1] = c_out
    
    return plan

def resize(array,plan,interpolation_order=3):
    
    """ Resizes data in cartesian coordinates. A plan must be supplied, either
    in the form of the plan generated by resize_plan or in the form of a tuple
    describing the shape in (rows,cols) of the resized data.
    
    arguments:
        data - data in cartesian coordinate
        plan or (r, R) - Either a description of unwrapping obtained by calling
            resize_plan(), or a 2-tuple (rows, cols) of new shape to interpolate.
        interpolation_order - coordinate transformation interpolation order.
            0 is nearest neighbor.
            1 is linear/bilinear
            3 is cubic/bicubic (default)
            4 and 5 order is also possible
    
    returns:
        resized - A 2d wrapped array according to the plan.
    """
    
    import scipy.ndimage
    
    assert isinstance(array,np.ndarray),  "input data must be ndarray"
    assert array.ndim == 2, "input data must be 2d"
    assert interpolation_order in range(6), "interpolation order must be 0-5"
    if isinstance(plan, np.ndarray):
        pass
    else:
        assert len(plan) == 2, "plan must be len-2 tuple/set/list or ndarray"
        plan = resize_plan((array.shape[0],array.shape[1]),(plan[0],plan[1]))

    # separate the plan and the shape
    r_out,c_out = plan[:,-1]
    plan        = plan[:,:-1]

    # map coordinates and resize the output
    resized = scipy.ndimage.map_coordinates(array, plan, order=interpolation_order)
    resized.shape = (r_out,c_out)
    
    return resized
