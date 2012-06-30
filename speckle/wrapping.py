""" A Library for wrapping and unwrapping arrays that have azimuthal symmetry.

Author: Daniel Parks (dparks@uoregon.edu)
"""

import numpy as np

def unwrap_plan(r,R,center,max_angle=None,modulo=None):
    """ Make the array which constitutes the unwrap plan. This is then passed
    to the actual unwrap function along with the array to be unwrapped. Generating
    a separate plan results in serious speed improvements if the unwrap plan
    is always the same.
    
    Required inputs:
    r: the interior radius if the unwrapping annulus. float or int.
    R: the exterior radius of the unwrapping annulus. float or int.
    center: a 2-tuple of integers giving the center of the annulus.
    
    Optional input:
    max_angle: the portion of the azimuthal coordinate to be unwrapped.
            default is 2pi.
    modulo: if put the coordinates through a modulo operation of this value.
        intended for use in unwrapping machine-centered speckle without
        requiring a fftshift. generally, this should be the array size.

    Returns:
    plan - an array object that contains the unwrapping plan.
        the plan is formatted as:
            [[y0,y1...yn,R],
             [x0,x1...xn,r]]
    """
    assert isinstance(r, (float, int)), "r must be int or float"
    assert isinstance(R, (float, int)), "R must be int or float"
    assert isinstance(center, (list, tuple, set)) and len(center) == 2, "center must be 2-tuple"
    # don't bother checking if center is int, just cast it. Found out that numpy.int64 type is not considered IntType
    #center = [int(i) for i in center]
    
    ur,uR = min([r,R]),max([r,R])
    assert ur > 0, "inner radius must be > 0"

    if max_angle == None: max_angle = 2*np.pi
    assert isinstance(max_angle, (float, int)), "max_angle must be float"
    max_angle = float(max_angle)

    # setup up polar arrays
    ucols = int(max_angle*uR)
    r,c = np.indices((uR-ur,ucols),float)
    phi = np.ones_like(r)*np.arange(ucols)*max_angle/ucols
    r += ur
    
    x = np.ravel(center[1]+r*np.cos(phi))
    y = np.ravel(center[0]+r*np.sin(phi))
    l = len(x)
    
    if modulo != None:
        assert isinstance(modulo,int), "modulo must be int"
        x = np.mod(x+modulo,modulo-1)
        y = np.mod(y+modulo,modulo-1)

    # combine into the plan array
    # Plan format:
    # [[y0,y1,y2...R],[x0,x1,x2...r]]
    plan = np.zeros((2,l+1),float)
    
    plan[0,:l] = y
    plan[1,:l] = x
    
    # very last entries in the plan are the radii. this means the plan contains all information for unwrapping.
    plan[0,-1] = uR
    plan[1,-1] = ur

    return plan

def wrap_plan(r,R,max_angle=None):
    """ Generate a plan to rewrap an array from polar coordinates into cartesian coordinates.
    Rewrapped array will be returned with coordinate center at array center.
    
    Required input:
    r: the inner raidus of the data to be wrapped.  int or float
    R: the outer radius of the data to be wrapped. int or float.
    
    Optional input:
    max_angle: if when unwrapping you used a max_angle other than the default value of 2*pi it should be resupplied here. radians.
    
    Returns:
    plan - a ndarray plan of coordinate wrapping map.  To be used in wrap_with_plan.
    """
    assert isinstance(R, (int, float)), "R must be float or int"
    assert isinstance(r, (int, float)), "r must be float or int"
    if max_angle == None: max_angle = 2*scipy.pi
    assert isinstance(max_angle,float), "max_angle must be float"
    
    R1 = max([r,R])
    R2 = min([r,R])
 
    # setup the cartesian arrays
    yArray,xArray = scipy.indices((2*R1,2*R1),float)-R1
    RArray = scipy.sqrt(yArray**2+xArray**2)-R2
    PhiArray = scipy.mod(scipy.angle(xArray+complex(0,1)*yArray)+2*scipy.pi,2*scipy.pi)
    PhiArray *= 1./PhiArray.max()*int(2*scipy.pi*R-1)

    # put them in the plan
    Plan = scipy.zeros((2,len(yArray),len(yArray[0])),float)
    Plan[0] = RArray
    Plan[1] = PhiArray
    
    return Plan

def unwrap(array,plan,interpolation_order=3,modulo=None):
    """ Given an array and a pre-generated plan, unwrap an array into polar coordinates
    
    Required input:
    array: data to be unwrapped. must be numpy ndarray
    plan or (r, R, center): Either a description of unwrapping, obtained by
        invoking unwrap_plan(), or the 3-tuple (r, R, center).  The latter method
        is useful if you only need to call unwrap() once.
    
    Optional input:
    interpolation_order: coordinate transformations require interpolation.
                        0 is nearest neighbor.
                        1 is linear/bilinear
                        3 is cubic/bicubic (default)
                        4 and 5 order is also possible

    returns:
    unwrapped_array: the unwrapped array. The size of this is (R-r,xxx)
    """
    import scipy.ndimage

    assert isinstance(array,scipy.ndarray),  "input data must be ndarray"
    assert interpolation_order in range(6), "interpolation order must be 0-5"
    if not isinstance(plan,scipy.ndarray):
        assert len(plan) == 3, "unwrap plan must be a len 2 list/tuple/set or a ndarray"
        r, R, center = plan
        plan = unwrap_plan(r, R, center, modulo=modulo)
    
    # separate plan into coordinate map and (r,R)
    R,r = plan[:,-1]
    plan = plan[:,:-1]
    
    l1,l2 = array.shape
    ymax = plan[0].max()
    xmax = plan[1].max()
    
    assert ymax <= l1 and xmax <= l2, "max unwrapped coordinates fall outside array.\ndid you use the correct plan for this array size?"
    
    # unwrap
    unwrapped = scipy.ndimage.map_coordinates(array,plan,order=interpolation_order)
    
    # the unwrapped version is 1d. reshape into the correct 2d arrangement (this is why the plan needs r and R)
    unwrapped.shape = (R-r,len(plan[0])/(R-r))
    plan.shape = (2,R-r,len(plan[0])/(R-r))
    
    
    return unwrapped,plan

def wrap(array,plan,interpolation_order=3):
    """ Wraps data from polar coordinates into cartesian coordinates. A plan must be supplied.
    This is basically just a wrapper to scipy.ndimage.map_coordinates with all the real work
    being done in generating the plan.
    
    Required input:
    array: data in polar coordinates (y axis is radial coordinate, x axis is angular coordinate)
    plan or (r, R): Either a description of unwrapping obtained by calling wrap_plan(), or a 2-tuple (r, R).

    Optional input:
    interpolation_order: coordinate transformations require interpolation.
                        0 is nearest neighbor.
                        1 is linear/bilinear
                        3 is cubic/bicubic (default)
                        4 and 5 order is also possible
    """
    import scipy
    import scipy.ndimage
    
    assert isinstance(array,scipy.ndarray),  "input data must be ndarray"
    assert interpolation_order in range(6), "interpolation order must be 0-5"
    if isinstance(plan, scipy.ndarray):
        pass
    else:
        assert len(plan) == 2, "plan must be len-2 tuple/set/list or ndarray"
        plan = wrap_plan(plan[0], plan[1])

    return scipy.ndimage.map_coordinates(array, plan, order=interpolation_order)
