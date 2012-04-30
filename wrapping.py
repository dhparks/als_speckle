import scipy
from types import *


def unwrap_plan(r,R,center,max_angle=None):
    """ Make the array which constitutes the unwrap plan. This is then passed
    to the actual unwrap function along with the array to be unwrapped. Generating
    a separate plan results in serious speed improvements if the unwrap plan
    is always the same
    
    Required inputs:
    r: the interior radius if the unwrapping annulus. float or int.
    R: the exterior radius of the unwrapping annulus. float or int.
    center: a 2-tuple of integers giving the center of the annulus.
    
    Optional input:
    max_angle: the portion of the azimuthal coordinate to be unwrapped.
            default is 2pi.
    """

    assert type(r) in (FloatType,IntType), "r must be int or float"
    assert type(R) in (FloatType,IntType), "R must be int or float"
    assert type(center) is TupleType and type(center[0]) is IntType and type(center[1]) is IntType, "center must be integer 2-tuple"
    if max_angle == None: max_angle = 2*scipy.pi
    assert type(max_angle) is FloatType, "max_angle must be float"

    # setup up polar arrays
    R2,R1,cx,cy = R,r,x,y
    
    cMax = int(MaxAngle*R2)
    RTable, CTable = scipy.indices((R2-R1,cMax),float)
    RTable += R1
    PhiTable = scipy.ones_like(RTable)*scipy.arange(cMax)*MaxAngle/cMax

    # now form the map
    Rows,Cols = RTable.shape
    N = Rows*Cols
    
    # basic trigonometry:
    # x = x0+r*cos(Theta)
    # y = y0+r*sin(Theta)
    x0Table = cx+RTable*scipy.cos(PhiTable)
    y0Table = cy+RTable*scipy.sin(PhiTable)
    
    # unravel to make 1-d (For speed? Why did I do this? Is this the required format?)
    x0Table = scipy.ravel(x0Table)
    y0Table = scipy.ravel(y0Table)
    
    # combine into the plan array
    # Plan format:
    # [[y0,y1,y2...R],[x0,x1,x2...r]]
    Plan = scipy.zeros((2,N+1),float)
    Plan[0,:-1] = y0Table
    Plan[1,:-1] = x0Table
    
    # very last entries in the plan are the radii. this means the plan contains all information for unwrapping.
    Plan[0,-1] = R
    Plan[1,-1] = r

    return Plan

def unwrap(array,plan,interpolation_order=3):
    import scipy.ndimage
    
    """ Given an array and a pre-generated plan, unwrap an array into polar coordinates
    
    Required input:
    array: data to be unwrapped
    plan:  description of unwrapping, obtained by invoking unwrap_plan()
    
    Optional input:
    interpolation_order: coordinate transformations require interpolation.
                        0 is nearest neighbor.
                        1 is linear/bilinear
                        3 is cubic/bicubic (default)
                        can go higher?"""
    
    # separate plan into coordinate map and (r,R)
    R,r = plan[:,-1]
    plan = plan[:,:-1]
    
    # unwrap
    unwrapped = scipy.ndimage.map_coordinates(data,plan,order=interpolation_order)
    
    # the unwrapped version is 1d. reshape into the correct 2d arrangement (this is why the plan needs r and R)
    unwrapped.shape = (R-r,len(plan[0])/(R-r))
    
    return unwrapped