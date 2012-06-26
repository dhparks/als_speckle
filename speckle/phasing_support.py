import numpy

def bound(data,threshold=1e-10,force_to_square=False,pad=0):
    # find the minimally bound non-zero region of the support. useful
    # for storing arrays so that the zero-padding for oversampling is avoided.
    
    data = numpy.where(data > threshold,1,0)
    rows,cols = data.shape
    
    rmin,rmax,cmin,cmax = 0,0,0,0
    
    for row in range(rows):
        if data[row,:].any():
            rmin = row
            break
            
    for row in range(rows):
        if data[rows-row-1,:].any():
            rmax = rows-row
            break
            
    for col in range(cols):
        if data[:,col].any():
            cmin = col
            break
    
    for col in range(cols):
        if data[:,cols-col-1].any():
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
        
    return numpy.array([rmin,rmax,cmin,cmax]).astype('int32')
    
def align_global_phase(data):
    """ Phase retrieval is degenerate to a global phase factor. This function tries to align the global phase rotation
    by minimizing the amount of power in the imag component. Real component could also be minimized with no effect
    on the outcome.
    
    arguments:
        data: 2d or 3d ndarray whose phase is to be aligned. Each frame of data is aligned independently.
        
    returns:
        complex ndarray of same shape as data"""
        
    from scipy.optimize import fminbound
    
    # check types
    assert isinstance(data,numpy.ndarray), "data must be array"
    assert data.ndim in (2,3), "data must be 2d or 3d"
    assert numpy.iscomplexobj(data), "data must be complex"
    was2d = False
    
    if data.ndim == 2:
        was2d = True
        data.shape = (1,data.shape[0],data.shape[1])
        
    for frame in data:
        x = frame.ravel()
        e = lambda p: numpy.sum(abs((x*numpy.exp(complex(0,1)*p)).imag))
        opt, val, conv, num = fminbound(e,0,2*numpy.pi,full_output=1)
        print opt
        frame *= numpy.exp(complex(0,1)*opt)
    
    if was2d: data = data[0]
    
    return data


