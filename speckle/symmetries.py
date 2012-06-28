import numpy
from . import wrapping, crosscorr
DFT = numpy.fft.fft2
IDFT = numpy.fft.ifft2

def make_cosines(components,N):
    """ Generates cosines to use in cosine decompsition of an autocorrelation.
    
    arguments:
        components: an iterable list of which cosine components to generate
        N: length of unwrapped autocorrelation
        
    returns:
        ndarray of shape (len(components),N) containing cosine values"""
        
    assert isinstance(components,(tuple,list,numpy.ndarray)), "components must be iterable"
    assert isinstance(N,int), "N must be int"

    x = (numpy.arange(N).astype(float))*(2*numpy.pi/N)
    cosines = numpy.zeros((len(components),N),float)
    for n,c in enumerate(components): cosines[n] = x*c
    return numpy.cos(cosines)

def decompose(ac,cosines):
    
    """ Do an explicit cosine decomposition by multiply-sum method
    
    arguments:
        ac: incoming angular autocorrelation
        cosines: array of evaluated cosine values
        
    returns:
        cosine spectrum, shape (len(ac),len(cosines))"""
        
    assert isinstance(ac,numpy.ndarray), "ac must be array"
    assert isinstance(cosines,numpy.ndarray), "cosines must be array"
    assert len(cosines[0]) == len(ac[0]), "cosines are wrong shape"
    
    N = float(len(ac[0]))
    decomposition = numpy.zeros((len(ac),len(cosines)),float)
    for y,row in enumerate(ac):
        for x, cosine in enumerate(cosines):
            decomposition[y,x] = numpy.sum(row*cosine)
    return decomposition

def rot_sym(speckles,plan=None,components=None,cosines=None):
    """ Given a speckle pattern, decompose its angular autocorrelation into a cosine series.
    
    arguments:
        speckle: the speckle pattern to be analyzed. Should be human-centered
            (ie, center of speckle is at center of array) rather than machine-centered
            (ie, center of speckle is at corner of array).
            
        plan: either an unwrap plan from wrapping.unwrap_plan or a tuple of form (r,R) or (r,R,(center))
            describing the range of radii to be analyzed. If nothing is supplied, the unwrapping
            will by default be as extensive as possible: (r,R) = (0,N/2).
            
        components: an iterable set of integers describing which cosine components to analyze.
            If nothing is supplied, this will be all even numbers between 2 and 20.
            
        cosines: an ndarray containing precomputed cosines. for speed. if cosines is supplied,
            components is ignored.
            
        returns:
            an ndarray of shape (R-r,len(components)) giving the cosine component values of the
            decomposition."""
            
    # check types
    assert isinstance(speckles,numpy.ndarray) and speckles.ndim == 2, "input data must be 2d array"
    assert isinstance(plan,(numpy.ndarray,tuple,list,type(None))), "plan type is unrecognized"
    assert isinstance(components,(numpy.ndarray,tuple,list,type(None))), "components are non-iterable"
    
    N,M = speckles.shape
    R = min([N,M])
    
    # do the unwrapping. behavior depends on what comes in as plan
    if isinstance(plan,numpy.ndarray):
        unwrapped = wrapping.unwrap(speckles,plan)
    if isinstance(plan,tuple):
        if len(plan) == 2: unwrapped = wrapping.unwrap(speckles,(plan[0],plan[1],(N/2,M/2)))
        if len(plan) == 3: unwrapped = wrapping.unwrap(speckles,plan)
    if plan == None: unwrapped = wrapping.unwrap(speckles,(0,R/2,(N/2,M/2)))
        
    # autocorrelate the unwrapped speckle. normalize each row individually.
    autocorrelation = crosscorr.crosscorr_axis(unwrapped,axis=1)
    for row,row_data in enumerate(autocorrelation.shape[0]):
        autocorrelation[row] = row_data*(1./abs(row_data).max())
    
    # generate components and cosines if necessary
    if components == None: components = numpy.arange(2,20,2).astype('float')
    if cosines == None: cosines = make_cosines(components,len(autocorrelation[0]))
    
    # run cosine decomposition
    decomposition = decompose(autocorrelation,cosines)

    return decomposition
    
    
    
    