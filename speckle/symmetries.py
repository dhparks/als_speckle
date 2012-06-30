""" CPU-based rotational symmetry code

Author: Daniel Parks (dhparks@lbl.gov)"""

import numpy as np
from . import shape, wrapping, crosscorr
DFT = np.fft.fft2
IDFT = np.fft.ifft2

class microscope():
    
    """ A class containing methods helpful in running a symmetry microscope analysis
    on an image. For now, this class is limited to simulations where the object being
    analyzed is square. Furthermore, speckle patterns are simulated at the same size
    as the object."""
    
    def __init__(self,object=None,unwrap=None,pinhole=None,coherence=None,components=None):
        
        # the class is structured in such a way that its parts can all be inited right now
        # or later on in the main script. the example will show both.
        
        self.can_has_object = False
        self.can_has_unwrap = False
        self.can_has_pinhole = False
        self.can_has_coherence = False
        self.cosines = None
        
        # if simulation information is supplied to __init__, set up those parts of the simulation.
        # otherwise, they will have to be done later from the main script.
        
        if object != None: self.set_object(object)
        if unwrap != None: self.set_unwrap(unwrap)
        if pinhole != None: self.set_pinhole(pinhole)
        if coherence != None: self.set_coherence(coherence)
        if components != None: self.set_cosines(components=components)
        
    def set_object(self,object):
        """ Takes a 2d array and makes it a correctly behaving object for symmetry microscope
        calculations (basically, make it cyclic).
        
        arguments:
            object: 2d numpy array
            
        returns: nothing, but generates class variables"""
        
        assert isinstance(object,np.ndarray) and object.ndim==2, "object must be 2d array"
        assert object.shape[0] == object.shape[1], "for now, object must be square"
        
        self.N = object.shape[0]
        
        # cyclify the object so that we can slice the object rather than roll. should probably
        # test which one is faster...
        #new_object = np.zeros((2*self.N,2*self.N),object.dtype)
        #new_object[0:self.N,0:self.N] = object
        #new_object[0:self.N,self.N:]  = object
        #new_object[self.N:,0:self.N]  = object
        #new_object[self.N:,self.N:]   = object
        self.object = object
        
        self.can_has_object = True
        
    def set_unwrap(self,params):
        """ Build and name the unwrap plan.
        
        Input:
            params: unwrap_r and unwrap_R. center is assumed to be at array center
        
        returns: nothing, but generates class variables"""
        
        assert self.can_has_object, "need to init object before unwrap"
        assert isinstance(params,(list,tuple,np.ndarray)), "unwrap params must be iterable"
        assert len(params) == 2, "must provide exactly two parameters: unwrap_r and unwrap_R"
        
        ur, uR = params[0],params[1]
        ur, uR = min([ur,uR]), max([ur,uR])
        self.rows = (uR-ur)
        assert ur > 0, "unwrap_r must be > 0"
        assert uR < self.N/2, "unwrap_R exceeds simulated speckle bounds"
        
        self.unwrap_plan = wrapping.unwrap_plan(ur,uR,(self.N/2,self.N/2))
        
        self.can_has_unwrap = True
        
    def set_pinhole(self,pinhole):
        
        """Set the pinhole function.
        
        arguments:
            pinhole: Either a number, in which case a circle is generated as the pinhole, or
            an array, in which case the array is set as the pinhole. This latter option allows
            for complicated illumination shapes to be supplied."""
        
        assert self.can_has_object, "need to init object before pinhole"
        assert isinstance(pinhole,(int,float,np.ndarray)), "pinhole must be number or array"
        
        if isinstance(pinhole,(int,float)):
            assert pinhole < self.N/2, "pinhole radius must be smaller than pinhole array size"
            #self.illumination = np.fft.fftshift(shape.circle((self.N,self.N),pinhole))
            self.illumination = shape.circle((self.N,self.N),pinhole)
            
        if isinstance(pinhole,np.ndarray):
            assert pinhole.shape == self.object.shape, "supplied pinhole must be same size as supplied object"
            self.illumination = pinhole
            
        self.can_has_pinhole = True
        
    def set_coherence(self,coherence):
        """ Make the coherence function"""
        
        assert self.can_has_object, "need to init object before coherence"
        assert isinstance(coherence,(tuple,list,np.ndarray)), "coherence needs to be tuple, list, or array"
        
        if isinstance(coherence,(list,tuple)):
            assert len(coherence) == 2, "coherence must have len 2"
            clx,cly = coherence[0],coherence[1]
            self.coherence = np.fft.fftshift(shape.gaussian((self.N,self.N),(cly,clx)))
        
        if isinstance(coherence,np.ndarray):
            assert coherence.shape == (self.N,self.N), "coherence must have NxN"
            self.coherence = coherence
            
        self.can_has_coherence = True
        
    def set_cosines(self,components=None,cosines=None):
        assert self.can_has_unwrap, "must set unwrap before cosines"
        
        if components != None:
            assert isinstance(components,(tuple,list,np.ndarray)), "components must be iterable"
            uL = (self.unwrap_plan.shape[1]-1)/self.rows
            self.cosines = make_cosines(components,uL)
            
        if cosines != None:
            # trust the user here...
            self.cosines = cosines
            
    def run_on_site(self,dy,dx,components=None,cosines=None):
        
        """ Run the symmetry microscope on the site of the object described by the roll coordinates
        dy, dx. Steps are:
        
        1. Roll the illumination
        2. Make the speckle
            2b: If coherence is specified, blur the speckle
        3. Unwrap the speckle
        4. Autocorrelate the unwrapped speckle
        5. Cosine-decompose the autocorrelation
        
        arguments:
            dy, dx: roll coordinates used as follows: np.roll(np.roll(d,r0,axis=0),r1,axis=1)
            components: (optional) decompose into these components
            cosines: (optional) precomputed cosines for speed
        """
        
        assert self.can_has_object, "no object loaded"
        assert self.can_has_pinhole, "no pinhole loaded"
        assert self.can_has_unwrap, "no unwrap loaded"
        assert isinstance(dy,int) and isinstance(dx,int), "site coordinates must be integer"
        
        # some helper functions
        rolls = lambda d,r0,r1: np.roll(np.roll(d,r0,axis=0),r1,axis=1)
        make_speckles = lambda x: np.fft.fftshift(abs(DFT(x))**2)
        blur_speckle = lambda x: abs(DFT(IDFT(x)*self.coherence))
        
        # go through the analysis. use the rot_sym function to do the unwrapping and decomposing.
        # if nothing is supplied for components or cosines, the defaults of rot_sym become active.
        self.illuminated = rolls(self.illumination,dy,dx)*self.object
        self.speckles = make_speckles(self.illuminated)
        
        if cosines == None: cosines = self.cosines
        if not self.can_has_coherence:
            self.spectrum = rot_sym(self.speckles,self.unwrap_plan,cosines=cosines)
        
        if self.can_has_coherence:
            self.blurred = blur_speckle(self.speckles)
            self.spectrum = rot_sym(self.blurred,self.unwrap_plan,cosines=cosines)

def make_cosines(components,N):
    """ Generates cosines to use in cosine decompsition of an autocorrelation.
    
    arguments:
        components: an iterable list of which cosine components to generate
        N: length of unwrapped autocorrelation
        
    returns:
        ndarray of shape (len(components),N) containing cosine values"""
        
    assert isinstance(components,(tuple,list,np.ndarray)), "components must be iterable"
    assert isinstance(N,int), "N must be int"

    x = (np.arange(N).astype(float))*(2*np.pi/N)
    cosines = np.zeros((len(components),N),float)
    for n,c in enumerate(components): cosines[n] = x*c
    return np.cos(cosines)

def decompose(ac,cosines):
    
    """ Do an explicit cosine decomposition by multiply-sum method
    
    arguments:
        ac: incoming angular autocorrelation
        cosines: array of evaluated cosine values
        
    returns:
        cosine spectrum, shape (len(ac),len(cosines))"""
        
    assert isinstance(ac,np.ndarray), "ac must be array"
    assert isinstance(cosines,np.ndarray), "cosines must be array"
    assert len(cosines[0]) == len(ac[0]), "cosines are wrong shape (c %s ac %s)"%(len(cosines[0]),len(ac[0]))
    
    N = float(len(ac[0]))
    decomposition = np.zeros((len(ac),len(cosines)),float)
    for y,row in enumerate(ac):
        for x, cosine in enumerate(cosines):
            decomposition[y,x] = np.sum(row*cosine)
    return decomposition/float(N)

def rot_sym(speckles,plan=None,components=None,cosines=None):
    """ Given a speckle pattern, decompose its angular autocorrelation into a cosine series.
    
    arguments:
        speckle: the speckle pattern to be analyzed. Should be human-centered
            (ie, center of speckle is at center of array) rather than machine-centered
            (ie, center of speckle is at corner of array).
            
        plan: either an unwrap plan from wrapping.unwrap_plan or a tuple of form (r,R) or (r,R,(center))
            describing the range of radii to be analyzed. If nothing is supplied, the unwrapping
            will by default be as extensive as possible: (r,R) = (0,N/2).
            
        components: (optional) an iterable set of integers describing which cosine components to analyze.
            If nothing is supplied, this will be all even numbers between 2 and 20.
            
        cosines: (optional) an ndarray containing precomputed cosines. for speed. if cosines is supplied,
            components is ignored.
            
        returns:
            an ndarray of shape (R-r,len(components)) giving the cosine component values of the
            decomposition."""
            
    # check types
    assert isinstance(speckles,np.ndarray) and speckles.ndim == 2, "input data must be 2d array"
    assert isinstance(plan,(np.ndarray,tuple,list,type(None))), "plan type is unrecognized"
    assert isinstance(components,(np.ndarray,tuple,list,type(None))), "components are non-iterable"
    
    N,M = speckles.shape
    R = min([N,M])
    
    # do the unwrapping. behavior depends on what comes in as plan
    if isinstance(plan,np.ndarray):
        unwrapped = wrapping.unwrap(speckles,plan)
    if isinstance(plan,tuple):
        if len(plan) == 2: unwrapped = wrapping.unwrap(speckles,(plan[0],plan[1],(N/2,M/2)))
        if len(plan) == 3: unwrapped = wrapping.unwrap(speckles,plan)
    if plan == None: unwrapped = wrapping.unwrap(speckles,(0,R/2,(N/2,M/2)))
        
    # autocorrelate the unwrapped speckle. normalize each row individually.
    autocorrelation = crosscorr.crosscorr(unwrapped,unwrapped,axes=(1,),shift=False)
    for row,row_data in enumerate(autocorrelation):
        autocorrelation[row] = row_data*(1./abs(row_data).max())
    
    # generate components and cosines if necessary
    if components == None: components = np.arange(2,20,2).astype('float')
    if cosines == None: cosines = make_cosines(components,len(autocorrelation[0]))
    
    # run cosine decomposition
    decomposition = decompose(autocorrelation,cosines)

    return decomposition
    
    
    
    