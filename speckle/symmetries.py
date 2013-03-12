""" CPU-based rotational symmetry code

Author: Daniel Parks (dhparks@lbl.gov)
"""
import numpy as np

from . import shape, wrapping, crosscorr

DFT = np.fft.fft2
IDFT = np.fft.ifft2
shift = np.fft.fftshift

class cpu_microscope():
    """ A class containing methods helpful in running a symmetry microscope
        analysis on an image. For now, this class is limited to simulations
        where the object being analyzed is square. Furthermore, speckle patterns
        are simulated at the same size as the object.
    """
    
    def __init__(self,device=None,object=None,unwrap=None,pinhole=None,ph_signal=None,coherence=None,components=None,specklesize=None,fb=10,corrwidth=360,returnables=('spectrum')):
        
        # the class is structured in such a way that its parts can all be inited right now
        # or later on in the main script. the example will show both.
        
        # device is a dummy input to equate the invocation of cpu_microscope and gpu_microscope
        if device != None:
            print "unncessary device input %s"%device
            print "ignoring that input"
        
        self.can_has_object = False
        self.can_has_unwrap = False
        self.can_has_pinhole = False
        self.can_has_coherence = False
        self.can_has_specklesize = False
        self.cosines = None
        self.returnables_list = returnables
        self.fb = fb # fakeblocker size
        
        # if simulation information is supplied to __init__, set up those parts of the simulation.
        # otherwise, they will have to be done later from the main script.
        
        if object != None: self.set_object(object)
        if pinhole != None: self.set_pinhole(pinhole,specklesize)
        if unwrap != None: self.set_unwrap(unwrap)
        if coherence != None: self.set_coherence(coherence)
        if components != None: self.set_cosines(corrwidth=corrwidth,components=components)
        
        self.returnables = {}
        
    def set_object(self,object):
        """ Put an object into the class namespace. The simulation object can
        be reset during the simulation, provided that it is the same shape as
        the object it replaces.
        
        arguments:
            object: 2d numpy array or, in the initial set, an integer. If an
                array, must be square."""
        
        assert isinstance(object,(np.ndarray,int))
        
        if self.can_has_object:
            # if an object has already been set, require the new object be an array
            assert isinstance(object,np.ndarray), "must update object with numpy array"
            assert object.ndim == 2, "object must be 2d"
            assert object.shape[0] == object.shape[1], "object must be square"
            self.object = object.astype(complex)
        
        if not self.can_has_object:
            # if an object hasn't already been set, allow either a number or an object;
            # strictly speaking the former is all that is needed to set the rest of
            # the class variables.
            if isinstance(object,int):
                self.obj_N = object
                self.object = np.zeros((self.obj_N,self.obj_N),complex)
            if isinstance(object,np.ndarray):
                assert object.ndim == 2, "object must be 2d"
                assert object.shape[0] == object.shape[1], "object must be square"
                self.object = object.astype(complex)
        
        self.can_has_object = True
        
    def set_unwrap(self,params):
        """ Build and name the unwrap plan. Can't be reset after initial setting.
        
        Input:
            params: unwrap_r and unwrap_R.
                center is assumed to be at array center
        
        returns: nothing, but generates class variables"""
        
        assert self.can_has_object, "need to init object before unwrap"
        assert isinstance(params,(list,tuple,np.ndarray)), "unwrap params must be iterable"
        assert len(params) == 2, "must provide exactly two parameters: unwrap_r and unwrap_R"
        
        if self.can_has_unwrap:
            print "setting unwrap during simulation not allowed"
            exit()
        
        ur, uR = params[0],params[1]
        self.ur, self.uR = min([ur,uR]), max([ur,uR])
        self.rows = (self.uR-self.ur)
        assert ur > 0, "unwrap_r must be > 0"
        assert uR < self.speckle_N/2, "unwrap_R exceeds simulated speckle bounds"
        
        self.unwrap_plan = wrapping.unwrap_plan(self.ur,self.uR,(self.speckle_N/2,self.speckle_N/2))
        
        self.can_has_unwrap = True
        
    def set_pinhole(self,pinhole,specklesize):
        
        """Set the pinhole function. Can be reset during the simulation if it is
        the same size as the old pinhole (the array size, not the radius).
        
        arguments:
            pinhole: Either a number, in which case a circle is generated as the
            pinhole, or an array, in which case the array is set as the pinhole.
            The latter option allows custom illuminations to be supplied."""

        assert self.can_has_object, "need to init object before pinhole"
        assert isinstance(pinhole,np.ndarray), "pinhole must be array"
        assert pinhole.ndim == 2, "pinhole must be 2d array"
        assert pinhole.shape[0] == pinhole.shape[1], "pinhole must be square"
        assert len(pinhole) <= self.obj_N, "pinhole size must be leq than object size"
        
        if not self.can_has_pinhole:
             self.pin_N = len(pinhole)
             self.illumination = pinhole

        # if the speckle will be resized, make the plan and put it on the gpu
        if not self.can_has_specklesize:
            if specklesize != None: self.speckle_N = specklesize
            if specklesize == None: self.speckle_N = self.pin_N
            self.can_has_specklesize = True
            
        if self.pin_N < self.obj_N:
            self.object = shift(self.object) # this makes the slicing easier
        
        # make the blocker: ball + stick
        #self.blocker = fs(io.open('../../jobs/symmetry microscope/images/blocker_mask.png').astype(np.float32))
        #self.blocker *= 1./self.blocker.max()
        
        #self.blocker = shape.circle((self.speckle_N,self.speckle_N),55)
        self.blocker = shape.rect((self.speckle_N,self.speckle_N),(self.fb,512),center=(512-self.fb/2,512+256))
        self.blocker = shift(np.clip(1-self.blocker,0,1))

        assert self.blocker.shape == (self.speckle_N,self.speckle_N), "blocker is wrong shape"
        
        self.can_has_pinhole = True
        
    def set_coherence(self,coherence):
        """ Make the coherence function"""
        
        assert self.can_has_object, "need to init object before coherence"
        assert isinstance(coherence,(tuple,list,np.ndarray)), "coherence needs to be tuple, list, or array"
        
        if isinstance(coherence,(list,tuple)):
            assert len(coherence) == 2, "coherence must have len 2"
            clx,cly = coherence[0],coherence[1]
            self.coherence = shift(shape.gaussian((self.speckle_N,self.speckle_N),(cly,clx)))
        
        if isinstance(coherence,np.ndarray):
            assert coherence.shape == (self.speckle_N,self.speckle_N), "coherence must be NxN"
            self.coherence = coherence
            
        self.can_has_coherence = True
     
    def set_cosines(self,corrwidth=None,components=None,cosines=None):
        assert self.can_has_unwrap, "must set unwrap before cosines"
        
        if components != None:
            assert isinstance(components,(tuple,list,np.ndarray)), "components must be iterable"
            
            if corrwidth == None: corrwidth = (self.unwrap_plan.shape[1]-1)/self.rows
            self.cosines = make_cosines(components,corrwidth)
            
        if cosines != None:
            # trust the user here...
            self.cosines = cosines
        
    def set_returnables(self,returnables=('spectrum')):
        
        """ Set which of the possible intermediate values are returned out of
        the simulation. Results are returned as a dictionary from which
        intermediates can be extracted through returnables['key'] where 'key' is
        the desired intermediate. Set after object to be safe. Fewer options are
        available here than in the gpu class.
        
        Available returnables:
            illuminated: the current view * the illumination function
            speckle: human-centered speckle of the current illumination
            speckle_blocker: speckle with a blocker of radius ur in the center
            blurred: blurred speckle
            blurred_blocker: blurred speckle with ur-blocker
            unwrapped: unwrapped speckle (or blurred).
            correlated: the angular autocorrelation
            spectrum: the angular ac decomposed into a cosine series.
            
        By default, the only returnable is 'spectrum', the final output.

        """
        
        available = ('illuminated','speckle','speckle_blocker',
                     'blurred','blurred_blocker','unwrapped','correlated',
                     'spectrum')
        
        # check types
        print returnables
        assert isinstance(returnables,(list,tuple)), "returnables must be list or tuple"
        assert all([isinstance(r,str) for r in returnables]), "all elements of returnables must be strings"
        
        self.returnables_list = []
        for r in returnables:
            if r not in available:
                print "requested returnable %s is unrecognized and will be ignored"%r
            else:
                self.returnables_list.append(r)
        
        if 'speckle_blocker' in returnables or 'blurred_blocker' in returnables:
            assert self.can_has_object, "set object before setting xxx_blocker returnable"
            self.blocker = 1-shape.circle((self.N,self.N),self.ur)
             
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
        
        assert self.can_has_object, "no object set"
        assert self.can_has_pinhole, "no pinhole set"
        assert self.can_has_unwrap, "no unwrap set"
        assert isinstance(dy,int) and isinstance(dx,int), "site coordinates must be integer but are %s %s"%(type(dy),type(dx))
        
        # some helper functions
        rolls = lambda d,r0,r1: np.roll(np.roll(d,r0,axis=0),r1,axis=1)
        blur_speckle = lambda x: abs(DFT(IDFT(x)*self.coherence))
        
        def make_speckles(x):
            # first, make the speckles by fft.
            # the, resize if necessary for the unwrap.
            speckles = shift(abs(DFT(x))**2)
            if self.speckle_N != self.pin_N:
                import scipy.misc.pilutil as smp
                import Image
                print "resizing"
                speckles = smp.fromimage(smp.toimage(speckles,mode='F').resize((self.speckle_N,self.speckle_N),Image.BILINEAR))
            return speckles
        
        def illuminate(dy,dx):
            if self.pin_N < self.obj_N:
                return rolls(self.object,dy,dx)[self.obj_N/2-self.pin_N/2:self.obj_N/2+self.pin_N/2,self.obj_N/2-self.pin_N/2:self.obj_N/2+self.pin_N/2]*self.illumination
            else:
                return rolls(self.illumination,dy,dx)*self.object

        # go through the analysis. use the rot_sym function to do the unwrapping and decomposing.
        # if nothing is supplied for components or cosines, the defaults of rot_sym become active.
        self.illuminated = illuminate(dy,dx)
        self.speckles = make_speckles(self.illuminated)
        
        if cosines == None: cosines = self.cosines
        if not self.can_has_coherence:
            to_rotsym = self.speckles
        if self.can_has_coherence:
            self.blurred = blur_speckle(self.speckles)
            to_rotsym = self.blurred
    
        returned_rs = rot_sym(to_rotsym,plan=self.unwrap_plan,cosines=self.cosines,get_back=self.returnables_list) # dont need to pass components, only cosines

        # build the output dictionary
        if 'illuminated' in self.returnables_list:
            temp = shift(self.illuminated)
            if isinstance(self.pr, (int,float)): temp = temp[self.N/2-self.pr:self.N/2+self.pr,self.N/2-self.pr:self.N/2+self.pr]
            self.returnables['illuminated'] = temp
        if 'speckle' in self.returnables_list: self.returnables['speckle'] = self.speckles
        if 'speckle_blocker' in self.returnables_list: self.returnables['speckle_blocker'] = self.speckles*self.blocker
        if 'blurred' in self.returnables_list and self.can_has_coherence: self.returnables['blurred'] = self.blurred
        if 'blurred_blocker' in self.returnables_list and self.can_has_coherence: self.returnables['blurred_blocker'] = self.blurred*self.blocker
        if 'unwrapped' in self.returnables_list: self.returnables['unwrapped'] = returned_rs['unwrapped']
        if 'correlated' in self.returnables_list: self.returnables['correlated'] = returned_rs['correlated']
        if 'spectrum' in self.returnables_list:self.returnables['spectrum'] = returned_rs['spectra'][0]
        if 'spectrum_ds' in self.returnables_list:self.returnables['spectrum_ds'] = returned_rs['spectra_ds'][0]

def make_cosines(components,N):
    """ Generates cosines to use in cosine decompsition of an autocorrelation.
    
    arguments:
        components: an iterable list of which cosine components to generate
        N: length of unwrapped autocorrelation
        
    returns:
        ndarray of shape (len(components),N) containing cosine values
    """
    assert isinstance(components,(tuple,list,np.ndarray)), "components must be iterable"
    assert isinstance(N,int), "N must be int, is %s"%N

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
        cosine spectrum, shape (len(ac),len(cosines))
    """
    assert isinstance(ac,np.ndarray), "ac must be array"
    assert isinstance(cosines,np.ndarray), "cosines must be array"
    assert len(cosines[0]) == len(ac[0]), "cosines are wrong shape (c %s ac %s)"%(len(cosines[0]),len(ac[0]))
    
    N = float(len(ac[0]))
    
    decomposition = np.zeros((len(ac),len(cosines)),float)
    for y,row in enumerate(ac):
        for x, cosine in enumerate(cosines):
            d = np.sum(row*cosine)
            decomposition[y,x] = np.sum(row*cosine)
    return decomposition/float(N)

def despike(ac,width=4):
    
    print ac.shape
    
    def _do(x,w):
        # cubic -> quadratic interpolation
        v0 = x[:,180-2*w]
        v1 = x[:,180-1*w]
        v2 = x[:,180+1*w]
        v3 = x[:,180+2*w]
        d1 = (v3-v2)/w
        d0 = (v1-v0)/w
        b  = (d1-d0)/(4*w)
        d  = (2*(v2+v1)-w*(d1-d0))/4
        col_start = 180-w
        for n in range(2*w): x[:,col_start+n] = b*(n-w)*(n-w)+d
        return x
        
    ac = _do(ac,width)
    ac = np.roll(ac,180,axis=1)
    ac = _do(ac,width)
    return np.roll(ac,90,axis=1)

def resize(data,shape):
    import scipy.misc.pilutil as smp
    import Image
    return smp.fromimage(smp.toimage(data,mode='F').resize(shape,Image.ANTIALIAS))

def rot_sym(speckles,plan=None,components=None,cosines=None,resize_to=360,get_back=()):
    """ Given a speckle pattern, decompose its angular autocorrelation into a
        cosine series.

    arguments:
        speckle: the speckle pattern to be analyzed.

        plan: (optional) either an unwrap plan from wrapping.unwrap_plan or a
            tuple of form (r,R) or (r,R,(center)) describing the range of radii
            to be analyzed. If center is not supplied, it uses (row/2, col/2).
            If nothing is supplied, the unwrapping will by centered at
            (row/2, col/2) and range from (0, min(row/2,col/2)).
            
        components: (optional) an iterable set of integers describing which
            cosine components to analyze. If nothing is supplied, this will be
            all even numbers between 2 and 20.

        cosines: (optional) an ndarray containing precomputed cosines. for
            speed. if cosines is supplied, components is ignored.

        get_back (optional): a tuple of keywords allowing a dictionary of intermediates to
            be returned. mainly for use with cpu_microscope in order to unify
            output syntax with gpu_microscope
            allowed kwords: 'spectra', 'unwrapped', 'correlation'.
            if specified, you get a dictionary back that you call like: output['spectra']

    returns:
        an ndarray of shape (R-r,len(components)) giving the cosine component
            values of the decomposition.
            
        others come back if get_back is specified but this is not the default behavior
    """
    # check types
    assert isinstance(speckles,np.ndarray) and speckles.ndim in (2,3), "input data must be 2d array"
    assert isinstance(plan,(np.ndarray,tuple,list,type(None))),        "plan type is unrecognized"
    assert isinstance(components,(np.ndarray,tuple,list,type(None))),  "components are non-iterable"
    
    was_2d = False
    if speckles.ndim == 2:
        speckles.shape = (1,speckles.shape[0],speckles.shape[1])
        was_2d = True
    
    L,N,M = speckles.shape #N probably = M but dont assume it
    R     = min([N,M])
    
    spectra      = []
    spectra_ds   = []
    unwrappeds   = []
    correlations = []
    correlations_ds = []
    
    # if plan comes in as a tuple, make the unwrap plan
    if isinstance(plan,tuple):
        if len(plan) == 2: plan = wrapping.unwrap_plan(plan[0],plan[1],(N/2,M/2))
        if len(plan) == 3: plan = wrapping.unwrap_plan(plan[0],plan[1],plan[2])
    if plan == None:       plan = wrapping.unwrap_plan(0,R/2,(N/2,M/2))
    
    # make the cosine components
    R,r = plan[:,-1]
    uw_cols = (len(plan[0])-1)/abs(R-r)
    if components == None: components = np.arange(2,20,2).astype('float')
    if cosines    == None: cosines    = make_cosines(components,int(uw_cols))
    
    # each frame is analyzed in the same sequence:
    # 1. unwrap
    # 2. correlate/normalize
    # 3. decompose
    for f,frame in enumerate(speckles):
        unwrapped = wrapping.unwrap(frame,plan)
        i0        = np.outer(np.average(unwrapped,axis=1)**2,np.ones(uw_cols)) # this is the denominator (<I>)^2
        autocorr  = crosscorr.crosscorr(unwrapped,unwrapped,axes=(1,),shift=False).real/uw_cols
        autocorr  = (autocorr-i0)/i0
        if resize_to != None:
            autocorr  = resize(autocorr,(resize_to,autocorr.shape[0]))
        despiked  = despike(autocorr.real,width=4)
        spectrum  = decompose(autocorr.real,cosines)
        spectrum_ds = decompose(despiked,cosines) # remove the spikes at 0 and pi

        unwrappeds.append(unwrapped)
        correlations.append(autocorr.real)
        correlations_ds.append(despiked.real)
        spectra.append(spectrum)
        spectra_ds.append(spectrum_ds)
    
    if was_2d: speckles.shape = (speckles.shape[1],speckles.shape[2])
    
    if get_back == ():
        return spectra
    if get_back != ():
        to_return = {}
        to_return['spectra'] = spectra
        to_return['spectra_ds'] = spectra_ds
        if 'unwrapped'     in get_back: to_return['unwrapped'] = unwrappeds
        if 'correlated'  in get_back: to_return['correlated'] = correlations
        if 'correlated_ds' in get_back: to_return['correlated_ds'] = correlations_ds
        return to_return
