""" CPU-based rotational symmetry code

Author: Daniel Parks (dhparks@lbl.gov)
"""
import numpy as np

from . import shape, wrapping, crosscorr

DFT = np.fft.fft2
shift = np.fft.fftshift

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

def fft_decompose(ac):
    """ Given a normalized rotational autocorrelation, decompose into
    a cosine series by fourier-transform method. The samples the
    spectrum up to the Nyquist limit. To limit the spectrum, it is
    necessary to slice the output somewhere on the column axis.
    Symmetries in the angular autocorrelation mean only the even components
    of the spectrum contain meaningful power, and only those components
    are returned.
    
    arguments:
        ac: incoming angular correlation.
        
    returns:
        cosine spectrum of shape (ac.shape[0],ac.shape[1]/4) """
        
    assert isinstance(ac,np.ndarray)
    
    if ac.ndim == 2:
        fft = DFT(ac,axes=(1,))
        return fft[:,2:ac.shape[1]/2+2:2]/ac.shape[1]
    if ac.ndim == 1:
        fft = np.fft.fft(ac)
        return fft[2:ac.shape[0]/2+2:2]/ac.shape[0]
        
def despike(ac,width=4):

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
    import scipy.misc as smp
    import Image
    return smp.fromimage(smp.toimage(data,mode='F').resize(shape,Image.ANTIALIAS))

def rot_sym(speckles,uwplan=None,resize_to=360,get_back=('spectra_ds',)):
    """ Given a speckle pattern, decompose its angular autocorrelation into a
        cosine series.

    arguments:
        speckle: the speckle pattern to be analyzed.

        uwplan: (optional) either an unwrap plan from wrapping.unwrap_plan or a
            tuple of form (r,R) or (r,R,(center)) describing the range of radii
            to be analyzed. If center is not supplied, it uses (row/2, col/2).
            If nothing is supplied, the unwrapping will by centered at
            (row/2, col/2) and range from (0, min(row/2,col/2)).
            
        resize_to: resize the unwrapped data (and consequently the correlation)
            to this width. number of rows remains unchanged. this should be a
            multiple of 4.

        get_back (optional): a tuple of keywords allowing a dictionary of intermediates to
            be returned. mainly for use with cpu_microscope in order to unify
            output syntax with gpu_microscope
            allowed kwords:
                'spectra' - all frames of the input decomposed up to the nyquist limit
                'unwrapped' - all frames unwrapped.
                'correlation'
            if specified, you get a dictionary back that you call like: output['spectra']

    returns:
        an ndarray of shape (R-r,len(components)) giving the cosine component
            values of the decomposition.
            
        others come back if get_back is specified but this is not the default behavior
    """
    # check types
    assert isinstance(speckles,np.ndarray) and speckles.ndim in (2,3), "input data must be 2d array"
    assert isinstance(uwplan,(np.ndarray,tuple,list,type(None))),      "plan type is unrecognized"
    assert len(get_back) > 0
    assert resize_to%4 == 0
    
    was_2d = False
    if speckles.ndim == 2:
        speckles.shape = (1,speckles.shape[0],speckles.shape[1])
        was_2d = True
    
    L,N,M = speckles.shape #N probably = M but dont assume it
    R     = min([N,M])

    # if plan comes in as a tuple, make the unwrap plan
    if isinstance(uwplan,tuple):
        if len(uwplan) == 2: uwplan = wrapping.unwrap_plan(plan[0],plan[1],(N/2,M/2))
        if len(uwplan) == 3: uwplan = wrapping.unwrap_plan(plan[0],plan[1],plan[2])
    if uwplan == None:       uwplan = wrapping.unwrap_plan(0,R/2,(N/2,M/2))
    R,r = uwplan[:,-1]
    cols = int((len(uwplan[0])-1)/abs(R-r))
    rows = int(R-r)

    # if resizing is requested, as it is by default, create a resizing plan
    if resize_to != None:
        rsplan = wrapping.resize_plan((rows,cols),(rows,resize_to))
        cols = resize_to
        
    # initialize the output
    if 'spectra'       in get_back: spectra      = np.zeros((L,rows,resize_to/4),np.float32)
    if 'spectra_ds'    in get_back: spectra_ds   = np.zeros((L,rows,resize_to/4),np.float32)
    if 'unwrapped'     in get_back: unwrappeds   = np.zeros((L,rows,resize_to),  np.float32)
    if 'correlated'    in get_back: correlations = np.zeros((L,rows,resize_to),  np.float32)
    if 'correlated_ds' in get_back: correlations = np.zeros((L,rows,resize_to),  np.float32)

    # each frame is analyzed in the same sequence:
    # 1. unwrap
    # 2. correlate/normalize
    # 3. despike
    # 3. decompose
    for f,frame in enumerate(speckles):
        
        # unwrap
        unwrapped = wrapping.unwrap(frame,uwplan)
        
        # maybe resize
        if resize_to != None: unwrapped = wrapping.resize(unwrapped,rsplan)
        
        # correlate and normalize
        i0        = np.outer(np.average(unwrapped,axis=1)**2,np.ones(cols)) # this is the denominator (<I>)^2
        autocorr  = crosscorr.crosscorr(unwrapped,unwrapped,axes=(1,),shift=False).real/cols
        autocorr  = autocorr/i0-1

        # maybe despike and decompose
        if 'correlated_ds' in get_back or 'spectra_ds' in get_back:
            despiked = despike(autocorr.real,width=4)
            spectrum_ds = np.abs(fft_decompose(despiked))
        
        # decompose into cosine series with fft
        spectrum = np.abs(fft_decompose(autocorr.real)) # I take the abs because there might be leakage into the imag component
        
        # add specified things to output
        if 'spectra'        in get_back: spectra[f]         = spectrum
        if 'spectra_ds'     in get_back: spectra_ds[f]      = spectrum_ds
        if 'unwrapped'      in get_back: unwrappeds[f]      = unwrapped
        if 'correlated'     in get_back: correlations[f]    = autocorr
        if 'correldated_ds' in get_back: correlations_ds[f] = despiked
        
    if was_2d: speckles.shape = (speckles.shape[1],speckles.shape[2])

    to_return = {}
    if 'spectra'       in get_back: to_return['spectra']       = spectra
    if 'spectra_ds'    in get_back: to_return['spectra_ds']    = spectra_ds
    if 'unwrapped'     in get_back: to_return['unwrapped']     = unwrappeds
    if 'correlated'    in get_back: to_return['correlated']    = correlations
    if 'correlated_ds' in get_back: to_return['correlated_ds'] = correlations_ds
    return to_return

def concentrations(data_in):
    # divide each component by the sum of values at the same |q|

    assert isinstance(data_in,np.ndarray)
    assert data_in.ndim == 3
    import time

    frames, rows, cols = data_in.shape
    
    # calculate the concentrations. numexpr does not speed this calculation (?!)
    p  = np.sum(data_in,axis=-1)       # sum along the component-value axis
    q = (data_in.transpose()/p.transpose()).transpose()
    c = (q**2).sum(axis=-1).transpose()

    return c, p.transpose()