# core
import numpy

from . import shape

DFT = numpy.fft.fft2
IDFT = numpy.fft.ifft2
shift = numpy.fft.fftshift

class CPUPR:
    
    # Implement phase retrieval as a class. An instance of this class is a reconstruction, and methods in the class are operations on the reconstruction.
    # For example, calling instance.hio() will advance the reconstruction by one iteration of the HIO algorithm; calling instance.update_support() will
    # update the support being used in the reconstruction in the manner of Marchesini's shrinkwrap.

    def __init__(self,N):
        self.N = N
        
    def load_data(self,modulus,support,update_sigma=None):

        
        # get the supplied data into the reconstruction namespace
        self.modulus  = modulus
        self.support  = support  # this is the active support, it can be updated
        self.support0 = support # this is the original support, it is read-only
        
        # generate some necessary files
        self.estimate   = ((numpy.random.rand(self.N,self.N)+complex(0,1)*numpy.random.rand(self.N,self.N))*self.support)
        
        if update_sigma != None:
            assert isinstance(update_sigma, (int,float)), "update_sigma must be float or int"
            self.blurkernel = DFT(shift(shape.gaussian((self.N,self.N),(update_sigma,update_sigma),center=None,normalization=None)))
        
    def iteration(self,algorithm,beta=0.8):
        
        assert algorithm in ['hio','er'], "real space enforcement algorithm %s is unknown"%algorithm
        
        psi = DFT(self.estimate)
        psi = self.modulus*psi/abs(psi)
        inverse = IDFT(psi)
        
        # enforce the real-space constraint
        if algorithm == 'hio': self.estimate = (1-self.support)*(self.estimate-beta*inverse)+self.support*inverse # hio support algorithm
        if algorithm == 'er': self.estimate = self.support*self.estimate
        
    def update_support(self,threshold = 0.25,retain_bounds=True):
        
        # auto-update the support by blurring the magnitude component of the estimate and retaining the selection of the blurred
        # signal greater than some threshold fraction of the signal maximum.
        
        # the retain_bounds flag is intended to keep the updated support from growing outside of the support boundaries supplied with
        # the first support estimate, the assumption being that that support was much too loose and updates should only get tighter.
        
        blur = lambda a,b: IDFT(DFT(a)*b)
        
        mag     = abs(self.estimate)  
        blurred = blur(mag,self.blurkernel)
        update  = numpy.where(blurred > blurred.max()*threshold,1,0)
        if retain_bounds: update *= self.support0
        self.support = update

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
        print abs(x.imag).sum(),opt,abs((x*numpy.exp(complex(0,1)*opt)).imag).sum()
        frame *= numpy.exp(complex(0,1)*opt)
        
        # minimizing the imaginary component can give a degenerate solution (ie, 0, pi)
        # now check the value of the real.sum() against -real.sum()
        s1 = frame.real.sum()
        s2 = (-1*frame).real.sum()
        if s2 > s1: frame *= -1
    
    if was2d: data = data[0]
    
    return data

def prtf(estimates,N):
    """Implements the PRTF which measures the reproducibility of the
    reconstructed phases. On the basis of this function claims
    of the resolution are often made.
    
    Inputs:
        estimates - an iterable set of aligned independent reconstructions
        N - the size of the array each estimate should be embedded in (this
        exists because the phasing library typically does not store the range
        of the image outside the support, as it is all zeros)
        
    Returns
        prtf: a 2d array of the PRTF at each reciprocal space value
        prtf_q: a 1d array of the PRTF averaged in annuli using the wrapping lib
    """
    
    assert isinstance(estimates,(numpy.ndarray,list,tuple)), "must be iterable"
    if isinstance(estimates,numpy.ndarray):
        assert estimates.ndim == 3, "must be 3d"
    if isinstance(estimates,(list,tuple)):
        assert len(estimates) > 1, "must be 3d"
    
    # compute the prtf by averaging the phase of all the trials
    phase_average = numpy.zeros((N,N),complex)
    estimate      = numpy.zeros((N,N),complex)
    
    print "averaging phases"
    for n in range(len(estimates)):
        print "  %s"%n
        sample = estimates[n]
        estimate[0:len(sample),0:len(sample[0])] = sample
        fourier = numpy.fft.fft2(estimate)
        phase = fourier/abs(fourier)
        phase_average += phase
    prtf = shift(abs(phase_average/len(estimates)))
    
    # unwrap and do the angular average
    import wrapping
    unwrapped = wrapping.unwrap(prtf,(0,N/2,(N/2,N/2)))
    prtf_q    = numpy.average(unwrapped,axis=1)
    
    return prtf, prtf_q

def rftf(estimate,goal_modulus,hot_pixels=False):
    """ Calculates the RTRF in coherent imaging which in analogy to
    crystallography attempts to quantify the Fourier error to determine
    the correctness of a reconstructed image. In contrast to the PRTF,
    this function measures deviations from the fourier goal modulus, whereas
    the PRTF measures the reproducibility of the phase.
    
    From Marchesini et al "Phase Aberrations in Diffraction Microscopy"
    arXiv:physics/0510033v2"
    
    Inputs:
        estimate: the averaged reconstruction
        goal_modulus: the fourier modulus of the speckles being reconstructed
        
    Returns:
        rtrf: a 2d array of the RTRF at each reciprocal space value
        rtrf_q : a 1d array of rtrf averaged in annuli using unwrap
    """
    
    assert isinstance(estimate,numpy.ndarray), "estimate must be ndarray"
    assert isinstance(goal_modulus,numpy.ndarray), "goal_modulus must be ndarray"
    assert estimate.shape <= goal_modulus.shape, "estimate must be smaller than goal_modulus"
    
    # form the speckle pattern
    new = numpy.zeros(goal_modulus.shape,estimate.dtype)
    new[0:estimate.shape[0],0:estimate.shape[1]] = estimate
    fourier = abs(numpy.fft.fft2(new))
    N = goal_modulus.shape[0]
    
    # line up with goal_modulus. based on the operation of the phasing
    # library, the goal_modulus will be centered either at (0,0) or (N/2,N/2).
    # aligning using align_frames is a bad idea because the reconstruction might
    # give a bad speckle pattern, throwing off alignment.
    diff1 = numpy.sum(abs(fourier-goal_modulus))          # goal modulus at (0,0)
    diff2 = numpy.sum(abs(shift(fourier)-goal_modulus))   # goal modulus at N/2
    
    if diff1 > diff2:
        fourier = shift(fourier)
        error   = (fourier-goal_modulus)**2/goal_modulus**2
        
    if diff1 < diff2:
        error = shift((fourier-goal_modulus)**2/goal_modulus**2)
        
    if hot_pixels:
        import conditioning
        error = conditioning.remove_hot_pixels(error)
        
    # calculate the rtrf from the error
    import wrapping
    rtrf      = numpy.sqrt(1./(1+error))
    if hot_pixels: rtrf = conditioning.remove_hot_pixels(rtrf,threshold=1.1)
    unwrapped = wrapping.unwrap(rtrf,(0,N/2,(N/2,N/2)))
    rtrf_q    = numpy.average(unwrapped,axis=1)
    
    return rtrf, rtrf_q
    
def refine_support(support,average_mag,blur=3,local_threshold=.2,global_threshold=0,kill_weakest=False):
    """ Given an average reconstruction and its support, refine the support
    by blurring and thresholding. This is the Marchesini approach (PRB 2003)
    for shrinkwrap.
    
    Inputs:
        support - the current support
        average_mag - the magnitude component of the average reconstruction
        blur - the stdev of the blurring kernel, in pixels
        threshold - the amount of the blurred max which is considered the object
        kill_weakest - if True, eliminate the weakest object in the support. this is
            for the initial refine of a multipartite holographic support as in the barker
            code experiment; one of the reference guesses may have nothing in it, and it
            should be eliminated.
        
    Output:
        the refined support
        
    """
    
    assert isinstance(support,numpy.ndarray),        "support must be array"
    assert support.ndim == 2,                        "support must be 2d"
    assert isinstance(average_mag,numpy.ndarray),    "average_mag must be array"
    assert average_mag.ndim == 2,                    "average_mag must be 2d"
    assert isinstance(blur,(float,int)),             "blur must be a number (is %s)"%type(blur)
    assert isinstance(global_threshold,(float,int)), "global_threshold must be a number (is %s)"%type(global_threshold)
    assert isinstance(local_threshold,(float,int)),  "lobal_threshold must be a number (is %s)"%type(local_threshold)
    
    refined   = numpy.zeros_like(support)
    
    from . import shape,masking
    kernel  = numpy.fft.fftshift(shape.gaussian(support.shape,(blur,blur)))
    kernel *= 1./kernel.sum()
    kernel  = numpy.fft.fft2(kernel)
    blurred = numpy.fft.ifft2(kernel*numpy.fft.fft2(average_mag)).real
    
    # find all the places where the blurred image passes the global threshold test
    global_passed = numpy.where(blurred > blurred.max()*global_threshold,1,0)
    
    # now find all the local parts of the support, and conduct local thresholding on each
    parts = masking.find_all_objects(support)
    print "refining"
    print parts.shape
    part_sums = numpy.ndarray(len(parts),float)
    for n,part in enumerate(parts):
        current      = blurred*part
        local_passed = numpy.where(current > current.max()*local_threshold, 1, 0)
        refined     += local_passed
        
        if kill_weakest: part_sums[n] = numpy.sum(average_mag*part)
        
    refined *= global_passed
    
    if kill_weakest:
        weak_part = parts[part_sums.argmin()]
        refined  *= 1-weak_part
    
    return refined,blurred
        
        
        
    
    
    
    