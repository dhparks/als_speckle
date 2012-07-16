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
        print opt
        frame *= numpy.exp(complex(0,1)*opt)
    
    if was2d: data = data[0]
    
    return data     
        
        
        
    
    
    
