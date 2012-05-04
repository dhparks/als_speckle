import scipy
from scipy.fftpack import fft2 as DFT
from scipy.fftpack import ifft2 as IDFT
from scipy.fftpack import fftshift

import pyopencl as cl
import pyopencl.array as cl_array
from pyopencl.elementwise import ElementwiseKernel
import pyfft

from types import *

import shape
import io2

from phasing_parameters import *

def condition_data(intensity,dark=None):
    # data coming in here is assumed to be ill-conditioned because it is basically right off the ccd.
    # this function applies some routine conditioning to try to get the data into a suitable form to be
    # passed to the reconstructer.
    
    pass

def speckle(input):
    return fftshift(abs(DFT(self.estimate))**2)
    
def build_kernel_file(c, d, fileName):
    kernelFile = open(fileName, 'r')
    kernelStr = kernelFile.read()

    # Load the program source
    program = cl.Program(c, kernelStr)

    # Build the program and check for errors
    program.build(devices=[d])

    return program

class CPUPR:
    
    # Implement phase retrieval as a class. An instance of this class is a reconstruction, and methods in the class are operations on the reconstruction.
    # For example, calling instance.hio() will advance the reconstruction by one iteration of the HIO algorithm; calling instance.update_support() will
    # update the support being used in the reconstruction in the manner of Marchesini's shrinkwrap.

    def __init__(self,N):
        self.N = N
        
    def load_data(self,modulus,support,update_sigma=None):
        print update_sigma
        
        # get the supplied data into the reconstruction namespace
        self.modulus  = modulus
        self.support  = support  # this is the active support, it can be updated
        self.support0 = support # this is the original support, it is read-only
        
        # generate some necessary files
        self.estimate   = ((scipy.rand(self.N,self.N)+complex(0,1)*scipy.rand(self.N,self.N))*self.support)
        
        if update_sigma != None:
            assert type(update_sigma) in (IntType,FloatType), "update_sigma must be float or int"
            self.blurkernel = DFT(fftshift(shape.gaussian((self.N,self.N),(update_sigma,update_sigma),center=None,normalization=None)))
        
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
        update  = scipy.where(blurred > blurred.max()*threshold,1,0)
        if retain_bounds: update *= self.support0
        self.support = update
        
    def save(self,savepath,savename,n,save_estimate=True,save_diffraction=False,save_support=False):
        if save_estimate:    io2.save_fits(savepath+'/'+savename+' '+str(iteration)+'.fits',             self.estimate,          components=['mag','phase'], overwrite=True)
        if save_support:     io2.save_fits(savepath+'/'+savename+' '+str(iteration)+' support.fits',     self.support,           components=['mag'], overwrite=True)
        if save_diffraction: io2.save_fits(savepath+'/'+savename+' '+str(iteration)+' diffraction.fits', speckle(self.estimate), components=['mag'], overwrite=True)
            
class GPUPR:

    def __init__(self,N):
        self.N = N

        ### initialize all the crap needed for gpu computations: context, commandqueue, fftplan, program kernels
        if verbose:
            print "pyopencl version: %s"%str(cl.VERSION)
            print "pyfft version:    %s"%str(pyfft.VERSION)
        
        # 1. set up the context and queue; adapted from OpenCL book. requires a gpu to run; no cpu fallback
        if verbose: print "making context"
        platforms = cl.get_platforms();
        if len(platforms) == 0: print "Failed to find any OpenCL platforms."; exit()
        if verbose: print "  Platform: %s"%platforms[0]

        devices = platforms[0].get_devices(cl.device_type.GPU)
        if len(devices) == 0: print "Could not find any GPU device"; exit()
        self.device = devices[0]
        #print "  Device:   %s"%devices[0]

        self.ctx = cl.Context([devices[0]]) # Create a context using the first gpu in the list of devices
        self.queue = cl.CommandQueue(self.ctx) # Create a command queue for that device
        
        # 2. make fft plan for a 2d array with length N
        if verbose: print "making fft plan"
        from pyfft.cl import Plan
        self.fftplan = Plan((self.N, self.N), queue=self.queue)
        
        # 3. make the kernels to enforce the fourier and real-space constraints
        if verbose: print "compling kernels"
        self.fourier_constraint = ElementwiseKernel(self.ctx,
            "float2 *psi, "                        # current estimate of the solution
            "float  *modulus, "                    # known fourier modulus
            "float2 *out",                         # output destination
            "out[i] = rescale(psi[i],modulus[i])", # operator definition
            "replace_modulus",
            preamble = """
            #define rescale(a,b) (float2)(a.x/hypot(a.x,a.y)*b,a.y/hypot(a.x,a.y)*b)
            """)
        if verbose: print "  fourier-space constraint"
        
        self.realspace_constraint_hio = ElementwiseKernel(self.ctx,
            "float beta, "       # feedback parameter
            "float *support, "   # support constraint array
            "float2 *psi_in, "   # estimate of solution before modulus replacement
            "float2 *psi_out, "  # estimate of solution after modulus replacement
            "float2 *out",       # output destination
            "out[i] = (1-support[i])*(psi_in[i]-beta*psi_out[i])+support[i]*psi_out[i]",
            "hio")
        if verbose: print "  real-space constraint (hio algorithm)"
        
        self.realspace_constraint_er = ElementwiseKernel(self.ctx,

            "float *support, "   # support constraint array
            "float2 *psi_out, "  # estimate of solution after modulus replacement
            "float2 *out",       # output destination
            "out[i] = support[i]*psi_out[i]",
            "hio")
        if verbose: print "  real-space constraint (hio algorithm)"
        
        # if the support will be updated with shrinkwrap, initialize some additional gpu kernels
        if shrinkwrap:
            
            #self.support_threshold = build_kernel_file(self.ctx, self.device, "%s/support_threshold.cl"%kernel_path)
            
            self.set_zero = ElementwiseKernel(self.ctx,
                "float2 *buff",
                "buff[i] = (float2)(0.0f, 0.0f)",
                "set_zero")
            
            self.copy_real = ElementwiseKernel(self.ctx,
                "float2 *in,"
                "float  *out",
                "out[i] = in[i].x",
                "set_zero")
            
            self.make_abs = ElementwiseKernel(self.ctx,
                "float2 *in,"
                "float2 *out",
                "out[i] = (float2)(hypot(in[i].x,in[i].y),0.0f)",
                "make_abs")
            
            self.blur_convolve = ElementwiseKernel(self.ctx,
                "float2 *toblur,"
                "float  *blurrer,"
                "float2 *blurred",
                "blurred[i] = (float2) (toblur[i].x*blurrer[i],toblur[i].y*blurrer[i])",
                "blur_convolve")
            
            self.support_threshold = ElementwiseKernel(self.ctx,
                "float2 *in,"
                "float *out,"
                "float t",
                "out[i] = isgreaterequal(in[i].x,t)",
                "support_threshold")
            
            self.bound_support = ElementwiseKernel(self.ctx,
                "float *s,"
                "float *s0",
                "s[i] = s[i]*s0[i]",
                "bound_support")

    def load_data(self,modulus,support,update_sigma=None):
        # put the support and the modulus on the gpu
        # also establish buffers for the intermediates
        
        # make sure the modulus and support are the correct single-precision dtype
        if modulus.shape != support.shape: exit()
        
        # transfer the data to the cpu. pyopencl handles buffer creation
        random_complex = ((scipy.rand(self.N,self.N)+complex(0,1)*scipy.rand(self.N,self.N))*support).astype('complex64')
        self.modulus  = cl_array.to_device(self.queue,modulus.astype('float32'))
        self.support  = cl_array.to_device(self.queue,support.astype('float32'))
        self.support0 = cl_array.to_device(self.queue,support.astype('float32'))
        
        # initialize gpu arrays for fourier constraint satisfaction
        self.psi_in      = cl_array.to_device(self.queue,random_complex)
        self.psi_out     = cl_array.empty(self.queue,(N,N),scipy.complex64) # need both in and out for hio algorithm
        self.psi_fourier = cl_array.empty(self.queue,(N,N),scipy.complex64) # to store fourier transforms of psi
        
        if update_sigma != None:
            
            # make gpu arrays for blurring the magnitude of the current estimate in order to update the support
            
            assert type(update_sigma) in (IntType,FloatType), "update_sigma must be float or int"
            blurkernel = abs(DFT(fftshift(shape.gaussian((self.N,self.N),(update_sigma,update_sigma),center=None,normalization=None))))
            self.blur_kernel   = cl_array.to_device(self.queue,blurkernel.astype('float32'))
            self.blur_temp     = cl_array.empty(self.queue,(N,N),scipy.complex64) # for holding the blurred estimate
            self.blur_temp_max = cl_array.empty(self.queue,(N,N),scipy.float32)
 
    def update_support(self,threshold = 0.2,retain_bounds=True):
            
        # zero the temp buffer for safety. this is basically a free operation
        self.set_zero(self.blur_temp)

        # make a blurry version of the abs of psi_out
        self.make_abs(self.psi_out,self.blur_temp).wait()
        self.fftplan.execute(self.blur_temp.data,wait_for_finish=True)
        self.blur_convolve(self.blur_temp,self.blur_kernel,self.blur_temp)
        self.fftplan.execute(self.blur_temp.data,inverse=True,wait_for_finish=True)
        
        # do the thresholding procedure and update the support.
        # thresholding and copying to the self.support buffer are in the same kernel
        self.copy_real(self.blur_temp,self.blur_temp_max).wait()
        m = (threshold*cl_array.max(self.blur_temp_max).get()).astype('float32')
        self.support_threshold(self.blur_temp,self.support,m).wait()
        
        io2.save_fits('%s/%s updated support %s.fits'%(savepath,savename,iteration), self.support.get(), components=['mag'], overwrite=True)
        
        # enforce the condition that the support shouldnt expand in size
        if retain_bounds: self.bound_support(self.support,self.support0)
        
        io2.save_fits('%s/%s updated support bounded %s.fits'%(savepath,savename,iteration), self.support.get(), components=['mag'], overwrite=True)
        
        print "updated"
         
    def iteration(self,algorithm,beta=0.8):

        # do a single iteration. algorithm is selectable through keyword

        assert algorithm in ['hio','er'], "real space enforcement algorithm %s is unknown"%algorithm

        # 1. fourier transform the data in psi_in, store the result in psi_fourier
        self.fftplan.execute(self.psi_in.data,data_out=self.psi_fourier.data)
        
        # 2. enforce the fourier constraint by replacing the current-estimated fourier modulus with the measured modulus from the ccd
        self.fourier_constraint(self.psi_fourier,self.modulus,self.psi_fourier)
        
        # 3. inverse fourier transform the new fourier estimate
        self.fftplan.execute(self.psi_fourier.data,data_out=self.psi_out.data,inverse=True)
        
        # 4. enforce the real space constraint. algorithm can be changed based on incoming keyword
        if algorithm == 'hio': self.realspace_constraint_hio(beta,self.support,self.psi_in,self.psi_out,self.psi_in)
        if algorithm == 'er':  self.realspace_constraint_er(self.support,self.psi_out,self.psi_in)
        
    def save(self,savepath,savename,n,save_estimate=True,save_diffraction=False,save_support=False):
        
        if save_estimate:    io2.save_fits(savepath+'/'+savename+' '+str(iteration)+'.fits',             self.psi_in.get(),          components=['mag','phase'], overwrite=True)
        if save_support:     io2.save_fits(savepath+'/'+savename+' '+str(iteration)+' support.fits',     self.support.get(),         components=['mag'], overwrite=True)
        if save_diffraction: io2.save_fits(savepath+'/'+savename+' '+str(iteration)+' diffraction.fits', speckle(self.psi_in.get()), components=['mag'], overwrite=True)
        
def _do_iterations(reconstruction):
    global iteration
    for iteration in range(iterations):

        if iteration%update_period == 0 and iteration > 0 and shrinkwrap:
            reconstruction.update_support()
            reconstruction.iteration('er')
        else:
            if iteration%100 != 0:
                reconstruction.iteration('hio')
            else:
                reconstruction.iteration('er')
        if iteration%100 == 0: print "  iteration ",iteration
        
if __name__== "__main__":
    # allow execution of the following code only when the phasing.py script is invoked directly
    
    data = io2.openfits(dataname)
    support = io2.openfits(supportname)
    
    if data_is_intensity:
        if darkname != None:
            dark = io2.openfits(darkname)
        else:
            dark = None
        data = condition_data(data,dark)
        
    # check sizes
    assert data.ndim == 2, "data has wrong dimensionality"
    assert support.ndim == 2, "support has wrong dimensionality"
    assert data.shape == support.shape, "data and support have different sizes"
    
    # initialize the reconstruction
    N = len(data)
    if device == 'CPU': reconstruction = CPUPR(N)
    if device == 'GPU': reconstruction = GPUPR(N) # need to supply N in order to build the plan for fft
    
    for trial in range(trials):
        
        print "trial ",trial
        
        # load data/make new seed
        sigma = None
        if shrinkwrap: sigma = shrinkwrap_sigma
        reconstruction.load_data(data,support,update_sigma = sigma)
        
        # iterate
        _do_iterations(reconstruction)

        # save result
        reconstruction.save(savepath,savename,trial)
        
        
        
        
    
    
    
