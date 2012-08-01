import numpy
import pyopencl.array as cla
from pyopencl.elementwise import ElementwiseKernel as cl_kernel

DFT = numpy.fft.fft2
IDFT = numpy.fft.ifft2
shift = numpy.fft.fftshift

class GPUPR:
    """ Implement phase retrieval on the GPU to take advantage of the parallel
    nature of the phase-retrieval algorithms. 
    
    Instantiating the class requires information about the gpu (device, context,
    queue) obtained from gpu.init(), and the simulation size (ie, 256x256,
    512x512) to initialize the FFT code. Right now only square simulation sizes
    are supported.
    
    Methods:
        load_data: puts constraint data onto the gpu. Requires 2 arrays, the
            fourier modulus and the support.
        seed: sets up an independent reconstruction by putting a new
            complex-random seed into GPU memory.
        iteration: runs a single iteration of either the HIO or the ER algorithm
        update_support: if using the shrinkwrap algorithm, this updates the
            support by blurring and thresholding.
    """
    
    def __init__(self,gpuinfo,N,shrinkwrap=False):
        self.N = N
        self.context,self.device,self.queue,self.platform = gpuinfo

        # 1. make fft plan for a 2d array with length N. this assumes complex data array
        # dtype (in c parlance: interleaved complex)
        from pyfft.cl import Plan
        self.fftplan = Plan((self.N, self.N), queue=self.queue)
        
        # 2. make the kernels to enforce the fourier and real-space constraints
        self.fourier_constraint = cl_kernel(self.context,
            "float2 *psi, "                        # current estimate of the solution
            "float  *modulus, "                    # known fourier modulus
            "float2 *out",                         # output destination
            "out[i] = rescale(psi[i],modulus[i])", # operator definition
            "replace_modulus",
            preamble = """
            #define rescale(a,b) (float2)(a.x/hypot(a.x,a.y)*b,a.y/hypot(a.x,a.y)*b)
            """)
        
        self.realspace_constraint_hio = cl_kernel(self.context,
            "float beta, "       # feedback parameter
            "float *support, "   # support constraint array
            "float2 *psi_in, "   # estimate of solution before modulus replacement
            "float2 *psi_out, "  # estimate of solution after modulus replacement
            "float2 *out",       # output destination
            "out[i] = (1-support[i])*(psi_in[i]-beta*psi_out[i])+support[i]*psi_out[i]",
            "hio")
        
        self.realspace_constraint_er = cl_kernel(self.context,
            "float *support, "   # support constraint array
            "float2 *psi_out, "  # estimate of solution after modulus replacement
            "float2 *out",       # output destination
            "out[i] = support[i]*psi_out[i]",
            "hio")

        # if the support will be updated with shrinkwrap, initialize some additional gpu kernels
        if shrinkwrap:
            
            self.set_zero = cl_kernel(self.context,
                "float2 *buff",
                "buff[i] = (float2)(0.0f, 0.0f)",
                "set_zero")
            
            self.copy_real = cl_kernel(self.context,
                "float2 *in,"
                "float  *out",
                "out[i] = in[i].x",
                "set_zero")
            
            self.make_abs = cl_kernel(self.context,
                "float2 *in,"
                "float2 *out",
                "out[i] = (float2)(hypot(in[i].x,in[i].y),0.0f)",
                "make_abs")
            
            self.blur_convolve = cl_kernel(self.context,
                "float2 *toblur,"
                "float  *blurrer,"
                "float2 *blurred",
                "blurred[i] = (float2) (toblur[i].x*blurrer[i],toblur[i].y*blurrer[i])",
                "blur_convolve")
            
            self.support_threshold = cl_kernel(self.context,
                "float2 *in,"
                "float *out,"
                "float t",
                "out[i] = isgreaterequal(in[i].x,t)",
                "support_threshold")
            
            self.bound_support = cl_kernel(self.context,
                "float *s,"
                "float *s0",
                "s[i] = s[i]*s0[i]",
                "bound_support")

    def load_data(self,modulus,support,update_sigma=None):
        """ Put the modulus and support into GPU memory. Allocate memory for intermediate results.
        
        Required arguments:
            modulus: 2d numpy array from prephasing output
            support: 2d numpy array, should be binary 0/1
            
        Optional arguments:
            update_sigma: the size of the blurring kernel used for shrinkwrap support updating."""
        
        # make sure the modulus and support are the correct single-precision dtype
        assert modulus.ndim == 2, "modulus must be 2d"
        assert support.ndim == 2, "support must be 2d"
        assert modulus.shape == support.shape, "modulus and support must have same shape"
        
        # transfer the data to the cpu. pyopencl handles buffer creation
        self.modulus  = cla.to_device(self.queue,modulus.astype('float32'))
        self.support  = cla.to_device(self.queue,support.astype('float32'))
        self.support0 = cla.to_device(self.queue,support.astype('float32'))
        
        # initialize gpu arrays for fourier constraint satisfaction
        self.psi_in      = cla.empty(self.queue,(self.N,self.N),numpy.complex64)
        self.psi_out     = cla.empty(self.queue,(self.N,self.N),numpy.complex64) # need both in and out for hio algorithm
        self.psi_fourier = cla.empty(self.queue,(self.N,self.N),numpy.complex64) # to store fourier transforms of psi
        
        if update_sigma != None:
            from .. import shape
            # make gpu arrays for blurring the magnitude of the current estimate in order to update the support
            assert isinstance(update_sigma, (int,float)), "update_sigma must be float or int"
            blurkernel = abs(DFT(shift(shape.gaussian((self.N,self.N),(update_sigma,update_sigma),center=None,normalization=None))))
            self.blur_kernel   = cla.to_device(self.queue,blurkernel.astype('float32'))
            self.blur_temp     = cla.empty(self.queue,(self.N,self.N),numpy.complex64) # for holding the blurred estimate
            self.blur_temp_max = cla.empty(self.queue,(self.N,self.N),numpy.float32)
            
    def seed(self):
        """ Replaces self.psi_in with random numbers. Use this method to restart the simulation without
        having to copy a whole bunch of extra data like the support and the speckle modulus."""
        x = numpy.random.rand(self.N,self.N)
        y = numpy.random.rand(self.N,self.N)
        z = (x+complex(0,1)*y)
        self.psi_in.set(numpy.complex64(z),queue=self.queue)
 
    def update_support(self,threshold = 0.2,retain_bounds=True):
        
        """ Update the support using the shrinkwrap blur-and-threshold approach.
        self.psi_out is blurred according to update_sigma. Threshold can be specified and refers to
        which fraction of the max intensity is considered signal. retain_bounds = True means that the
        estimated support cannot exceed the initial specified support, useful if the initial
        support is known to be very loose. """
            
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
        m = (threshold*cla.max(self.blur_temp_max).get()).astype('float32')
        self.support_threshold(self.blur_temp,self.support,m).wait()
        
        # enforce the condition that the support shouldnt expand in size
        if retain_bounds: self.bound_support(self.support,self.support0)
        
        print "updated"
         
    def iteration(self,algorithm,beta=0.8):
        """ Do a single iteration of a phase retrieval algorithm.
        
        Arguments:
            algorithm: can be either 'hio' or 'er'
            beta: if using the hio algorithm, this sets the feeback parameter. default is 0.8"""

        assert algorithm in ('hio','er'), "real space enforcement algorithm %s is unknown"%algorithm

        # 1. fourier transform the data in psi_in, store the result in psi_fourier
        self.fftplan.execute(self.psi_in.data,data_out=self.psi_fourier.data)
        
        # 2. enforce the fourier constraint by replacing the current-estimated fourier modulus with the measured modulus from the ccd
        self.fourier_constraint(self.psi_fourier,self.modulus,self.psi_fourier)
        
        # 3. inverse fourier transform the new fourier estimate
        self.fftplan.execute(self.psi_fourier.data,data_out=self.psi_out.data,inverse=True)
        
        # 4. enforce the real space constraint. algorithm can be changed based on incoming keyword
        if algorithm == 'hio': self.realspace_constraint_hio(beta,self.support,self.psi_in,self.psi_out,self.psi_in)
        if algorithm == 'er':  self.realspace_constraint_er(self.support,self.psi_out,self.psi_in)



        
        
        
        
    
    
    
