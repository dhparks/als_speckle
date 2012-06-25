# core
import numpy
import pyopencl as cl
import pyopencl.array as cl_array
from pyopencl.elementwise import ElementwiseKernel
import pyfft

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
        self.estimate   = ((numpy.rand(self.N,self.N)+complex(0,1)*numpy.rand(self.N,self.N))*self.support)
        
        if update_sigma != None:
            assert isinstance(update_sigma, (int,float)), "update_sigma must be float or int"
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
        
        DFT = numpy.fft.fft2
        IDFT = numpy.fft.ifft2
        
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

class GPUPR:
    
    """ Implement phase retrieval on the GPU to take advantage of the embarassingly-parallel nature of the
    phase-retrieval algorithms. 
    
    Instantiating the class requires information about the gpu (device, context, queue) obtained from
    gpu.init(), and the simulation size (ie, 256x256, 512x512) to initialize the FFT code.
    
    Methods:
        load_data: puts constraint data onto the gpu. Requires 2 arrays, the fourier modulus and the support.
        seed: sets up an independent reconstruction by putting a new complex-random seed into GPU memory
        iteration: runs a single iteration of either the HIO or the ER algorithm
        update_support: if using the shrinkwrap algorithm, this updates the support by blurring and thresholding.
    """
    
    def __init__(self,gpuinfo,N,shrinkwrap=False):
        self.N = N
        self.context,self.device,self.queue = gpuinfo

        # 1. make fft plan for a 2d array with length N
        from pyfft.cl import Plan
        self.fftplan = Plan((self.N, self.N), queue=self.queue)
        
        # 2. make the kernels to enforce the fourier and real-space constraints
        self.fourier_constraint = ElementwiseKernel(self.context,
            "float2 *psi, "                        # current estimate of the solution
            "float  *modulus, "                    # known fourier modulus
            "float2 *out",                         # output destination
            "out[i] = rescale(psi[i],modulus[i])", # operator definition
            "replace_modulus",
            preamble = """
            #define rescale(a,b) (float2)(a.x/hypot(a.x,a.y)*b,a.y/hypot(a.x,a.y)*b)
            """)
        
        self.realspace_constraint_hio = ElementwiseKernel(self.context,
            "float beta, "       # feedback parameter
            "float *support, "   # support constraint array
            "float2 *psi_in, "   # estimate of solution before modulus replacement
            "float2 *psi_out, "  # estimate of solution after modulus replacement
            "float2 *out",       # output destination
            "out[i] = (1-support[i])*(psi_in[i]-beta*psi_out[i])+support[i]*psi_out[i]",
            "hio")
        
        self.realspace_constraint_er = ElementwiseKernel(self.context,
            "float *support, "   # support constraint array
            "float2 *psi_out, "  # estimate of solution after modulus replacement
            "float2 *out",       # output destination
            "out[i] = support[i]*psi_out[i]",
            "hio")

        # if the support will be updated with shrinkwrap, initialize some additional gpu kernels
        if shrinkwrap:
            
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
        assert modulus.shape == support.shape, "modulus and support must have same shape"
        
        # transfer the data to the cpu. pyopencl handles buffer creation
        self.modulus  = cl_array.to_device(self.queue,modulus.astype('float32'))
        self.support  = cl_array.to_device(self.queue,support.astype('float32'))
        self.support0 = cl_array.to_device(self.queue,support.astype('float32'))
        
        # initialize gpu arrays for fourier constraint satisfaction
        self.psi_in      = cl_array.empty(self.queue,(self.N,self.N),numpy.complex64)
        self.psi_out     = cl_array.empty(self.queue,(self.N,self.N),numpy.complex64) # need both in and out for hio algorithm
        self.psi_fourier = cl_array.empty(self.queue,(self.N,self.N),numpy.complex64) # to store fourier transforms of psi
        
        if update_sigma != None:
            import shape.gaussian
            # make gpu arrays for blurring the magnitude of the current estimate in order to update the support
            assert isinstance(update_sigma, (int,float)), "update_sigma must be float or int"
            blurkernel = abs(DFT(fftshift(shape.gaussian((self.N,self.N),(update_sigma,update_sigma),center=None,normalization=None))))
            self.blur_kernel   = cl_array.to_device(self.queue,blurkernel.astype('float32'))
            self.blur_temp     = cl_array.empty(self.queue,(self.N,self.N),numpy.complex64) # for holding the blurred estimate
            self.blur_temp_max = cl_array.empty(self.queue,(self.N,self.N),numpy.float32)
            
    def seed(self):
        # use this method to restart the simulation without having to copy a whole bunch of extra data
        # like the support and the speckle modulus. this releases the old psi_in from memory and puts
        # a new random seed into the self.psi_in name.
        x = numpy.random.rand(self.N,self.N)
        y = numpy.random.rand(self.N,self.N)
        z = (x+complex(0,1)*y)
        #self.psi_in.release()
        self.psi_in = cl_array.to_device(self.queue,numpy.complex64(z))
 
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
        
        # enforce the condition that the support shouldnt expand in size
        if retain_bounds: self.bound_support(self.support,self.support0)
        
        print "updated"
         
    def iteration(self,algorithm,beta=0.8):

        # do a single iteration. algorithm is selectable through keyword

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
    
    imag_power = lambda p,x: numpy.sum(abs((x*numpy.exp(complex(0,1)*p)).imag))
    
    if data.ndim == 2:
        was2d = True
        data.shape = (1,data.shape[0],data.shape[1])
        
    for frame in data:
        x = frame.ravel()
        e = lambda p: numpy.sum(abs((x*numpy.exp(complex(0,1)*p)).imag))
        opt, val, conv, num = fminbound(e,0,2*numpy.pi,full_output=1)
        frame *= numpy.exp(complex(0,1)*opt)
    
    if was2d: data = data[0]
    
    return data


        
        
        
        
    
    
    
