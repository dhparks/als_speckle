import numpy
import pyopencl as cl
import pyopencl.array as cla
from pyopencl.elementwise import ElementwiseKernel as cl_kernel
import pyfft
import gpu
import string
kp = string.join(gpu.__file__.split('/')[:-1],'/')+'/kernels/'

from .. import wrapping # for prtf

class GPUPR:
    
    """ Implement phase retrieval on the GPU to take advantage of the
    embarassingly-parallel nature of the phase-retrieval algorithms. 
    
    Instantiating the class requires information about the gpu (device, context,
    queue) obtained from gpu.init(), and the simulation size (ie, 256x256,
    512x512) to initialize the FFT code.
    
    methods:
        load_data: puts constraint data onto the gpu. Requires 2 arrays, the
            fourier modulus and the support.
        seed: sets up an independent reconstruction by putting a new
            complex-random seed into GPU memory
        iteration: runs a single iteration of either the HIO or the ER algorithm
        update_support: if using the shrinkwrap algorithm, this updates the
            support by blurring and thresholding.        
    """
    
    def __init__(self,device=None):
        
        self.ints = (int,numpy.int32,numpy.int64) # integer types
        self.iters = (list,tuple,numpy.ndarray)     # iterable types
        self.floats = (float,numpy.float32,numpy.float64)
        
        assert isinstance(device,tuple) and len(device) == 4, "device information improperly formed"
        self.context,self.device,self.queue,self.platform = device
        
        self.can_has_modulus = False
        self.can_has_support = False
        self.can_has_psff    = False
        self.can_iterate     = False
        self.N               = None
        
        self.can_has_estimate = False # for rl deconvolution
        
        self._make_kernels()
        
        self.array_dtypes = ('float32','complex64')

    def _make_kernels(self):

        self.rl_make_estimate = cl_kernel(self.context,
            "float *in,"   # this is always self.modulus                           
            "float2 *out", # this is always estimate
            "out[i] = (float2)(in[i]*in[i],0)",
            "rl_make_estimate")
        
        self.square_diff = cl_kernel(self.context,
            "float *in1,"
            "float *in2,"
            "float *out",
            "out[i] = pown(fabs(in1[i]-in2[i]),2)",
            "square_diff")
        
        # compile basic mathematical functions for multiplication, division, abs
        self.mult_f_f   = gpu.build_kernel_file(self.context, self.device, kp+'basic_multiply_f_f.cl')   # float*float
        self.mult_f_f2  = gpu.build_kernel_file(self.context, self.device, kp+'basic_multiply_f_f2.cl')  # float*complex
        self.mult_f2_f2 = gpu.build_kernel_file(self.context, self.device, kp+'basic_multiply_f2_f2.cl') # complex*complex
        
        self.div_f_f    = gpu.build_kernel_file(self.context, self.device, kp+'basic_divide_f_f.cl')     # float/float
        self.div_f_f2   = gpu.build_kernel_file(self.context, self.device, kp+'basic_divide_f_f2.cl')    # float/complex
        self.div_f2_f   = gpu.build_kernel_file(self.context, self.device, kp+'basic_divide_f2_f.cl')    # complex/float
        self.div_f2_f2  = gpu.build_kernel_file(self.context, self.device, kp+'basic_divide_f2_f2.cl')   # complex/complex
        
        self.abs_f2_f   = gpu.build_kernel_file(self.context, self.device, kp+'basic_abs_f2_f.cl')       # abs cast to float
        self.abs_f2_f2  = gpu.build_kernel_file(self.context, self.device, kp+'basic_abs_f2_f2.cl')      # abs kept as cmplx
        
        self.complex_sqrt = gpu.build_kernel_file(self.context, self.device, kp+'basic_sqrt_f2.cl')      # square of complex number

        # compile kernels specific to phasing
        self.phasing_er      = self.mult_f_f2
        self.phasing_hio     = gpu.build_kernel_file(self.context, self.device, kp+'phasing_hio.cl')
        self.phasing_fourier = gpu.build_kernel_file(self.context, self.device, kp+'phasing_fourier.cl')
        

        # 4. make unwrapping and resizing plans for computing the prtf on the gpu
        #uy, ux = wrapping.unwrap_plan(0, self.N/2, (0,0), modulo=self.N)[:,:-1]
        #self.unwrap_cols   = len(ux)/(self.N/2)
        #self.unwrap_x_gpu  = cla.to_device(self.queue,ux.astype(numpy.float32))
        #self.unwrap_y_gpu  = cla.to_device(self.queue,uy.astype(numpy.float32))
        #self.unwrapped_gpu = cla.empty(self.queue,(self.N/2,self.unwrap_cols), numpy.float32)

    def _cl_mult(self,in1,in2,out):
        """ Wrapper function to the various array-multiplication kernels. Every
        combination of input requires a different kernel. This function checks
        dtypes and automatically selects the appropriate kernel.
        
        in1, in2 are the input arrays being multiplied
        out is the output array where the result is stored
        
        All passed arguments should be the pyopencl array, NOT the data attribute.
        """
        
        d1, d2, d3 = in1.dtype, in2.dtype, out.dtype
        assert d1 in self.array_dtypes, "in1 dtype not recognized; is %s"%d1
        assert d2 in self.array_dtypes, "in2 dtype not recognized; is %s"%d2
        assert d3 in self.array_dtypes, "out dtype not recognized; is %s"%d3
        assert in1.shape == in2.shape and in1.shape == out.shape, "all arrays must have same shape"
        N = in1.size
        
        if d1 == 'float32':
            if d2 == 'float32':
                func = self.mult_f_f
                assert d3 == 'float32', "float * float = float"
                arg1 = in1
                arg2 = in2
                
            if d2 == 'complex64':
                func = self.mult_f_f2
                assert d3 == 'complex64', "float * complex = complex"
                arg1 = in1
                arg2 = in2
                
        if d2 == 'complex64':
            if d2 == 'float32':
                func = self.mult.f_f2
                assert d3 == 'complex64', "float * complex = complex"
                arg1 = in2
                arg2 = in1
                
            if d2 == 'complex64':
                func = self.mult_f2_f2
                assert d3 == 'complex64', "complex * complex = complex"
                arg1 = in1
                arg2 = in2
                
        func.execute(self.queue,(N,), arg1.data, arg2.data, out.data)
        
    def _cl_abs(self,in1,out):
        """ Wrapper func to the various abs kernels. Checks types of in1 and out
        to select appropriate kernel. """
        
        d1, d2 = in1.dtype, out.dtype
        assert d1 in self.array_dtypes, "in1 dtype not recognized"
        assert d2 in self.array_dtypes, "out dtype not recognized"
        assert in1.shape == out.shape,  "in1 and out must have same shape"
        N = in1.size
    
        assert d1 == 'complex64', "no abs func for in1 dtype float"
        if d2 == 'float32': func = self.abs_f2_f
        if d2 == 'complex64': func = self.abs_f2_f2
        
        func.execute(self.queue,(N,),in1.data,out.data)
        
    def _cl_div(self,in1,in2,out):
        """ Wrapper func to various division kernels. Checks type of in1, in2,
        and out to select the appropriate kernels.
        
        in1, in2 are the numerator and denominator so ORDER MATTERS
        out is the output
        
        All arguments should be the pyopencl array, NOT the data attribute.
        """
        
        d1, d2, d3 = in1.dtype, in2.dtype, out.dtype
        assert d1 in self.array_dtypes, "in1 dtype not recognized; is %s"%d1
        assert d2 in self.array_dtypes, "in2 dtype not recognized; is %s"%d2
        assert d3 in self.array_dtypes, "out dtype not recognized; is %s"%d3
        assert in1.shape == in2.shape and in1.shape == out.shape, "all arrays must have same shape"
        N = in1.size
            
        if d1 == 'float32':
            if d2 == 'float32':
                func = self.div_f_f
                assert d3 == 'float32', "float / float = float"
                
            if d2 == 'complex64':
                func = self.div_f_f2
                assert d3 == 'complex64', "float / complex = complex"

        if d2 == 'complex64':
            if d2 == 'float32':
                func = self.div_f2_f
                assert d3 == 'complex64', "complex / float = complex"
                
            if d2 == 'complex64':
                func = self.div_f2_f2
                assert d3 == 'complex64', "complex / complex = complex"
                
        func.execute(self.queue,(N,),in1.data,in2.data,out.data)

    def load_data(self,modulus=None,support=None,update_sigma=None,psf=None):
        """ Load objects into the GPU memory. This is all handled through one
        function which tracks the sizes of the arrays. After the initial loading,
        the size of the objects is LOCKED and cannot be changed.
        
        When loaded, the modulus, support, and psf must be equal sizes.
        The calculation must have a support and a modulus to procede.
        If a psf is loaded, it is also assumed that the fourier modulus is
        partially coherent, and algorithms which account for this partial
        coherence are employed."""
        
        # check types, sizes, etc
        types = (type(None),numpy.ndarray)
        assert isinstance(modulus,types), "modulus must be ndarray if supplied"
        assert isinstance(support,types), "support must be ndarray if supplied"
        assert isinstance(psf,types), "psf must be ndarray if supplied"
        
        # load or replace modulus
        if isinstance(modulus,numpy.ndarray):
            assert modulus.ndim == 2, "modulus must be 2 dimensional"
            assert modulus.shape[0] == modulus.shape[1], "modulus must be square"
            
            if self.can_has_modulus:
                assert modulus.shape == (self.N,self.N), "modulus size cannot change when reloading!"
                self.modulus.set(modulus.astype(numpy.float32),queue=self.queue)
                
            if not self.can_has_modulus:
                self.can_has_modulus = True
                self.modulus = cla.to_device(self.queue,modulus.astype(numpy.float32))
            
        # load or replace support
        if isinstance(support,numpy.ndarray):
            assert support.ndim == 2, "support must be 2 dimensional"
            assert support.shape[0] == support.shape[1], "support must be square"
            
            if self.can_has_support:
                assert support.shape == (self.N,self.N), "support size cannot change when reloading!"
                self.support.set(support.astype(numpy.float32),queue=self.queue)  # to update
                self.support0.set(support.astype(numpy.float32),queue=self.queue) # read only?
                
            if not self.can_has_support:
                self.can_has_support = True
                self.support  = cla.to_device(self.queue,support.astype(numpy.float32)) # to update
                self.support0 = cla.to_device(self.queue,support.astype(numpy.float32)) # read only?
        
        # load or replace psf
        if isinstance(psf,numpy.ndarray):
            assert psf.ndim == 2, "psf must be 2 dimensional"
            assert psf.shape[0] == psf.shape[1], "psf must be square"
            
            psff = numpy.fft.fft2(psf) # psf needs to come in corner-centered!
            
            if self.can_has_psff:
                assert psf.shape == (self.N,self.N), "psf size cannot change when reloading!"
                self.psff.set(psff.astype(numpy.float32),queue=self.queue)
                
            if not self.can_has_psff:
                self.can_has_psff = True
                self.psff = cla.to_device(self.queue,psff.astype(numpy.float32))
        
        # verify that everything currently loaded has the same shape. this is
        # inefficient but reliable.
        m = self.can_has_modulus
        s = self.can_has_support
        p = self.can_has_psff
        
        if m:
            if s: assert self.modulus.shape == self.support.shape, "modulus and support shapes differ"
            if p: assert self.modulus.shape == self.psff.shape,    "modulus and psff shapes differ"
        if s:
            if m: assert self.support.shape == self.modulus.shape, "modulus and support shapes differ"
            if p: assert self.support.shape == self.psff.shape,    "support and psff shapes differ"
        if p:
            if s: assert self.psff.shape == self.support.shape,    "psff and support shapes differ"
            if m: assert self.psff.shape == self.modulus.shape,    "psff and modulus shapes differ"
            
        if m: self.N = self.modulus.shape[0]
        if s: self.N = self.modulus.shape[0]
        if p: self.N = self.modulus.shape[0]
        if self.N != None:
            self.N2 = self.N*self.N
            from pyfft.cl import Plan
            self.fftplan = Plan((self.N, self.N), queue=self.queue)
        
        # initialize gpu arrays for fourier constraint satisfaction
        if m or s or p:
            self.psi_in      = cla.empty(self.queue,(self.N,self.N),numpy.complex64) # holds the estimate of the wavefield
            self.psi_out     = cla.empty(self.queue,(self.N,self.N),numpy.complex64) # need both in and out for hio algorithm
            self.psi_fourier = cla.empty(self.queue,(self.N,self.N),numpy.complex64) # to store fourier transforms of psi
            self.fourier_div = cla.empty(self.queue,(self.N,self.N),numpy.complex64)   # stores divisor generate from psi_fourier
            self.scratch_f   = cla.empty(self.queue,(self.N,self.N),numpy.float32)   # general purpose NxN float buffer
            
        # set the flag which allows iteration.
        if m and s: self.can_iterate = True
            
    def richardson_lucy(self,iterations):
        """ Given:
                1. the speckle pattern
                2. an estimate of the psf (coherence function)
            use the richardson-lucy algorithm to attempt a deconvolution of the
            speckle to restore the fully-coherent speckle. """

        # steps in richardson-lucy algorithm:
        # 1. form an estimate of the perfectly coherent image (ie random seed)
        # 2. form the inverse psf
        # 3. in a iterative fashion, do the following:
        #       4. b = convolve(est, psf)
        #       5. d = pc/b
        #       6. e = convolve(d, ipsf)
        #       7. est *= e
        # this go can for a number of iterations, or until a termination condition is reached?
        # therefore, we need the following buffers:
        
        assert self.can_has_psf, "cant do richardson lucy without a psf!"
        assert isinstance(iterations,self.ints)
        
        if not self.can_has_estimate:
            self.rl_est   = cla.empty(self.queue,(self.N,self.N),numpy.complex64)
            self.rl_pc    = cla.empty(self.queue,(self.N,self.N),numpy.complex64)
            self.rl_blur1 = cla.empty(self.queue,(self.N,self.N),numpy.complex64)
            self.rl_div   = cla.empty(self.queue,(self.N,self.N),numpy.complex64)
            self.rl_blur2 = cla.empty(self.queue,(self.N,self.N),numpy.complex64)
            self.can_has_estimate = True    
        self.rl_make_estimate(self.modulus,self.rl_est)
        self.rl_make_estimate(self.modulus,self.rl_pc)
    
        for i in range(iterations):
            _convolvef(self.rl_est,self.psff,self.rl_blur1)
            self._cl_div(self.rl_pc,self.rl_blur1,self.rl_div)
            _convolvef(self.rl_div,self.psffr,self.rl_blur2)
            self._cl_mult(self.rl_est,self.rl_blur2,self.rl_est)
    
    def _convolvef(self,d1,d2,out):
        # calculate a convolution when d1 must be transformed but d2 is already
        # transformed. the multiplication function depends on the dtype of d2.
        
        assert d1.dtype == 'complex64',  "in _convolvef, input d1 has wrong dtype for fftplan"
        assert out.dtype == 'complex64', "in _convolvef, output has wrong dtype"
        
        self._cl_fft(in1=data,out1=out)
        self._cl_mult(out,d2,out)
        self._cl_fft(in1=out,out1=out,inverse=True)
    
    def seed(self,supplied=None):
        """ Replaces self.psi_in with random numbers. Use this method to restart
        the simulation without having to copy a whole bunch of extra data like
        the support and the speckle modulus.
        
        arguments:
            supplied - (optional) can set the seed with a precomputed array.
        """
        
        assert self.N != None, "cannot seed without N being set. load data first."
        
        if supplied != None:
            self.psi_in.set(supplied.astype(numpy.complex64))
        if supplied == None:
            x = numpy.random.rand(self.N,self.N)
            y = numpy.random.rand(self.N,self.N)
            z = (x+complex(0,1)*y)
            self.psi_in.set(numpy.complex64(z),queue=self.queue)
 
    def update_support(self,threshold = 0.2,retain_bounds=True):
        
        """ Update the support using the shrinkwrap blur-and-threshold approach.
        self.psi_out is blurred according to update_sigma. Threshold can be
        specified and refers to which fraction of the max intensity is
        considered signal. retain_bounds = True means that the estimated support
        cannot exceed the initial specified support, useful if the initial
        support is known to be very loose. """
            
        # zero the temp buffer for safety. this is basically a free operation
        self.set_zero(self.blur_temp)

        # make a blurry version of the abs of psi_out with _convolvef
        self.make_abs(self.psi_out,self.blur_temp).wait()
        self._convolvef(self.blur_temp,self.blur_kernel,self.blur_temp)
        
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
        
        # 2. from the data in psi_fourier, build the divisor. depending on whether self.psff has
        # been defined, this will be differently executed. _build_divisor_2 implements clark2011 (apl)
        def _build_divisor_1(self,psi,div):
            self._cl_abs.execute(psi,div)
        def _build_divisor_2(self,psi,div,psf):
            assert psi.dtype == 'complex64', "in _build_divisor_2, psi is wrong dtype"
            assert div.dtype == 'complex64', "in _build_divisor_2, div is wrong dtype"
            assert psf.dtype == 'float32'  , "in _build_divisor_2, psf is wrong dtype"
            self._cl_abs(psi,div)      # put the abs of psi in div
            self._cl_mult(div,div,div) # square div in place
            
            # start convolution
            self._cl_fft(div.data,wait_for_finish=True)              # fft div in place, div must be COMPLEX
            self._cl_mult(div,psf,div)                                       # multiply by psff
            self.fftplan.execute(div.data,inverse=True,wait_for_finish=True) # ifft in place
            # end convolution
            
            # sqrt makes the pc modulus from intensity
            self.complex_sqrt.execute(self.queue,(self.N2,),div,div)    # sqrt div in place
            
        if not self.can_has_psff: _build_divisor_1(self,self.psi_fourier,self.fourier_div) # fully coherent, no psf
        if self.can_has_psff:     _build_divisor_2(self,self.psi_fourier,self.fourier_div,self.psff) # partially coherent, has a psf

        # 3. enforce the fourier constraint
        self.phasing_fourier.execute(self.queue,(self.N2,), self.psi_fourier.data,self.fourier_div.data,self.modulus.data,self.psi_fourier.data)

        # 4. inverse fourier transform the new fourier estimate
        self.fftplan.execute(self.psi_fourier.data,data_out=self.psi_out.data,inverse=True)
        
        # 5. enforce the real space constraint. algorithm can be changed based on incoming keyword
        if algorithm == 'hio':
            self.phasing_hio.execute(self.queue,(self.N2,),                 # opencl stuff
                                    numpy.float32(beta), self.support.data, # inputs
                                    self.psi_in.data, self.psi_out.data,    # inputs
                                    self.psi_in.data)                       # output
        if algorithm == 'er':
            self.phasing_er.execute(self.queue,(self.N2,),                  # opencl stuff
                                    self.support.data,self.psi_out.data,    # inputs
                                    self.psi_in.data)                       # output
        
    def iterate(self,iterations,update_period=0):
    
        """ Run iterations. This is the primary method for the class.
    
        Arguments:
            phasing_instance - an instance of either the GPUPR or CPUPR class.
            iterations       - how many iterations to run total
            shrinkwrap       - whether to run the shrinkwrap functions
            update_period    - (optional) if using shrinkwrap, how many iterations
                between running the update functions
        """
             
        assert isinstance(iterations,self.ints), "iterations must be integer, is %s"%type(iterations)
        #assert isinstance(shrinkwrap,bool), "shrinkwrap must be boolean, is %s"%type(shrinkwrap)
        assert self.can_iterate, "cant iterate before loading support and modulus."
       
        for iteration in range(iterations):
            
            self.i_n = iteration
            if (iteration+1)%100 != 0:
                self.iteration('hio')
            else: self.iteration('er')
            if iteration%100 == 0: print "  iteration %s"%iteration
            
            #if shrinkwrap and iteration%update_period == 0 and iteration > 0:
            #    self.update_support()
            #    self.iteration('er')
            #else:
            #    # alternate 99 hio / 1 er. this ratio can be changed
            #    if (iteration+1)%100 != 0: self.iteration('hio')
            #    else: self.iteration('er')
            #if iteration%100 == 0: print "  iteration %s"%iteration
         
    def get(self,array):
        # wraps get to unify cpu/gpu api
        return array.get()

    def fourier_error(self):
        # 1. FFT psi in
        # 2. Create fourier modulus, store in scratch_f
        # 3. Subtract the current modulus from the goal modulus
        # 4. Square the difference, then sum. This is the fourier error
        
        self.realspace_constraint_er(self.support,self.psi_in,self.scratch_c)
        self.fftplan.execute(self.scratch_c.data,data_out=self.psi_fourier.data)
        self.make_abs_f(self.psi_fourier,self.scratch_f)
        self.square_diff(self.scratch_f,self.modulus,self.scratch_f)
        
        return cla.sum(self.scratch_f).get()
        
def covar_results(gpuinfo,data,threshold=0.85):
    
    import sys
    from pyfft.cl import Plan
    import gpu
    kp = gpu.__file__.replace('gpu.pyc','kernels/')
    
    context,device,queue,platform = gpuinfo
    
    assert isinstance(data,numpy.ndarray), "data must be an array"
    data = abs(data).astype(numpy.complex64)
    
    frames, rows, cols = data.shape

    # put the data in a square array with length a power of 2 (required for pyfft)
    r2 = int(round(numpy.log2(rows)))+1
    c2 = int(round(numpy.log2(cols)))+1
    if r2 >= c2: N = 2**(r2)
    else: N = 2**(c2)
    
    new_data = numpy.zeros((frames,N,N),numpy.complex64)
    new_data[:,:rows,:cols] = data
    
    # set up buffers
    cc       = numpy.zeros((frames,frames),float) # in host memory
    covars   = numpy.zeros((frames,frames),float) # in host memory
    gpu_data = cla.empty(queue, (frames,N,N), numpy.complex64)
    dft1     = cla.empty(queue, (N,N), numpy.complex64) # hold first dft to crosscorrelate
    dft2     = cla.empty(queue, (N,N), numpy.complex64) # hold second dft to crosscorrelate
    product  = cla.empty(queue, (N,N), numpy.complex64) # hold the product of conj(dft1)*dft2
    corr     = cla.empty(queue, (N,N), numpy.float32)   # hold the abs of idft(product)
        
    # make the gpu kernels
    fft_N = Plan((N,N), queue=queue)
    
    conj_mult = cl_kernel(context,
                "float2 *dft1,"  
                "float2 *dft2,"  
                "float2 *out",  
                "out[i] = (float2)(dft1[i].x*dft2[i].x+dft1[i].y*dft2[i].y,dft1[i].x*dft2[i].y-dft1[i].y*dft2[i].x)",
                "conj_mult")
                
    make_abs = cl_kernel(context,
                "float2 *in,"
                "float *out",
                "out[i] = hypot(in[i].x,in[i].y)",
                "make_abs")
                
    slice_covar = gpu.build_kernel_file(context, device, kp+'slice_covar.cl')
    
    # put the data on the gpu
    gpu_data.set(new_data)
    
    # precompute the dfts by running fft_interleaved as a batch. store in-place.
    print "precomputing ffts"
    fft_N.execute(gpu_data.data,batch=frames)
    print "done"
    
    # now iterate through the CCs, cross correlating each pair of dfts
    iter = 0
    total = frames**2/2
    for n in range(frames):
        print "covar %s"%n

        # get the first frame buffered
        slice_covar.execute(queue, (N,N), # opencl stuff
            gpu_data.data, dft1.data,     # src and dest
            numpy.int32(n)).wait()           # which frame to grab
                       
        for m in range(frames-n):
            m += n
        
            # get the second frame buffered
            slice_covar.execute(queue, (N,N), # opencl stuff
                gpu_data.data, dft2.data,     # src and dest
                numpy.int32(m)).wait()        # which frame to grab
                
            # multiply conj(dft1) and dft2. store in product. inverse transform
            # product; keep in place. make the magnitude of product in corr. take
            # the max of corr and return it to host.
            conj_mult(dft1,dft2,product)
            
            if n == 1 and m == 2:
                gpu_product = product.get()
                gpu_dft1 = dft1.get()
                gpu_dft2 = dft2.get()
            
            fft_N.execute(product.data,inverse=True)
            make_abs(product,corr)
            max_val = cla.max(corr).get()
            cc[n,m] = max_val
            cc[m,n] = max_val
            
            iter += 1

    # now turn the cc values into normalized covars:
    # covar(i,j) = cc(i,j)/sqrt(cc(i,i)*cc(j,j))
    for n in range(frames):
        for m in range(frames-n):
            m += n
            covar = cc[n,m]/numpy.sqrt(cc[n,n]*cc[m,m])
            covars[n,m] = covar
            covars[m,n] = covar
            
    covars = numpy.nan_to_num(covars)
    
    stats = None
            
    if threshold > 0:
        # count which reconstructions are most like each other
        rows, cols = covars.shape
        stats = numpy.zeros((rows,3),float)
        for row in range(rows):
            average = numpy.average(covars[row])
            passed  = numpy.sum(numpy.where(covars[row] > threshold,1,0))
            stats[row] = row, average, passed
        
    return cc, covars, stats
        
    
    
    
