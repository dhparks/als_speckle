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
            "out[i] = (float2)((1-support[i])*(psi_in[i].x-beta*psi_out[i].x)+support[i]*psi_out[i].x,(1-support[i])*(psi_in[i].y-beta*psi_out[i].y)+support[i]*psi_out[i].y)",
            "hio")
        
        self.realspace_constraint_er = cl_kernel(self.context,
            "float *support, "   # support constraint array
            "float2 *psi_out, "  # estimate of solution after modulus replacement
            "float2 *out",       # output destination
            "out[i] = support[i]*psi_out[i]",
            "hio")
        
        self.make_abs = cl_kernel(self.context,
            "float2 *in,"
            "float2 *out",
            "out[i] = (float2)(hypot(in[i].x,in[i].y),0.0f)",
            "make_abs")
        
        self.make_abs_f = cl_kernel(self.context,
            "float2 *in,"
            "float *out",
            "out[i] = hypot(in[i].x,in[i].y)",
            "make_abs_f")
        
        self.square_diff = cl_kernel(self.context,
            "float *in1,"
            "float *in2,"
            "float *out",
            "out[i] = pown(fabs(in1[i]-in2[i]),2)",
            "square_diff")
        
        self.copy_f2 = cl_kernel(self.context,
            "float2 *in,"
            "float2 *out",
            "out[i] = (float2)(in[i].x,in[i].y)",
            "copyf2")

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

        # 4. make unwrapping and resizing plans for computing the prtf on the gpu
        uy, ux = wrapping.unwrap_plan(0, self.N/2, (0,0), modulo=self.N)[:,:-1]
        self.unwrap_cols = len(ux)/(N/2)
        self.unwrap_x_gpu  = cla.to_device(self.queue,ux.astype(numpy.float32))
        self.unwrap_y_gpu  = cla.to_device(self.queue,uy.astype(numpy.float32))
        self.unwrapped_gpu = cla.empty(self.queue,(self.N/2,self.unwrap_cols), numpy.float32)
        self.map_coords    = gpu.build_kernel_file(self.context, self.device, kp+'map_coords_buffer.cl') # for unwrapping and resizing

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
        self.modulus  = cla.to_device(self.queue,modulus.astype(numpy.float32))
        self.support  = cla.to_device(self.queue,support.astype(numpy.float32))
        self.support0 = cla.to_device(self.queue,support.astype(numpy.float32))
        
        # initialize gpu arrays for fourier constraint satisfaction
        self.psi_in      = cla.empty(self.queue,(self.N,self.N),numpy.complex64) # holds the estimate of the wavefield
        self.psi_out     = cla.empty(self.queue,(self.N,self.N),numpy.complex64) # need both in and out for hio algorithm
        self.psi_fourier = cla.empty(self.queue,(self.N,self.N),numpy.complex64) # to store fourier transforms of psi
        self.psi_fourier2 = cla.empty(self.queue,(self.N,self.N),numpy.complex64) # to store fourier transforms of psi
        self.scratch_c   = cla.empty(self.queue,(self.N,self.N),numpy.complex64) # general purpose NxN complex buffer
        self.scratch_f   = cla.empty(self.queue,(self.N,self.N),numpy.float32 )  # general purpose NxN float buffer
        
        if update_sigma != None:
            from .. import shape
            # make gpu arrays for blurring the magnitude of the current estimate in order to update the support
            assert isinstance(update_sigma, (int,float)), "update_sigma must be float or int"
            blurkernel = abs(numpy.fft.fft2(numpy.fft.fftshift(shape.gaussian((self.N,self.N),(update_sigma,update_sigma),center=None,normalization=None))))
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
         
    def iteration(self,algorithm,beta=0.8,iter=None):
        """ Do a single iteration of a phase retrieval algorithm.
        
        Arguments:
            algorithm: can be either 'hio' or 'er'
            beta: if using the hio algorithm, this sets the feeback parameter. default is 0.8"""

        assert algorithm in ('hio','er'), "real space enforcement algorithm %s is unknown"%algorithm

        # 1. fourier transform the data in psi_in, store the result in psi_fourier
        self.fftplan.execute(self.psi_in.data,data_out=self.psi_fourier.data)
        self.copy_f2(self.psi_fourier,self.psi_fourier2)
        
        # 2. enforce the fourier constraint by replacing the current-estimated fourier modulus with the measured modulus from the ccd
        self.fourier_constraint(self.psi_fourier,self.modulus,self.psi_fourier)
        #elf.copy_f2(self.psi_fourier,self.psi_fourier2)
        
        # 3. inverse fourier transform the new fourier estimate
        self.fftplan.execute(self.psi_fourier.data,data_out=self.psi_out.data,inverse=True)
        
        # 4. enforce the real space constraint. algorithm can be changed based on incoming keyword
        if algorithm == 'hio':
            self.realspace_constraint_hio(beta,
                                          self.support,
                                          self.psi_in,
                                          self.psi_out,
                                          self.psi_in)
        if algorithm == 'er':  self.realspace_constraint_er(self.support,self.psi_out,self.psi_in)

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
        
    def prtf(self,data=None):

        # 1: make fourier abs
        if isinstance(data,cl.array.Array):
            touw = data
        if isinstance(data,numpy.ndarray):
            print data.dtype
            print data.shape
            touw = cla.to_device(self.queue,data.astype(numpy.complex64))
        if data == None:
            touw = self.psi_in
            self.realspace_constraint_er(self.support,touw,self.scratch_c)
        
        self.fftplan.execute(self.scratch_c.data,data_out=self.psi_fourier.data)
        self.make_abs_f(self.psi_fourier,self.scratch_f)
        
        # 2: unwrap using map_coords
        # inputs:
        # queue, size of output
        # input, size of input
        # output, size of output
        # plan_x, plan_y, interpolation order
        self.map_coords.execute(self.queue, (self.unwrap_cols,self.N/2),                                   
                   self.scratch_f.data,     numpy.int32(self.N),           numpy.int32(self.N),           
                   self.unwrapped_gpu.data, numpy.int32(self.unwrap_cols), numpy.int32(self.N/2),                
                   self.unwrap_x_gpu.data,  self.unwrap_y_gpu.data,        numpy.int32(3)).wait()
        temp1 = self.unwrapped_gpu.get()
        
        self.map_coords.execute(self.queue, (self.unwrap_cols,self.N/2),                                   
                   self.modulus.data,     numpy.int32(self.N),           numpy.int32(self.N),           
                   self.unwrapped_gpu.data, numpy.int32(self.unwrap_cols), numpy.int32(self.N/2),                
                   self.unwrap_x_gpu.data,  self.unwrap_y_gpu.data,        numpy.int32(3)).wait()
        temp2 = self.unwrapped_gpu.get()

        # calculate power ratios as a function of radius
        s1 = numpy.sum(temp1**2,axis=1)*(numpy.arange(0,self.N/2)+1) # reconstruction
        s2 = numpy.sum(temp2**2,axis=1)*(numpy.arange(0,self.N/2)+1) # modulus; this could be precomputed...

        return s1/s2,s1,s2
        

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
        
    
    
    
