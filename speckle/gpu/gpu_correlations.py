# core
import numpy as np
import pyopencl as cl
import pyopencl.array as cla
from pyopencl.elementwise import ElementwiseKernel as EK
from pyfft.cl import Plan as fft_plan
import time
import string

# common libs. do some ugly stuff to get the path set to the kernels directory
from .. import wrapping,shape,crosscorr,io
import gpu
kp = string.join(gpu.__file__.split('/')[:-1],'/')+'/kernels/'

# cpu fft
DFT = np.fft.fft2
IDFT = np.fft.ifft2
fftshift = np.fft.fftshift
fs = fftshift

class gpu_microscope():
    
    """Methods for running a symmetry-microscope simulation on a gpu using
    pyopencl and pyfft libraries"""
    
    def __init__(self,device):
        """ Get the class running with gpu info. Acceptance of input has now
        been moved into a separate function, as that is how it was used in
        all my usage code anyways."""
        
        assert isinstance(device,tuple) and len(device) == 4, "device information improperly formed"
        self.context, self.device, self.queue, self.platform = device
        
        # these turn to true in the load function. based on what has been
        # loaded, execution may be disallowed (ie, need an object and illumination function)
        self.can_has_object      = False
        self.can_has_unwrap      = False
        self.can_has_pinhole     = False
        self.can_has_coherence   = False
        self.can_has_cosines     = False
        self.resize_speckles     = False
        self.can_has_blocker     = False
        self.can_has_psff        = False
        self.can_run_scan        = False
        
        # these must all be set for execution to proceed
        self.obj_N = None
        self.pin_N = None
        self.psf_N = None
        
        self._make_kernels()
        
        # returnables. self.keywords gives all the outputs that can be generated,
        # usually for debugging purposes.
        self.returnables = {}
        self.keywords = ('object','sliced','illuminated','speckle','speckle_blocker',
                        'blurred','blurred_blocker','unwrapped','resized','correlated',
                        'spectrum','spectrum_ds','correlated_ds','rspeckle')
        
        self.norm_mode = 0
        self.interp_order = 1
        
        # set up timings
        self.speckle_time = 0
        self.resize_time1 = 0
        self.blur_time    = 0
        self.unwrap_time  = 0
        self.resize_time2 = 0
        self.decomp_time  = 0
        self.despike_time = 0
        self.correl_time  = 0
        self.resize_time3 = 0
        self.decomp_time  = 0
        self.despike_time = 0
        
    def load(self,object=None,pinhole=None,psf=None,unwrap=None,components=None,returnables=None,blocker=None):
        """ This function is the recommended way to set simulation parameters, as
        it does type and consistency checking.
        
        Fields which must be set:
        1. object  - the scattering object. must be a 2d array.
        2. pinhole - the illumination function. must be a 2d square array.
        3. psf     - the point spread function which simulates partial spatial
            coherence. must be a 2d array. should be corner-centered and real
            positive, although only real-valued is checked. THE SIZE OF PSF SETS
            THE SIZE OF THE SPECKLE PATTERN.
        4. unwrap  - the range of unwrapping the speckle pattern.
            format is (inner_radius, outer_radius). cannot be changed later.
        5. cosines - a list of components ie (4,6,8,10,12,14,16). the cosine
            values will be generated in this function.
        6. returnables - the desired output keywords
        7. blocker - width of the blocker stick in pixels. not yet implemented.
        """
        
        # on the first pass, require that object, pinhole, psf, unwrap, and cosines ALL be specified.
        if self.can_has_object == False:
            assert object != None, "must load an object"
            assert pinhole != None, "must load a pinhole"
            assert psf != None, "must load a psf"
            assert components != None, "must load cosine components"
            assert unwrap != None, "must specify unwrap"
        
        # load the object
        if object != None:
            assert isinstance(object,np.ndarray), "object must be numpy array"
            assert object.ndim == 2, "object must be 2d"
            assert object.shape[0] == object.shape[1], "object must be square (for now?)"
            
            if self.can_has_object:
                assert object.shape == self.master_object.shape, "new object must be same shape as old object"
                self.master_object.set(object.astype(self.master_object.dtype))
                
            if not self.can_has_object:
                self.obj_N = object.shape[0]
                self.master_object = cla.to_device(self.queue,object.astype(np.complex64))
                self.can_has_object = True
                
        # load the pinhole
        if pinhole != None:
            assert isinstance(pinhole,np.ndarray), "pinhole must be a numpy array"
            assert pinhole.ndim == 2, "pinhole must be 2d"
            assert pinhole.shape[0] == pinhole.shape[1], "pinhole must be square"
            
            if self.can_has_pinhole:
                assert pinhole.shape == self.pinhole.shape, "new pinhole must have same shape as old pinhole"
                self.pinhole.set(pinhole.astype(self.pinhole.dtype))
                
            if not self.can_has_pinhole:
                self.pin_N = pinhole.shape[0]
                self.pinhole = cla.to_device(self.queue,pinhole.astype(np.complex64))
                
                # allocate memory for intermediate results in the speckle calculation. these all need to be (pin_N x pin_N) 
                self.active_object = cla.empty(self.queue, (self.pin_N,self.pin_N), np.complex64)
                self.illuminated   = cla.empty(self.queue, (self.pin_N,self.pin_N), np.complex64)
                self.far_field     = cla.empty(self.queue, (self.pin_N,self.pin_N), np.complex64)
                self.speckles      = cla.empty(self.queue, (self.pin_N,self.pin_N), np.complex64)
                self.speckles_f   = cla.empty(self.queue, (self.pin_N,self.pin_N), np.float32)
                self.can_has_pinhole = True
                
            assert ispower2(self.pin_N), "pinhole size must be a power of 2"
                
        # load the psf
        if psf != None:
            assert isinstance(psf,np.ndarray), "psf must be a numpy array"
            assert psf.ndim == 2,                 "psf must be 2 dimensional"
            assert psf.shape[0] == psf.shape[1],  "psf must be square"
            
            if np.iscomplexobj(psf):
                print "warning, casting complex psf to float!"
                psf = psf.astype(np.float32)

            psff = np.fft.fft2(psf)
            
            if self.can_has_psff:
                assert psff.shape == (self.psf_N,self.psf_N), "new psf must have same size as old psf"
                self.psff.set(psff.astype(self.psff.dtype),queue=self.queue)
                
            if not self.can_has_psff:
                self.can_has_psff = True
                self.psff = cla.to_device(self.queue,psff.astype(np.complex64))
                self.psf_N = psf.shape[0]
                
            assert ispower2(self.psf_N), "psf size must be a power of 2"
                
        # from psf and pinhole, see if the speckles will be resized. if so, make
        # the resizing plan
        assert self.psf_N <= self.pin_N, "psf size must be <= pinhole size"
        if self.psf_N < self.pin_N:
            self.speckle_N = self.psf_N
            resizeplan   = wrapping.resize_plan((self.pin_N,self.pin_N),(self.psf_N,self.psf_N),target='gpu')
            self.r_x_gpu = cla.to_device(self.queue, resizeplan[1].astype(np.float32))
            self.r_y_gpu = cla.to_device(self.queue, resizeplan[0].astype(np.float32))
            self.resized = cla.empty(self.queue, (self.speckle_N,self.speckle_N), np.complex64)  # buffer to hold resized speckle
            self.resized_f = cla.empty(self.queue, (self.speckle_N,self.speckle_N), np.float32)  # buffer to hold resized speckle
            
            io.save('rxgpu.fits',self.r_x_gpu.get())
            io.save('rygpu.fits',self.r_y_gpu.get())
            
            self.resize_speckles = True
            self.to_blur = self.resized
        else:
            self.resize_speckles = False
            self.speckle_N = self.pin_N
            self.to_blur = self.speckles
            
        self.pN2 = self.pin_N*self.pin_N
        self.sN2 = self.speckle_N*self.speckle_N
        self.speckle_sum = cla.empty(self.queue, (self.speckle_N,self.speckle_N), np.complex64) # this holds the sum of ALL the speckle patterns
        self.blurred     = cla.empty(self.queue, (self.speckle_N,self.speckle_N), np.complex64) # this holds the blurred speckles.
        self.blurred_f   = cla.empty(self.queue, (self.speckle_N,self.speckle_N), np.float32) # this holds the blurred speckles.

        # create the unwrapping plans
        if unwrap != None:
            
            if self.can_has_unwrap:
                print "cannot reset unwrapping plans"
            
            if not self.can_has_unwrap:
                
                assert isinstance(unwrap,(list,tuple,np.ndarray)), "unwrap must be iterable type"
                assert len(unwrap) == 2, "unwrap must be length 2"
                for x in unwrap: assert isinstance(x,(int,np.int32,np.int64)), "elements in wrap must be integer"
                
                self.ur   = min([unwrap[0],unwrap[1]])
                self.uR   = max([unwrap[0],unwrap[1]])
                self.rows = self.uR-self.ur
                
                
                # make the unwrap plan. put the buffers into gpu memory
                uy,ux = wrapping.unwrap_plan(self.ur,self.uR,(0,0),modulo=self.speckle_N)[:,:-1]
                self.unwrap_cols = len(ux)/self.rows
                self.unwrap_x_gpu  = cla.to_device(self.queue,ux.astype(np.float32))
                self.unwrap_y_gpu  = cla.to_device(self.queue,uy.astype(np.float32))
                self.unwrapped_gpu = cla.empty(self.queue,(self.rows,self.unwrap_cols), np.float32)
                
                # make resizing plans and buffers for row correlations
                r512_x, r512_y  = self._resize_uw_plan(self.rows,self.unwrap_cols,512)
                r360_x, r360_y  = self._resize_uw_plan(self.rows,512,360)
                self.r512_x_gpu = cla.to_device(self.queue, r512_x.astype(np.float32))  # plan
                self.r512_y_gpu = cla.to_device(self.queue, r512_y.astype(np.float32))  # plan
                self.r360_x_gpu = cla.to_device(self.queue, r360_x.astype(np.float32))  # plan
                self.r360_y_gpu = cla.to_device(self.queue, r360_y.astype(np.float32))  # plan
                self.resized512 = cla.empty(self.queue,r512_x.shape,      np.float32)   # result, also stores correlations
                self.resized360 = cla.empty(self.queue,r360_x.shape,      np.float32)   # holds data after resizing to 360 pixels in theta
                self.despiked1  = cla.empty(self.queue,r360_x.shape,      np.float32)   # holds data after resizing to 360 pixels in theta
                self.despiked2  = cla.empty(self.queue,r360_x.shape,      np.float32)
                self.rowaverage = cla.empty(self.queue,(r512_x.shape[0],),np.float32) # correlation normalizations
                self.correl_sum = cla.empty(self.queue,r360_x.shape,      np.float32)
                self.correl_sum_ds = cla.empty(self.queue,r360_x.shape,   np.float32)
                
                # make fft plan for correlating rows.
                
                self.resized512z     = cla.empty(self.queue,r512_x.shape,np.float32) # an array of zeros for the xaxis-only split fft
        
                # zero out some buffers
                self.set_zero_f(self.resized512z)
                self.set_zero_f(self.correl_sum)
                self.set_zero_f(self.correl_sum_ds)
                
                self.can_has_unwrap = True
        
        # we perform the following ffts:
        # 1. transform illuminated into farfield (pin_N, complex)
        # 2. blur the speckles (speckle_N,complex)
        # 3. row correlations (512,complex)
        self.fftplan_speckle = fft_plan((self.pin_N,self.pin_N), queue=self.queue) # (pin_N, complex)
        self.fftplan_blur_f2 = fft_plan((self.speckle_N,self.speckle_N),queue=self.queue)
        self.fftplan_correls = fft_plan((512,), dtype=np.float32,queue=self.queue)
        
        # create cosines for the decomposition.
        if components != None:

            if not self.can_has_cosines:
                assert isinstance(components,(tuple,list,np.ndarray)), "components must be iterable"
                for component in components: assert isinstance(component,int), "each component must be int"
    
                self.nc = len(components)
                cosines = np.zeros((self.nc,360),float)
                angles  = np.arange(360)*2*np.pi/360.
                for n, c in enumerate(components): cosines[n] = np.cos(angles*c)
                self.cosines = cosines
            
                # put cosines on gpu
                # initialize the spectrum array
                self.cosines_gpu      = cla.to_device(self.queue,self.cosines.astype(np.float32))
                self.spectrum_gpu     = cla.empty(self.queue,(self.rows,self.nc), np.float32) # for correlating with spikes
                self.spectrum_gpu_sum = cla.empty(self.queue,(self.rows,self.nc), np.float32) # for correlating with spikes
                self.spectrum_gpu_ds1 = cla.empty(self.queue,(self.rows,self.nc), np.float32) # for correlating without spikes
                self.spectrum_gpu_ds2 = cla.empty(self.queue,(self.rows,self.nc), np.float32) # for correlating without spikes (second method)
                self.set_zero_f(self.spectrum_gpu_sum)
                self.can_has_cosines = True
            
        if returnables != None:
            
            # check types
            assert isinstance(returnables,(list,tuple)), "returnables must be list or tuple"
            assert all([isinstance(r,str) for r in returnables]), "all elements of returnables must be strings"
        
            self.returnables_list = []
            for r in returnables:
                if r not in self.keywords:
                    print "requested returnable %s is unrecognized and will be ignored"%r
                else:
                    self.returnables_list.append(r)
            print self.returnables_list
            
        if self.can_has_object and self.can_has_pinhole and self.can_has_psff and self.can_has_unwrap and self.can_has_cosines:
            self.can_run_scan = True

    def _make_kernels(self):
        
        """ This is a function just for organizational reasons. Logically,
        make_kernels is entirely part of __init__()."""
        
        #assert self.can_has_unwrap, "need to set unwrap before building kernels to test x-axis support in pyfft"
        
        # build basic mathematical kernels
        #self.mult_f_f = gpu.build_kernel_file(self.context, self.device, kp+'basic_multiply_f2_f2.cl') # complex*complex
        #self.mult_f_f2 = gpu.build_kernel_file(self.context, self.device, kp+'basic_multiply_f2_f2.cl') # complex*complex
        #self.mult_f2_f = gpu.build_kernel_file(self.context, self.device, kp+'basic_multiply_f2_f2.cl') # complex*complex
        self.mult_f2_f2 = gpu.build_kernel_file(self.context, self.device, kp+'basic_multiply_f2_f2.cl') # complex*complex
        
        #self.div_f_f    = gpu.build_kernel_file(self.context, self.device, kp+'basic_divide_f_f.cl')     # float/float
        #self.div_f_f2   = gpu.build_kernel_file(self.context, self.device, kp+'basic_divide_f_f2.cl')    # float/complex
        #self.div_f2_f   = gpu.build_kernel_file(self.context, self.device, kp+'basic_divide_f2_f.cl')    # complex/float
        #self.div_f2_f2  = gpu.build_kernel_file(self.context, self.device, kp+'basic_divide_f2_f2.cl')   # complex/complex
        
        self.abs_f2_f     = gpu.build_kernel_file(self.context, self.device, kp+'basic_abs_f2_f.cl')       # abs cast to float
        self.abs_f2_f2    = gpu.build_kernel_file(self.context, self.device, kp+'basic_abs_f2_f2.cl')      # abs kept as cmplx
        self.abs_split_f  = gpu.build_kernel_file(self.context, self.device, kp+'basic_abs_split_f.cl')    # abs kept as cmplx
        self.abs_split_f2 = gpu.build_kernel_file(self.context, self.device, kp+'basic_abs_split_f2.cl')
        
        self.add_f_f = gpu.build_kernel_file(self.context, self.device, kp+'basic_add_f_f.cl')
        #self.add_f_f2 = gpu.build_kernel_file(self.context, self.device, kp+'basic_add_f_f.cl')
        #self.add_f2_f = gpu.build_kernel_file(self.context, self.device, kp+'basic_add_f_f.cl')
        #self.add_f2_f2 = gpu.build_kernel_file(self.context, self.device, kp+'basic_add_f_f.cl')
        
        self.complex_sqrt = gpu.build_kernel_file(self.context, self.device, kp+'basic_sqrt_f2.cl')      # square of complex number
        
        # recast float to complex and vice versa
        self.copy_f_f   = gpu.build_kernel_file(self.context, self.device, kp+'copy_f_f.cl') # cast complex to float
        self.copy_f_f2  = gpu.build_kernel_file(self.context, self.device, kp+'copy_f_f2.cl') # cast float to complex
        self.copy_f2_f  = gpu.build_kernel_file(self.context, self.device, kp+'copy_f2_f.cl')
        self.copy_f2_f2 = gpu.build_kernel_file(self.context, self.device, kp+'copy_f2_f2.cl')
        
        # Build OpenCL programs from external kernel files.
        self.slice_view    = gpu.build_kernel_file(self.context, self.device, kp+'slice_view.cl')        # for extracting a small array from a larger one
        self.map_coords_f  = gpu.build_kernel_file(self.context, self.device, kp+'map_coords_f.cl') # for unwrapping and resizing float array
        self.average_rows  = gpu.build_kernel_file(self.context, self.device, kp+'correl_denoms.cl')     # for getting correlation normalizations
        self.corr_norm     = gpu.build_kernel_file(self.context, self.device, kp+'correl_norm.cl')       # for doing correlation normalization
        self.cosine_reduce = gpu.build_kernel_file(self.context, self.device, kp+'cosine_reduce.cl')     # for doing cosine spectrum reduction
        self.despike       = gpu.build_kernel_file(self.context, self.device, kp+'despike.cl')           # cubic interpolation to remove autocorrelation spike
        self.despike2      = gpu.build_kernel_file(self.context, self.device, kp+'despike2.cl')
    
        self.set_zero_f = EK(self.context,
            "float *a",
            "a[i] = 0.0f",
            "setzero")
        
        self.set_zero_f2 = EK(self.context,
            "float2 *a",
            "a[i] = (float2)(0.0f,0.0f)",
            "setzerof2")

    def _cl_add(self,in1,in2,out):
        d1, d2, d3 = in1.dtype, in2.dtype, out.dtype
        assert d1 in self.array_dtypes, "in1 dtype not recognized; is %s"%d1
        assert d2 in self.array_dtypes, "in2 dtype not recognized; is %s"%d2
        assert d3 in self.array_dtypes, "out dtype not recognized; is %s"%d3
        assert in1.shape == in2.shape and in1.shape == out.shape, "all arrays must have same shape"
        N = in1.size
        
        if d1 == 'float32':
            if d2 == 'float32':
                func = self.add_f_f
                assert d3 == 'float32', "float + float = float"
                arg1 = in1
                arg2 = in2
                
            if d2 == 'complex64':
                func = self.add_f_f2
                assert d3 == 'complex64', "float + complex = complex"
                arg1 = in1
                arg2 = in2
                
        if d2 == 'complex64':
            if d2 == 'float32':
                func = self.add_f_f2
                assert d3 == 'complex64', "float + complex = complex"
                arg1 = in2
                arg2 = in1
                
            if d2 == 'complex64':
                func = self.mult_f2_f2
                assert d3 == 'complex64', "complex + complex = complex"
                arg1 = in1
                arg2 = in2
                
        func.execute(self.queue,(N,), arg1.data, arg2.data, out.data)
        
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
                func = self.mult_f_f2
                assert d3 == 'complex64', "float * complex = complex"
                arg1 = in2
                arg2 = in1
                
            if d2 == 'complex64':
                func = self.mult_f2_f2
                assert d3 == 'complex64', "complex * complex = complex"
                arg1 = in1
                arg2 = in2
                
        func.execute(self.queue,(N,), arg1.data, arg2.data, out.data)
        
    def _cl_abs(self,in1=None,in2=None,out=None):
        """ Wrapper func to the various abs kernels. Checks types of in1 and out
        to select appropriate kernel. """
        
        # allowed cases
        # 1. in1, out
        # 2. in1, in2, out
        
        assert in1 != None
        assert out != None

        d1, d2 = in1.dtype, out.dtype

        assert d1 in self.array_dtypes, "in1 dtype not recognized"
        assert d2 in self.array_dtypes, "out dtype not recognized"
        assert in1.shape == out.shape,  "in1 and out must have same shape"
        
        if in2 != None:
            d3 == in2.dtype
            assert d3 in self.array_dtypes
            assert in2.shape == in1.shape

        N = in1.size
    
        if in2 == None:
            assert d1 == 'complex64', "no abs func for in1 dtype float"
            if d2 == 'float32':   func = self.abs_f2_f
            if d2 == 'complex64': func = self.abs_f2_f2
            func.execute(self.queue,(N,),in1.data,out.data)
            
        if in2 != None:
            assert d1 == 'float32', "split data must be float"
            assert d3 == 'float32', "split data must be float"
            if d2 == 'float32':   func = self.abs_split_f
            if d2 == 'complex64': func = self.abs_split_f2
            func.execute(self.queue,(N,),in1.data,in2.data,out.data)

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
        
    def _cl_map2d(self,in1,out,x_plan,y_plan):
        """ Wrapper for the various places in which map_coords_f gets called."""
        
        # check types
        s = out.shape
        assert in1.dtype == 'float32'
        assert out.dtype == 'float32'
        assert x_plan.dtype == 'float32'
        assert y_plan.dtype == 'float32'
        assert s == x_plan.shape
        assert s == y_plan.shape
        
        r_in = np.int32(in1.shape[0])
        c_in = np.int32(in1.shape[1])
        r_out = s[0]
        c_out = s[1]
        
        self.map_coords_f.execute(self.queue,(c_out,r_out),in1.data,c_in,r_in,out.data,x_plan.data,y_plan.data,np.int32(self.interp_order)).wait()
        
    def _cl_copy(self,in1,out):
        """ Copies in1 to out with correct casting function. If in1 is complex
        and out is float, retain only the real component. """
        
        d1 = in1.dtype
        d2 = out.dtype
        
        assert in1.shape == out.shape
        assert d1 in ('float32','complex64')
        assert d2 in ('float32','complex64')
        N = d1.size
        
        if d1 == 'complex64':
            if d2 == 'complex64':
                func = self.copy_f2_f2
            if d2 == 'float32':
                func = self.copy_f2_f
        if d1 == 'float32':
            if d2 == 'complex64':
                func = self.copy_f2_f
            if d2 == 'float32':
                func = self.copy_f_f
                
        func.execute(self.queue,(N,),in1.data,out.data)

    def _adjust_rows(self):
        """ Change the value of unwrapped_R or unwrapped_r in the case of a
        GeForce + Mac when (unwrapped_R-unwrapped_r) is a power of 2. For
        unknown reasons (most likely Apple's OpenCL compiler), this combination
        leads to NaNs. This definitely happens on OSX 10.6.8 + GF9400M; other
        combinations are unknown."""
        
        print "Warning! Numbers of unwrapped rows that are powers of 2 on NVidia+Apple can lead to NaN."
        
        if self.uR < self.pin_N/2:
            self.uR += -1
            self.rows += -1
            print "subtracted 1 from unwrapped_R to avoid the issue"
            
        if self.uR >= self.pin_N/2:
                self.ur += 1
                self.rows += -1
                print "added 1 to unwrapped_r to avoid the issue"
                
        if self.uR >= self.pin_N/2 and self.ur == 0:
            print "check output spectra for NaN. if present adjust unwrapped_r or unwrapped_R"

    def _slice_object(self,top_row,left_col):
        """ Copy an object from self.master_object into self.active_active. The
        size of the sliced array is the same as that of the object specified
        when instantiating the class. If the master_object and active_object are
        the same size, this function basically becomes np.roll.
        
        Inputs are:
            top_row: y coordinate for slice 
            left_col: x coordinate for slice
            site (optional): 
            
        returnables:
            object: the master (unsliced) object 
            sliced: the sliced (active) object
        """
        
        self.slice_time0 = time.time()

        self.slice_view.execute(self.queue, (self.pin_N,self.pin_N), # opencl stuff. array size is size of sliced array
                   self.master_object.data,                          # input: master_object
                   self.active_object.data,                          # output: active_object
                   np.int32(self.obj_N), np.int32(self.pin_N),       # array dimensions: N_in, N_out
                   np.int32(top_row), np.int32(left_col)).wait()     # coords
        self.slice_time1 = time.time()

        if 'object' in self.returnables_list: self.returnables['object'] = self.master_object.get()
        if 'sliced' in self.returnables_list: self.returnables['sliced'] = self.active_object.get()

    def _make_speckle(self):
        """Turn the current occupant of self.active_object into a fully speckle pattern.
        This method accepts no input besides the optional site which is used only for saving symmetry microscope
        output at particular illumination locations. The speckle pattern is stored in self.speckles_re.
        
        Available returnables:
            illuminated: the product of self.active_object and self.pinhole.
            speckle: the fully coherent speckle pattern.
            speckle_blocker: the speckle pattern with a blocker of radius self.ur in the center.
        """

        make_time0 = time.time()
        
        # calculate the exit wave by multiplying the illumination
        self._cl_mult(self.active_object ,self.pinhole, self.illuminated)

        if 'illuminated' in self.returnables_list: self.returnables['illuminated'] = self.illuminated.get()
        
        # from the exit wave, make the speckle by abs(fft)**2
        self.fftplan_speckle.execute(data_in=self.illuminated.data, data_out=self.far_field.data,wait_for_finish=True)
        self._cl_abs(in1=self.far_field,out=self.speckles)
        self._cl_mult(self.speckles,self.speckles,self.speckles)
        
        make_time1 = time.time()
        
        # resize the speckles if needed
        if self.resize_speckles:
            # copy re to float, map, copy re to complex
            self._cl_copy(self.speckles,self.speckles_f)
            self._cl_map2d(self.speckles_f,self.resized_f,self.r_x_gpu,self.r_y_gpu)
            self._cl_copy(self.resized_f,self.resized)
            
        make_time2 = time.time()
        
        self.speckle_time += make_time1-make_time0
        self.resize_time1 += make_time2-make_time1
            
        # include the blocker
        if 'speckle' in self.returnables_list:
            self.returnables['speckle'] = fftshift(self.speckles_f.get())
        if 'rspeckle' in self.returnables_list and self.resize_speckles:
            self.returnables['rspeckle'] = fftshift(self.resized_f.get())
        if 'speckle_blocker' in self.returnables_list:
            self.returnables['speckle_blocker'] = fftshift(self.speckles_re.get())*self.blocker

    def _blur(self):
        """ Turn the fully-coherent speckle pattern in self.intensity_sum_re into a partially coherent
        speckle pattern through convolution with a mutual intensity function. This method accepts no input.
        The blurred speckle is saved in self.intensity_sum_re.

        Available returnables:
            blurred: blurred speckle
            blurred_blocker: blurred speckle with a blocker in the center
        """

        blur_time0 = time.time()

        # fft, multiply, ifft
        self.fftplan_blur_f2.execute(data_in=self.to_blur.data, data_out=self.blurred.data, wait_for_finish=True)
        self._cl_mult(self.blurred,self.psff,self.blurred)
        self.fftplan_blur_f2.execute(self.blurred.data, wait_for_finish=True, inverse=True)
        self._cl_copy(self.blurred,self.blurred_f) # keep only the real component
        blur_time1 = time.time()
        
        self.blur_time += blur_time1-blur_time0

        if 'blurred' in self.returnables_list: self.returnables['blurred'] = fftshift(abs(self.blurred.get()))
        if 'blurred_blocker' in self.returnables_list: self.returnables['blurred_blocker'] = fftshift(abs(self.speckles_re.get()))*self.blocker
    
    def _unwrap_speckle(self):
        """Remap the speckle into polar coordinate, and then resize it to 512 pixels along the x-axis.
        This method accepts no input.
        
        Available returnables:
            unwrapped: unwrapped speckle
            resized: unwrapped speckle resized to 512 pixels wide"""

        # unwrap, then resize to 512 columns. this can probably be made into a single operation.
        unwrap_time0 = time.time()
        self._cl_map2d(self.blurred_f,self.unwrapped_gpu,self.unwrap_x_gpu,self.unwrap_y_gpu)
        unwrap_time1 = time.time()
        self._cl_map2d(self.unwrapped_gpu,self.resized512,self.r512_x_gpu,self.r_512_y_gpu)        
        unwrap_time2 = time.time()
        
        self.unwrap_time += unwrap_time1-unwrap_time0
        self.resize_time2 += unwrap_time2-unwrap_time1
        
        if 'unwrapped' in self.returnables_list: self.returnables['unwrapped'] = self.unwrapped_gpu.get()
        if 'resized'   in self.returnables_list: self.returnables['resized']   = self.resized512.get()

    def _rotational_correlation(self,fallback='cpu'):
        """ Perform an angular correlation of a speckle pattern by doing a
        cartesian correlation of a speckle pattern unwrapped into polar
        coordinates. This method accepts no input. Data must be in
        self.resized512. Output is stored in place. After correlating the data
        is resized to 360 pixels in the angular axis.
        
        Available returnables:
            correlated: the angular ac after normalization, resized to 360
            pixels in the angular coordinate.
        """
        
        corr_time0 = time.time()
        
        # zero the buffers
        self.set_zero_f(self.resized512z)
        self.set_zero_f(self.rowaverage)
            
        # get the denominaters for the wochner normalization
        if self.norm_mode == 0:
            self.average_rows.execute(self.queue, (self.rows,), self.resized512.data, self.rowaverage.data, np.int32(512),np.int32(0))
            
        # calculate the autocorrelation of the rows using a 1d plan and the batch command in pyfft
        self.fftplan_correls.execute(self.resized512.data,self.resized512z.data,wait_for_finish=True,batch=self.rows)
        self._cl_abs(in1=self.resized_512,in2=self.resized512_z,out=self.resized_512)
        self.set_zero_f(self.resized512z)
        self.fftplan_correls.execute(self.resized512.data,self.resized512z.data,batch=self.rows,inverse=True,wait_for_finish=True)

        if self.norm_mode == 1:
            self.average_rows.execute(self.queue, (self.rows,), self.resized512.data, self.rowaverage.data, np.int32(512), np.int32(self.norm_mode))
        
        # normalize the autocorrelations according to self.norm_mode
        # if self.norm_mode = 0, use the wochner convention
        # if self.norm_mode = 1, divide each row by the value at column 1
        self.corr_norm.execute(self.queue, (512,self.rows), self.resized512.data, self.rowaverage.data, self.resized512.data, np.int32(self.norm_mode))
        
        corr_time1 = time.time()
        
        # resize the normalized correlation to 360 pixels in theta, so each pixel is a one degree step
        self._cl_map_2d(self.resized512,self.resized360,self.r360_x_gpu,self.r360_y_gpu)
        self._cl_add(self.correl_sum, self.resized360,self.correl_sum)
        corr_time2 = time.time()

        if 'correlated' in self.returnables_list: self.returnables['correlated'] = self.resized360.get()
        
        self.correl_time += corr_time1-corr_time0
        self.resize_time3 += corr_time2-corr_time1
  
    def _decompose_spectrum(self):
        """ Decompose the angular correlation in self.resized360 into a cosine spectrum.
        This method accepts no input. Because the amount of data in the self.spectrum_gpu
        object is very small, moving the spectrum back into host memory is a cheap transfer.
        
        Available returnables:
            spectrum (the default)
        """

        assert self.can_has_cosines, "cant decompose until cosines are set!"

        decomp_time0 = time.time()
        
        self.set_zero_f(self.spectrum_gpu)            # zero the buffer self.spectrum_gpu
        self.cosine_reduce.execute(                 # run the cosine reduction kernel
            self.queue,(int(self.nc),int(self.rows)),
            self.resized360.data,
            self.cosines_gpu.data,
            self.spectrum_gpu.data,
            np.int32(self.nc))
        decomp_time1 = time.time()
        
        # now despike the correlation and redo the decomposition. this will test how much the
        # spikes really matter in the concentration metric
        self._cl_copy(self.resized360,self.despiked1)
        self.despike.execute(self.queue,(self.rows,), self.despiked1.data, self.despiked1.data, np.int32(4), np.float32(4))
        self._cl_add(self.correl_sum_ds,self.despiked1,self.correl_sum_ds)
        self.set_zero_f(self.spectrum_gpu_ds1)       # zero the buffer self.spectrum_gpu_ds
        self.cosine_reduce.execute(                 # run the cosine reduction kernel
            self.queue,(self.nc,self.rows),
            self.despiked1.data,
            self.cosines_gpu.data,
            self.spectrum_gpu_ds1.data,
            np.int32(self.nc))
        decomp_time2 = time.time()
        
        self.decomp_time  += decomp_time1-decomp_time0
        self.despike_time += decomp_time2-decomp_time1
        
        self._cl_add(self.spectrum_gpu_sum,self_spectrum_gpu_ds1,self.spectrum_gpu_sum)

        if 'spectrum' in self.returnables_list:
            self.returnables['spectrum']     = self.spectrum_gpu.get()
        if 'spectrum_ds' in self.returnables_list:
            self.returnables['spectrum_ds'] = self.spectrum_gpu_ds1.get()
        if 'correlated_ds' in self.returnables_list:
            self.returnables['correlated_ds'] = self.despiked1.get()

    def _resize_uw_plan(self,rows,columns_in,columns_out):
    
        # uR-ur gives the number of rows
        # columns_in gives the number of columns in the image being resize
        # columns_out gives the number of columns in the image after resizing
    
        cols1 = np.arange(columns_out).astype(np.float32)*columns_in/float(columns_out)
        rows1 = np.arange(rows).astype(np.float32)
        return np.outer(np.ones(rows,int),cols1), np.outer(rows1,np.ones(columns_out))

    def _resize_speckle_plan(self,n1,n2):
        rows, cols = np.indices((n2,n2),float)*float(n1)/float(n2)
        return rows, cols

    def run_on_site(self,y,x):
        
        """ Run the symmetry microscope on the site of the object described by the roll coordinates
        dy, dx. Steps are:
        
        1. Roll the illumination
        2. Make the speckle
            2b: If coherence is specified, blur the speckle
        3. Unwrap the speckle
        4. Autocorrelate the unwrapped speckle
        5. Cosine-decompose the autocorrelation
        
        This function aggregates the more atomistic methods into a single function.
        
        arguments:
            dy, dx: roll coordinates used as follows: np.roll(np.roll(d,r0,axis=0),r1,axis=1)
            components: (optional) decompose into these components
            cosines: (optional) precomputed cosines for speed
        """
        
        assert self.can_has_object, "no object set"
        assert self.can_has_pinhole, "no pinhole set"
        assert self.can_has_unwrap, "no unwrap set"
        assert self.can_has_cosines, "no cosines set"
        assert isinstance(y,int) and isinstance(x,int), "site coordinates must be integer"
        
        self._slice_object(y,x)
        self._make_speckle()
        self._blur()
    
        # in the cpu microscope, these are handled by symmetries.rot_sym
        self._unwrap_speckle()
        self._rotational_correlation()
        self._decompose_spectrum()

    def print_timings(self):
        total_time = self.speckle_time+self.resize_time1+self.blur_time+self.unwrap_time+self.resize_time2+self.decomp_time+self.despike_time+self.correl_time+self.resize_time3+self.decomp_time+self.despike_time
        print "speckle time: %.3e (%.2f %%)"%(self.speckle_time,self.speckle_time/total_time*100)
        print "resize1 time: %.3e (%.2f %%)"%(self.resize_time1,self.resize_time1/total_time*100)
        print "blur time:    %.3e (%.2f %%)"%(self.blur_time,self.blur_time/total_time*100)
        print "unwrap time:  %.3e (%.2f %%)"%(self.unwrap_time,self.unwrap_time/total_time*100)
        print "resize2 time: %.3e (%.2f %%)"%(self.resize_time2,self.resize_time2/total_time*100)
        print "correl time:  %.3e (%.2f %%)"%(self.correl_time,self.correl_time/total_time*100)
        print "resize3 time: %.3e (%.2f %%)"%(self.resize_time3,self.resize_time3/total_time*100)
        print "decomp time:  %.3e (%.2f %%)"%(self.decomp_time,self.decomp_time/total_time*100)
        print "despike time: %.3e (%.2f %%)"%(self.despike_time,self.despike_time/total_time*100)


def ispower2(N):
    return (N&(N-1)) == 0

# old code no longer being used for dichroic magnetism
"""
        self.multiply_buffers = EK(self.context,
            "float2 *a, " # illumination
            "float2 *b, " # transmission
            "float2 *c",  # output is complex type so float2
            "c[i] = (float2)(a[i].x*b[i].x, a[i].x*b[i].y+a[i].y*b[i].x)",
            "mult_buffs")
            
        self.subtract_charge = EK(self.context,
            "float *charge, "   # charge
            "float *total, "    # charge+magnetic
            "float *magnetic, " # output
            "float ratio",      # ratio
            "magnetic[i] = total[i]-charge[i]*ratio",
            "sub_charge")

        # calculate the charge signal and load it into memory. assume this is just the fourier intensity of the illumination function.
        ChargeSignal       = abs(DFT(illumination))**2
        self.charge_signal = cla.to_device(self.queue,ChargeSignal.astype(np.float32))
        self.charge_max    = cla.max(self.charge_signal).get()

        if magnetic:
            
            contrast = 2
            n0 = np.complex64(complex(0.998482,0.0019125))
            dn = contrast*np.complex64(complex(-3.03667e-05,3.825e-05)) # magnetic modulation to index of refraction
            wavelength = 1240e-9/780.
            scale = 2*scipy.pi*600e-9/wavelength # just a scaling factor for the transmission function

            for polarization in [-1.,1.]: # each polarization give a separate speckle pattern
                # calculate the transmission through the domains
                self.transmit.execute(self.queue,(self.N**2,),                                                                      # opencl stuff
                                      np.float32(scale), np.float32(polarization), np.complex64(n0), np.complex64(dn),  # constants
                                      self.active_object.data, self.scratch1.data).wait()                                          # in/out buffers
                if 'transmission' in self.returnables_list and self.ipath != None:
                    Temp = self.scratch1.get()
                    io2.save('%s/%s_%s transmission polarization_%s.fits'%(self.ipath,self.int_name,polarization),Temp,components='real')
    
                # calculate the exit wave by multiplying the illumination
                self.multiply_buffers(self.illumination_gpu, self.scratch1, self.scratch1).wait()
                if 'illuminated' in self.returnables_list and self.ipath != None:
                    Temp = self.scratch1.get()
                    io2.save('%s/%s_%s illuminated polarization_%s.fits'%(self.ipath,self.int_name,site,polarization),Temp)
                
                # calculate the fft of the wavefield
                self.fftplan_speckle.execute(self.scratch1.data, data_out=self.scratch2.data,wait_for_finish=True)
                if 'fft' in self.returnables_list and self.ipath != None:
                    Temp = self.scratch2.get()
                    io2.save('%s/%s_%s fft polarization_%s.fits'%(self.ipath,self.int_name,site,polarization),Temp)
                
                # calculate the intensity of the fft
                self.add_intensity(self.scratch2, self.intensity_sum_re).wait()
                if 'intensity' in self.returnables_list and self.ipath != None:
                    Temp = self.intensity_sum_re.get()
                    io2.save('%s/%s_%s intensity_sum polarization_%s.fits'%(self.ipath,self.int_name,site,polarization),Temp)
                
            # subtract the charge signal from the sum. keep the result in self.intensity_sum
            signal_max = cla.max(self.intensity_sum_re).get()
            ratio = signal_max/self.charge_max
            self.subtract_charge(self.charge_signal,self.intensity_sum_re,self.intensity_sum_re,np.float32(ratio))
            if 'isolated' in self.returnables_list and self.ipath != None:
                Temp = fftshift(self.intensity_sum_re.get())
                io2.save('%s/%s_%s isolated %s.fits'%(self.ipath,self.int_name,site,contrast),Temp,components='real')"""

