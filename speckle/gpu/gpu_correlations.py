# core
import numpy as np
import pyopencl as cl
import pyopencl.array as cla
from pyopencl.elementwise import ElementwiseKernel as EK
from pyfft.cl import Plan as fft_plan
import time
import string

# common libs. do some ugly stuff to get the path set to the kernels directory
from .. import wrapping,shape
import gpu
kp = string.join(gpu.__file__.split('/')[:-1],'/')+'/kernels/'

# cpu fft
DFT = np.fft.fft2
IDFT = np.fft.ifft2
fftshift = np.fft.fftshift

# import matplotlib
# matplotlib.use('Agg')
# import pylab

class gpu_microscope():
    """Methods for running a symmetry-microscope simulation on a gpu using
    pyopencl and pyfft libraries.
    """

    def __init__(self,device=None,object=None,unwrap=None,pinhole=None,components=None,coherence=None,ph_signal=False,returnables=('spectrum',)):

        """ Get the class running. At minimum, accept gpu information and
        compile kernels. Information about the object, pinhole, unwrap region
        coherence properties of the illumination, and cosine components for the
        decomposition may also be supplied to __init__ or later on to the
        methods called within __init__."""

        assert isinstance(device,tuple) and len(device) == 4, "device information improperly formed"

        self.context, self.device, self.queue, self.platform = device
        self.make_kernels()
        self.returnables_list = returnables
        self.interpolation_order = 1
        self.ph_signal = ph_signal
        
        self.can_has_object = False
        self.can_has_unwrap = False
        self.can_has_pinhole = False
        self.can_has_coherence = False
        self.can_has_cosines = False
        self.cosines = None
        
        if object != None: self.set_object(object)
        if unwrap != None: self.set_unwrap(unwrap)
        if pinhole != None: self.set_pinhole(pinhole)
        if components != None: self.set_cosines(components)
        if coherence != None: self.set_coherence(coherence)
        
        self.returnables = {}
    
    def set_object(self,object):
        
        """ Set the object on which the symmetry microscope is run. Can be
        changed without restarting the class but the new object size has to be
        the old object size.
        
        arguments:
            object: an integer or numpy array. if an integer, memory is
            allocated on the gpu for an upcoming NxN float array if an array,
            the array is put onto the gpu.

            The reason to accept a number is just to allow instantiation of all
            the buffers before an actual object is passed in.
        """
        
        assert isinstance(object,(int,np.ndarray)), "object spec must be size or array"
        
        if not self.can_has_object:
            if isinstance(object,np.ndarray):
                assert object.ndim == 2, "object must be 2d"
                assert object.shape[0] == object.shape[1], "object must be square"
                self.N = len(object)
                self.master_object = cla.to_device(self.queue,object.astype(np.float32))
                #self.set_zero(self.master_object)
            if isinstance(object,int):
                self.N = object
                self.master_object = cla.empty(self.queue,(self.N,self.N),np.float32)
                
            # allocate memory for intermediate results in the speckle calculation. these all need to be NxN clas 
            self.active_object    = cla.empty(self.queue, (self.N,self.N), np.float32  )
            self.scratch1         = cla.empty(self.queue, (self.N,self.N), np.complex64)
            self.scratch2         = cla.empty(self.queue, (self.N,self.N), np.complex64)
            self.fft_signal       = cla.empty(self.queue, (self.N,self.N), np.complex64)
            self.intensity_sum_re = cla.empty(self.queue, (self.N,self.N), np.float32  )
            self.intensity_sum_im = cla.empty(self.queue, (self.N,self.N), np.float32  )
            if not self.ph_signal:
                self.scratch1re   = cla.empty(self.queue, (self.N,self.N), np.float32)
            
            # make fft plan for complex (interleaved); for speckle
            self.fftplan_speckle = fft_plan((self.N,self.N), queue=self.queue)
            
        if self.can_has_object:
            # this means an object has already been set and now we're just updating it
            assert isinstance(object,np.ndarray) and object.ndim==2, "object must be 2d array"
            assert object.shape == (self.N,self.N), "object shape differs from initialization values"
            self.master_object.set(object.astype(self.master_object.dtype))

        self.can_has_object = True
        
    def set_unwrap(self,params):
        """ Build and name the unwrap plan. Cannot be changed without restarting
        the class.
        
        arguments:
            params: (unwrap_r, unwrap_R). no support or need to provide support
            for user-supplied center."""
        
        if self.can_has_unwrap:
            print "no current support for updating unwrap during simulation"
            exit()
        
        assert self.can_has_object, "need to init object before unwrap"
        assert isinstance(params,(list,tuple,np.ndarray)), "unwrap params must be iterable"
        assert len(params) == 2, "must provide exactly two parameters: unwrap_r and unwrap_R"
        
        ur, uR = params[0],params[1]
        assert isinstance(ur,(float,int)) and isinstance(uR,(float,int)), "ur and uR must be numbers"
        self.ur, self.uR = min([ur,uR]), max([ur,uR])
        self.rows = self.uR-self.ur
        
        assert self.ur > 0,       "unwrap_r must be > 0"
        assert self.uR > self.ur, "unwrap_R must be > unwrap_r"
        assert uR < self.N/2,     "unwrap_R exceeds simulated speckle bounds"
        
        # make the unwrap plan. put the buffers into gpu memory
        uy,ux = wrapping.unwrap_plan(self.ur,self.uR,(0,0),modulo=self.N)[:,:-1]
        self.unwrap_cols = len(ux)/self.rows
        self.unwrap_x_gpu  = cla.to_device(self.queue,ux.astype(np.float32))
        self.unwrap_y_gpu  = cla.to_device(self.queue,uy.astype(np.float32))
        self.unwrapped_gpu = cla.empty(self.queue,(self.rows,self.unwrap_cols), np.float32)
        
        # make resizing plans and buffers for row correlations
        r512_x, r512_y   = self._resize_plan(self.rows,self.unwrap_cols,512)
        r360_x, r360_y   = self._resize_plan(self.rows,512,360)
        self.r512_x_gpu  = cla.to_device(self.queue, r512_x.astype(np.float32)) # plan
        self.r512_y_gpu  = cla.to_device(self.queue, r512_y.astype(np.float32)) # plan
        self.r360_x_gpu  = cla.to_device(self.queue, r360_x.astype(np.float32)) # plan
        self.r360_y_gpu  = cla.to_device(self.queue, r360_y.astype(np.float32)) # plan
        self.resized512  = cla.empty(self.queue,r512_x.shape,      np.float32)  # result, also stores correlations
        self.resized360  = cla.empty(self.queue,r360_x.shape,      np.float32)  # holds data after resizing to 360 pixels in theta
        self.corr_denoms = cla.empty(self.queue,(r512_x.shape[0],),np.float32)  # correlation normalizations

        # make fft plan for correlating rows.
        self.fftplan_correls = fft_plan((512,), dtype=np.float32, queue=self.queue)

        # check for the apple/nvidia problem
        if 'Apple' in str(self.platform) and 'GeForce' in str(self.device) and ispower2(self.rows): self.adjust_rows()

        self.can_has_unwrap = True
     
    def set_pinhole(self,pinhole):
        
        """Set the pinhole function. Can be changed without restarting the class
        but pinhole size must be object size.
        
        arguments:
            pinhole: Either a number, in which case a circle is generated as the
            pinhole with the argument as radius, or an array, in which case the
            array is set as the pinhole. This latter option allows for
            complicated illumination shapes to be supplied."""
        
        assert self.can_has_object, "need to init object before pinhole"
        assert isinstance(pinhole,(int,float,np.ndarray)), "pinhole must be number or array"
            
        if isinstance(pinhole, (int,float)):
            assert pinhole < self.N/2, "pinhole radius must be smaller than pinhole array size"
            illumination = fftshift(shape.circle((self.N,self.N),pinhole))
            self.pr = pinhole
            self.pinhole_area = np.sum(illumination)

        if isinstance(pinhole,np.ndarray):
            assert pinhole.shape == self.object.shape, "supplied pinhole must be same size as supplied object"
            illumination = pinhole
            self.pr = None
        
        if not self.can_has_pinhole:
            self.illumination_gpu = cla.to_device(self.queue,illumination.astype(np.complex64))
        else: self.illumination_gpu.set(illumination.astype(self.illumination_gpu.dtype))
        
        self.can_has_pinhole = True
        
    def set_coherence(self,coherence):
        
        """ Set the coherence function. Object must be set first. Can be changed
        without re-instantiating the class.
        
        arguments:
            coherence: a 2-tuple where [0] is the horizontal coherence length
            and [1] is the vertical coherence length. These values must be
            given in PIXELS, so it is the users responsibility to understand
            what the pixel pitch is in their calculations."""
        
        assert self.can_has_object, "must init object before coherence"
        assert isinstance(coherence,(list,tuple)) and len(coherence) == 2, "coherence should be a 2-tuple"
        assert isinstance(coherence[0],(int,float)) and isinstance(coherence[1],(int,float)), "coherence vals should be float or int"
        
        # make gaussian kernel
        cl_x_px, cl_y_px = coherence
        gaussian = fftshift(shape.gaussian((self.N,self.N),(cl_y_px,cl_x_px),normalization=1.0))
        #gaussian = fftshift(_gaussian_kernel(cl_y_px,cl_x_px,self.N))
        
        if not self.can_has_coherence:
            self.blur_kernel = cl.array.to_device(self.queue,gaussian.astype(np.float32))
            self.blurred_image  = cla.empty(self.queue, (self.N,self.N), np.float32)
            self.blurred_image2  = cla.empty(self.queue, (self.N,self.N), np.float32)
        
            # make fft plan for blurring. because the speckle intensity is in a float32 array,
            # the fft plan expects data in split format: two float32 buffers which are re and im components
            self.fftplan_blurred = fft_plan((self.N,self.N), dtype=np.float32, queue=self.queue)
        
        if self.can_has_coherence:
            self.blur_kernel.set(gaussian.astype(self.blur_kernel.dtype))
        
        self.can_has_coherence = True
        
    def make_kernels(self):
        
        """ This is a function just for organizational reasons. Logically,
        make_kernels is entirely part of __init__() but must follow set_unwrap()
        due to pyfft x-axis support issues."""
        
        #assert self.can_has_unwrap, "need to set unwrap before building kernels to test x-axis support in pyfft"
        
        # Build OpenCL programs from external kernel files.
        self.slice_view    = gpu.build_kernel_file(self.context, self.device, kp+'slice_view.cl')    # for extracting a small array from a larger one
        self.map_coords    = gpu.build_kernel_file(self.context, self.device, kp+'map_coords_buffer.cl') # for unwrapping and resizing
        self.sum_rows      = gpu.build_kernel_file(self.context, self.device, kp+'correl_denoms.cl') # for getting correlation normalizations
        self.row_divide    = gpu.build_kernel_file(self.context, self.device, kp+'row_divide.cl')    # for doing correlation normalization
        self.cosine_reduce = gpu.build_kernel_file(self.context, self.device, kp+'cosine_reduce.cl') # for doing cosine spectrum reduction
        self.slice_row     = gpu.build_kernel_file(self.context, self.device, kp+'slice_row_f_f.cl') # slice row to correlate
        self.put_row       = gpu.build_kernel_file(self.context, self.device, kp+'put_row_f_f.cl')   # put correlated row back in buffer
            
        # more kernels. due to their simplicity these are not put into external files
        self.illuminate_re = EK(self.context,
            "float2 *a, " # illumination (possibly complex)
            "float *b, "  # transmission/object
            "float2 *c",  # output (possibly complex)
            "c[i] = (float2)(a[i].x*b[i],a[i].y*b[i])",
            "illluminate_re")
        
        self.set_zero = EK(self.context,
            "float *a",
            "a[i] = 0.0f",
            "setzero")
        
        self.blur_convolve = EK(self.context,
            "float *a,"  # convolvee, float
            "float *b,"  # convolver, float
            "float *c",  # convolved, float
            "c[i] = a[i]*b[i]",
            "complex_mult")
        
        self.add_intensity = EK(self.context,
            "float2 *a, " # the incoming complex wavefield
            "float  *b",  # the existing intensity distribution
            "b[i] = b[i] + pown(hypot(a[i].x,a[i].y),2)",
            "add_int")
    
        self.make_intensity = EK(self.context,
            "float *real_comp,"
            "float *imag_comp,"
            "float *intensity",
            "intensity[i] = real_comp[i]*real_comp[i]+imag_comp[i]*imag_comp[i]",
            "make_intensity")
        
        self.scalar_mult = EK(self.context,
            "float *input,"
            "float a",
            "input[i] = a*input[i]",
            "scalar_mult")
        
        self.square_modulus = EK(self.context,
            "float2 *fft",
            "fft[i] = (float2)(fft[i].x*fft[i].x+fft[i].y*fft[i].y, 0.0f)",
            "square_modulus")
        
        self.copy_real = EK(self.context,
            "float2 *carray,"
            "float *rarray",
            "rarray[i] = carray[i].x",
            "copy_real")
        
        self.subtract_pinhole = EK(self.context,
            "float2 *scratch,"
            "float2* illumination,"
            "float scale",
            "scratch[i] = (float2)(scratch[i].x-scale*illumination[i].x,scratch[i].y-scale*illumination[i].y)",
            "sub_pin")
        
    def set_cosines(self,components):
        
        """ Precompute the cosines for the cosine decomposition of the angular
        autocorrelations. Requires object and unwrap to be set. Can't be changed
        without re-instantiating the class.
        
        arguments:
            components: a list or array of integers describing which symmetry
            components are to be decomposed e.g, [4,6,8,10]."""
        
        assert self.can_has_unwrap, "must set object and unwrap before cosines"
        assert isinstance(components,(tuple,list,np.ndarray)), "components must be iterable"
        for component in components: assert isinstance(component,int), "each component must be int"
        
        if self.can_has_cosines:
            print "currently no support for changing components during simulation"
            exit()

        # make cosine array
        self.nc = len(components)
        cosines = np.zeros((self.nc,360),float)
        angles  = np.arange(360)*2*np.pi/360.
        for n, c in enumerate(components): cosines[n] = np.cos(angles*c)
        
        # put on gpu
        self.cosines = cla.to_device(self.queue,cosines.astype(np.float32))

        # initialize the spectrum array
        self.spectrum_gpu = cla.empty(self.queue,(self.rows,self.nc), np.float32)

        self.can_has_cosines = True
 
    def adjust_rows(self):
        """ Change the value of unwrapped_R or unwrapped_r in the case of a
        GeForce + Mac when (unwrapped_R-unwrapped_r) is a power of 2. For
        unknown reasons (most likely Apple's OpenCL compiler), this combination
        leads to NaNs. This definitely happens on OSX 10.6.8 + GF9400M; other
        combinations are unknown."""
        
        print "Warning! Numbers of unwrapped rows that are powers of 2 on NVidia+Apple can lead to NaN."
        
        if self.uR < self.N/2:
            self.uR += -1
            self.rows += -1
            print "subtracted 1 from unwrapped_R to avoid the issue"
            
        if self.uR >= self.N/2:
                self.ur += 1
                self.rows += -1
                print "added 1 to unwrapped_r to avoid the issue"
                
        if self.uR >= self.N/2 and self.ur == 0:
            print "check output spectra for NaN. if present adjust unwrapped_r or unwrapped_R"

    def set_returnables(self,returnables=('spectrum',)):
        
        """ Set which of the possible intermediate values are returned out of
        the simulation. Results are returned as a dictionary from which
        intermediates can be extracted through returnables['key'] where 'key' is
        the desired intermediate. Set after object to be safe. 
        
        Available returnables:
            object: the master object from which views are sliced. this is the
                object set by set_object()
            sliced: the current sliced view. when the master object is the same
                size as the simulation size, slicing is just rolling
            illuminated: the current slice * the illumination function
            speckle: human-centered speckle of the current illumination
            speckle_blocker: speckle with a blocker of radius ur in the center
            blurred: blurred speckle
            blurred_blocker: blurred speckle with ur-blocker
            unwrapped: unwrapped speckle (or blurred).
            resized: the unwrapped speckle resized to width 512 columns
            correlated: the angular autocorrelation, resized to width 360
            spectrum: the angular ac decomposed into a cosine series.
            
        By default, the only returnable is 'spectrum', the final output.
        
        Advisory note: since the point of running on the GPU is !SPEED!, and
        pulling data off the GPU is slow, use of returnables should be limited
        except for debugging.
        """
        
        available = ('object','sliced','illuminated','speckle','speckle_blocker',
                     'blurred','blurred_blocker','unwrapped','resized','correlated',
                     'spectrum')
        
        # check types
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
            
    def slice_object(self,top_row,left_col):
        """ Copy an object from self.master_object into self.active_active. The size of the sliced
        array is the same as that of the object specified when instantiating the class.
        
        Inputs are:
            top_row: y coordinate for slice 
            left_col: x coordinate for slice
            site (optional): 
            
        returnables:
            object: the master (unsliced) object 
            sliced: the sliced (active) object
        """
        
        self.slice_time0 = time.time()
        self.slice_view.execute(self.queue, (self.N,self.N),      # opencl stuff
                   self.master_object.data,                       # input: master_object
                   self.active_object.data,                       # output: active_object
                   np.int32(self.N),np.int32(self.N),             # array dimensions
                   np.int32(top_row), np.int32(left_col)).wait()  # coords
        self.slice_time1 = time.time()

        if 'object' in self.returnables_list: self.returnables['object'] = self.master_object.get()
        if 'sliced' in self.returnables_list: self.returnables['sliced'] = self.active_object.get()

    def make_speckle(self):
        """Turn the current occupant of self.active_object into a fully speckle pattern.
        This method accepts no input besides the optional site which is used only for saving symmetry microscope
        output at particular illumination locations. The speckle pattern is stored in self.intensity_sum_re.
        
        Available returnables:
            illuminated: the product of self.active_object and self.pinhole.
            speckle: the fully coherent speckle pattern.
            speckle_blocker: the speckle pattern with a blocker of radius self.ur in the center.
        """

        # reset intensity_sum for each iteration by filling with zeros
        self.set_zero(self.intensity_sum_re)
        self.set_zero(self.intensity_sum_im)
    
        self.make_time0 = time.time()
        
        # calculate the exit wave by multiplying the illumination
        self.illuminate_re(self.illumination_gpu, self.active_object, self.scratch1).wait()
        
        if self.pr != None and not self.ph_signal:
            self.copy_real(self.scratch1,self.scratch1re)
            object_sum = cla.sum(self.scratch1re).get()
            self.subtract_pinhole(self.scratch1,self.illumination_gpu,np.float32(object_sum/self.pinhole_area))
        
        if 'illuminated' in self.returnables_list:
            temp = fftshift(self.scratch1.get())
            if isinstance(self.pr, (int,float)): temp = temp[self.N/2-self.pr:self.N/2+self.pr,self.N/2-self.pr:self.N/2+self.pr]
            self.returnables['illuminated'] = temp
        
        # calculate the fft of the wavefield
        self.fftplan_speckle.execute(self.scratch1.data, data_out=self.scratch2.data, wait_for_finish=True)
        
        # turn the complex field (scratch2) into speckles (intensity_sum_re)
        self.add_intensity(self.scratch2, self.intensity_sum_re).wait()
                
        if 'speckle' in self.returnables_list: self.returnables['speckle'] = fftshift(self.intensity_sum_re.get())
        if 'speckle_blocker' in self.returnables_list: self.returnables['speckle_blocker'] = fftshift(self.intensity_sum_re.get())*self.blocker
            
        self.make_time1 = time.time()

    def blur(self):
        """ Turn the fully-coherent speckle pattern in self.intensity_sum_re into a partially coherent
        speckle pattern through convolution with a mutual intensity function. This method accepts no input.
        The blurred speckle is saved in self.intensity_sum_re.

        Available returnables:
            blurred: blurred speckle
            blurred_blocker: blurred speckle with a blocker in the center
        """

        self.blur_time0 = time.time()

        # fft, multiply, ifft
        self.fftplan_blurred.execute(self.intensity_sum_re.data, self.intensity_sum_im.data, wait_for_finish=True)
        self.blur_convolve(self.intensity_sum_re,self.blur_kernel,self.intensity_sum_re)
        self.blur_convolve(self.intensity_sum_im,self.blur_kernel,self.intensity_sum_im)
        self.fftplan_blurred.execute(self.intensity_sum_re.data, self.intensity_sum_im.data,
                                     wait_for_finish=True, inverse=True)
        
        # preserve power
        #p2 = cla.sum(self.blurred_image).get()
        #self.scalar_mult(self.blurred_image,np.float32(p1/p2))
        self.blur_time1 = time.time()

        if 'blurred' in self.returnables_list: self.returnables['blurred'] = fftshift(abs(self.intensity_sum_re.get()))
        if 'blurred_blocker' in self.returnables_list: self.returnables['blurred_blocker'] = fftshift(abs(self.intensity_sum_re.get()))*self.blocker
    
    def unwrap_speckle(self):
        """Remap the speckle into polar coordinate, and then resize it to 512 pixels along the x-axis.
        This method accepts no input.
        
        Available returnables:
            unwrapped: unwrapped speckle
            resized: unwrapped speckle resized to 512 pixels wide"""
            
        # can this be combined into a single operation through a new map? ie, resize the unwrap map, then do a single
        # pass of map_coords using the hybrid plan?
        
        #unwrap
        self.unwrap_time0 = time.time()
        self.map_coords.execute(self.queue, (self.unwrap_cols,self.rows),                                             # opencl stuff
                   self.intensity_sum_re.data, np.int32(self.N),           np.int32(self.N),                          # input
                   self.unwrapped_gpu.data,    np.int32(self.unwrap_cols), np.int32(self.rows),                       # output
                   self.unwrap_x_gpu.data,     self.unwrap_y_gpu.data,     np.int32(self.interpolation_order)).wait() # plan, interpolation order
        self.unwrap_time1 = time.time()
        
        # resize
        self.map_coords.execute(self.queue, (512,self.rows),
                   self.unwrapped_gpu.data, np.int32(self.unwrap_cols), np.int32(self.rows),                        # input
                   self.resized512.data,    np.int32(512),              np.int32(self.rows),                        # output
                   self.r512_x_gpu.data,    self.r512_y_gpu.data,          np.int32(self.interpolation_order)).wait()  # plan, interpolation order            
        self.unwrap_time2 = time.time()
        
        if 'unwrapped' in self.returnables_list: self.returnables['unwrapped'] = self.unwrapped_gpu.get()
        if 'resized' in self.returnables_list: self.returnables['resized'] = self.resized512.get()
      
    def rotational_correlation(self,fallback='cpu'):
        """ Perform an angular correlation of a speckle pattern by doing a
        cartesian correlation of a speckle pattern unwrapped into polar
        coordinates. This method accepts no input. Data must be in
        self.resized512. Output is stored in place. After correlating the data
        is resized to 360 pixels in the angular axis.
        
        For fastest performance, this method requires installation of the hacked
        pyfft code to allow for single-axis transformations. Both CPU and GPU
        fallbacks are provided, but are much slower.
        
        Available returnables:
            correlated: the angular ac after normalization, resized to 360
            pixels in the angular coordinate.
        """
        
        self.correlation_time0 = time.time()
        
        # zero the buffers
        self.set_zero(self.resized512z)
        self.set_zero(self.corr_denoms)
            
        # get the denominaters
        self.sum_rows.execute(self.queue, (self.rows,), self.resized512.data, self.corr_denoms.data,np.int32(self.rows))
        
        # calculate the autocorrelation of the rows using a 1d plan and the batch command in pyfft
        self.fftplan_correls.execute(self.resized512.data,self.resized512z.data,wait_for_finish=True,batch=self.rows)
        self.make_intensity(self.resized512,self.resized512z,self.resized512)
        self.set_zero(self.resized512z)
        self.fftplan_correls.execute(self.resized512.data,self.resized512z.data,batch=self.rows,inverse=True,wait_for_finish=True)
        
        # normalize the autocorrelations by dividing each row by the normalization constants found in corr_denoms.
        self.row_divide.execute(self.queue, (512,self.rows), self.resized512.data, self.corr_denoms.data, self.resized512.data)
    
        self.map_coords.execute(self.queue, (360,self.rows),
            self.resized512.data,  np.int32(512),     np.int32(self.rows),                        # input
            self.resized360.data,  np.int32(360),     np.int32(self.rows),                        # output
            self.r360_x_gpu.data,  self.r360_y_gpu.data, np.int32(self.interpolation_order)).wait()  # plan, interpolation order 

        if 'correlated' in self.returnables_list: self.returnables['correlated'] = self.resized360.get()

        self.correlation_time1 = time.time()

    def decompose_spectrum(self):
        """ Decompose the angular correlation in self.resized360 into a cosine spectrum.
        This method accepts no input. Because the amount of data in the self.spectrum_gpu
        object is very small, moving the spectrum back into host memory is a cheap transfer.
        
        Available returnables:
            spectrum (the default)
        """

        assert self.can_has_cosines, "cant decompose until cosines are set!"

        self.decomp_time0 = time.time()
        
        self.set_zero(self.spectrum_gpu)            # zero the buffer self.spectrum_gpu
        self.cosine_reduce.execute(                 # run the cosine reduction kernel
            self.queue,(self.nc,self.rows),
            self.resized360.data,
            self.cosines.data,
            self.spectrum_gpu.data,
            np.int32(self.nc))
        self.decomp_time1 = time.time()
        
        if 'spectrum' in self.returnables_list: self.returnables['spectrum'] = self.spectrum_gpu.get()
        
    def _resize_plan(self,rows,columns_in,columns_out):
    
    # uR-ur gives the number of rows
    # columns_in gives the number of columns in the image being resize
    # columns_out gives the number of columns in the image after resizing
    
        cols1 = np.arange(columns_out).astype(np.float32)*columns_in/float(columns_out)
        rows1 = np.arange(rows).astype(np.float32)
        return np.outer(np.ones(rows,int),cols1), np.outer(rows1,np.ones(columns_out))    
    
    def run_on_site(self,y,x):
        
        """ Run the symmetry microscope on the site of the object described by the roll coordinates
        (y,x). Steps are:
        
        1. Roll the illumination
        2. Make the speckle
            2b: If coherence is specified, blur the speckle
        3. Unwrap the speckle
        4. Autocorrelate the unwrapped speckle
        5. Cosine-decompose the autocorrelation
        
        This function aggregates the more atomistic methods into a single function.
        
        arguments:
            y, x: row and column describing the center of the pinhole. for example,
            y = x = 0 puts the pinhole in the corner. y = x = N/2 puts the pinhole
            in the visible-center.
        """
        
        assert self.can_has_object,  "no object set"
        assert self.can_has_pinhole, "no pinhole set"
        assert self.can_has_unwrap,  "no unwrap set"
        assert self.can_has_cosines, "no cosines set"
        assert isinstance(y,int) and isinstance(x,int), "site coordinates must be integer"
        
        self.slice_object(y,x)
        self.make_speckle()
        if self.can_has_coherence: self.blur() # in the case of full coherence, this is skipped for a speed gain
    
        # in the cpu microscope code, the following steps are handled by symmetries.rot_sym
        self.unwrap_speckle()
        self.rotational_correlation()
        self.decompose_spectrum()

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

