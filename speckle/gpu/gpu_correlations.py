# core
import numpy
import pyopencl as cl
import pyopencl.array as cla
from pyopencl.elementwise import ElementwiseKernel as EK
import time
import string

# common libs. do some ugly stuff to get the path set to the kernels directory
from .. import wrapping,shape
import gpu
kp = string.join(gpu.__file__.split('/')[:-1],'/')+'/kernels/'

# cpu fft
DFT = numpy.fft.fft2
from numpy.fft import fftshift

import matplotlib
matplotlib.use('Agg')
import pylab

class angular_correlations():

    def __init__(self,gpuinfo,object_or_N,ur,uR,pinhole,cosine_components,interrupts=()):

        # this stuff is ALL REQUIRED to run the full simulation, so it is all initialized at the start.
        # work flow is variation over coordinate and sequential analysis to fourier spectrum, rather than compute all
        # the speckle patterns, then turn all those into correlations, etc.
        
        # since this is intended to be common code for the group, check types to make sure n00bs aren't putting in garbage
        assert isinstance(object_or_N,(int,numpy.ndarray)), "object for angular correlations must be ndarray or int"
        assert isinstance(ur,int) and isinstance(uR,int), "ur and uR (for unwrapping) must be integer type"
        ur,uR = min([ur,uR]), max([ur,uR])
        assert isinstance(pinhole,(int,float,numpy.ndarray)), "pinhole must be a radius (float or integer type) or a custom supplied array"
        assert isinstance(cosine_components,(list,tuple,numpy.ndarray)), "cosine components must be iterable: list, tuple, ndarray"
        for component in cosine_components: assert isinstance(component,int), "each cosine component must be an integer"
        assert ur > 0, "unwrapped_r must be > 0"
        
        self.context, self.device, self.queue, platform = gpuinfo
        if isinstance(object_or_N,numpy.ndarray):
            self.N = len(object_or_N)
            self.master_object = cla.to_device(self.queue,object_or_N.astype(numpy.float32))
        if isinstance(object_or_N,int):
            self.N = object_or_N
            self.master_object = cla.empty(self.queue,(self.N,self.N),numpy.float32)
        self.interrupts = interrupts
        self.ur = ur
        self.uR = uR
        self.rows = uR-ur
        assert self.uR <= self.N/2, "unwrapped_R must be < N/2"
        
        # check for the apple/nvidia problem
        if 'Apple' in str(platform) and 'GeForce' in str(self.device) and ispower2(self.rows): self.adjust_rows()
        print self.ur,self.uR,self.rows

        Pitch = 35e-9
        CL_x = 3e-6
        CL_y = 4e-6
        self.interpolation_order = 1
        self.cosine_components = cosine_components
        
        # make the fft plans. 1. NxN (for speckle), 2. NxN split (for blurring) 3a. (uR-ur)x512 (for correlating) 3b. (512,) for correlating without hacked pyfft
        from pyfft.cl import Plan as fft_plan
        self.fftplan_speckle = fft_plan((self.N,self.N),                      queue=self.queue)
        self.fftplan_blurred = fft_plan((self.N,self.N), dtype=numpy.float32, queue=self.queue)
        try:
            # this plan requires real and imaginary components to be separately specified
            self.fftplan_correls = fft_plan((self.rows,512), dtype=numpy.float32, queue=self.queue, axes=(1,))
            self.pyfft_x = True
        except:
            print "couldn't build axis=1 fft. instead, will build a 1d fft and loop to correlate rows, which is much slower!"
            print "for fast 1d ffts, install plan.py and cl.py from the repo into the pyfft directory"
            self.fftplan_correls = fft_plan((512,),dtype=numpy.float32,queue=self.queue)
            self.pyfft_x = False

        # Build OpenCL programs from external kernel files.
        self.slice_view    = _build_kernel_file(self.context, self.device, kp+'slice_view.cl')    # for extracting a small array from a larger one
        self.map_coords    = _build_kernel_file(self.context, self.device, kp+'map_coords_buffer.cl')    # for unwrapping and resizing
        self.sum_rows      = _build_kernel_file(self.context, self.device, kp+'correl_denoms.cl') # for getting correlation normalizations
        self.row_divide    = _build_kernel_file(self.context, self.device, kp+'row_divide.cl')    # for doing correlation normalization
        self.cosine_reduce = _build_kernel_file(self.context, self.device, kp+'cosine_reduce.cl') # for doing cosine spectrum reduction
        
        if not self.pyfft_x:
            self.slice_row = _build_kernel_file(self.context, self.device, kp+'slice_row_f_f.cl') # slice row to correlate
            self.put_row   = _build_kernel_file(self.context, self.device, kp+'put_row_f_f.cl')   # put correlated row back in buffer
        
        # more kernels. due to their simplicity these are not put into external files
        self.illuminate_re = EK(self.context,
            "float2 *a, " # illumination
            "float *b, "  # transmission/object
            "float2 *c",  # output
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
        
        # illumination function
        if isinstance(pinhole, (int,float)):
            illumination = shape.circle((self.N,self.N),pinhole)
            #illumination = shape.square((self.N,self.N),2*pinhole)
            self.pinhole_r = pinhole
        else: illumination = pinhole
        self.illumination_gpu = cla.to_device(self.queue,illumination.astype(numpy.complex64))

        # calculate the charge signal and load it into memory. assume this is just the fourier intensity of the illumination function.
        ChargeSignal       = abs(DFT(illumination))**2
        self.charge_signal = cla.to_device(self.queue,ChargeSignal.astype(numpy.float32))
        self.charge_max    = cla.max(self.charge_signal).get()

        # allocate memory for intermediate results in the speckle calculation. these all need to be NxN clas 
        self.active_object    = cla.empty(self.queue, (self.N,self.N), numpy.float32  )
        self.scratch1         = cla.empty(self.queue, (self.N,self.N), numpy.complex64)
        self.scratch2         = cla.empty(self.queue, (self.N,self.N), numpy.complex64)
        self.fft_signal       = cla.empty(self.queue, (self.N,self.N), numpy.complex64)
        self.intensity_sum_re = cla.empty(self.queue, (self.N,self.N), numpy.float32  )
        self.intensity_sum_im = cla.empty(self.queue, (self.N,self.N), numpy.float32  )
        self.blurred_image    = cla.empty(self.queue, (self.N,self.N), numpy.float32  )
        self.blurred_image2   = cla.empty(self.queue, (self.N,self.N), numpy.float32  )
        
        cl_x_px = CL_x/Pitch
        cl_y_px = CL_y/Pitch
        gaussian = fftshift(_gaussian_kernel(cl_y_px,cl_x_px,self.N))
        self.blur_kernel = cl.array.to_device(self.queue,gaussian.astype(numpy.float32))

        # make the unwrap plan. put the buffers into gpu memory
        xTable,yTable,self.unwrap_cols = _unwrap_plan(self.N,self.ur,self.uR,self.N/2,self.N/2)
        print self.unwrap_cols
        self.unwrap_x_gpu  = cla.to_device(self.queue,xTable.astype(numpy.float32))
        self.unwrap_y_gpu  = cla.to_device(self.queue,yTable.astype(numpy.float32))
        self.unwrapped_gpu = cla.empty(self.queue,(self.rows,self.unwrap_cols), numpy.float32)
        
        # make resizing plans and buffers for row correlations
        r512_x, r512_y   = _resize_plan(self.ur,self.uR,self.unwrap_cols,512)
        r360_x, r360_y   = _resize_plan(self.ur,self.uR,512,360)
        self.r512_x_gpu  = cla.to_device(self.queue, r512_x.astype(numpy.float32)) # plan
        self.r512_y_gpu  = cla.to_device(self.queue, r512_y.astype(numpy.float32)) # plan
        self.r360_x_gpu  = cla.to_device(self.queue, r360_x.astype(numpy.float32)) # plan
        self.r360_y_gpu  = cla.to_device(self.queue, r360_y.astype(numpy.float32)) # plan
        self.resized512  = cla.empty(self.queue,r512_x.shape,      numpy.float32)  # result, also stores correlations
        self.resized360  = cla.empty(self.queue,r360_x.shape,      numpy.float32)  # holds data after resizing to 360 pixels in theta
        self.corr_denoms = cla.empty(self.queue,(r512_x.shape[0],),numpy.float32)  # correlation normalizations
        
        # depending on presence of axis = 1 support in pyfft, set up the correct buffers
        if self.pyfft_x:
            self.resized512z = cla.empty(self.queue,r512_x.shape,numpy.float32) # an array of zeros for the xaxis-only split fft
            self.set_zero(self.resized512z)
        if not self.pyfft_x:
            self.active_row  = cla.empty(self.queue,(512,),numpy.float32)
            self.dummy_row   = cla.empty(self.queue,(512,),numpy.float32)
        
        # these interrupts need a blocker
        if 'speckle_blocker' in interrupts or 'blurred_blocker' in interrupts:
            self.blocker     = 1-shape.circle((self.N,self.N),self.ur)
        
        # make the cosine array and the buffers for cosine reduction
        # make the buffer for doing row-correlations
        components = numpy.zeros((len(self.cosine_components),360),float)
        angles = numpy.arange(360)*2*numpy.pi/360.
        for n, c in enumerate(self.cosine_components): components[n] = numpy.cos(angles*c)
        self.cosines = cla.to_device(self.queue,components.astype(numpy.float32))

        # initialize the spectrum array
        self.spectrum_gpu   =  cla.empty(self.queue,(self.rows,len(self.cosine_components)), numpy.float32)
        
        # initialize the returns dictionary
        self.returnables = {}
        
    def adjust_rows(self):
        """ Change the value of unwrapped_R or unwrapped_r in the case of a GeForce
        card on Apple when unwrapped_R-unwrapped_r is a power of 2. For unknown reasons
        (most likely Apple's OpenCL compiler), this combination can lead to NaNs."""
        
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
                             
    def load_object(self,object):
        """ Load an object into the self.master_object GPU memory space. It must be the same
        size that the class instance was initialized with."""
        
        assert isinstance(object,numpy.ndarray) and object.ndim==2, "object must be 2d array"
        assert object.shape == (self.N,self.N), "object shape differs from initialization values"
        
        self.master_object.set(object.astype(self.master_object.dtype))

    def change_interrupts(self,interrupts=()):
        self.interrupts = interrupts
        if 'speckle_blocker' in interrupts or 'blurred_blocker' in interrupts:
            self.blocker = 1-shape.circle((self.N,self.N),self.ur)

    def slice_object(self,top_row,left_col):
        """ Copy an object from self.master_object into self.active_active. The size of the sliced
        array is the same as that of the object specified when instantiating the class.
        
        Inputs are:
            top_row: y coordinate for slice 
            left_col: x coordinate for slice
            site (optional): 
            
        Interrupts:
            object: the master (unsliced) object 
            sliced: the sliced (active) object
        """
        
        self.slice_time0 = time.time()
        self.slice_view.execute(self.queue, (self.N,self.N),             # opencl stuff
                   self.master_object.data,                              # input: master_object
                   self.active_object.data,                              # output: active_object
                   numpy.int32(self.N),numpy.int32(self.N),             # array dimensions
                   numpy.int32(top_row), numpy.int32(left_col)).wait()   # coords
        self.slice_time1 = time.time()

        if 'object' in self.interrupts: self.returnables['object'] = self.master_object.get()
        if 'sliced' in self.interrupts: self.returnables['sliced'] = self.active_object.get()

    def make_speckle(self):
        """Turn the current occupant of self.active_object into a fully speckle pattern.
        This method accepts no input besides the optional site which is used only for saving symmetry microscope
        output at particular illumination locations. The speckle pattern is stored in self.intensity_sum_re.
        
        Available interrupts:
            illuminated: the product of self.active_object and self.pinhole.
            speckle: the fully coherent speckle pattern.
            speckle_blocker: the speckle pattern with a blocker of radius self.ur in the center.
        """

        # reset intensity_sum for each iteration by filling with zeros
        self.set_zero(self.intensity_sum_re)
        self.set_zero(self.intensity_sum_im)
    
        self.make_time0 = time.time()
        
        # calculate the exit wave by multiplying the illumination
        to_return = []
        return_something = False
        self.illuminate_re(self.illumination_gpu, self.active_object, self.scratch1).wait()
        if 'illuminated' in self.interrupts:
            Temp = self.scratch1.get()
            try:
                x = self.pinhole_r+15
                Temp = Temp[self.N/2-x:self.N/2+x,self.N/2-x:self.N/2+x]
            except: pass
            self.returnables['illuminated'] = Temp
        
        # calculate the fft of the wavefield
        self.fftplan_speckle.execute(self.scratch1.data, data_out=self.scratch2.data,wait_for_finish=True)
        
        # turn the complex field (scratch2) into speckles (intensity_sum_re)
        self.add_intensity(self.scratch2, self.intensity_sum_re).wait()
                
        if 'speckle' in self.interrupts: self.returnables['speckle'] = fftshift(self.intensity_sum_re.get())
        if 'speckle_blocker' in self.interrupts: self.returnables['speckle_blocker'] = fftshift(self.intensity_sum_re.get())*self.blocker
            
        self.make_time1 = time.time()

    def blur_speckle(self):
        """ Turn the fully-coherent speckle pattern in self.intensity_sum_re into a partially coherent
        speckle pattern through convolution with a mutual intensity function. This method accepts no input.
        The blurred speckle is saved in self.intensity_sum_re.

        Available interrupts:
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
        #self.scalar_mult(self.blurred_image,numpy.float32(p1/p2))
        self.blur_time1 = time.time()

        if 'blurred' in self.interrupts: self.returnables['blurred'] = fftshift(abs(self.intensity_sum_re.get()))
        if 'blurred_blocker' in self.interrupts: self.returnables['blurred_blocker'] = fftshift(abs(self.intensity_sum_re.get()))*self.blocker
    
    def unwrap_speckle(self):
        """Remap the speckle into polar coordinate, and then resize it to 512 pixels along the x-axis.
        This method accepts no input.
        
        Available interrupts:
            unwrapped: unwrapped speckle
            resized: unwrapped speckle resized to 512 pixels wide"""
            
        # can this be combined into a single operation?
        
        #unwrap
        self.unwrap_time0 = time.time()
        self.map_coords.execute(self.queue, (self.unwrap_cols,self.rows),                                                   # opencl stuff
                   self.intensity_sum_re.data, numpy.int32(self.N),           numpy.int32(self.N),                          # input
                   self.unwrapped_gpu.data,    numpy.int32(self.unwrap_cols), numpy.int32(self.rows),                       # output
                   self.unwrap_x_gpu.data,     self.unwrap_y_gpu.data,        numpy.int32(self.interpolation_order)).wait() # plan, interpolation order
        self.unwrap_time1 = time.time()
            
        # resize
        self.map_coords.execute(self.queue, (512,self.rows),
                   self.unwrapped_gpu.data, numpy.int32(self.unwrap_cols), numpy.int32(self.rows),                        # input
                   self.resized512.data,    numpy.int32(512),              numpy.int32(self.rows),                        # output
                   self.r512_x_gpu.data,    self.r512_y_gpu.data,          numpy.int32(self.interpolation_order)).wait()  # plan, interpolation order            
        self.unwrap_time2 = time.time()
        
        if 'unwrapped' in self.interrupts: self.returnables['unwrapped'] = self.unwrapped_gpu.get()
        if 'resized' in self.interrupts: self.returnables['resized'] = self.resized512.get()

    def gpu_autocorrelation(self,fftplan,re_data,im_data,data_out=None):
        """ A helper function which does the autocorrelation of complex data according to the supplied plan.
         fftplan should expect split (not interleaved) data. The real data is in re_data and the imaginary
         data is in im_data. The output, which must be real because its an autocorrelation, is stored in
         re_data unless a separate buffer is specified by optional argument data_out. im_data is overwritten
         with zero so be careful!"""
         
        fftplan.execute(re_data.data,im_data.data,wait_for_finish=True)              # forward transform
        self.make_intensity(re_data,im_data,re_data)                                 # re**2 + im**2; store in re
        self.set_zero(im_data)                                                       # set im to 0
        if data_out == None:
            fftplan.execute(re_data.data,im_data.data,inverse=True,wait_for_finish=True)
        else:
            fftplan.execute(re_data.data,im_data.data,data_out=data_out.data,inverse=True,wait_for_finish=True)
            
    def rotational_correlation(self):
        """ Perform an angular correlation of a speckle pattern by doing a cartesian correlation of
        a speckle pattern unwrapped into polar coordinates. This method accepts no input. Data must be
        in self.resized512. Output is stored in place. After correlating the data is resized to 360 pixels
        in the angular axis.
        
        This method requires installation of the hacked pyfft code to allow for single-axis transformations.
        
        Available interrupts:
            correlation: the autocorrelated data after normalization
        """
        
        # fft, square, ifft each row of the image
        self.correlation_time0 = time.time()

        self.set_zero(self.corr_denoms) # have to use this function because array.set() is crashing for unknown reasons
        
        if self.pyfft_x:
            # zero the buffer for imaginary data
            self.set_zero(self.resized512z)
            
            # get the denominaters
            self.sum_rows.execute(self.queue, (self.rows,),
                              self.resized512.data,
                              self.corr_denoms.data,numpy.int32(self.rows))
            
            self.gpu_autocorrelation(self.fftplan_correls,self.resized512,self.resized512z)
        
        if not self.pyfft_x:
            
            # get the denominaters
            self.sum_rows.execute(self.queue, (self.rows,),
                              self.resized512.data,
                              self.corr_denoms.data,numpy.int32(1))
            
            for r in range(self.rows):
                # slice out active row. autocorrelate the row. put the autocorrelation back in.
                offset = numpy.int32(512*r)
                self.set_zero(self.dummy_row)
                self.slice_row.execute(self.queue,(512,),self.resized512.data,self.active_row.data,offset)
                self.gpu_autocorrelation(self.fftplan_correls,self.active_row,self.dummy_row)
                self.put_row.execute(self.queue,(512,),self.active_row.data,self.resized512.data,offset)
        
        self.row_divide.execute(self.queue, (512,self.rows), # normalize the row_corr_re output by dividing each row by the
            self.resized512.data,                            # normalization constants found in corr_denoms
            self.corr_denoms.data,
            self.resized512.data) # output is kept in-place

        self.map_coords.execute(self.queue, (360,self.rows),
            self.resized512.data,  numpy.int32(512),     numpy.int32(self.rows),                        # input
            self.resized360.data,  numpy.int32(360),     numpy.int32(self.rows),                        # output
            self.r360_x_gpu.data,  self.r360_y_gpu.data, numpy.int32(self.interpolation_order)).wait()  # plan, interpolation order 

        if 'correlation' in self.interrupts: self.returnables['correlation'] = self.resized360.get()

        self.correlation_time1 = time.time()

    def decompose_spectrum(self):
        """ Decompose the angular correlation in self.resized360 into a cosine spectrum.
        This method accepts no input. Because the amount of data in the self.spectrum_gpu
        object is very small, moving the spectrum back into host memory is a cheap transfer.
        
        Available interrupts:
            spectrum
        """

        self.decomp_time0 = time.time()
        
        self.set_zero(self.spectrum_gpu)            # zero the buffer self.spectrum_gpu
        self.cosine_reduce.execute(                 # run the cosine reduction kernel
            self.queue,(len(self.cosine_components),self.rows),
            self.resized360.data,
            self.cosines.data,
            self.spectrum_gpu.data,
            numpy.int32(len(self.cosine_components)))
        self.decomp_time1 = time.time()
        
        if 'spectrum' in self.interrupts: self.returnables['spectrum'] = self.spectrum_gpu.get()

def make_raster_coords(N,xstep,ystep,size=None):

    if size == None:
        start, stop = 0,N
    else:
        assert size%2 == 0, "size to make_coords must be even"
        start, stop = N/2-size/2,N/2+size/2

    x_coords = numpy.arange(start,stop,xstep)
    y_coords = numpy.arange(start,stop,ystep)
    return x_coords, y_coords

def _resize_plan(ur,uR,columns_in,columns_out):
    
    # uR-ur gives the number of rows
    # columns_in gives the number of columns in the image being resize
    # columns_out gives the number of columns in the image after resizing
    
    cols = numpy.arange(columns_out).astype(numpy.float32)*columns_in/float(columns_out)
    rows = numpy.arange(uR-ur).astype(numpy.float32)
    return numpy.outer(numpy.ones(uR-ur,int),cols), numpy.outer(rows,numpy.ones(columns_out))

def _unwrap_plan(N,ur,uR,x,y,MaxAngle=None):
    import scipy
    if MaxAngle == None: MaxAngle = 2*numpy.pi
    
    # setup up polar arrays
    R2,R1,cx,cy = uR,ur,x,y
    
    cMax = int(MaxAngle*R2)
    RTable, CTable = numpy.indices((R2-R1,cMax),float)
    RTable += R1
    PhiTable = numpy.ones_like(RTable)*numpy.arange(cMax)*MaxAngle/cMax
    Rows,Cols = RTable.shape

    # trigonometry
    x0Table = cx+RTable*numpy.cos(PhiTable)
    y0Table = cy+RTable*numpy.sin(PhiTable)

    return numpy.mod(x0Table+N/2,N),numpy.mod(y0Table+N/2,N),Cols # fft comes back with 0-freq @ (0,0) so we need to modulo the arithmetic
    
def _gaussian_kernel(sx,sy,N):
    kernel_x = numpy.array([0,1,0]).astype(numpy.float32)
    kernel_y = numpy.array([0,1,0]).astype(numpy.float32)
    
    if sx > 0:
        L = N
        x = numpy.arange(L).astype('float')
        kernel_x = numpy.exp(-1*abs(L/2-x)**2/(2*sx**2)).astype(numpy.float32)
        
    if sy > 0:
        L = N
        x = numpy.arange(L).astype('float')
        kernel_y = numpy.exp(-1*abs(L/2-x)**2/(2*sy**2)).astype(numpy.float32)
        
    return numpy.outer(kernel_x,kernel_y)

def _build_kernel_file(c,d,fileName):
    kernelFile = open(fileName, 'r')
    kernelStr = kernelFile.read()

    # Load the program source
    program = cl.Program(c, kernelStr)

    # Build the program and check for errors
    program.build(devices=[d])

    return program

def _build_kernel(c,d,kernel):
    
    # Load the program source
    program = cl.Program(c, kernel)

    # Build the program and check for errors
    program.build(devices=[d])

    return program

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

        if magnetic:
            
            contrast = 2
            n0 = numpy.complex64(complex(0.998482,0.0019125))
            dn = contrast*numpy.complex64(complex(-3.03667e-05,3.825e-05)) # magnetic modulation to index of refraction
            wavelength = 1240e-9/780.
            scale = 2*scipy.pi*600e-9/wavelength # just a scaling factor for the transmission function

            for polarization in [-1.,1.]: # each polarization give a separate speckle pattern
                # calculate the transmission through the domains
                self.transmit.execute(self.queue,(self.N**2,),                                                                      # opencl stuff
                                      numpy.float32(scale), numpy.float32(polarization), numpy.complex64(n0), numpy.complex64(dn),  # constants
                                      self.active_object.data, self.scratch1.data).wait()                                          # in/out buffers
                if 'transmission' in self.interrupts and self.ipath != None:
                    Temp = self.scratch1.get()
                    io2.save('%s/%s_%s transmission polarization_%s.fits'%(self.ipath,self.int_name,polarization),Temp,components='real')
    
                # calculate the exit wave by multiplying the illumination
                self.multiply_buffers(self.illumination_gpu, self.scratch1, self.scratch1).wait()
                if 'illuminated' in self.interrupts and self.ipath != None:
                    Temp = self.scratch1.get()
                    io2.save('%s/%s_%s illuminated polarization_%s.fits'%(self.ipath,self.int_name,site,polarization),Temp)
                
                # calculate the fft of the wavefield
                self.fftplan_speckle.execute(self.scratch1.data, data_out=self.scratch2.data,wait_for_finish=True)
                if 'fft' in self.interrupts and self.ipath != None:
                    Temp = self.scratch2.get()
                    io2.save('%s/%s_%s fft polarization_%s.fits'%(self.ipath,self.int_name,site,polarization),Temp)
                
                # calculate the intensity of the fft
                self.add_intensity(self.scratch2, self.intensity_sum_re).wait()
                if 'intensity' in self.interrupts and self.ipath != None:
                    Temp = self.intensity_sum_re.get()
                    io2.save('%s/%s_%s intensity_sum polarization_%s.fits'%(self.ipath,self.int_name,site,polarization),Temp)
                
            # subtract the charge signal from the sum. keep the result in self.intensity_sum
            signal_max = cla.max(self.intensity_sum_re).get()
            ratio = signal_max/self.charge_max
            self.subtract_charge(self.charge_signal,self.intensity_sum_re,self.intensity_sum_re,numpy.float32(ratio))
            if 'isolated' in self.interrupts and self.ipath != None:
                Temp = fftshift(self.intensity_sum_re.get())
                io2.save('%s/%s_%s isolated %s.fits'%(self.ipath,self.int_name,site,contrast),Temp,components='real')"""

