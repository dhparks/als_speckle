import numpy
import speckle
import scipy.misc as smp

global use_gpu

try:
    import speckle.gpu as gpulib
    import pyopencl
    import pyopencl.array as cla
    import pyfft
    use_gpu = True
    
except ImportError:
    use_gpu = False
    f = numpy.fft.fft2
    g = numpy.fft.ifft2
    s = numpy.fft.fftshift

class fth():
    
    """ Class for fourier-transform holography interface. This is a basic
    class which assumes that the reference is delta-like, and consequently does
    not require any iterative phasing methods. """
    
    def __init__(self,force_cpu=False):
        
        # load the gpu if available
        if use_gpu:
            try:
                self.gpu = gpulib.gpu.init()
                self.context,self.device,self.queue,self.platform = self.gpu
            except gpulib.GPUInitError:
                use_gpu = False
        
        if force_cpu: use_gpu = False
        
        self.fourier_data  = None
        self.fourier_image = None
        self.rs_data       = None
        self.rs_image      = None
    
    def open_data(self,path):
        
        """ Open the data at path. Form the following important quantities:
        1. Array of far-field data self.fourier_data
        2. Array of real-space data self.rs_data
        3. HSV image of real-space data self.rs_image
        """

        # open. if trivially 3d, convert to 2d by taking first frame. if 3d and
        # has frames, raise an error as this is not supported.
        fourier_data = speckle.io.open(path).astype(numpy.float32)
        if fourier_data.ndim == 3:
            if fourier_data.shape[0] > 1:
                pass
                # actually, raise an error
            else:
                fourier_data = fourier_data[0]
        self.fourier_data = fourier_data
            
        if fourier_data.shape[0] != fourier_data.shape[1]:
            # embed in a square array, otherwise DFT sampling is wonky
            l = min(fourier_data.shape)
            g = max(fourier_data.shape)
            new = numpy.zeros((g,g),numpy.float32)
            new[:fourier_data.shape[0],:fourier_data.shape[1]] = fourier_data
            self.fourier_data = new
            
        rolls = lambda d,r1,r2: numpy.roll(numpy.roll(d,r1,axis=0),r2,axis=1)
        r, c  = numpy.unravel_index(self.fourier_data.argmax(),fourier_data.shape)
            
        # put the data in realspace through ifft2
        self.rs_data = s(g(rolls(self.fourier_data,-r,-c)))
        
        # make images with linear scaling, sqrt scaling, log scaling. this might
        # take a little bit of time.
        mag, phase = abs(self.rs_data), numpy.angle(self.rs_data)
        self.rs_image_linear = io.complex_hsv_image(self.rs_data)
        self.rs_image_sqrt   = io.complex_hsv_image(np.sqrt(mag)*np.exp(complex(0,1)*phase))
        self.rs_image_log    = io.complex_hsv_image(np.log(mag)*np.exp(complex(0,1)*phase))
        
        self.fourier_image_linear = smp.toimage(self.fourier_data)
        self.fourier_image_sqrt =   smp.toimage(numpy.sqrt(self.fourier_data))
        self.fourier_image_log =    smp.toimage(numpy.log(self.fourier_data))

    def select_region(self,region):
        pass
    
    def propagate(self):
        """ This method runs the back propagation routine. While the data is
        internally propagated at a large power of 2 to allow propagation limit
        to be fairly large, the data returned to the user is only that specified
        in the selected region. """
        
        if use_gpu:
            propagated,images = gpulib.gpu_propagate.gpu_propagate_distance(gpu_info,obj,distance,energy,pitch,subregion=(4*pixel_radius,),silent=False,im_convert='hsv')
        if not use_gpu:
            stuff()
        pass
    
    def scale_image(self):
        """ This method rescales the data so that it is easier to see the
        various correlation terms of the inverted hologram. """
        
        pass
    
    def calculate_acutance():
        pass
    
    def apodize():
        pass
        
    
    
    
    