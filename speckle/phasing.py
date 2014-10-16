# implement coherent phasing algorithms on cpu or gpu if available.
# it is assumed that if a gpu is available, it is desired.
# the goal of this code is to completely unify the phasing interface,
# which was previously split between cpu and gpu files.

import numpy as np
import wrapping, masking, gpu, sys, time, io
from math import sin, cos
w = sys.stdout.write
common = gpu.common

try:
    import string
    import pyopencl
    import pyopencl.array as cla
    import pyfft
    HAVE_GPU = True
except ImportError:
    HAVE_GPU = False
    
try:
    import numexpr
    HAVE_NUMEXPR = True
except ImportError:
    HAVE_NUMEXPR = False
    
try:
    import pyfftw
    HAVE_FFTW = True
except ImportError:
    HAVE_FFTW = False
    
class phasing(common):
    
    """ A class to hold a reconstruction and methods to operate on the
    reconstruction. The GPU and CPU interfaces are identical, and
    handled internally in the methods.
    
    To start a reconstruction, call the class.
    
    To load data into a reconstruction, call load(), which must be accessed
    with keyword arguments. For example, after opening a prepared Fourier,
    load it into the class by load(modulus=$modulus_data$).
    
    For the reconstruction to iterate, the following data must be loaded:
        1. The goal fourier modulus. This should be corner-centered. 'modulus'
        2. The estimated support. 'support'
        3. The number of independent trials. 'numtrials'
        4. (Optional) An estimate of the coherence factor, which should
            also be corner-centered. 'ipsf'
    Data can be loaded in any order. Data can also be changed, eg, to update
    the estimate of the support.
    
    Typical basic usage might be as follows:
        reconstructor = speckle.phasing.phasing()
        reconstructor.load(modulus=$modulus_data, support=$support_data, trials=10)
        reconstructor.iterate(500)
        results = reconstructor.get(reconstructor.save_buffer)
        
    More advanced usage (for example: refinement of the support):
        reconstructor = speckle.phasing.phasing()
        reconstructor.load(modulus=$modulus_data, support=$support_data, trials=10)
        reconstructor.iterate(500)
        results = reconstructor.get(reconstructor.save_buffer)
        
        average_result = results.mean()
        new_support = speckle.phasing.refine_support(reconstructor.get(reconstructor.support), average_result)
        reconstructor.load(support=new_support)
        reconstructor.iterate(500)
        results = reconstructor.get(reconstructor.save_buffer)
            
    Publicly available methods in this class are:
    
        get() -  retrieve class attributes; same interface between cpu and gpu.
        iterate() - run iterations on the current trial
        load() - load data into the class
        optimize_coherence(): attempt to optimize the coherence function using
            one of two methods. first, an estimate can be made using
            richardson-lucy deconvolution. second, an estimate can be made
            assuming that the coherence function is gaussian in form.
        seed() - make a complex random number to use as the initial guess
        status() - reports the current status of the reconstruction
    """
    
    def __init__(self, force_cpu=False, force_np_fft=False, gpu_info=None):

        # load the gpu if available
        # keep context, device, queue, platform, and kp in the parent namespace
        # (still available to self)
        if HAVE_GPU and not force_cpu:
            common.project = 'phasing'
            self.start(gpu_info)
        else:
            self.use_gpu = False

        # tell the methods in common which device we're using
        common.use_gpu = self.use_gpu 

        if self.use_gpu:
            self.compute_device = 'gpu'  
        else:
            self.compute_device = 'cpu'
        
        # can switch between np.fft and pyfft by use of the force_np_fft
        # switch. mainly this is just for testing
        if not self.use_gpu:
            
            if HAVE_FFTW and not force_np_fft:
                print "using fftw"
                self.fft2 = pyfftw.interfaces.numpy_fft.fft2
                self.ifft2 = pyfftw.interfaces.numpy_fft.ifft2
                
            else:
                print "using np.fft"
                self.fft2 = np.fft.fft2
                self.ifft2 = np.fft.ifft2
            
        # state variables. certain of these must be changed from zero for the
        # reconstruction to proceed. these track which data is loaded into the
        # class. 0 = nothing; 1 = partial; 2 = complete.
        self.modulus_state = 0
        self.support_state = 0
        self.ipsf_state = 0
        self.buffer_state = 0
        self.rl_state = 0
        self.spectrum_state = 0
        self.can_iterate = False
        
        self.numtrial = 0
        self.array_dtypes = ('float32', 'complex64')
        
        self.ints = (int, np.int8, np.int16, np.int32, np.uint8)
        self.floats = (float, np.float16, np.float32, np.float64)
        self.float2s = (complex, np.complex64, np.complex128)
        self.iterables = (list, tuple, np.ndarray)
        
        # declare attributes for reconstruction
        self.fftplan = None
        self.psi_fourier = None
        self.psi_out = None
        self.psi_in = None
        self.modulus = None
        self.fourier_tmp = None
        self.fourier_div = None
        self.N = None
        self.support = None
        self.size = None
        self.shape = None
        self.reconstruction = None
        
        # attributes for saving trials to buffer
        self.numtrials = None
        self.r0 = None
        self.c0 = None
        self.rows = None
        self.cols = None
        self.savebuffer = None

        # attributes for coherence correction
        self.N_spctrm = None
        self.ipsf = None
        self.sweights = None
        self.rescales = None

        # declare attributes for coherence optimization
        self.optimize_ac = None
        self.optimize_ac2 = None
        self.rl_sum = None
        self.rl_d = None
        self.rl_p = None
        self.rl_u = None
        self.optimize_d = None
        self.optimize_g = None
        self.optimize_m = None
        self.optimize_bl2 = None
        self.rl_blur = None
        self.rl_ph = None
        self.optimize_bl = None
        self.counter = None
        
    def iterate(self, iterations, order=None, silent=True, beta=0.8,
                debug_on=None):
    
        """ Run iterations. This is the primary method for the class. When the
        iterations terminate, the result is copied to self.save_buffer.
        
        required input:
            iterations -- The total number of iterations to run
            
        optional input:
            order -- A specification of which algorithms to run. Currently,
                this code supports the following algorithms:
                
                hio, er, sf, raar, dm

                The order should be an iterable of entries, each entry
                giving an algorithm and then a number of iterations. Example:
                
                order = (('hio',5),('er',5),('sf',1))
                
                tells the program to do 5 iterations of hio, 5 iterations of er
                and finally 1 iteration of sf. The sequence is looped until
                the total given number of iterations is reached. The number of
                iterations described within order DOES NOT have to be a divisor
                of iterations passed to iterate(). So, the above example order
                could be combined with iterations = 13 to give
                
                hio, hio, hio, hio, hio, er, er, er, er, er, sf, hio, hio
                
                at which point the program finishes.
                
                Default value is (('hio',99),('er',1))
                
            silent - (default True) If set to True, outputs the iteration
                number. Can also be set to an integer value, in which case
                the iteration number is printed out every int(silent)
                iterations.  This is useful to monitor progress.
                
            debug_on -- if a tuple of integers is supplied here, the machine
                will save debugging output from the _iteration method during
                the listed iterations.
            
        The keyword argument silent (default True) can be set to False, in
        which case every iteration spits out an update, or a number, in which
        case those iterations where iteration%silent == 0 will report.
        """
        
        if debug_on == None:
            debug_on = ()
             
        if not self.can_iterate:
            print "cant iterate before loading support and modulus."
            for n in ('modulus', 'support', 'buffer'):
                print "modulus state: "+str(self.modulus_state)
                print "support state: "+str(self.support_state)
                print "buffer  state: "+str(self.buffer_state)
            
        iterations = int(iterations)
        
        # build the iteration list
        assert isinstance(order, (type(None), list, tuple)),\
        "algorithm order must be iterable"
        
        if order == None:
            order = (('hio', 99), ('er', 1))

        tmp = []
        for entry in order:
            try:
                t = str(entry[0])
                i = int(entry[1])
            except (IndexError, TypeError):
                print "entry %s in iteration order is improperly formatted"
                exit()
            tmp.append([t for j in range(i)])
        order_list = [x for entry in tmp for x in entry]
        k = len(order_list)
                
        # run the iterations
        for iteration in range(iterations):

            if iteration in debug_on:
                self._iteration(order_list[iteration%k], iteration=iteration,
                                beta=beta, debug=True)
            else:
                self._iteration(order_list[iteration%k], iteration=iteration,
                                beta=beta)

            # print an update to the terminal to show the program is
            # still running
            if silent != True:
                if isinstance(silent, int):
                    if (iteration+1)%silent == 0:
                        print "  iteration %s"%(iteration+1)
                if isinstance(silent, bool):
                    print "  iteration %s"%(iteration+1)
            
        # copy the current reconstruction to the save buffer
        if self.use_gpu:
            self._kexec('copy_to_buffer_f2', self.savebuffer, self.psi_in,
                        self.c0, self.r0, self.numtrial, self.N,
                        shape=(self.cols, self.rows))
        else:
            sliced = self.psi_in[self.r0:self.r0+self.rows, \
                                 self.c0:self.c0+self.cols]
            
            self.savebuffer[self.numtrial] = sliced.astype(np.complex64)

        self.numtrial += 1

    def load(self, modulus=None, support=None, ipsf=None, spectrum=None,
             numtrials=None):
        
        """ Load data into the reconstruction object. This function accepts
        only keyword arguments, which can be the following: 
        
        modulus - the Fourier modulus to invert. Needs to have q = 0 at the
            corner. Should be a 2d square float array.
            
        support - an initial estimate of the support of the object. Should
            be a 2d square array.
        
        ipsf - an estimate of the ifft of the point-spread function. this
            is also called the complex coherence factor or just coherence
            function. if it is supplied, the fourier projection used in the
            reconstruction will be the PC projection, which convolves
            the estimated fourier modulus with the psf to generate a blurry
            fourier modulus
        
        spectrum - a 2d array (2 x L) where L is the number of bins in
            an estimate of the spectrum of incident illumination.
            spectrum[0] should be the values of the energy in eV.
            spectrum[1] should be the relative weights of the corresponding
            energies. If spectrum is supplied and a GPU is present, the
            reconstruction will take into account the dilation of the speckle
            pattern due to the temporal coherence. The CPU code path of this
            script DOES NOT support this type of coherence correction.
        
        numtrials - how many independent reconstructions to perform (int)
        
        Data can be loaded in any order through any number of calls to
        load(). This allows, for example, the coherence estimate to be updated
        during the course of the reconstruction, or for the support to
        be refined.
        
        For reconstructions to proceed, it is necessary to have specified
        the modulus, the support, and the numtrials. """
        
        # check types, sizes, etc
        types = (type(None), np.ndarray)
        assert isinstance(modulus, types), \
        "modulus must be ndarray if supplied"
        
        assert isinstance(support, types), \
        "support must be ndarray if supplied"
        
        assert isinstance(ipsf, types), \
        "psf must be ndarray if supplied"
        
        ### first, do all the loading that has no dependencies
        
        # load the modulus. should be (NxN) array. N should be a power of 2.
        if modulus != None:
            # modulus is the master from which all the other arrays take
            # their required size
            
            assert modulus.ndim == 2
            modulus = modulus.astype(np.float32)
            
            if self.modulus_state == 2:
                assert modulus.shape == self.shape
            if self.modulus_state != 2:
                assert modulus.shape[0] == modulus.shape[1]
                
                self.N = modulus.shape[0]
                self.shape = modulus.shape
                self.size = modulus.size
                
                # allocate memory for the modulus
                self.modulus = self._allocate(self.shape, np.float32, 'modulus')
                
                # allocate NxN complex buffers for iterations
                names = ('psi_in', 'psi_out', 'psi_fourier', 'fourier_div',\
                         'fourier_tmp', 'challenge')
                s = "self.%s = self._allocate(self.shape, np.complex64, \
                        name='%s')"
                for n in names:
                    exec(s%(n, n))
                
            # make the fft plan. because this is remade every time, don't
            # expect it to go stale.
            if self.use_gpu:
                from pyfft.cl import Plan
                self.fftplan = Plan((self.N, self.N), queue=self.queue)
                
            # load the modulus
            self.modulus = self._set(modulus, self.modulus)
            self.modulus_state = 2
            
        # load or replace the support. should be a 2d array. size must be
        # smaller than modulus in all dimensions (or equally sized to modulus)
        if support != None:
            
            assert support.ndim == 2, "support is %s-dim"%support.ndim
            
            assert support.shape[0] == support.shape[1], \
            "support shape is %s %s, must be square"%(support.shape)
            
            support = support.astype(np.float32)
            support *= 1./support.max()

            # load the support into memory. because we only check the size
            # later after both support and modulus have been loaded, we must
            # re-allocate memory every time.
            if self.support_state == 2:
                self.numtrial = 0
            self.support = self._allocate(support.shape, np.float32, 'support')
            self.support = self._set(support, self.support)
            
            # make the new bounding region.
            # do not yet allocate memory for the buffer.
            bounds = masking.bounding_box(support, pad=10)
            self.r0, self.rows = bounds[0], int(bounds[1]-bounds[0])
            self.c0, self.cols = bounds[2], int(bounds[3]-bounds[2])

            self.support_state = 1
            
        # get the number of trials into memory. unlimited (limited by memory)
        if numtrials != None:
            self.numtrials = int(numtrials)
            if self.buffer_state == 0:
                self.buffer_state = 1
            
        # load or replace ipsf. should be same size as modulus.
        if ipsf != None:
            
            assert ipsf.ndim == 2
            assert ipsf.shape[0] == ipsf.shape[1]
            ipsf = ipsf.astype(np.complex64)
            
            if self.ipsf_state == 2:
                assert ipsf.shape == self.ipsf.shape
            self.ipsf = self._allocate(ipsf.shape, np.complex64, 'ipsf')
            self.ipsf = self._set(ipsf.astype(np.complex64), self.ipsf)
            self.ipsf_state = 1
            
        # load or replace energy spectrum. energy spectrum must be an array
        # with shape (2, N), where N is the number of sampling points in the
        # spectrum. format of spectrum is [(energies),(weights)]
        if spectrum != None:
            
            assert isinstance(spectrum, np.ndarray)
            assert spectrum.ndim == 2
            assert spectrum.shape[0] == 2
            
            energies = spectrum[0]
            weights = spectrum[1]
            
            # turn the energies into rescale values
            center_e = energies[weights.argmax()]
            rescales = energies/center_e
            n_energy = energies.shape[0]
            
            # allocate memory for the rescaling factors and their
            # spectral weights
            self.rescales = self._allocate(n_energy, np.float32, 'rescales')
            self.sweights = self._allocate(n_energy, np.float32, 'sweights')
            self.rescales = self._set(rescales.astype(np.float32), \
                                      self.rescales)
            self.sweights = self._set(weights.astype(np.float32), \
                                      self.sweights)
            self.N_spctrm = np.int32(n_energy)
            self.spectrum_state = 2

        #### load information with dependencies
        if self.modulus_state == 2:
            
            if self.ipsf_state > 0:
                assert self.ipsf.shape == self.shape
                self.ipsf_state = 2

            # if a support is loaded, ensure it is the correct size
            if self.support_state > 0:
                
                # this means we have a self.modulus and a self.support
                if self.support.shape != self.modulus.shape:
                    
                    assert self.rows <= self.modulus.shape[0]
                    assert self.cols <= self.modulus.shape[1]
                    
                    resupport = self.get(self.support)
                    resupport = resupport[self.r0:self.r0+self.rows,\
                                          self.c0:self.c0+self.cols]
                    new = np.zeros(self.shape, np.float32)
                    new[:self.rows, :self.cols] = resupport
                    
                    # make the new bounding region
                    bounds = masking.bounding_box(new, force_to_square=True, \
                                                  pad=4)
                    self.r0, self.rows = bounds[0], int(bounds[1]-bounds[0])
                    self.c0, self.cols = bounds[2], int(bounds[3]-bounds[2])

                    self.support = self._allocate(self.shape, np.float32, \
                                                  name='support')
                    self.support = self._set(new, self.support)
                
                assert self.support.shape == self.shape
                self.support_state = 2
                
                # allocate memory for savebuffer
                if self.buffer_state > 0:
                    shape = (self.numtrials+1, self.rows, self.cols)
                    tmp1 = self.buffer_state == 1
                    tmp2 = self.buffer_state == 2
                    if tmp1 or (tmp2 and self.savebuffer.shape != shape):
                        dt = np.complex64
                        self.savebuffer = self._allocate(shape, dt, 'savebuffer')
                        self.buffer_state = 2
            
        # set the flag which allows iteration.
        tmp1 = self.modulus_state == 2
        tmp2 = self.support_state == 2
        tmp3 = self.buffer_state == 2
        if tmp1 and tmp2 and tmp3:
            self.can_iterate = True

    def optimize_coherence(self, best_estimate, modulus=None, force_cpu=False,
                           load=True, silent=False, iterations=500,
                           method='richardson_lucy', is_fourier=False):
    
        """
        This function presents methods to optimize the estimated coherence
        function from best_estimate. Usually, best_estimate is the set of
        reconstructions using the current estimate of the coherence function.
        The code does the optimization for each frame in best_estimate, then
        takes the average.
        
        Required arguments:
            best_estimate: real-space reconstructed image(s). Must be numpy
            array, 2d or 3d.
            
        Optional arguments:
            load: (default True) If True, load the estimated coherence function
            into self.ipsf, saving user effort.
        
            modulus: fourier modulus of solution. If None, use whatever
            is in self.modulus.
            
            force_cpu: (default False). If True, execute coherence optimization
            on cpu.
            
            silent: (default False). If True, prints some debugging output.
            
            iterations: (default 500). For the richardson_lucy method of
            estimating the coherence, this is the number of iterations to
            use in the deconvolution.
            
            method: (default 'richardson_lucy') Can be either 'richardson_lucy'
            or 'gaussian'. If the former, will estimate the coherence function
            by Richardson-Lucy deconvolution. If the latter, will assume that
            the form of the coherence function is a gaussian, and perform an
            optimization to find the optimal standard deviations.
        
        Returns:
        
            If method == 'richardson_lucy', returns the average estimated
            coherence function (for example to save).
            
            If method == 'gaussian', returns the list of estimated stdevs.
        
        """
        
        import shape

        # check some types
        assert method in ('richardson_lucy', 'gaussian'),\
        "unknown coherence method %s"%method
        
        if method == 'richardson_lucy':
            try:
                iterations = int(iterations)
            except:
                print "couldnt cast iterations %s to int"%iterations

        # check gpu
        old_use_gpu = self.use_gpu
        if force_cpu:
            self.use_gpu = False

        self.counter = 0
        
        print best_estimate.shape
            
        def _allocate_g():
            """ Allocate memory for arrays used in the optimizer"""
            
            self.optimize_ac = self._allocate(s, c, 'optimize_ac')
            self.optimize_ac2 = self._allocate(s, c, 'optimize_ac2')
            self.optimize_m = self._allocate(s, f, 'optimize_m')
            
            # this holds the blurry image
            self.optimize_bl = self._allocate(s, c, 'optimize_bl')
            
            # this holds the blurry image, abs-ed to f
            self.optimize_bl2 = self._allocate(s, f, 'optimize_bl2')
            
            # this holds the difference
            self.optimize_d = self._allocate(s, c, 'optimize_d')
            
            # this holds the gaussian
            self.optimize_g = self._allocate(s, f, 'optimize_g')   
            
            self.optimize_m = self._set(modulus.astype(f), self.optimize_m)
    
            # to make a gaussian, first make coordinate arrays
            rows, cols = np.indices(modulus.shape)-modulus.shape[0]/2.
            rows = np.fft.fftshift(rows)
            cols = np.fft.fftshift(cols)
            optcls = np.zeros((2, best_estimate.shape[0]), f)
            
            return rows, cols, optcls

        def _allocate_rl():
            """ Allocate memory for arrays used in the deconvolver"""
            if self.rl_state != 2:
                self.rl_d = self._allocate(s, c, 'rl_d')
                self.rl_u = self._allocate(s, c, 'rl_u')
                self.rl_p = self._allocate(s, c, 'rl_p')
                self.rl_ph = self._allocate(s, c, 'rl_ph')
                self.rl_sum = self._allocate(s, c, 'rl_sum')
                self.rl_blur = self._allocate(s, c, 'rl_blur')
                self.rl_state = 2
         
            # zero rl_sum, which holds the sum of the deconvolved frames  
            if self.use_gpu:
                self._cl_zero(self.rl_sum)
            else:
                self.rl_sum = np.zeros_like(self.rl_sum)
        
        def _finalize_g():
            """ Having optimized the coherence estimate for each frame,
            form the average to use as a composite estimate """

            ave_cl = np.average(optcls, axis=1)
    
            use_gpu = old_use_gpu
            common.use_gpu = old_use_gpu
        
            if load:
                # reset the ipsf
                gy = np.exp(-rows**2/(2*ave_cl[0]**2))
                gx = np.exp(-cols**2/(2*ave_cl[1]**2))
                self.load(ipsf=(gy*gx)) # set this to false to skip loading
        
            if not silent:
                print "recovered coherence lengths"
                print ave_cl
                print np.std(optcls, axis=1)
                print optcls
                print ""
                
            return optcls
            
        def _finalize_rl():
            """ Finalize RL deconvolution """
            
            # to be power preserving, the ipsf should be 1 at (0,0)?
            ipsf = self.get(self.rl_u)
            ipsf *= 1./abs(ipsf[0, 0])
            
            if load:
                self.load(ipsf=ipsf)
            self.rl_state = 0 # why?
            
            return np.fft.fftshift(ipsf)
        
        def _opt_iter_g(cl):
            """ Optimize the gaussian """
            
            
            self.counter += 1
            
            # make the gaussian
            if self.use_gpu:
                self._kexec('gaussian_f', self.optimize_g, cl[0],
                            cl[1], shape=modulus.shape)
                
            else:
                gy = np.exp(-rows**2/(2*cl[0]**2))
                gx = np.exp(-cols**2/(2*cl[1]**2))
                self.optimize_g = gy*gx
                
            # blur the speckle pattern
            if self.use_gpu:
                self._cl_mult(self.optimize_g, self.optimize_ac,
                              self.optimize_ac2)
                self._fft2(self.optimize_ac2, self.optimize_bl)
                self._cl_abs(self.optimize_bl, self.optimize_bl)
                self._cl_sqrt(self.optimize_bl, self.optimize_bl2)

            else:
                temp = np.abs(np.fft.fft2(self.optimize_g*self.optimize_ac))
                self.optimize_bl2 = np.sqrt(temp)
                
            # calculate the absolute difference and return the sum of the square
            if self.use_gpu:
                
                self._kexec('subtract_f_f', self.optimize_bl2,
                            self.optimize_m, self.optimize_bl2)
                
                self._cl_abs(self.optimize_bl2, self.optimize_bl2,
                             square=True)
                
                x = cla.sum(self.optimize_bl2).get()

            else:
                diff = self.optimize_bl2-self.optimize_m
                x = np.sum(np.abs(diff)**2)
            
            return x
        
        def _opt_iter_rl():
            """
            # implement the rl deconvolution algorithm.
            # explanation of quantities:
            # p is the point spread function (the reconstructed intensity)
            # d is the measured partially coherent intensity
            # u is the estimate of the psf, what we are trying to reconstruct
            """
            
            # convolve u and p. this should give a blurry intensity
            if self.use_gpu:
                self._fft2(self.rl_u, self.rl_blur)
                self._cl_mult(self.rl_blur, self.rl_p, self.rl_blur)
                self._fft2(self.rl_blur, self.rl_blur, inverse=True)
            else:
                self.rl_blur = np.fft.ifft2(np.fft.fft2(self.rl_u)*self.rl_p)
                
            # divide d by the convolution
            if self.use_gpu:
                self._cl_div(self.rl_d, self.rl_blur, self.rl_blur)
            else:
                self.rl_blur = self.rl_d/self.rl_blur

            # convolve the quotient with phat
            if self.use_gpu:
                self._fft2(self.rl_blur, self.rl_blur)
                self._cl_mult(self.rl_blur, self.rl_ph, self.rl_blur)
                self._fft2(self.rl_blur, self.rl_blur, inverse=True)
            else:
                tmp = np.fft.fft2(self.rl_blur)*self.rl_ph
                self.rl_blur = np.fft.ifft2(tmp)
                
            # multiply u and blur to get a new estimate of u
            if self.use_gpu:
                self._cl_mult(self.rl_u, self.rl_blur, self.rl_u)
            else:
                self.rl_u *= self.rl_blur

        def _postprocess_g():
            """ Post process the Gaussian optimization """
            optcls[:, n] = p1
        
        def _postprocess_rl():
            """Post process the RL deconvolution """

            # make rl_u into the ipsf with ans fft
            if self.use_gpu:
                self._fft2(self.rl_u, self.rl_u)
            else:
                self.rl_u = np.fft.fft2(self.rl_u)
                
            # add rl_u to rl_sum
            if self.use_gpu:
                self._cl_add(self.rl_sum, self.rl_u, self.rl_sum)
            else:
                self.rl_sum += self.rl_u.astype(c)
                
            # reset types to single precision from double precision for cpu path
            d = np.float32
            if not self.use_gpu:
                self.rl_u = self.rl_u.astype(d)
                self.rl_p = self.rl_p.astype(d)
                self.rl_blur = self.rl_blur.astype(d)
                
            if not silent:
                print "finished richardson-lucy estimate %s of %s"\
                %(n+1, best_estimate.shape[0])

        def _preprocess_g():
            """ Preprocess the Gaussian optimization """
            
            # load the autocorrelation
            self.optimize_ac = self._set(active.astype(c), self.optimize_ac)
            
            # fft, abs, square, ifft
            if self.use_gpu:
                self._fft2(self.optimize_ac, self.optimize_ac)
                self._cl_abs(self.optimize_ac, self.optimize_ac)
                self._cl_mult(self.optimize_ac, self.optimize_ac, \
                              self.optimize_ac)
                self._fft2(self.optimize_ac, self.optimize_ac, inverse=True)
            else:
                tmp = np.fft.fft2(self.optimize_ac)
                tmp = np.abs(tmp)**2
                tmp = np.fft.ifft2(tmp)
                self.optimize_ac = tmp
        
        def _preprocess_rl():
            """ Preprocess the RL deconvolution
            
            self.rl_p is the "best estimate" in real space. Make its fourier
            intensity.
            # precompute the fourier transform for the convolutions.
            # make modulus into intensity
            """
            
            g = np.fft.fftshift(shape.gaussian(s, (s[0]/4, s[1]/4)))
            self.rl_p = self._set(active.astype(c), self.rl_p)
            m2 = (modulus**2)
            m2 /= m2.sum()
            self.rl_d = self._set(m2.astype(c), self.rl_d)
            self.rl_u = self._set(g.astype(c), self.rl_u)

            if self.use_gpu:
                
                # fourier modulus of best_estimate if best_estimate
                # is real-space
                if not is_fourier:
                    self._fft2(self.rl_p, self.rl_p) 
                    self._cl_abs(self.rl_p, self.rl_p)
                
                # square to make intensity
                self._cl_mult(self.rl_p, self.rl_p, self.rl_p) 
                rlpsum = self.rl_p.get().sum().real
                div = (1./rlpsum).astype(np.float32)
                self._cl_mult(self.rl_p, div, self.rl_p)
                
                # fft, precomputed for convolution
                self._fft2(self.rl_p, self.rl_p) 

                # precompute p-hat
                self._cl_copy(self.rl_p, self.rl_ph)
                self._kexec('flip_f2', self.rl_p, self.rl_ph, shape=s)

                
            else:
                d = np.complex64
                self.rl_p = np.abs(np.fft.fft2(self.rl_p))**2 # intensity
                self.rl_p /= self.rl_p.sum()
                self.rl_p = np.fft.fft2(self.rl_p).astype(d) # precomputed fft
                self.rl_ph = self.rl_p[::-1, ::-1] # phat

        def _start(best_estimate, modulus):
            """ Get the modulus, make the fft plan etc """
            
            # if best_estimate is on gpu, take it off
            try:
                best_estimate.isgpu
                best_estimate = best_estimate.get()
            except AttributeError:
                pass
            
            # if modulus is on gpu, take if off
            if modulus == None:
                modulus = self.modulus
            try:
                modulus.isgpu
                modulus = modulus.get()
            except AttributeError:
                pass
                
            # modulus sets the shape. if best_estimate is a different
            # shape, embed it in an array of the correct shape.
            assert isinstance(modulus, np.ndarray)
            assert modulus.ndim == 2
            assert modulus.shape[0] == modulus.shape[1]
            
            # best estimate needs to be either a 2d or 3d array. if 3d, the
            # optimization will be performed on every frame.
            assert isinstance(best_estimate, np.ndarray)
            assert best_estimate.ndim in (2, 3)
            
            if best_estimate.ndim == 2:
                best_estimate.shape = (1,)+best_estimate.shape
            
            assert best_estimate.shape[1] <= modulus.shape[0]
            assert best_estimate.shape[2] <= modulus.shape[1]
            
            # make the fft plan if one does not exist already
            if self.use_gpu and self.modulus_state != 2:
                from pyfft.cl import Plan
                self.fftplan = Plan(modulus.shape, queue=self.queue)
                
            return best_estimate, modulus
               
        # start is the initial code common to both methods. it gets the data
        # off the gpu (if its there), casts to 3d, puts back on the gpu if
        # necessary, and checks sizes and types.
        estimate, modulus = _start(best_estimate, modulus)
        
        # allocate memory
        s, f, c = modulus.shape, np.float32, np.complex64
        active = np.zeros_like(modulus).astype(c)
        
        if method == 'gaussian':
            rows, cols, optcls = _allocate_g()
            
        if method == 'richardson_lucy':
            _allocate_rl()
        
        # loop over frames; do analysis on each
        for n, frame in enumerate(estimate):
            
            # slice out the active data
            active[:frame.shape[0], :frame.shape[1]] = frame

            if method == 'gaussian':
                
                from scipy.optimize import fmin
                
                # make the autocorrelation of the current data
                _preprocess_g()
            
                # run the optimizer
                p0 = (modulus.shape[0]/2, modulus.shape[1]/2)
                p1 = fmin(_opt_iter_g, p0, disp=False, ftol=.1, xtol=0.1)
                
                # save optimal lengths to do statistics.
                _postprocess_g()
                
            if method == 'richardson_lucy':
                
                # load initial values and preprocess
                _preprocess_rl()
                
                # now run the deconvolver for a set number of iterations
                for opt_i in range(iterations):
                    _opt_iter_rl()
                
                # make rl_u into ipsf with fft; add to sum
                _postprocess_rl()
                    
        self.use_gpu = old_use_gpu
                    
        if method == 'gaussian':
            return _finalize_g()
            
        if method == 'richardson_lucy':
            return _finalize_rl()

    def reconstruct(self, iterations=100, rounds=1, order=None, refine_support=True, refinement_parameters=None):
        """ Wrap the seeding and iterating behind a single method.
        
        If refine_support = True, the support will be refined at the
        end of each round on the basis of the average reconstruction.
        Parameters for support refinement can be passed as a dictionary
        to kwarg refinement_parameters. Allowed values in the dictionary
        are:
            1. local_threshold
            2. global_threshold
            3. blur_sigma
            4. kill_weakest
        Please consult the documentation for refine_support in the phasing
        module for information regarding these parameters. If parameters
        are not given, sensible defaults are used.
        """
        
        for r in rounds:
        
            for trial in range(self.trials):
                self.seed()
                self.iterate(iterations, silent=100)
                
            # once the trials are complete, get the savebuffer into
            # guaranteed cpu space, align the global phase, and form
            # the average over trials
            save_buffer = self.get(self.savebuffer)
            save_buffer = align_global_phase(save_buffer)
            save_buffer[-1] = np.mean(save_buffer, axis=0)
            
            self.reconstruction = save_buffer[-1]
            
            # if we are refining the support, do so on the basis
            # of the average reconstruction
            if refine_support:
                
                assert isinstance(refinement_parameters, (type(None), dict))
                
                rp = {'local_threshold':0.08, 'blur_sigma':3, 'global_threshold':0.0,
                      'kill_weakest':False}
                
                if refinement_parameters != None:
                    rp.update(refinement_parameters)
                
                refined = refine_support(
                    self.support[self.r0:self.r0+self.rows,self.c0:self.c0+self.cols],
                    save_buffer[-1],
                    blur=rp['blur_sigma'],
                    local_threshold=rp['local_threshold'],
                    global_threshold=rp['global_threshold'])[0]
            
                new_support = np.zeros(self.shape, np.float32)
                new_support[0:refined.shape[0], 0:refined.shape[1]] = refined

                # load the refined support
                self.load(support=new_support)

    def seed(self, supplied=None):
        """ Replaces self.psi_in with random numbers. Use this method to
        restart the simulation without having to copy a whole bunch of
        extra data like the support and the speckle modulus.
        
        arguments:
            supplied - (optional) can set the seed with a precomputed array.
        """
        
        assert self.N != None, \
        "cannot seed without N being set. load data first."
        
        if supplied == None:
            supplied = np.random.rand(self.N, self.N)
            supplied += 1j*np.random.rand(self.N, self.N)
            
        if supplied != None:
            # if a guess is supplied, first make it the correct size
            # through embedding (if necessary), then by rolling to
            # align with the support
            s = supplied
            assert s.ndim == 2
            assert s.shape[0] <= self.N
            assert s.shape[1] <= self.N

            if s.shape != (self.N, self.N):
                s2 = np.zeros(self.support.shape, np.complex64)
                s2[:s.shape[0], :s.shape[1]] = s.astype(np.complex64)
                s = s2
                
            from conditioning import align_frames
            supplied = align_frames(s, align_to=self.get(self.support), \
                                    use_mag_only=True)

        self.psi_in = self._set(supplied.astype(np.complex64), self.psi_in)
        
        if self.use_gpu:
            self._cl_mult(self.psi_in, self.support, self.psi_in)
        else:
            self.psi_in *= self.support

    def status(self):
        """ Print the current data status of the reconstruction object """
        
        ro = {2:'complete', 1:'incomplete', 0:'nothing'}
        
        print "states:"
        print "  modulus %s %s"%(ro[self.modulus_state], self.shape)
        print "  support %s"%ro[self.support_state]
        print "  ipsf    %s"%ro[self.ipsf_state]
        print "  buffer  %s"%ro[self.buffer_state]
        print "  spectrm %s"%ro[self.spectrum_state]
        print "  device  %s"%self.compute_device

    def _convolvef(self, to_convolve, kernel, convolved=None):
        """
        Calculate a convolution when to_convolve must be transformed but
        kernel is already transformed. the multiplication function depends
        on the dtype of kernel.
        """
        
        if self.use_gpu:
            assert to_convolve.dtype == 'complex64',\
            "in _convolvef, input to_convolve has wrong dtype for fftplan"
            
            assert convolved.dtype == 'complex64', \
            "in _convolvef, convolvedput has wrong dtype"
        
            self._fft2(to_convolve, convolved)
            self._cl_mult(convolved, kernel, convolved)
            self._fft2(convolved, convolved, inverse=True)
            
        if not use_gpu:
            return self.ifft2(self.fft2(to_convolve)*kernel)

    def _fft2(self, data_in, data_out, inverse=False):
        """ unified wrapper for fft.
        note that this does not expose the full functionatily of the pyfft
        plan because of assumptions regarding the input (eg, is complex)
        """
        
        if self.use_gpu:
            self.fftplan.execute(data_in=data_in.data, \
                                 data_out=data_out.data, inverse=inverse)
        else:

            if inverse:
                tmp = self.ifft2(data_in)
                data_out = self.ifft2(data_in)
            else:
                tmp = self.ifft2(data_in)
                data_out = self.fft2(data_in)
            data_out = data_out.astype(np.complex64)
                
        return data_out

    def _iteration(self, algorithm, beta=0.8, iteration=None, debug=False):
        """ Do a single iteration of a phase retrieval algorithm.
        
        Arguments:
            algorithm: Can be any of the following
                er   - error reduction
                hio  - hybrid input/output
                raar - relaxed averaged alternating reflectors
                sf   - solvent flipping
                dm   - difference map (uses elser's feedback recommendations)

            beta: For the algorithms with a feedback parameter (hio, raar, dm)
                this is that parameter. The default value is 0.8
                
            --- parameters for developer use ---
            iteration - iteration number
            debug     - True or False; if True, will save some output to help
                        with bugs
                """
                
        # first, build some helper functions which abstract the cpu/gpu issue
        def _build_divisor():
            """ Build the fourier divisor used in satisfying the fourier
            constraint """
            
            psi = self.psi_fourier
            div = self.fourier_div
            
            # gpu code path
            if self.use_gpu:
                
                # convert to modulus
                self._cl_abs(psi, div)
                
                # if ipsf exists, make the modulus blurry through
                # convolution 
                if self.ipsf_state == 2:
                    
                    psf = self.ipsf
                    
                    # square
                    self._cl_mult(div, div, div)
                
                    # convolve. abs fixes leakage into imag component
                    self._convolvef(div, psf, div)
                    self._cl_abs(div, div)

                    # sqrt
                    self._cl_sqrt(div, div)
                    
                # if an energy spectrum has been supplied, make blurry
                # through a dilatory convolution (expensive!)
                if self.spectrum_state == 2:
                    
                    tmp = self.fourier_tmp
                    
                    # square
                    self._cl_mult(div, div, div)
                    
                    # do the radial rescale, an expensive calculation. this
                    # should be kept in a temp buffer then copied in order
                    # to prevent a race condition.
                    self._kexec('radial_rescale', div, self.rescales, \
                                self.sweights, self.N_spctrm, tmp, \
                                shape=self.shape)
                    self._cl_copy(tmp, div)
                    
                    # sqrt
                    self._cl_sqrt(div, div)
            
            # cpu code path
            else:
                
                if HAVE_NUMEXPR:
                    if self.ipsf_state != 2:
                        div = numexpr.evaluate("abs(psi)")
                    
                    # blur through convolution
                    if self.ipsf_state == 2:
                        psf = self.ipsf
                        div = numexpr.evaluate("abs(div)**2")
                        div2 = self._fft2(div, div)
                        div = numexpr.evaluate("div*psf")
                        div = self._fft2(div, div, inverse=True)
                        div = numexpr.evaluate("sqrt(abs(div))")
                        
                else:
                    div = np.abs(psi)
                    
                    if self.ipsf_state == 2:
                        psf = self.ipsf
                        div = div**2
                        div = self._fft2(div, div)
                        div = div*psf
                        div = self._fft2(div, div, inverse=True)
                        div = np.sqrt(np.abs(div))
                    
            return div

        def _fourier_constraint(data, out):
            """ Enforce the Fourier constraint (modulus replacement) """
            s = np.fft.fftshift
            
            # 1. fourier transform the data. store in psi_fourier
            self.psi_fourier = self._fft2(data, self.psi_fourier)
            if debug:
                io.save('i%s psi_fourier_1.fits'%iteration, \
                        s(self.get(self.psi_fourier)), components='polar')

            # 2. from the data in psi_fourier, build the divisor. partial
            # coherence correction is enabled by
            # 1. supplying an estimate of ipsf (transverse coherence)
            # 2. supplying an estimate of the spectrum (longitudinal coherence).
            # currently the longitudinal correction runs only on the
            # gpu codepath.
            self.fourier_div = _build_divisor()
            if debug:
                io.save('i%s fourier_div.fits'%iteration, \
                        s(self.get(self.fourier_div)), components='polar')
            
            # 3. execute the magnitude replacement
            if self.use_gpu:
                self._kexec('fourier', self.psi_fourier, self.fourier_div,\
                            self.modulus, self.psi_fourier)
            else:
                m, pf, fd = self.modulus, self.psi_fourier, self.fourier_div
                if HAVE_NUMEXPR:
                    self.psi_fourier = numexpr.evaluate("m*pf/abs(fd)").astype(np.complex64)
                else:
                    self.psi_fourier = m*pf/np.abs(fd)
            if debug:
                io.save('i%s psi_fourier_2.fits'%iteration, \
                        s(self.get(self.psi_fourier)), components='polar')
                
            # 4. inverse fourier transform the new fourier estimate
            out = self._fft2(self.psi_fourier, out, inverse=True)
            if debug:
                io.save('i%s out.fits'%iteration, \
                        s(self.get(self.psi_fourier)), components='polar')
            
            return out
              
        #### define the algorithm functions
        def _hio(psi_in, psi_out, support):
            """# hio algorithm, given psi_in and psi_out"""
            if self.use_gpu:
                self._kexec('hio', np.float32(beta), support, \
                            psi_in, psi_out, psi_in)
            else:
                if HAVE_NUMEXPR:
                    s = "(1-support)*(psi_in-beta*psi_out)+support*psi_out"
                    psi_in = numexpr.evaluate(s)
                else:
                    psi_in = (1-support)*(psi_in-beta*psi_out)+support*psi_out
            return psi_in
            
        def _er(psi_in, psi_out, support):
            """ ER algorithm"""
            if self.use_gpu:
                self._kexec('er', support, psi_out, psi_in)
            else:
                if HAVE_NUMEXPR:
                    psi_in = numexpr.evaluate("support*psi_in")
                else:
                    psi_in = support*psi_in
            return psi_in
        
        def _raar(psi_in, psi_out, support):
            """ RAAR algorithm """
            if self.use_gpu:
                self._kexec('raar', beta, support, psi_in, psi_out, psi_in)
            else:
                if HAVE_NUMEXPR:
                    s = "(1-support)*(beta*psi_in+(1-2*beta)*psi_out)\
                        +support*psi_out"
                    psi_in = numexpr.evaluate(s)
                else:
                    psi_in = (1-support)*(beta*psi_in+(1-2*beta)*psi_out)\
                    +support*psi_out
            return psi_in
        
        def _sf(psi_in, psi_out, support):
            """ SF algorithm"""
            if self.use_gpu:
                self._kexec('sf', support, psi_in, psi_out, psi_in)
            else:
                if HAVE_NUMEXPR:
                    s = "(1-support)*(-1*psi_in)+support*psi_out"
                    psi_in = numexpr.evaluate(s)
                else:
                    psi_in = (1-support)*(-1*psi_in)+support*psi_out
            return psi_in
        
        def _dm(psi_in, psi_out, support):
            """ Difference map algorithm """
            # difference map requires some additional fourier transforms.
            # we have already formed the fourier projection of the estimate,
            # but need to do it again on a modification of the estimate.
            
            # Elser's recommendations
            gamma_m = 1./beta 
            gamma_s = -gamma_m
            
            # make the modified estimate. enforce fourier constraint in-place
            if self.use_gpu: self._kexec('dm1', gamma_m, support, psi_in,\
                                    self.fourier_tmp)
            else:
                if HAVE_NUMEXPR:
                    s = "support*psi_in-gamma_m*(1-support)*psi_in"
                    self.fourier_tmp = numexpr.evaluate(s)
                else:
                    self.fourier_tmp = support*psi_in-gamma_m*(1-support)*psi_in
                    
            self.fourier_tmp = _fourier_constraint(self.fourier_tmp, \
                                                   self.fourier_tmp)
            
            # enforce the real-space constraint
            if self.use_gpu:
                self._kexec('dm2', beta, gamma_s, support, psi_in, psi_out, \
                            self.fourier_tmp, psi_in)
            else:
                if HAVE_NUMEXPR:
                    s1 = "2*psi_out-beta*self.fourier_tmp+\
                         beta*((1+gamma_s)*psi_out-gamma_s*psi_in)"
                    s2 = "psi_in-beta*psi_out"
                    s3 = "support*t1+(1-support)*t2"
                    t1 = numexpr.evaluate(s1)
                    t2 = numexpr.evaluate(s2)
                    psi_in = numexpr.evaluate(s3)
                else:
                    t1 = 2*psi_out-beta*self.fourier_tmp+\
                         beta*((1+gamma_s)*psi_out-gamma_s*psi_in)
                    t2 = psi_in-beta*psi_out
                    psi_in = support*t1+(1-support)*t2
            return psi_in
            
        # check algorithm request
        algos = {'hio':_hio, 'er':_er, 'raar':_raar, 'sf':_sf, 'dm':_dm}
        
        assert algorithm in algos.keys(), \
        "real space enforcement algorithm %s is unknown"%algorithm
        
        # 1. enforce the fourier constraint. this is the same for all
        # algorithms. however, the way the constraint is satisfied depends
        # on coherence information.
        self.psi_out = _fourier_constraint(self.psi_in, self.psi_out).astype(np.complex64)
        
        # 2. enforce the real space constraint. algorithm can be changed
        # based on incoming keyword
        self.psi_in = algos[algorithm](self.psi_in, self.psi_out, self.support).astype(np.complex64)
        
def align_global_phase(data_in):
    """ Phase retrieval is degenerate to a global phase factor. This function
    tries to align the global phase rotation by minimizing the sum of the abs
    of the imag component.
    
    arguments:
        data: 2d or 3d ndarray whose phase is to be aligned. Each frame of data
            is aligned independently.
        
    returns:
        complex ndarray of same shape as data"""
        
    from scipy.optimize import fminbound
    
    # work on a copy
    data = np.copy(data_in)
    
    # check types
    assert isinstance(data, np.ndarray), "data must be array"
    assert data.ndim in (2, 3), "data must be 2d or 3d"
    assert np.iscomplexobj(data), "data must be complex"
    was2d = False

    if data.ndim == 2:
        was2d = True
        data.shape = (1, data.shape[0], data.shape[1])
        
    I = 0+1j
    
    for frame in data:

        fr = frame.real
        fi = frame.imag
        e = lambda p: np.sum(np.abs(fr*sin(p)+fi*cos(p))**2)

        opt = fminbound(e, 0, 2*np.pi, xtol=1e-9, full_output=1)
            
        frame *= np.exp(1j*opt[0])
        
        # minimizing the imaginary component can give a degenerate solution
        # (ie, 0, pi) now check the value of the real.sum() against -real.sum()
        s1 = frame.real.sum()
        s2 = (-1*frame).real.sum()
        if s2 > s1:
            frame *= -1

    if was2d:
        data = data[0]
    
    return data

def prtf(estimates, N=None, silent=True, prtfq=False):
    """ Implements the PRTF which measures the reproducibility of the
    reconstructed phases. On the basis of this function claims
    of the resolution are often made.
    
    Inputs:
        estimates - an iterable set of aligned independent reconstructions
        N - the size of the array each estimate should be embedded in (this
        exists because the phasing library typically does not store the range
        of the image outside the support, as it is all zeros)
        prtfq - (default False) If True, will also unwrap and compute the
        prtf as a function of |q|.
        
    Returns
        prtf: a 2d array of the PRTF at each reciprocal space value
        prtf_q: a 1d array of the PRTF averaged in annuli using wrapping.
        prtf_q is only returned if optional input prtfq=True.
    """
    
    assert isinstance(estimates, (np.ndarray, list, tuple)),\
    "must be iterable"
    
    if isinstance(estimates, np.ndarray):
        assert estimates.ndim == 3, "must be 3d"
        
    if isinstance(estimates, (list, tuple)):
        
        assert len(estimates) > 1, "must be 3d"
    
    shift = np.fft.fftshift
    
    f, r, c = estimates.shape
    if N == None:
        N = max([r, c])
    
    # compute the prtf by averaging the phase of all the trials
    phase_average = np.zeros((N, N), np.complex64)
    
    if not silent:
        print "averaging phases"
    for n in range(f):
        if not silent:
            print "  %s"%n
        fourier = np.fft.fft2(estimates[n, :r, :c])
        phase = fourier/np.abs(fourier)
        phase_average += phase
    prtf_array = shift(np.abs(phase_average/f))
    
    # unwrap and do the angular average
    if prtfq:
        unwrapped = wrapping.unwrap(prtf_array, (0, N/2, (N/2, N/2)))
        prtf_q = np.average(unwrapped, axis=1)
        return prtf_array, prtf_q
    
    if not prtfq:
        return prtf_array
    
def gpu_prtf(gpuinfo, estimates, N=None, prtfq=False):
    """ Try to run the prtf calculation on the GPU.
    Obviously this requires a GPU and enabling libraries (pyopencl, pyfft)
    before it can work. Interface is identical to standard prtf function with
    the exception of requiring the gpuinfo.
    
    N needs to be a power of 2 for pyfft to work.
    
    """
    context, device, queue, platform = gpuinfo

    if not use_gpu:
        print "cant do gpu prtf without gpu libraries"
        exit()

    # necessary imports
    from pyfft.cl import Plan
    
    L = estimates.shape[0]
    
    # make fft plan
    fftplan = Plan((N, N), queue=queue)
        
    # build kernels
    kp = string.join(gpu.__file__.split('/')[:-1], '/')+'/kernels/'
    prtf_accumulate = gpu.build_kernel_file(context, device, \
                                            kp+'phasing_prtf_accumulate2.cl')

    # allocate buffers
    accumulation = cla.zeros(queue, (N, N), np.float32)
    transformtmp = cla.zeros(queue, (N, N), np.complex64)
    
    # put data on gpu
    new = np.zeros((estimates.shape[0], N, N), np.complex64)
    es = estimates.shape
    new[:, :es.shape[1], :es.shape[2]] = estimates.astype(np.complex64)
    frames = cla.to_device(queue, new)

    # fft
    fftplan.execute(data_in=frames.data, data_out=transformtmp.data,
                    batch=L)
    
    # accumulate along the frame axis
    prtf_accumulate.execute(queue, (N, N), None, np.int32(L),
                            transformtmp.data, accumulation.data)
        
    # get the data off the gpu. shift.
    prtf_array = np.fft.fftshift(accumulation.get())
    
    # unwrap and do the angular average
    if prtfq:
        unwrapped = wrapping.unwrap(prtf_array, (0, N/2, (N/2, N/2)))
        prtf_q = np.average(unwrapped, axis=1)
        return prtf_array, prtf_q
    
    if not prtfq:
        return prtf_array

def rftf(estimate, goal_modulus, hot_pixels=False, ipsf=None,
         rftfq=False, scale=False):
    
    """ Calculates the RTRF in coherent imaging which in analogy to
    crystallography attempts to quantify the Fourier error to determine
    the correctness of a reconstructed image. In contrast to the PRTF,
    this function measures deviations from the fourier goal modulus, whereas
    the PRTF measures the reproducibility of the phase.
    
    From Marchesini et al "Phase Aberrations in Diffraction Microscopy"
    arXiv:physics/0510033v2"
    
    Inputs:
        estimate: the averaged reconstruction
        goal_modulus: the fourier modulus of the speckles being reconstructed
        
    Optional inputs:
        hot_pixels: if True, will remove hot pixels from the error function.
        ipsf: If not none, will incorporate the supplied inverse point-spread
            function (ie coherence function) into the rftf by blurring the
            intensity of the average speckle.
        rftfq: If true, will return both the 2d rftf as an array as well as the
            azimuthal average
        scale: If true, will rescale the RFTF from its typical min/max range
            of [1/sqrt(2),1] to [0,1]
        
    Returns:
        rtrf: a 2d array of the RTRF at each reciprocal space value
        rtrf_q : a 1d array of rtrf averaged in annuli using unwrap, but only
            when rftfq=True.
    """
    
    assert isinstance(estimate, np.ndarray), "estimate must be ndarray"
    assert isinstance(goal_modulus, np.ndarray), "goal_modulus must be ndarray"
    
    assert estimate.shape <= goal_modulus.shape,\
    "estimate must be smaller than goal_modulus"
    
    shift = np.fft.fftshift
    
    # form the speckle pattern
    new = np.zeros(goal_modulus.shape, estimate.dtype)
    new[0:estimate.shape[0], 0:estimate.shape[1]] = estimate
    fourier = np.abs(np.fft.fft2(new))
    N = goal_modulus.shape[0]
    
    if ipsf != None:
        assert isinstance(ipsf, np.ndarray)
        assert ipsf.ndim == 2
        assert ipsf.shape == goal_modulus.shape
        
        # make the partially coherent modulus. ipsf should have max at corner.
        fourier = np.sqrt(np.abs(np.fft.ifft2(np.fft.fft2(fourier)*ipsf)))

    # line up with goal_modulus. based on the operation of the phasing
    # library, the goal_modulus will be centered either at (0,0) or (N/2,N/2).
    # aligning using align_frames is a bad idea because the reconstruction
    # might give a bad speckle pattern, throwing off alignment.
    diff1 = np.sum(np.abs(fourier-goal_modulus)) # goal modulus at (0,0)
    diff2 = np.sum(np.abs(shift(fourier)-goal_modulus)) # goal modulus at N/2
    
    if diff1 > diff2:
        fourier = shift(fourier)
        error = (fourier-goal_modulus)**2/goal_modulus**2
        
    if diff1 < diff2:
        error = shift((fourier-goal_modulus)**2/goal_modulus**2)
        
    if hot_pixels:
        from conditioning import remove_hot_pixels
        error = remove_hot_pixels(error)
        
    # calculate the rtrf from the error
    rftf_array = np.sqrt(1./(1+error))
    
    if scale:
        x = 1./np.sqrt(2)
        rftf_array = (rftf_array-x)/(1-x)
    
    if hot_pixels:
        rftf_array = remove_hot_pixels(rftf_array, threshold=1.1)

    if rftfq:
        unwrapped = wrapping.unwrap(rftf_array, (0, N/2, (N/2, N/2)))
        rftf_q = np.average(unwrapped, axis=1)
        return rftf_array, rftf_q
    
    else:
        return rftf_array

def gpu_rftf(gpuinfo, estimates, goal_modulus, ipsf=None, rftfq=False):
    """ Try to run the prtf calculation on the GPU.
    Obviously this requires a GPU and enabling libraries (pyopencl, pyfft)
    before it can work. Interface is identical to standard prtf function with
    the exception of requiring the gpuinfo.
    
    N needs to be a power of 2 for pyfft to work.
    
    """
    context, device, queue, platform = gpuinfo
    
    if not use_gpu:
        print "cant do gpu rftf with gpu libraries"
        exit()

    # necessary imports
    from pyfft.cl import Plan
    
    # check types
    assert isinstance(estimates, np.ndarray)
    assert isinstance(goal_modulus, np.ndarray)
    assert estimates.ndim == 3
    assert goal_modulus.ndim == 2
    assert goal_modulus.shape[0] == goal_modulus.shape[1]
    
    if ipsf != None:
        assert isinstance(ipsf, np.ndarray)
        assert ipsf.shape == goal_modulus.shape
        
    # get the array size from modulus
    N = goal_modulus.shape[0] # array size
    L = estimates.shape[0] # frames

    # make fft plan
    fftplan = Plan((N, N), queue=queue)
        
    # build kernels
    build = lambda f: gpu.build_kernel_file(context, device, kp+f)
    
    kp = string.join(gpu.__file__.split('/')[:-1], '/')+'/kernels/'
    rftf_accumulate = build('phasing_rftf_accumulate.cl')
    abs2 = build('common_abs2_f2_f2.cl')
    abs1 = build('common_abs_f2_f2.cl')
    mult = build('common_multiply_axis_f_f2.cl')
    sqrt = build('common_sqrt_f2.cl')

    # allocate buffers
    accumulation = cla.zeros(queue, (N, N), np.float32)
    transformtmp = cla.zeros(queue, (L, N, N), np.complex64)
    
    # put data on gpu
    new = np.zeros((L, N, N), np.complex64)
    es = estimates.shape
    
    new[:, :es.shape[1], :es.shape[2]] = estimates.astype(np.complex64)
    
    frames = cla.to_device(queue, new)
    if ipsf != None:
        gpu_ipsf = cla.to_device(queue, ipsf.astype(np.float32))
    modulus = cla.to_device(queue, goal_modulus.astype(np.float32))
    
    # fourier transform all the frames
    fftplan.execute(data_in=frames.data, data_out=transformtmp.data, batch=L)
    
    # if we have an ipsf, do the convolution of the abs2
    if ipsf != None:
        abs2.execute(queue, (L*N*N,), None, transformtmp.data,
                     transformtmp.data)
        
        fftplan.execute(data_in=transformtmp.data, data_out=transformtmp.data,
                        batch=L)
        
        mult.execute(queue, (N, N), None, np.int32(L), gpu_ipsf.data,
                     transformtmp.data, transformtmp.data)
        
        fftplan.execute(data_in=transformtmp.data, data_out=transformtmp.data,
                        inverse=True, batch=L)
        
        sqrt.execute(queue, (L*N*N,), None, transformtmp.data,
                     transformtmp.data)
        
    if ipsf == None:
        abs1.execute(queue, (L*N*N,), None, transformtmp.data,
                     transformtmp.data)

    # accumulate across the frame axis
    rftf_accumulate.execute(queue, (N*N,), None, np.int32(L), modulus.data,
                            accumulation.data, transformtmp.data)
    rftf_done = np.fft.fftshift(accumulation.get())
    
    # unwrap and do the angular average
    if rftfq:
        unwrapped = wrapping.unwrap(rftf_done, (0, N/2, (N/2, N/2)))
        rftf_q = np.average(unwrapped, axis=1)
        return rftf_done, rftf_q
    
    else:
        return rftf_done

def center_of_mass_average(imgs):
    """ Given a set of reconstructions, use the center of mass of the magnitude
    to generate an average.
    
    Inputs:
        imgs - a 3d array
        
    Output:
        the average (complex valued) array
    """
    
    assert isinstance(imgs, np.ndarray)
    assert imgs.ndim in (2, 3)
    
    if imgs.ndim == 2:
        was_2d = True
        imgs.shape = (1,)+imgs.shape
    
    rows, cols = np.indices(imgs.shape[1:])
    
    # helper function to compute center of mass
    def _com(frame):
        """ Helper, defines COM """
        f = np.sum(frame)
        return int(np.sum(rows*frame)/f), int(np.sum(cols*frame)/f)
    
    # compute the COM of each frame, then add it to the running total
    total = np.zeros(imgs.shape[1:], imgs.dtype)
    for n, img in enumerate(imgs):
        r, c = _com(np.abs(img))
        if n == 0:
            r0, c0 = r, c
        dr, dc = r-r0, c-c0
        total += np.roll(np.roll(img, -dr, axis=0), -dc, axis=1)
        
    return total
        
def refine_support(support, average_mag, blur=3, local_threshold=.2,
                   global_threshold=0, kill_weakest=False):
    """ Given an average reconstruction and its support, refine the support
    by blurring and thresholding. This is the Marchesini approach (PRB 2003)
    for shrinkwrap.
    
    Inputs:
        support - the current support
        average_mag - the magnitude component of the average reconstruction
        blur - the stdev of the blurring kernel, in pixels
        threshold - the amount of blurred max which is considered the object
        kill_weakest - if True, eliminate the weakest object in the support.
            This is for the initial refine of a multipartite holographic
            support as in the barker code experiment; one of the reference
            guesses may have nothing in it, and should therefore be eliminated.
        
    Output:
        the refined support
        
    """
    
    assert isinstance(support, np.ndarray), "support must be array"
    assert support.ndim == 2, "support must be 2d"
    assert isinstance(average_mag, np.ndarray), "average_mag must be array"
    assert average_mag.ndim == 2, "average_mag must be 2d"
    
    assert isinstance(blur, (float, int)), \
    "blur must be a number (is %s)"%type(blur)
    
    assert isinstance(global_threshold, (float, int)), \
    "global_threshold must be a number (is %s)"%type(global_threshold)
    
    assert isinstance(local_threshold, (float, int)),\
    "local_threshold must be a number (is %s)"%type(local_threshold)
    
    refined = np.zeros_like(support)
    average_mag = np.abs(average_mag) # just to be sure...
    
    import shape
    print "in refine_support, blur %s"%blur
    kernel = np.fft.fftshift(shape.gaussian(support.shape, (blur, blur)))
    kernel *= 1./kernel.sum()
    kernel = np.fft.fft2(kernel)
    blurred = np.fft.ifft2(kernel*np.fft.fft2(average_mag)).real
    
    # find all the places where the blurred image passes
    # the global threshold test
    global_passed = np.where(blurred > blurred.max()*global_threshold, 1, 0)

    # now find all the local parts of the support, and conduct local
    # thresholding on each
    parts = masking.find_all_objects(support)
    part_sums = np.ndarray(len(parts), float)
    
    for n, part in enumerate(parts):
        current = blurred*part
        local_passed = np.where(current > current.max()*local_threshold, 1, 0)
        refined += local_passed
        
        if kill_weakest:
            part_sums[n] = np.sum(average_mag*part)
        
    refined *= global_passed
    
    if kill_weakest:
        weak_part = parts[part_sums.argmin()]
        refined *= 1-weak_part
    
    return refined, blurred

def covar_results(gpuinfo, data, threshold=0.85, mask=None):
    """ A GPU-only method which computes the pair-wise cross-correlations
    of data to do configuration sorting. Basically just a bunch of FFTs."""
    
    def _reshape_data(data_in):
        """ Beat the data into shape. cast it to the correct complex64, then
        change the size to a power of 2 if necessary. """
        
        data = data_in.astype(np.complex64)
        frames, rows, cols = data.shape
    
        ispower2 = lambda num: ((num & (num - 1)) == 0) and num != 0
            
        if rows != cols:
            if rows > cols:
                cols = rows
            if rows < cols:
                rows = cols
        if ispower2(rows):
            N = rows
        else:
            r2 = int(round(np.log2(rows)))+1
            N = 2**(r2)
            new_data = np.zeros((frames, N, N), np.complex64)
            new_data[:, :rows, :cols] = data
            data = new_data
            
        return data, frames, N
    
    def _prep_gpu():
        """ Helper to set up the GPU stuff """
        
        # load gpu libs
        if self.use_gpu:

            context = gpuinfo[0]
            queue = gpuinfo[2]
            device = gpuinfo[1]

            from pyopencl.elementwise import ElementwiseKernel as cl_kernel
            from pyfft.cl import Plan
            kp = gpu.__file__.replace('gpu.pyc', 'kernels/')
            
        else:
            print "phasing.covar_results failed on imports"
            exit()
        
        # allocate memory
        cc = np.zeros((frames, frames), float)
        covars = np.zeros((frames, frames), float)
        
        g = {}
        g['q'] = queue
        g['d1'] = cla.empty(queue, (N, N), np.complex64)
        g['d2'] = cla.empty(queue, (N, N), np.complex64)
        g['corr'] = cla.empty(queue, (N, N), np.float32)
        g['prod'] = cla.empty(queue, (N, N), np.complex64)
        g['data'] = cla.empty(queue, (frames, N, N), np.complex64)
        g['data'].set(data)
        
        # make the gpu kernels
        fft_N = Plan((N, N), queue=queue)
        
        k = {}
        k['conj_mult'] = cl_kernel(context,
                                   "float2 *d1," "float2 *d2,"  "float2 *out",
                                   "out[i] = (float2)(\
                                             d1[i].x*d2[i].x+d1[i].y*d2[i].y,\
                                             d1[i].x*d2[i].y-d1[i].y*d2[i].x)",
                                   "conj_mult")
                    
        k['make_abs'] = cl_kernel(context, "float2 *in," "float *out",
                                  "out[i] = hypot(in[i].x, in[i].y)",
                                  "make_abs")
      
        k['slice'] = gpu.build_kernel_file(context,
                                           device,
                                           kp+'phasing_copy_from_buffer.cl')
        
        return cc, covars, g, k, fft_N
    
    def _calculate_covars():
        # now iterate through the CCs, cross correlating each pair of dfts
        for n in range(frames):
    
            # get the first frame buffered
            k['slice'].execute(g['q'], (N, N), None, g['d1'].data,
                               g['data'].data, np.int32(0), np.int32(0),
                               np.int32(n), np.int32(N))
            
            # find the indices of which frames we need to correlate
            row = mask[n, n:]
            do = np.nonzero(row)[0]+n
    
            for m in do:
                
                # get the second frame buffered
                k['slice'].execute(g['q'], (N, N), None, g['d2'].data,
                                   g['data'].data, np.int32(0), np.int32(0),
                                   np.int32(m), np.int32(N))
                        
                # multiply conj(dft1) and dft2. store in product. inverse
                # transform  product; keep in place. make the magnitude of
                # product in corr. take the max of corr and return it to host.
                k['conj_mult'](g['d1'], g['d2'], g['prod']).wait()
                    
                fft_N.execute(g['prod'].data, inverse=True)
                k['make_abs'].execute(g['prod'], g['corr']).wait()
                    
                max_val = cla.max(g['corr']).get()
                cc[n, m] = max_val
                cc[m, n] = max_val
                
        # now turn the cc values into normalized covars:
        # covar(i,j) = cc(i,j)/sqrt(cc(i,i)*cc(j,j))
        for n in range(frames):
            
            # find the indices of which frames we need to correlate
            row = mask[n, n:]
            do = np.nonzero(row)[0]+n
            
            for m in do:
                covar = cc[n, m]/np.sqrt(cc[n, n]*cc[m, m])
                covars[n, m] = covar
                covars[m, n] = covar
                
        return cc, np.nan_to_num(covars)
    
    # check types
    assert isinstance(gpuinfo, tuple) and len(gpuinfo) == 4
    assert isinstance(data, np.ndarray) and data.ndim == 3
    if mask == None:
        mask = 1.0
    assert isinstance(mask, (np.ndarray, float, np.float32, np.float64))
    
    # upcast to power-of-2 length for gpu ffts
    data, frames, N = _reshape_data(data)
        
    # make the sampling mask. at whichever point mask > 0, we
    # do that pair-wise correlation. (this is done when mask
    # is a float, skipped when mask is an array).
    if not isinstance(mask, np.ndarray):
        new_mask = np.random.rand(frames, frames)
        new_mask = np.where(new_mask < mask, 1, 0)
        mask = new_mask

    # set up gpu
    cc, covars, g, k, fft_N = _prep_gpu()
    
    # precompute the dfts by running fft as a batch. store in-place.
    fft_N.execute(g['data'].data, batch=frames, wait_for_finish=True)
    
    # calculate the pair-wise covariances
    cc, covars = _calculate_covars()
    
    # calculate some stats
    stats = None
    if threshold > 0:
        # count which reconstructions are most like each other
        rows = covars.shape[0]
        stats = np.zeros((rows, 3), float)
        for row in range(rows):
            average = np.average(covars[row])
            passed = np.sum(np.where(covars[row] > threshold, 1, 0))
            stats[row] = row, average, passed
        
    return cc, covars, stats, g['data'].get()