# implement coherent phasing algorithms on cpu or gpu if available.
# it is assumed that if a gpu is available, it is desired.
# the goal of this code is to completely unify the phasing interface,
# which was previously split between cpu and gpu files.

global use_gpu

import numpy as np
import wrapping,masking,gpu,sys
w = sys.stdout.write
common = gpu.common

try:
    import string
    import pyopencl
    import pyopencl.array as cla
    import pyfft
    use_gpu = True
    
except ImportError:
    use_gpu = False
    
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
    
            
    Publicly available methods in this class are:
    
        get() - method to unify retrieving class attributes between cpu and gpu.
        iterate() - run iterations on the current trial
        load() - load data into the class
        optimize_gaussian_coherence() - try to optimize the ipsf assuming
            the coherence function is gaussian
        richardson_lucy() - implements richardson-lucy deconvolution
        richardsona_lucy_clark() - implements richardson-lucy deconvolution 
            specifically for the algorithm found in Clark et al Nat. Comm 2012.
        seed() - make a complex random number to use as the initial guess
        status() - reports the current status of the reconstruction
        
    Probably the richardson_lucy() method should be removed.
    """
    
    def __init__(self,force_cpu=False):
        global use_gpu
        
        # load the gpu if available
        # keep context, device, queue, platform, and kp in the parent namespace (still available to self)
        if use_gpu:
            common.project = 'phasing'
            use_gpu = self.start() 
        if force_cpu: use_gpu = False
        common.use_gpu = use_gpu # tell the methods in common which device we're using

        if use_gpu: self.compute_device = 'gpu'  
        else: self.compute_device = 'cpu'
            
        # state variables. certain of these must be changed from zero for the
        # reconstruction to proceed. these track which data is loaded into the
        # class. 0 = nothing; 1 = partial; 2 = complete.
        self.modulus_state  = 0
        self.support_state  = 0
        self.ipsf_state     = 0
        self.buffer_state   = 0
        self.rl_state       = 0
        self.spectrum_state = 0
        self.can_iterate    = False
        
        self.numtrial = 0
        self.array_dtypes = ('float32','complex64')
        
        self.ints      = (int,np.int8,np.int16,np.int32,np.uint8)
        self.floats    = (float,np.float16,np.float32,np.float64)
        self.float2s   = (complex,np.complex64,np.complex128)
        self.iterables = (list,tuple,np.ndarray)

    def iterate(self,iterations,silent=True):
    
        """ Run iterations. This is the primary method for the class. When the
        iterations terminate, the result is copied to self.save_buffer.
        
        The keyword argument silent (default True) can be set to False, in which
        case every iteration spits out an update, or a number, in which case
        those iterations where iteration%silent == 0 will report.

        """
             
        if not self.can_iterate:
            print "cant iterate before loading support and modulus."
            for n in ('modulus','support','buffer'):
               print "modulus state: "+str(self.modulus_state)
               print "support state: "+str(self.support_state)
               print "buffer  state: "+str(self.buffer_state)
            
        iterations = int(iterations)
       
        for iteration in range(iterations):
            
            # run the algorithm for a single iteration
            if (iteration+1)%100 != 0: self._iteration('hio',iteration=iteration)
            else:                      self._iteration('er')

            # print an update to the terminal so that it doesnt look like a crash
            if silent != True:
                if isinstance(silent,int):
                    if iteration%silent==0:
                        print "  iteration %s"%iteration
                if isinstance(silent,bool):
                    print "  iteration %s"%iteration
            
        # copy the current reconstruction to the save buffer
        if not use_gpu: self.psi_in = self.psi_in.astype(np.complex64)
        if use_gpu: self._kexec('copy_to_buffer', self.savebuffer,self.psi_in,self.c0,self.r0, self.numtrial,self.N,shape=(self.cols,self.rows))
        else: self.savebuffer[self.numtrial] = self.psi_in[self.r0:self.r0+self.rows,self.c0:self.c0+self.cols]

        self.numtrial += 1

    def load(self,modulus=None,support=None,ipsf=None,spectrum=None,numtrials=None):
        
        # check types, sizes, etc
        types = (type(None),np.ndarray)
        assert isinstance(modulus,types), "modulus must be ndarray if supplied"
        assert isinstance(support,types), "support must be ndarray if supplied"
        assert isinstance(ipsf,types),     "psf must be ndarray if supplied"
        
        ### first, do all the loading that has no dependencies
        
        # load the modulus. should be (NxN) array. N should be a power of 2.
        if modulus != None:
            # modulus is the master from which all the other arrays take
            # their required size
            
            assert modulus.ndim == 2
            modulus = modulus.astype(np.float32)
            
            if self.modulus_state == 2: assert modulus.shape == self.shape
            if self.modulus_state != 2:
                assert modulus.shape[0] == modulus.shape[1]
                
                self.N     = modulus.shape[0]
                self.shape = modulus.shape
                self.size  = modulus.size
                
                # allocate memory for the modulus
                self.modulus = self._allocate(self.shape,np.float32,'modulus')
                
                # allocate NxN complex buffers for iterations
                names = ('psi_in','psi_out','psi_fourier','fourier_div','fourier_tmp')
                for n in names: exec("self.%s = self._allocate(self.shape,np.complex64,name='%s')"%(n,n))
                
            # make the fft plan
            if use_gpu:
                from pyfft.cl import Plan
                self.fftplan = Plan((self.N, self.N), queue=self.queue)
                
            # load the modulus
            self.modulus = self._set(modulus,self.modulus)
            self.modulus_state = 2
            
        # load or replace the support. should be a 2d array. size must be smaller
        # than modulus in all dimensions (or equally sized to modulus)
        if support != None:
            
            assert support.ndim == 2
            assert support.shape[0] == support.shape[1]
            support = support.astype(np.float32)
            support *= 1./support.max()

            # load the support into memory. because we only check the size later after both
            # support and modulus have been loaded, we must re-allocate memory every time.
            if self.support_state == 2: self.numtrial = 0
            self.support = self._allocate(support.shape,np.float32,'support')
            self.support = self._set(support,self.support)
            
            # make the new bounding region. do not yet allocate memory for the buffer.
            bounds = masking.bounding_box(support,force_to_square=False,pad=4)
            self.r0, self.rows = bounds[0], int(bounds[1]-bounds[0])
            self.c0, self.cols = bounds[2], int(bounds[3]-bounds[2])

            self.support_state = 1
            
        # get the number of trials into memory. unlimited (limited by memory)
        if numtrials != None:
            self.numtrials = int(numtrials)
            if self.buffer_state == 0: self.buffer_state = 1
            
        # load or replace ipsf. should be same size as modulus.
        if ipsf != None:
            
            assert ipsf.ndim == 2
            assert ipsf.shape[0] == ipsf.shape[1]
            ipsf = ipsf.astype(np.complex64)
            
            if self.ipsf_state == 2:
                assert ipsf.shape == self.ipsf.shape
            self.ipsf = self._allocate(ipsf.shape,np.complex64,'ipsf')
            self.ipsf = self._set(ipsf.astype(np.complex64),self.ipsf)
            self.ipsf_state = 1
            
        # load or replace energy spectrum. energy spectrum must be an array
        # with shape (2, N), where N is the number of sampling points in the
        # spectrum. format of spectrum is [(energies),(weights)]
        if spectrum != None:
            
            assert isinstance(spectrum,np.ndarray)
            assert spectrum.ndim == 2
            assert spectrum.shape[0] == 2
            
            energies = spectrum[0]
            weights  = spectrum[1]
            
            # turn the energies into rescale values
            center_e = energies[weights.argmax()]
            rescales = energies/center_e
            n_energy = energies.shape[0]
            
            # allocate memory for the rescaling factors and their
            # spectral weights
            self.rescales = self._allocate(n_energy,np.float32,'rescales')
            self.sweights = self._allocate(n_energy,np.float32,'sweights')
            self.rescales = self._set(rescales.astype(np.float32),self.rescales) # rescaling factors
            self.sweights = self._set(weights.astype(np.float32) ,self.sweights) # spectral weights
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
                    
                    resupport = self.get(self.support)[self.r0:self.r0+self.rows,self.c0:self.c0+self.cols]
                    new = np.zeros(self.shape,np.float32)
                    new[:self.rows,:self.cols] = resupport
                    
                    # make the new bounding region
                    bounds    = masking.bounding_box(new,force_to_square=True,pad=4)
                    self.r0, self.rows = bounds[0], int(bounds[1]-bounds[0])
                    self.c0, self.cols = bounds[2], int(bounds[3]-bounds[2])

                    self.support = self._allocate(self.shape,np.float32,name='support')
                    self.support = self._set(new,self.support)
                
                assert self.support.shape == self.shape
                self.support_state = 2
                
                # allocate memory for savebuffer
                if self.buffer_state == 1 or (self.buffer_state == 2 and self.savebuffer.shape != (self.numtrials,self.rows,self.rows)):
                    self.savebuffer = self._allocate((self.numtrials+1,self.rows,self.cols),np.complex64,'savebuffer')
                    self.buffer_state = 2
            
        # set the flag which allows iteration.
        if self.modulus_state == 2 and self.support_state == 2 and self.buffer_state == 2: self.can_iterate = True

    def optimize_coherence(self,best_estimate,modulus=None,force_cpu=False,load=True,silent=False):
    
        """ fill this in later with code to unify optimize_gaussian_coherence() and
        optimize_richardson_lucy() """
        
        pass

    def optimize_gaussian_coherence(self, best_estimate, modulus=None, force_cpu=False, load=True, silent=False):
        """ Given a "best estimate" reconstruction, find an ipsf
        function which when used to blur the "best estimate" speckle pattern
        closely matches the known partially coherent speckle pattern.
        
        By default this uses self.modulus as the modulus but a modulus
        can also be supplied through the modulus kwarg. If supplied, it is
        assu  med that it is a modulus instead of an intensity and has been
        properly rolled to corner.
        """
        
        from scipy.optimize import fmin
        
        global use_gpu
        old_use_gpu = use_gpu
        if force_cpu:
            use_gpu = False
            common.use_gpu = False
        
        self.counter = 0

        def _preprocess():
            # fft, abs, square, ifft
            
            if use_gpu:
                
                self._fft2(self.optimize_ac,self.optimize_ac)
                self._cl_abs(self.optimize_ac,self.optimize_ac)
                self._cl_mult(self.optimize_ac,self.optimize_ac,self.optimize_ac)
                self._fft2(self.optimize_ac,self.optimize_ac,inverse=True)
                
            else:
                
                self.optimize_ac = np.fft.ifft2(abs(np.fft.fft2(self.optimize_ac))**2)
                
        def _opt_iter(cl):
            
            self.counter += 1
            
            # make the gaussian
            if use_gpu:
                self._kexec('gaussian_f',self.optimize_g,cl[0],cl[1],shape=modulus.shape)
                
            else:
                gy = np.exp(-rows**2/(2*cl[0]**2))
                gx = np.exp(-cols**2/(2*cl[1]**2))
                self.optimize_g = gy*gx
                
            # blur the speckle pattern
            if use_gpu:
                
                self._cl_mult(self.optimize_g, self.optimize_ac,self.optimize_ac2)
                self._fft2(   self.optimize_ac2,self.optimize_bl)
                self._cl_abs( self.optimize_bl,self.optimize_bl) # now a blurry intensity
                self._cl_sqrt(self.optimize_bl,self.optimize_bl2) # now a blurry modulus

            else:
                temp = abs(np.fft.fft2(self.optimize_g*self.optimize_ac))
                power_out = np.sum(temp)
                self.optimize_bl2 = np.sqrt(temp)
                
            # calculate the absolute difference and return the sum
            if use_gpu:
                
                self._kexec('subtract_f_f',self.optimize_bl2,self.optimize_m,self.optimize_bl2)
                self._cl_abs(     self.optimize_bl2,self.optimize_bl2,square=True)
                x = cla.sum(self.optimize_bl2).get()

            else:
                diff = self.optimize_bl2-self.optimize_m
                x = np.sum(abs(self.optimize_bl2-self.optimize_m)**2)
            
            return x
    
        #### first, get the machine in a known state. this lowers efficiency
        # but sure makes coding easier
    
        # if best_estimate is on gpu, take it off
        try:
            best_estimate.isgpu
            best_estimate = best_estimate.get()
        except AttributeError: pass
        
        # if modulus is on gpu, take if off
        if modulus == None: modulus = self.modulus
        try:
            modulus.isgpu
            modulus = modulus.get()
        except AttributeError: pass
            
        # modulus sets the shape. if best_estimate is a different
        # shape, embed it in an array of the correct shape.
        assert isinstance(modulus,np.ndarray)
        assert modulus.ndim == 2
        assert modulus.shape[0] == modulus.shape[1]
        
        # best estimate needs to be either a 2d or 3d array. if 3d, the
        # optimization will be performed on every frame.
        assert isinstance(best_estimate,np.ndarray)
        assert best_estimate.ndim in (2,3)
        
        was_2d = False
        if best_estimate.ndim == 2:
            was_2d = True
            best_estimate.shape = (1,)+best_estimate.shape
        
        assert best_estimate.shape[1] <= modulus.shape[0]
        assert best_estimate.shape[2] <= modulus.shape[1]
        
        # make the fft plan
        if use_gpu and self.modulus_state != 2:
            from pyfft.cl import Plan
            self.fftplan = Plan(modulus.shape, queue=self.queue)

        # allocate memory for arrays used in the optimizer
        self.optimize_ac  = self._allocate(modulus.shape,np.complex64,'optimize_ac')
        self.optimize_ac2 = self._allocate(modulus.shape,np.complex64,'optimize_ac2')
        self.optimize_m   = self._allocate(modulus.shape,np.float32,'optimize_m')
        self.optimize_bl  = self._allocate(modulus.shape,np.complex64,'optimize_bl') # this holds the blurry image
        self.optimize_bl2 = self._allocate(modulus.shape,np.float32,'optimize_bl2') # this holds the blurry image, abs-ed to f
        self.optimize_m   = self._set(modulus.astype(np.float32),self.optimize_m)
        self.optimize_d   = self._allocate(modulus.shape,np.complex64,'optimize_d') # this holds the difference
        self.optimize_g   = self._allocate(modulus.shape,np.float32,'optimize_g') # this holds the gaussian
    
        # to make a gaussian, we will need to put the coordinate array into namespace.
        # this is the array that gets used to generate the gaussian.
        rows, cols = np.indices(modulus.shape)-modulus.shape[0]/2.
        rows = np.fft.fftshift(rows)
        cols = np.fft.fftshift(cols)
    
        # iterate over the frame axis, optimizing the gaussian coherence for each frame of data.
        active = np.zeros_like(modulus).astype(np.complex64)
        optcls = np.zeros((2,best_estimate.shape[0]),np.float32)
        
        for n, frame in enumerate(best_estimate):
            
            # slice out the active data
            active[:frame.shape[0],:frame.shape[1]] = frame
            
            # make the autocorrelation of the current data
            self.optimize_ac  = self._set(active.astype(np.complex64),self.optimize_ac)
            _preprocess()
        
            # run the optimizer
            p0 = (modulus.shape[0]/2,modulus.shape[1]/2)
            p1 = fmin(_diff,p0,disp=False,ftol=.1,xtol=0.1)
            
            # p1 is the optimized coherence length. save it to do statistics.
            optcls[:,n] = p1
            
        # now that we have done the optimization for each frame, try using
        # the average of the recovered coherence lengths as the
        ave_cl = np.average(optcls,axis=1)

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
            print np.std(optcls,axis=1)
            print optcls
            print ""
        
        return optcls

    def optimize_richardson_lucy(self, best_estimate, modulus=None, force_cpu=False, load=True, silent=False ,iterations=500):
        """ Given a "best estimate" reconstruction, find an ipsf
        function which when used to blur the "best estimate" speckle pattern
        closely matches the known partially coherent speckle pattern.
        
        By default this uses self.modulus as the modulus but a modulus
        can also be supplied through the modulus kwarg. If supplied, it is
        assumed that it is a modulus instead of an intensity and has been
        properly rolled to corner.
        """
        
        from . import shape, io
        
        global use_gpu
        old_use_gpu = use_gpu
        if force_cpu:
            use_gpu = False
            common.use_gpu = False
        
        self.counter = 0

        def _preprocess():
            # self.rl_p is the "best estimate" in real space. make its fourier intensity.
            # precompute the fourier transform for the convolutions.
            # make modulus into intensity

            if use_gpu:
                self._fft2(   self.rl_p, self.rl_p)             # fourier space
                self._cl_abs( self.rl_p, self.rl_p)             # modulus of best_estimate
                self._cl_mult(self.rl_p, self.rl_p, self.rl_p)  # square to make intensity
                rlpsum = self.rl_p.get().sum().real
                div = (1./rlpsum).astype(np.float32)
                self._cl_mult(self.rl_p,div,self.rl_p)
                self._fft2(   self.rl_p, self.rl_p)             # fft, precomputed for convolution
                self._cl_copy(self.rl_p,self.rl_ph)
                self._kexec(  'flip_f2', self.rl_p, self.rl_ph, shape=s) # precompute p-hat

            else:
                self.rl_p  = abs(np.fft.fft2(self.rl_p))**2   # intensity
                self.rl_p /= self.rl_p.sum()
                self.rl_p  = np.fft.fft2(self.rl_p).astype(d) # precomputed fft
                self.rl_ph = self.rl_p[::-1,::-1]             # phat
         
        device = 'gpu'
        if not use_gpu: device = 'cpu'
         
        def _opt_iter():
            # implement the rl deconvolution algorithm.
            # explanation of quantities:
            # p is the point spread function (in this case the reconstructed intensity)
            # d is the measured partially coherent intensity
            # u is the estimate of the psf, what we are trying to reconstruct
            
            sl = [501,]
            
            # convolve u and p. this should give a blurry intensity
            if use_gpu:
                self._fft2(self.rl_u, self.rl_blur)
                self._cl_mult(self.rl_blur,self.rl_p,self.rl_blur)
                self._fft2(self.rl_blur,self.rl_blur,inverse=True)
            else:
                self.rl_blur = np.fft.ifft2(np.fft.fft2(self.rl_u)*self.rl_p)
                
            if opt_i in sl:
                rl_u = self.get(self.rl_u)
                rl_b = self.get(self.rl_blur)
                io.save('%s rl_u %s.fits'%(device,opt_i),rl_u)
                io.save('%s rl_blur %s.fits'%(device,opt_i),rl_b)
                
            # divide d by the convolution
            if use_gpu:
                self._cl_div(self.rl_d,self.rl_blur,self.rl_blur)
            else:
                self.rl_blur = self.rl_d/self.rl_blur
            
            if opt_i in sl:
                rl_d = self.get(self.rl_d)
                rl_b = self.get(self.rl_blur)
                io.save('%s rl_blur %s.fits'%(device,opt_i),rl_b)

            # convolve the quotient with phat
            if use_gpu:
                self._fft2(self.rl_blur,self.rl_blur)
                self._cl_mult(self.rl_blur,self.rl_ph,self.rl_blur)
                self._fft2(self.rl_blur,self.rl_blur,inverse=True)
            else:
                self.rl_blur = np.fft.ifft2(np.fft.fft2(self.rl_blur)*self.rl_ph)
                
            if opt_i in sl:
                rl_d = self.get(self.rl_d)
                rl_b = self.get(self.rl_blur)
                io.save('%s rl_blur3 %s.fits'%(device,opt_i),rl_b)
                
            # multiply u and blur to get a new estimate of u
            if use_gpu: self._cl_mult(self.rl_u,self.rl_blur,self.rl_u)
            else: self.rl_u *= self.rl_blur
            
            if opt_i in sl:
                rl_u = self.get(self.rl_u)
                io.save('%s rl_u2 %s.fits'%(device,opt_i),rl_u)
                #exit()
                
        #### first, get the machine in a known state. this lowers efficiency
        # but sure makes coding easier
    
        # if best_estimate is on gpu, take it off
        try:
            best_estimate.isgpu
            best_estimate = best_estimate.get()
        except AttributeError: pass
        
        # if modulus is on gpu, take if off
        if modulus == None: modulus = self.modulus
        try:
            modulus.isgpu
            modulus = modulus.get()
        except AttributeError: pass
            
        # modulus sets the shape. if best_estimate is a different
        # shape, embed it in an array of the correct shape.
        assert isinstance(modulus,np.ndarray)
        assert modulus.ndim == 2
        assert modulus.shape[0] == modulus.shape[1]
        
        # best estimate needs to be either a 2d or 3d array. if 3d, the
        # optimization will be performed on every frame.
        assert isinstance(best_estimate,np.ndarray)
        assert best_estimate.ndim in (2,3)
        
        was_2d = False
        if best_estimate.ndim == 2:
            was_2d = True
            best_estimate.shape = (1,)+best_estimate.shape
        
        assert best_estimate.shape[1] <= modulus.shape[0]
        assert best_estimate.shape[2] <= modulus.shape[1]
        
        # make the fft plan
        if use_gpu and self.modulus_state != 2:
            from pyfft.cl import Plan
            self.fftplan = Plan(modulus.shape, queue=self.queue)

        ### allocate memory
        s = modulus.shape
        d = np.complex64
        if self.rl_state != 2:
            self.rl_d    = self._allocate(s,d,'rl_d')
            self.rl_u    = self._allocate(s,d,'rl_u')
            self.rl_p    = self._allocate(s,d,'rl_p')
            self.rl_ph   = self._allocate(s,d,'rl_ph')
            self.rl_sum  = self._allocate(s,d,'rl_sum')
            self.rl_blur = self._allocate(s,d,'rl_blur')
            self.rl_state = 2
         
        # zero rl_sum, which holds the sum of the deconvolved frames  
        if use_gpu: self._cl_zero(self.rl_sum)
        else: self.rl_sum = np.zeros_like(self.rl_sum)
        
        # iterate over the frames
        active = np.zeros_like(modulus).astype(np.complex64)
        for m, frame in enumerate(best_estimate):
            
            # slice out the active data
            active[:frame.shape[0],:frame.shape[1]] = frame
        
            # load initial values and preprocess
            g = np.fft.fftshift(shape.gaussian(s,(150,150)))
            self.rl_p = self._set(active.astype(d),self.rl_p)
            m2  = (modulus**2)
            m2 /= m2.sum()
            self.rl_d = self._set(m2.astype(d),self.rl_d)
            self.rl_u = self._set(g.astype(d),self.rl_u)
            _preprocess()
            
            # now run the deconvolver for a set number of iterations
            for opt_i in range(iterations): _opt_iter() # iterate
            
            # make rl_u into the ipsf with ans fft
            if use_gpu: self._fft2(self.rl_u, self.rl_u)
            else: self.rl_u = np.fft.fft2(self.rl_u)
            
            # add rl_u to rl_sum
            if use_gpu: self._cl_add(self.rl_sum,self.rl_u,self.rl_sum)
            else: self.rl_sum += self.rl_u.astype(np.complex64)
            
            # reset types to sp for cpu path
            if not use_gpu:
                self.rl_u    = self.rl_u.astype(d)
                self.rl_p    = self.rl_p.astype(d)
                self.rl_blur = self.rl_blur.astype(d)
            
            if not silent:
                print "finished richardson-lucy estimate %s of %s"%(m,best_estimate.shape[0])
            
        # to be power preserving, the ipsf should be 1 at (0,0)?
        ipsf = self.get(self.rl_u)
        ipsf *= 1./abs(ipsf[0,0])
        
        if load: self.load(ipsf=ipsf)
        self.rl_state = 0
        
        return np.fft.fftshift(ipsf)

    def richardson_lucy_clark(self,best_estimate,modulus=None):

        # attempt to deconvolve the blurry modulus using the intensity
        # from the best_estimate. for now, assume best_estimate is the
        # real-space version
        
        assert use_gpu

        def _preprocess():
            # self.rl_p is the best estimate in real space. make its fourier intensity.
            # precompute the fourier transform for the convolutions.
            # make modulus into intensity
            
            if use_gpu:
                self._fft2(   self.rl_p, self.rl_p)             # fourier space
                self._cl_abs( self.rl_p, self.rl_p)             # modulus
                self._cl_mult(self.rl_p, self.rl_p, self.rl_p)  # intensity
                self._kexec(  'flip_f2', self.rl_p, self.rl_ph, shape=s) # precompute p-hat
                self._fft2(   self.rl_p, self.rl_p)             # fft, precomputed for convolution
                self._fft2(   self.rl_ph,self.rl_ph)            # fft, precomputed for convolution
                
                
            else:
                self.rl_p  = abs(np.fft.fft2(self.rl_p))**2 # intensity
                self.rl_p  = np.fft.fft2(self.rl_p)         # precomputed fft
                self.rl_ph = self.rl_p[::-1,::-1]              # phat
                
        def _rl():
            # implement the rl deconvolution algorithm
            
            # convolve u and p. this should give a blurry intensity
            if use_gpu:
                self._fft2(self.rl_u, self.rl_blur)
                self._cl_mult(self.rl_blur,self.rl_p,self.rl_blur)
                self._fft2(self.rl_blur,self.rl_blur,inverse=True)
            else:
                self.rl_blur = np.fft.ifft2(np.fft.fft2(self.rl_u)*self.rl_p)
                
            # divide d by the convolution
            if use_gpu: self._cl_div(self.rl_d,self.rl_blur,self.rl_blur)
            else: self.rl_blur = self.rl_d/self.rl_blur
            
            
            # convolve the quotient with phat
            if use_gpu:
                self._fft2(self.rl_blur,self.rl_blur)
                self._cl_mult(self.rl_blur,self.rl_ph,self.rl_blur)
                self._fft2(self.rl_blur,self.rl_blur,inverse=True)
            else:
                self.rl_blur = np.fft.ifft2(np.fft.fft2(self.rl_blur)*self.rl_ph)
                
            # multiply u and blur to get a new estimate of u
            if use_gpu: self._cl_mult(self.rl_u,self.rl_blur,self.rl_u)
            else: self.rl_u *= self.rl_blur

        #### first, get the machine in a known state. this lowers efficiency
        # but sure makes coding easier
    
        # if best_estimate is on gpu, take it off
        try:
            best_estimate.isgpu
            best_estimate = best_estimate.get()
        except AttributeError: pass
        
        # if modulus is on gpu, take if off
        if modulus == None: modulus = self.modulus
        try:
            modulus.isgpu
            modulus = modulus.get()
        except AttributeError: pass
            
        # modulus sets the shape. if best_estimate is a different
        # shape, embed it in an array of the correct shape.
        
        assert isinstance(modulus,np.ndarray)
        assert modulus.ndim == 2
        assert modulus.shape[0] == modulus.shape[1]
        s = modulus.shape
        d = np.complex64
        
        assert isinstance(best_estimate,np.ndarray)
        assert best_estimate.ndim == 2
        assert best_estimate.shape[0] <= modulus.shape[0]
        assert best_estimate.shape[1] <= modulus.shape[1]
        
        if best_estimate.shape != modulus.shape:
            new = np.zeros_like(modulus).astype(np.complex64)
            new[:best_estimate.shape[0],:best_estimate.shape[1]] = best_estimate
            best_estimate = new 
            
        if use_gpu and self.modulus_state != 2:
            from pyfft.cl import Plan
            self.fftplan = Plan(modulus.shape, queue=self.queue)
            
        # if the rl buffers have not yet been allocated, do so
        if self.rl_state != 2:
            self.rl_d    = self._allocate(s,d,'rl_d')
            self.rl_u    = self._allocate(s,d,'rl_u')
            self.rl_p    = self._allocate(s,d,'rl_p')
            self.rl_ph   = self._allocate(s,d,'rl_ph')
            self.rl_blur = self._allocate(s,d,'rl_blur')
            self.rl_state = 2
            
        # put best_estimate into p and modulus into d, then make them
        # into the right things for fast calculation with _preprocess()
        # start rl_u as a bunch of ones
        from .. import shape
        modulus = modulus**2
        g = np.fft.fftshift(shape.gaussian(s,(150,150)))
        self.rl_p = self._set(best_estimate.astype(d),self.rl_p)
        self.rl_d = self._set(modulus.astype(d),self.rl_d)
        self.rl_u = self._set(g.astype(d),self.rl_u)
        _preprocess()

        # for this version of richardson lucy, d is the blurry intensity
        # p is the coherent intensity, and u is the coherence function.
        for n in range(500): _rl() # iterate

        # make rl_u into the ipsf with an fft
        if use_gpu: self._fft2(self.rl_u, self.rl_u)
        else: self.rl_u = np.fft.fft2(self.rl_u)
        
        # to be power preserving, the ipsf should be 1 at (0,0)?
        ipsf = self.get(self.rl_u)
        ipsf *= 1./abs(ipsf[0,0])
        
        self.load_data(ipsf=ipsf)
        w("updated ipsf\n")

    def richardson_lucy(self,iterations,blurry,square=False):

        # given self.ipsf, attempt to deconvolve the blurry input

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
        
        assert self.ipsf_state == 2

        def _rl_gpu():
            
            # make ipsfr
            self._kexec('reverse_f2',self.ipsf,self.ipsfr,shape=self.shape)
            
            # make the initial estimate by squaring the modulus
            if square: self._cl_mult(blurry,blurry,self.rl_est)   
            else:      self._cl_copy(blurry,self.rl_est)
            self._cl_copy(self.rl_est,self.rl_pc)
        
            for i in range(iterations):
                if i%100 == 0: print i
                self._convolvef(self.rl_est, self.ipsf,     self.rl_blur1)
                self._cl_div(   self.rl_pc,  self.rl_blur1, self.rl_div  )
                self._convolvef(self.rl_div, self.ipsfr,    self.rl_blur)
                self._cl_mult(  self.rl_est, self.rl_blur, self.rl_est  )
    
        def _rl_cpu():
                
            self.ipsfr = self.ipsf[::-1,::-1]
            self.rl_est = (abs(blurry)**2).astype(np.complex64)
            self.rl_pc  = np.copy(self.rl_est)
            
            for i in range(iterations):
                self.rl_blur1 = _convolvef(self.rl_est,self.ipsf)
                self.rl_div   = self.rl_pc/self.rl_blur1
                self.rl_blur = _convolvef(self.rl_div,self.ipsfr)
                self.rl_est  *= self.rl_blur
                
        if use_gpu: _rl_gpu()
        else:       _rl_cpu()

    def seed(self,supplied=None):
        """ Replaces self.psi_in with random numbers. Use this method to restart
        the simulation without having to copy a whole bunch of extra data like
        the support and the speckle modulus.
        
        arguments:
            supplied - (optional) can set the seed with a precomputed array.
        """
        
        assert self.N != None, "cannot seed without N being set. load data first."
        
        if supplied == None: supplied = np.random.rand(self.N,self.N)+complex(0,1)*np.random.rand(self.N,self.N)
        self.psi_in = self._set(supplied.astype(np.complex64),self.psi_in)
        
        if use_gpu: self._cl_mult(self.psi_in,self.support,self.psi_in)
        else: self.psi_in *= self.support

    def status(self):
        
        ro = {2:'complete',1:'incomplete',0:'nothing'}
        
        print "states:"
        print "  modulus %s"%ro[self.modulus_state]
        print "  support %s"%ro[self.support_state]
        print "  ipsf    %s"%ro[self.ipsf_state]
        print "  buffer  %s"%ro[self.buffer_state]
        print "  spectrm %s"%ro[self.spectrum_state]
        print "  device  %s"%self.compute_device

    def _convolvef(self,to_convolve,kernel,convolved=None):
        # calculate a convolution when to_convolve must be transformed but kernel is already
        # transformed. the multiplication function depends on the dtype of kernel.
        
        if use_gpu:
            assert to_convolve.dtype == 'complex64',  "in _convolvef, input to_convolve has wrong dtype for fftplan"
            assert convolved.dtype == 'complex64', "in _convolvef, convolvedput has wrong dtype"
        
            self._fft2(to_convolve,convolved)
            self._cl_mult(convolved,kernel,convolved)
            self._fft2(convolved,convolved,inverse=True)
            
        if not use_gpu:
            return np.fft.ifft2(np.fft.fft2(to_convolve)*kernel)

    def _fft2(self,data_in,data_out,inverse=False):
        # unified wrapper for fft.
        # note that this does not expose the full functionatily of the pyfft
        # plan because of assumptions regarding the input (eg, is complex)
        
        if use_gpu: self.fftplan.execute(data_in=data_in.data,data_out=data_out.data,inverse=inverse)
        else:
            if inverse: data_out = np.fft.ifft2(data_in)
            else      : data_out = np.fft.fft2(data_in)
        return data_out

    def _iteration(self,algorithm,beta=0.8,iteration=None):
        """ Do a single iteration of a phase retrieval algorithm.
        
        Arguments:
            algorithm: can be either 'hio' or 'er'
            beta: if using the hio algorithm, this sets the feeback parameter.
                default is 0.8"""

        assert algorithm in ('hio','er'), "real space enforcement algorithm %s is unknown"%algorithm

        # first, build some helper functions which abstract the cpu/gpu issue
        def _build_divisor():
            
            psi = self.psi_fourier
            div = self.fourier_div
            
            # step one: convert to modulus
            if use_gpu: self._cl_abs(psi,div)
            else      : div = abs(psi)
            
            # step two: if a transverse coherence estimate has been supplied
            # through the ipsf, use it to blur the modulus.
            if self.ipsf_state == 2:
                
                psf = self.ipsf
                
                if use_gpu:
                    
                    # square
                    self._cl_mult(div,div,div)
                
                    # convolve
                    self._convolvef(div,psf,div)
                    self._cl_abs(div,div)

                    # sqrt
                    self._cl_sqrt(div,div)
                    
                else:
                    
                    div = psi**2
                    div = np.fft.ifft2(np.fft.fft2(div)*psf)
                    div = np.sqrt(abs(div))
                    
            # step three: if a longitudinal coherence estimate has been supplied
            # through the spectrum, use it to blur the modulus.
            if self.spectrum_state == 2:
                
                if use_gpu:
                    
                    tmp = self.fourier_tmp
                    
                    # square
                    self._cl_mult(div,div,div)
                    
                    # do the radial rescale, an expensive calculation. this should be kept in a temp buffer
                    # then copied in order to prevent a race condition.
                    self._kexec('radial_rescale',div,self.rescales,self.sweights,self.N_spctrm,tmp,shape=self.shape)
                    self._cl_copy(tmp,div)
                    
                    # sqrt
                    self._cl_sqrt(div,div)
                    
                else:
                    print "cpu codepath not supported yet for radial rescale; skipping"
                    
            return div

        def _fourier_constraint(data,div,mod,out):
            if use_gpu: self._kexec('fourier',data,div,mod,out)
            else:            out = mod*data/abs(div)
            return out
                
        def _hio(psi_in,psi_out,support):
            # hio algorithm, given psi_in and psi_out
            if use_gpu: self._kexec('hio',np.float32(beta),support,psi_in,psi_out,psi_in)
            else:            psi_in = (1-support)*(psi_in-beta*psi_out)+support*psi_out
            return psi_in
            
        def _er(psi_in,psi_out,support):
            if use_gpu: self._kexec('er',support,psi_out,psi_in)
            else           : psi_in = support*psi_in
            return psi_in
            
        # now run an iteration of the phasing algorithms
        # 1. fourier transform the data in psi_in, store the result in psi_fourier
        self.psi_fourier = self._fft2(self.psi_in,self.psi_fourier)

        # 2. from the data in psi_fourier, build the divisor. partial coherence
        # correction is enabled by 1. supplying an estimate of ipsf (transverse
        # coherence) 2. supplying an estimate of the spectrum (longitudinal
        # coherence). currently the longitudinal correction runs only on the
        # gpu codepath.
        self.fourier_div = _build_divisor()

        # 3. enforce the fourier constraint
        self.psi_fourier = _fourier_constraint(self.psi_fourier,self.fourier_div,self.modulus,self.psi_fourier)

        # 4. inverse fourier transform the new fourier estimate
        self.psi_out = self._fft2(self.psi_fourier,self.psi_out,inverse=True)
        
        # 5. enforce the real space constraint. algorithm can be changed based on incoming keyword
        if algorithm == 'hio': self.psi_in = _hio(self.psi_in,self.psi_out,self.support)
        if algorithm == 'er':  self.psi_in = _er(self.psi_in,self.psi_out,self.support)

def align_global_phase(data):
    """ Phase retrieval is degenerate to a global phase factor. This function
    tries to align the global phase rotation by minimizing the sum of the abs of
    the imag component.
    
    arguments:
        data: 2d or 3d ndarray whose phase is to be aligned. Each frame of data
            is aligned independently.
        
    returns:
        complex ndarray of same shape as data"""
        
    from scipy.optimize import fminbound
    
    # check types
    assert isinstance(data,np.ndarray), "data must be array"
    assert data.ndim in (2,3), "data must be 2d or 3d"
    assert np.iscomplexobj(data), "data must be complex"
    was2d = False
    
    if data.ndim == 2:
        was2d = True
        data.shape = (1,data.shape[0],data.shape[1])
        
    for frame in data:
        x = frame.ravel()
        e = lambda p: np.sum(abs((x*np.exp(complex(0,1)*p)).imag))
        opt, val, conv, num = fminbound(e,0,2*np.pi,full_output=1)
        #print abs(x.imag).sum(),opt,abs((x*np.exp(complex(0,1)*opt)).imag).sum()
        frame *= np.exp(complex(0,1)*opt)
        
        # minimizing the imaginary component can give a degenerate solution (ie, 0, pi)
        # now check the value of the real.sum() against -real.sum()
        s1 = frame.real.sum()
        s2 = (-1*frame).real.sum()
        if s2 > s1: frame *= -1
    
    if was2d: data = data[0]
    
    return data

def prtf(estimates,N=None):
    """Implements the PRTF which measures the reproducibility of the
    reconstructed phases. On the basis of this function claims
    of the resolution are often made.
    
    Inputs:
        estimates - an iterable set of aligned independent reconstructions
        N - the size of the array each estimate should be embedded in (this
        exists because the phasing library typically does not store the range
        of the image outside the support, as it is all zeros)
        
    Returns
        prtf: a 2d array of the PRTF at each reciprocal space value
        prtf_q: a 1d array of the PRTF averaged in annuli using the wrapping lib
    """
    
    assert isinstance(estimates,(np.ndarray,list,tuple)), "must be iterable"
    if isinstance(estimates,np.ndarray):
        assert estimates.ndim == 3, "must be 3d"
    if isinstance(estimates,(list,tuple)):
        assert len(estimates) > 1, "must be 3d"
    
    shift = np.fft.fftshift
    
    f, r, c       = estimates.shape
    if N == None: N = max([r,c])
    
    # compute the prtf by averaging the phase of all the trials
    phase_average = np.zeros((N,N),complex)
    estimate      = np.zeros((N,N),complex)
    
    print "averaging phases"
    for n in range(f):
        print "  %s"%n
        estimate[:r,:c] = estimates[n]
        fourier = np.fft.fft2(estimate)
        phase = fourier/abs(fourier)
        phase_average += phase
    prtf = shift(abs(phase_average/f))
    
    # unwrap and do the angular average
    import wrapping
    unwrapped = wrapping.unwrap(prtf,(0,N/2,(N/2,N/2)))
    prtf_q    = np.average(unwrapped,axis=1)
    
    return prtf, prtf_q

def rftf(estimate,goal_modulus,hot_pixels=False,ipsf=None):
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
        
    Returns:
        rtrf: a 2d array of the RTRF at each reciprocal space value
        rtrf_q : a 1d array of rtrf averaged in annuli using unwrap
    """
    
    assert isinstance(estimate,np.ndarray), "estimate must be ndarray"
    assert isinstance(goal_modulus,np.ndarray), "goal_modulus must be ndarray"
    assert estimate.shape <= goal_modulus.shape, "estimate must be smaller than goal_modulus"
    
    shift = np.fft.fftshift
    
    # form the speckle pattern
    new = np.zeros(goal_modulus.shape,estimate.dtype)
    new[0:estimate.shape[0],0:estimate.shape[1]] = estimate
    fourier = abs(np.fft.fft2(new))
    N = goal_modulus.shape[0]
    
    if ipsf != None:
        assert isinstance(ipsf,np.ndarray)
        assert ipsf.ndim == 2
        assert ipsf.shape == goal_modulus.shape
        
        # make the partially coherent modulus. ipsf should have max at corner.
        fourier = fourier**2
        ac      = np.fft.fft2(fourier)
        pc      = np.fft.ifft2(ac*ipsf)
        fourier = np.sqrt(pc)

    # line up with goal_modulus. based on the operation of the phasing
    # library, the goal_modulus will be centered either at (0,0) or (N/2,N/2).
    # aligning using align_frames is a bad idea because the reconstruction might
    # give a bad speckle pattern, throwing off alignment.
    diff1 = np.sum(abs(fourier-goal_modulus))          # goal modulus at (0,0)
    diff2 = np.sum(abs(shift(fourier)-goal_modulus))   # goal modulus at N/2
    
    if diff1 > diff2:
        fourier = shift(fourier)
        error   = (fourier-goal_modulus)**2/goal_modulus**2
        
    if diff1 < diff2:
        error = shift((fourier-goal_modulus)**2/goal_modulus**2)
        
    if hot_pixels:
        import conditioning
        error = conditioning.remove_hot_pixels(error)
        
    # calculate the rtrf from the error
    import wrapping
    rtrf      = np.sqrt(1./(1+error))
    if hot_pixels: rtrf = conditioning.remove_hot_pixels(rtrf,threshold=1.1)
    unwrapped = wrapping.unwrap(rtrf,(0,N/2,(N/2,N/2)))
    rtrf_q    = np.average(unwrapped,axis=1)
    
    return rtrf, rtrf_q

def center_of_mass_average(imgs):
    """ Given a set of reconstructions, use the center of mass of the magnitude
    to generate an average.
    
    Inputs:
        imgs - a 3d array
        
    Output:
        the average (complex valued) array
    """
    
    assert isinstance(imgs,np.ndarray)
    assert imgs.ndim == 3
    
    rows, cols = np.indices(imgs.shape[1:])
    
    # helper function to compute center of mass
    def _com(frame):
        f = np.sum(frame)
        return int(np.sum(rows*frame)/f), int(np.sum(cols*frame)/f)
    
    # compute the COM of each frame, then add it to the running total
    total = np.zeros(imgs.shape[1:],imgs.dtype)
    for n,img in enumerate(imgs):
        r, c = _com(abs(img))
        if n == 0:
            r0, c0 = _com(abs(img))
        dr, dc = r-r0, c-c0
        total += np.roll(np.roll(img,-dr,axis=0),-dc,axis=1)
        
    return total
        
def refine_support(support,average_mag,blur=3,local_threshold=.2,global_threshold=0,kill_weakest=False):
    """ Given an average reconstruction and its support, refine the support
    by blurring and thresholding. This is the Marchesini approach (PRB 2003)
    for shrinkwrap.
    
    Inputs:
        support - the current support
        average_mag - the magnitude component of the average reconstruction
        blur - the stdev of the blurring kernel, in pixels
        threshold - the amount of the blurred max which is considered the object
        kill_weakest - if True, eliminate the weakest object in the support.
            This is for the initial refine of a multipartite holographic support
            as in the barker code experiment; one of the reference guesses may
            have nothing in it, and should therefore be eliminated.
        
    Output:
        the refined support
        
    """
    
    assert isinstance(support,np.ndarray),        "support must be array"
    assert support.ndim == 2,                        "support must be 2d"
    assert isinstance(average_mag,np.ndarray),    "average_mag must be array"
    assert average_mag.ndim == 2,                    "average_mag must be 2d"
    assert isinstance(blur,(float,int)),             "blur must be a number (is %s)"%type(blur)
    assert isinstance(global_threshold,(float,int)), "global_threshold must be a number (is %s)"%type(global_threshold)
    assert isinstance(local_threshold,(float,int)),  "lobal_threshold must be a number (is %s)"%type(local_threshold)
    
    refined     = np.zeros_like(support)
    average_mag = abs(average_mag) # just to be sure...
    
    import shape,masking
    kernel  = np.fft.fftshift(shape.gaussian(support.shape,(blur,blur)))
    kernel *= 1./kernel.sum()
    kernel  = np.fft.fft2(kernel)
    blurred = np.fft.ifft2(kernel*np.fft.fft2(average_mag)).real
    
    # find all the places where the blurred image passes the global threshold test
    global_passed = np.where(blurred > blurred.max()*global_threshold,1,0)
    
    # now find all the local parts of the support, and conduct local thresholding on each
    parts = masking.find_all_objects(support)
    part_sums = np.ndarray(len(parts),float)

    for n,part in enumerate(parts):
        current      = blurred*part
        local_passed = np.where(current > current.max()*local_threshold, 1, 0)
        refined     += local_passed
        
        if kill_weakest: part_sums[n] = np.sum(average_mag*part)
        
    refined *= global_passed
    
    if kill_weakest:
        weak_part = parts[part_sums.argmin()]
        refined  *= 1-weak_part
    
    return refined,blurred

def covar_results(gpuinfo,data,threshold=0.85,mask=None):
    """ A GPU-only method which computes the pair-wise cross-correlations
    of data to do configuration sorting. Basically just a bunch of FFTs."""
    
    # check types
    assert isinstance(gpuinfo,tuple) and len(gpuinfo) == 4
    assert isinstance(data,np.ndarray) and data.ndim == 3
    if mask == None: mask = 1.0
    assert isinstance(mask,(np.ndarray,float,np.float32,np.float64))

    # load gpu libs
    try:
        context,device,queue,platform = gpuinfo
        import sys
        import pyopencl.array as cla
        from pyopencl.elementwise import ElementwiseKernel as cl_kernel
        from pyfft.cl import Plan
        from . import gpu
        kp = gpu.__file__.replace('gpu.pyc','kernels/')
    except:
        "phasing.covar_results failed on imports"
        exit()

    # beat the data into shape. cast it to the correct sp complex, then
    # change the size to a power of 2 if necessary.
    data = data.astype(np.complex64)
    frames, rows, cols = data.shape

    ispower2 = lambda num: ((num & (num - 1)) == 0) and num != 0
    if rows != cols:
        if rows > cols: cols = rows
        if rows < cols: rows = cols
    if ispower2(rows):
        N = rows
    else:
        r2 = int(round(np.log2(rows)))+1
        N = 2**(r2)
        new_data = np.zeros((frames,N,N),np.complex64)
        new_data[:,:rows,:cols] = data
        data = new_data
        
    # make the sampling mask. at whichever point mask > 0, we
    # do that pair-wise correlation.
    if isinstance(mask,np.ndarray): pass
    else:
        new_mask = np.random.rand(frames,frames)
        new_mask = np.where(new_mask < mask,1,0)
        mask = new_mask
    
    # set up buffers
    cc       = np.zeros((frames,frames),float) # in host memory
    covars   = np.zeros((frames,frames),float) # in host memory
    gpu_data = cla.empty(queue, (frames,N,N), np.complex64)
    dft1     = cla.empty(queue, (N,N), np.complex64) # hold first dft to crosscorrelate
    dft2     = cla.empty(queue, (N,N), np.complex64) # hold second dft to crosscorrelate
    product  = cla.empty(queue, (N,N), np.complex64) # hold the product of conj(dft1)*dft2
    corr     = cla.empty(queue, (N,N), np.float32)   # hold the abs of idft(product)
        
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
                
    slice_covar = gpu.build_kernel_file(context, device, kp+'phasing_copy_from_buffer.cl')
    
    # put the data on the gpu
    gpu_data.set(data)
    
    # precompute the dfts by running fft_interleaved as a batch. store in-place.
    print "precomputing ffts"
    fft_N.execute(gpu_data.data,batch=frames)
    print "done"
    
    # now iterate through the CCs, cross correlating each pair of dfts
    iter = 0
    total = frames**2/2
    for n in range(frames):
        print n

        # get the first frame buffered
        slice_covar.execute(queue,(N,N),dft1.data,gpu_data.data,np.int32(0),np.int32(0),np.int32(n),np.int32(N))

        for m in range(frames-n):
            m += n
        
            if mask[n,m] > 1e-6:
                
                # get the second frame buffered
                slice_covar.execute(queue,(N,N),dft2.data,gpu_data.data,np.int32(0),np.int32(0),np.int32(m),np.int32(N))
                    
                # multiply conj(dft1) and dft2. store in product. inverse transform
                # product; keep in place. make the magnitude of product in corr. take
                # the max of corr and return it to host.
                conj_mult(dft1,dft2,product)
                
                fft_N.execute(product.data,inverse=True)
                make_abs(product,corr)
                max_val = cla.max(corr).get()
                cc[n,m] = max_val
                cc[m,n] = max_val
                print max_val

            iter += 1

    # now turn the cc values into normalized covars:
    # covar(i,j) = cc(i,j)/sqrt(cc(i,i)*cc(j,j))
    for n in range(frames):
        for m in range(frames-n):
            m += n
            if mask[n,m] > 1e-6:
                covar = cc[n,m]/np.sqrt(cc[n,n]*cc[m,m])
                covars[n,m] = covar
                covars[m,n] = covar
            
    covars = np.nan_to_num(covars)
    
    stats = None
            
    if threshold > 0:
        # count which reconstructions are most like each other
        rows, cols = covars.shape
        stats = np.zeros((rows,3),float)
        for row in range(rows):
            average = np.average(covars[row])
            passed  = np.sum(np.where(covars[row] > threshold,1,0))
            stats[row] = row, average, passed
        
    return cc, covars, stats, gpu_data.get()