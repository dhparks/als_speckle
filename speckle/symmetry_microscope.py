# implement the symmetry microscope on cpu or gpu if available.
# it is assumed that if a gpu is available, it is desired.
# the goal of this code is to completely unify the phasing interface,
# which was previously split between cpu and gpu files.


global use_gpu

import numpy as np
import wrapping,masking,gpu,sys,time,warnings
w = sys.stdout.write
common = gpu.common
shift = np.fft.fftshift

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import io

try:
    import string
    import pyopencl
    import pyopencl.array as cla
    from pyfft.cl import Plan as fft_plan
    use_gpu = True
        
except ImportError:
    use_gpu = False
    import symmetries
    
try:
    import numexpr
    have_numexpr = True
except ImportError:
    have_numexpr = False
    
try:
    import pyfftw
    pyfftw.interfaces.cache.enable()
    have_fftw = True
except ImportError:
    have_fftw = False

class microscope(common):
    
    def __init__(self,gpu_info=None,force_cpu=False):
        global use_gpu
        
        # load the gpu if available
        # keep context, device, queue, platform, and kp in the superset namespace.
        if use_gpu:
            common.project = 'sm'
            use_gpu = self.start(gpu_info=gpu_info) 
        if force_cpu:
            use_gpu = False
        common.use_gpu = use_gpu # tell the methods in common which device we're using

        if use_gpu:
            self.compute_device = 'gpu'
            self.correl1 = gpu.build_kernel_file(self.context, self.device, self.kp+'sm_correl_denoms.cl')
            self.correl2 = gpu.build_kernel_file(self.context, self.device, self.kp+'sm_correl_norm.cl')
        else:
            self.compute_device = 'cpu'

        # these attributes change value in the load function. based on what has been
        # loaded, execution may be disallowed (ie, need an sample and illumination function).
        # 0 indicates no information, 1 partial information, 2 complete information
        self.sample_state       = 0
        self.illumination_state = 0
        self.ipsf_state         = 0
        self.unwrap_state       = 0
        self.blocker_state      = 0
        self.can_run_scan       = False
        self.resize_speckles    = False
 
        self.array_dtypes = ('float32','complex64')
        
        # these class variables control analysis flow and which intermediates are pulled
        # in order to return as output. the keywords are the types of output that can be
        # added to the output dictionary. returnables is the output dictionary.
        self.keywords = ('sliced','illuminated','speckle',
                        'blurred','unwrapped','correlated',
                        'spectrum','spectrum_ds','correlate_ds','rspeckle')
        self.returnables = {}
        self.sumspectra = False
        self.despike = True

        # set up timings. this is for optimization and gpu/cpu comparisons.
        # the method for these attributes is timings()
        self.times = {}
        time_keys = ('slice','illum','spckl','resiz','blur ','unwrp',
                     'dcomp','dspik','corrl','mastr','reduc','crrl1',
                     'crrl2','crrl3','dcrrl','fillz')
        self.times = {key:0 for key in time_keys}
        self.counter = 0
        
        # these variables get populated during load but are instantiated here to avoid errors
        self.N1, self.N2, self.ur, self.uR, self.uwp_params = None, None, None, None, None

    def load(self,sample=None,illumination=None,ipsf=None,unwrap=None,returnables=('spectrum_ds',),blocker=None):
        """ Load data into the class namespace. While some data is required to
        run the microscope, required data does not have to be loaded at once.
        For example, load() may be called several times to change the sample
        (see the microscope example script).
        
        Loadable items:
        
            sample -- (required) a numpy array representing the object over which
                the microscope runs. Physically, this should represent the
                transmission function in the Born approximation.
                
            illumination -- (required) a numpy array representing the
                illumination function over the sample. Does not have to be
                commensurate with sample. Must be square.
                
            unwrap -- (required) a 2-tuple of (inner-radius, outer-radius) for
                the unwrapping. NOTE: if ipsf is specified, the outer-radius
                must be fit within the size of ipsf.
                
            ipsf -- (optional) a numpy array representing the coherence
                function, which is the inverse fourier transform of the point-
                spread function hence IPSF. If smaller than illumination,
                the far-field speckle pattern will be downsampled to the size
                of ipsf through linear interpolation. Must be square.
                
            blocker -- (optional) a numpy array representing the blocker.
                Physically, this should be valued between 0 and 1, with 0
                being the region of the detector blocked by the blocker and 1
                being the region of the detector not blocked by the blocker.
                
            returnables -- (optional, various) a list of keywords corresponding
                to what output from the simulation is desired. Some are mainly
                for debugging. Available keywords:
                
                'sliced': the current slice of sample
                'illuminated': the illuminated sample (slice*illumination)
                'speckle': the coherent speckle pattern
                'rspeckle': the downsized speckle pattern (not always available)
                'blurred': the blurred speckle pattern (not always available)
                'unwrapped': the unwrapped speckle pattern
                'correlated': the angular correlation
                'correlate_ds': the de-spiked angular correlation 
                'spectrum': the cosine spectrum of the correlation
                'spectrum_ds': the cosine spectrum of the de-spiked correlation
                
                The default value is 'spectrum_ds'. Results are returned as a
                dictionary accessed by:
                
                microscope.returnables['keyword'] = numpy-array
                
                If using the GPU, specifying more returnable output will slow
                the speed of calculation due to increased transfers from device
                to host memory.
        """
        
        def _load_sample(sample):
            
            assert isinstance(sample,np.ndarray), "sample is %s"%type(sample)
            assert sample.ndim == 2
            sample = sample.astype(np.complex64)
            
            def _need_new(sample):
                a = self.sample_state == 2
                try:    b = self.sample.shape == sample.shape
                except: b = False
                return (a and b) or (not a)
                
            if _need_new(sample): self.sample = self._allocate(sample.shape,np.complex64,name='sample')
            self.sample = self._set(sample,self.sample)
            self.sample_state = 2
                
            self.N0a, self.N0b = self.sample.shape
            
        def _load_illumination(illumination):
            
            def _need_new(illumination):
                a = self.illumination_state == 2
                try:    b = self.illumination.shape != illumination.shape
                except: b = False
                return (not a) or (a and b)

            assert self._is_square_array(illumination)
            
            illumination = illumination.astype(np.complex64)
            self.N1  = illumination.shape[0]
            self.N11 = illumination.size
            s, d = illumination.shape, np.complex64
            
            # see if we need to allocate memory, then set illumination
            if _need_new(illumination): self.illumination = self._allocate(s,d,name='illumination')
            self.illumination = self._set(illumination,self.illumination)
              
            # allocate memory for all the buffers of same size and shape as illumination
            self.sliced      = self._allocate(s,d,name='sliced')
            self.speckles    = self._allocate(s,d,name='speckles')
            self.far_field   = self._allocate(s,d,name='far_field')
            self.illuminated = self._allocate(s,d,name='illuminated')
            
            # make fft plans. when no gpu is present, try to use fftw. when fftw
            # is not present, use numpy.fft
            if use_gpu:
                self.fftplan_speckle = fft_plan((self.N1,self.N1), queue=self.queue)
            else:
                self.ffplan_speckle = None
                if have_fftw:
                    self.fft2  = pyfftw.interfaces.numpy_fft.fft2
                    self.ifft2 = pyfftw.interfaces.numpy_fft.ifft2
                else:
                    self.fft2  = np.fft.fft2
                    self.ifft2 = np.fft.ifft2

            self.illumination_state = 2
            
        def _load_ipsf(ipsf):
            
            assert self._is_square_array(ipsf)
            
            ipsf = ipsf.astype(np.complex64)
            self.N2  = ipsf.shape[0]
            self.N22 = ipsf.size
            s,d,d2 = ipsf.shape, np.complex64, np.float32

            if (self.ipsf_state == 2 and s != ipsf.shape) or (self.ipsf_state != 2):
                self.ipsf = self._allocate(s,d,name='ipsf')
            self.ipsf = self._set(ipsf,self.ipsf)

            self.blurred   = self._allocate(ipsf.shape,d,name='blurred')
            self.blurred_f = self._allocate(ipsf.shape,d2,name='blurred_f')
            self.ipsf_state = 2
            
        def _load_unwrap(unwrap):

            assert isinstance(unwrap,(list,tuple,np.ndarray)), "unwrap must be iterable type"
            assert len(unwrap) == 2, "unwrap must be length 2"
            ur, uR = int(unwrap[0]),int(unwrap[1])
                
            self.ur   = min([ur,uR])
            self.uR   = max([ur,uR])
            self.rows = self.uR-self.ur
            if use_gpu: assert self.rows%16 == 0, "uR-ur must be divisible by 16 for gpu speed reasons"
            
            # allocate memory for cosine decompositions
            s4, d = (self.rows,128), np.complex64
            
            self.spectrum      = self._allocate(s4,d,name='spectrum')
            self.spectrumds    = self._allocate(s4,d,name='spectrumds')
            self.spectrumsum   = self._allocate(s4,d,name='spectrumsum')
            self.spectrumsumds = self._allocate(s4,d,name='spectrumsumds')
            
            self.spectrum_buffer = self._allocate_local(16*16*8,'spectrum_buffer')
            
            self.unwrap_state = 1
            
        def _load_returnables(returnables):    
        
            # check types
            assert isinstance(returnables,(list,tuple)), "returnables must be list or tuple"
            assert all([isinstance(r,str) for r in returnables]), "all elements of returnables must be strings"
        
            self.returnables_list = []
            for r in returnables:
                if r not in self.keywords:
                    sys.stdout.write("requested returnable %s is unrecognized and will be ignored\n"%r)
                else:
                    self.returnables_list.append(r)

        def _make_resize_plan():
            
            s1 = (self.N1,self.N1) # old size, set by illumination
            s2 = (self.N2,self.N2) # new size, set by ipsf
            d  = np.float32
            
            # make the resizing plan
            if use_gpu:
                plan = wrapping.resize_plan(s1,s2,target='gpu')
                self.rpx = self._allocate(s2,d,name='rpx')
                self.rpy = self._allocate(s2,d,name='rpy')
                self.rpx = self._set(plan[0].astype(np.float32),self.rpx)
                self.rpy = self._set(plan[1].astype(np.float32),self.rpy)
            else:
                plan = wrapping.resize_plan(s1,s2,target='cpu')
                self.r_out, self.c_out = plan[:,-1]
                self.rp = plan[:,:-1] # cpu plan
                
            self.resized_speckles = self._allocate(s2,np.complex64,name='resized_speckles')

        def _make_unwrap_plan():
            # make the unified unwrap/resize plan
            
            if self.uR > self.N2/2:
                warnings.warn('\nThe supplied unwrap outer radius larger than the size of the speckle to unwrap.\nSetting outer radius to largest allowable value.\n')
                self.uR == self.N2/2
                self.rows = self.uR-self.uR
                if use_gpu: assert self.rows%16 == 0, "uR-ur must be divisble by 16 for gpu speed reasons"
            
            uplan = wrapping.unwrap_plan(self.ur,self.uR,(0,0),target='gpu') # y,x
            upy   = wrapping.resize(uplan[0],(self.rows,512))
            upx   = wrapping.resize(uplan[1],(self.rows,512))
            upy   = np.mod(upy+self.N2,self.N2-1)
            upx   = np.mod(upx+self.N2,self.N2-1)
            s3, d1, d2 = (self.rows,512), np.complex64, np.float32
            
            if not use_gpu:
                # cpu plan must be a different shape for ndimage map_coordinates
                upy, upx = upy.ravel(), upx.ravel()
                self.up = np.array([np.append(upy,self.uR),np.append(upx,self.ur)])
                
            # put the plan into memory.
            if use_gpu:
                self.upx = self._allocate(s3,d2,name='upx')
                self.upy = self._allocate(s3,d2,name='upy')
                self.upx = self._set(upx.astype(d2),self.upx)
                self.upy = self._set(upy.astype(d2),self.upy)
                
            # allocate memory
            self.unwrapped  = self._allocate(s3,d1,name='unwrapped')
            self.despiked   = self._allocate(s3,d1,name='despiked')
            self.rowaverage = self._allocate((self.rows,),d2,name='rowaverage')
                
            # write down the parameters for this plan so that we don't have to remake it on
            # every re-load of sample or ipsf or illumination
            self.uwp_params = (self.ur,self.uR,self.N2)
            
            # correlation plan for gpu
            if use_gpu: self.fftplan_correls = fft_plan((512,),queue=self.queue)
                
            self.unwrap_state = 2

        #### load the pieces of information which do not require dependencies.
        if ipsf         != None: _load_ipsf(ipsf)
        if sample       != None: _load_sample(sample)
        if unwrap       != None: _load_unwrap(unwrap)
        if returnables  != None: _load_returnables(returnables)
        if illumination != None: _load_illumination(illumination)

        #### now calculate information with multiple dependencies
        
        if self.illumination_state == 2:
            
            if self.ipsf_state != 2:
                # have illumination but no coherence
                self.N2, self.resize_speckles = self.N1, False
                
            if self.ipsf_state == 2:
                
                if self.N1 == self.N2:
                    # have illumination and equally sized coherence
                    self.resize_speckles = False
                    
                if (self.N1 != self.N2 and (ipsf != None or illumination != None)):
                    # have illumination and differently size coherence and at least
                    # one of them is being loaded RIGHT NOW
                    self.resize_speckles = True
                    _make_resize_plan()
                    
                if use_gpu: self.fftplan_blur = fft_plan((self.N2,self.N2),np.complex64,queue=self.queue)
            
            # see if we need a new unwrap plan. we require an unwrap plan before we can run
            # the microscope.
            if self.unwrap_state > 0 and (unwrap != None or illumination != None or ipsf != None) and self.uwp_params != (self.ur,self.uR,self.N2):
                _make_unwrap_plan()

        if self.sample_state == 2 and self.illumination_state == 2 and self.unwrap_state == 2:
            self.can_run_scan = True
     
    def run_on_site(self,y,x):
        
        """ Run the symmetry microscope on the site of the object described by the raster coordinates
        dy, dx. Steps are:
        
        1. Slice the sample
        2. Make the speckle
            2b: If coherence is specified, blur the speckle
        3. Unwrap the speckle
        4. Autocorrelate the unwrapped speckle
        5. Cosine-decompose the autocorrelation (in reality, fft and take even columns)
        
        This function aggregates the more atomistic methods into a single function.
        
        arguments:
            dy, dx: roll coordinates used as follows: np.roll(np.roll(d,r0,axis=0),r1,axis=1)
            components: (optional) decompose into these components
        """
        
        assert self.can_run_scan

        time0 = time.time()
        self._slice_sample(y,x)
        self._make_speckle()
        if self.ipsf_state == 2: self._blur()

        # strictly speaking these could all be one method.
        # segregation is just for logical clarity.
        self._unwrap_speckle()
        self._rotational_correlation()
        self._decompose_spectrum()

        self.counter += 1
        self.times['mastr'] += time.time()-time0

    def status(self):
        """ Report on the status of the microscope vis-a-vis what required
        elements have been loaded. These attributes can all be accessed
        externally (if the right words are known!) . This just gives
        a slightly nicer formatting. """
        
        ro = {0:'no',1:'partial',2:'complete'}
        
        sys.stdout.write("Microscope status\n")
        sys.stdout.write("  device: %s\n"%(self.compute_device))
        sys.stdout.write("  required elements\n")
        sys.stdout.write("    sample: %s\n"%ro[self.sample_state])
        sys.stdout.write("    ilmntn: %s\n"%ro[self.illumination_state])
        sys.stdout.write("    unwrap: %s\n"%ro[self.unwrap_state])
        sys.stdout.write("  can scan? %s\n"%self.can_run_scan)
        sys.stdout.write("  optional elements\n")
        sys.stdout.write("    ipsf:   %s\n"%ro[self.ipsf_state])
        sys.stdout.write("    blockr: %s\n"%ro[self.blocker_state])
        sys.stdout.write("  decisisions\n")
        if self.ipsf_state == 2:
            sys.stdout.write("    * Because ipsf is specified, speckle will be blurred\n")
        if self.resize_speckles:
            sys.stdout.write("    * Because illlumination and ipsf are different sizes,\n")
            sys.stdout.write("      speckle will be resized before blurring.\n")
        
    def timings(self):
        mt = self.times['mastr']
        for key in self.times.keys():
            if key != 'mastr':
                t = self.times[key]
                sys.stdout.write("%s time:  %.3e (%.2f %%)\n"%(key,t,t/mt*100))
        sys.stdout.write("mastr time: %.3e\n"%mt)

    def _is_square_array(self, array):
        # check if an array is square. this gets used a lot in loading.
        a = isinstance(array,np.ndarray)
        b = array.ndim == 2
        c = array.shape[0] == array.shape[1]
        return a and b and c

    def _blur(self):
        
        """ Blur the current speckle pattern.
        1. FFT the speckle pattern.
        2. Multiply by FFT by ipsf
        3. Inverse transform the product
        4. To be safe, take abs of transform
        """

        time0 = time.time()
        
        # select the correct array to blur
        if self.resize_speckles: to_blur = self.resized_speckles
        else:                    to_blur = self.speckles
        
        # perform the convolution
        if use_gpu:
            self.fftplan_blur.execute(data_in=to_blur.data,data_out=self.blurred.data,wait_for_finish=True)
            self._cl_mult(self.blurred,self.ipsf,self.blurred)
            self.fftplan_blur.execute(self.blurred.data,wait_for_finish=True,inverse=True)
            self._cl_abs(self.blurred,self.blurred)
                
        else:
            self.blurred = np.abs(self.ifft2(self.fft2(to_blur)*self.ipsf))
            
        self.times['blur '] += time.time()-time0

        if 'blurred' in self.returnables_list: self.returnables['blurred'] = shift(self.get(self.blurred).real)
        if 'blurred_blocker' in self.returnables_list: self.returnables['blurred_blocker'] = shift(self.get(self.blurred)).real*blocker

    def _decompose_spectrum(self):
        """ Decompose the rotational correlation into a cosine spectrum via fft.
        Also, try to despike the correlation, then decompose the despiked data.
        
        spectrum-with-spike is in self.spectrum
        spectrum-without-spike is in self.spectrumds
        """

        if use_gpu:
            
            entries = [(self.spectrum,self.unwrapped),]
            time7 = time.time()
            
            # copy the correlation
            if 'spectrum_ds' in self.returnables or 'correlated_ds' in self.returnables:
                self._cl_copy(self.unwrapped,self.despiked)
                time0    = time.time()
                self._kexec('despike',self.despiked,self.despiked,np.int32(4),np.float32(5),shape=(self.rows,))
                time1    = time.time()
                entries += (self.spectrumds,self.despiked)
                self.times['dspik'] += time1-time0

            # correlate and pull out the even components.
            for entry in entries:
                time6 = time.time()
                #entry[0].fill(np.complex64(0))
                time5 = time.time()
                self.fftplan_correls.execute(entry[1].data,batch=self.rows,wait_for_finish=True) # fft
                time3 = time.time()
                self._kexec('cosine_reduce',entry[1],entry[0],self.spectrum_buffer,shape=(128,int(self.rows)),local=(16,16))
                time4 = time.time()
                self.times['reduc'] += time4-time3
                self.times['dcrrl'] += time3-time5
                self.times['fillz'] += time5-time6

            if self.sumspectra:
                self.spectrumsum.__add__(self.spectrum)
                if 'spectrum_ds' in self.returnables: self.spectrumsumds.__add__(self.spectrumds)
                
            time2 = time.time()

            self.times['dcomp'] += time2-time7
            

        else:
            # cpu path uses function in symmetries library
            import symmetries
            
            entries = [(self.unwrapped,self.spectrum),]
            
            if 'spectrum_ds' in self.returnables:
                time0         = time.time()
                self.despiked = symmetries.despike(self.unwrapped,width=5)
                time1         = time.time()
                entries      += (self.despiked,self.spetrum_ds)
                self.times['dspik'] += time1-time0
                
            for entry in entries:
                time2    = time.time()
                entry[1] = symmetries.fft_decompose(entry[0])
                time3    = time.time()
                self.times['dcomp'] += time3-time2
        
        if 'spectrum' in self.returnables_list:      self.returnables['spectrum']      = self.get(self.spectrum)
        if 'spectrum_ds' in self.returnables_list:   self.returnables['spectrum_ds']   = self.get(self.spectrumds)
        if 'correlated_ds' in self.returnables_list: self.returnables['correlated_ds'] = self.get(self.despiked)    

    def _fft2(self,data_in,data_out,inverse=False,plan=None):
        # unified wrapper for fft.
        # note that this does not expose the full functionatily of the pyfft
        # plan because of assumptions regarding the input (eg, is complex)
        
        if use_gpu: plan.execute(data_in=data_in.data,data_out=data_out.data,inverse=inverse,wait_for_finish=True)
        else:
            if inverse: return np.fft.ifft2(data_in)
            else      : return np.fft.fft2(data_in)

    def _make_speckle(self):
        """ Create the coherent speckle pattern.
        1. Multiply the illumination and the current sample slice
        2. FFT the product
        3. square-modulus the FFT
        """
        def _illuminate():
            time0 = time.time()
            if use_gpu: self._cl_mult(self.sliced ,self.illumination, self.illuminated)
            else:       self.illuminated = self.sliced*self.illumination
            self.times['illum'] += time.time()-time0

        def _speckles():
            time0 = time.time()
            if use_gpu:
                self._fft2(self.illuminated,self.far_field,plan=self.fftplan_speckle)
                self._cl_abs(self.far_field,self.speckles,square=True)
                #self._cl_mult(self.speckles,self.speckles,self.speckles)
            else:
                far_field = self.fft2(self.illuminated)
                if have_numexpr: self.speckles = numexpr.evaluate('abs(far_field)**2')
                else:            self.speckles = np.abs(self.far_field)**2
            self.times['spckl'] += time.time()-time0
                
        def _resize():
            time0 = time.time()
            if use_gpu: self._cl_map2d(self.speckles,self.resized_speckles,self.rpx,self.rpy)
            else: self.resized_speckles = map_coords(self.speckles,self.rp)
            self.times['resiz'] += time.time()-time0
            
        _illuminate()                      # calculate the exit wave by multiplying the illumination
        _speckles()                        # from the exit wave, make the speckle by abs(fft)**2
        if self.resize_speckles: _resize() # resize the speckles if needed
        
        # returnables section
        if 'illuminated' in self.returnables_list: self.returnables['illuminated'] = self.get(self.illuminated)
        if 'speckle'     in self.returnables_list: self.returnables['speckle']     = shift(self.get(self.speckles))  
        if 'rspeckle'    in self.returnables_list and self.resize_speckles: self.returnables['rspeckle'] = shift(self.get(self.resized_speckles))  
        if 'speckle_blocker' in self.returnables_list:  self.returnables['speckle_blocker'] = shift(self.get(self.speckles))*self.blocker
         
    def _rotational_correlation(self):

        """ Do the autocorrelation on unwrapped along the column (angle) axis.
        
        1. Calculate average along rows.
        2. FFT along column axis
        3. square-modulus FFT
        4. Inverse FFT along column axis
        5. Divide each row by previously compute average**2
        
        Input is self.unwrapped
        Output is self.unwrapped (in place)"""
        
        def _row_ave():
            
            if use_gpu:
                
                #self._kexec('correl_denoms',self.unwrapped,self.rowaverage,np.int32(0),shape=(self.rows,))
                #self._kexec('correl_denoms',self.unwrapped,self.rowaverage,np.int32(0),shape=(self.rows,))
                ls = 2
                self.correl1.execute(self.queue,(self.rows,),(ls,),self.unwrapped.data, self.rowaverage.data, pyopencl.LocalMemory(ls*4))
                
            else:
                r = np.average(self.unwrapped,axis=1)
                c = np.ones((512,),np.float32)
                self.rowaverage = np.outer(r,c)
                
        def _corr_norm():

            if use_gpu:
                #self._kexec('correl_norm',self.unwrapped,self.rowaverage,self.unwrapped,shape=(512,self.rows))
                self.correl2.execute(self.queue, (512, self.rows), (16, 16), self.unwrapped.data, self.rowaverage.data, pyopencl.LocalMemory(16*4),self.unwrapped.data)
                
            else:
                self.unwrapped /= (self.rowaverage**2)
                self.unwrapped += -1
        
        corr_time0 = time.time()
        
        t0 = time.time()
        _row_ave() # calculate the wochner normalization
        t1 = time.time()

        # calculate the autocorrelation of the rows using a 1d plan and the batch command in pyfft
        if use_gpu:
            self.fftplan_correls.execute(self.unwrapped.data,batch=self.rows)
            self._cl_abs(self.unwrapped,self.unwrapped,square=True)
            self.fftplan_correls.execute(self.unwrapped.data,batch=self.rows,inverse=True)
            self.queue.finish()

        else:
            t1 = self.fft2(self.unwrapped,axes=(1,))
            if have_numexpr: t2 = numexpr.evaluate('abs(t1)**2')
            else:            t2 = np.abs(t1)**2
            self.unwrapped = self.ifft2(t2,axes=(1,))

        t2 = time.time()
        _corr_norm() # normalize the autocorrelations
        t3 = time.time()
    
        self.times['crrl1'] += t1-t0
        self.times['crrl2'] += t2-t1
        self.times['crrl3'] += t3-t2

        self.times['corrl'] += time.time()-corr_time0
        
        if 'correlated' in self.returnables_list: self.returnables['correlated'] = self.get(self.unwrapped).real
      
    def _slice_sample(self,top_row,left_col):
        """ Copy a subarray out of self.sample and into self.sliced.
        This is the current view of the sample which gets made into a speckle
        pattern.
        
        For GPU code path, sample and illumination can be any relative size, as
        long as illumination is square.
        
        For CPU code path, sample >= illumination in both dimensions.
        
        Boundary conditions on self.sample are cyclic. """
        
        time0 = time.time()
        if use_gpu:
            a, b, c, d = np.int32(self.N0a), np.int32(self.N0b), np.int32(top_row), np.int32(left_col)
            self._kexec('slice_view',self.sample,self.sliced,a,b,c,d,shape=(self.N1,self.N1))
            # look into the built in opencl method for slicing a rectangle
 
        else:
            
            # for the cpu path, we roll and slicing using indexes. this requires
            # that the sample be at least as large as the illumination in both axes
            i = self.illumination.shape[0] # square
            j = i/2
            s = self.sample.shape
            
            rolls = lambda d,r0,r1: np.roll(np.roll(d,-r0,axis=0),-r1,axis=1)
            assert s[0] >= i and s[1] >= i
            
            self.sliced = rolls(self.sample,top_row,left_col)[s[0]/2-j:s[0]/2+j,s[1]/2-j:s[1]/2+j]
            
        if 'sample' in self.returnables_list: self.returnables['sample'] = self.get(self.sample).real
        if 'sliced' in self.returnables_list: self.returnables['sliced'] = self.get(self.sliced).real
            
        self.times['slice'] += time.time()-time0
            
    def _unwrap_speckle(self):
        """ Remap the (blurred) speckle into polar coordinates. The unwrap plan
        simultaneously resizes to 512 pixels in width. """

        # select the correct array to unwrap
        if self.ipsf_state == 2: to_unwrap = self.blurred
        else: to_unwrap = self.speckles

        time0 = time.time()
        if use_gpu:
            self._cl_map2d(to_unwrap,self.unwrapped,self.upx,self.upy,shape=self.upx.shape)
            self._cl_abs(self.unwrapped,self.unwrapped)
        else:
            self.unwrapped = wrapping.unwrap(to_unwrap.real,self.up,interpolation_order=1)
        
        self.times['unwrp'] += time.time()-time0
        
        if 'unwrapped' in self.returnables_list: self.returnables['unwrapped'] = self.get(self.unwrapped).real

        