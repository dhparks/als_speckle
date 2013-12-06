# unified cpu/gpu interface to run

global use_gpu

import numpy as np
from .. import gpu, shape, io
import sys, warnings, time
w = sys.stdout.write
common = gpu.common

shift = np.fft.fftshift

try:
    import string
    import pyopencl
    import pyopencl.array as cla
    from pyfft.cl import Plan as fft_plan
    use_gpu = True
    
except ImportError:
    use_gpu = False
    
try:
    import pyfftw
    pyfftw.interfaces.cache.enable()
    DFT = pyfftw.interfaces.numpy_fft.fft2
    IDFT = pyfftw.interfaces.numpy_fft.ifft2
except ImportError:
    have_fftw = False
    

class generator(common):
    
    """ Copd-type domain generation. Requires a user-generated envelope.
    To run a domain simulation:
    1. Instantiate the class; foo = path.generator()
    2. Load an envelope function "bar"; foo.load(envelope=bar)
    3. Seed the simulation; foo.seed(supplied=optional)
    4. Iterate: foo.iterate()
    5. Repeat 4 until convergence is reached: foo.converged = True
    """
    
    def __init__(self,force_cpu=False,gpu_info=None):
        global use_gpu
        
        # load the gpu if available
        # keep context, device, queue, platform, and kp in the superset namespace.
        if use_gpu:
            common.project = 'domains'
            use_gpu = self.start(gpu_info) 
        if force_cpu: use_gpu = False
        common.use_gpu = use_gpu # tell the methods in common which device we're using

        if use_gpu: self.compute_device = 'gpu'  
        else: self.compute_device = 'cpu'
            
        # these are gpu memory objects associated with making envelopes or with rescaling
        # envelopes
        self.keywords = ('converged','envelope','speckles')
        self.returnables = {}
        
        # these are the acceptable array dtypes for gpu computation.
        # there should be no reason to require double precision floats
        self.array_dtypes = ('float32','complex64')
        
        # most of the arguments that come in need to be made into class variables to be accessible
        # to the other functions during the course of the simulation
        self.alpha     = np.float32(.5)
        self.m_turnon  = 1
        self.converged_at = 0.002
        self.converged = False
        self.cutoff    = 200
        self.spa_max   = 0.1
        self.powerlist = [] # for convergence tracking
        self.goal_m    = 0  # set to 0 just as initial value; can be changed
        self.extrastop = False # for sending a stop signal during a trajectory
        self.only_walls = False
        self.transitions   = 0
        self.sites = 0
        
        self.envelope_state = 0
        self.ggr_state = 0
             
    def check_convergence(self,iteration):
        
        def _diff():
            if use_gpu: self._kexec('absdiff',self.incoming,self.domains_f,self.domain_diff)
            else: self.domain_diff = abs(self.incoming-self.domains_f)

        _diff()
        
        self.power = self._sum(self.domain_diff)/self.N2
        self.powerlist.append(self.power)
        print iteration, self.power
        
        # set the convergence condition
        if self.power <= self.converged_at: self.converged = True
        if iteration > self.cutoff: self.converged = True

        # if converged, prep output
        if self.converged:
            if 'converged' in self.returnables_list: self.returnables['converged'] = self.get(self.domains_f)
            if 'speckles'  in self.returnables_list: self.returnables['speckles']  = shift(np.abs(self.speckles.get()))

    def get(self,something):
        if use_gpu: return something.get()
        else: return something
                
    def iterate(self,iteration):
        
        def _findwalls():
            # find all the sites in the array which constitute domain walls.
            # by definition, a site in a domain wall has a nearest-neighbor
            # with an opposite sign
            if use_gpu:
                #self._kexec('findwalls',self.domains,self.allwalls,self.poswalls,self.negwalls,shape=(self.N,self.N),local=(8,8))
                self.findwalls.execute(self.queue,(self.N,self.N),(16,16),
                                       self.domains.data,
                                       self.allwalls.data,
                                       self.poswalls.data,
                                       self.negwalls.data,
                                       self.localbuffer,
                                       self.scratch.data).wait()
            else:
                rolls = lambda d, r0, r1: np.roll(np.roll(d,r0,axis=0),r1,axis=1)
            
                neighbors = 8
                if neighbors == 4: p = ((1,0),(-1,0),(0,1),(0,-1))
                if neighbors == 8: p = ((1,-1),(1,0),(1,1),(0,-1),(0,1),(-1,-1),(-1,0),(-1,1))
            
                allwalls = np.zeros_like(self.domains)
                for o in p:
                    rolled = rolls(self.domains,o[0],o[1])
                    allwalls += 1-(np.sign(self.domains*rolled)+1)/2
        
                self.allwalls = allwalls.astype(bool).astype(int)
                self.negwalls = self.allwalls*np.where(self.domains < 0,1,0)
                self.poswalls = self.allwalls*np.where(self.domains > 0,1,0)
        
        def _ising():
            # enforce the +- 1 bias
            if use_gpu:
                self._kexec('ising_tanh',np.float32(2),self.domains_f)
                #self._kexec('ising',np.float32(0.5),np.int32(0),self.domains_f)
            else:
                self.domains_f = (1+self.alpha)*self.domains_f-self.alpha*self.domains_f**3
                np.clip(self.domains_f,-1,1,out=self.domains_f)

        def _bound():
            # bound the spins to the range (-1,1)
            if use_gpu: self._kexec('bound',self.domains_f,self.domains_f)
            else: np.clip(self.domains_f,-1,1,out=self.domains_f)
        
        def _update_domains():
            if use_gpu: self._kexec('update',self.domains_f,self.incoming,self.allwalls)
            else: self.domains_f = self.domains_f*w+(1-w)*self.incoming
            
        def _make_available(which):
            # find out which spins are available for changing.
            if use_gpu: self._kexec('mkavailable',self.available,which,self.negpins,self.pospins)
            else: self.available = which*self.negpins*self.pospins
            
        def _promote_spins(x):
            # add an amount x to the spins identified by the mask self.available
            if use_gpu: self._kexec('promote',self.domains_f,self.available,np.float32(x))
            else:
                self.domains_f += self.available*x
                np.clip(self.domains_f,-1,1,out=self.domains_f)
        
        if use_gpu:
            self._cl_copy(self.domains,self.incoming) # basically, just discard the imaginary component
        else: self.incoming = self.domains.real
        
        # rescale the domains. this operates on the class variables so no arguments are passed.
        # self.domains_f stores the rescaled real-valued domains. the rescaled domains are bounded to the range +1 -1.
        self._rescale_speckle(iteration=iteration)
        if use_gpu: self._cl_copy(self.domains,self.domains_f)
        else: self.domains_f = self.domains.real # _f for "float"

        #_findwalls() # now find the domain walls. modifications to the domain pattern due to rescaling only take place in the walls
        _ising() # run the ising bias; includes bounding

        if 'bounded' in self.returnables_list: self.returnables['bounded'] = self.get(self.domains_f)
        
        # if making an ordering island, this is the command that enforces the border condition
        #if use_boundary and n > boundary_turn_on: self.enforce_boundary(self.domains,self.boundary,self.boundary_values)
        
        # so now we have self.incoming (old domains) and self.domains (rescaled domains). we want to use self.walls to enforce changes to the domain pattern
        # from rescaling only within the walls. because updating can change wall location, also refind the walls.
        if self.only_walls:
            _findwalls()
            _update_domains()
        
        if iteration > self.m_turnon:
            
            _findwalls()
            if 'walls'     in self.returnables_list: self.returnables['walls']     = self.get(self.allwalls)
            if 'pos_walls' in self.returnables_list: self.returnables['pos_walls'] = self.get(self.poswalls)
            if 'neg_walls' in self.returnables_list: self.returnables['neg_walls'] = self.get(self.negwalls)

            # now attempt to adjust the net magnetization in real space to achieve the target magnetization.
            net_m    = self._sum(self.domains_f)
            needed_m = self.goal_m-net_m
            if needed_m > 0: which, cfunc, sign = self.negwalls, min, 1
            if needed_m < 0: which, cfunc, sign = self.poswalls, max, -1
                
            _make_available(which)
            sites = self._sum(self.available,d=np.int32)
            amount = cfunc([sign*self.spa_max,needed_m/sites])
                
            _promote_spins(amount) # promote the spins; includes bounding
        
            if 'promoted' in self.returnables_list: self.returnables['promoted'] = self.get(self.domains_f)
            
        # now copy domains_f back to domains for another fft
        if use_gpu: self._cl_copy(self.domains_f,self.domains)
        else: self.domains = np.copy(self.domains_f)
        if 'domains' in self.returnables_list: self.returnables['domains'] = self.get(self.domains_f)
       
    def load(self,envelope=None,goal_growth_rate=None,returnables=('converged','speckles')):
        
        if envelope != None:
            assert isinstance(envelope,np.ndarray), "envelope is %s"%type(sample)
            assert envelope.ndim == 2
            
            s = envelope.shape
            
            # we're assuming that the incoming function is the modulus
            envelope = np.fft.fftshift(envelope.astype(np.float32))
            envelope += -envelope.min()
            envelope *= 1./envelope.max()
            
            if self.envelope_state == 2:
                assert envelope.shape == self.envelope.shape
                self.envelope = self._set(envelope,self.envelope)
            
            if self.envelope_state != 2:
                # with the size of the envelope known, allocate memory
                # for the other objects: fourier_domains, speckles, etc
                
                s, d1, d2, d3 = envelope.shape, np.complex64, np.float32, np.uint8
                self.N  = s[0]
                self.N2 = s[0]**2
                
                d1_names = ('domains','fourier_domains','speckles','ipsf') # complex64
                d2_names = ('envelope','domains_f','speckles_f','incoming','domain_diff','kicker','available','scratch') # float32
                d3_names = ('allwalls','poswalls','negwalls','negpins','pospins') #uint (8) -> uchar
                
                # allocate memory and namespace for the names in the above tuples
                for n in d1_names: exec("self.%s = self._allocate(s,d1,name='%s')"%(n,n))
                for n in d2_names: exec("self.%s = self._allocate(s,d2,name='%s')"%(n,n))
                for n in d3_names: exec("self.%s = self._allocate(s,d3,name='%s')"%(n,n))
                
                self.localbuffer = pyopencl.LocalMemory(18*18*4)
                
                # build the kernels which use local memory
                if use_gpu:
                    self.findwalls = gpu.build_kernel_file(self.context, self.device, self.kp+'domains_findwalls.cl')
            
                # these don't actually have anything to do with the envelope but this is a
                # convenient spot to put the allocations
                names = ('m0_1','m0_2')
                for n in names: exec("self.%s = self._allocate((1,),np.complex64,name='%s')"%(n,n))
                
                # move the envelope to device
                self.envelope = self._set(envelope,self.envelope)
                
                # make a gaussian on host and set it on device
                l = self.N/(2*np.pi*3)
                k = np.fft.fftshift(shape.gaussian(s,(l,l)))
                self.ipsf = self._set(k.astype(np.complex64),self.ipsf)
                
                # if using the gpu, make the fft plan
                if use_gpu: self.fftplan = fft_plan(s, queue=self.queue)
                
                # set the pinnings to one, indicating no initial values are to be preserved
                ones = np.ones(s,d3)
                self.negpins = self._set(ones,self.negpins)
                self.pospins = self._set(ones,self.pospins)
                
                # note done with loading the envelope
                self.envelope_state = 2
                
        if goal_growth_rate != None:
            
            assert self.can_has_domains, "must set domains before ggr"
            assert isinstance(ggr,tuple) and len(ggr) == 2, "ggr must be a 2-tuple"
        
            growth_rate,ncrossings = ggr
        
            window_length      = 10  # can be changed but not exposed for simplicity      
            rate               = (1+growth_rate)**(1./window_length)-1
            self.plan          = self._ggr_make_plan(self.m0,rate,0.02,50)
            self.target        = 0
            self.optimized_spa = 0.05

            self.direction = np.sign(self.m0-self.plan[-1])
            
            self.can_has_ggr = True
            
        if self.ggr_state == 1 and self.envelope_state == 2:
            
            self.next_crossing = 0.0
            self.crossed       = False
            self.ggr_tracker   = np.zeros((len(self.plan),3),float)
            
            self.spa_buffer    = self._allocate((self.N,self.N),np.float32)
            self.whenflipped   = self._allocate((self.N,self.N),np.int32)
            
            # build the lookup table for the recency enforcement
            # these parameters can be changed but are not exposed to the user to keep things simple
            rmin, rmax, rrate = 0.05, 2., 0.5
            x = np.arange(len(self.plan)).astype('float')
            recency_need = rmin*rmax*np.exp(rrate*x)/(rmax+rmin*np.exp(rrate*x))
            self.recency_need = cla.to_device(self.queue,recency_need.astype(np.float32))
            
            # self.crossings are the values of m_out which, when crossed over, generate a signal
            # to save the output to make a movie out of or whatever
            if isinstance(ncrossings,(int,float)): self.crossings = np.arange(0,1,1./ncrossings)[1:]
            if isinstance(ncrossings,(list,tuple,np.ndarray)): self.crossings = ncrossings
            if ncrossings != None: self.next_crossing = self.crossings[-1]
            
            self.direction = np.sign(self.m0-self.plan[-1])
            
            self.ggr_state = 2

        if returnables != None:
            
            # check types
            assert isinstance(returnables,(list,tuple)), "returnables must be list or tuple"
            assert all([isinstance(r,str) for r in returnables]), "all elements of returnables must be strings"
        
            self.returnables_list = []
            for r in returnables:
                if r not in self.keywords:
                    sys.stdout.write("requested returnable %s is unrecognized and will be ignored\n"%r)
                else:
                    self.returnables_list.append(r)
                
    def seed(self,supplied=None):
        """ Replaces self.domains with random numbers.
        
        arguments:
            supplied - (optional) can set the seed with a precomputed array.
        """
        
        if self.envelope_state != 2:
            warnings.warn('cannot seed the simulation without an envelope being set first')
        
        if self.envelope_state == 2:
            # this indicates space has also been allocated for self.domains
            if supplied == None: supplied = np.random.rand(self.N,self.N)-0.5
            self.domains = self._set(supplied.astype(np.complex64),self.domains)
            self.converged = False
            print "seeded"
            
    def kick(self,amount):
        """ If the simulation has reached an acceptable degree of convergence,
        "kick" the speckle pattern so that the domains re-converge into a slightly
        different configuration. For XPCS simuluations. """
        
        # fourier transform the domains. multiply the magnitude component
        # by random*
        
        # set the kicker
        kicker = 1+amount*np.random.randn(self.N,self.N).astype(np.float32)
        kicker[kicker < 0] = 0
        self.kicker = self._set(kicker,self.kicker)
        
        # form the fourier representation
        if use_gpu: self._fft2(self.domains,self.fourier_domains)
        else: self.fourier_domains = self._fft2(self.domains,self.fourier_domains)
        
        # multiply fourier_domains by the kicker
        if use_gpu: self._cl_mult(self.kicker,self.fourier_domains,self.fourier_domains)
        else:       self.fourier_domains *= self.kicker
        
        # inverse transform the domains, ending the kick
        if use_gpu: self._fft2(self.fourier_domains,self.domains,inverse=True)
        else: self.domains = IDFT(self.fourier_domains)
        
        self.converged = False
        
    def kick2(self,amount):
        """ If the simulation has reached an acceptable degree of convergence,
        "kick" the speckle pattern so that the domains re-converge into a slightly
        different configuration. For XPCS simuluations. """
        
        # fourier transform the domains. multiply the magnitude component
        # by random*
        
        # set the kicker
        kicker = 1+amount*np.random.randn(self.N,self.N).astype(np.float32)
        kicker[kicker < 0] = 0
        self.kicker = self._set(kicker,self.kicker)
        
        # form the fourier representation
        if use_gpu: self._fft2(self.domains,self.fourier_domains)
        else: self.fourier_domains = self._fft2(self.domains,self.fourier_domains)
        
        # multiply fourier_domains by the kicker
        if use_gpu: self._cl_add(self.kicker,self.fourier_domains,self.fourier_domains)
        else:       self.fourier_domains += self.kicker
        
        # inverse transform the domains, ending the kick
        if use_gpu: self._fft2(self.fourier_domains,self.domains,inverse=True)
        else: self.domains = np.fft.ifft2(self.fourier_domains)
        
        self.converged = False
        
    def kick_rs(self,amount):
        kicker = 1+amount*np.random.randn(self.N,self.N).astype(np.float32)
        kicker[kicker < 0] = 0
        self.kicker = self._set(kicker,self.kicker)
        if use_gpu: self._cl_mult(self.kicker,self.domains,self.domains)
        else:       self.domains *= kicker
        
        self.converged = False
        
    def kick_rs2(self,amount):
        kicker = 1+amount*np.random.randn(self.N,self.N).astype(np.float32)
        kicker[kicker < 0] = 0
        self.kicker = self._set(kicker,self.kicker)
        if use_gpu: self._cl_add(self.kicker,self.domains,self.domains)
        else:       self.domains += kicker
        
        self.converged = False

    def _blur(self):
        
        if use_gpu:
            self._fft2(self.speckles,self.speckles)
            self._cl_mult(self.speckles,self.ipsf,self.speckles)
            self._fft2(self.speckles,self.speckles,inverse=True)
            self._cl_abs(self.speckles,self.speckles)
                
        else:
            self.speckles = np.abs(IDFT(DFT(self.speckles)*self.ipsf))

    def _fft2(self,data_in,data_out,inverse=False,plan=None):
        # unified wrapper for fft.
        # note that this does not expose the full functionatily of the pyfft
        # plan because of assumptions regarding the input (eg, is complex)

        if use_gpu:
            if plan == None: plan = self.fftplan
            plan.execute(data_in=data_in.data,data_out=data_out.data,inverse=inverse,wait_for_finish=True)
        else:
            if inverse: return IDFT(data_in)
            else      : return DFT(data_in)

    def _preserve_power(self,stage):
        
        """ Preserve the power of the speckle pattern OUTSIDE of the 0,0
        component across the blurring process. The 0,0 component is extracted
        before blurring and replaced after blurring so as to not disrupt the
        average magnetization """
        
        
        assert stage in ('in','out'), "unrecognized _preserve_power stage %s"%stage

        if use_gpu:
            if stage == 'in':
                self._kexec('getm0',self.fourier_domains,self.m0_1,shape=(1,))
                self._cl_copy(self.speckles,self.speckles_f) # have to copy to a float array to run cla.sum
                self.power_in = self._sum(self.speckles_f)-abs(self.m0_1.get())[0]**2
                self._kexec('replace_dc1',self.speckles,np.int32(self.N),shape=(1,))
            if stage == 'out':
                self._kexec('getm0',self.speckles,self.m0_2,shape=(1,))
                self._cl_copy(self.speckles,self.speckles_f) # have to copy to a float array to run cla.sum
                self.power_out = self._sum(self.speckles_f)-abs(self.m0_2.get()[0])
                r = np.sqrt(self.power_in/self.power_out)
                self._cl_mult(np.float32(r),self.fourier_domains,self.fourier_domains)
                self._kexec('replace_dc2',self.fourier_domains,self.m0_1,shape=(1,))
                
        else:
            if stage == 'in':
                self.m0_1 = self.fourier_domains[0,0]
                self.power_in = self.speckles.sum()-abs(self.m0_1)**2
                self.speckles[0,0] = (self.speckles[1,0]+self.speckles[1,1]+self.speckles[0,1])/3.
                
            if stage == 'out':
                self.m0_2 = self.speckles[0,0]
                self.power_out = self.speckles.sum()-self.m0_2
                r = np.sqrt(self.power_in/self.power_out)
                #print self.power_in,self.power_out,r
                self.fourier_domains *= r
                self.fourier_domains[0,0] = self.m0_1
    
    def _rescale_speckle(self,iteration=None):
        """ The heart of the algorithm. """
        
        # this function implements the fourier operation: rescaling the envelope
        # order of operations:
        # 1. fft the domains
        # 2. record the amount of incoming speckle power with _preserve_power('in')
        # 3. blur with blur_kernel and _blur to "despeckle"
        # 4. rescale using blurred speckle as part of the input
        # 5. normalize outgoing speckle power to incoming power with _preserve_power('out')
        # 6. ifft the domains

        # form the speckle pattern by abs(fft)**2.
        if use_gpu:
            self._fft2(self.domains,self.fourier_domains)
            self._cl_abs(self.fourier_domains,self.speckles,square=True)
        else:
            self.fourier_domains = self._fft2(self.domains,self.fourier_domains)
            self.speckles = np.abs(self.fourier_domains)**2

        if 'speckle_in' in self.returnables_list:
            self.make_speckle(self.domains_dft_re,self.domains_dft_im, self.speckles)
            self.returnables['speckle_in'] = shift(np.abs(self.speckles.get()))

        # this is the first half of the power preservation. it extracts the dc component
        # from the speckle pattern to ensure that the average magnetization remain uneffected
        # by the rescaling process.
        self._preserve_power('in')

        # convolve self.speckles with self.ipsf to form blurry speckles. store blurry speckles in
        # self.speckles (ie, in place)
        self._blur() 
        if 'blurred' in self.returnables_list: self.returnables['blurred'] = shift(np.abs(self.speckles.get()))
        
        # multiply the domain pattern by the rescaler function. calculate the new speckle for power comparison.
        if use_gpu:
            self._kexec('erescale',self.fourier_domains,self.envelope,self.speckles)
            #self._cl_abs(self.fourier_domains,self.speckles,square=True)
        else:
            self.fourier_domains *= np.sqrt(self.envelope/self.speckles)
            self.speckles = np.abs(self.fourier_domains)**2

        # this is the second half of the power preservation, which ensures that the total power of the
        # speckle pattern remains unchanged over the rescaling, and which also puts the dc component
        # back in place.
        self._preserve_power('out')

        if 'rescaled' in self.returnables_list:
            self.returnables['rescaled'] = shift(abs(self.speckles.get()))

        # inverse transform the rescaled speckles
        if use_gpu:
            self._fft2(self.fourier_domains,self.domains,inverse=True)
        else:
            self.domains = IDFT(self.fourier_domains)

    def _sum(self,array,d=None):
        if use_gpu:
            assert array.dtype != 'complex64'
            x = cla.sum(array,dtype=d).get()
            return x
        else:
            x = np.sum(array)
            return x

    def set_returnables(self,returnables=('converged',)):
        
        """ Set which of the possible intermediate values are returned out of
        the simulation. Results are returned as a dictionary from which
        intermediates can be extracted through returnables['key'] where 'key' is
        the desired intermediate.
        
        Available returnables:
            *** normal returnables
            converged: domain image when convergence is reached
            domains: domains after an iteration; for making convergence movies?
            envelope: the goal envelope functioning as fourier constraint
            
            *** debugging returnables dealing with rescaling
            rescaled: rescaled speckle; note: includes dc component
            rescaler: rescaler function
            walls1: all incoming walls
            walls2: all walls after rescaling
            (neg/pos)_walls(1/2): negative (postive) walls before (after) rescaling
            bounded: domains bounded to +- 1 after rescaling
            promoted: domains after spin promotion
            
        By default, the only returnable is 'converged', the final output.
        
        Advisory note: since the point of running on the GPU is !SPEED!, and
        pulling data off the GPU is slow, use of returnables should be limited
        except for debugging.
        """
        
        available = ('converged','domains','rescaled','rescaler','blurred',
                     'walls1','walls2','pos_walls1','pos_walls2','neg_walls1',
                     'neg_walls2','bounded','promoted','envelope','speckle_in')
        
        # check types
        assert isinstance(returnables,(list,tuple)), "returnables must be list or tuple"
        assert all([isinstance(r,str) for r in returnables]), "all elements of returnables must be strings"
        
        self.returnables_list = []
        for r in returnables:
            if r not in available:
                print "requested returnable %s is unrecognized and will be ignored"%r
            else:
                self.returnables_list.append(r)

        # simple kernels written as pyopencl objects
        self.ggr_promote_spins = EK(self.context, # differs from above in that a different output is used: unify later?
            "float *domains,"
            "float *sites,"
            "float *out,"
            "float spa",
            "out[i] = domains[i]+sites[i]*spa",
            "ggr_promote_spins")
        
        self.enforce_boundary = EK(self.context,
            "float *domains,"
            "float *boundary,"
            "float *bv",
            "domains[i] = domains[i]*(1-boundary[i])+boundary[i]*bv[i]",
            "enforce_boundary")
        
        self.only_in_walls = EK(self.context,
            "float *new,"
            "float *old,"
            "float *walls",
            "new[i] = new[i]*walls[i]+old[i]*(1-walls[i])",
            "only_in_walls")

    def ggr_iteration(self,iteration):
        
        """ Run a single iteration of the ggr (goal-growth-rate) algorithm.
        This algorithm attempts to change the total magnetization of the sample
        between iterations by a certain fixed multiple according to
        m@iteration(n-1)/m@iteration(n).  The reason for this algorithm is to
        allow gentle worm-like growth at high magnetization."""
        
        assert self.can_has_domains, "no domains set!"
        assert self.can_has_envelope, "no goal envelope set!"
        assert self.can_has_ggr, "must set ggr before running ggr_iteration"

        # get the target net magnetization
        net_m       = cla.sum(self.domains).get()
        self.goal_m = self.plan[iteration+1]
        needed_m    = self.goal_m*self.N2-net_m
        self.target = np.sign(needed_m).astype(np.float32)
        
        # copy the current domain pattern to self.incoming
        self.copy(self.domains,self.incoming)
        
        # find the domain walls. these get used in self.make_available. make the correct sites available for modification
        self.findwalls.execute(self.queue,(self.N,self.N),self.domains.data,self.allwalls.data,self.poswalls.data,self.negwalls.data,np.int32(self.N))
        self.make_available1(self.available,self.allwalls,self.negpins,self.pospins)
        #if net_m > self.goal_m: self.make_available1(self.available,self.poswalls,self.negpins,self.pospins)
        #if net_m < self.goal_m: self.make_available1(self.available,self.negwalls,self.negpins,self.pospins)
        
        # run the ising bias
        self.ising(self.domains,self.alpha)
        
        # rescale the domains. this operates on the class variables so no arguments are passed. self.domains stores the rescaled real-valued domains.
        # the rescaled domains are bounded to the range +1 -1.
        # change in the domain pattern is allowed to happen only in the walls.
        # enforce the recency condition to prevent domain splittings (basically, make it hard to revert changes from long ago)
        self._rescale_speckle()
        self.bound(self.domains,self.domains)
        self.only_in_walls(self.domains,self.incoming,self.available)
        self.recency.execute(self.queue,(self.N2,),
                             self.whenflipped.data,self.domains.data,self.incoming.data,
                             self.recency_need.data,self.target,np.int32(iteration)).wait()

        # since the domains have been updated, refind the walls
        self.findwalls.execute(self.queue,(self.N,self.N),self.domains.data,self.allwalls.data,self.poswalls.data,self.negwalls.data,np.int32(self.N))

        # now adjust the magnetization so that it reaches the target
        net_m       = cla.sum(self.domains).get()
        needed_m    = self.goal_m*self.N2-net_m
        self.target = np.sign(needed_m).astype(np.float32)
        if net_m > 0: self.make_available2(self.available,self.poswalls,self.domains,self.target)
        if net_m < 0: self.make_available2(self.available,self.negwalls,self.domains,self.target)
        
        # now we need to run an optimizer to find the correct value for spa. this should result in an
        # update to the class variable self.optimized_spa. optimized_spa is a class variable because spa
        # changes slowly so using the old value as the starting point in the optimization gives a speed up.
        opt_out = fminbound(self._ggr_spa_error,-1,1,full_output=1)
        self.optimized_spa = opt_out[0]
        
        # use the optimized spa value to actually promote the spins in self.domains
        self.ggr_promote_spins(self.domains,self.available,self.domains,self.target*self.optimized_spa)
        self.bound(self.domains,self.domains)
        m_out   = (cla.sum(self.domains).get())/self.N2
        
        # update the whenflipped data, which records the iteration when each pixel changed sign
        self.update_whenflipped.execute(self.queue,(self.N2,),self.whenflipped.data,self.domains.data,self.incoming.data,np.int32(iteration))
        
        print "%.4d, %.3f, %.3f, %.3f"%(iteration, self.goal_m, m_out, self.optimized_spa)
        self.ggr_tracker[iteration] = iteration, self.optimized_spa, m_out
        
        # set a flag to save output. it is better to check m_out than to trust that the
        # self.goal_m, asked for in self.plan, has actually been achieved
        if m_out < self.next_crossing:
            print "***"
            self.checkpoint    = True
            self.crossed       = self.next_crossing
            try:
                self.crossings     = self.crossings[:-1]
                self.next_crossing = self.crossings[-1]
            except IndexError:
                self.next_crossing = 0.01
        else: self.checkpoint = False

    def _ggr_make_plan(self,m0,rate,transition,finishing):
        # make the plan of magnetizations for the ggr algorithm to target 
        
        taper = lambda x: 1-rate*np.tanh(50*(1-x))
        
        plan = []
        plan.append(m0)
        m = m0
        
        # transition is where the plan transitions from goal growth rate (exponential decay behavior)
        # to a constant drive down to 0 net magnetization. typically transition should be 0.2 or less
        # as this is where one seems to see "breathing" behavior in domain simulations

        while m >= transition:
            m1 = plan[-1]
            m2 = m1*taper(m1)
            plan.append(m2)
            m = m2
            
        while m > 0:
            m1 = plan[-1]
            m2 = m1-transition/finishing
            plan.append(m2)
            m = m2
            
        return plan

    def _ggr_spa_error(self,spa):
        # promote the available spins by value target*spa. store in the spa_buffer. bound spa_buffer.
        # calculate the new total magnetization for this spa value. difference of total and desired is the error function.
        
        self.ggr_promote_spins(self.domains,self.available,self.spa_buffer,self.target*spa)
        self.bound(self.spa_buffer,self.spa_buffer)
        buffer_average = (cla.sum(self.spa_buffer).get())/self.N2
        e = abs(buffer_average-self.goal_m)
        #print "    %.6e, %.3e"%(spa,e)
        return e

