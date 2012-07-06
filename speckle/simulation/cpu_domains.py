# core
import numpy as np
from scipy.optimize import fminbound
import string

# common libs. do some ugly stuff to get the path set to the kernels directory
from .. import shape, wrapping
import speckle

# cpu fft
DFT = np.fft.fft2
IDFT = np.fft.ifft2
shift = np.fft.fftshift

class generator():
    
    """ CPU-executing copd-type domain generation. """
    
    def __init__(self,device=None,domains=None,alpha=0.5,converged_at=0.002,ggr=None,returnables=('converged',)):
        
        if device != None:
            "device %s provided to cpu_domains, which requires no device."
            "that information is being ignored"
        
        self.can_has_domains = False
        self.can_has_envelope = False
        self.can_has_ggr = False
        
        if domains != None: self.set_domains(domains)
        if ggr != None: self.set_ggr(ggr)
        self.returnables_list = returnables
        self.returnables = {}

        # most of the arguments that come in need to be made into class variables to be accessible
        # to the other functions during the course of the simulation
        self.alpha     = alpha
        self.m_turnon  = 5
        self.converged_at = converged_at
        self.spa_max   = 0.4
        self.powerlist = [] # for convergence tracking
    
    def set_domains(self,domains):
        
        """ Load an object into the self.domains memory space or tell the
        simulation what size object to expect in the future so that the rest of
        the simulation can be properly initialized.
        
        This method allows the domain image to be changed suddenly during the
        simulation.
        
        Accepts one or the other:
            1. A numpy array. Must be square. If an array has been loaded in the
                past (or the array size has been set earlier), this new array
                must be the same size.
            2. An integer which sets the simulation grid size. A domain image
                must still be supplied in the future for the simulation to
                function properly. """
        
        assert isinstance(domains,(int,np.ndarray)), "domains must be int or array"
        
        def helper_load(d):
            x = len(d)
            self.m0 = np.sum(domains)/self.N2
            self.domains  = d
        
        if self.can_has_domains:
            # this means an domains has already been set and now we're just updating it
            assert isinstance(domains,np.ndarray) and domains.ndim==2, "domains must be 2d array"
            assert domains.shape == (self.N,self.N), "domains shape differs from initialization values"
            helper_load(domains.astype(self.domains.dtype))
            
        if not self.can_has_domains:
            if isinstance(domains,np.ndarray):
                assert domains.ndim == 2, "domains must be 2d"
                assert domains.shape[0] == domains.shape[1], "domains must be square"
                self.N = len(domains)
                self.N2 = self.N**2
                helper_load(domains)

            if isinstance(domains,int):
                self.N = domains
                self.N2 = domains**2
                self.domains = np.zeros((self.N,self.N),np.float32)
                
            # now that self.N has been established, initialize all the buffers for intermediates
            # that require self.N. many of these are more appropriately associated with the envelope
            # than the domains per se be the location of the initialization is not particularly important
            self.r_array       = shape.radial((self.N,self.N))
            self.phi_array     = np.mod(shape.angular((self.N,self.N))+2*np.pi,2*np.pi)
            
            self.negpins = np.ones((self.N,self.N),np.float32)
            self.pospins = np.ones((self.N,self.N),np.float32)

        self.can_has_domains = True
    
    def set_envelope(self,parameters_list):
        
        """ Build or set a goal envelope to function as the fourier constraint
        in the simulation. The object must be set before setting the envelope.
        The syntax of this method is, unfortunately, relatively complicated.
        
        Accepts: a list of parameters describing the goal envelope.

         Each entry in the list is a list
         Each entry has as element 0 either 'isotropic', 'modulation', 'goal_m',
            or 'supplied'.
         'isotropic' entries have a shape and some numerical parameters
               for example: ['isotropic', 'lorentzian', number1, number2, etc]
         'modulation' entries have two additional parameters which specifies the
            order and phase offset of the modulation, after which is a shape
            which describes where the modulation is located in the r coordinate.
               for example: ['modulation', 4, 0, 'lorentzian', x, y, z] gives a
               4th order symmetry with no phase offset, confined to some range
               of r by a lorentzian envelope with parameters x, y, z.
         'supplied' opens the listed file with syntax ['supplied', path_to_file]
            file must be supplied in FITS format. modulations and isotropic
            additions can still occur after loading the supplied file.
         'goal_m' is used to change the targeted net magnetization. It is
            between 0 and 1.
          Note that order of elements is important since they are applied going
            down the line without any type of sorting.
        
        """
        
        assert self.can_has_domains, "must set domains before setting envelope"
        
        if not self.can_has_envelope:
            # these are gpu memory objects associated with making envelopes or with rescaling
            # envelopes
            self.m0_1 = 0
            self.m0_2 = 0
            self.power = 0
            self.transitions = 0
        
        temp_envelope = np.zeros((self.N,self.N),float)

        for element in parameters_list:
            
            assert element[0] in ('isotropic','modulation','supplied','goal_m'), "command type %s unrecognized in set_envelope; command: %s"%(element[0],element)

            if element[0] == 'isotropic':
                parameters = element[1:]
                temp_envelope  += function_eval(self.r_array,parameters)
                
            if element[0] == 'modulation':
                strength   = element[1]
                symmetry   = element[2]
                phase      = element[3]
                parameters = element[4:]
     
                # make the angular and radial parts of the modulation           
                redbundled = ['sinusoid', strength, symmetry, phase]
                angular    = function_eval(self.phi_array,redbundled)
                radial     = function_eval(self.r_array,parameters)
                
                # modulate the envelope
                temp_envelope = temp_envelope*(1-radial)+temp_envelope*radial*angular
                
            if element[0] == 'supplied':
                from .. import io
                path = element[1]
                supplied = io.openfits(path).astype('float')
                assert supplied.shape == temp_envelope.shape, "supplied envelope and simulation size aren't the same"
                temp_envelope += supplied

            if element[0] == 'goal_m':
                
                # this sets the goal net magnetization
                # (a separate idea from the self.goal_m which happens with goal growth rate simulations)
                self.goal_m = element[1]*self.N*self.N

        # find the intensity center of the envelope. based on how far it is from the center,
        # make the blur kernel. sigma is a function of intensity center to avoid too much
        # blurring for scattering centered near q=0.
        unwrapped = np.sum(wrapping.unwrap(temp_envelope,(0,self.N/2,(self.N/2,self.N/2))),axis=1)
        center    = unwrapped.argmax()
        
        # find the fwhm. the amount of blurring depends on the width of the speckle ring in order
        # to avoid the envelope collapsing
        hm    = (unwrapped.max()-unwrapped.min())/2.+unwrapped.min()
        left  = abs(unwrapped[:center]-hm).argmin()
        right = abs(unwrapped[center:]-hm).argmin()+center
        fwhm  = abs(left-right)
        
        self.sigma = fwhm/8.
        self.set_coherence(self.sigma,self.sigma)
    
        temp_envelope += -temp_envelope.min()
        temp_envelope *= 1./temp_envelope.max()
                
        self.goal_envelope = shift(temp_envelope.astype(np.float32))
        self.can_has_envelope = True
        self.converged = False
        self.transitions += 1
        
        if 'envelope' in self.returnables_list: self.returnables['envelope'] = temp_envelope
         
    def set_ggr(ggr):
        
        assert self.can_has_domains, "must set domains before ggr"
        assert isinstance(ggr,tuple) and len(ggr) == 2, "ggr must be a 2-tuple"
        
        growth_rate,ncrossings = ggr
        
        window_length      = 10  # can be changed but not exposed for simplicity      
        rate               = (1+growth_rate)**(1./window_length)-1
        self.plan          = self._ggr_make_plan(self.m0,rate,0.02,50)
        self.target        = 0
        self.optimized_spa = 0.05

        if not self.can_has_ggr:
            self.next_crossing = 0.0
            self.crossed       = False
            self.ggr_tracker   = np.zeros((len(self.plan),3),float)
            self.spa_buffer    = cla.empty(self.queue,(self.N,self.N),np.float32)
            self.whenflipped   = cla.empty(self.queue,(self.N,self.N),np.int32)

            # build the lookup table for the recency enforcement
            # these parameters can be changed but are not exposed to the user to keep things simple
            rmin, rmax, rrate = 0.05, 2., 0.5
            x = np.arange(len(self.plan)).astype('float')
            recency_need = rmin*rmax*np.exp(rrate*x)/(rmax+rmin*np.exp(rrate*x))
            self.recency_need = cla.to_device(self.queue,recency_need.astype(np.float32))
    
            self.set_zero(self.whenflipped)
        
            # self.crossings are the values of m_out which, when crossed over, generate a signal
            # to save the output to make a movie out of or whatever
            if isinstance(ncrossings,(int,float)): self.crossings = np.arange(0,1,1./ncrossings)[1:]
            if isinstance(ncrossings,(list,tuple,np.ndarray)): self.crossings = ncrossings
            if ncrossings != None: self.next_crossing = self.crossings[-1]
        
        self.direction = np.sign(self.m0-self.plan[-1])
        
        self.can_has_ggr = True
     
    def set_coherence(self,clx,cly):
        
        assert isinstance(clx,(int,float)), "coherence parameter clx must be float or int"
        assert isinstance(cly,(int,float)), "coherence parameter cly must be float or int"

        self.blur_kernel = DFT(shift(shape.gaussian((self.N,self.N),(cly,clx)))).real
        self.can_has_coherence = True
        
    def set_returnables(self,returnables=('converged',)):
        
        """ Set which of the possible intermediate values are returned out of
        the simulation. Results are returned as a dictionary from which
        intermediates can be extracted through returnables['key'] where 'key' is
        the desired intermediate.
        
        Available returnables:
            converged: domain image when convergence is reached
            domains: domains after an iteration; for making convergence movies
            
            debugging returnables dealing with rescaling
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
                     'neg_walls2','bounded','promoted','available','incoming_dft')
        
        # check types
        assert isinstance(returnables,(list,tuple)), "returnables must be list or tuple"
        assert all([isinstance(r,str) for r in returnables]), "all elements of returnables must be strings"
        
        self.returnables_list = []
        for r in returnables:
            if r not in available:
                print "requested returnable %s is unrecognized and will be ignored"%r
            else:
                self.returnables_list.append(r)
            
    def one_iteration(self,iteration):
        
        """ Iterate through one cycle of the domain simulation. The sequence of
        operations closely follows that of diffractive imaging."""
        
        # iterate through one cycle of the simulation.
        assert self.can_has_domains, "no domains set!"
        assert self.can_has_envelope, "no goal envelope set!"
        assert self.can_has_coherence, "no blurring function set!"
        
        ising = lambda x: (1+self.alpha)*x-self.alpha*(x**3)
        
        # first, copy the current state of the domain pattern to a holding buffer ("incoming")
        #self.copy(self.domains,self.incoming)
        self.incoming = np.copy(self.domains)
        
        # now find the domain walls. modifications to the domain pattern due to rescaling only take place in the walls
        #self.findwalls.execute(self.queue,(self.N,self.N),self.domains.data,self.allwalls.data,self.poswalls.data,self.negwalls.data,np.int32(self.N))
        self._find_walls(neighbors=8)
        
        if 'walls1' in self.returnables_list: self.returnables['walls1'] = self.allwalls
        if 'pos_walls1' in self.returnables_list: self.returnables['pos_walls1'] = self.poswalls
        if 'neg_walls1' in self.returnables_list: self.returnables['neg_walls1'] = self.negwalls
        
        # run the ising bias
        #self.ising(self.domains,self.alpha)s
        self.domains = ising(self.domains)
        
        # rescale the domains. this operates on the class variables so no arguments are passed. self.domains stores the rescaled real-valued domains. the rescaled
        # domains are bounded to the range +1 -1.
        self._rescale_speckle()
        self.domains = self.domains.real
        np.clip(self.domains,-1,1,out=self.domains)
        if 'bounded' in self.returnables_list: self.returnables['bounded'] = self.domains
        
        # if making an ordering island, this is the command that enforces the border condition
        #if use_boundary and n > boundary_turn_on: self.enforce_boundary(self.domains,self.boundary,self.boundary_values)
        
        # so now we have self.incoming (old domains) and self.domains (rescaled domains). we want to use self.walls to enforce changes to the domain pattern
        # from rescaling only within the walls. because updating can change wall location, also refind the walls.
        #self.update_domains(self.domains,self.incoming,self.allwalls)
        
        if iteration > self.m_turnon:
            self._find_walls(neighbors=8)
            
            if 'walls2' in self.returnables_list: self.returnables['walls2'] = self.allwalls
            if 'pos_walls2' in self.returnables_list: self.returnables['pos_walls2'] = self.poswalls
            if 'neg_walls2' in self.returnables_list: self.returnables['neg_walls2'] = self.negwalls
            
            # now attempt to adjust the net magnetization in real space to achieve the target magnetization.
            net_m = np.sum(self.domains.real)
            needed_m = self.goal_m-net_m
    
            if needed_m > 0:
                self.available = self.negwalls*self.negpins*self.pospins
                sites = np.sum(self.available)
                spa = min([self.spa_max,needed_m/sites])
                
            if needed_m < 0:
                self.available = self.poswalls*self.negpins*self.pospins
                sites = np.sum(self.available)
                spa = max([-1*self.spa_max,needed_m/sites])
            
            self.domains += self.available*spa
            
            if 'available' in self.returnables_list: self.returnables['available'] = self.domains
            if 'promoted' in self.returnables_list: self.returnables['promoted'] = self.domains
            
            np.clip(self.domains,-1,1,out=self.domains)

        if 'domains' in self.returnables_list: self.returnables['domains'] = self.domains

    def ggr_iteration(self,iteration):
        
        assert self.can_has_domains, "no domains set!"
        assert self.can_has_envelope, "no goal envelope set!"
        assert self.can_has_ggr, "must set ggr before running ggr_iteration"

        # get the target net magnetization
        net_m       = cla.sum(self.domains)
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
        self.rescale_speckle()
        self.bound(self.domains,self.domains)
        self.only_in_walls(self.domains,self.incoming,self.available)
        self.recency.execute(self.queue,(self.N2,),
                             self.whenflipped.data,self.domains.data,self.incoming.data,
                             self.recency_need.data,self.target,np.int32(iteration)).wait()

        # since the domains have been updated, refind the walls
        self.findwalls.execute(self.queue,(self.N,self.N),self.domains.data,self.allwalls.data,self.poswalls.data,self.negwalls.data,np.int32(self.N))

        # now adjust the magnetization so that it reaches the target
        net_m       = cla.sum(self.domains)
        needed_m    = self.goal_m*self.N2-net_m
        self.target = np.sign(needed_m).astype(np.float32)
        spa         = self.optimized_spa
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
        m_out   = (cla.sum(self.domains))/self.N2
        m_error = abs(m_out-self.goal_m)
        
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

    def _find_walls(self,neighbors=4):
        # find the walls
        
        rolls = lambda d, r0, r1: np.roll(np.roll(d,r0,axis=0),r1,axis=1)
        
        if neighbors == 4: p = ((1,0),(-1,0),(0,1),(0,-1))
        if neighbors == 8: p = ((1,-1),(1,0),(1,1),(0,-1),(0,1),(-1,-1),(-1,0),(-1,1))
        
        allwalls = np.zeros_like(self.domains)
        for o in p:
            rolled = rolls(self.domains,o[0],o[1])
            allwalls += 1-(np.sign(self.domains*rolled)+1)/2
    
        self.allwalls = allwalls.astype(bool).astype(int)
        self.negwalls = self.allwalls*np.where(self.domains < 0,1,0)
        self.poswalls = self.allwalls*np.where(self.domains > 0,1,0)

    def _rescale_speckle(self):
        
        # this function implements the fourier operation: rescaling the envelope
        # order of operations:
        # 1. fft the domains
        # 2. record the amount of incoming speckle power with _preserve_power('in')
        # 3. blur the speckle to prep rescaling
        # 4. rescale using blurred speckle as part of the input
        # 5. normalize outgoing speckle power to incoming power with _preserve_power('out')
        # 6. ifft the domains

        # just to be safe, reset the temporary buffer which holds imaginary components

        dft_domains = DFT(self.domains)
        mag, phase = abs(dft_domains),np.angle(dft_domains)
        dft_speckle = abs(dft_domains)**2
        m0_1 = dft_domains[0,0]
        
        if 'incoming_dft' in self.returnables_list: self.returnables['incoming_dft'] = shift(abs(mag))
        
        # calculate the incoming power
        power_in = np.sum(dft_speckle)-abs(m0_1)**2
        new_00 = 0.
        for coords in ((1,-1),(1,0),(1,1),(0,-1),(0,1),(-1,-1),(-1,0),(-1,1)):
            new_00 += dft_speckle[coords[0],coords[1]]/8.
        dft_speckle[0,0] = new_00
        
        # blur the speckle to form the rescaler, then rescale the dft_domains
        blurred = IDFT(DFT(dft_speckle)*self.blur_kernel)
        rescaler = np.sqrt(self.goal_envelope/blurred)
        np.nan_to_num(rescaler)
        dft_domains *= rescaler
        
        # record the outgoing power. multiple everything but the (0,0) by a ratio
        # of the powers. put the original (0,0) back in place
        dft_speckle2 = abs(dft_domains)**2
        power_out = np.sum(dft_speckle2)-dft_speckle2[0,0]
        ratio = np.sqrt(power_in.real/power_out.real)
        dft_domains *= ratio
        dft_domains[0,0] = m0_1
        
        self.domains = IDFT(dft_domains)

        if 'rescaler' in self.returnables_list: self.returnables['rescaler'] = shift(abs(rescaler))
        if 'rescaled' in self.returnables_list: self.returnables['rescaled'] = shift(abs(dft_speckle2))

    def _ggr_make_plan(self,m0,rate,transition,finishing):
        # make the plan of magnetizations for the ggr algorithm to target 
        
        taper = lambda x: 1-rate*np.tanh(50*(1-x))
        
        plan = []
        plan.append(m0)
        m = m0
        n = 1
        
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
        buffer_average = (cla.sum(self.spa_buffer))/self.N2
        e = abs(buffer_average-self.goal_m)
        #print "    %.6e, %.3e"%(spa,e)
        return e

    def check_convergence(self):
        
        # calculate the difference of the previous domains (self.incoming) and the current domains (self.domains).
        self.power = np.sum(abs(self.domains-self.incoming))/self.N2
        self.powerlist.append(self.power)
        
        # set the convergence condition
        if self.power >  self.converged_at: self.converged = False
        if self.power <= self.converged_at: self.converged = True
        
        if 'converged' in self.returnables_list and self.converged: self.returnables['converged'] = self.domains
        
def function_eval(x,Parameters):
    # for making the envelope
    Type = Parameters[0]
    
    assert Type in ('linear','lorentzian','lorentzian_sq','gaussian','tophat','sinusoid','uniform'), "function type %s unrecognized in make_envelope->function_eval"%Type
    
    if Type == 'linear':
        Slope,yIntercept = Parameters[1:]
        return Slope*x+yIntercept
        
    if Type == 'lorentzian':
        Scale,Center,Width = Parameters[1:]
        return float(Scale)/(1+(1/float(Width)**2)*(x-float(Center))**2)
        
    if Type == 'lorentzian_sq':
        Scale,Center,Width = Parameters[1:]
        return float(Scale)/(1+(1/float(Width)**2)*(x-float(Center))**2)**2
        
    if Type == 'gaussian':
        Scale,Center,Sigma = Parameters[1:]
        return Scale*np.exp(-((x-Center)**2)/(2*Sigma**2))
        
    if Type == 'tophat':
        Scale,Center,Width = Parameters[1:]
        Hat = np.where(x < Center+Width/2.,1,0)*np.where(x >= Center-Width/2.,1,0)
        return Hat
        
    if Type == 'sinusoid':
        Strength,Symmetry,Rotation = Parameters[1:]
        return 1+Strength*0.5*(np.cos(Symmetry*(x+Rotation))-1)
        
    if Type == 'uniform':
        return np.ones_like(x)
        
def make_speckle(array):
    return shift(abs(DFT(array))**2)