# core
import numpy
import pyopencl as cl
import pyopencl.array as cl_array
from pyopencl.elementwise import ElementwiseKernel as EK
from scipy.optimize import fminbound

# common libs
import shape,io,wrapping,gpu

# cpu fft
DFT = numpy.fft.fft2
from numpy.fft import fftshift

class generator():
    
    def __init__(self,gpuinfo,N,domains,alpha=0.5,interrupts=(),ggr=None):
        
        # most of the arguments that come in need to be made into class variables to be accessible
        # to the other functions during the course of the simulation
        self.context   = gpuinfo[0]
        self.device    = gpuinfo[1]
        self.queue     = gpuinfo[2]
        self.N         = N
        self.N2        = N**2
        self.alpha     = alpha
        self.m_turnon  = 5
        self.max_power = 0.002
        self.spa_max   = 0.4
        self.interrupts = interrupts
        
        ### initialize all the kernels
        
        # fft
        from pyfft.cl import Plan as fft_plan
        self.fftplan_split = fft_plan((self.N,self.N), dtype=numpy.float32, queue=self.queue) # fft plan which accepts 2 float32 inputs as re and im

        # complicated (or non-elementwise) kernels built from opencl/c99 code
        self.get_m0                = gpu.build_kernel_file(self.context, self.device, 'kernels/get_m0.cl')
        self.replace_dc_component1 = gpu.build_kernel_file(self.context, self.device, 'kernels/replace_dc_1.cl')
        self.replace_dc_component2 = gpu.build_kernel_file(self.context, self.device, 'kernels/replace_dc_2.cl')
        self.envelope_rescale      = gpu.build_kernel_file(self.context, self.device, 'kernels/envelope_rescale.cl')
        self.findwalls             = gpu.build_kernel_file(self.context, self.device, 'kernels/find_walls.cl') # looks at the 8 nearest neighbors.
        self.recency               = gpu.build_kernel_file(self.context, self.device, 'kernels/recency.cl')
        self.update_whenflipped    = gpu.build_kernel_file(self.context, self.device, 'kernels/update_whenflipped.cl')
    
        # simple kernels written as pyopencl objects
        self.copy = EK(self.context,
            "float *in,"
            "float *out",
            "out[i] = in[i]",
            "copy")
    
        self.bound = EK(self.context,
            "float *in,"
            "float *out",
            "out[i] = clamp(in[i],-1.0f,1.0f)",
            "bound")
        
        self.ising = EK(self.context,
            "float *domains,"
            "float a",
            "domains[i] = (1+a)*domains[i]-a*domains[i]*domains[i]*domains[i]",
            "ising")
        
        self.scalar_multiply = EK(self.context,
            "float c,"
            "float *array",
            "array[i] = array[i]*c",
            "ising")
    
        self.set_zero = EK(self.context,
            "float *array",
            "array[i] = 0.0f",
            "setzero")
        
        self.set_ones = EK(self.context,
            "float *array",
            "array[i] = 1.0f",
            "setones")
        
        self.set_zerof2 = EK(self.context,
            "float2 *array",
            "array[i] = (float2)(0.0f,0.0f)",
            "setzerof2")
    
        self.make_speckle = EK(self.context,
            "float *in_re,"
            "float *in_im,"
            "float *out", 
            "out[i] = pown(in_re[i],2)+pown(in_im[i],2)",
            "getmag")
        
        self.array_multiply_f_f = EK(self.context,
            "float *a,"       # convolvee, float
            "float *b,"       # convolver, float (kernel)
            "float *c",       # convolved, float
            "c[i] = a[i]*b[i]",
            "array_multiply_f_f")
        
        self.update_domains = EK(self.context,
            "float *domains,"
            "float *incoming,"
            "float *walls",
            "domains[i] = walls[i]*domains[i]+(1-walls[i])*incoming[i]", # update only the walls
            "update_domains")
        
        self.make_available1 = EK(self.context,
            "float *available,"
            "float *walls,"
            "float *pospins,"
            "float *negpins",
            "available[i] = walls[i]*pospins[i]*negpins[i]",
            "make_available1")
        
        self.make_available2 = EK(self.context,
            "float *available,"
            "float *walls,"
            "float *domains,"
            "float target",
            "available[i] = clamp(clamp(sign(domains[i])*target,0.0f,1.0f)+walls[i],0.0f,1.0f)",
            "make_available2")
        
        self.promote_spins = EK(self.context, # promote spins within the location given by *sites
            "float *domains,"
            "float *sites,"
            "float x",
            "domains[i] = domains[i]+sites[i]*x",
            "promote_spins")
        
        self.ggr_promote_spins = EK(self.context, # differs from above in that a different output is used: unify later?
            "float *domains,"
            "float *sites,"
            "float *out,"
            "float spa",
            "out[i] = domains[i]+sites[i]*spa",
            "ggr_promote_spins")
        
        self.array_diff = EK(self.context,
            "float *a1,"
            "float *a2,"
            "float *a3",
            "a3[i] = fabs(a1[i]-a2[i])",
            "domain_diff")
        
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

        ### initialize all the arrays 
    
        # initialize the arrays for making envelopes
        self.transitions = 0
        self.goal_envelope = cl_array.empty(self.queue,(self.N,self.N),numpy.float32)
        self.r_array       = shape.radial((self.N,self.N))
        self.phi_array     = numpy.mod(shape.angular((self.N,self.N))+2*numpy.pi,2*numpy.pi)
        self.powerlist     = [] # for convergence tracking
        
        # initialize buffers and kernels for speckle rescaling (ie, the function F)
        self.domains_dft_re = cl_array.empty(self.queue,(self.N,self.N),numpy.float32)
        self.domains_dft_im = cl_array.empty(self.queue,(self.N,self.N),numpy.float32)
        self.speckle        = cl_array.empty(self.queue,(self.N,self.N),numpy.float32)
        self.dummy_zeros    = cl_array.empty(self.queue,(self.N,self.N),numpy.float32)
        self.rescaler       = cl_array.empty(self.queue,(self.N,self.N),numpy.float32)
        self.domain_diff    = cl_array.empty(self.queue,(self.N,self.N),numpy.float32)
        self.m0_1           = cl_array.empty(self.queue,(1,), numpy.complex64)
        self.m0_2           = cl_array.empty(self.queue,(1,), numpy.complex64)
        self.power          = cl_array.empty(self.queue,(2,), numpy.float32)

        # initialize buffers and kernels for real-space and magnetization constraints (ie, the function M)
        self.incoming  = cl_array.empty(self.queue,(self.N,self.N),numpy.float32)
        self.allwalls  = cl_array.empty(self.queue,(self.N,self.N),numpy.float32)
        self.poswalls  = cl_array.empty(self.queue,(self.N,self.N),numpy.float32)
        self.negwalls  = cl_array.empty(self.queue,(self.N,self.N),numpy.float32)
        self.negpins   = cl_array.empty(self.queue,(self.N,self.N),numpy.float32) # 1 means ok to change, 0 means retain value
        self.pospins   = cl_array.empty(self.queue,(self.N,self.N),numpy.float32)
        self.available = cl_array.empty(self.queue,(self.N,self.N),numpy.float32)
        
        self.set_ones(self.negpins)
        self.set_ones(self.pospins)
        
        ### initialize the domain image; put on gpu as self.domains
        x = len(domains)
        assert x >= self.N, "must supply a seed with correct size"
        if x > self.N: domains = domains[x/2-self.N/2:x/2+self.N/2,x/2-self.N/2:x/2+self.N/2]
        m0 = numpy.sum(domains)/self.N2
        self.domains  = cl_array.to_device(self.queue,domains.astype(numpy.float32))
        
        if ggr != None:
            
            growth_rate,ncrossings = ggr
        
            # running the ggr routine requires additional objects for the spa optimization
            window_length      = 10  # can be changed but not exposed for simplicity      
            rate               = (1+growth_rate)**(1./window_length)-1                 
            self.spa_buffer    = cl_array.empty(self.queue,(self.N,self.N),numpy.float32)
            self.whenflipped   = cl_array.empty(self.queue,(self.N,self.N),numpy.int32)
            self.plan          = self._ggr_make_plan(m0,rate,0.02,50)
            self.target        = 0
            self.optimized_spa = 0.05
            self.next_crossing = 0.0
            self.crossed       = False
            self.ggr_tracker   = numpy.zeros((len(self.plan),3),float)
            
            # build the lookup table for the recency enforcement
            # these parameters can be changed but are not exposed to the user to keep things simple
            rmin, rmax, rrate = 0.05, 2., 0.5
            x = numpy.arange(len(self.plan)).astype('float')
            recency_need = rmin*rmax*numpy.exp(rrate*x)/(rmax+rmin*numpy.exp(rrate*x))
            self.recency_need = cl_array.to_device(self.queue,recency_need.astype(numpy.float32))
        
            self.set_zero(self.whenflipped)
            
            # self.crossings are the values of m_out which, when crossed over, generate a signal
            # to save the output to make a movie out of or whatever
            if isinstance(ncrossings,(int,float)): self.crossings = numpy.arange(0,1,1./ncrossings)[1:]
            if isinstance(ncrossings,(list,tuple,numpy.ndarray)): self.crossings = ncrossings
            if ncrossings != None: self.next_crossing = self.crossings[-1]
            
            self.direction = numpy.sign(m0-self.plan[-1])
            print self.crossings
            
    def make_blur_kernel(self,clx,cly):
        gaussian         = abs(DFT(fftshift(_gaussian_kernel(cly,clx,self.N))))
        assert gaussian.shape == (self.N,self.N), "gaussian blur kernel is wrong shape!"
        self.blur_kernel = cl.array.to_device(self.queue,gaussian.astype(numpy.float32))
        self.blur_exists = True
        
    def make_envelope(self,parameter_list):

        # Parameters come in as a list.
        # Each entry in the list is a list
        # Each entry has as element 0 either 'isotropic' or 'modulation'
        # 'isotropic' entries have a shape and some numerical parameters
        #       for example: ['isotropic', 'lorentzian', number1, number2, etc]
        # 'modulation' entries have two additional parameters which specifies the order and phase offset
        #  of the modulation, after which is a shape which describes where the modulation is located in r
        #       for example: ['modulation', 4, 0, 'lorentzian', number1, number2, etc]
        #       so that is a 4th order symmetry with no phase offset, confined to some range of r by a lorentzian envelope.
        # 'supplied' opens the listed file with syntax ['supplied', path_to_fits_file]. file must be supplied in FITS format. modulations and isotropic additions can still occur.
        # 'goal_m' is used to change the targeted net magnetization. It is between 0 and 1.
        #  Note that order of elements is important since they are applied going down the line without any type of sorting.
        
        temp_envelope = numpy.zeros((self.N,self.N),float)

        for element in parameter_list:

            if element[0] == 'isotropic':
                parameters = element[1:]
                temp_envelope  += function_eval(self.r_array,parameters)
                
            if element[0] == 'modulation':
                strength   = element[1]
                symmetry   = element[2]
                phase      = element[3]
                parameters = element[4:]
     
                # make the angular part of the modulation           
                redbundled = ['sinusoid', strength, symmetry, phase]
                angular    = function_eval(self.phi_array,redbundled)
                
                # make the radial part of the modulation
                radial     = function_eval(self.r_array,parameters)
                
                # modulate the envelope
                temp_envelope = temp_envelope*(1-radial)+temp_envelope*radial*angular
                
            if element[0] == 'supplied':
                path = element[1]
                supplied = io.openfits(path).astype('float')
                assert supplied.shape == temp_envelope.shape, "supplied envelope and simulation size aren't the same"

            if element[0] == 'goal_m':
                
                # this sets the goal net magnetization
                # (a separate idea from the self.goal_m which happens with goal growth rate simulations)
                self.goal_m = element[1]*self.N*self.N

        # find the intensity center of the envelope. based on how far it is from the center,
        # make the blur kernel. sigma is a function of intensity center to avoid too much
        # blurring for scattering centered near q=0.
        unwrapped = numpy.sum(wrapping.unwrap(temp_envelope,(0,self.N/2,(self.N/2,self.N/2))),axis=1)
        center    = unwrapped.argmax()
        
        # find the fwhm. the amount of blurring depends on the width of the speckle ring in order
        # to avoid the envelope collapsing
        hm    = (unwrapped.max()-unwrapped.min())/2.+unwrapped.min()
        left  = abs(unwrapped[:center]-hm).argmin()
        right = abs(unwrapped[center:]-hm).argmin()+center
        fwhm  = abs(left-right)
        
        self.sigma = fwhm/8.
        self.make_blur_kernel(self.sigma,self.sigma)
    
        temp_envelope += -temp_envelope.min()
        temp_envelope *= 1./temp_envelope.max()
                
        self.goal_envelope.set(fftshift(temp_envelope.astype(numpy.float32)))

        if 'envelope' in self.interrupts:
            io.save('output/gpudg/envelope %s.fits'%self.transitions,temp_envelope)
            io.save('output/gpudg/blurkernel %s.fits'%self.transitions,fftshift(self.blur_kernel.get()))
            
        if 'envelope_img' in self.interrupts:
            io.save('output/gpudg/envelope %s.png'%self.transitions,temp_envelope)
            io.save('output/gpudg/blurkernel %s.png'%self.transitions,fftshift(self.blur_kernel.get()))
        
        self.transitions += 1

    def rescale_speckle(self):
        # this function implements the fourier operation: rescaling the envelope

        # just to be safe, reset the temporary buffer which holds imaginary components
        self.set_zero(self.dummy_zeros)

        # fourier transform the domains. store the dft as split components in domains_dft_re and domains_dft_im
        self.fftplan_split.execute(data_in_re  = self.domains.data,        data_in_im  = self.dummy_zeros.data,
                                   data_out_re = self.domains_dft_re.data, data_out_im = self.domains_dft_im.data,
                                   wait_for_finish = True)

        # make the components into speckle. get m0 and the speckle power.
        # m0 is a float2 which stores the (0,0) of re and im
        # replace the (0,0) component with the average of the surrounding neighbors to prevent envelope distortion due to non-zero average magnetization
        self.get_m0.execute(self.queue,(1,), self.domains_dft_re.data, self.domains_dft_im.data, self.m0_1.data)
        self.make_speckle(self.domains_dft_re,self.domains_dft_im, self.speckle)
        power1 = cl_array.sum(self.speckle).get()-(self.m0_1.get()[0])**2
        self.replace_dc_component1.execute(self.queue,(1,), self.speckle.data, self.speckle.data, numpy.int32(self.N))

        # blur with the convolution kernel using internal kernels
        self.fftplan_split.execute(self.speckle.data, self.dummy_zeros.data, wait_for_finish=True)                # NxN fft with re in self.speckle_magnitude and im in dummy_zeros
        self.array_multiply_f_f(self.speckle,self.blur_kernel,self.speckle)                                       # magnitude currently holds the re part of the fft
        self.array_multiply_f_f(self.dummy_zeros,self.blur_kernel,self.dummy_zeros)                               # dummy_zeros currently holds the im part of the fft
        self.fftplan_split.execute(self.speckle.data, self.dummy_zeros.data, wait_for_finish=True, inverse=True)  # fft back. re part in speckle, im part (= 0) in dummy_zeros. in-place replacement.            
                                     
        # rescale the fourier transform by the ratio of the blurred-speckle to the goal-envelope. this enforces the fourier constraint.
        # save the rescaler function for debugging but it never gets used outside of this kernel
        self.envelope_rescale.execute(self.queue,(self.N2,),
                                      self.domains_dft_re.data, self.domains_dft_im.data,
                                      self.goal_envelope.data,  self.speckle.data, self.rescaler.data) 

        # preserve the total amount of speckle power outside the (0,0) component
        self.get_m0.execute(self.queue,(1,), self.domains_dft_re.data, self.domains_dft_im.data, self.m0_2.data)
        self.make_speckle(self.domains_dft_re,self.domains_dft_im, self.speckle) # for computing out-going power
        power2 = cl_array.sum(self.speckle).get()-(self.m0_2.get()[0])**2
        ratio = (numpy.sqrt(power1/power2)).astype(numpy.float32)
        self.scalar_multiply(ratio,self.domains_dft_re)
        self.scalar_multiply(ratio,self.domains_dft_im)
        
        # put m0_1 back as the (0,0) component so that the average magnetization is not effected by the rescaling
        self.replace_dc_component2.execute(self.queue,(1,), self.domains_dft_re.data, self.domains_dft_im.data, self.m0_1.data)
        self.make_speckle(self.domains_dft_re,self.domains_dft_im, self.speckle)

        # transform back to real space
        self.fftplan_split.execute(data_in_re  = self.domains_dft_re.data, data_in_im  = self.domains_dft_im.data,
                                   data_out_re = self.domains.data,        data_out_im = self.dummy_zeros.data,
                                   wait_for_finish = True, inverse=True)

    def _ggr_make_plan(self,m0,rate,transition,finishing):
        # make the plan of magnetizations for the ggr algorithm to target 
        
        taper = lambda x: 1-rate*numpy.tanh(50*(1-x))
        
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

    def ggr_iteration(self,iteration):

        # get the target net magnetization
        net_m       = cl_array.sum(self.domains).get()
        self.goal_m = self.plan[iteration+1]
        needed_m    = self.goal_m*self.N2-net_m
        self.target = numpy.sign(needed_m).astype(numpy.float32)
        
        # copy the current domain pattern to self.incoming
        self.copy(self.domains,self.incoming)
        self.set_zero(self.debug_want)
        self.set_zero(self.debug_need)
        
        # find the domain walls. these get used in self.make_available. make the correct sites available for modification
        self.findwalls.execute(self.queue,(self.N,self.N),self.domains.data,self.allwalls.data,self.poswalls.data,self.negwalls.data,numpy.int32(self.N))
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
                             self.recency_need.data,self.target,numpy.int32(iteration),
                             self.debug_want.data,self.debug_need.data).wait()

        # since the domains have been updated, refind the walls
        self.findwalls.execute(self.queue,(self.N,self.N),self.domains.data,self.allwalls.data,self.poswalls.data,self.negwalls.data,numpy.int32(self.N))

        # now adjust the magnetization so that it reaches the target
        net_m       = cl_array.sum(self.domains).get()
        needed_m    = self.goal_m*self.N2-net_m
        self.target = numpy.sign(needed_m).astype(numpy.float32)
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
        m_out   = (cl_array.sum(self.domains).get())/self.N2
        m_error = abs(m_out-self.goal_m)
        
        # update the whenflipped data, which records the iteration when each pixel changed sign
        self.update_whenflipped.execute(self.queue,(self.N2,),self.whenflipped.data,self.domains.data,self.incoming.data,numpy.int32(iteration))
        
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

    def _ggr_spa_error(self,spa):
        # promote the available spins by value target*spa. store in the spa_buffer. bound spa_buffer.
        # calculate the new total magnetization for this spa value. difference of total and desired is the error function.
        
        self.ggr_promote_spins(self.domains,self.available,self.spa_buffer,self.target*spa)
        self.bound(self.spa_buffer,self.spa_buffer)
        buffer_average = (cl_array.sum(self.spa_buffer).get())/self.N2
        e = abs(buffer_average-self.goal_m)
        #print "    %.6e, %.3e"%(spa,e)
        return e

    def one_iteration(self,iteration):
        # iterate through one cycle of the simulation.
        
        # first, copy the current state of the domain pattern to a holding buffer ("incoming")
        self.copy(self.domains,self.incoming)
        
        # now find the domain walls. modifications to the domain pattern due to rescaling only take place in the walls
        self.findwalls.execute(self.queue,(self.N,self.N),self.domains.data,self.allwalls.data,self.poswalls.data,self.negwalls.data,numpy.int32(self.N))
        
        # run the ising bias
        self.ising(self.domains,self.alpha)
        
        # rescale the domains. this operates on the class variables so no arguments are passed. self.domains stores the rescaled real-valued domains. the rescaled
        # domains are bounded to the range +1 -1.
        self.rescale_speckle()
        self.bound(self.domains,self.domains)
        
        # if making an ordering island, this is the command that enforces the border condition
        #if use_boundary and n > boundary_turn_on: self.enforce_boundary(self.domains,self.boundary,self.boundary_values)
        
        # so now we have self.incoming (old domains) and self.domains (rescaled domains). we want to use self.walls to enforce changes to the domain pattern
        # from rescaling only within the walls. because updating can change wall location, also refind the walls.
        #self.update_domains(self.domains,self.incoming,self.allwalls)
        
        if iteration > self.m_turnon:
            self.findwalls.execute(self.queue,(self.N,self.N),self.domains.data,self.allwalls.data,self.poswalls.data,self.negwalls.data,numpy.int32(self.N))
            
            # now attempt to adjust the net magnetization in real space to achieve the target magnetization.
            net_m = cl_array.sum(self.domains).get()
            needed_m = self.goal_m-net_m
    
            if needed_m > 0:
                self.make_available(self.available,self.negwalls,self.negpins,self.pospins)
                sites = cl_array.sum(self.available).get()
                spa = min([self.spa_max,needed_m/sites])
                
            if needed_m < 0:
                self.make_available(self.available,self.poswalls,self.negpins,self.pospins)
                sites = cl_array.sum(self.available).get()
                spa = max([-1*self.spa_max,needed_m/sites])

            self.promote_spins(self.domains,self.available,spa)
            self.bound(self.domains,self.domains)

    def check_convergence(self):
        
        # calculate the difference of the previous domains (self.incoming) and the current domains (self.domains).
        self.array_diff(self.incoming,self.domains,self.domain_diff)
        
        # sum the difference array and divide by the area of the simulation as a metric of how much the two domains
        # configurations differ. 
        power = (cl_array.sum(self.domain_diff).get())/self.N2
        self.powerlist.append(power)
        
        # set the convergence condition
        if power >  self.max_power: return False
        if power <= self.max_power: return True
        
def _gaussian_kernel(sx,sy,N):
    x = numpy.arange(N).astype('float')
    kernel_x = numpy.exp(-1*abs(N/2-x)**2/(2*sx**2)).astype(numpy.float32)
    kernel_y = numpy.exp(-1*abs(N/2-x)**2/(2*sy**2)).astype(numpy.float32)
    return numpy.outer(kernel_x,kernel_y)
        
def function_eval(x,Parameters):
    # for making the envelope
    Type = Parameters[0]
    
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
        return Scale*numpy.exp(-((x-Center)**2)/(2*Sigma**2))
        
    if Type == 'top hat':
        Scale,Centr,Width,Bevel = Parameters[1:]
        Hat = numpy.where(x < Center+Width/2.,1,0)*numpy.where(x >= Center-Width/2.,1,0)
        if Bevel > 0:
            Convolver = libarrayops.RollArray(libgenerate.GaussianArray(Bevel/Rescale,Bevel/Rescale,(Size,Size)))
            Hat = abs(libarrayops.Convolve(Convolver,Hat))
        
    if Type == 'sinusoid':
        Strength,Symmetry,Rotation = Parameters[1:]
        return 1+Strength*0.5*(numpy.cos(Symmetry*(x+Rotation))-1)
        
    if Type == 'uniform':
        return numpy.ones_like(x)
        
def make_speckle(array):
    return fftshift(abs(DFT(array))**2)