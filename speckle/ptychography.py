import numpy as np
import gpu
import time
import io
    
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
    
io.set_overwrite(True)
I = complex(0, 1)
    
class ptychography(gpu.common):

    """ A class to provide an API for ptychography scripts. This class makes
    liberal use of eval and exec to manage the variable number of
    illumination sites, as addressing 3d arrays on the gpu turns out
    to be sort of slow. """
    
    def __init__(self, force_cpu=False, force_np_fft=False, gpu_info=None):

        # load the gpu if available
        # keep context, device, queue, platform, and kp
        # in the parent namespace (still available to self)
        if force_cpu:
            self.use_gpu = False
        if not force_cpu:
            gpu.common.project = 'ptycho'
            self.start(gpu_info) 

        if self.use_gpu:
            self.compute_device = 'gpu'  
        else:
            self.compute_device = 'cpu'
        
        if HAVE_FFTW and not force_np_fft:
            self.fft2 = pyfftw.interfaces.numpy_fft.fft2
            self.ifft2 = pyfftw.interfaces.numpy_fft.ifft2
        else:
            self.fft2 = np.fft.fft2
            self.ifft2 = np.fft.ifft2
            
        test = np.random.random((16, 16))+1j*np.random.random((16, 16))
        self.test = test.astype(np.complex64)

        # state variables. certain of these must be changed from zero for the
        # reconstruction to proceed. these track which data is loaded into the
        # class. 0 = nothing; 1 = partial; 2 = complete.
        
        # for data generation only
        self.sample_state = 0
        self.can_generate = 0
        
        # for data reconstruction only
        self.modulus_state = 0
        self.object_state = 0
        self.can_reconstruct = 0
        
        # for data reconstruction and data generation
        self.probe_state = 0
        self.coordinates_state = 0
        self.ipsf_state = 0

        # dtype definitions
        self.array_dtypes = ('float32', 'complex64')
        self.ints = (int, np.int8, np.int16, np.int32, np.uint8)
        self.floats = (float, np.float16, np.float32, np.float64)
        self.float2s = (complex, np.complex64, np.complex128)
        self.iterables = (list, tuple, np.ndarray)
        
        # thresholds etc
        self.probe_update_threshold = 50
        
        # these strings get executed by exec to allocate
        # and set memory on the gpu
        self.alloc = 'self.%s = self._allocate(%s, %s, name="%s")'
        self.setstr = 'self.%s = self._set(%s, self.%s)'
        self.setstr2 = 'self.%s = self._set(%s.astype(%s), self.%s)'

        self.w_alloc = 'self.wave%s = self._allocate(self.shape, \
                       np.complex64, name="wave%s")'

        self.f_alloc = 'self.%s = self._allocate(self.shape, \
                       np.complex64, name="%s")'
        
        self.m_alloc = 'self.modulus%s = self._allocate(self.shape, \
                       np.float32, name="modulus%s")'
        
        self.w_set = 'self.wave%s = self._set(\
                     tmp.astype(np.complex64), self.wave%s)'
        
        self.m_set = 'self.modulus%s = self._set(\
                     modulus[n].astype(np.float32), self.modulus%s)'
        
        ##### attributes which are actually defined later, 
        ##### but are listed here for convenience and readability
        self.fftplan = None
        
        # for fourier constraint
        self.fourier_div = None
        self.psi_fourier = None
        self.psi_out = None
        self.psi_in = None
        self.fourier_div = None
        
        # for object and object update
        self.object = None
        self.o_top = None
        self.o_btm = None
        self.o_update = None
        self.o_update_kernel = None
        
        # for wave updates
        self.product = None 
        
        # for probe and probe update
        self.probe = None
        self.p_top = None
        self.p_btm = None
        self.pstar = None
        self.p2 = None
        
        # for generating data from a sample
        self.sample = None
        self.modulus_frames = None
        
        # for reconstructing data from moduli
        self.modulus = None
        self.iteration = None
        self.iterations = None
        self.frames_n = None
        self.frames_m = None
        self.r_coords = None
        self.c_coords = None
        self.propagator = None
        self.coordinates = None
        self.ipsf = None
        self.shape = None
        self.size = None
        
        # timings (help with optimizing)
        self.fft_time = 0
        self.o_update_time = 0

    def clear_coordinates(self):
        self.coordinates = None
        self.coordinates_state = 0
        
    def clear_modulus(self):
        self.modulus = None
        self.modulus_state = 0
        
    def clear_probe(self):
        self.probe = None
        self.probe_state = 0
        
    def clear_ipsf(self):
        self.ipsf = None
        self.ipsf_state = 0
           
    def clear_object(self):
        self.object = None
 
    def generate_diffraction_patterns(self):
        """ Generate ptychography diffraction patterns from:
            1. the sample function
            2. the probe function
            3. the probe coordinates
            
            Results are kept in self.modulus as moduli (not intensities)
            and with DC frequency at the corners. """

        def _make(sample, probe, ipsf, coord):
            tmp = self._rolls(sample, -coord[0], -coord[1])
            tmp = tmp[:probe.shape[0], :probe.shape[1]]
            tmp = np.abs(self.fft2(tmp*probe))
            if self.ipsf_state == 2:
                tmp = np.sqrt(np.abs(self.fft2(self.ifft2(tmp**2)*ipsf)))
            return tmp

        self._check_states_for_generation()

        if self.can_generate:

            sample = self.get(self.sample)
            probe = self.get(self.probe)
            try:
                ipsf = self.get(self.ipsf)
            except:
                ipsf = None
            
            self.modulus_frames = np.array([_make(sample, probe, ipsf, coord)\
                                            for coord in self.coordinates])
            self.load(modulus=self.modulus_frames)

    def load(self, sample=None, modulus=None, coordinates=None, probe=None,
             ipsf=None):
        
        """ Load data into the ptychography class.
        
        Available keyword inputs:
            sample: if generating simulated diffraction patterns, supply
                the real-space sample here
            modulus: if reconstructing diffraction patterns, supply
                them here as an array of shape (M,N,N)
            coordinates: for generating and reconstructing, supply
                coordinates in the form [(y0,x0),(y1,x1)...]
            probe: for generating and reconstructing, supply a probe
                function here
            ipsf: for generating and reconstructing, supply the idft
                of the point-spread function here.
                
        Data can be loaded in pieces. For example, the moduli can all
        be kept but the ipsf or coordinates can change (this is useful
        for optimizing...).
        
        Is there a way to make the dependency check graph-based?
        
        """
        
                
        # check types, sizes, etc
        types = (type(None), np.ndarray)
        assert isinstance(sample, types), "sample must be ndarray if supplied"
        assert isinstance(modulus, types), "modulus must be ndarray if supplied"
        assert isinstance(probe, types), "probe must be ndarray if supplied"
        assert isinstance(ipsf, types), "ipsf must be ndarray if supplied"
        
        ### first, do all the loading that has no dependencies
        
        # load the modulus. should be (MxNxN) array. N should be a power of 2.
        # M is the number of diffraction patterns.
        if modulus != None:
            
            assert modulus.ndim == 3
            modulus = modulus.astype(np.float32)
            
            if self.modulus_state == 2:
                assert modulus.shape == (self.frames_m,)+self.shape
            
            if self.modulus_state != 2:
                assert modulus.shape[-2] == modulus.shape[-1]
                
                self.frames_m = modulus.shape[0]
                self.frames_n = modulus.shape[-1]
                self.shape = modulus[0].shape
                self.size = modulus[0].size

                # allocate memory for psi_n^j and fourier intermediates
                nrr = np.random.random
                for wave_n in range(self.frames_m):
                    tmp = (nrr(self.shape)+I*nrr(self.shape))
                    tmp = tmp.astype(np.complex64)
                    name = 'wave'+str(wave_n)
                    exec(self._alloc_str(name, self.shape, 'np.complex64'))
                    exec(self._set_str(name, 'tmp'))

                # allocate buffers for fourier space operations
                for name in ('psi_out', 'psi_fourier', 'fourier_div',
                             'psi_in', 'product'):
                    exec(self._alloc_str(name, self.shape, 'np.complex64'))

            # make the fft plan.
            if self.use_gpu:
                from pyfft.cl import Plan
                self.fftplan = Plan((self.frames_n, self.frames_n), \
                    queue=self.queue)
                
            # load the modulus
            
            for w_n in range(self.frames_m):
                name = 'modulus'+str(w_n)
                exec(self._alloc_str(name, self.shape, 'np.float32'))
                exec(self._set_str(name, 'modulus[w_n]', dtype='np.float32'))
                
            self.modulus_state = 2
            
            # get new kernels built, along with executable strings for them
            # SAUSAGE GETTING MADE
            if self.use_gpu:
                self._build_o_update_kernel()
                #self._build_p_update_kernel()
            
            # clear the object
            self.clear_object()
            
        # load the probe (for generation of diffraction patterns) or an
        # initial estimate of the probe (for reconstruction). needs to
        # be square
        if probe != None:
            
            assert probe.ndim == 2
            assert probe.shape[0] == probe.shape[1]
            
            self.probe = self._allocate(probe.shape, np.complex64, 'probe')
            self.probe = self._set(probe.astype(np.complex64), self.probe)

            # need these for implementing the update
            self.p_top = self._allocate(probe.shape, np.complex64, 'probe')
            self.p_btm = self._allocate(probe.shape, np.float32, 'probe')
            
            self.probe_state = 2
            
        # load the sample for generation of patterns
        if sample != None:
            
            assert sample.ndim == 2
            
            self.sample = self._allocate(sample.shape, np.complex64, 'sample')
            self.sample = self._set(sample.astype(np.complex64), self.sample)
            self.sample_state = 2
            
        # load the ipsf for generation or reconstruction
        if ipsf != None:
            
            assert ipsf.ndim == 2
            assert ipsf.shape[0] == ipsf.shape[1]
        
            self.ipsf = self._allocate(ipsf.shape, np.complex64, 'ipsf')
            self.ipsf = self._set(ipsf.astype(np.complex64), self.ipsf)
            self.ipsf_state = 2
        
        # load the coordinates of the probe for generation or
        # reconstruction. needs to be an iterable of iterables of length2.
        # for example: [(y0,x0),(y1,x1)....(yn,xn)]
        if coordinates != None:
            
            assert isinstance(coordinates, self.iterables)
            for coord in coordinates:
                assert isinstance(coord, self.iterables)
                assert len(coord) == 2
                
            self.coordinates = coordinates
            self.r_coords = self._allocate((len(self.coordinates),), \
                np.int32, 'r_coords')
            self.c_coords = self._allocate((len(self.coordinates),), \
                np.int32, 'c_coords')
            
            self.coordinates_state = 2
            
    def reconstruct(self, iterations=15, propagator='auto'):
        
        """ Reconstruct a set of overlapping diffraction patterns.
        
        Needs modulus, probe, and coordinates to all be loaded. """

        self._check_states_for_reconstruction()
        self._change_coordinates()

        if not self.use_gpu and HAVE_FFTW:
            pyfftw.interfaces.cache.enable()

        if self.can_reconstruct:
            
            if self.object == None: self._make_object()
            self._set_propagator(propagator)
        
            for self.iteration in range(iterations):
                
                # 1. update the object (updates o_top, o_btm, object)
                self._update_object()
                
                # 2. update the probe (but not always!)
                if self._can_update_probe():
                    self._update_probe()

                # 3. update the waves
                self._update_waves()

            self.iterations = iterations
            
        if not self.use_gpu and HAVE_FFTW:
            pyfftw.interfaces.cache.disable()

    def reset_phi(self):
        """ Replace all self.waveX with random complex numbers """
        
        for wave_n in range(self.frames_m):
            tmp = np.random.random(self.shape)+1j*np.random.random(self.shape)
            exec('self.wave%s = self._set(tmp.astype(np.complex64),\
                 self.wave%s)'%(wave_n, wave_n))

    def _alloc_str(self, name, shape, dtype):
        return self.alloc%(name, shape, dtype, name)
    
    def _set_str(self, name, arrayname, dtype=None):
        if dtype == None:
            return self.setstr%(name, arrayname, name)
        else:
            return self.setstr2%(name, arrayname, dtype, name)

    def _change_coordinates(self):
        # displace all the coordinates by the min
        
        rows = np.array([x[0] for x in self.coordinates]).astype(np.int32)
        cols = np.array([x[1] for x in self.coordinates]).astype(np.int32)  
        rows -= rows.min()
        cols -= cols.min()
        
        self.r_coords = self._set(rows, self.r_coords)
        self.c_coords = self._set(cols, self.c_coords)
        
        tmp = [(rows[i], cols[i]) for i in  range(len(self.coordinates))]
        self.coordinates = tmp

    def _build_o_update_kernel(self):
        """ The o update requires looping over a variable number of buffers.
        For unknown reasons, combining them into a 3d array seems to seriously
        hurt performance. Instead, try making a kernel which passes ALL the
        buffers as arguments. """
        
        # these are components for the o update kernel
        header = """
        __kernel void execute(
            __global float2* object,
            __global float2* probe,
            __global int* rc,
            __global int* cc,
            int L,
            %s) {
        """
        
        repeat = """
        //////// begin repeated block
            p_i   = i-rc[%s];
            p_j   = j-cc[%s];
            
            // only try to grab elements if the index is valid!
            if (p_i > -1 && p_i < L && p_j > -1 && p_j < L) {
        
                p_idx = p_i*L+p_j;
                
                p = probe[p_idx];
                w = wave%s[p_idx];
                
                sum2 += (float2) (p.x*w.x+p.y*w.y,p.x*w.y-p.y*w.x);
                sum1 += p.x*p.x+p.y*p.y;
            }
        //////// end repeated block
        """
        
        indexes = """
            int i = get_global_id(0);
            int j = get_global_id(1);
            int J = get_global_size(1);
            
            float2 sum2 = (float2) (0.0, 0.0); // for o_top
            float  sum1 = 0.0; // for o_btm
            
            int o_idx = i*J+j;
            int p_idx;
            int p_i;
            int p_j;
            
            float2 p; // variable for selecting data from probe
            float2 w; // variable for selecting data from waves
            
        """
        
        final = """

            float divx = sum2.x/sum1;
            float divy = sum2.y/sum1;
            
            if (isnan(divx) || isnan(divy)) {
                divx = 0;
                divy = 0;
            }

            object[o_idx] = (float2) (divx, divy);
            }"""
        
        # build the kernel string
        tmp = ['__global float2* wave%s'%i for i in range(self.frames_m)]
        global_ptrs = ',\n'.join(tmp)
        repeats = ''.join([repeat%(i, i, i) for i in range(self.frames_m)])
        k_str = header%global_ptrs+indexes+repeats+final

        # compile the kernel
        self.o_update_kernel = gpu.build_kernel(self.context,
                                                self.device, k_str)
        
        # make a string with all the right arguments which can be
        # invoked as a command using exec
        
        arrays = 'self.queue,self.object.shape,None,self.object.data, \
                  self.probe.data,self.r_coords.data, \
                  self.c_coords.data,np.int32(self.frames_n),'
        execute = 'self.o_update_kernel.execute('
        tmp = ['self.wave%s.data'%i for i in range(self.frames_m)]
        joined = ','.join(tmp)
        execute += arrays+joined+').wait()'
        
        self.o_update = execute
        
    def _build_p_update_kernel(self):
        
        header = """__kernel void execute(
            __global float2* probe, // this is where we write to
            __global float2* object, // this is the estimate of the object
            __global int* rc, // row coordinates
            __global int* cc, // col coordinates
            int L,            // object columns
            %s) {
            """

        static_in = """
            // indices
            int i = get_global_id(0);
            int j = get_global_id(1);
            int N = get_global_size(1);
    
            // sums
            float2 sum1 = (float2) (0, 0);
            float  sum2 = 0;
    
            // others
            float2 o; // data from object
            float2 w; // data from wave
            int r;    // row offset
            int c;    // col offset"""
            
        static_out = """ // write to output
        
            float divx = sum1.x/sum2;
            float divy = sum1.y/sum2;
            
            if (isnan(divx)) {
                divx = 0;
                divy = 0;
            }
        
            probe[i*N+j] = (float2) (divx, divy);
            }"""
            
        repeat = """
        
            // repeated section. get the row and column offsets; pull
            // object data; add correct product/quotient to sums
            r = rc[%s];
            c = cc[%s];
    
            o = object[(i+r)*L+j+c];
            w = wave%s[i*N+j];
    
            sum1 += (float2) (o.x*w.x+o.y*w.y, -o.y*w.x-o.x*w.y);
            sum2 += o.x*o.x+o.y*o.y;"""

        # build the kernel string
        tmp = ['__global float2* wave%s'%i for i in range(self.frames_m)]
        global_ptrs = ',\n'.join(tmp)
        repeats = ''.join([repeat%(i, i, i) for i in range(self.frames_m)])
        k_str = header%global_ptrs+static_in+repeats+static_out
        
        # compile the kernel
        self.o_update_kernel = gpu.build_kernel(self.context,
                                                self.device, k_str)
        
        # make a string with all the right arguments which can be
        # invoked as a command using exec
        
        arrays = 'self.queue, self.probe.shape,None, self.probe.data, \
                  self.object.data, self.r_coords.data, \
                  self.c_coords.data, np.int32(self.frames_n),'
        execute = 'self.o_update_kernel.execute('
        joined = ','.join(['self.wave%s.data'%i for i in range(self.frames_m)])
        execute += arrays+joined+').wait()'

        self.p_update = execute
        
    def _can_update_probe(self):
        """ Helper: run probe update this iteration? """
        b1 = self.iteration > self.probe_update_threshold
        b2 = self.probe_update_threshold > 0
        return b1 and b2

    def _check_states_for_generation(self):
        
        """ Check that all necessary data has been loaded and
        is compatible for generation of diffraction patterns """
        
        self.can_generate = False
        generate_problems = []
        
        if self.sample_state == 0:
            generate_problems.append("no sample")
            
        if self.probe_state == 0:
            generate_problems.append("no probe")
            
        if self.coordinates_state == 0:
            generate_problems.append("no coordinates")

        # need sample, probe, coordinates
        b1 = self.sample_state == 2
        b2 = self.probe_state == 2
        b3 = self.coordinates_state == 2
        if b1 and b2 and b3:
            
            # sample needs to be bigger than probe
            if self.sample.shape >= self.probe.shape:
                self.can_generate = True
            else:
                generate_problems.append('sample array is smaller than probe')
                
        if self.probe_state == 2 and self.ipsf_state == 2:
            
            if self.ipsf.shape != self.probe.shape:
                self.can_generate = False
                generate_problems.append('ipsf and probe are different size')
                
        if not self.can_generate:
            print "cant generate diffraction patterns:"
            for message in generate_problems:
                print "    "+message

    def _check_states_for_reconstruction(self):
        """ Check loading and consistency of data for reconstruction"""
                
        self.can_reconstruct = False
        problems = []
        
        # need modulus, probe, coordinates
        b1 = self.modulus_state == 2
        b2 = self.probe_state == 2
        b3 = self.coordinates_state == 2
        if b1 and b2 and b3:
            
            self.can_reconstruct = True
            
            # probe and modulus need to be the same size
            if self.shape != self.probe.shape:
                self.can_reconstruct = False
                problems.append('probe, modulus are different shapes')
                
        if self.modulus_state == 2 and self.ipsf_state == 2:
            if self.ipsf.shape != self.shape:
                self.can_reconstruct = False
                problems.append('ipsf, modulus are different shapes')
                
        if self.modulus_state == 2 and self.coordinates_state == 2:
            if self.frames_m != len(self.coordinates):
                self.can_reconstruct = False
                problems.append('number of moduli and coordinates are unequal')
                
        if not self.can_reconstruct:
            print "cant reconstruct diffraction patterns"

            for message in problems:
                print "    "+message

    def _convolvef(self, to_convolve, kernel, convolved=None):
        """
        calculate a convolution when to_convolve must be transformed but
        kernel is already transformed. the multiplication function depends
        on the dtype of kernel. """
        
        if self.use_gpu:
            msg1 = "input to_convolve has wrong dtype for fftplan"
            msg2 = "convolved output has wrong dtype"
            assert to_convolve.dtype == 'complex64', msg1
            assert convolved.dtype == 'complex64', msg2
        
            self._fft2(to_convolve, convolved)
            self._cl_mult(convolved, kernel, convolved)
            self._fft2(convolved, convolved, inverse=True)
            
        if not self.use_gpu:
            return self.ifft2(self.fft2(to_convolve)*kernel)

    def _fft2(self, data_in, data_out, inverse=False):
        """ A unified wrapper for ffts. If using the gpu, runs
        self.fftplan on data_in, storing it in data_out. If using
        the cpu, runs fftw or np.fft.ff2 on data_in, storing it
        in data_out.
        
        If inverse = True, the inverse transform is calculated. This
        convention was adopted from pyfft."""
        
        if self.use_gpu:
            self.fftplan.execute(data_in=data_in.data, data_out=data_out.data,
                                 inverse=inverse)
        else:
            if inverse:
                data_out = self.ifft2(data_in)
            else:
                data_out = self.fft2(data_in)
                
        return data_out

    def _fourier_constraint(self, psi_in, constraint_modulus, out):
        
        """ Enforce the usual fourier constraint. """
        
        def _divisor():
            
            psi = self.psi_fourier
            div = self.fourier_div
            
            # step one: convert to modulus
            if self.use_gpu:
                self._cl_abs(psi, div)
            else:
                div = np.abs(psi)
            
            # step two: if a transverse coherence estimate has been supplied
            # through the ipsf, use it to blur the modulus.
            if self.ipsf_state == 2 and self.propagator == 'pc':
                
                psf = self.ipsf
                
                if self.use_gpu:
                    self._cl_mult(div, div, div)
                    self._convolvef(div, psf, div)
                    self._cl_abs(div, div)
                    self._cl_sqrt(div, div)
                    
                else:
                    div = np.sqrt(self._convolvef(div**2, psf))
                    div = np.sqrt(np.abs(div))
                    
            return div

        # transform psi_in; store in self.psi_fourier
        self.psi_fourier = self._fft2(psi_in, self.psi_fourier)

        # make the divisor (this is where partial coherence correction goes)
        self.fourier_div = _divisor()

        # magnitude replacement
        if self.use_gpu:
            self._kexec('fourier', self.psi_fourier, self.fourier_div,
                        constraint_modulus, self.psi_fourier)
        else:
            modulus = constraint_modulus
            psif = self.psi_fourier
            fdiv = self.fourier_div
            if HAVE_NUMEXPR:
                self.psi_fourier = numexpr.evaluate("modulus*psif/abs(fdiv)")
                self.psi_fourier = self.psi_fourier.astype(np.complex64)
            else:
                self.psi_fourier = modulus*psif/np.abs(fdiv)

        # inverse transform
        out = self._fft2(self.psi_fourier, out, inverse=True)
                
        return out
 
    def _make_object(self):
        
        # make the object (thibault's O(r)) based on probe
        # size and coordinates.
        
        rows = [x[0] for x in self.coordinates]
        cols = [x[1] for x in self.coordinates]
        
        nrows = max(rows)-min(rows)+self.probe.shape[0]
        ncols = max(cols)-min(cols)+self.probe.shape[1]
        
        # make the size of the object divisible by 16 in each
        # axis for gpu computations. (workgroup (16,16) is usually
        # fastest). for cpu, modulo doesnt matter
        if self.use_gpu:
            if nrows%16 != 0:
                nrows = (nrows/16+1)*16
            if ncols%16 != 0:
                ncols = (ncols/16+1)*16
        
        obj = np.random.rand(nrows, ncols)+1j*np.random.rand(nrows, ncols)

        # we need a top, bottom, and quotient for the object
        self.object = self._allocate(obj.shape, np.complex64, name='object')
        self.object = self._set(obj.astype(np.complex64), self.object)
        self.object.fill(0)
        
        self.o_top = self._allocate(obj.shape, np.complex64, name='o_top')
        self.o_btm = self._allocate(obj.shape, np.float32, name='o_btm')

    def _rolls(self, array, roll_y, roll_x):
        return np.roll(np.roll(array, roll_y, 0), roll_x, 1)

    def _set_propagator(self, propagator):
        
        assert isinstance(propagator, str)
        propagator = propagator.lower()
        assert propagator in ('auto', 'pc', 'fc')
        
        if propagator == 'auto' and self.ipsf_state == 2:
            propagator = 'pc'
        if propagator == 'auto' and self.ipsf_state != 2:
            propagator = 'fc'
        self.propagator = propagator

    def _update_coherence(self, iterations=100):
        """ Update the estimate of the coherence function by
        using Richardson-Lucy deconvolution of the measured modulus
        by the current estimate of the modulus. We do a deconvolution
        for each view, then use the average outcome as the new ipsf.
        
        This method is a cut-down method of that originally written for
        phasing; only RLD is offered as an estimate of ipsf.
        """
        
        import shape

        # check gpu
        global use_gpu
        old_use_gpu = use_gpu
        device = 'gpu'
        if force_cpu:
            use_gpu = False
            common.use_gpu = False
            device = 'cpu'

        self.counter = 0
        
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
            if use_gpu:
                self._cl_zero(self.rl_sum)
            else:
                self.rl_sum = np.zeros_like(self.rl_sum)
            
        def _finalize_rl():
            """ Finalize RL deconvolution """
            
            # to be power preserving, the ipsf should be 1 at (0,0)?
            ipsf = self.get(self.rl_u)
            ipsf *= 1./abs(ipsf[0, 0])
            
            # load new ipsf
            self.load(ipsf=ipsf)
            self.rl_state = 0 # why?
        
        def _opt_iter_rl():
            """
            # implement the rl deconvolution algorithm.
            # explanation of quantities:
            # p is the point spread function (the reconstructed intensity)
            # d is the measured partially coherent intensity
            # u is the estimate of the psf, what we are trying to reconstruct
            """
            
            # convolve u and p. this should give a blurry intensity
            if use_gpu:
                self._fft2(self.rl_u, self.rl_blur)
                self._cl_mult(self.rl_blur, self.rl_p, self.rl_blur)
                self._fft2(self.rl_blur, self.rl_blur, inverse=True)
            else:
                self.rl_blur = np.fft.ifft2(np.fft.fft2(self.rl_u)*self.rl_p)
                
            # divide d by the convolution
            if use_gpu:
                self._cl_div(self.rl_d, self.rl_blur, self.rl_blur)
            else:
                self.rl_blur = self.rl_d/self.rl_blur

            # convolve the quotient with phat
            if use_gpu:
                self._fft2(self.rl_blur, self.rl_blur)
                self._cl_mult(self.rl_blur, self.rl_ph, self.rl_blur)
                self._fft2(self.rl_blur, self.rl_blur, inverse=True)
            else:
                tmp = np.fft.fft2(self.rl_blur)*self.rl_ph
                self.rl_blur = np.fft.ifft2(tmp)
                
            # multiply u and blur to get a new estimate of u
            if use_gpu:
                self._cl_mult(self.rl_u, self.rl_blur, self.rl_u)
            else:
                self.rl_u *= self.rl_blur
        
        def _postprocess_rl():
            """Post process the RL deconvolution """

            # make rl_u into the ipsf with an fft
            if use_gpu:
                self._fft2(self.rl_u, self.rl_u)
            else:
                self.rl_u = np.fft.fft2(self.rl_u)
                
            # add rl_u to rl_sum
            if use_gpu:
                self._cl_add(self.rl_sum, self.rl_u, self.rl_sum)
            else:
                self.rl_sum += self.rl_u.astype(c)
                
            # reset types to single precision from double precision for cpu path
            d = np.float32
            if not use_gpu:
                self.rl_u = self.rl_u.astype(d)
                self.rl_p = self.rl_p.astype(d)
                self.rl_blur = self.rl_blur.astype(d)
                
            if not silent:
                print "finished richardson-lucy estimate %s of %s"\
                %(n+1, self.frame_m)

        def _preprocess_rl():
            """ Preprocess the RL deconvolution
            
            self.rl_p is the "best estimate" in real space. Make its fourier
            intensity.
            # precompute the fourier transform for the convolutions.
            # make modulus into intensity
            """
            
            m2 = (modulus**2)
            m2 /= m2.sum()
            g = np.fft.fftshift(shape.gaussian(s, (s[0]/4, s[1]/4)))
            
            self.rl_p = self._set(active.astype(c), self.rl_p)
            self.rl_d = self._set(m2.astype(c), self.rl_d)
            self.rl_u = self._set(g.astype(c), self.rl_u)

            if use_gpu:
                # fourier modulus of best_estimate
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

        def _start(frame_n):
            """ Get the modulus and estimate for the current view number"""
            
            wave = self.get(eval('self.wave%s'%frame_n))
            fmod = self.get(eval('self.modulus'%frame_n))
            
            tmp = np.zeros(fmod.shape, c)
            if wave.shape != fmod.shape:
                tmp[:wave.shape[0], :wave.shape[1]] = wave
                wave = tmp
            
            return wave, fmod
               
        # allocate memory. s, f, c are local variables
        # for this function only. active_ holds the
        # current view and its goal modulus
        s, f, c = self.shape, np.float32, np.complex64
        _allocate_rl()
        
        # loop over the frames, performing a de
        for frame_n in range(self.frame_m):
            active, modulus = _start(frame_n)
            
            # load initial values and preprocess
            _preprocess_rl()
            
            # now run the deconvolver for a set number of iterations
            for opt_i in range(iterations):
                _opt_iter_rl()
            
            # make rl_u into ipsf with fft; add to sum
            _postprocess_rl()
                    
        use_gpu = old_use_gpu
        common.use_gpu = old_use_gpu
            
        _finalize_rl()

    def _update_object(self):
        # implement equation s7 in thibault science 2008 "high resolution..."
        
        if self.use_gpu:
            exec(self.o_update)

        if not self.use_gpu:
            self.o_top.fill(0)
            self.o_btm.fill(0)
            self.pstar = np.conjugate(self.probe)
            self.p2 = np.abs(self.probe)**2

            for wave_n, coord in enumerate(self.coordinates):
                row1, row2 = coord[0], coord[0]+self.shape[0]
                col1, col2 = coord[1], coord[1]+self.shape[1]
                
                self.o_top[row1:row2, col1:col2] += \
                self.pstar*eval('self.wave%s'%wave_n)
                
                self.o_btm[row1:row2, col1:col2] += self.p2
                
            self.object = np.nan_to_num(self.o_top/self.o_btm)
  
    def _update_probe(self):
        # implement equation s8 in thibault science 2008 "high resolution..."
        
        self.p_top.fill(0)
        self.p_btm.fill(0)

        for wave_n, coord in enumerate(self.coordinates):
            row1, row2 = coord[0], coord[0]+self.shape[0]
            col1, col2 = coord[1], coord[1]+self.shape[1]
            if self.use_gpu:
                self._kexec('update_p', self.p_top, self.p_btm, self.object,
                            eval('self.wave%s'%wave_n), row1, col1,
                            self.object.shape[1], shape=self.shape)
            else:
                tmp = self.object[row1:row2, col1:col2]
                self.p_top += np.conjugate(tmp)*eval('self.wave%s'%wave_n)
                self.p_btm += np.abs(tmp)**2

    def _update_waves(self):
        # implement equation s9 in thibault science 2008 "high resolution..."
        
        for wave_n, coord in enumerate(self.coordinates):
            
            # form the product o*p and psi_in for the constraint
            row1, row2 = coord[0], coord[0]+self.shape[0]
            col1, col2 = coord[1], coord[1]+self.shape[1]
            if self.use_gpu:
                self._kexec('dm_product', self.probe, self.object,
                            self.product, eval('self.wave%s'%wave_n),
                            self.psi_in, row1, col1, self.object.shape[1],
                            shape=self.probe.shape)
            else:
                self.product = self.probe*self.object[row1:row2, col1:col2]
                self.psi_in = 2*self.product-eval('self.wave%s'%wave_n)

            # satisfy the fourier constraint
            fc = self._fourier_constraint
            self.psi_out = fc(self.psi_in, eval('self.modulus%s'%wave_n),
                              self.psi_out)

            # update the wave
            if self.use_gpu:
                self._kexec('update_wave', eval('self.wave%s'%wave_n),
                            self.psi_out, self.product)
            else:
                to_exec = 'self.wave%s = self.wave%s+self.psi_out-self.product'
                exec(to_exec%(wave_n, wave_n))

def find_overlap_displacement(aperture, overlap_ratio):
    """ Given an aperture function, attempt to calculate how many
    pixels the aperture must be displaced to generate a desired
    overlap ratio. The overlap ratio is defined as:
    
    2*sum(aperture*displaced_aperture)/sum(aperture+displaced_aperture)
    
    So two non-overlapping apertures will have ratio 0 and two
    toally overlapping apertures will have ratio 1. The overlap_ratio
    therefore occurs in the range [0,1]."""
    
    assert isinstance(aperture, np.ndarray)
    assert aperture.ndim == 2
    overlap_ratio = float(overlap_ratio)
    assert overlap_ratio >= 0
    assert overlap_ratio <= 1
    
    aperture = np.abs(aperture)
    
    overlap = 1
    shift = 0
    while overlap >= overlap_ratio:
        shift += 1
        displaced = np.roll(aperture, shift, 0)
        overlap = 2.*np.sum(aperture*displaced)/np.sum(aperture+displaced)
    
    if shift > 0:
        return shift-1
    else:
        return 0
    
def make_raster(grid_size, delta, center=None):
    """ Generate raster-grid coordinates for ptychography.
    
    Required input:
        grid_size: (rows,columns)
        delta: (row_delta, column_delta) step size of along each axis.
        
    Optional input:
        center: (row,column)
        
    Returns:
        list of tuples, each tuple is a (row,column) coordinate pair.
        
    """
    
    assert isinstance(grid_size, (list, tuple, np.ndarray))
    assert len(grid_size) == 2
    
    assert isinstance(delta, (list, tuple, np.ndarray))
    assert len(delta) == 2
    
    has_center = False
    assert isinstance(center, (list, tuple, np.ndarray, type(None), str))
    if isinstance(center, (list, tuple, np.ndarray)):
        assert len(center) == 2
        has_center = True
    if isinstance(center, str):
        center = (0, 0)
        has_center = True
        
    import itertools
    
    rows = np.arange(grid_size[0])*delta[0]
    if has_center:
        rows += center[0]-rows.mean()
    rows = rows.astype(int)
    
    cols = np.arange(grid_size[1])*delta[1]
    if has_center:
        cols += center[1]-cols.mean()
    cols = cols.astype(int)
    
    return [(x[0], x[1]) for x in itertools.product(rows, cols)]
    
    
    

    