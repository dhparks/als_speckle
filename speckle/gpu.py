import numpy as np

class arrayWrapper(object):
    
    def __init__(self, array, name):
        self.array = array
        self.name = name
        self.isgpu = True
        
    def __getattr__(self, x):
        if x == 'name':
            return self.name
        if x == 'isgpu':
            return self.isgpu
        return getattr(self.array, x)
    
class common:
    """ A superset class for the various unified cpu/gpu modules. The purpose
    of this class is just to hold those methods which are common to all gpu
    operations. """
    
    def __init__(self):
        pass

    def _allocate(self, size, dtype, name=None):
        """ Wrapper to define new arrays whether gpu or cpu path"""
        if self.use_gpu:
            import pyopencl.array as cla
            x = cla.empty(self.queue, size, dtype)
            y = arrayWrapper(x, name)
            return y
        else:
            return np.zeros(size, dtype)
        
    def _allocate_local(self, size, name=None):
        if self.use_gpu:
            import pyopencl
            x = pyopencl.LocalMemory(size)
            x.name = name
            return x
        else:
            return None
    
    def _cl_abs(self, in1, out, square=False):
        """ Wrapper func to the various abs kernels. Checks types of in1 and out
        to select appropriate kernel. """
        
        d1, d2 = in1.dtype, out.dtype
        assert d1 in self.array_dtypes, "in1 dtype not recognized"
        assert d2 in self.array_dtypes, "out dtype not recognized"
        assert in1.shape == out.shape,  "in1 and out must have same shape"
        N = in1.size
    
        if square == False:
            # usual abs operation
            if d1 == 'float32':
                assert d2 == 'float32'
                func = 'abs_f'
            if d1 == 'complex64':
                if d2 == 'float32': func = 'abs_f2_f'
                if d2 == 'complex64': func = 'abs_f2_f2'
        
        if square == True:
            # square the abs. this saves some time.
            if d1 == 'float32':
                assert d2 == 'float32'
                func = 'abs2_f'
            if d1 == 'complex64':
                if d2 == 'float32': func = 'abs2_f2_f'
                if d2 == 'complex64': func = 'abs2_f2_f2'
        
        self._kexec(func,in1,out)
   
    def _cl_add(self, in1, in2, out, **kwargs):
        d1, d2, d3 = in1.dtype, in2.dtype, out.dtype
        assert d1 in self.array_dtypes, "in1 dtype not recognized; is %s"%d1
        assert d2 in self.array_dtypes, "in2 dtype not recognized; is %s"%d2
        assert d3 in self.array_dtypes, "out dtype not recognized; is %s"%d3
        assert in1.shape == in2.shape and in1.shape == out.shape, "all arrays must have same shape"
        N = in1.size
        
        if d1 == 'float32':
            if d2 == 'float32':
                func = 'add_f_f'
                assert d3 == 'float32', "float + float = float"
                arg1 = in1
                arg2 = in2
                
            if d2 == 'complex64':
                func = 'add_f_f2'
                assert d3 == 'complex64', "float + complex = complex"
                arg1 = in1
                arg2 = in2
                
        if d2 == 'complex64':
            if d2 == 'float32':
                func = 'add_f_f2'
                assert d3 == 'complex64', "float + complex = complex"
                arg1 = in2
                arg2 = in1
                
            if d2 == 'complex64':
                func = 'add_f2_f2'
                assert d3 == 'complex64', "complex + complex = complex"
                arg1 = in1
                arg2 = in2
                
        self._kexec(func, arg1, arg2, out, **kwargs)
        
    def _cl_copy(self, in1, out):
        d1, d2 = in1.dtype, out.dtype
        assert d1 in self.array_dtypes, "in1 dtype not recognized"
        assert d2 in self.array_dtypes, "out dtype not recognized"
        assert in1.shape == out.shape,  "in1 and out must have same shape"
        N = in1.size
        
        if d1 == 'float32':
            if d2 == 'float32':   func = 'copy_f_f'
            if d2 == 'complex64': func = 'copy_f_f2'
        if d1 == 'complex64':
            if d2 == 'complex64':
                func = 'copy_f2_f2'
            if d2 == 'float32':
                func = 'copy_f2_f'
            
        self._kexec(func,in1,out)
            
    def _cl_div(self,in1,in2,out):
        """ Wrapper func to various division kernels. Checks type of in1, in2,
        and out to select the appropriate kernels.
        
        in1, in2 are the numerator and denominator so ORDER MATTERS
        out is the output
        
        All arguments should be the pyopencl array, NOT the data attribute.
        """
        
        d1, d2, d3 = in1.dtype, in2.dtype, out.dtype
        assert d1 in self.array_dtypes, "in1 dtype not recognized; is %s"%d1
        assert d2 in self.array_dtypes, "in2 dtype not recognized; is %s"%d2
        assert d3 in self.array_dtypes, "out dtype not recognized; is %s"%d3
        assert in1.shape == in2.shape and in1.shape == out.shape, "all arrays must have same shape"
        N = in1.size
            
        arg1, arg2 = in1, in2

        if d1 == 'float32':
            if d2 == 'float32':
                func = 'divide_f_f'
                assert d3 == 'float32', "float / float = float"
                
            if d2 == 'complex64':
                func = 'divide_f_f2'
                assert d3 == 'complex64', "float / complex = complex"

        if d1 == 'complex64':
            if d2 == 'float32':
                func = 'divide_f2_f'
                assert d3 == 'complex64', "complex / float = complex"
                
            if d2 == 'complex64':
                func = 'divide_f2_f2'
                assert d3 == 'complex64', "complex / complex = complex"

        self._kexec(func, arg1, arg2, out)
        
    def _cl_map2d(self, in1, out, x_plan, y_plan, **kwargs):
        """ Wrapper for two places where map_coords_f gets called."""
        
        # check types
        s, d1, d2 = out.shape, in1.dtype, out.dtype
        
        assert d1 == d2
        assert x_plan.dtype == 'float32'
        assert y_plan.dtype == 'float32'
        assert s == x_plan.shape
        assert s == y_plan.shape
        
        r_in = np.int32(in1.shape[0])
        c_in = np.int32(in1.shape[1])
        r_out = s[0]
        c_out = s[1]
        
        if d1 == 'float32':   func = 'map_coords_f'
        if d1 == 'complex64': func = 'map_coords_f2'
        
        self._kexec(func,in1,c_in,r_in,out,x_plan,y_plan,np.int32(1),**kwargs)
                
    def _cl_mult(self, in1, in2, out, **kwargs):
        """ Wrapper function to the various array-multiplication kernels. Every
        combination of input requires a different kernel. This function checks
        dtypes and automatically selects the appropriate kernel.
        
        in1, in2 are the input arrays being multiplied
        out is the output array where the result is stored
        
        All passed arguments should be the pyopencl array, NOT the data attribute.
        """
        
        d1, d2, d3 = in1.dtype, in2.dtype, out.dtype
        acceptable_types = ('float32','complex64')

        assert d1 in acceptable_types, "in1 dtype not recognized; is %s"%d1
        assert d2 in acceptable_types, "in2 dtype not recognized; is %s"%d2
        assert d3 in acceptable_types, "out dtype not recognized; is %s"%d3
        N = in1.size
        
        def _assert_shapes(*args):
            # this makes sure that all the arrays have the same shape.
            # not all inputs are required to be arrays.
            s = None
            for a in args:
                if s == None:
                    try:
                        a.isgpu
                        s = a.shape
                    except AttributeError: pass
                if s != None:
                    try:
                        a.isgpu
                        s = a.shape
                        assert s == a.shape, "arrays are incommensurate %s %s"%(s,a.shape)
                    except AttributeError: pass

        _assert_shapes(in1,in2,out)

        try:
            in1.isgpu
            
            try:
                in2.isgpu
        
                # this means both are arrays
                if (d1 == 'complex64' or d2 == 'complex64'): assert d3 == 'complex64'
                
                if d1 == 'float32':
                    if d2 == 'float32':
                        if d3 == 'float32':   func = 'multiply_f_f'
                        if d3 == 'complex64': func = 'multiply_f_f_f2'
                        arg1 = in1
                        arg2 = in2
                    if d2 == 'complex64': func,arg1,arg2 = 'multiply_f_f2', in1, in2
                    
                if d1 == 'complex64':
                    if d2 == 'float32':   func,arg1,arg2 = 'multiply_f_f2',  in2, in1
                    if d2 == 'complex64': func,arg1,arg2 = 'multiply_f2_f2', in1, in2
                        
            except AttributeError: # this means in2 is a scalar
                assert d2 == 'float32'
                assert d1 == d3
                arg1 = in2
                arg2 = in1
                if d1 == 'float32':   func = 'multiply_s_f'
                if d1 == 'complex64': func = 'multiply_s_f2'
                
        except AttributeError: # this means in1 is a scalar
            assert d1 == 'float32'
            assert d2 == d3
            arg1 = in1
            arg2 = in2
            if d1 == 'float32':   func = 'multiply_s_f'
            if d2 == 'complex64': func = 'multiply_s_f2'
            
        self._kexec(func,arg1,arg2,out,kwargs)

    def _cl_sqrt(self, in1, out):
        """ Wrapper func to the various abs kernels. Checks types of in1 and out
        to select appropriate kernel. """
        
        d1, d2 = in1.dtype, out.dtype
        assert d1 in self.array_dtypes, "in1 dtype not recognized"
        assert d2 in self.array_dtypes, "out dtype not recognized"
        assert in1.shape == out.shape,  "in1 and out must have same shape"
        N = in1.size

        if d1 == 'complex64':
            if d2 == 'complex64': func = 'sqrt_f2'
            if d2 == 'float32': func = 'sqrt_f2_f'
        if d1 == 'float32':
            assert d2 == 'float32'
            func = 'sqrt_f'
        
        self._kexec(func,in1,out)
        
    def _cl_zero(self, in1):
        """ Wrapper func to set_zero kernels. """
        
        d1 = in1.dtype
        assert d1 in self.array_dtypes
        if d1 == 'float32':   func = 'set_zero_f'
        if d1 == 'complex64': func = 'set_zero_f2'
        
        self._kexec(func,in1)
        
    def _kexec(self, name, *args, **kwargs):
        """ Wrapper function to execute elementwise kernels. If the kernel
        does not exist, try to build it. If the named file cannot be found,
        terminate execution of the program.
        """
        
        assert isinstance(name,str), "name is %s"%name
        import pyopencl.array as cla
        import pyopencl._cl as cl_
        
        # build the command from name and *args and **kwargs. right now the only
        # kwarg is shape, but other options can be added.
        if 'shape' in kwargs:
            shape = kwargs['shape']
            assert isinstance(shape,tuple)
            for x in shape: assert isinstance(x,int)
            
        else:
            sizes = []
            for arg in args:
                try:
                    arg.isgpu
                    sizes.append(arg.size)
                except AttributeError:
                    pass
            shape = '(%s,)'%max(sizes)

        if 'local' in kwargs:
            local = kwargs['local']
            assert isinstance(local,(tuple,type(None)))
            try:
                for x in local: assert isinstance(x,int)
            except TypeError: pass
        else:
            local = None

        # build the command string for execution
        cmd = 'self.%s.execute(self.queue,%s,%s'%(name, shape, local)

        for arg in args:
            if   isinstance(arg, (arrayWrapper, cla.Array)):
                cmd += ',self.%s.data'%arg.name
            elif isinstance(arg, cl_.LocalMemory):
                cmd += ',self.%s'%arg.name
            elif isinstance(arg, self.ints):
                cmd += ',np.int32(%s)'%arg
            elif isinstance(arg, self.floats):
                cmd += ',np.float32(%s)'%arg
            elif isinstance(arg, self.float2s):
                cmd += ',np.complex64(%s)'%arg # not tested
            elif isinstance(arg, dict):
                pass
            else:
                print('no matched instance in cmd %s'%name)
        cmd += ').wait()'

        # try to run the command. this fails if the kernel hasnt yet been built
        try:
            exec(cmd)
        except AttributeError:
        
            # try to build the kernel
            cmd1 = "self.%s = build_kernel_file(self.context, self.device, self.kp+'%s_%s.cl')"%(name, self.project, name)
            cmd2 = "self.%s = build_kernel_file(self.context, self.device, self.kp+'common_%s.cl')"%(name, name)
            
            try:
                exec(cmd1)
            except IOError:
                try:
                    exec(cmd2)
                except IOError:
                    print "problem building kernel file"
                    print cmd1
                    print cmd2
                    print self.kp
                    exit()

            exec(cmd)

    def _set(self, what, where):
        """ Wrapper to move "what" to "where" regardless of device.
        Only applicable to arrays. Care must be taken that, in the case of
        GPUs, memory is allocated first. For CPUs, allocation gets handled
        dynamically by the interpreter. """

        assert what.shape == where.shape, "incompatible shapes %s %s"%(what.shape, where.shape)
        assert what.dtype == where.dtype, "incompatible types %s %s"%(what.dtype, where.dtype)
        if self.use_gpu:
            import pyopencl.array as cla
            where.set(what, queue=self.queue)
        else:
            where = what
        return where
    
    def get(self, something):
        if self.use_gpu:
            try:
                something.isgpu
                return something.get()
            except AttributeError:
                return something
        if not self.use_gpu:
            return something
    
    def start(self, gpu_info=None):
        # inits the gpu. this is the necessary method ONLY IF
        # GPU.INIT WAS NOT CALLED EARLIER TO TURN ON THE GPU.
        
        self.ints    = (int, np.int8, np.int16, np.int32, np.uint8)
        self.floats  = (float, np.float16, np.float32, np.float64)
        self.float2s = (complex, np.complex64, np.complex128)
        
        try:
            import string
            self.kp = string.join(__file__.split('/')[:-1],'/')+'/kernels/'
            if gpu_info == None:
                self.context, self.device, self.queue, self.platform = init()
                self.gpu_info = self.context, self.device, self.queue, self.platform
                import pyfft
            else:
                self.gpu_info = gpu_info
                self.context, self.device, self.queue, self.platform = self.gpu_info
            self.use_gpu = True
        
        except:
            print "couldnt init gpu, reverting to cpu"
            self.use_gpu = False

class GPUInitError(Exception):
    def __init__(self, msg, platform=None, device=None, context=None, queue=None):
        self.msg = msg
        self.platform = platform
        self.device = device
        self.context = context
        self.queue = queue
    def __str__(self):
        out = '%s \n  platform: %s\n  device: %s\n  context: %s\n  queue: %s'%(self.msg,self.platform,self.device,self.context,self.queue)
        return out
    
def init_cpu_cl():
    """ Initialize the CPU with all the pyopencl magic words. Runs slower than GPU, but should use all cores"""

    p_success = False
    d_success = False
    c_success = False
    q_success = False
    
    try: import pyopencl
    except ImportError:
        raise GPUInitError("cant init gpu because no pyopencl")
    
    try:
        platform  = pyopencl.get_platforms()[1]
        p_success = True
    except (pyopencl.LogicError,IndexError):
        error_msg = 'no platform'
        raise GPUInitError(error_msg)
        
    if p_success:
        
        try:
            # now that we found the platform, get the device
            try:
                CPUs = platform.get_devices(pyopencl.device_type.CPU)
            except: 
                CPUs = []
            if len(CPUs) > 0:
                if 'Apple' in str(platform) and 'Intel(R)' in str(CPUs[0]):
                    raise GPUInitError('no gpu and apple+intel crashes fft')
                else:
                    device = CPUs[0]
            if len(CPUs) == 0: raise GPUInitError('platform exists but no devices?!')
            d_success = True
            
        except pyopencl.LogicError:
            error_msg = 'logic error getting devices'
            raise GPUInitError(error_msg,platform=platform)
            
    if d_success:
        try:
            context = pyopencl.Context([device])
            c_success = True
        except pyopencl.LogicError:
            error_msg = 'logic error getting context'
            raise GPUInitError(error_msg,platform=platform,device=device)
            
    if c_success:
        try:
            queue = pyopencl.CommandQueue(context)
            q_success = True
        except pyopencl.LogicError:
            error_msg = 'logic error making queue'
            raise GPUInitError(error_msg,platform=platform,device=device,context=context)
            
    return context,device,queue,platform

def init():
    """ Initialize the GPU with all the pyopencl magic words. By default, takes the first
    GPU in the list. Multi-GPU computers is too esoteric to consider."""

    p_success = False
    d_success = False
    c_success = False
    q_success = False
    
    import time
    
    try: import pyopencl
    except ImportError:
        raise GPUInitError("cant init gpu because no pyopencl")
    
    try:
        platform  = pyopencl.get_platforms()[0]
        p_success = True
    except (pyopencl.LogicError,IndexError):
        error_msg = 'no platform'
        raise GPUInitError(error_msg)

    if p_success:
        
        try:
            # now that we found the platform, get the device
            try:
                GPUs = platform.get_devices(pyopencl.device_type.GPU)
            except:
                GPUs = []
            try:
                CPUs = platform.get_devices(pyopencl.device_type.CPU)
            except: 
                CPUs = []
            if len(GPUs) > 0: device = GPUs[0]
            if len(GPUs) == 0 and len(CPUs) > 0:
                if 'Apple' in str(platform) and 'Intel(R)' in str(CPUs[0]):
                    raise GPUInitError('no gpu and apple+intel crashes fft')
                else:
                    device = CPUs[0]
            if len(GPUs) == 0 and len(CPUs) == 0: raise GPUInitError('platform exists but no devices?!')
            d_success = True
            
        except pyopencl.LogicError:
            error_msg = 'logic error getting devices'
            raise GPUInitError(error_msg,platform=platform)

    # Beginning 1 March 2013 this sleep command is necessary to prevent the GPU driver
    # from hanging. The underlying cause is unknown.
    time.sleep(0.5)
        
    if d_success:
        try:
            context = pyopencl.Context([device])
            c_success = True
        except pyopencl.LogicError:
            error_msg = 'logic error getting context'
            raise GPUInitError(error_msg,platform=platform,device=device)

    if c_success:
        try:
            queue = pyopencl.CommandQueue(context)
            q_success = True
        except pyopencl.LogicError:
            error_msg = 'logic error making queue'
            raise GPUInitError(error_msg,platform=platform,device=device,context=context)

    return context,device,queue,platform

def build_kernel(c,d,kernel):
    
    try: import pyopencl
    except ImportError:
        raise GPUInitError("cant build kernel because no pyopencl")
    
    # Load the program source
    program = pyopencl.Program(c, kernel)

    # Build the program and check for errors
    program.build(devices=[d])

    return program

def build_kernel_file(c,d,fileName):
    kernelFile = open(fileName, 'r')
    kernelStr = kernelFile.read()
    program = build_kernel(c,d,kernelStr)

    return program

def valid(gpu_info):
    """ Attempt to ensure that the current gpu info is correctly formed by
    checking the types of the tuple elements. This is not robust
    as the context and queue may have been corrupted elsewhere. To check
    the integrity of the GPU context, use gpu.challenge.
    
    
    Argument: gpu_info - that which is returned by gpu.init()
    Returns: a boolean True if gpu_info conforms; False otherwise
    """
    
    import pyopencl
    
    assert isinstance(gpu_info,tuple) and len(gpu_info) == 4, "gpu info malformed %s"%gpu_info
    t1 = isinstance(gpu_info[0],pyopencl._cl.Context)
    t2 = isinstance(gpu_info[1],pyopencl._cl.Device)
    t3 = isinstance(gpu_info[2],pyopencl._cl.CommandQueue)
    t4 = isinstance(gpu_info[3],pyopencl._cl.Platform)
    return (t1 and t2 and t3 and t4)
    
def challenge(gpu_info):
    
    # try to run an easy kernel in order to generate an error
    c ,d, q, p = gpu_info
    kp   = string.join(__file__.split('/')[:-1],'/')+'/kernels/'
    abs2 = gpu.build_kernel_file(c, d, kp+'common_abs2_f2_f2.cl')

    test_data = np.arange(128).astype(np.complex64)
    gpu_data  = cla.to_device(queue, c)
    abs2.execute(queue,(int(L),),gpu_data.data,gpu_data.data)
    
    