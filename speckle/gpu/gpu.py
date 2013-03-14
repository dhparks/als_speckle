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
    
try: import pyopencl
except ImportError:
    raise GPUInitError("importing speckle.gpu failed; presumably no pyopencl install\nfalling back to cpu-python generator code")

def init_cpu_cl():
    """ Initialize the CPU with all the pyopencl magic words. Runs slower than GPU, but should use all cores"""

    p_success = False
    d_success = False
    c_success = False
    q_success = False
    
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
    time.sleep(1)
        
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
    
    # Load the program source
    program = pyopencl.Program(c, kernel)

    # Build the program and check for errors
    program.build(devices=[d])

    return program

def build_kernel_file(c,d,fileName):
    kernelFile = open(fileName, 'r')
    kernelStr = kernelFile.read()

    # Load the program source
    program = pyopencl.Program(c, kernelStr)

    # Build the program and check for errors
    program.build(devices=[d])

    return program