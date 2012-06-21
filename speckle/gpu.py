import pyopencl

def init(which=0):
    """ Initialize the GPU with all the pyopencl magic words. By default, takes the first
    GPU in the list. Multi-GPU computers is too esoteric to consider."""
    assert isinstance(which,int)
    platforms = pyopencl.get_platforms()
    devices = platforms[0].get_devices(cl.device_type.GPU)
    context = pyopencl.Context([devices[0]])
    device = devices[0]
    queue = pyopencl.CommandQueue(context)
    return context,device,queue

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
    
