import pyopencl

def init():
    """ Initialize the GPU with all the pyopencl magic words. By default, takes the first
    GPU in the list. Multi-GPU computers is too esoteric to consider."""
    
    platforms = pyopencl.get_platforms()
    device    = platforms[0].get_devices(pyopencl.device_type.GPU)[0]
    context   = pyopencl.Context([device])
    queue     = pyopencl.CommandQueue(context)
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
    
