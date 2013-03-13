# import the common libraries
import numpy
import sys
sys.path.insert(0,'..')
import speckle
import speckle.gpu as gpulib
phasing = speckle.phasing

# import the parameters for the reconstruction
import phasing_parameters as params

# if asking for a gpu, check to see if it exists
def check_gpu():
    """ See if a gpu is available and wanted. If not, fall back to cpu.
    
    Returns:
        use_gpu: True or False
        device_info: for initializing gpu. cpu is returned None as a dummy."""

    if params.device == 'gpu':
        try:
            import speckle.gpu as gpulib # this can throw an error
        except gpulib.gpu.GPUInitError as error:
            print error,"\nfalling back to cpu"
            params.device == 'cpu'
            
    if params.device == 'gpu':
        try:
            print "trying to get gpuinfo"
            gpuinfo = gpulib.gpu.init() # this can throw various errors
            print "got it"
        except gpulib.gpu.GPUInitError as error:
            print error, "\nfalling back to cpu"
            params.device = 'cpu'
            
    if params.device == 'gpu': return True, gpuinfo
    if params.device == 'cpu': return False, None

def reconstruct(device_info):
    # canonical code to demonstrate library functions

    # open the data specified in dataname and supportname
    data    = speckle.io.openfits(params.dataname)
    support = speckle.io.open(params.supportname).astype('float')
    support *= 1./support.max()
    
    # find the region which bounds the support. this will prevent having to store and save a bunch
    # of redundant zeros. as used here, this will generate square bounding with a ring of zeros around
    # the outside for margin of error.
    bounds = speckle.masking.bounding_box(support,force_to_square=True,pad=4)
    r0,c0  = bounds[0],bounds[2]
    rows   = int(bounds[1]-bounds[0])
    cols   = int(bounds[3]-bounds[2])
        
    # initialize the GPUPR class provided by the phasing library. initialize the savebuffer in host memory
    N              = len(data)
    reconstruction = phasing_code(device=device_info,N=N,shrinkwrap=params.shrinkwrap)
    save_buffer    = numpy.zeros((params.trials+1,rows,cols),numpy.complex64)
    
    # load the data onto the gpu using the GPUPR methods
    if params.shrinkwrap: reconstruction.load_data(data,support,update_sigma=params.shrinkwrap_sigma)
    if not params.shrinkwrap: reconstruction.load_data(data,support)
    
    for trial in range(params.trials):
        # each reconstruction follows the same procedures but with a different random input which leads
        # to a semi-unique reconstruction. the simulation is seeded with a random start, then the simulation
        # is iterated to a solution, then the solution is copied back to host memory from the GPU.
        print "trial ",trial
        reconstruction.seed() # seed the simulation with random numbers.
        reconstruction.iterate(params.iterations) # iterate
        save_buffer[trial] = reconstruction.get(reconstruction.psi_in)[r0:r0+rows,c0:c0+cols] # copy solution to host

    # roll the phase for global phase alignment. executes on cpu.
    print "rolling global phases"
    save_buffer = phasing.align_global_phase(save_buffer)
    
    # sum along the frame axis then save the output. save the components as fits for future work and a sqrt
    # image of the magnitude for inspection with a program like GIMP or photoshop
    save_buffer[-1] = numpy.sum(save_buffer,0)
    speckle.io.save('%s/%s summed.fits'%(params.savepath,params.savename), save_buffer[-1], components='cartesian', overwrite=True)
    speckle.io.save('%s/%s summed sqrt.png'%(params.savepath,params.savename), numpy.sqrt(abs(save_buffer[-1])), components='complex_hls')

if __name__ == '__main__':
    use_gpu, device_info = check_gpu()
    if use_gpu:
        import speckle.gpu as gpulib
        phasing_code = gpulib.gpu_phasing.GPUPR
    if not use_gpu:
        phasing_code = phasing.CPUPR
    reconstruct(device_info)
