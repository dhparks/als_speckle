# import the common libraries
import numpy
import speckle
import speckle.gpu as gpulib
phasing = speckle.phasing

# import the parameters for the reconstruction
import phasing_parameters as params

def do_iterations(gpupr):
    # conduct phase retrieval iterations by calling GPUPR methods.
    for iteration in range(params.iterations):
        if params.shrinkwrap:
            if iteration%params.update_period == 0 and iteration > 0:
                gpupr.update_support()
                gpupr.iteration('er')
        if not params.shrinkwrap:
            if (iteration+1)%100 != 0: gpupr.iteration('hio')
            else: gpupr.iteration('er')
        if iteration%100 == 0: print "  iteration ",iteration

def reconstruct_gpu(gpuinfo):
    # canonical code to demonstrate library functions

    # open the data specified in dataname and supportname
    data = speckle.io.openfits(params.dataname)
    support = speckle.io.openimage(params.supportname).astype('float')
    support *= 1./support.max()
    
    # find the region which bounds the support. this will prevent having to store and save a bunch
    # of redundant zeros. as used here, this will generate square bounding with a ring of zeros around
    # the outside for margin of error.
    bounds = speckle.masking.bounding_box(support,force_to_square=True,pad=4)
    r0,c0 = bounds[0],bounds[2]
    rows = int(bounds[1]-bounds[0])
    cols = int(bounds[3]-bounds[2])
        
    # initialize the GPUPR class provided by the phasing library. initialize the savebuffer in host memory
    N = len(data)
    reconstruction = gpulib.gpu_phasing.GPUPR(gpuinfo,N,shrinkwrap=params.shrinkwrap)
    save_buffer = numpy.zeros((params.trials+1,rows,cols),numpy.complex64)
    
    # load the data onto the gpu using the GPUPR methods
    if params.shrinkwrap: reconstruction.load_data(data,support,update_sigma=phasing.shrinkwrap_sigma)
    if not params.shrinkwrap: reconstruction.load_data(data,support)
    
    for trial in range(params.trials):
        # each reconstruction follows the same procedures but with a different random input which leads
        # to a semi-unique reconstruction. the simulation is seeded with a random start, then the simulation
        # is iterated to a solution, then the solution is copied back to host memory from the GPU.
        print "trial ",trial
        reconstruction.seed() # seed the simulation with random numbers.
        do_iterations(reconstruction) # iterate
        save_buffer[trial] = reconstruction.psi_in.get()[r0:r0+rows,c0:c0+cols] # copy solution to host

    # roll the phase for global phase alignment. executes on cpu.
    save_buffer = phasing.align_global_phase(save_buffer)
    
    # sum along the frame axis then save the output. save the components as fits for future work and a sqrt
    # image of the magnitude for inspection with a program like GIMP or photoshop
    save_buffer[-1] = numpy.sum(save_buffer,0)
    speckle.io.save('%s/%s summed.fits'%(params.savepath,params.savename), save_buffer[-1], components='cartesian', overwrite=True)
    speckle.io.save('%s/%s summed sqrt.png'%(params.savepath,params.savename), numpy.sqrt(abs(save_buffer[-1])), color_map="B")

# initialize the gpu (get the context etc) then run the reconstruction
info = gpulib.gpu.init()
reconstruct_gpu(info)
