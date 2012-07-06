# core libraries
import numpy,sys
fftshift = numpy.fft.fftshift
fft2 = numpy.fft.fft2
import time

# common libraries
import speckle
import speckle.simulation

# set the io default to overwrite files
speckle.io.set_overwrite(True)

def check_gpu():
    global gpuinfo
    # instantiate the generator. initialize the gpu if necessary. there are several ways
    # this can fail so this is a little bit messy. if you KNOW FOR SURE that the system
    # has an opencl runtime and pyopencl is installed and there is a supported GPU in the
    # system then this can be simplified (see further down)
    if dp.device == 'gpu':
        try:
            import speckle.gpu as gpulib
        except gpulib.gpu.GPUInitError as errormsg:
            # import crashed without pyopencl
            print errormsg,"\nfalling back to cpu"
            dp.device == 'cpu'
            
    if dp.device == 'gpu':
        try:
            gpuinfo = gpulib.gpu.init()
        except gpulib.gpu.GPUInitError as errormsg:
            # gpulib.gpu_init() crashed due to an opencl problem (such as lack of supported gpu)
            print errormsg, "\nfalling back to cpu"
            dp.device = 'cpu'
            
    if dp.device == 'gpu': return True, gpuinfo
    if dp.device == 'cpu': return False, None

def iterate(dg_instance):
    # Now run the simulation until the domain pattern has converged with a
    # reasonable degree of self-consistency. The simulation is incremented along
    # iterations by the one_iteration() method, which takes as argument the
    # iteration number. If reasonable self-consistency is not reached within 200
    # iterations the simulation is stopped.
    print "iterating"
    iteration = 0
    while not dg_instance.converged:
        dg_instance.one_iteration(iteration)
        dg_instance.check_convergence()
        if iteration == 200: dg_instance.converged = True
        iteration += 1
        
        #domains = dg_instance.returnables['domains']
        #speckle.io.save('%s/domains debug %s.png'%(dp.save_to,iteration),domains,components='real')
        
        print "  iteration %s, power %.3e"%(iteration,dg_instance.power)
    print "  convergence reached after %s iterations\n"%iteration
        
    # pull the domain image and the envelope off the gpu and back into host memory.
    envelope = dg_instance.returnables['envelope']
    domains = dg_instance.returnables['converged']

    return domains,envelope

import domains_parameters as dp

# generate a random seed file for domain generation. range of values is +- 1
seed = 2*numpy.random.rand(dp.size,dp.size)-1
speckle.io.save('%s/domains seed.png'%(dp.save_to),seed,components='real')
r = dp.size/2048.

# check the opencl runtime and gpu support. possible outcomes:
# want gpu; gpu available: get back gpuinfo
# want gpu; not available: fall back to cpu
# want cpu               : use cpu
use_gpu, device_info = check_gpu()

# instantiate the simulation on whichever device. at this point the simulation has
# been abstracted away from the hardware from the user POV
if use_gpu: import speckle.simulation.gpu_domains as domainslib
if not use_gpu: import speckle.simulation.cpu_domains as domainslib
domain_simulation = domainslib.generator(device=device_info,domains=dp.size,returnables=dp.domain_returnables,converged_at=dp.converged_at)

# simplified gpu instantiation, for use when the GPU and runtimes are known to be present:
# import speckle.gpu
# gpuinfo = speckle.gpu.gpu.init()
# import speckle.simulation.gpu_domains as domainslib
# domain_simulation = domainslib.generator(gpuinfo,domains=seed,returnables=dp.domain_returnables,converged_at=dp.converged_at)
    
# put the seed into the simulation
time0 = time.time()
domain_simulation.set_domains(seed)

# run each stage of the trajectory, saving output from each
for n,stage in enumerate(dp.trajectory):
    domain_simulation.set_envelope(stage)          # set the goal despeckle envelope 
    domains,envelope = iterate(domain_simulation)  # iterate to a self-consistent solution
    
    # save output
    print "elapsed time: %s"%(time.time()-time0)
    speckles = domainslib.make_speckle(domains)
    speckle.io.save('%s/%s melted domains %s %s stage %s.png'%(dp.save_to,dp.device,dp.center,dp.fwhm,n+1),domains,components=('real'))
    speckle.io.save('%s/%s goal envelope %s %s stage %s.fits'%(dp.save_to,dp.device,dp.center,dp.fwhm,n+1),envelope,components=('real'))
    speckle.io.save('%s/%s melted speckle %s %s stage %s.png'%(dp.save_to,dp.device,dp.center,dp.fwhm,n+1),numpy.sqrt(speckles),color_map='B')
    


