# core libraries
import numpy
from numpy.fft import fftshift
from numpy.fft import fft2 as DFT

# common libraries
import speckle
import speckle.gpu as gpulib
dg = gpulib.gpu_domain_generator

# set the io default to overwrite files
speckle.io.set_overwrite(True)

# plotting
import matplotlib
matplotlib.use('Agg')
import pylab

def fit_speckle(data):
    from scipy.optimize import leastsq
    
    unwrapped = speckle.wrapping.unwrap(data,(0,N/2,(N/2,N/2)))
    yd = numpy.sum(unwrapped,axis=1)
    yd *= 1./yd.max()
    xd = numpy.arange(N/2).astype('float')
    tofit = numpy.array([xd,yd])
    fitted = speckle.fit.lorentzian_sq(tofit)
    
    eval_l2 = lambda p,x: p[0]/((x-p[1])**2./p[2]**2.+1)**2.+p[3]
    
    pylab.plot(xd,yd)
    pylab.plot(xd,eval_l2(fitted.final_params,xd))
    
    c0 = center*N/2048.
    w0 = fwhm*N/2048.
    
    pylab.xlim([c0-2*w0,c0+2*w0])
    pylab.savefig('out/envelope fit %s %s.png'%(center,fwhm))
    
    fitted.final_params[1] *= 2048./N
    fitted.final_params[2] *= 2048./N*1.29 # convert from w in the fit to fwhm in the report
    
    return fitted.final_params

def simulate_domains(gpuinfo,domain_N,center,fwhm,seed):
    
    converged = False
    domain_iteration = 0
    alpha = 0.3
    
    print ""
    print "simulating domains with L^2 envelope:"
    print "  size   = %s"%domain_N
    print "  center = %.2f -> %.2f"%(center,center*domain_N/2048.)
    print "  fwhm   = %.2f -> %.2f\n"%(fwhm,fwhm*domain_N/2048.)
    
    # Instantiate the simulation and make the goal envelope.
    print "instantiating gpu domain generator class"
    domain_simulation = dg.generator(gpuinfo,domain_N,seed,alpha,interrupts=())
    print "  done"

    # Make the goal envelope. In this case, the envelope is an isotropic squared lorentzian with no
    # symmetries. The 'center' and 'width' keywords which come into the simulation are what would
    # be measured on the 2048 ccd. The goal magnetization is also set in the make_envelope function;
    # here, it is set to zero by the ['goal_m',0] list element
    print "making envelope"
    width = fwhm/(2*numpy.sqrt(numpy.sqrt(2)-1))
    domain_simulation.make_envelope([['isotropic','lorentzian_sq',1,center*domain_N/2048.,width*domain_N/2048.],['goal_m',0]])
    print "  done"

    # Now run the simulation until the domain pattern has converged with a reasonable degree
    # of self-consistency. The simulation is incremented along iterations by the
    # gpudg.generator.one_iteration() method, which takes as argument the iteration number so that
    # the methods in the class can know what time it is. If reasonable self-consistency is not
    # reached within 200 iterations the simulation is stopped.
    print "iterating"
    while not converged:
        domain_simulation.one_iteration(domain_iteration)
        converged = domain_simulation.check_convergence()
        if domain_iteration == 200: converged = True
        domain_iteration += 1
    print "  convergence reached after %s iterations\n"%domain_iteration
        
    # pull the domain image off the gpu and back into host memory
    domains = domain_simulation.domains.get()

    return domains

# initialize the gpu
info = gpulib.gpu.init()

# open the seed file for domain generation
seed = speckle.io.openfits('resources/real phi0 1024 random.fits').astype('float32')

# specify the basic envelope parameters. these describe the lineshape as it would be
# measured on a 2048x2048 ccd on beamline 12.0.2 at 780eV.
center = 240
fwhm = 20.
N = 512

domains = simulate_domains(info,N,center,fwhm,seed)
speckles = dg.make_speckle(domains)

# fit the envelope paramters. these should be reasonably close to the input parameters.
# the fit width of the speckle will probably be a little less than the input goal
# due to the envelope rescaling algorithm convolving with a gaussian in r.
fit_params = fit_speckle(speckles)
print "fit speckle to L^2:"
print "  center: %.2f"%fit_params[1]
print "  width:  %.2f (will be slightly less than goal)"%fit_params[2]

speckle.io.save('out/melted domains %s %s.png'%(center,fwhm),domains,components=('real'))
speckle.io.save('out/melted speckle %s %s.png'%(center,fwhm),numpy.sqrt(speckles),color_map='B')

