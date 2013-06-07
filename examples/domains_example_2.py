# core libraries
import numpy
import time

# common libraries
import speckle
from speckle.simulation import domain_generator as dg
speckle.io.set_overwrite(True)

import domains_parameters as dp

# instantiate the generator. default device is gpu, if available.
# cpu can be forced by supplying force_cpu=True at instantiation.
generator = dg.generator()
#generator = dg.generator(force_cpu=True)

# set some values. the generator has defaults of these but this
# is how they can be overridden. to find the other attributes you'll
# probably have to look in the class method __init__()
generator.converged_at = dp.converged_at
generator.cutoff       = dp.cutoff

# allocate an array to hold all the frames
out = numpy.zeros((dp.kicks,dp.N,dp.N),numpy.uint8)

# in this example, we will MAKE the envelope from scratch
r = speckle.shape.radial((dp.N,dp.N))
c = dp.N*300/2048.
w = dp.N*110/2048.
e = 1./(((r-c)/w)**2+1)**2

generator.load(envelope=e)
generator.seed()

for k in range(dp.kicks):
    
    # run the generator to convergence
    iteration = 0
    while not generator.converged:
        generator.iterate(iteration)
        generator.check_convergence(iteration)
        iteration += 1
    print k, iteration
        
    # if we're here, the current domain pattern has converged, so pull
    # it out of the class and save them in a low-bit form in save_buffer
    domains = numpy.copy(generator.get(generator.domains)) # have to make a copy to protect original on cpu path
    domains += -domains.min()
    domains *= 255./domains.max()
    out[k] = domains.astype(numpy.uint8)
    
    # now kick the converged domain pattern still in the simulation.
    generator.kick_rs(dp.kick_amount)
    
speckle.io.save('kicked2 domains k%s.fits'%dp.kick_amount,out,components='real')


