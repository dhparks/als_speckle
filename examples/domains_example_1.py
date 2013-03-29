# core libraries
import sys
sys.path.insert(0,'..')
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
# probably have to look in the class __init__()
generator.converged_at = dp.converged_at
generator.cutoff       = dp.cutoff

for n, envelope in enumerate(dp.envelopes):
    
    # open the envelope and load it into the class
    envelope = speckle.io.open(envelope)
    generator.load(envelope=envelope)
    
    # we might want to reset the domain image
    if n == 0 or (n > 0 and dp.reset_each):
        generator.seed()
        iteration = 0

    # now we iterate to convergence
    time0 = time.time()
    while not generator.converged:
        
        generator.iterate(iteration)
        generator.check_convergence(iteration)
        iteration += 1
        
    print time.time()-time0

    speckle.io.save('converged domains %s.fits'%n,generator.returnables['converged'],components='real')
    


