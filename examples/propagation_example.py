# illustrate use of functions in shape and propagate. propagate uses the cpu
# code path, which is slow.
import numpy, speckle

# first, establish simulation parameters. these are physically reasonable
# but not necessarily achievable at bl1202. the sample will be a circular
# pinhole.

N      = 256  # N**2 pixels in simulations
energy = 5e2   # eV
pitch  = 50e-9 # meters
radius = 2e-6  # meters

# generate the sample with shape
pixel_radius = radius/pitch
sample = speckle.shape.circle((N,N),pixel_radius,AA=True)

# choose the distances to propagate and run the propagator. to save space and
# memory, i'm not saving the whole array but rather a subsection around the
# origin; this is set through subarraysize=...
distances = numpy.arange(-400,400,4)*1e-6 # meters
propagated = speckle.propagate.propagate_distance(sample,distances,energy,pitch,subarraysize=4*pixel_radius,silent=False)

