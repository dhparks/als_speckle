# illustrate use of functions in shape and propagate. propagate uses the cpu
# code path, which is slow.
import numpy, speckle

# first, establish simulation parameters. these are physically reasonable
# but not necessarily achievable at bl1202. the object will be a circular
# pinhole.

N      = 1024  # N**2 pixels in simulations
energy = 5e2   # eV
pitch  = 50e-9 # meters
radius = 4e-6  # meters

# generate the object with shape
pixel_radius = radius/pitch
test_object  = speckle.shape.circle((N,N),pixel_radius,AA=True)

# choose the distances to propagate and run the propagator. to save space and
# memory, i'm not saving the whole array but rather a subsection around the
# origin; this is set through subarraysize=...
distances = numpy.arange(-200,200,4)*1e-6 # meters
propagated = speckle.propagate.propagate_distance(test_object,distances,energy,pitch,subregion=4*pixel_radius,silent=False)


# save the output
speckle.io.save('propagated circle.fits',propagated.astype('complex64'),components='polar',overwrite=True)
speckle.io.save('propagated circle hls.jpg',propagated.astype('complex64')[0],components='complex_hls',overwrite=True)
