
#### EXPLANATION OF EXAMPLE ####
# this example demonstrates use of the speckle.propagate library, including
# the use of apodize and acutance. The simulation proceeds along the following
# steps:
#
# 0. Parameters are set
# 1. A fictitious object is created and forward-propagated by distance z_sim
# 2. At the aperture plane, the propagated field is multiplied by the aperture
# 3. The filtered field is back propagated to the sample plane.
# 4. For comparison, the filtered field is apodized and again back propagated.
# 5. The acutance is calculated for both propagation cases.
# 6. Output is saved

import numpy, speckle, matplotlib
DFT, IDFT, shift = numpy.fft.fft2, numpy.fft.ifft2, numpy.fft.fftshift
speckle.io.set_overwrite(True)

#### parameters section ####
N        = 1024  # pixels
pitch    = 50e-9 # meters
energy   = 500   # eV
p_radius = 2e-6  # meters
o_radius = 3     # pixels
z_sim    = 80e-6 # meters

#### simulation section ####
# this code builds the random test object and the post-scattering aperture
print "building"
density = 1e-2
n_obj   = N**2*density
coords  = tuple((numpy.random.rand(2,n_obj)*N).astype(int))
object  = numpy.zeros((N,N),float)
ball    = shift(speckle.shape.circle((N,N),o_radius))

convolve = lambda a,b: IDFT(DFT(a)*DFT(b))
object[coords] = 1
object = abs(convolve(object,ball)).real
object = 1-numpy.clip(object,0,1)

# make the aperture. you might try changing the shape to an ellipse or square
# to see what effect that has (downstream changes might be necessary after
# a change in aperture shape)
aperture = speckle.shape.circle((N,N),p_radius/pitch)

# Propagate the object forward the distance z, then multiply by a pinhole.
# This is ideally the solution from a phase retrieval problem.
print "forward propagating"
propagated = speckle.propagate.propagate_one_distance(object,energy,z=z_sim,pixel_pitch=pitch)
propagated *= aperture

# Now let's pretend we don't know the solution to the back propagation problem
# and propagate the "recovered" wavefield at the aperture plane through a
# series of trial distances. Propagate in both directions, as the sign of
# "backwards" might not be known.
print "backward propagating + acutance (not apodized)"
step = 4
distances = numpy.arange(-200,200,step)*1e-6
back1 = speckle.propagate.propagate_distance(propagated,distances,energy,pitch,subarraysize=4*p_radius/pitch)
acutance1 = speckle.propagate.acutance(back1)

# back propagating a hard edge can result in ringing, so apodize the aperture
# function, then recalculate the back propagation and acutance. please
# note that I am apodizing the pinhole/support, NOT the data. In analyzing the
# result from a CDI experiment, the recovered support should be used. The
# default values for apodize should be ok, but you might try setting kt=x
# (where x is between 0 and 1) in apodize (default is kt=0.1)
print "backward propagating + acutance (apodized)"
filter = speckle.propagate.apodize(aperture)
filtered = filter*propagated
back2 = speckle.propagate.propagate_distance(filtered,distances,energy,pitch,subarraysize=4*p_radius/pitch)
acutance2 = speckle.propagate.acutance(back2)

#### output section ####
# now lets plot the acutance and save the output from the (known) solution,
# found at z = z_sim. also save some intermediates from the simuluation.
print "saving output"
matplotlib.pyplot.plot(acutance1,'b-',label='not apodized')
matplotlib.pyplot.plot(acutance2,'r-',label='apodized')
matplotlib.pyplot.legend()
matplotlib.pyplot.savefig('back propagation acutances.png')

solution_frame = len(distances)/2-z_sim/1e-6/step
solution1 = back1[solution_frame]
solution2 = back2[solution_frame]

speckle.io.save('built object.fits',object)
speckle.io.save('propagated to pinhole.fits',propagated)
speckle.io.save('apodized at pinhole.fits',filtered)
speckle.io.save('apodizer.fits',filter)
speckle.io.save('back propagated.fits',solution1)
speckle.io.save('back propgated filtered.fits',solution2)






