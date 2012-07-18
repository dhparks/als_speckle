# demonstrate use of the propagation library for focusing
# back-propagated wavefields

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

aperture = speckle.shape.circle((N,N),p_radius/pitch)

# propagate the object forward the distance z, then multiply by a pinhole.
# this is ideally the solution from a phase retrieval problem (ie, step
# one of the barker code analysis)
print "forward propagating"
propagated = speckle.propagate.propagate_one_distance(object,energy,z=z_sim,pixel_pitch=pitch)
propagated *= aperture

# now let's pretend we don't know the solution to the back propagation problem
# and propagate the "recovered" wavefield at the aperture plane through a
# series of trial distances.
print "backward propagating + acutance (not apodized)"
step = 4
distances = numpy.arange(-200,200,step)*1e-6
back1 = speckle.propagate.propagate_distance(propagated,distances,energy,pitch,subarraysize=4*p_radius/pitch)
acutance1 = speckle.propagate.acutance(back1)

# back propagating a hard edge can result in ringing, so apodize the aperture
# function, then recalculate the back propagation and acutance. please
# note that I am apodizing the pinhole/support, NOT the data. the default
# values for apodize should be ok, but you might try setting kt=x
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





