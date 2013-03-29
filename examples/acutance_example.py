
#### EXPLANATION OF EXAMPLE ####
# this example demonstrates use of the speckle.propagate library, including
# the use of apodize and acutance. The simulation proceeds along the following
# steps:
#
# 0. Parameters are set
# 1. A fictitious sample is created and forward-propagated by distance z_sim
# 2. At the aperture plane, the propagated field is multiplied by the aperture
# 3. The apodizered field is back propagated to the sample plane.
# 4. For comparison, the apodizered field is apodized and again back propagated.
# 5. The acutance is calculated for both propagation cases.
# 6. Output is saved

import numpy, matplotlib
import speckle
DFT, IDFT, shift = numpy.fft.fft2, numpy.fft.ifft2, numpy.fft.fftshift
speckle.io.set_overwrite(True)
matplotlib.use('Agg') # for headless
import matplotlib.pyplot as plt

#### parameters section ####
N        = 1024  # pixels
pitch    = 50e-9 # meters
energy   = 500   # eV
p_radius = 2e-6  # meters
o_radius = 3     # pixels
z_sim    = 80e-6 # meters
use_gpu  = True # if True, will try to use gpu to propagate

#### simulation section ####
# this code builds the random test sample and the post-scattering aperture
print "building"
density = 1e-2
n_obj   = N**2*density
coords  = tuple((numpy.random.rand(2,n_obj)*N).astype(int))
sample  = numpy.zeros((N,N),float)
ball    = shift(speckle.shape.circle((N,N),o_radius))

convolve = lambda a,b: IDFT(DFT(a)*DFT(b))
sample[coords] = 1
sample = abs(convolve(sample,ball)).real
sample = 1-numpy.clip(sample,0,1)

# make the aperture. you might try changing the shape to an ellipse or square
# to see what effect that has (downstream changes might be necessary after
# a change in aperture shape)
aperture = speckle.shape.circle((N,N),p_radius/pitch)

# Propagate the sample forward the distance z, then multiply by a pinhole.
# This is ideally the solution from a phase retrieval problem.
print "forward propagating"
propagated = speckle.propagate.propagate_one_distance(sample,energy,z=z_sim,pixel_pitch=pitch)
propagated *= aperture

# see if a gpu is available
if use_gpu:
    try: gpu_info = speckle.gpu.init()
    except: use_gpu = False
        
# Now let's pretend we don't know the solution to the back propagation problem
# and propagate the "recovered" wavefield at the aperture plane through a
# series of trial distances. Propagate in both directions, as the sign of
# "backwards" might not be known.
print "backward propagating + acutance (not apodized)"
step = 4
distances = numpy.arange(-200,200,step)*1e-6
if use_gpu: back1 = speckle.propagate.propagate_distances(propagated,distances,energy,pitch,subregion=4*p_radius/pitch,silent=False,gpu_info=gpu_info)
else:       back1 = speckle.propagate.propagate_distances(propagated,distances,energy,pitch,subregion=4*p_radius/pitch,silent=False)
acutance1 = speckle.propagate.acutance(back1)

# back propagating a hard edge can result in ringing, so apodize the aperture
# function, then recalculate the back propagation and acutance. please
# note that I am apodizing the pinhole/support, NOT the data. In analyzing the
# result from a CDI experiment, the recovered support should be used. The
# default values for apodize should be ok, but you might try setting kt=x
# (where x is between 0 and 1) in apodize (default is kt=0.1)
print "backward propagating + acutance (apodized)"
apodizer  = speckle.propagate.apodize(aperture)
apodized  = apodizer*propagated
if use_gpu: back2 = speckle.propagate.propagate_distances(apodized,distances,energy,pitch,subregion=4*p_radius/pitch,silent=False,gpu_info=gpu_info)
else:       back2 = speckle.propagate.propagate_distances(apodized,distances,energy,pitch,subregion=4*p_radius/pitch,silent=False)
acutance2 = speckle.propagate.acutance(back2)

#### output section ####
# now lets plot the acutance and save the output from the (known) solution,
# found at z = z_sim. also save some intermediates from the simuluation.
print "saving output"
plt.plot(acutance1,'b-',label='not apodized')
plt.plot(acutance2,'r-',label='apodized')
plt.legend()
plt.savefig('back propagation acutances.png')

solution_frame = len(distances)/2-z_sim/1e-6/step
solution1 = back1[solution_frame]
solution2 = back2[solution_frame]

speckle.io.save('built sample.png',sample)
speckle.io.save('propagated to pinhole.png',propagated,components='complex_hls')
speckle.io.save('apodized at pinhole.png',apodized,components='complex_hls')
speckle.io.save('apodizer.fits',apodizer)
speckle.io.save('back propagated.png',solution1,components='complex_hls')
speckle.io.save('back propgated apodized.png',solution2,components='complex_hls')






