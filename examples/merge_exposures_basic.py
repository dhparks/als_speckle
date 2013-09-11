import numpy, speckle
speckle.io.set_overwrite(True)

# This script illustrates how to use the merging routines found in the
# conditioning module of the speckle library. First we make a simple simulated
# set of exposures (including counting statistics), then merge them.

### data simulation section ###

# make an airy pattern
N    = 512
R    = 15
obj  = speckle.shape.circle((N,N),R)
airy = numpy.fft.fftshift(abs(numpy.fft.fft2(obj))**2)

# Divide the airy pattern into blocker-in and blocker-out images.
# In the blocker_out image, assume we count until the brightest pixel has 5k
# photons. In the blocker_in image, count until the brightest pixel has 1k
# photons. Include counting statistics and a dark-current background.

blocker = 1-speckle.shape.circle((N,N),2*R)

blocker_out  = airy
blocker_out *= 15e3/blocker_out.max()
blocker_out  = numpy.random.poisson(blocker_out)
blocker_out += 400+15*numpy.random.randn(N,N)

blocker_in  = airy*blocker
blocker_in *= 1e3/blocker_in.max()
blocker_in  = numpy.random.poisson(blocker_in)
blocker_in += 400+15*numpy.random.randn(N,N)

# Sometimes, the blocker_out and blocker_in data collected during an experiment
# are out of alignment because moving the blocker perturbs the CCD cantilever.
# So, simulate the data becoming out of alignment.
r1, r2 = int(numpy.random.randn()*5), int(numpy.random.randn()*5)
blocker_in  = numpy.roll(numpy.roll(blocker_in,r1,0),r2,1)

### data merging section ###

# For the merging routine, we need to specify two regions: a fill_region
# which generally corresponds to the blocker, and a match_region which
# is the region inside of which counts are matched. For experimental data,
# it is easiest to draw these regions in ds9 and save them as a .reg file.
# For this example I will just define the regions using the shape module.
# *** PLEASE READ THE DOCUMENTATION FOR speckle.conditioning.merge ***

fill_region  = speckle.shape.circle((N,N),1.2*R+5,AA=0)
match_region = speckle.shape.square((N,N),32,center=(N/2,N/2+2*R+16+2))
merged       = speckle.conditioning.merge(blocker_in,blocker_out,
                                          fill_region,fit_region=match_region)

# now I will align the data prior to merging, which should give
# a much better result. we will align by cross-correlation method in the
# match_region.
merged2 = speckle.conditioning.merge(blocker_in,blocker_out,fill_region,
                                     fit_region=match_region,align=True)

### output section
speckle.io.save('blocker_in.fits',  blocker_in)
speckle.io.save('blocker_out.fits', blocker_out)
speckle.io.save('merged.fits',      merged)
speckle.io.save('merged2.fits',     merged2)



