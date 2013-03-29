# demonstrate wrapping library functionality
import numpy
import sys
sys.path.insert(0,'..')
import speckle
DFT, IDFT, shift = numpy.fft.fft2, numpy.fft.ifft2, numpy.fft.fftshift
speckle.io.set_overwrite(True)

#### parameters section ####
N        = 1024  # simulation size; pixels
density  = 1e-2  # density of "particles"
p_radius = 100   # pinhole radius; pixels
unwrap_r = 70    # unwrapping inner radius; pixels
unwrap_R = 200   # unwrapping outer radious; pixels

#### simulation section ####

# Build random sample; make speckles from sample
sample = numpy.zeros((N,N),float)
n_obj  = N**2*density
coords = tuple((numpy.random.rand(2,n_obj)*N).astype(int))

sample[coords] = 1
sample *= speckle.shape.circle((N,N),p_radius)

make_speckles = lambda x: shift(abs(DFT(x))**2).real
speckles = make_speckles(sample)

# Unwrap speckles. Here, I will first generate an unwrapping plan.
# When unwrapping a single speckle pattern, the plan can also be built for you
# behind the scenes. However, if unwrapping many speckle patterns with the
# same parameters, pre-building a plan is MUCH faster.
uwplan = speckle.wrapping.unwrap_plan(unwrap_r,unwrap_R,(N/2,N/2))
unwrapped = speckle.wrapping.unwrap(speckles,uwplan)

# Now rewrap! Here, I will bypass building an explicit plan and instead just
# pass in the values of r and R; a plan will be created behind the scenes.
# Notice that the dimensions of wrapped are not the same as the original
# speckle pattern unless unwrap_R = N/2
wrapped = speckle.wrapping.wrap(unwrapped,(unwrap_r,unwrap_R))

# Because we already went to the trouble of making up a speckle pattern we should
# look for angular symmetries. having already unwrapped the pattern we could use
# some of the lower-level functions in speckle.symmetries. Here I will use the
# highest-level function, "rot_sym". In place of plan=uwplan I could also supply
# plan=(unwrap_r,unwrap_R).
# IMPORTANT: ROT_SYM RETURNS A DICTIONARY
spectrum = speckle.symmetries.rot_sym(speckles,uwplan=uwplan)['spectra_ds'] # decomposed up to nyquist limit. even components only.

# If I also wanted the angular autocorrelation in addition to its decomposition
# I have to specify things a little differently.
output = speckle.symmetries.rot_sym(speckles,uwplan=uwplan,get_back=('spectra','correlated'))
angular_correlation = output['correlated']
angular_decomposition = output['spectra'] # this is the same result as spectrum = ... above

#### output section ####
speckle.io.save('unwrapping sample.fits',sample)
speckle.io.save('unwrapping speckle.fits',speckles)
speckle.io.save('unwrapped speckle.fits',unwrapped)
speckle.io.save('re-wrapped speckle.fits',wrapped)
speckle.io.save('angular spectrum.fits',spectrum)
speckle.io.save('angular correlation.fits',angular_correlation)
