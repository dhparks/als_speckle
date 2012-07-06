# this example supports execution on either the cpu or the gpu.
# if the gpu code cannot be loaded because the correct libraries are not installed
# or the device is not supported, the script will automatically fall back to
# the cpu. options for device: 'gpu','cpu'
device = 'gpu'

# this option sets what is considered 'converged'. it is defined basically
# as abs(sum(new_domains-old_domains))/size**2
converged_at = 0.002

# this example will simulate isotropic domains which evolve into symmetric
# domains. in this 2-stage process, the envelope will be changed to impart symmetries
# after the isotropic speckle has converged.
# center is the center of the scattering ring as seen on the 2048-ccd
# fwhm is the width of the lorentzian**2 on the 2048-ccd
# symmetries is the symmetry order for ordered stages and must be even
center = 250
fwhm = 100
symmetry = 4

# 
# the seed file is a set of random numbers which will be melted into domains.
# this seed file is 1024x1024 but the size of the simulation is set by the
# parameter "size". see open_seed() for coding details.
size = 512
seed_name = 'domain generator/resources/real phi0 1024 random.fits'

# these set which information to return from the simulation. 'converged'
# and 'envelope' are generally the only ones which matter; there are many more
# options available but they are mainly for debugging
domain_returnables = ('converged','envelope')

# this example first evolves the random seed into an isotropic domain pattern, and
# then into a series of ordered patterns. each of these stages requires a new
# goal envelope. the descriptions of these envelopes are aggregated below into
# the "trajectory" list. unfortunately the syntax for the envelopes can be complicated.
import numpy
c = 1./(2*numpy.sqrt(numpy.sqrt(2)-1)) # converts fwhm to correct lorentzian_sq width value
r = size/2048. # converts center and fwhm to correct values for the simulation size
trajectory = [[['isotropic','lorentzian_sq',1,center*r,fwhm*r*c],['goal_m',0]],
              [['isotropic','lorentzian_sq',1,center*r,fwhm*r*c],['modulation',1,symmetry,0,'tophat',1,center*r,fwhm*r*c],['modulation',1,symmetry,0,'tophat',1,2*center*r,fwhm*r*c]],
              [['isotropic','lorentzian_sq',1,center*r,fwhm*r*c],['modulation',1,symmetry,0,'uniform']],
              [['supplied','domain generator/resources/supplied_envelope.fits']]]

# finally, choose the output directory
save_to = 'domain generator/out'


