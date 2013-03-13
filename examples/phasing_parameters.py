""" Parameters and filenames to execute the phasing_example script.
To adopt phasing_example into general purpose canonical phasing code,
most changes should be in phasing_parameters."""

device = 'gpu'

dataname    = '../../jobs/out/stripestest/stripes_speckle_mag.fits'
supportname = '../../jobs/out/stripestest/stripes_support_mag.fits'

# the primary reconstruction parameters: trials and iterations (per trial)
trials     = 1
iterations = 500

# define the output path
savepath = '../../jobs/out/stripestest/'
savename = 'testing'

# set these for shrinkwrap
shrinkwrap = False
shrinkwrap_sigma = 3
update_period = 50