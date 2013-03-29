import sys
import numpy as np

path = '../exampledata' # I'm just using this to shorten paths below

# specify the objects. these are the files that will be rastered across to look
# for symmetries.
samples  = ['%s/domains 296 114 1_real.png'%path,
            '%s/domains 296 114 2_real.png'%path]

# Specify the illumination function. in this example, I supply a magnitude
# file for the illumination; see start_illumination() to change this scheme.
# The shape of the files must be the same. The size of the illumination sets
# the size of the speckle pattern (ie 2048x2048) so you need to pay attention to
# what illumination width gives the correct speckle size.
mag_file   = '%s/ex_pinhole110_mag.fits'%path
phase_file = None

# Specify the coherence factor. This is the inverse fourier transform of the
# point-spread function. For simplificity, we will just use a circular gaussian.
# The coherence function should be constructed so that the 0-frequency component
# is in the (0,0) pixel ("machine-centered") instead of the (N/2,N/2) pixel.
coherence_file = '%s/ex_coherence_mag.fits'%path

# Specify the parameters for unwrapping the speckle pattern. unwrap_r and
# unwrap_R are the inner and outer wrapping radii, respectively.
unwrap_r  = 80
unwrap_R  = 250

# Specify the spacing of the illumination sites.
# In this example, the list of coordinates is recalculated every time a new
# object file is loaded for scanning.
step_size = 16

# this is mainly just for my debugging.
troubleshoot_returnables = ('illuminated','blurred','unwrapped','resized','correlated','spectrum','spectrum_ds','speckle')

# choose which parts of the simulation to run.
run_microscope  = True # this runs the symmetry microscope on each of the images according to the microscope parameters
troubleshoot    = False # debugging: runs on one site of the first object to make sure the code executes properly
get_list        = [0,500] # list of which site numbers to get particulars from

# Designate a basic output path for results.
# !!you MUST have write permissions to this path !!
# If comine_output is true, the spectra are all combined into one big 4d file containing
# spectra from all samples in the list above. If combine_spectra = False, the
# spectra from the samples will be saved individually in 3d files.
output_path = path
combine_output = True  # if True, the spectra are all combined into one big file containing all object results
output_labels = [1,2] # for labeling output in conjugation with identifier (below)

# this gets prepended to the spectra names. basically a "project name"
identifier = 'symmetry microscope demo'



