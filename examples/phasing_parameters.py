""" Parameters and filenames to execute the basic_phasing_example and
advanced_phasing_example scripts."""

iterations = 500 # how many iterations per trials
trials     = 10   # how independent trials per round
rounds     = 5    # how many rounds

dataname    = 'path to conditioned modulus; max pixel should be in corner'
supportname = 'path to support file, can be .png, .fits, or .reg'

# define the output path
savepath = 'path to output folder'
savename = 'namebase for output files'

# these pertain to support refinement
blur_sigma       = 2.0
global_threshold = 0.0
local_threshold  = 0.07
