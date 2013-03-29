""" Parameters and filenames to execute the barker_phasing script."""

iterations = 200 # how many iterations per trials
trials     = 10  # how independent trials per round
rounds     = 3   # how many rounds

dataname    = 'path to conditioned modulus; max pixel should be in corner'
supportname = 'path to support file, can be .png, .fits, or .reg'

# define the output path
savepath = 'path to output folder'
savename = 'namebase for output files'

# these pertain to support refinement
blur_sigma       = 2.0
global_threshold = 0.0
local_threshold  = 0.07
