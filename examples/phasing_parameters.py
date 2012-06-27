""" Parameters and filenames to execute the phasing_example script.
To adopt phasing_example into general purpose canonical phasing code,
most changes should be in phasing_parameters."""

# select the bcmo data here based on temperature. the support identifier
# is chosen from the dictionary. these are the most refined supports used
# to reconstruct the experimental data. for experimenting with the example
# code, try switching the identifiers so that the wrong support is selected;
# this should cause a deterioration of the reconstruction.
Tb = 218
sn_tb = {163:'38d',191:'35e',218:'29e',246:'22d',274:'18d',309:'10e'}
sn = sn_tb[Tb]

dataname = 'data/bcmo holo conditioned tb %s 512_mag.fits'%Tb
supportname = 'supports/bcmo_june_%s.png'%sn

# the primary reconstruction parameters: trials and iterations (per trial)
trials = 10
iterations = 500

# define the output path
savepath = 'out'
savename = 'bcmo tb%s june%s 512 i%s n%s'%(Tb,sn,iterations,trials)

# set these for shrinkwrap
shrinkwrap = False
shrinkwrap_sigma = 3
update_period = 50
