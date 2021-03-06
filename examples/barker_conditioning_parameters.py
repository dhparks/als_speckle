# define paths and file locations. it is assumed that the dark
# and data files are in data_path. output will be saved to data_path
data_path = '../../jobs/data/barker'
dark_name = 'barker-500ev-long-background.fits'
data_name = 'barker-500ev-long.fits'
dust_mask_name = 'barker_dust3.png'

# some size parameters. look in code for details
correlation_box = 256
frame_align_box = 256
  
use_old_files = True # if True, will look for intermediates created on previous conditioning runs. for debugging. 

