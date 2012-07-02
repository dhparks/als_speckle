where = 'gpu' # options: 'gpu','cpu'

# first, specify the colloid/random-walk simulation parameters. note that these parameters
# are also used elsewhere; for example, the symmetry microscope, which runs after the random-walk,
# looks for output files whose names depend on brownian step etc.
N            = 1024 # simulation grid size
density      = 1e-2 # density of ball centers
frames       = 1 # number of simulation frames. for gpu g2 calculation, make a power of 2
brownianstep = 0.50 # stdev of brownian motion step size
ballradius   = 1    # object size; this doesnt change the speckles, just gives an intensity profile

# second, specify the parameters for the symmetry microscope
pinhole    = 64           # pinhole radius
unwrap_r   = 70           # inner radius to unwrap
unwrap_R   = unwrap_r+255 # outer radius to unwrap; uR-ur should be a power of 2?
step_size  = 64           # step size between illumination sites
view_size  = 128          # this limits the field of view. the total number of sites is (view_size/step_size)**2
components = [2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32] # cosine components for the decomposition

# candidate symmetries are detected in this simulation by finding a cosine component which
# dominates the total cosine power at a given |q|. candidate_threshold defines how much of the total
# power a cosine component must possess to be a candidate. interrupts is an advanced option which
# specifies which simulation output should be saved for each candidate; recommend against changing.
candidate_threshold = 0.6
candidate_returnables = ('illuminated','speckle_blocker','correlated','spectrum')

# finally, choose which parts of the simulation to run.
make_samples    = False # this makes images of balls according to the first set of parameters
run_microscope  = False # this runs the symmetry microscope on each of the images according to the microscope parameters
find_candidates = True # find candidates according to the candidate_threshold

# designate a basic output path for results. sub directories will be created. you MUST have write permissions
# in this specified path
output_path = 'colloid output'

