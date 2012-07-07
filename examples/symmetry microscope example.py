# core
import numpy as np
import time
DFT = np.fft.fft2
IDFT = np.fft.ifft2
fftshift = np.fft.fftshift

# shared
import speckle
import speckle.simulation
speckle.io.set_overwrite(True)

def check_parameters():
    
    """ Make sure no nonsense came in through symmetry_microscope_parameters.
    Some of these checks are repeated in later code. Additional and more
    rigorous checks also happen later"""
    
    assert sp.make_samples in (0,1,True,False), "make_samples must be boolean-evaluable"
    
    if sp.make_samples:
        assert sp.device in ('cpu','gpu'), "unrecognized device %s"%sp.device
        assert isinstance(sp.N,int), "array size must be integer"
        assert sp.N in (32,64,128,256,512,1024,2048,4096), "array size must be power of 2"
        assert isinstance(sp.density,float), "density must be float"
        assert sp.density < 1, "density must be < 1"
        assert isinstance(sp.frames, int) and sp.frames >= 1, "number of frames must be int >= 1"
        assert isinstance(sp.brownianstep,(float,int)), "brownianstep must be a number"
        assert isinstance(sp.ballradius,(float,int)), "ball radius must be a number"
        
    assert sp.run_microscope in (0,1,True,False), "run_microscope must be boolean-evaluable"
    assert sp.find_candidates in (0,1,True,False), "find_candidates must be boolean-evaluable"
    
    if sp.run_microscope or sp.find_candidates:
        assert isinstance(sp.pinhole,(int,float)), "in this simulation, pinhole must be a number"
        assert isinstance(sp.unwrap_r,int), "unwrap_r must be int"
        assert isinstance(sp.unwrap_R,int), "unwrap_R must be int"
        assert sp.unwrap_R > sp.unwrap_r, "unwrap_R must be larger than unwrap_r"
        assert isinstance(sp.step_size,int), "step_size must be int"
        assert isinstance(sp.view_size,int), "view_size must be int"
        assert isinstance(sp.components,(list,tuple,np.ndarray)), "components must be iterable type"
        assert all([isinstance(e,int) for e in sp.components]), "each component must be int"
        
    if sp.find_candidates:
        assert isinstance(sp.candidate_threshold,float), "candidate threshold must be float"
        assert sp.candidate_threshold > 0 and sp.candidate_threshold < 1, "threshold must be > 0, < 1"
        assert isinstance(sp.candidate_returnables,(list,tuple)), "candidate returnables must be an iterable type"
        assert all([isinstance(e,str) for e in sp.candidate_returnables]), "each returnable label must be a string"
        
    assert isinstance(sp.output_path,str), "output_path must be a string"

def check_gpu():
    """ See if a gpu is available and wanted. If not, fall back to cpu.
    
    Returns:
        use_gpu: True or False
        device_info: for initializing gpu. cpu is returned None as a dummy."""

    if sp.device == 'gpu':
        try:
            import speckle.gpu as gpulib # this can throw an error
        except gpulib.gpu.GPUInitError as error:
            print error,"\nfalling back to cpu"
            sp.device == 'cpu'
            
    if sp.device == 'gpu':
        try:
            print "trying to get gpuinfo"
            gpuinfo = gpulib.gpu.init() # this can throw various errors
            print "got it"
        except gpulib.gpu.GPUInitError as error:
            print error, "\nfalling back to cpu"
            sp.device = 'cpu'
            
    if sp.device == 'gpu': return True, gpuinfo
    if sp.device == 'cpu': return False, None

def make_paths():
    
    """ Makes paths for output"""
    
    import os
    
    assert os.path.isdir(sp.output_path), "output path does not exist!"
    ball_path = '%s/real space images/balls density%s radius%s step%s'%(sp.output_path,sp.density,sp.ballradius,sp.brownianstep)
    analysis_path = '%s/analysis/bd%s br%s bs%s pr%s'%(sp.output_path,sp.density,sp.ballradius,sp.brownianstep,sp.pinhole)
    
    try: os.makedirs(ball_path)
    except: pass
    
    try: os.makedirs(analysis_path)
    except: pass
    
    return ball_path, analysis_path

def make_samples():
    
    """ Make and save a set of images showing the brownian motion of a set of
    particles. Uses the random_walk class in speckle.simulation.
    
    parameters: N, density, frames, ballradius, brownianstep"""
    
    rw = speckle.simulation.random_walk
    
    print "making %s sample images"%nf
    objs = int(sp.N**2*sp.density)
    balls = rw(sp.N,objs,sp.ballradius)
    for frame in range(nf):
        print "  "+str(frame)
        balls.displace(sp.brownianstep)
        image = balls.place_on_grid()
        speckle.io.save('%s/%s.png'%(ball_path,frame),image)
        
def open_sample(name):
    
    """ Open an image and turn it into an array. The scattering model
    here is that for magnetic dichroism because it gives a minimal
    airy pattern"""
    
    data = speckle.io.openimage(name)
    data = data.astype('float')
    data = 2*data/data.max()-1
    return data

def modulo_indices(index,shape):
    """ Given an array index and array shape, returns the location of the index as a list of
    indices in order of axis speed. For example, given a 2d array will return (row_val,col_val).
    This is useful for something like turning argmax into coordinates.
    """
    nd = len(shape)
    indices = []
    for n in range(nd-1):
        length = np.product(shape[n+1:])
        indices.append(index/length)
        index = index%length
    indices.append(index%length)
    return indices

def open_spectra(name):
    
    # do some stuff with splits as regex to separate out the _mag
    # and then extract the fp value to rescale correctly
    # from int16 back to float
    
    data = speckle.io.openfits(name).astype('float32')
    maxval = name.split('maxval_')[1].split('_mag')[0]
    data *= float(maxval)/2**16.
    return data

def raster_spectra(sm_instance):
    
    """ Run the symmetry microscope analysis on each of the frames generated
    by make_samples() at the coordinates described by make_raster_coords().
    
    sm_instance is an instance of a symmetry microscope class, in this case
    generated by either speckle.gpu.gpu_correlations.gpu_microscope or by
    speckle.symmetries.cpu_microscope.
    
    Output is saved in analysis_path which was defined by make_paths()
    Format of the output spectra is (frame, site, |q|, m). Dtype is uint16 to
    save space (float32 and float64 get really huge!)
    """
    
    print "running symmetry microscope, %s sites/frame"%(nx*ny)
    
    # save results in spectra
    spectra = np.zeros((nf,nx*ny,nq,nc),float)

    for frame in range(nf):
        time0 = time.time()
        print "  frame %s"%frame
        
        # open a sample frame from disk and load into the class
        open_name = '%s/%s_mag.png'%(ball_path,frame)
        sample = open_sample(open_name)
        sm_instance.set_object(sample)

        # raster the probe over ycoords, xcoords. after running
        # the simulation at each site, copy the spectrum to spectra
        site = 0
        for row in ycoords:
            for col in xcoords:
                print "    %s %s"%(row,col)
                sm_instance.run_on_site(row,col)
                spectra[frame,site] = sm_instance.returnables['spectrum']
                site += 1
        
    # save; cast the results to integer to reduce space.
    maxval = spectra.max()
    spectra *= 65535/maxval
    spectra = spectra.astype('uint16')
    save_name = '%s spectra bd%s br%s bs%s pr%s maxval_%.4e.fits'%(sp.device,sp.density,sp.ballradius,sp.brownianstep,sp.pinhole,maxval)
    speckle.io.save('%s/%s'%(analysis_path,save_name),spectra)

def get_particulars(sm_instance,record):
    
    """ Get the relevant output from candidates identified by find_candidates.
    Run the symmetry microscope on the coordinates described in record; save
    the output described by sp.candidate_returnables
    
    arguments:
        sm_instance: an instance of a microscope class
        record: the ouput of find_candidates, listing which illumination sites
            in which random-walk frames gave speckle with a dominant symmetry.
            this could also be supplied by hand as a list of tuples if you wanted
            to look at some sites manually.
    """
    
    loaded = None
    for entry in record:
        print "getting particulars from frame/raster/row/col:"
        frame, record_site = int(entry[0]), int(entry[1])

        # if frame isnt the current sample on the gpu, load it
        if frame != loaded:
            open_name = '%s/%s_mag.png'%(ball_path,frame)
            data = open_sample(open_name)
            sm_instance.set_object(data)
            loaded = frame
            
        # record_site stores the raster number. convert the raster number to
        # y and x coordinates via modulo_indices
        ny, nx = modulo_indices(record_site,(len(ycoords),len(xcoords)))
        row,col = ycoords[ny], xcoords[nx]
        print "  %s %s %s %s"%(frame,record_site,row,col) 
        
        # do the speckle formation and decomposition at coords (row,col)
        sm_instance.run_on_site(row,col)
        
        # save the output showing the symmetry candidate.
        for key in sm_instance.returnables.keys():
            name = '%s/%s %s %s %s.png'%(analysis_path,sp.device,frame,record_site,key)
            speckle.io.save(name,sm_instance.returnables[key],components='real')
    
def find_candidates():
    
    """ Look through the spectra to find visually striking symmetry candidates.
    A candidate symmetry m at wavevector |q| is here defined as one whose
    cosine value a_{|q|,m} is greather then some percentage of the sum of the
    all the a_{|q|,m} at that |q|. The threshold percentage is controlled by
    sp.candidate_threshold and runs from 0 to 1.
    
    There is no condition placed on the width of the symmetry in |q|, so most of
    these will probably trigger on something of width 1 pixel.
    """
    
    # open the data from run_microscope
    import glob
    file = glob.glob('%s/%s spectra bd%s br%s bs%s pr%s maxval*.fits'%(analysis_path,sp.device,sp.density,sp.ballradius,sp.brownianstep,sp.pinhole))[0]
    data = open_spectra(file)
    assert data.ndim in (3,4), "data is wrong shape"
    if data.ndim == 3: data.shape = (1,data.shape[0],data.shape[1],data.shape[2])
   
    # divide each component by the sum of values at the same |q|
    power = np.sum(data,axis=3) # sum along the component-value axis
    ranks = np.zeros(data.shape,'uint8') # do this as integer for space and speed reasons
    passed = np.zeros(data.shape,bool)
    t = int(256*sp.candidate_threshold)
    print "looking for candidates in frame:"
    for i in range(data.shape[0]):
        print "  %s"%i
        for j in range(data.shape[1]):
            current_power = power[i,j]
            components = data[i,j].transpose()
            ranks[i,j] = (256*(components/current_power).transpose()).astype('uint8')
            passed[i,j] = np.where(ranks[i,j] > t, True, False)
   
    # filter to find which q,m components pass the threshold test.
    npassed = np.sum(passed)
    if npassed == 0:
        print "found no candidates"
        exit()
    print "  found %s candidates"%npassed
    
    # now, turn filtered into a list of (frame, site, row, col) indices using
    # some modulo arithmetic in modulo_indices()        
    passed = passed.ravel()
    get = np.argsort(passed)[-npassed:]
    record = np.zeros((npassed,5),int)
    uniques = []

    shape = ranks.shape
    for n,index in enumerate(get):
        # get indices via modulo arithmetic.
        frame, site, row, col = modulo_indices(index,shape)
        record[n] = frame, site, row, col, ranks[frame,site,row,col]
        uniques.append('%s_%s'%(frame,site))
        
    # record stores all the (frame, site, row, col) tuples which pass the test,
    # but for examining candidates we really just want to know which are the
    # unique (frame, site) pairs. these are saved in unique_record
    uniques = list(set(uniques))
    unique_record = np.zeros((len(uniques),2),int)
    for n,entry in enumerate(uniques):
        splits = entry.split('_')
        unique_record[n] = splits[0],splits[1]
    print "    %s are unique frame/site combinations"%len(unique_record)

    # re-order records by frame
    indices = np.argsort(record[:,0])
    record = record[indices]
    
    indices = np.argsort(unique_record[:,0])
    unique_record = unique_record[indices]
        
    record_save_name = '%s/candidate record bd%s br%s bs%s pr%s.fits'%(analysis_path,sp.density,sp.ballradius,sp.brownianstep,sp.pinhole)
    unique_save_name = '%s/candidate uniques bd%s br%s bs%s pr%s.fits'%(analysis_path,sp.density,sp.ballradius,sp.brownianstep,sp.pinhole)

    speckle.io.save(record_save_name,record)
    speckle.io.save(unique_save_name,unique_record)
    return unique_record

def make_raster_coords(N,xstep,ystep,size=None):

    if size == None:
        start, stop = 0,N
    else:
        assert size%2 == 0, "size to make_coords must be even"
        start, stop = 0,size

    x_coords = np.arange(start,stop,xstep)
    y_coords = np.arange(start,stop,ystep)
    return x_coords, y_coords

import symmetry_microscope_parameters as sp
check_parameters()

nf = sp.frames
nc = len(sp.components)
nq = sp.unwrap_R-sp.unwrap_r

### now lets get down to tacks. first, set up paths for output
ball_path, analysis_path = make_paths()

# next, make random walk images. these get saved in ball_path
if sp.make_samples:
    make_samples()

# instantiate a symmetry microscope. if user wants a gpu, first check runtimes
# and gpu support using check_gpu().
if sp.run_microscope or sp.find_candidates:

    # make the (x,y) coordinates where the images will be analyzed
    xcoords, ycoords = make_raster_coords(sp.N,sp.step_size,sp.step_size,size=sp.view_size)
    nx = len(xcoords)
    ny = len(ycoords)
    print "done with coords"
    
    # figure out which device to run on, gpu or cpu
    use_gpu, device_info = check_gpu()
    if use_gpu:
        import speckle.gpu
        microscope_code = speckle.gpu.gpu_correlations.gpu_microscope
    if not use_gpu:
        microscope_code = speckle.symmetries.cpu_microscope
        
    # instantiate the microscope with the array size, unwrap parameters, etc.
    # an actual array will be loaded into the class later using set_object().
    symmetry_microscope = microscope_code(
                device      = device_info, # if cpu, this is a dummy input
                object      = sp.N, # here, it is also acceptable to pass an array
                unwrap      = (sp.unwrap_r,sp.unwrap_R),
                pinhole     = sp.pinhole,
                components  = sp.components,
                returnables = ('spectrum',))
    
# run the microscope by rastering around the sample. examine raster_spectra() for details
# about how to handle the class properly.
if sp.run_microscope:
    raster_spectra(symmetry_microscope)

# find candidate symmetries, then go to those sites with the symmetry
# microscope again and save inspectable output by turning on more returnables.
if sp.find_candidates:
    record = find_candidates()
    symmetry_microscope.set_returnables(sp.candidate_returnables)
    get_particulars(symmetry_microscope,record)
