# core
import numpy as np
DFT = np.fft.fft2
IDFT = np.fft.ifft2
fftshift = np.fft.fftshift

import speckle
speckle.io.set_overwrite(True)

def check_parameters():
    
    """ Make sure no nonsense came in through symmetry_microscope_parameters.
    Some of these checks are repeated in later code. Additional and more
    rigorous checks also happen later"""
    
    assert sp.make_samples in (0,1,True,False), "make_samples must be boolean-evaluable"
    
    if sp.make_samples:
        assert sp.device in ('cpu','gpu'), "unrecognized device %s"%sp.device
        assert isinstance(sp.N,int), "array size must be integer"
        assert isp2(sp.N), "array size must be power of 2"
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
        assert isinstance(sp.stepsize,int), "stepsize must be int"
        assert isinstance(sp.view_size,int), "view_size must be int"
        assert isinstance(sp.components,(list,tuple,np.ndarray)), "components must be iterable type"
        assert all([isinstance(e,int) for e in sp.components]), "each component must be int"
        
    if sp.find_candidates:
        assert isinstance(sp.candidate_threshold,float), "candidate threshold must be float"
        assert isinstance(sp.candidate_returnables,(list,tuple)), "candidate returnables must be an iterable type"
        assert all([isinstance(e,str) for e in sp.candidate_returnables]), "each returnable label must be a string"
        
    assert isinstance(sp.output_path,str), "output_path must be a string"

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

def samples():
    
    """ Make and save a set of images showing the brownian motion of a set of
    particles. Uses the random_walk class in generators.
    
    parameters: N, density, frames, ballradius, brownianstep"""
    
    import sys
    sys.path.append('../generators')
    import random_walk as rw
    
    print "making %s sample images"%nf
    objs = int(sp.N**2*sp.density)
    balls = rw.random_walk(objs,sp.ballradius)
    for frame in range(nf):
        print "  "+str(frame)
        balls.displace(sp.brownianstep)
        image = balls.make()
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
    # sm_instance stands for "symmetry microscope instance"
    
    print "running symmetry microscope, %s sites/frame"%(nx*ny)
    
    # allocate memory to store all the spectra that will be generated
    spectra = np.zeros((nf,nx*ny,nq,nc),float)

    for frame in range(nf):
        print "  frame %s"%frame
        
        # open a sample frame from disk and load into the class
        open_name = '%s/%s_mag.png'%(ball_path,frame)
        sample = open_sample(open_name)
        sm_instance.set_object(sample)

        # now run the speckle generation and decomposition over the sample
        # at the raster coordinates specified in ycoords, xcoords. after running
        # the simulation at each site, copy the spectrum to the saving array
        site = 0
        for row in ycoords:
            for col in xcoords:
                print "    %s %s"%(row,col)
                sm_instance.run_on_site(row,col)
                spectra[frame,site] = sm_instance.returnables['spectrum']
                site += 1
                
    # save; cast the results to integer to reduce space. the loss of dynamic range
    # here is acceptable.
    maxval = spectra.max()
    spectra *= 65535/maxval
    spectra = spectra.astype('uint16')
    save_name = '%s/%s spectra bd%s br%s bs%s pr%s maxval_%.4e.fits'%(analysis_path,sp.device,sp.density,sp.ballradius,sp.brownianstep,sp.pinhole,maxval)
    speckle.io.save(save_name,spectra)

def get_particulars(sm_instance,record):
    
    """ Get the relevant output from candidates identified by find_candidates.
    Same code structure as raster_scan() but only runs on a few sites
    and saves different output.
    
    arguments:
        sm_instance: an instance of a microscope class
        record: the ouput of find_candidates, listing which illumination sites
            in which random-walk frames gave speckle with a dominant symmetry.
            this could also be supplied by hand as a list of tuples if you wanted
            to look at some sites manually.
    """
    
    loaded = None
    for entry in record:
        frame, record_site = entry[0], entry[1]
        
        # if frame isnt the current sample on the gpu, load it
        if frame != loaded:
            open_name = '%s/%s_mag.png'%(ball_path,frame)
            data = open_sample(open_name)
            sm_instance.set_object(data)
            loaded = frame
            
        # raster scan through the coordinate list just like in raster_spectra.
        # however, actually do the computation only if the site in the record entry
        # is the current raster site.
        raster_site = 0
        for row in ycoords:
            for col in xcoords:
                if record_site == raster_site:
        
                    sm_instance.run_on_site(row,col)
        
                    # save the output showing the symmetry candidate.
                    for key in sm_instance.returnables.keys():
                        name = '%s/%s %s %s %s.png'%(analysis_path,sp.device,frame,record_site,key)
                        speckle.io.save(name,sm_instance.returnables[key],components='real')
                    raster_site += 1
    
def find_candidates():
    
    import glob
    
    # find the coordinates of those components which pass the candidate test.
    # the test is that the component value should be more than 60% of the spectrum
    # at that |q|, ie, it should be dominant.
    
    file = glob.glob('%s/cpu spectra bd%s br%s bs%s pr%s maxval*.fits'%(analysis_path,sp.density,sp.ballradius,sp.brownianstep,sp.pinhole))[0]
    data = open_spectra(file)
   
    power = np.sum(data,axis=3) # sum along the component-value axis
    ranks = np.zeros(data.shape,'uint8')
    
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            current_powers = powers[i,j]
            components = data[i,j].transpose()
            ranks[i,j] = (256*((components/current_powers).transpose())).astype('uint8')
   
    # filtered marks in "space" where the candidates are
    t = (256*sp.candidate_threshold).astype('uint8')
    filtered = np.where(ranks > t,1,0)

    # now, turn filtered into a list of (frame, site, row, col) indices
    filtered = filtered.ravel()
    passed = np.sum(filtered) 
    get = np.argsort(filtered)[-passed:]
    record = np.zeros((passed,5),int)

    kept = 0
    filtered_shape = filtered.shape
    for n,index in enumerate(get):
        # get indices via modulo arithmetic.
        frame, site, row, col = modulo_indices(index,filtered_shape)
        if col > 0 and col < 10:
            # not interested in m = 2 or m > 20
            record[kept] = frame, site, row, col, ranks[frame,site,row,col]
            kept += 1
    
    # re-order record by frame
    record = record[0:kept]
    indices2 = np.argsort(record[:,0])
    record = record[indices2]
        
    save_name = '%s/candidate record bd%s br%s bs%s pr%s.fits'%(analysis_path,sp.density,sp.ballradius,sp.brownianstep,sp.pinhole)
    speckle.io.save(save_name,record)
    return record

def make_raster_coords(N,xstep,ystep,size=None):

    if size == None:
        start, stop = 0,N
    else:
        assert size%2 == 0, "size to make_coords must be even"
        start, stop = 0,size

    x_coords = np.arange(start,stop,xstep)
    y_coords = np.arange(start,stop,ystep)
    return x_coords, y_coords

### throat clearing

import symmetry_microscope_parameters as sp
check_parameters()

nf = sp.frames
nc = len(sp.components)
nq = sp.unwrap_R-sp.unwrap_r

### now lets get down to tacks. first, set up paths
ball_path, analysis_path = make_paths()

# next, make random walk images. these get saved in ball_path
if sp.make_samples: samples()

# now instantiate a symmetry microscope. notice that the invocation is
# essentially identical for cpu vs gpu, only differing in that the gpu instance
# requires information about the gpu and in the segretation of the gpu code
# away from the cpu code (gpu: speckle.gpu.gpu_correlations vs cpu: speckle.symmetries)
if sp.run_microscope or sp.find_candidates:
    
    # see if a gpu is available. if not, fall back to cpu.
    # this is the same strategy as used in domains_example
    if sp.device == 'gpu':
        try:
            import speckle.gpu as gpulib # this can throw an error
        except gpulib.gpu.GPUInitError as error:
            print error,"\nfalling back to cpu"
            sp.device == 'cpu'
            
    if sp.device == 'gpu':
        try:
            gpuinfo = gpulib.gpu.init() # this can throw various errors
            microscope_code = gpulib.gpu_correlations.gpu_microscope
        except gpulib.gpu.GPUInitError as error:
            print error, "\nfalling back to cpu"
            sp.device = 'cpu'

    if sp.device == 'cpu': microscope_code = speckle.symmetries.cpu_microscope
        
    # now that we've found out if a gpu is available or not, instantiate the class
    # and load information into it. the loading order is important! many things
    # depend on the size of the simulation (sp.N) and it must be first. here, set_object
    # just tells the class the size of the simulation; later, real arrays will be passed
    # into the class.
    if sp.device == 'gpu': symmetry_microscope = microscope_code(gpuinfo)
    if sp.device == 'cpu': symmetry_microscope = microscope_code()
    symmetry_microscope.set_object(sp.N)
    symmetry_microscope.set_pinhole(sp.pinhole)
    symmetry_microscope.set_unwrap((sp.unwrap_r,sp.unwrap_R))
    symmetry_microscope.set_cosines(sp.components)
    symmetry_microscope.set_returnables(('spectrum',))
    
    # for reference, here is an alternate way to instantiate the gpu microscope class where
    # the elements are loaded in at init instead of through the set_xxx methods.
    # in this case, __init__ handles the above ordering automatically.
    """ symmetry_microscope = microscope_code(gpuinfo,
                object      = sp.N,
                unwrap      = (sp.unwrap_r,sp.unwrap_R),
                pinhole     = sp.pinhole,
                components  = sp.components,
                returnables = ('spectrum',))"""
    
    # now make the (x,y) coordinates where the images will be analyzed
    xcoords, ycoords = make_raster_coords(sp.N,sp.step_size,sp.step_size,size=sp.view_size)
    nx = len(xcoords)
    ny = len(ycoords)
    
# run the microscope by rastering around the sample. examine raster_spectra() for details
# about how to handle the class properly. the pinhole, the unwrap, and the cosine component
# were passed at instantiation, but at every frame of the simulated random walk the object
# will be reloaded
if sp.run_microscope: raster_spectra(symmetry_microscope)

# find candidate symmetries, then go to those sites with the symmetry
# microscope again and save inspectable output by turning on more returnables.
if sp.find_candidates:
    record = find_candidates()
    microscope.set_returnables(sp.candidate_returnables)
    get_particulars(microscope,record)

    

    
    





