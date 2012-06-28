# core
import numpy
DFT = numpy.fft.fft2
IDFT = numpy.fft.ifft2
fftshift = numpy.fft.fftshift

import speckle
import speckle.gpu as gpulib
gc = gpulib.gpu_correlations
speckle.io.set_overwrite(True)

class random_walk():
    """ Methods for generating random-walk simulations (more accurately, wiener-processes)"""

    def __init__(self,n,radius):
        
        assert isinstance(n,int), "number of balls must be int"
        assert isinstance(radius, (int,float)), "ball radius must be float or int"
        
        if radius == 0:
            self.do_convolve == False
        if radius > 0:
            self.dft_object = DFT(fftshift(speckle.shape.circle((params.N,params.N),radius)))
            self.do_convolve = True
        
        self.coordinates = (params.N*numpy.random.rand(n,2)).astype(float)

    def displace(self,stepsize):
        
        assert isinstance(stepsize,(int,float)), "step size must be float or int"
        
        deltas = stepsize*numpy.random.randn(len(self.coordinates),2)
        self.coordinates += deltas
        self.coordinates = numpy.mod(self.coordinates,params.N)
        
    def make(self):
        
        grid = numpy.zeros((params.N,params.N),float)
    
        # i don't know how to do this without a loop, barf
        sites = len(self.coordinates)
        for i in range(sites):
            row, col = self.coordinates[i,:]
            grid[int(row),int(col)] = 1
            
        if self.do_convolve:
            grid = numpy.clip(abs(IDFT(DFT(grid)*self.dft_object)),0,1)
        return 1-grid

def make_paths():
    import os
    assert os.path.isdir(params.output_path), "output path does not exist!"
    ball_path = '%s/real space images/balls density%s radius%s step%s'%(params.output_path,params.density,params.ballradius,params.brownianstep)
    analysis_path = '%s/analysis/bd%s br%s bs%s pr%s'%(params.output_path,params.density,params.ballradius,params.brownianstep,params.pinhole)
    try: os.makedirs(ball_path)
    except: pass
    try: os.makedirs(analysis_path)
    except: pass
    return ball_path, analysis_path

def samples():
    print "making %s sample images"%nf
    objs = int(params.N**2*params.density)
    balls = random_walk(objs,params.ballradius)
    for frame in range(nf):
        print "  "+str(frame)
        balls.displace(params.brownianstep)
        image = balls.make()
        speckle.io.save('%s/%s.png'%(ball_path,frame),image)
        
def open_sample(name):
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
        length = numpy.product(shape[n+1:])
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
    
def microscope_stages(sm_instance,y,x):
    # combine pedagogically-separate but logically-continuous (in the sense of not requiring
    # additional input) symmetry microscope stages into one command.
    sm_instance.slice_object(y,x)
    sm_instance.make_speckle()
    #sm_instance.blur_speckle(site) # uncomment to simulate partial coherence; slows simulation considerably due to additional ffts
    sm_instance.unwrap_speckle()
    sm_instance.rotational_correlation()
    sm_instance.decompose_spectrum()
    
    spectrum = sm_instance.returnables['spectrum']
    print spectrum.shape
    for n in range(spectrum.shape[0]):
        print spectrum[n,0]
    exit()
    
    spectrum = sm_instance.spectrum_gpu.get()
    return spectrum

def symmetry_microscope(sm_instance):
    
    print "running symmetry microscope, %s sites/frame"%(nx*ny)
    
    # allocate memory to store all the spectra that will be generated
    try: spectra = numpy.zeros((nf,nx*ny,nq,nc),float)
    except MemoryError:
        print "Had a malloc error in symmetry_microscope()!\nSuggest simulating fewer frames or sites"
        exit()
    
    for frame in range(nf):
        print "  frame %s"%frame
        
        # open a sample frame from disk and load onto the gpu
        open_name = '%s/%s_mag.png'%(ball_path,frame)
        sample = open_sample(open_name)
        sm_instance.load_object(sample)

        # now run the speckle generation and decomposition over the sample
        # at the raster coordinates specified in ycoords, xcoords
        site = 0
        for row in ycoords:
            for col in xcoords:
                spectra[frame,site] = microscope_stages(sm_instance,row,col)
                print spectra[frame,site].max()
                site += 1
                
    # save the results as integer to reduce space
    maxval = spectra.max()
    print maxval
    spectra *= 65535/maxval
    spectra = spectra.astype('uint16')
    save_name = '%s/spectra bd%s br%s bs%s pr%s maxval_%.4e.fits'%(analysis_path,params.density,params.ballradius,params.brownianstep,params.pinhole,maxval)
    speckle.io.save(save_name,spectra)

def get_particulars(sm_instance,record):
    
    loaded = -1
    
    for entry in record:
        frame, record_site = entry[0], entry[1]
        
        # if frame isnt the current sample on the gpu, load it
        if frame != loaded:
            open_name = '%s/%s_mag.png'%(ball_path,frame)
            data = open_sample(open_name)
            sm_instance.load_object(data)
            loaded = frame
            
        # raster scan through the coordinate list just like in symmetry_microscope.
        # however, actually do the computation only if the site in the record entry
        # is the current raster site. it would be smarter to just do some more modulo
        # arithmetic to find out (row,col)...
        raster_site = 0
        for row in ycoords:
            for col in xcoords:
                if record_site == raster_site:
                    microscope_stages(sm_instance,row,col)
                    
                    # save the output showing the symmetry candidate. the necessary
                    # data has been pulled off the gpu behind the scenes.
                    speckle.io.save('%s/%s %s illuminated.png'%(analysis_path,frame,record_site),sm_instance.returnables['illuminated'],components='real')
                    speckle.io.save('%s/%s %s correlation.fits'%(analysis_path,frame,record_site),sm_instance.returnables['correlation'])
                    speckle.io.save('%s/%s %s speckle.jpg'%(analysis_path,frame,record_site),sm_instance.returnables['speckle_blocker'],color_map='B')

def find_candidates():
    
    import glob
    
    # find the coordinates of those components which pass the candidate test.
    # the test is that the component value should be more than 60% of the spectrum
    # at that |q|, ie, it should be dominant.
    
    file = glob.glob('%s/spectra bd%s br%s bs%s pr%s maxval*.fits'%(analysis_path,params.density,params.ballradius,params.brownianstep,params.pinhole))[0]
    data = open_spectra(file)
   
    power = numpy.sum(data,axis=3) # sum along the component-value axis
    ranks = numpy.zeros(data.shape,'uint8')
    
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            current_powers = powers[i,j]
            components = data[i,j].transpose()
            ranks[i,j] = (256*((components/current_powers).transpose())).astype('uint8')
   
    # filtered marks in "space" where the candidates are
    t = (256*params.candidate_threshold).astype('uint8')
    filtered = numpy.where(ranks > t,1,0)

    # now, turn filtered into a list of (frame, site, row, col) indices
    x,y,z,t = filtered.shape
    filtered = filtered.ravel()
    passed = numpy.sum(filtered) 
    get = numpy.argsort(filtered)[-passed:]
    record = numpy.zeros((passed,5),int)

    kept = 0
    for n,index in enumerate(get):
        # get indices via modulo arithmetic.
        frame, site, row, col = modulo_indices(index,filtered.shape)
        if col > 0 and col < 10:
            # not interested in m = 2 or m > 20
            record[kept] = frame, site, row, col, ranks[frame,site,row,col]
            kept += 1
    
    # re-order record by frame
    record = record[0:kept]
    indices2 = numpy.argsort(record[:,0])
    record = record[indices2]
        
    save_name = '%s/candidate record bd%s br%s bs%s pr%s.fits'%(analysis_path,params.density,params.ballradius,params.brownianstep,params.pinhole)
    speckle.io.save(save_name,record)
    return record

import colloid_parameters as params

# rename some stuff for convenience
nf = params.frames
nc = len(params.components)
nq = params.unwrap_R-params.unwrap_r

# first, set up paths
ball_path, analysis_path = make_paths()

# next, make random walk images
if params.make_samples: samples()

# now initialize the gpu and instantiate the symmetry microscope gpu code.
if params.run_microscope or params.find_candidates:
    gpuinfo = gpulib.gpu.init()
    microscope = gc.angular_correlations(gpuinfo,params.N,params.unwrap_r,params.unwrap_R,params.pinhole,params.components,interrupts=('spectrum'))
    xcoords, ycoords = gc.make_raster_coords(params.N,params.step_size,params.step_size,size=params.view_size)
    nx = len(xcoords)
    ny = len(ycoords)
    
# run the microscope
if params.run_microscope: symmetry_microscope(microscope)

# find candidate symmetries, then go to those sites with the symmetry
# microscope again and save inspectable output by turning on "interrupts"
if params.find_candidates:
    record = find_candidates()
    microsope.change_interrupts(params.interrupts)
    get_particulars(record)

    

    
    





