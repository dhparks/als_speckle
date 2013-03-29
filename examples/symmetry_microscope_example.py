# core
import numpy as np
import itertools

# shared
import speckle
speckle.io.set_overwrite(True)

def make_concentrations(data_in):
    frames, rows, cols = data_in.shape
    powers = np.sum(data_in,axis=-1)       # sum along the component-value axis
    out    = np.zeros((rows,frames),float) # the concentrations
    q2     = (data_in.transpose()/powers.transpose()).transpose()
    c2     = (q2**2).sum(axis=-1).transpose()
    return c2, powers.transpose()

def open_sample(name):
    
    """ Open an image and turn it into an array. The scattering model
    here is that for magnetic dichroism because it gives a minimal
    airy pattern"""
    
    data = speckle.io.openimage(name)
    data = data.astype('float')
    data = 2*data/data.max()-1
    
    return data

def open_illumination():
    illumination = speckle.io.open(params.mag_file)
    illumination *= 1./illumination.max()
    if params.phase_file != None:
        phase = speckle.io.open(params.phase_file)
        illumination *= np.exp(complex(0,1)*phase)
    return illumination

def save_spectra(spectra,name_base):
    # save; cast the results to integer to reduce space.
    
    save_name1 = '%s.fits'%(name_base)
    speckle.io.save('%s/averaged %s'%(params.output_path,save_name1),np.average(spectra,axis=0),components='real')
    
    maxval = spectra.max()
    minval = spectra.min()
    spectra += -minval
    spectra *= 65535/maxval
    spectra = spectra.astype('uint16')
    save_name = '%s maxval_%.4e minval_%.4e.fits'%(name_base,maxval,minval)
    sys.stdout.write('saving spectra: %s\n'%save_name)
    speckle.io.save('%s/%s'%(params.output_path,save_name),spectra)
    
def open_spectra(name):
    
    # do some stuff with splits as regex to separate out the _mag
    # and then extract the fp value to rescale correctly
    # from int16 back to float
    
    data = speckle.io.openfits(name).astype('float32')
    maxval = name.split('maxval_')[1].split('_mag')[0]
    data *= float(maxval)/2**16.
    return data

def raster_spectra():
    
    """ Run the symmetry microscope analysis on each of the objects at the
    coordinates described by make_raster_coords().
    
    """
    
    # make the (x,y) coordinates where the images will be analyzed
    
    combined_spectra = []
    
    import time
    time0 = time.time()
    for n, name in enumerate(params.samples):
        
        # open the sample and load it into the microscope
        sys.stdout.write('loading sample...')
        x = open_sample(name)
        microscope.load(sample=x,returnables=('spectrum_ds','spectrum'))
        sys.stdout.write('done\n')
        
        # based on the size of the object and the step size specified in the
        # parameters, calculate the new raster coordinates. these coordinates
        # describe where the CENTER of the ILLUMINATION array will be found.
        # also allocate memory to hold all the spectra from this sample.
        xcoords, ycoords = make_raster_coords()
        current_spectra  = np.zeros((len(xcoords)*len(ycoords),microscope.rows,128),np.float32)
        
        
        # now iterate over the coordinates, creating a rotational correlation
        # spectrum at each.
        for m,site in enumerate(itertools.product(xcoords,ycoords)):
            col, row = site
            sys.stdout.write('calculating row %s, col %s        \r'%(row,col))
            microscope.run_on_site(row,col)
            current_spectra[m] = microscope.returnables['spectrum_ds']
        sys.stdout.write('\n')
        speckle.io.save('gpu sum.fits',microscope.spectrumsum.get(),components='real')
        exit()
    
        # now we either save or bundle with the other spectra depending on
        # what is specified in params.combine_spectra
        
        if params.combine_output: combined_spectra.append(current_spectra)
        else: save_spectra(current_spectra,'%s %s spectra_ds'%(params.identifier,params.output_labels[n]))
        speckle.io.save('testing spectra ds.fits',current_spectra)
        
    sys.stdout.write("\n\nTimings:\n")
    sys.stdout.write("time measured in example file %s\n"%(time.time()-time0))
    sys.stdout.write("times measured in microscope class:\n")
    microscope.timings()
    sys.stdout.write("average time per site: %s\n"%(microscope.master_time/(len(params.samples)*len(xcoords)*len(ycoords))))
    sys.stdout.write("average interal sites per second: %s\n"%((len(params.samples)*len(xcoords)*len(ycoords))/microscope.master_time))
        
    # after iterating over all samples, save the combined spectra
    if params.combine_output: save_spectra(np.array(combined_spectra),'%s combined spectra_ds'%params.identifier)

def make_raster_coords():

    yr, xr = microscope.sample.shape
    offset = microscope.illumination.shape[0]
    
    x_coords = np.arange(0,yr,params.step_size)-offset/2
    y_coords = np.arange(0,yr,params.step_size)-offset/2
    
    return x_coords, y_coords

def troubleshoot():
    
    sample = open_sample(params.samples[0])
    microscope.load(sample=sample,returnables=params.troubleshoot_returnables)
    
    # run the microscope at site 0,0
    microscope.run_on_site(-256,-512)

    # save the output showing the symmetry candidate.
    for key in microscope.returnables.keys():
        name = '%s/%s troubleshooting_1 %s.fits'%(params.output_path,microscope.compute_device,key)
        speckle.io.save(name,microscope.returnables[key],components='real')
        print name
        
    microscope.run_on_site(0,0)

    # save the output showing the symmetry candidate.
    for key in microscope.returnables.keys():
        name = '%s/%s troubleshooting_2 %s.fits'%(params.output_path,microscope.compute_device,key)
        speckle.io.save(name,microscope.returnables[key],components='real')
        print name

import symmetry_microscope_parameters as params

nf = len(params.samples)
nq = params.unwrap_R-params.unwrap_r

# open the illumination
sys.stdout.write('opening illumination and coherence files...')
illumination = open_illumination()

# open the coherence function
coherence = speckle.io.open(params.coherence_file)
sys.stdout.write('done\n')

# instantiate a symmetry microscope. if user wants a gpu, first check runtimes
# and gpu support using check_gpu().
if params.run_microscope or params.troubleshoot:
    
    # import the microscope code
    sys.stdout.write('importing the unified code...')
    import speckle.symmetry_microscope as sm
    sys.stdout.write(' done\n')
    
    # instantiate the symmetry microscope code. no input is accepted at
    # initialization. during instantiation, the script looks for a gpu and if
    # one is not available, falls back to the cpu code-path.
    sys.stdout.write('instantiating the microscope class...')
    #microscope = sm.microscope()
    microscope = sm.microscope(force_cpu=True)
    sys.stdout.write('done\n')
    
    # load the known information into the microscope. this includes the illumination,
    # coherence (ipsf), unwrap values, and cosine components. later, in
    # raster_spectra(), the sample files will be opened and loaded. at that
    # point, the microscope can be scanned.
    sys.stdout.write('loading data into microscope...')
    microscope.load(illumination=illumination, ipsf=coherence, unwrap=(params.unwrap_r,params.unwrap_R))
    #microscope.load(illumination=illumination, unwrap=(params.unwrap_r,params.unwrap_R))
    sys.stdout.write('done\n')
    
    # this will verify the status of the microscope as not ready to scan
    sys.stdout.write('\nWithout a sample loaded (that happens next), the microscope cant scan:\n')
    microscope.status()
    sys.stdout.write('\n')
    
# run the microscope by rastering around the sample. examine raster_spectra() for details
# about how to handle the class properly.
if params.run_microscope:
    raster_spectra() # this runs the microscope.
    
if params.troubleshoot:
    troubleshoot()
