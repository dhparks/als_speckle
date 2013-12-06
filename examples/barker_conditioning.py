import os,string
isfile = os.path.isfile
import speckle,numpy
speckle.io.set_overwrite(True)
import time

# see if a gpu is available.
gpu_info = speckle.gpu.init()
if gpu_info != None: use_gpu = True
else: use_gpu = False

def average_dark(i=3):
    
    """ This opens the dark file (at path dark_file) and computes an average
    along the frame axis. If the sum already exists, this function will open the
    saved file and return it rather than duplicate the calculations."""
    
    # from info in barker_conditioning_parameters, make the paths
    dark_file = '%s/%s'%(bcp.data_path,bcp.dark_name)
    out_file  = '%s/barker conditioned dark'%(bcp.data_path)

    print "  working on dark file"
    
    if isfile('%s_mag.fits'%out_file) and bcp.use_old_files:
        print "    found saved calculation; opening"
        cleaned = speckle.io.open('%s_mag.fits'%out_file).astype(numpy.float32)
        
    else:
        assert isfile(dark_file), "dark file %s not found! check path?"%dark_file
        print "    round raw dark. conditioning dark file."
        
        # average the dark data along the frame axis if it is 3d
        dark  = speckle.io.open(dark_file)
        if dark.ndim == 2: average = dark
        if dark.ndim == 3: average = numpy.mean(dark,axis=0)
        
        # put the average through a median filter to reduce noise
        if use_gpu: 
            
            # gpu code path
            import string
            import pyopencl as cl
            import pyopencl.array as cla
            
            # build the kernel
            context, device, queue, platform = gpu_info
            kp = string.join(speckle.gpu.__file__.split('/')[:-1],'/')+'/kernels/'
            gpu_median5 = speckle.gpu.build_kernel_file(context, device, kp+'medianfilter5.cl')

            # put the data onto the gpu, then run the median filter the specified number of times.
            gpu_data = cla.to_device(queue,average.astype(numpy.float32))
            for n in range(i): gpu_median5.execute(queue, (average.shape[0],average.shape[1]), gpu_data.data, gpu_data.data)
            cleaned = gpu_data.get()

        if not use_gpu:
            
            # cpu codepath

            from scipy.signal import medfilt
            cleaned = numpy.copy(data)
            for n in range(i): cleaned = medfilt(cleaned)
            
        # save the data
        speckle.io.save('%s.fits'%out_file,cleaned)
        
    # return the sum
    return cleaned

def correlate_frames():
    """ Drift in the sample means that the speckle patterns collected over time
    fall into different configurations. Here, we calculate all pair-wise cross-
    correlation coefficients between frames i and j. This resulting (n x n) array
    of correlation coefficients will be used to separate the different
    configurations in a separate function. Can be run on either the cpu or gpu.
    """
    
    print "  correlating signal frames to separate configurations"
    
    in_file  = '%s/%s'%(bcp.data_path,bcp.data_name)
    out_file = '%s/barker correlations'%bcp.data_path
    
    if isfile('%s_mag.fits'%out_file) and bcp.use_old_files:
        print "    find old file; opening"
        corrs = speckle.io.open('%s_mag.fits'%out_file).astype(numpy.float32)
    
    else:
        
        L = bcp.correlation_box
        
        # slice out a region around the brightest portion of each frame.
        # note that for data which has been polluted by cosmic rays, for example,
        # doing this first will give bad results. for REALLY dirty data, the
        # order of operations should be to condition each frame, then do the
        # frame correlations, then add the configurations.
        print "    slicing"
        shape  = speckle.io.get_fits_dimensions(in_file)
        if len(shape) == 3:
            maxima = numpy.zeros((shape[0],L,L),numpy.float32)
            for f in range(shape[0]):
                frame     = speckle.io.openframe(in_file,f)
                mr,mc     = numpy.unravel_index(frame.argmax(),frame.shape) # brightest pixel (row,col)
                maxima[f] = frame[mr-L/2:mr+L/2,mc-L/2:mc+L/2]
        
        # pairwise correlations between all frame pairs
        print "    correlating"
        corrs  = speckle.crosscorr.pairwise_covariances(maxima,gpu_info=gpu_info)
           
        # save the cross-correlations
        speckle.io.save('%s.fits'%out_file,corrs)
        
    return corrs

def average_signal():
    # make the dust plan
    
    # open each frame.
    # fill in the dust
    # remove the hot pixels
    # subtract the background
    # if n > 0, align to the initial frame
    
    """ This function averages the signal similarly to the way the dark file was
    averaged. However, in the signal file, the data frames must be sorted into
    configurations using the correlations calculated in correlate_frames(), then
    have dust and hot-pixels removed, then the background subtracted. After
    alignment, the frames may then be averaged.
    """
    
    in_file  = '%s/%s'%(bcp.data_path,bcp.data_name)
    out_file = '%s/barker sums'%bcp.data_path

    if bcp.dust_mask_name != None: dm = '%s/%s'%(bcp.data_path,bcp.dust_mask_name)
    else: dm = None
    
    print "  working on signal average"
    
    if isfile('%s_mag.fits'%out_file) and bcp.use_old_files:
        print "    found old file; opening"
        config_sums = speckle.io.open('%s_mag.fits'%out_file).astype(numpy.float32)

    else:
        
        assert isfile(in_file), "didnt find the input data %s"%in_file
        
        rolls = lambda d,rr, rc: numpy.roll(numpy.roll(d,rr,axis=0),rc,axis=1)

        print "    planning dust removal"
        # open the dust masks and make plans
        if dm != None:
            dust_mask = numpy.flipud(speckle.conditioning.open_dust_mask(dm))
            dust_plan = speckle.conditioning.plan_remove_dust(dust_mask)
        
        # using the correlations data, sort the frames into configurations
        print "    finding configurations"
        configs = speckle.conditioning.sort_configurations(frame_corrs,louvain=True)
        
        # for each configuration, iterate over frames. for each frame,
        # align to a reference, remove dust, remove hot pixels, subtract dark,
        # and add to the configuration sum.
        config_sums = numpy.zeros(((len(configs),)+dust_mask.shape),numpy.float32)
        for nc, config in enumerate(configs):
            
            print "working on configuration %s of %s"%(nc+1,len(configs))
            
            # set up the frame sum of this configuration
            config_sum = numpy.zeros_like(dust_mask)
            
            # slice out a portion of the first frame in the configuration to
            # serve as alignment fiducial
            L         = bcp.frame_align_box
            ref_frame = speckle.io.open(in_file)[config[0]]
            mr, mc    = numpy.unravel_index(ref_frame.argmax(),ref_frame.shape)
            reference = ref_frame[mr-L/2:mr+L/2,mc-L/2:mc+L/2]
            
            # pre-calculate the roll coordinates for all frames. this saves
            # recalculating the FFT of the reference frame each time.
            frames = speckle.io.open(in_file)[config,mr-L/2:mr+L/2,mc-L/2:mc+L/2]
            coords = speckle.conditioning.align_frames(frames,align_to=reference,return_type='coordinates')

            for nf,f in enumerate(config):
                
                frame = speckle.io.open(in_file)[f]
                frame = speckle.conditioning.subtract_dark(frame,average_dark)
                if dm!= None:
                    frame = speckle.conditioning.remove_dust(frame,dust_mask,dust_plan=dust_plan)[0]
                frame = speckle.conditioning.remove_hot_pixels(frame,gpu_info=gpu_info,threshold=1.5)
                
                x, y  = coords[nf]
                config_sum += rolls(frame,coords[nf][0],coords[nf][1])
                
            config_sums[nc] = config_sum
        
        # save the configurations
        speckle.io.save('%s.fits'%out_file,config_sums)
    
    # return the sum
    return config_sums
     
def prep_speckles():
    # resize through band limiting
    # move max to corner
    # sqrt to set modulus
   
    in_file = '%s/barker sum config1_mag.fits'%bcp.data_path
    out_file = '%s/barker hologram.fits'%bcp.data_path
    
    L = 512

    # each frame of data is a different configuration
    data = cleaned_signal
    for nf, frame in enumerate(data):
        
        # put the brightest pixel at (0,0)
        indx  = numpy.unravel_index(frame.argmax(),frame.shape)
        frame = numpy.roll(numpy.roll(frame,-indx[0],axis=0),-indx[1],axis=1)
        
        # inverse transform the data. cut the oversampling ratio. save it.
        iframe = numpy.fft.fftshift(numpy.fft.ifft2(frame))
        s      = iframe.shape
        cut    = iframe[s[0]/2-L/2:s[0]/2+L/2,s[1]/2-L/2:s[1]/2+L/2]
        
        speckle.io.save('%s/inverse data config%s.fits'%(bcp.data_path,nf),cut,components='polar')
        speckle.io.save('%s/inverse data config%s.jpg'%(bcp.data_path,nf), cut,components='complex_hsv')
        
        mag, phase = numpy.abs(cut),numpy.angle(cut)
        mag = numpy.log(mag)
        speckle.io.save('%s/inverse data rescaled config%s.jpg'%(bcp.data_path,nf), mag*numpy.exp(phase*complex(0,1)),components='complex_hsv')
        
        # prepare the speckle pattern for phasing
        speckles = numpy.sqrt(numpy.abs(numpy.fft.fft2(cut)))
        speckle.io.save('%s/barker phasing 512 config%s.fits'%(bcp.data_path,nf),speckles)
         
def dumb_average_signal():
    # this is basically just to get the location of the dust
    
    in_file  = '%s/barker-500ev-long.fits'%data_path
    out_file = '%s/barker average dumb'%data_path
    
    if isfile('%s_mag.fits'%out_file):
        summed = speckle.io.open('%s_mag.fits'%out_file).astype(numpy.float32)
        return summed

    else:
    
        print in_file
        assert isfile(in_file), "signal infile not found!"
        
        shape     = speckle.io.get_fits_dimensions(in_file)
        if len(shape) == 2: shape = (1,shape[0],shape[1])
        print shape
        summed    = numpy.zeros((shape[1],shape[2]),numpy.float32)
    
        for n in range(shape[0]):
            frame = speckle.io.openframe(in_file,n)
            summed += frame.astype(numpy.float32)
        summed *= 1./shape[0]    
        summed = speckle.conditioning.remove_hot_pixels(summed)
            
        speckle.io.save('%s.fits'%out_file,summed)
        
        rc = abs(summed-ad)**(0.25)
        blocker = 1-speckle.shape.circle((summed.shape[0],summed.shape[1]),45,center=(945,945),AA=False)
        speckle.io.save('%s_forgimp1.png'%out_file,rc)
        
        rcm = rc.min()
        print rcm
        rc = rc*blocker+rcm*(1-blocker)
        speckle.io.save('%s_forgimp2.png'%out_file,rc)
        
        return summed
    
import barker_conditioning_parameters as bcp

print "running barker conditioning script"
average_dark   = average_dark()
frame_corrs    = correlate_frames()
cleaned_signal = average_signal()
prep_speckles()