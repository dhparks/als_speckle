import os,string
isfile = os.path.isfile
import speckle,numpy
speckle.io.set_overwrite(True)
import time

import matplotlib
matplotlib.use('TkAgg')
import pylab

def average_dark():
    
    """ This opens the dark file (at path dark_file) and computes an average
    along the frame axis. If the sum already exists, this function will open the
    saved file and return it rather than duplicate the calculations."""
    
    # from info in barker_conditioning_parameters, make the paths
    dark_file = '%s/%s'%(bcp.data_path,bcp.dark_name)
    out_file  = '%s/barker average bg'%(bcp.data_path)

    print "  working on dark file"
    
    if isfile('%s_mag.fits'%out_file) and bcp.use_old_files:
        print "    found saved calculation; opening"
        dark_sum = speckle.io.open('%s_mag.fits'%out_file).astype(numpy.float32)
    else:
        assert isfile(dark_file), "dark file %s not found! check path?"%dark_file
        print "    round raw dark. calculating frame average."
        
        # check the shape of the dark. if 2d, make trivially 3d
        shape = speckle.io.get_fits_dimensions(dark_file)
        if len(shape) == 2: shape = (1,)+tuple(shape)
        
        # because the dark file may be very large, loading it all into memory
        # is prohibitive. instead, open one frame at a time and add it to a running
        # sum of all the frames.
        dark_sum = numpy.zeros(shape[1:],numpy.float32)
        for n in range(shape[0]):
            frame     = speckle.io.openframe(dark_file,n)
            dark_sum += frame.astype(numpy.float32)
        dark_sum *= 1./shape[0] # turn the sum into an average
        
        # save the output to avoid repeating this calculation if the analysis is re-run.
        # to avoid opening the file, delete it or set bcp.use_old_files = False.
        speckle.io.save('%s.fits'%out_file,dark_sum)
        
    # return the sum
    return dark_sum

def clean_dark(i=3):
    """ This function uses a median filter to remove hot pixels.
    If a gpu is available and requested through bcp.device, run the
    median filter on the gpu instead."""
    
    print "    cleaning with median filter"

    # make paths
    in_file = '%s/barker average bg_mag.fits'%bcp.data_path
    out_file = '%s/barker cleaned bg_mag'%bcp.data_path
    
    if isfile('%s_mag.fits'%out_file) and bcp.use_old_files:
        print "    found old cleaned dark file; opening"
        cleaned = speckle.io.open('%s_mag.fits'%out_file).astype(numpy.float32)
    
    else:
        
        assert isfile(in_file), "couldnt find file %s"%in_file
        data = speckle.io.open(in_file).astype(numpy.float32)
        
        if use_gpu: # gpu codepath

            # put the data onto the gpu, then run the median filter the specified number of times.
            gpu_data = cla.to_device(queue,data.astype(numpy.float32))
            for n in range(i): gpu_median5.execute(queue,(data.shape[0],data.shape[1]), gpu_data.data, gpu_data.data)
            cleaned = gpu_data.get()

        if not use_gpu: # cpu codepath

            from scipy.signal import medfilt
            cleaned = numpy.copy(data)
            for n in range(i): cleaned = medfilt(cleaned)
            
        # save the data
        speckle.io.save('%s.fits'%out_file,cleaned)

    # return the cleaned dark
    return cleaned

def correlate_frames(L=256):
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
        correlations = speckle.io.open('%s_mag.fits'%out_file).astype(numpy.float32)
    
    else:
        # to speed the calculation, only correlate a box of size L around the
        # central maximum in each frame. 
        shape  = speckle.io.get_fits_dimensions(in_file)
        if len(shape) == 2:
            shape = (1,shape[0],shape[1])
        maxima = numpy.zeros((shape[0],L,L),numpy.float32)
        
        # slice the out the central maxima
        for n in range(shape[0]):
            frame       = speckle.io.openframe(in_file,n)
            mr,mc       = numpy.unravel_index(frame.argmax(),(frame.shape)) # brightest pixel (row,col)
            maxima[n]   = frame[mr-L/2:mr+L/2,mc-L/2:mc+L/2]
    
        print "    correlating..."
        #if use_gpu:     correlations = gpulib.gpu_phasing.covar_results(gpu_info,subarray,threshold=0.9)[1] # accidentally deleted this file, oops
        #if not use_gpu: correlations = speckle.crosscorr.pairwise_covariances(maxima)
           
        # save the cross-correlations
        speckle.io.save('%s.fits'%out_file,correlations)
        
    return correlations

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
    
    This function is simplified from the most general case because we know that
    there are only two configurations in the speckle patterns. This is not
    always the case.
    """
    
    in_file  = '%s/%s'%(bcp.data_path,bcp.data_name)
    out_file = '%s/barker sum config1'%bcp.data_path
    dm = '%s/%s'%(bcp.data_path,bcp.dust_mask_name)
    
    print "  working on signal average"
    
    if isfile('%s_mag.fits'%out_file) and bcp.use_old_files:
        print "    found old file; opening"
        sum1 = speckle.io.open('%s_mag.fits'%out_file).astype(numpy.float32)

    else:
        
        assert isfile(in_file), "didnt find the input data %s"%in_file
        
        print "    planning"
        # open the dust masks and make plans
        dust_mask = numpy.flipud(speckle.conditioning.open_dust_mask(dm))
        dust_plan = speckle.conditioning.plan_remove_dust_new(dust_mask)
        
        # set up the sums for:
        # 0. all data
        # 1. configuration 1
        # 2. configuration 2
        
        shape = speckle.io.get_fits_dimensions(in_file)
        if len(shape) == 2: shape = (1,shape[0],shape[1])
        sum0  = numpy.zeros(shape[1:],numpy.float32)
        sum1  = numpy.zeros(shape[1:],numpy.float32)
        sum2  = numpy.zeros(shape[1:],numpy.float32)

        # open frames for configuration1 and configuration2. these
        # will serve as the frames against which we align other frames in the
        # configuration. alignment will be done using central speckle only.

        print "    getting alignment frames"

        L = 256

        frame1   = speckle.io.openframe(in_file,0)
        mr1, mc1 =  numpy.unravel_index(frame1.argmax(),frame1.shape)
        align1   = frame1[mr1-L/2:mr1+L/2,mc1-L/2:mc1+L/2]
        
        frame2   = speckle.io.openframe(in_file,frame_corrs[0].argmin())
        mr2, mc2 =  numpy.unravel_index(frame2.argmax(),frame2.shape)
        align2   = frame2[mr2-L/2:mr2+L/2,mc2-L/2:mc2+L/2]

        # now that the alignment frames have been extracted, go through each
        # frame, discover its configuration, do the processing, and add it
        # to the correct sum.
        
        rolls = lambda d, rr, rc: numpy.roll(numpy.roll(d,rr,axis=0),rc,axis=1)
        
        hot_time = 0
        dust_time = 0
        align_time = 0
        open_time = 0
        dark_time = 0
        
        for n in range(shape[0]):
            
            print "    working on frame %s of %s"%(n,shape[0])
            
            # find the configuration
            if frame_corrs[0,n] >= 0.97:
                add_to   = sum1
                align_to = align1
                r,c      = mr1, mc1
                
            if frame_corrs[0,n] <  0.97:
                add_to   = sum2
                align_to = align2
                r,c      = mr2, mc2
                
            # open the data, subtract background, remove dust, remove hot pixels
            # we can do dark subtraction simply because acquisition parameters were the same
            frame = speckle.io.openframe(in_file,n)
            frame = abs(frame-cleaned_dark)
            frame = speckle.conditioning.remove_dust_new(frame,dust_mask,dust_plan=dust_plan)[0]
            
            if use_gpu:
                # gpu code path for hot pixels. execute two kernels. first, medfilt.
                # second, replace hot pixels with median.
                if n == 0:
                    gpu_data_hot  = cla.empty(queue, frame.shape, numpy.float32)
                    gpu_data_cold = cla.empty(queue, frame.shape, numpy.float32)
                gpu_data_hot.set(frame.astype(numpy.float32))
                gpu_median3.execute(queue,frame.shape,  gpu_data_hot.data,gpu_data_cold.data)
                gpu_hotpix.execute(queue, (frame.size,),gpu_data_hot.data,gpu_data_cold.data,numpy.float32(1.25))
                frame = gpu_data_hot.get()
                
            if not use_gpu:
                # cpu code path for hot pixel removal
                frame = speckle.conditioning.remove_hot_pixels(frame)
                
            time4 = time.time()

            # figure out how much we have to roll the data to align it. slice out a
            # subframe to speed this calculation
            sub  = frame[r-L/2:r+L/2,c-L/2:c+L/2]
            x, y = speckle.conditioning.align_frames(sub,align_to=align_to,return_type='coordinates')[0]
    
            # using the coordinates from the subframe, shift the whole frame.
            add_to += rolls(frame,x,y)
            
        # save configuration 1
        speckle.io.save('%s.fits'%out_file,sum1)
    
    # return the sum
    return sum1
     
def prep_speckles():
    # resize through band limiting
    # move max to corner
    # sqrt to set modulus
   
    in_file = '%s/barker sum config1_mag.fits'%bcp.data_path
    out_file = '%s/barker hologram.fits'%bcp.data_path
   
    levels = [5,4,1]
    
    L = 512
    x = 1024
    z = 256

    # put the brightest pixel at (0,0)
    data = cleaned_signal
    indx = numpy.unravel_index(data.argmax(),data.shape)
    data = numpy.roll(numpy.roll(data,-indx[0],axis=0),-indx[1],axis=1)
    
    # inverse transform the data. cut the oversampling ratio. save it.
    idata = numpy.fft.fftshift(numpy.fft.ifft2(data))
    cut   = idata[x-L/2:x+L/2,x-L/2:x+L/2]
    
    speckle.io.save('%s/inverse data.fits'%bcp.data_path,cut,components='polar')
    speckle.io.save('%s/inverse data.jpg'%bcp.data_path, cut,components='complex_hls')
    
    mag, phase = abs(cut),numpy.angle(cut)
    mag = numpy.sqrt(numpy.sqrt(mag))
    speckle.io.save('%s/inverse data rescaled.jpg'%bcp.data_path, mag*numpy.exp(phase*complex(0,1)),components='complex_hls')
    
    # prepare the speckle pattern for phasing
    speckles = numpy.sqrt(abs(numpy.fft.fft2(cut)))
    speckle.io.save('%s/barker phasing 512.fits'%bcp.data_path,speckles)
         
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
    
# see if a gpu is available. output is in "global" so it doesn't need to be
# passed to the various functions
try:
    use_gpu, device_info = check_gpu()
    
    # load gpu stuff
    import speckle.gpu as gpulib
    import pyopencl as cl
    import pyopencl.array as cla
    kp = string.join(gpulib.__file__.split('/')[:-1],'/')+'/kernels/'
    context, device, queue, platform = device_info
    
    gpu_median3 = gpulib.gpu.build_kernel_file(context, device, kp+'medianfilter3.cl') # for signal
    gpu_median5 = gpulib.gpu.build_kernel_file(context, device, kp+'medianfilter5.cl') # for dark
    gpu_hotpix  = gpulib.gpu.build_kernel_file(context, device, kp+'remove_hot_pixels.cl')
    
    print "using gpu"
except:
    use_gpu = False

average_dark   = average_dark()
cleaned_dark   = clean_dark()
frame_corrs    = correlate_frames()
cleaned_signal = average_signal()
prep_speckles()