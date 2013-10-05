### backend for iterative imaging and back-propagation interface
import numpy
import sys
sys.path.insert(0,'..')
import speckle
import time
import os
import scipy.misc as smp
import Image
import math
speckle.io.set_overwrite(True)

class backend():

    def __init__(self,gpu_info=None):
    
        # this class receives gpu info from the flask_server, which
        # is the original instantiator of the gpu context.
        
        self.machine = speckle.phasing.phasing(gpu_info=gpu_info)
        
        self.reconstructions = {}
        self.modulus = None
        self.support = None
        self.loaded  = False

        if gpu_info != None:
            self.use_gpu = True
            self.gpu     = gpu_info
        else:
            self.use_gpu = False
        
    def make_blocker(self,power):
        
        self.blocker_power = power
        
        if power <= 1:
        
            # unwrap self.rs_data and integrate radially. define a blocker which
            # blocks 90% or 95% of the total power of the inverted hologram.
            
            uwp = speckle.wrapping.unwrap_plan(0,self.data_shape[0]/2,(self.data_shape[0]/2,self.data_shape[1]/2),columns=360)
            uw  = speckle.wrapping.unwrap(numpy.abs(self.rs_data),uwp)
            rad = numpy.sum(uw,axis=1)*numpy.arange(uw.shape[0])
            
            r_power = numpy.sum(rad)
            r_cut   = numpy.abs(numpy.cumsum(rad)/r_power-power).argmin()
            
            self.blocker = 1-speckle.shape.circle(self.data_shape,r_cut)
            
        if power > 1:
            
            self.blocker = 1-speckle.shape.circle(self.data_shape,power)
            
    def save_zoom_images(self):
            
            # make images with linear scaling, sqrt scaling, log scaling. this might
            # take a little bit of time.
            mag, phase = abs(self.rs_data), numpy.angle(self.rs_data)
            mag *= self.blocker
            
            # resize, then save the color and scale permutations of first_frame.
            linr = mag
            sqrt = numpy.sqrt(mag)
            logd = numpy.log((mag-mag.min())/mag.max()*1000+1)
      
            #self.rs_image_linear = speckle.io.complex_hsv_image(linr)
            #self.rs_image_sqrt   = speckle.io.complex_hsv_image(sqrt*numpy.exp(complex(0,1)*phase))
            self.rs_image_log    = speckle.io.complex_hsv_image(logd*numpy.exp(complex(0,1)*phase))
            
            imgs = {'logd':smp.toimage(self.rs_image_log)}
            n    = 0
            ds   = self.rs_data.shape
            w, h = self.rs_data.shape
            self.zoom_widths = []
            
            l_min, step, l_max  = 128., 0.8, max(ds)
            step   = 0.8
            nzooms = int(math.floor(math.log(l_min/l_max)/math.log(step))+1)

            big_image = Image.new('RGB',(self.resize*nzooms,self.resize))
            
            for n in range(nzooms):
                w1, h1 = int(w*(step**n)), int(h*(step**n))
                box    = (ds[1]/2-w1/2,ds[0]/2-h1/2,ds[1]/2+w1/2,ds[0]/2+h1/2)
                for key in imgs.keys():
                    y = imgs[key].crop(box).resize((self.resize,self.resize),Image.ANTIALIAS)
                    y.save("./static/imaging/images/zoom_%s_%s_%s_%s.png"%(self.data_id,n,self.blocker_power,key))
                    self.zoom_widths.append(w1)
                    big_image.paste(y,(self.resize*n,0))
            
            big_image.save('./static/imaging/images/zooms_%s_%s_%s.png'%(self.data_id,self.blocker_power,key))
            big_image.save('./static/imaging/images/zooms_%s_%s_%s.jpg'%(self.data_id,self.blocker_power,key))

            self.zooms = nzooms
        
    def load_data(self,path,project,blocker=0.8,resize=300):
    
        # open and prepare data for display. depending on project,
        # the data gets prepared in different ways. for example, when the
        # project is fth, the central maximum gets blockered more extensively
        # than when the project is cdi.
        
        def _invert():
            
            # center the speckle pattern. roll to corner. invert
            data = speckle.conditioning.find_center(self.fourier_data,return_type='data')
            rolled = numpy.fft.fftshift(data)
            return numpy.fft.fftshift(numpy.fft.fft2(rolled)), rolled
        
        # assign a unique id to the current data
        self.data_id = int(time.time())
        
        # this is the size of the saved images
        self.resize = resize
        
        # clear old files; otherwise they will accumulate massively
        for folder in ('static/imaging/images','static/imaging/csv'):
            for f in os.listdir(folder):
                os.remove(os.path.join(folder,f))
                
        # open the data and get the shape. if the data is trivially 3d,
        # convert trivially to 2d. if the data is substantively 3d, convert
        # to 2d by averaging along the frame axis.
        fourier_data = speckle.io.open(path).astype(numpy.float32)
        if fourier_data.ndim == 3:
            if fourier_data.shape[0] == 1: fourier_data = fourier_data[0]
            if fourier_data.shape > 1: fourier_data = numpy.average(fourier_data,axis=0)
        self.fourier_data = fourier_data
        
        # if not square, embed in a square array
        if self.fourier_data.shape[0] != self.fourier_data.shape[1]:
            g   = max(self.fourier_data.shape)
            new = numpy.zeros((g,g),numpy.float32)
            new[:self.fourier_data.shape[0],:self.fourier_data.shape[1]] = self.fourier_data
            self.fourier_data = new
        self.data_shape = self.fourier_data.shape
        
        # now process the data into loadable images
        self.rs_data, self.fourier_data = _invert()
        self.make_blocker(blocker)
        self.save_zoom_images()
        
        # mark data as not loaded in case it will be reconstructed
        self.loaded = False
        self.reconstructions = {}

    def make_support(self, regions):
        
        # given an incoming dictionary regions processed fomr the json
        # request to the flask_server, build an initial support for cdi
        
        self.support = numpy.zeros(self.rs_data.shape,numpy.float32)
        
        z  = int(regions['zoom'])
        w  = self.zoom_widths[z]
        x  = self.data_shape[0]/2

        del regions['zoom']
        for key in regions.keys():
            r  = regions[key]
            r0 = int(r['rmin']/300.*w+x-w/2)
            r1 = int(r['rmax']/300.*w+x-w/2)
            c0 = int(r['cmin']/300.*w+x-w/2)
            c1 = int(r['cmax']/300.*w+x-w/2)
            self.support[r0:r1,c0:c1] = 1

        self.loaded = False
        self.reconstruction = {}
            
    def propagate(self,params,project):
        """ This method runs the back propagation routine. While the data is
        internally propagated at a large power of 2 to allow propagation limit
        to be fairly large, the data returned to the user is only that specified
        in the selected region. """
        
        def _save_acutance(z):
            acutancef = open('static/imaging/csv/acutance_%s_%s.csv'%(self.data_id,self.bp_id),'w')
            acutancef.write("z,acutance\n")
            for zn, av in zip(z,self.acutance):
                row = '%s,%.3f\n'%(zn,av)
                acutancef.write(row)
            acutancef.close()
        
        # assign a unique id to the back-propagation results. this prevents
        # the browser from caching the results
        self.bp_id = int(time.time())
        
        # from the parameters, calculate and slice the selected data pixels
        if project == 'fth':
            w  = self.zoom_widths[params['zoom']]
            rd = self.rs_data.shape[0]/2-w/2
            cd = self.rs_data.shape[1]/2-w/2
            r0 = int(params['rmin']*w+rd)
            r1 = int(params['rmax']*w+rd)
            c0 = int(params['cmin']*w+cd)
            c1 = int(params['cmax']*w+cd)
            
            if (r1-r0)%2 == 1: r1 += 1
            if (c1-c0)%2 == 1: c1 += 1
        
            # slice the data
            d  = self.rs_data[r0:r1,c0:c1]
            
        if project == 'cdi':
            r, c = self.reconstructions[params['round']].shape
            
            r0 = int(params['rmin']*r)
            r1 = int(params['rmax']*r)
            c0 = int(params['cmin']*c)
            c1 = int(params['cmax']*c)
        
            if (r1-r0)%2 == 1: r1 += 1
            if (c1-c0)%2 == 1: c1 += 1
            if r1 > r: r1 += -1
            if c1 > c: c1 += -1
        
            d = self.reconstructions[params['round']][r0:r1,c0:c1]
            #speckle.io.save('sliced for propagation.fits',d)

        # set parameters correctly
        sr = max([r1-r0,c1-c0])
        p  = params['pitch']*1e-9
        e  = params['energy']
        z1 = params['zmin']
        z2 = params['zmax']
        
        # make the range of distances. z2 > z1, always. libjpeg requires that
        # the maximum size of an image be (2**16, 2**16), so if the number of
        # propagations exceeds the maximum allowed clip the range of values.
        if z1 > z2: z1, z2 = z2, z1
        
        r = math.floor(2**16/int(self.resize))
        if z2-z1+1 > r**2:
            diff = r**2-(z2-z1)
            if diff%2 == 1: diff += 1
            z2 -= diff/2
            z1 += diff/2
        
        z = numpy.arange(z1,z2+1)
        

        # embed the data in a sea of zeros. if requested, attempt to apodize.
        # apodization typically requires an explicit estimate of the support.
        # here, i try to estimate the support from the sliced data.
        mask, d2 = speckle.propagate.apodize(d,threshold=0.01,sigma=3)
        if params['apodize']: d = d2

        data = numpy.zeros((1024,1024),numpy.complex64)
        data[512-d.shape[0]/2:512+d.shape[0]/2,512-d.shape[1]/2:512+d.shape[1]/2] = d
        
        # propagate
        pd = speckle.propagate.propagate_distances
        if self.use_gpu:
            propagated, images = pd(data,z*1e-6,e,p,subregion=sr,gpu_info=self.gpu,im_convert=True,silent=False)
        else:
            propagated, images = pd(data,z*1e-6,e,p,subregion=sr,im_convert=True,silent=False)
        self.propagated = propagated
        
        # convert to PIL objects which are an intermediate representation of the
        # data which can be quickly saved to jpeg etc
        t0 = time.time()
        rgb, images = [], []
        for n, frame in enumerate(propagated): rgb.append(speckle.io.complex_hsv_image(frame/numpy.abs(frame).max()))
        for array in rgb: images.append(smp.toimage(array))
        t1 = time.time()
        
        # save the images as a single large image which only requires a single GET
        # request to the webserver. g is the dimensions of the grid in terms of
        # number of images.
        imgx, imgy = images[0].size
        g = int(math.floor(math.sqrt(len(z)))+1)
        big_image = Image.new('RGB',(g*imgx,g*imgy))
        for n, img in enumerate(images):
            r, c = n/g, n%g
            big_image.paste(img,(imgx*c,imgy*r))
        big_image = big_image.resize((g*self.resize,g*self.resize),Image.BILINEAR)
        big_image.save('./static/imaging/images/bp_%s_%s.jpg'%(self.data_id,self.bp_id))

        #for zn, img in zip(z,images):
        #    img.save('./static/imaging/images/bp_%s_%s_%s.png'%(self.data_id,self.bp_id,int(zn)))
            
        # debug: save the fits
        #speckle.io.save('propagated_%s_%s.fits'%(self.data_id,self.bp_id),self.propagated)
    
        # calculate the acutance, then save it to a csv
        self.acutance  = numpy.array(speckle.propagate.acutance(self.propagated,mask=mask))
        self.acutance /= self.acutance.max()
        _save_acutance(z)
        
    def reconstruct(self,params):
        
        def _load_new_support():
            tmp = self.fourier_data
            
            # make modulus if necessary (specified in front end!)
            if not ismodulus: tmp = numpy.sqrt(tmp)

            # resize for speed. need to be the next power of 2 larger
            # than twice the maximum dimension of the support.
            import math
            bb = speckle.masking.bounding_box(self.support,force_to_square=True)
            rs = bb[1]-bb[0]
            cs = bb[3]-bb[2]
            g  = int(2**(math.floor(math.log(2*max([rs,cs]))/math.log(2))+1))
            
            self.reconstruct_shape = (g,g)
            tmp     = numpy.fft.fftshift(tmp)
            resized = speckle.wrapping.resize(tmp,(g,g))
            tmp     = numpy.fft.fftshift(tmp)
            self.modulus = tmp
            
            # load the data
            self.machine.load(modulus=tmp)
            self.loaded = True
            
            # slice the support and load it
            support = numpy.zeros_like(self.modulus)
            support[0:rs,0:cs] = self.support[bb[0]:bb[1],bb[2]:bb[3]]
            self.machine.load(support=self.support)
            
        def _refine_support(r_average):
            ref = speckle.phasing.refine_support
            r0, r1 = self.machine.r0, self.machine.r0+self.machine.rows
            c0, c1 = self.machine.c0, self.machine.c0+self.machine.cols
            refined = ref(self.support[r0:r1,c0:c1], r_average,                             
                                blur=sw_sigma,                      
                                local_threshold=sw_cutoff,
                                global_threshold=0.01)[0]
            self.support[:,:] = 0
            self.support[:self.machine.rows,:self.machine.cols] = refined
            self.machine.load(support=self.support)
            
        def _save_images(ave1):
            
            ave2 = numpy.sqrt(numpy.abs(ave1))*numpy.exp(complex(0,1)*numpy.angle(ave1))
            hsv  = speckle.io.complex_hsv_image
            
            for entry in [(ave1,'linr'),(ave2,'sqrt')]:
                data, scale = entry
                g,h = data.shape
                l   = max([g,h])
                new_d = numpy.zeros((l,l),numpy.complex64)
                new_d[:g,:h] = data
                img = smp.toimage(hsv(new_d))
                img.resize((300,300),Image.ANTIALIAS)
                
                img.save("static/imaging/images/r_%s_%s_%s.png"%(self.data_id,self.r_id,scale))
        
        # this function broadly duplicates the functionality of 
        # advanced_phasing_example. however, it does not automatically
        # loop over the refinement rounds, but instead runs a single
        # round at each invocation. this allows an image of the reconstruction
        # to be reloaded into the frontend.
        
        # passed params: iterations, numtrials, ismodulus, sigma, threshold
        ismodulus  = params['ismodulus']
        numtrials  = params['numtrials']
        iterations = params['iterations']
        sw_sigma   = params['sw_sigma']
        sw_cutoff  = params['sw_cutoff']

        # tell the machine how many trials
        self.machine.load(numtrials=numtrials)
        
        # during the first round, self.loaded = False; at this time
        # we do final processing of the modulus, including resizing.
        # otherwise we just refine the support.
        if not self.loaded: _load_new_support()
        else: _refine_support(self.most_recent)
  
        # for the given number of trials, make a new random estimate
        # and iterate the desired number of times. default scheme
        # is 99HIO + 1ER
        for trial in range(numtrials):
            self.machine.seed()
            self.machine.iterate(iterations,silent=100)
            
        # once the trials are all finished, process the results into an
        # average by aligning the phase and averaging along the frame axis
        savebuffer = self.machine.get(self.machine.savebuffer)
        savebuffer = speckle.phasing.align_global_phase(savebuffer)
        r_average  = numpy.mean(savebuffer[:-1],axis=0)
        r_sum      = numpy.sum(savebuffer[:-1],axis=0)
        savebuffer[-1] = r_average
        
        # now save the average formed above. this will get displayed in the
        # front end. must be embedded, then resized to 300x300 (subject to change?)
        self.r_id = int(time.time())
        _save_images(r_average)
        self.reconstructions[str(self.r_id)] = r_average
        self.most_recent = r_average
        
        # calculate the rftf
        rftf, self.rftfq = speckle.phasing.rftf(r_average,self.modulus,rftfq=True,scale=True,hot_pixels=True)
