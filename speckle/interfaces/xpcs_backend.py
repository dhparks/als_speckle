# core
import numpy as np
import time
import itertools
import os
import Image
import scipy

# common libs
from .. import io, xpcs, fit, wrapping

# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg # turn this on to embed a matplotlib Figure object in a Tk canvas instance
int_types = (int,np.int8,np.int16,np.int32,np.int64)

class backend():
    """ Class for xpcs methods, intended for use with graphical interface.
    However, can also be used for commandline or script operations. """
    
    def __init__(self,session_id,gpu_info=None):

        self.regions = {}
        self.form ='decayexp'
        self.session_id = session_id
        self.set_gpu(gpu_info)

    def set_gpu(self,gpu_info):
        self.gpu = gpu_info
        if gpu_info != None: self.use_gpu = True
        if gpu_info == None: self.use_gpu = False

    def load_data(self,project,folder):
        
        """ Get all the relevant information about the file at path. Because
        XPCS datasets can be very large (1GB+) and only a portion of the data
        will be analyzed at a time, limit the initial data to the following:
        
        1. The first frame, which will be presented as an image for region
        selection
        
        2. The shape
        
        3. Rescale and offsets for coordinate transformations
        
        """
        
        def _save_scales():
            # rescale the first frame into sqrt and log
            scales = {}
            
            tmp  = self.first_frame-self.first_frame.min()
            tmp *= 10000/tmp.max()
            tmp += 1
            
            scales['linear'] = tmp
            scales['sqrt']   = np.sqrt(tmp)
            scales['log']    = np.log(tmp)
            
            # for each scale option, create 3 color options. save as datasprites,
            # which is what the webbrowser will load.
            big_image = Image.new('RGB',(3*512,3*512))
            for n, key in enumerate(['linear','sqrt','log']):
                imgl = scipy.misc.toimage(scales[key])
                for m, colormap in enumerate(['L','A','B']):
                    imgc = imgl
                    if colormap != 'L':
                        imgc.putpalette(io.color_maps(colormap))
                        imgc = imgc.convert("RGB")
                    big_image.paste(imgc,(n*512,m*512))
            big_image.save("static/xpcs/images/datasprites_session%s_id%s.jpg"%(self.session_id,self.data_id))
            
        def _resize():
            # support non-square data for calculating a rescale factor and offsets
            # needed to transform gui coordinates into data coordinates
            self.rescale = 512./max(self.data_shape[1:])
            if self.rescale > 1: self.rescale = 1 # for now, only shrink arrays
            self.roffset = int((512-self.data_shape[1]*self.rescale)/2)
            self.coffset = int((512-self.data_shape[2]*self.rescale)/2)
                
            # now open the first frame. resize the maximum dimension of the first
            # frame to 512 pixels before saving.
            ff  = io.open(self.data_path)[0].astype(np.float32)
            ff -= ff.min()
            ff  = wrapping.resize(ff,(int(self.data_shape[1]*self.rescale),int(self.data_shape[2]*self.rescale)))
            self.first_frame = np.zeros((512,512),np.float32)
            self.first_frame[self.roffset:self.roffset+ff.shape[0],self.coffset:self.coffset+ff.shape[1]] = ff

        self.data_id = self._new_id()
        old_name     = '%s/%sdata_session%s.fits'%(folder,project,self.session_id)
        new_name     = '%s/%sdata_session%s_id%s.fits'%(folder,project,self.session_id,self.data_id)
        os.rename(old_name,new_name)

        self.data_path = new_name
        
        # first, check the shape and get the number of frames
        self.data_shape = io.get_fits_dimensions(new_name)
        self.frames = self.data_shape[0]
        
        if len(self.data_shape) !=  3 or (len(self.data_shape) == 3 and self.data_shape[0] == 1):
            raise TypeError("Data is 2 dimensional")
            
        # resize the data to fit within a 512x512 image. this also calculates
        # scale factors and offsets necessary for coordinate transformations between
        # the coordinates as seen in the browswer and defined here in the backend
        _resize()

        # make the intensity image(s)
        _save_scales()
        
        # reset the regions every time new data is loaded
        self.regions = {}

    def _transform_coords(self,coords):
        rmin = np.clip((coords[0]-self.roffset)/self.rescale,0,self.data_shape[1])
        rmax = np.clip((coords[1]-self.roffset)/self.rescale,0,self.data_shape[1])
        cmin = np.clip((coords[2]-self.coffset)/self.rescale,0,self.data_shape[2])
        cmax = np.clip((coords[3]-self.coffset)/self.rescale,0,self.data_shape[2])
        return rmin, rmax, cmin, cmax

    def update_region(self,uid,coords):
        
        # update the coordinates of a region. if there is no region associated
        # with uid, make it.
        rmin, rmax, cmin, cmax = self._transform_coords(coords)
        
        uid = str(uid)
        
        if uid in self.regions.keys():
            
            # update coordinates
            if (self.regions[uid].rmin != rmin):
                self.regions[uid].rmin = rmin
                self.regions[uid].changed = True
            
            if (self.regions[uid].rmax != rmax):
                self.regions[uid].rmax = rmax
                self.regions[uid].changed = True
                
            if (self.regions[uid].cmin != cmin):
                self.regions[uid].cmin = cmin
                self.regions[uid].changed = True
                
            if (self.regions[uid].cmax != cmax):
                self.regions[uid].cmax = cmax
                self.regions[uid].changed = True
        
        else:
            
            # create new region
            here = region()
            here.unique_id = uid
            
            here.rmin = rmin
            here.rmax = rmax
            here.cmin = cmin
            here.cmax = cmax
            
            here.changed = True
            
            self.regions[uid] = here
            self.newest = uid

    def calculate(self):
        
        # iterate through the regions. when one is encountered with
        # .changed == True, recalculate the intensity, g2, and fit to g2.
        
        recalculate, refit = [], []
        for region_key in self.regions.keys():
            if self.regions[region_key].changed:
                recalculate.append(region_key)
                refit.append(region_key)
            if self.refitg2:
                refit.append(region_key)
        refit = list(set(refit))    

        print "recalculating regions: %s"%recalculate
        
        for region_key in recalculate:

            here = self.regions[region_key]

            # if out-of-bounds (unusual), the g2 calculation will fail so
            # return data which indicates an anomalous selection
            if (here.rmin == here.rmax or here.cmin == here.cmax):
                g2 = np.ones((self.frames/2,),np.float32)
                i  = np.ones((self.frames,),np.float32)
            
            # for in-bounds data (usually the case!), calculate g2(tau) for
            # each pixel, then average over pixels. when finished, add to object
            else:
                data  = io.open(self.data_path)[:,here.rmin:here.rmax,here.cmin:here.cmax].astype(np.float32)
                g2all = np.nan_to_num(xpcs.g2(data,gpu_info=self.gpu))-1
                g2    = self._qave(g2all)
                i     = self._qave(data)
                
                # find the point where g2 falls below 1e-6
                cutoff, k = 0, 0
                while cutoff == 0 and k < len(g2)-1:
                    if g2[k] >= 1e-6 and g2[k+1] < 0: cutoff = k
                    k += 1
                    
                g2[k-1:len(g2)] = 0

            here.g2        = g2
            here.intensity = i
            here.changed   = False
            
        print "refitting regions: %s"%refit
        
        for region_key in refit:
            
            here = self.regions[region_key]

            c = (here.g2[0])/2
            to_fit = np.array([np.arange(len(here.g2)),here.g2])
            mask = np.exp(-(here.g2-c)**2/(.5))
            mask = np.where(here.g2 > 0, 1, 0)
            if self.form == 'decayexpbeta': fitted = fit.decay_exp_beta(to_fit.transpose(),mask=mask,weighted=False)
            if self.form == 'decayexp':     fitted = fit.decay_exp(to_fit.transpose(),mask=mask,weighted=False)
            if self.form == 'decayexpbetanooffset': fitted = fit.decay_exp_beta_no_offset(to_fit.transpose(),mask=mask,weighted=False)
            self.functional = fitted.functional
            self.fit_keys   = fitted.params_map
            
            # add the data to the region
            here.fit_vals   = fitted.final_evaluated
            here.fit_params = fitted.final_params
        
    def csv_output(self):
        
        # 3 files get save at each dump.
        # 1. tau, g2, fit_eval (all regions in this file)
        # 2. fit parameters (all regions in this file)
        # 3. a "core dump" with several sections
        #   i. region coordinates
        #  ii. tau, g2, fit_val for all regions
        # iii. fit parameters of all regions

        # open the three files used to save analysis
        self.file_id = self._new_id()
        analysisf    = open('static/xpcs/csv/analysis_session%s.csv'%self.session_id,'w')
        g2f          = open('static/xpcs/csv/g2_session%s_id%s.csv'%(self.session_id,self.file_id),'w')
        fitsf        = open('static/xpcs/csv/fit_session%s_id%s.csv'%(self.session_id,self.file_id),'w')
    
        # form the header rows for each file
        srk   = self.regions.keys()
        sfk   = self.fit_keys.keys()
        gkeys = ['tau',]+['%s'%k for k in srk]+['%s_fit'%k for k in srk]
        fkeys = ['%s'%k for k in sfk]
        fvals = ['%s'%(self.fit_keys[k]) for k in sfk] # things like: a, b, beta, etc

        aheader = 'regionId,rmin,rmax,cmin,cmax\n'
        gheader = ','.join(gkeys)+'\n'
        fheader = 'regionId,'+','.join(fvals)+'\n'
        
        # form the g2 array, which holds all the g2 data resolved against region name
        g2_array = np.zeros((self.frames/2,1+2*len(self.regions)),np.float32)
        for n, key in enumerate(gkeys):
            if key == "tau":
                g2_array[:,n] = np.arange(self.frames/2)+1
            if "_fit" in key:
                g2_array[:,n] = self.regions[key.replace("_fit","")].fit_vals
            if key != "tau" and "_fit" not in key:
                g2_array[:,n] = self.regions[key].g2
          
        # remove the old analysis file
        if os.path.isfile('/static/xpcs/csv/analysis.csv'): os.remove('static/csv/analysis')
    
        # write region coordinates
        analysisf.write(aheader)
        for rk in self.regions.keys():
            region = self.regions[rk]
            line = "%s,%s,%s,%s,%s\n"%(region.unique_id,region.rmin,region.rmax,region.cmin,region.cmax)
            analysisf.write(line)
        analysisf.write("\n")
        
        # write g2 values to both the g2 file and the analysis file
        g2f.write(gheader)
        analysisf.write(gheader)
        for row in g2_array:
            out_row = ",".join("{0}".format(n) for n in row)+"\n"
            analysisf.write(out_row)
            g2f.write(out_row)
        g2f.close()
        analysisf.write("\n")
        
        # write fit parameters to both the fit file and the analysis file
        fitsf.write(fheader)
        analysisf.write(fheader)
        for key in self.regions.keys():
            r = self.regions[key]
            row = str(r.unique_id)+","+",".join("{0:.3f}".format(n) for n in r.fit_params)+"\n"
            analysisf.write(row)
            fitsf.write(row)
        
        fitsf.close()
        analysisf.close()
 
    def _qave(self,data):
        assert data.ndim > 1
        aves = np.zeros(data.shape[0],np.float32)
        for n, frame in enumerate(data): aves[n] = np.average(frame)
        return aves
    
    def _qstdev(self,data):
        assert data.ndim == 3
        stdevs = np.zeros(data.shape[0],np.float32)
        for n, frame in enumerate(data): stdevs[n] = np.std(frame)
    
    def _tave(self,data):
        return np.mean(data,axis=0)
    
    def _new_id(self):
        return str(int(time.time()*10))

    def check_data(self,data_name):
        
        # Check the incoming data for the attributes necessary for imaging
        # IMPORTANT: this assumes that the flask_server has already checked
        # the mimetype and extension of the data, and found that it is
        # in fact a FITS file. hopefully this limits the ability of a user
        # to upload executable/malicious data which could be executed by
        # opening the file.

        try: data_shape = io.get_fits_dimensions(data_name)
        except:
            error = "couldn't get fits dimensions; file may be invalid"
            return False, error

        # basically, the only requirement for xpcs data is that the data be
        # 3d, and not trivially so.
        if len(data_shape) < 3:
            return False, "Fits files for xpcs must be 3d; this is %sd"%len(data_shape)
        
        if data_shape[0] < 2 or data_shape[1] < 2 or data_shape[2] < 2:
            return False, "fits file is only trivially 3d with shape %s"((data_shape),)
        
        # success! data seems ok to basic checks
        return True, None
    
        
class region():
    """ empty class for holding various attributes of a selected region.
    Just basically easier than using a dictionary.
    
    1. data
    2. g2 data
    3. other stuff
    
    """
    
    def __init__(self):

        # identity
        self.unique_id = None
        self.color     = None
        self.changed   = None
        
        # coordinates
        self.rmin = None
        self.rmax = None
        self.cmin = None
        self.cmax = None
        
        # data
        self.intensity  = None
        self.g2         = None
        self.fit_params = None
        self.fit_vals   = None
    
        
        
        
        
        
        
        
        
        
    

