# core
import numpy as np
import time
import itertools
import os

# common libs
from .. import io, xpcs, fit, wrapping

# matplotlib
import matplotlib.pyplot as plt

# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg # turn this on to embed a matplotlib Figure object in a Tk canvas instance
int_types = (int,np.int8,np.int16,np.int32,np.int64)

class xpcs_backend():
    """ Class for xpcs methods, intended for use with graphical interface.
    However, can also be used for commandline or script operations. """
    
    def __init__(self):

        self.regions = {}
        self.plottables = {}
        self.figure = plt.figure()
        self.g2plot = self.figure.add_subplot(111)
        self.plot_id = 0

    def open_data(self,path):
        
        """ Get all the relevant information about the file at path. Because
        XPCS datasets can be very large (1GB+) and only a portion of the data
        will be analyzed at a time, limit the initial data to the following:
        
        1. The first frame, which will be presented as an image for region
        selection
        
        2. The shape
        
        """

        self.data_path = path

        # first, check the shape
        self.data_shape = io.get_fits_dimensions(path)
        
        if len(self.data_shape) !=  3:
            pass
            # actually, Raise an Error
            
        if len(self.data_shape) == 3 and self.data_shape[0] == 1:
            pass
            # raise the same error
            
        # now open the first frame
        self.first_frame = wrapping.resize(io.openframe(path,frame=0),(512,512))
        self.frames = self.data_shape[0]
        
        # remove everything in the static images folder
        files = os.listdir('static/images')
        for f in files: os.remove(os.path.join('static/images',f))
        
        # resize, then save the color and scale permutations of first_frame.
        linr = self.first_frame
        sqrt = np.sqrt(self.first_frame)
        log  = np.log(self.first_frame/self.first_frame.min())
        
        for color in ('L','B','A'):
            io.save('static/images/data_linear_%s.jpg'%(color),linr, color_map=color, append_component=False, overwrite=True)
            io.save('static/images/data_sqrt_%s.jpg'%(color),  sqrt, color_map=color, append_component=False, overwrite=True)
            io.save('static/images/data_log_%s.jpg'%(color),   log,  color_map=color, append_component=False, overwrite=True)

    def add_region(self,uid,coords,color):
        
        # instantiate a new instance of the region class
        this_region = region()
        this_region.unique_id = uid
        
        # populate the new region with incoming spec
        this_region.rmin = coords[0]
        this_region.rmax = coords[1]
        this_region.cmin = coords[2]
        this_region.cmax = coords[3]
        
        this_region.color = color
        
        # when .changed is True, prior to plotting we have to recalculate:
        # intensity, g2, fit
        this_region.changed = True
        
        self.regions[uid] = this_region
        
    def update_region(self,uid,coords):
        
        self.regions[uid].rmin = coords[0]
        self.regions[uid].rmax = coords[1]
        self.regions[uid].cmin = coords[2]
        self.regions[uid].cmax = coords[3]
        self.regions[uid].changed = True
        
    def recalculate(self):
        
        # iterate through the regions. when one is encountered with
        # .changed == True, recalculate the intensity, g2, and fit to g2.
        
        for region_key in self.regions.keys():
            
            this_region = self.regions[region_key]
            
            if this_region.changed:
                
                # get the data
                data  = io.open(self.data_path)[:,this_region.rmin:this_region.rmax,this_region.cmin:this_region.cmax].astype(np.float32) 
                
                # calculate g2(tau) for each pixel, then average over pixels
                g2all = xpcs.g2(data)
                g2    = self._qave(g2all)
            
                # run the fit
                to_fit    = np.array([np.arange(len(g2)),g2])
                exp_fit   = fit.decay_exp_beta(to_fit.transpose())
                
                # add the data to the region
                this_region.g2         = g2
                this_region.intensity  = self._qave(data)
                this_region.fit_vals   = exp_fit.final_evaluated
                this_region.fit_params = exp_fit.final_params
                
                this_region.changed = False
        
    def dump(self):
        # save a csv of the current g2 calculations
        
        keys = ['tau',]+self.regions.keys()
        print keys
        
        g2_array    = np.zeros((1+len(self.regions),self.frames/2),np.float32)
        g2_array[0] = np.arange(self.frames/2)
        for n,key in enumerate(keys[1:]):
            g2_array[n+1] = self.regions[key].g2
        g2_array = g2_array.transpose()

        self.csv_path = 'static/csv/g2_%s.csv'%(int(time.time()))
        
        outf = open(self.csv_path,'w')
        outf.write(','.join(keys)+"\n")
        for row in g2_array:
            out_row = ",".join("{0}".format(n) for n in row)
            outf.write(out_row+"\n")
        outf.close()
        
    def plot(self):
        
        # declare the plot
        f = plt.figure()
        g = f.add_subplot(111)
        
        print len(self.regions.keys())
        
        # plot g2 and fits
        for region_key in self.regions.keys():
            
            this_region = self.regions[region_key]
            x = np.arange(len(this_region.g2))
            g.plot(x,this_region.g2,'-',color=this_region.color)
            g.plot(x,this_region.fit_vals,'--',color=this_region.color)
            
        self.g2_path = 'static/images/g2plot_%s.png'%(int(time.time()))
        f.savefig(self.g2_path)
        plt.clf()

    def _fft_numerator(self,data):
        # embed the data in an array of twice the length
        # calculate the autocorrelation along the frame axis.
        # normalize by the number of frames which contributed to each term.
        
        f = np.fft.fftn
        g = np.fft.ifftn
        
        print data.shape
        
        if data.ndim == 3:
            n,r,c = data.shape
            t     = np.zeros((2*n,r,c),data.dtype)
            t[0:n] = data
            x     = abs(g(abs(f(t,axes=(0,)))**2,axes=(0,)))[0:n]
            
        if data.ndim == 1:
            n     = data.shape[0]
            t     = np.zeros((2*n,),data.dtype)
            t[0:n] = data
            x     = abs(g(abs(f(t))**2))[0:n]
        
        for m in range(n):
            x[m] *= 1./(n-m)
            
        return x
        
    def _qave(self,data):
        assert data.ndim > 1
        for n in range(data.ndim-1):
            data = np.average(data,axis=-1)
        return data
    
    def _tave(self,data):
        return np.mean(data,axis=0)

    def new_background(self,params):
        # resave self.first_frame as a new image for interface presentation
        
        scaling, color, path = params
        
        # rescale
        try:
            power   = float(scaling)
            to_save = self.first_frame**power
        except:
            pass
        if scaling == 'sqrt':   to_save = np.sqrt(self.first_frame)
        if scaling == 'log':    to_save = np.log(self.first_frame)
        if scaling == 'linear': to_save = self.first_frame
        
        # save
        io.save(path,to_save,color_map=color,append_component=False,overwrite=True)
        
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
    
        
        
        
        
        
        
        
        
        
    

