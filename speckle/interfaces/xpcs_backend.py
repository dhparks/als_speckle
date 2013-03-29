# core
import numpy as np
import time

# common libs
import speckle

# matplotlib
import matplotlib.pyplot as plt

# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg # turn this on to embed a matplotlib Figure object in a Tk canvas instance
int_types = (int,np.int8,np.int16,np.int32,np.int64)

class xpcs():
    """ Class for xpcs methods, intended for use with graphical interface.
    However, can also be used for commandline or script operations. """
    
    def __init__(self):

        self.regions = {}
        self.plottables = {}
        self.figure = plt.figure()
        self.g2plot = self.figure.add_subplot(111)

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
        self.data_shape = speckle.io.get_fits_dimensions(path)
        
        if len(self.data_shape) !=  3:
            pass
            # actually, Raise an Error
            
        if len(self.data_shape) == 3 and self.data_shape[0] == 1:
            pass
            # raise the same error
            
        # now open the first frame
        self.first_frame = speckle.io.openframe(path,frame=0)
        self.frames = self.data_shape[0]
        
    def new_region(self,coords):
        """ Slice the data as selected in the GUI. Then compute the g2 function
        for the region. Normalization is standard.

        coords is a 4-tuple of (rmin, rmax, cmin, cmax).
        """
        
        ########## Get the data
        
        assert isinstance(coords,tuple)
        assert len(coords) == 4
        for n in coords: assert isinstance(n,int_types)
        
        rmin, rmax, cmin, cmax = coords
        rows = rmax-rmin
        cols = cmax-cmin
        
        # region dict holds all the information and analysis for the selected
        # region. region_dict is accessed through dictionary self.regions
        this_region = region()
        
        # loop through the number of frames, opening only the relevant portion.
        data = np.zeros((self.frames,rows,cols),np.float32)
        for f in range(self.frames):
            data[f] = speckle.io.openframe(self.data_path,frame=f)[rmin:rmax,cmin:cmax]
        this_region.data = data
        print "opened"
    
        ########## Calculate g2
        qave    = self._qave(data)                  # calculate the q-average for the current region data
        #tave    = self._tave(data)                 # calculate the t-average for the current region data
        #qtave   = self._tave(qave)                 # calculate the q-t-average for the current region data
        #numer   = self._fft_numerator(qave)        # calculate the numerator of all the g2 functions with fft
        g2      = speckle.xpcs.g2_plain_norm(qave,self.frames).ravel()  # calculate the g2 function. THIS FUNCTION CAN CHANGE
        print "g2"
        
        this_region.g2 = g2
        
        ########## Fit g2 to a known form
        to_fit    = np.zeros((2,self.frames),np.float32)
        to_fit[0] = np.arange(self.frames).astype(np.float32)
        to_fit[1] = g2
        exp_fit   = speckle.fit.decay_exp_beta(to_fit.transpose())
        print "fit"
        
        this_region.fit_vals   = exp_fit.final_evaluated
        this_region.fit_params = exp_fit.final_params
            
        # done with analysis, add region to master dictionary
        region_key  = '%s,%s,%s,%s'%coords
        this_region.region_key = region_key
        self.regions[region_key] = this_region
        print "done"

        print self.regions[region_key].data.shape
        
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
            data = np.mean(data,axis=-1)
        return data
    
    def _tave(self,data):
        return np.mean(data,axis=0)

    def clear_data(self,regions=None):
        """ Forcibly clear the specified regions from memory.
        
        Note to self for gpu:
            region_dict[key].data.release()
            del region_dict[data_key]
            
        if the key remains in the dictionary, attempts to access the data
        will cause a segmentation fault.
            
        """
        
        if regions == None: regions = self.regions.keys()
        
        try:
            for region in keys():
                
                region_dict = self.regions[region]
                data_keys = region_dict.keys()
                for data_key in data_keys:
                    #
                    del region_dict[data_key]
                
                del self.regions[region]

        except KeyError:
            print "region not in dictionary"
            
    def add_to_plot(self,region,what):
        
        assert what in ('data','fit')
        assert region in self.regions.keys()
        
        this_region = self.regions[region]
        
        x = np.arange(self.frames)
        if what == 'data':
            self.g2plot.loglog(x,this_region.g2)              # plot data
            this_region.g2_plot_n = len(self.g2plot.lines)-1  # record number of plot
        if what == 'fit':
            self.g2plot.loglog(x,this_region.fit_vals)
            this_region.fit_plot_n = len(self.g2plot.lines)-1
 
    def remove_from_plot(self,region,what):
        
        assert region in self.regions.keys()
        assert what in ('data','fit')
        
        this_region = self.regions[region]
        
        # get the location of the plot in the self.g2plot.lines list
        if what == 'data': n = this_region.g2_plot_n
        if what == 'fit':  n = this_region.fit_plot_n
        
        # remove the plt and update the listing
        del self.g2plot.lines[n]
        n = None
        
        # now we need to find those plots with higher number, and decrement them
        for key in self.regions.keys():
            a_region = self.regions[key]
            if a_region.g2_plot_n  > n: a_region.g2_plot_n  += -1
            if a_region.fit_plot_n > n: a_region.fit_plot_n += -1
        
class region():
    """ empty class for holding various attributes of a selected region.
    Just basically easier than using a dictionary.
    
    1. data
    2. g2 data
    3. other stuff
    
    """
    
    def __init__(self):
    
        self.data     = None
        self.g2       = None
        self.fit_vals = None
        self.fit_params = None
        self.g2_plot_n = None
        self.fit_plot_n = None
    
        
        
        
        
        
        
        
        
        
    

