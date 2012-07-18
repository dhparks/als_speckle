#!/usr/bin/env python
""" An GUI example of binning the signal from the photon counting detector.
This GUI program allows yone to load a g_2 file (and correct overflows) and bin
it in space and time to create a movie.  This uses Enthought's traits and
traitsui library (http://code.enthought.com/projects/traits/).
"""

import os
import numpy as np
from threading import Thread

from traits.api import Button, Enum, File, Int, Float, Bool, HasTraits, Str
from traitsui.api import View, Item, Group, StatusItem

import speckle

def remove_corner_photons(data):
    """ Remove the photons from the corners.
    
    When the detector cannot find the start or stop pulse, it registers a count
    in one of the corners or edges.  A ratio of the number of corner photons to
    real photons should be < 5%.
    """
    assert isinstance(data, np.ndarray) and data.ndim in (2,3), "data must be 2d or 3d ndarray"
    if data.ndim == 3:
        data[:, :, -1] = 0
        data[:, -1, :] = 0
    elif data.ndim == 2:
        data[:, -1] = 0
        data[-1, :] = 0
    return data

class G2Fast(HasTraits):
    fitsfile = File("", filter=['*.fits'], desc="Fast FITS file to decode", label="Fast g2 file", exists=True)
    
    xybin = Int(8, label="x-y binning")
    stepinterval = Float(40.0e-9, label = "clock interval (s)")
    timebin = Float(1.0, label = "frame time (s)")
    removeCornerPhotons = Bool(True, label = "Remove bad photons")

#   fitsinfo = String("", label="BCS Information")
    process = Button("Process file")
    
    _CO = "Correct integer overflow"
    _MOV = "Create movie"
    _BOTH = "Correct overflow/Create movie"
    action = Enum( (_BOTH, _CO, _MOV), label = "action")
    
    outfile = File("", exists=False, auto_set=True, desc="Output File", label="Save As..")
    
    status = Str
    
    view = View(Item('fitsfile'),
                Group( Group( Item('xybin'), Item('removeCornerPhotons'), ),
                        Group(Item('timebin'), Item('stepinterval') ), orientation = 'horizontal' ),
                Item('action'),
                Item('outfile', width=400),
                Item('process', show_label=False, width=30, style='custom'),
                title="Fast FITS binning",
                statusbar = StatusItem( name = "status")
                )
    
    def _process_fired(self):
        thread = Thread (target = self.processData)
        thread.start()
    
    def genHeader(self):
        hdr = {}
        hdr["ORIG_X"] = 4096
        hdr["ORIG_Y"] = 4096
        hdr["HBIN"] = self.xybin
        hdr["VBIN"] = self.xybin
        hdr["EXPOSURE"] = self.timebin
        hdr["ACT"] = self.timebin
        hdr["KCT"] = self.timebin
        hdr["DATATYPE"] = 'Photons'
        hdr["ORIGFILE"] = self.fitsfile.encode('ascii')
        hdr["CLOCKINT"] = self.stepinterval
        hdr["NOTE"] = "Data has been processed into in an image"
        locdate = speckle.io.get_fits_acq_time(self.fitsfile)
        if locdate:
            hdr["DATETIME"] = locdate.strftime("%Y/%m/%d %H:%M:%S")
        return hdr
    
    def processData(self):
        self.status = "Opening %s" % self.fitsfile
        
        if self.action in (self._BOTH, self._CO):
            self.status = "Correcting integer overflows"
            self.data = speckle.io.open_photon_counting_fits(self.fitsfile, correct=True)
        else:
            self.data = speckle.io.open_photon_counting_fits(self.fitsfile)

        
        if self.action in (self._BOTH, self._MOV):
            self.status = "Binning data into %1.2f s frames" % self.timebin
            self.data = speckle.xpcs.sp_bin_by_space_and_time(self.data, self.timebin, xybin = self.xybin, counterTime = self.stepinterval)
        
        if self.removeCornerPhotons:
            self.status = "Removing corner photons"
            self.data = remove_corner_photons(self.data)
        
        if os.path.exists(self.outfile):
            print "path exists"
            self.outfile = self.outfile.replace(".fits", "-new.fits")
        
        self.status = "Writing %s" % self.outfile
        speckle.io.writefits(self.outfile, self.data, headerItems=self.genHeader())
        
        self.status = "Done."

if __name__ == '__main__':
    G2Fast().configure_traits()
