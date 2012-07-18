#!/usr/bin/env python
""" beamline_parameters.py: get and print the beamline parameters from a FITS
file. The beamline parameters must be identical to the "BCS Log Files/"
directory that is located at the beamline.  This also uses Enthought's traits
and traitsui library (http://code.enthought.com/projects/traits/).
"""

import os

from traits.api import *
from traitsui.api import View, Item

import speckle

class FitsParameters(HasTraits):
    fitsfile = File("", filter=['*.fits'], desc="FITS file to lookup", label="FITS file")
    bcs_dir = Directory("%s/Work/Data/Beamline Controls/BCS Log Files/" % os.path.expanduser('~'), desc="Beamline Control System logfiles", label="BCS directory")
    fitsinfo = String("", label="BCS Information")
    get_info = Button("Load log data")
    
    view = View(Item('fitsfile'),
                Item('bcs_dir'),
                Item('fitsinfo', width=400, height=400, style='custom'),
                Item('get_info', show_label=False, springy=True, style='custom'),
                title="FITS Information"
                )

    def _get_info_fired(self):
        self.params = speckle.io.get_beamline_parameters(self.fitsfile, self.bcs_dir)
        for k in sorted(self.params.keys()):
            self.fitsinfo += '%s : %s\n' % (k.encode('ascii', errors='ignore'), self.params[k].encode('ascii', errors='ignore'))
        self.fitsinfo = self.fitsinfo[0:-1] # remove last "\n"

if __name__ == '__main__':
    FitsParameters().configure_traits()

