""" A speckle correnation library for Beamline 12.0.2 at the Advanced Light
Source at Lawrance Berkeley National Laboratory. This library contains
functions that allow the user to analyze data from experiments conducted at
the ALS. This includes magnetic memory, rotational symmetries, imaging, near
field propagation, and x-ray photon correlation spectroscopy.

The beamline webpages are here:
http://ssg.als.lbl.gov/ssgbeamlines/beamline12-0-2
https://sites.google.com/a/lbl.gov/coherent-scattering-beamline/

Author: Keoki Seu (kaseu@lbl.gov)
Author: Daniel Parks (dhparks@lbl.gov)

"""
__version_info__ = ('0', '2', '0')
__version__ = '.'.join(__version_info__)

# if you make a new file/module name, put it here.  These are alphabetized.
__all__ = [
    "averaging",
    "conditioning",
    "crosscorr",
    "fit",
    "gpu",
    "io",
    "masking",
    "phasing",
    "propagate",
    "scattering",
    "shape",
    "symmetries",
    "wrapping",
    "xpcs",
]

for mod in __all__:
    exec("import %s" % mod)
del mod
