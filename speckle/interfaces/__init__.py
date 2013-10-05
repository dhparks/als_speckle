""" Interface backends for GUI frontends intended to facilitate on-line analysis
of coherent scattering experiments at beamline 12.0.2

The beamline webpages are here:
http://ssg.als.lbl.gov/ssgbeamlines/beamline12-0-2
https://sites.google.com/a/lbl.gov/coherent-scattering-beamline/

Author: Daniel Parks (dhparks@lbl.gov)

"""
__version_info__ = ('2013', '10', '4')
__version__ = '.'.join(__version_info__)

# if you make a new file/module name, put it here.  These are alphabetized.
__all__ = [
    "imaging_backend",
    "xpcs_backend"
]

for mod in __all__:
    exec("import %s" % mod)
del mod
