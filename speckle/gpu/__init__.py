""" A GPU-accelerated submodule to the speckle correlation library for
beamline 12.0.2 at the Advanced Light Source. Provided here are functions
which benefit from GPU acceleration, in particular phasing algorithms
and simulations heavily reliant on discrete fourier transforms. GPU kernels
written in OpenCL/C99 are found in speckle/gpu/kernels.

Author: Daniel Parks (dhparks@lbl.gov)

"""

have_pyopencl = True
have_pyfft = True
try:
    import pyopencl
except ImportError:
    have_pyopencl = False

try:
    import pyfft
except ImportError:
    have_pyfft = False
    
__all__ = [
    "gpu",
    "gpu_propagate",
    "gpu_phasing",
    "gpu_correlations",
    "speckle_statistics"
]

if have_pyopencl and have_pyfft:
    for mod in __all__:
        exec("import %s" % mod)
    del mod
    print "gpu functions enabled"
else:
    print "gpu functions not enabled"
