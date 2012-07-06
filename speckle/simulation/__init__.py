""" Simulations of coherent scattering events such as random walking balls
illuminated by a pinhole and generation of time-stamped single-photon events.

Author: Keoki Seu (kaseu@lbl.gov)
Author: Daniel Parks (dhparks@lbl.gov)

"""

__all__ = [
    "singlephoton",
    "random_walk",
    "gpu_domains",
    "cpu_domains",
]

for mod in __all__:
    exec("from %s import *" % mod)
del mod
