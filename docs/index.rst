.. Speckle documentation master file, created by
   sphinx-quickstart on Fri Jul  6 14:20:38 2012.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Speckle's documentation!
===================================

Speckle is a module for analyzing data from the coherent scattering beamline
`12.0.2 <https://sites.google.com/a/lbl.gov/coherent-scattering-beamline/>`_
at the `Advanced Light Source <http://www-als.lbl.gov>`_.  The module requires
`numpy <http://numpy.org>`_ and `scipy <http://scipy.org>`_ modules, and it has
GPU support when `PyOpenCL <http://mathema.tician.de/software/pyopencl/>`_ and
`PyFFT <http://packages.python.org/pyfft/>`_ are installed.


Features of the library:
    * Analyze X-ray photon correlation spectroscopy (XPCS) datasets
    * Fit 1d (x,y), 2d and 3d (image) datasets to many functions (decayed exponentials with beta parameter, Lorentzians, Gaussians)
    * Phase retrieval and data conditioning with GPU support
    * Unwrapping and wrapping centrosymmetric data
    * Rotational symmetry analysis
    * Q-dependent (rotational) and spatial memory analysis
    * Simulations of single-photon XPCS data, domain generation, and random walks
    * Endstation-specific items such as Q-calculations and conversion from detected events to photons


Module contents:

.. toctree::
   :maxdepth: 4

   speckle


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

