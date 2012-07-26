try:
    from setuptools.core import setup
except ImportError:
    from distutils.core import setup
    
dependencies = ['scipy', 'numpy']

# run glob on the gpu/kernels directory to get a list
# of all the .cl files to copy
import glob
kernel_files = glob.glob('./speckle/gpu/kernels/*.cl')

setup(
    name='speckle',
    version='0.3',
    packages=['speckle','speckle.gpu','speckle.simulation'],
    requires = dependencies,
    data_files = ['speckle/gpu/kernels', kernel_files, 'test', 'examples'],
    author = "Daniel Parks, Keoki Seu",
    author_email = "dhparks@lbl.gov, kaseu@lbl.gov",
    description = "Library for speckle analysis at Beamline 12.0.2 at the Advanced Light Source",
    url = "https://sites.google.com/a/lbl.gov/coherent-scattering-beamline/"
)
