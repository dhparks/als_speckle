try:
    from setuptools.core import setup
except ImportError:
    from distutils.core import setup

setup(
    name='speckle',
    version='0.3',
    packages=['speckle'],
    requires = ['scipy', 'numpy'],
    package_data = { 'speckle': ['gpu/kernels/*.cl']},
    author = "Daniel Parks, Keoki Seu",
    author_email = "dhparks@lbl.gov, kaseu@lbl.gov",
    description = "Library for speckle analysis at Beamline 12.0.2 at the Advanced Light Source",
    url = "https://sites.google.com/a/lbl.gov/coherent-scattering-beamline/"
)
