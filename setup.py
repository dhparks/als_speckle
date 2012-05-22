from distutils.core import setup

dependencies = ['scipy', 'numpy']

setup(
    name='speckle',
    version='0.1',
    packages=['speckle'],
    requires = dependencies,
    author = "Daniel Parks, Keoki Seu",
    author_email = "dhparks@lbl.gov, kaseu@lbl.gov",
    description = "Library for speckle analysis at Beamline 12.0.2 at the Advanced Light Source",
    url = "https://sites.google.com/a/lbl.gov/coherent-scattering-beamline/"
)