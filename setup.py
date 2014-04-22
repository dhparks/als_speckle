from distutils.core import setup

setup(
    name='speckle',
    version='14.03.26',
    packages=[ 'speckle', 'speckle.simulation'],
    requires = ['scipy', 'numpy'],
    package_data = { 'speckle': ['kernels/*.cl']},
    author = "Daniel Parks, Keoki Seu",
    author_email = "dhparks@lbl.gov, kaseu@lbl.gov",
    description = "Library for speckle analysis at Beamline 12.0.2 at the Advanced Light Source",
    url = "https://sites.google.com/a/lbl.gov/coherent-scattering-beamline/"
)
