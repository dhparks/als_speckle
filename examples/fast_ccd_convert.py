import speckle, numpy

file_in  = ''
file_out =''
data = speckle.io.open_photon_counting_fits(file).astype(numpy.float32)
speckle.io.save(file_out,data)