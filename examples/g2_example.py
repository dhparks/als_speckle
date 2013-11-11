import speckle, time, numpy
import matplotlib.pyplot as plt

### this example demonstrates how to do some g2 calculations on a 3d dataset.
### to adapt this example for your own analysis, you will at a minimum need
### to change the path to the data file, which in this example doesn't actually
### point to anything. additionally, it is likely that your analysis needs will
### differ from the analysis presented in this example.

# 1. open the data
datafile  = 'NAME OF YOUR DATA HERE'
data      = speckle.io.open(datafile)
(t, r, c) = data.shape
print "data has shape %s"%((data.shape),)

# 2. run the g2 correlation. this is much faster if the necessary libraries to
# use gpu acceleration have been installed. if no gpu is present, the analysis
# will be run on the cpu. THE G2 FUNCTION HAS MANY OPTIONS SO READ ITS
# DOCUMENTATION TO LEARN ABOUT MORE FUNCTIONALITY.
try: gpu_info = speckle.gpu.init()
except: gpu_info = None

# calculate g2 at ALL x,y coordinates
t0 = time.time()
all_g2  = speckle.xpcs.g2(data,gpu_info=gpu_info)
print "calculated g2 on all pixels; time: %.3f seconds"%(time.time()-t0)

# calculate g2 at only SOME x,y coordinates by slicing data
t0 = time.time()
some_g2 = speckle.xpcs.g2(data[:,r/2-r/8:r/2+r/3,c/2-c/8:c/2+c/8],gpu_info=gpu_info)
print "calculated g2 on some pixels; time: %.3f seconds"%(time.time()-t0)

# calculate with the same coordinates, but use a different normalization
# scheme (the so-called "standard" normalization)
t0 = time.time()
other_g2 = speckle.xpcs.g2(data[:,r/2-r/8:r/2+r/3,c/2-c/8:c/2+c/8],gpu_info=gpu_info,norm="standard",qAvg=("circle",5))
print "calculated g2 with different norm; time: %.3f seconds"%(time.time()-t0)

# 3. Now we probably want to reduce the dimensionality of the data a little bit,
# so we will unwrap all_g2 into polar coordinates, then average along the azimuthal
# coordinate. this will give us an average g2 as a function of radius.
plan         = speckle.wrapping.unwrap_plan(5, min([r,c])/2-1, (r/2,c/2), columns=360)
unwrapped_g2 = speckle.wrapping.unwrap(all_g2,plan)
azimuthal_g2 = numpy.average(unwrapped_g2,axis=-1).transpose()
print "shape of unwrapped g2 is %s"%(unwrapped_g2.shape,)
print "shape of azimuthal g2 is %s"%(azimuthal_g2.shape,)

# 4. save the azimuthal g2
speckle.io.save('azimuthal_g2.fits',azimuthal_g2)

# 5. an example of how to fit some of the data to a functional form, in
# this case decay_exp_beta. Then we plot it. This loop fits and plots 3 slices
# of azimuthal_g2
r = azimuthal_g2.shape[0]
x = numpy.arange(azimuthal_g2.shape[1])
for row in [0, r/2, -1]:
    y      = azimuthal_g2[row]
    data   = [x, y]
    fitted = speckle.fit.decay_exp_beta(data)
    plt.plot(x,y,'o',markersize=3)
    plt.plot(x,fitted.final_evaluated,'-',linewidth=2)
plt.savefig('g2 plots.png')
plt.clf()





