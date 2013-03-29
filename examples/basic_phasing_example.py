### this file demonstrates how to use the phasing library in a basic
### reconstruction. modulus and support data are opened and loaded,
### then some reconstructions take place and output is saved.

# import the common libraries
import numpy
import sys
sys.path.insert(0,'..')
import speckle

# import the parameters for the reconstruction
import phasing_parameters as params

def reconstruct():
    # canonical code to demonstrate library functions

    # open and load the modulus data
    data = speckle.io.open(params.dataname).astype(numpy.float32)
    r.load(modulus=data)
    
    # open and load the support data
    support = speckle.io.open(params.supportname).astype('float')
    support *= 1./support.max()
    r.load(support=support)

    # for the specified number of trials, make a new random estimate
    # and iterate it the specified number of iterations.
    # iteration scheme is 99 HIO, 1 ER
    for trial in range(params.trials):
        print "trial ",trial
        r.seed()
        r.iterate(params.iterations)
    
    # now that all the independent trials are reconstructed,
    # pull data out of the class and in to local namespace.
    save_buffer = r.get(r.savebuffer)

    # roll the phase for global phase alignment. executes on cpu because
    # executing the optimization algorithms on the gpu is painful.
    print "rolling global phases"
    save_buffer = phasing.align_global_phase(save_buffer)
    
    # sum along the frame axis then save the output. save the components as fits for future work and a sqrt
    # image of the magnitude for inspection with a program like GIMP or photoshop
    save_buffer[-1] = numpy.sum(save_buffer,0)
    
    # save some output in the specified output directory
    speckle.io.save('%s/%s summed.fits'%(params.savepath,params.savename), save_buffer[-1], components='cartesian', overwrite=True)
    speckle.io.save('%s/%s summed sqrt.png'%(params.savepath,params.savename), numpy.sqrt(abs(save_buffer[-1])), components='complex_hls')

if __name__ == '__main__':
    # instantiate the class
    r = speckle.phasing.phasing()
    
    # run the reconstruction algorithm
    reconstruct()
