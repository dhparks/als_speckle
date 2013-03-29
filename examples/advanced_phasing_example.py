### this file demonstrates how to use the phasing library in a more sophisticated
### reconstruction. this example goes through several "rounds", at the end of
### which the support and an estimate of the coherence function get updated.
### note that in the first two updates only the support gets updated. in later
### updates, both the support and coherence are refined.

# import the common libraries
import numpy
import sys
sys.path.insert(0,'..')
import speckle

# import the parameters for the reconstruction
import advanced_phasing_parameters as params

def reconstruct():

    # open and load the modulus data
    data = speckle.io.open(params.dataname).astype(numpy.float32)
    r.load(modulus=data, numtrials=params.numtrials)

    # behavior regarding the support changes based on the round number
    for ur in params.update_rounds:
        
        if ur == 0:
            support = speckle.io.open(params.supportname).astype('float')
            support *= 1./support.max()
            
        if ur > 0:
            
            # here we refine the support using marchesini's method
            pbs = params.blur_sigma
            plt = params.local_threshold
            pgt = params.global_threshold
            s1 = support[r.r0:r.r0+r.rows,r.c0:r.c0+r.cols] # take the minimally bounded support
            
            # refine the support. embed it in an array of the correct size.
            support = phasing.refine_support(s1,save_buffer[-1],blur=pbs,local_threshold=plt,global_threshold=pgt)[0]
            new_support = numpy.zeros(data.shape,numpy.float32)
            new_support[0:support.shape[0],0:support.shape[1]] = support
            support = new_support
            
        # load the support into the class
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
        save_buffer = phasing.align_global_phase(save_buffer)
        
        # sum along the frame axis then save the output. save the components as fits for future work and a sqrt
        # image of the magnitude for inspection with a program like GIMP or photoshop
        save_buffer[-1] = numpy.sum(save_buffer,0)
        
        # in later rounds, optimize the coherence function (assumes gaussian form)
        # notice that we are passing the average reconstruction from host memory
        # back into the class, which is a little ham-fisted.
        if update_round > 2: r.optimize_gaussian_coherence(save_buffer[-1])
    
        # save some output in the specified output directory
        speckle.io.save('%s/%s summed.fits'%(params.savepath,params.savename), save_buffer[-1], components='cartesian', overwrite=True)
        speckle.io.save('%s/%s summed sqrt.png'%(params.savepath,params.savename), numpy.sqrt(abs(save_buffer[-1])), components='complex_hls')

if __name__ == '__main__':
    # instantiate the class
    r = speckle.phasing.phasing()
    
    # run the reconstruction algorithm
    reconstruct()
