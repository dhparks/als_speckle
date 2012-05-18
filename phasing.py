import scipy
from scipy.fftpack import fft2 as DFT
from scipy.fftpack import ifft2 as IDFT
from scipy.fftpack import fftshift

import pyopencl as cl
import pyopencl.array as cl_array
from pyopencl.elementwise import ElementwiseKernel
import pyfft

from types import *

import shape
import io2
import conditioning

from phasing_parameters import *

def init_gpu():
    # create the queue and context so that they are available as pointers?
    
    platforms = cl.get_platforms();
    if len(platforms) == 0: print "Failed to find any OpenCL platforms."; exit()

    devices = platforms[0].get_devices(cl.device_type.GPU)
    if len(devices) == 0: print "Could not find any GPU device"; exit()
    device = devices[0]

    context = cl.Context([devices[0]]) # Create a context using the first gpu in the list of devices
    queue = cl.CommandQueue(context) # Create a command queue for that device
    
    return context, queue, device

def condition_data(dataname,darkname,dust_plan=None,align_params=None,threshold=0.9):
    
    # data coming in here is assumed to be ill-conditioned because it is basically right off the ccd.
    # this function applies some routine conditioning to try to get the data into a suitable form to be
    # passed to the reconstructer.
    
    # 1. optimally subtract dark frame from each image by looking at DC in corners
    # 2. do hot-pixel removal on each frame
    # 3. select a subarray and align each frame to a reference (probably the zero frame)
    # 4. to deal with drift, cross-correlate each frame with every other frame. only those frames which
    #    are sufficiently similar will be included in the sum
    # 4. sum along the frame axis
    
    # check some assumptions regarding inputs
    assert type(align_params) in [NoneType,ListType,TupleType,scipy.ndarray],  "subarray must be None or iterable"
    if type(align_params) != NoneType:
        assert len(align_params) == 4,                                         "subarray must be length-4"
        for n in range(4):
            assert align_params[n] in [IntType,FloatType],                     "all data in subarray must be float or int"
    
    # check the header of the data file. ensure the file is either 2d or 3d. check that if there is a
    # dark file, the shape of each frame of data is the same as of dark.
    data_header = io2.openheader(dataname)
    data_shape = [0,0,0]
    for n in range(3):
        try: data_shape[n] = data_header['NAXIS%s'%(3-n)]
        except: pass
    print "data shape:",data_shape    
    
    if darkname == None:
        dark = None
    else:
        dark_header = io2.openheader(darkname)
        dark_shape = [0,0,0]
        for n in range(3):
            try: dark_shape[n] = int(dark_header['NAXIS%s'%(3-n)])
            except: pass
        assert data_shape[1:] == dark_shape[1:], "data and dark acquisitions must be same size"
        print "dark shape:", dark_shape
        
        # open the dark as a sum along the frame axis. There is nothing fancy to be done with the dark frames.
        if dark_shape[0] > 0:
            dark = scipy.zeros(dark_shape[1:],float)
            for n in range(dark_shape[0]):
                dark += io2.openframe(darkname,n)
            dark *= 1./dark_shape[0]
            
        if dark_shape[0] == 0: dark = io2.openfits(darkname)
        
        # now fix hot pixels and dust spots
        dark = conditioning.hot_pixels(dark,t=1.2)
        if dust_plan != None: dark = conditioning.remove_dust(dark,dust_plan,use_old_plan=True)

    # process the data file. much more complicated!
    frames = data_shape[0]
    if frames == 0: conditioning.subtract_background(io2.openfits(dataname),dark)
    if frames > 0:
        
        # this algorithm tries to minimize the amount of memory used, which can otherwise become problematic when very large datasets
        # must be conditioned. the cost of this optimization is that some operations must either be inexact or duplicated;
        # for example, sub-frame cross-correlation alignment done without dark subtraction might be less accurate, but
        # if dark frame subtraction is done before alignment it also must be done again when the signal is summed across frames.
        
        # get the sub-frames for alignment
        print "doing sub-frame alignment"
        first_frame = io2.openframe(dataname,0)
        if align_params == None: # choose sensible defaults
            maxloc = first_frame.argmax()
            maxrow,maxcol = maxloc/first_frame.shape[1],maxloc%first_frame.shape[1]
            rmin,rmax,cmin,cmax = maxrow-64,maxrow+64,maxcol-64,maxcol+64
        else: rmin,rmax,cmin,cmax = align_params
        to_align = scipy.zeros((frames,rmax-rmin,cmax-cmin),float)
        print "  reading frames"
        for n in range(frames):
            to_align[n] = io2.openframe(dataname,n)[rmin:rmax,cmin:cmax]
            
        # get the alignment coordinates
        print "  doing analysis"
        align_coords = align_frames(to_align,return_type='coordinates')
        
        # figure out which frames to sum. this takes largers arrays than the alignment
        # and all frames must be correlated against each other.
        print "doing sub-frame cc analysis to sort which frames to sum"
        f = 1
        delta_r,delta_c = rmax-rmin,cmax-cmin
        ave_r,ave_c = int((rmax+rmin)/2),int((cmax+cmin)/2)
        
        if delta_r > delta_c:
            if delta_r%2 == 1:
                delta_r += 1
                if rmin > 0:  rmin += -1
                if rmin == 0: rmax += 1
            cmin,cmax = ave_c-delta_r/2,ave_c+delta_r/2
            
        if delta_c > delta_r:
            if delta_c%2 == 1:
                delta_c += 1
                if cmin > 0:  cmin += -1
                if cmin == 0: cmax += 1
            rmin,rmax = ave_r-delta_c/2,ave_r+delta_c/2

        print "  reading frames"
        to_correlate = scipy.zeros((frames,delta_r,delta_c),float)
        for n in range(frames):
            to_correlate[n] = io2.openframe(dataname,n)[rmin:rmax,cmin:cmax]
        print "  doing analysis"
        correlation_matrix = conditioning.covariance_matrix(to_correlate)
        best = scipy.sum(correlation_matrix,0).argmax()
        to_sum_list = scipy.where(correlation_matrix[best] > threshold,1,0)

        # do the summation
        print "summing frames with dark subtraction"
        summed = scipy.zeros_like(first_frame)
        for n in range(frames):
            if to_sum_list[n]:
                rolls = align_coords[n]
                frame = io2.openframe(dataname,n)                                      # open frame
                frame = conditioning.remove_dust(frame,dust_plan,use_old_plan=True)    # remove dust spots
                #frame = conditioning.subtract_background(frame,dark)                   # subtract background
                frame = scipy.roll(scipy.roll(frame,rolls[0],axis=0),rolls[1],axis=1)  # align to master
                summed += frame                                                        # add frame to sum
        
        print "removing hot pixels from sum with median filter"
        summed = conditioning.hot_pixels(summed)
        
        # finally, roll the data so that the max is at the corners and then take the square root to make it modulus instead of mod**2
        maxloc = summed.argmax()
        maxrow, maxcol = maxloc/len(summed),maxloc%len(summed)
        summed = scipy.roll(scipy.roll(summed,-maxrow,axis=0),-maxcol,axis=1)
        print "done"
        return scipy.sqrt(summed)

def speckle(input):
    # return coherent speckle pattern of input
    return fftshift(abs(DFT(self.estimate))**2)

def roll_phase(data):
    # Phase retrieval is degenerate to a global phase factor. This function tries to align the global phase rotation
    # by minimizing the amount of power in the imag component. Real component could also be minimized with no effect
    # on the outcome.
    
    # check types
    assert type(data) == scipy.ndarray, "data must be array"
    assert data.ndim in [2,3], "data must be 2d or 3d"
    
    imag_component = lambda p,x: (x*scipy.exp(complex(0,1)*p)).imag
    
    from scipy.optimize import leastsq
    if data.ndim == 2:
        p = 0
        p = leastsq(imag_component,p,args=(data.ravel()))[0]
        data *= scipy.exp(complex(0,1)*p)
        
    if data.ndim == 3:
        for frame in data:
            p = 0
            p = leastsq(imag_component,p,args=(frame.ravel()))[0]
            frame *= scipy.exp(complex(0,1)*p)
    
    return data

class CPUPR:
    
    # Implement phase retrieval as a class. An instance of this class is a reconstruction, and methods in the class are operations on the reconstruction.
    # For example, calling instance.hio() will advance the reconstruction by one iteration of the HIO algorithm; calling instance.update_support() will
    # update the support being used in the reconstruction in the manner of Marchesini's shrinkwrap.

    def __init__(self,N):
        self.N = N
        
    def load_data(self,modulus,support,update_sigma=None):
        print update_sigma
        
        # get the supplied data into the reconstruction namespace
        self.modulus  = modulus
        self.support  = support  # this is the active support, it can be updated
        self.support0 = support # this is the original support, it is read-only
        
        # generate some necessary files
        self.estimate   = ((scipy.rand(self.N,self.N)+complex(0,1)*scipy.rand(self.N,self.N))*self.support)
        
        if update_sigma != None:
            assert type(update_sigma) in (IntType,FloatType), "update_sigma must be float or int"
            self.blurkernel = DFT(fftshift(shape.gaussian((self.N,self.N),(update_sigma,update_sigma),center=None,normalization=None)))
        
    def iteration(self,algorithm,beta=0.8):
        
        assert algorithm in ['hio','er'], "real space enforcement algorithm %s is unknown"%algorithm
        
        psi = DFT(self.estimate)
        psi = self.modulus*psi/abs(psi)
        inverse = IDFT(psi)
        
        # enforce the real-space constraint
        if algorithm == 'hio': self.estimate = (1-self.support)*(self.estimate-beta*inverse)+self.support*inverse # hio support algorithm
        if algorithm == 'er': self.estimate = self.support*self.estimate
        
    def update_support(self,threshold = 0.25,retain_bounds=True):
        
        # auto-update the support by blurring the magnitude component of the estimate and retaining the selection of the blurred
        # signal greater than some threshold fraction of the signal maximum.
        
        # the retain_bounds flag is intended to keep the updated support from growing outside of the support boundaries supplied with
        # the first support estimate, the assumption being that that support was much too loose and updates should only get tighter.
        
        blur = lambda a,b: IDFT(DFT(a)*b)
        
        mag     = abs(self.estimate)  
        blurred = blur(mag,self.blurkernel)
        update  = scipy.where(blurred > blurred.max()*threshold,1,0)
        if retain_bounds: update *= self.support0
        self.support = update
        
    def save(self,savepath,savename,n,save_estimate=True,save_diffraction=False,save_support=False):
        if save_estimate:    io2.save_fits(savepath+'/'+savename+' '+str(iteration)+'.fits',             self.estimate,          components=['mag','phase'], overwrite=True)
        if save_support:     io2.save_fits(savepath+'/'+savename+' '+str(iteration)+' support.fits',     self.support,           components=['mag'], overwrite=True)
        if save_diffraction: io2.save_fits(savepath+'/'+savename+' '+str(iteration)+' diffraction.fits', speckle(self.estimate), components=['mag'], overwrite=True)
            
class GPUPR:

    def __init__(self,context,queue,device,N,bounds=None):
        self.N = N
        self.queue = queue
        self.context = context
        self.device = device

        # 1. make fft plan for a 2d array with length N
        if verbose: print "making fft plan"
        from pyfft.cl import Plan
        self.fftplan = Plan((self.N, self.N), queue=self.queue)
        
        # 2. make the kernels to enforce the fourier and real-space constraints
        if verbose: print "compling kernels"
        self.fourier_constraint = ElementwiseKernel(self.context,
            "float2 *psi, "                        # current estimate of the solution
            "float  *modulus, "                    # known fourier modulus
            "float2 *out",                         # output destination
            "out[i] = rescale(psi[i],modulus[i])", # operator definition
            "replace_modulus",
            preamble = """
            #define rescale(a,b) (float2)(a.x/hypot(a.x,a.y)*b,a.y/hypot(a.x,a.y)*b)
            """)
        
        self.realspace_constraint_hio = ElementwiseKernel(self.context,
            "float beta, "       # feedback parameter
            "float *support, "   # support constraint array
            "float2 *psi_in, "   # estimate of solution before modulus replacement
            "float2 *psi_out, "  # estimate of solution after modulus replacement
            "float2 *out",       # output destination
            "out[i] = (1-support[i])*(psi_in[i]-beta*psi_out[i])+support[i]*psi_out[i]",
            "hio")
        
        self.realspace_constraint_er = ElementwiseKernel(self.context,
            "float *support, "   # support constraint array
            "float2 *psi_out, "  # estimate of solution after modulus replacement
            "float2 *out",       # output destination
            "out[i] = support[i]*psi_out[i]",
            "hio")
        
        if save_to_buffer:
            
            # set up the buffer which will store the reconstructions. this has to be done in __init__ rather than
            # load_data because load_data gets called for each trial, but this buffer stores teh outcome of all the trials
            
            self.bounds = bounds
            self.rows = int(bounds[1]-bounds[0])
            self.cols = int(bounds[3]-bounds[2])
            self.x0 = scipy.int32(bounds[2])
            self.y0 = scipy.int32(bounds[0])
            t = int(trials)
            
            self.save_buffer = cl_array.empty(self.queue,(t,self.rows,self.cols),scipy.complex64)
            
            # this copies a subarray from a 2d array into the 3d self.save_buffer
            self.copy_to_buffer = cl.Program(self.context,
            """__kernel void execute(__global float2 *dst, __global float2 *src, int x0, int y0, int rows, int cols, int n, int N)
            {
                int i_dst = get_global_id(0);
                int j_dst = get_global_id(1);
            
                // i_dst and j_dst are the coordinates of the destination. we "simply" need to turn them into 
                // the correct indices to move values from src to dst.
                
                int dst_index = (n*rows*cols)+(j_dst*rows)+i_dst; // (frames)+(rows)+cols
                int src_index = (i_dst+x0)+(j_dst+y0)*N; // (cols)+(rows)
                
                dst[dst_index] = src[src_index];
            }""").build(devices=[self.device])

        # if the support will be updated with shrinkwrap, initialize some additional gpu kernels
        if shrinkwrap:
            
            self.set_zero = ElementwiseKernel(self.ctx,
                "float2 *buff",
                "buff[i] = (float2)(0.0f, 0.0f)",
                "set_zero")
            
            self.copy_real = ElementwiseKernel(self.ctx,
                "float2 *in,"
                "float  *out",
                "out[i] = in[i].x",
                "set_zero")
            
            self.make_abs = ElementwiseKernel(self.ctx,
                "float2 *in,"
                "float2 *out",
                "out[i] = (float2)(hypot(in[i].x,in[i].y),0.0f)",
                "make_abs")
            
            self.blur_convolve = ElementwiseKernel(self.ctx,
                "float2 *toblur,"
                "float  *blurrer,"
                "float2 *blurred",
                "blurred[i] = (float2) (toblur[i].x*blurrer[i],toblur[i].y*blurrer[i])",
                "blur_convolve")
            
            self.support_threshold = ElementwiseKernel(self.ctx,
                "float2 *in,"
                "float *out,"
                "float t",
                "out[i] = isgreaterequal(in[i].x,t)",
                "support_threshold")
            
            self.bound_support = ElementwiseKernel(self.ctx,
                "float *s,"
                "float *s0",
                "s[i] = s[i]*s0[i]",
                "bound_support")

    def load_data(self,modulus,support,update_sigma=None):
        # put the support and the modulus on the gpu
        # also establish buffers for the intermediates
        
        # make sure the modulus and support are the correct single-precision dtype
        if modulus.shape != support.shape: exit()
        
        # transfer the data to the cpu. pyopencl handles buffer creation
        random_complex = ((scipy.rand(self.N,self.N)+complex(0,1)*scipy.rand(self.N,self.N))*support).astype('complex64')
        self.modulus  = cl_array.to_device(self.queue,modulus.astype('float32'))
        self.support  = cl_array.to_device(self.queue,support.astype('float32'))
        self.support0 = cl_array.to_device(self.queue,support.astype('float32'))
        
        # initialize gpu arrays for fourier constraint satisfaction
        self.psi_in      = cl_array.to_device(self.queue,random_complex)
        self.psi_out     = cl_array.empty(self.queue,(self.N,self.N),scipy.complex64) # need both in and out for hio algorithm
        self.psi_fourier = cl_array.empty(self.queue,(self.N,self.N),scipy.complex64) # to store fourier transforms of psi
        
        if update_sigma != None:
            
            # make gpu arrays for blurring the magnitude of the current estimate in order to update the support
            
            assert type(update_sigma) in (IntType,FloatType), "update_sigma must be float or int"
            blurkernel = abs(DFT(fftshift(shape.gaussian((self.N,self.N),(update_sigma,update_sigma),center=None,normalization=None))))
            self.blur_kernel   = cl_array.to_device(self.queue,blurkernel.astype('float32'))
            self.blur_temp     = cl_array.empty(self.queue,(self.N,self.N),scipy.complex64) # for holding the blurred estimate
            self.blur_temp_max = cl_array.empty(self.queue,(self.N,self.N),scipy.float32)
 
    def update_support(self,threshold = 0.2,retain_bounds=True):
            
        # zero the temp buffer for safety. this is basically a free operation
        self.set_zero(self.blur_temp)

        # make a blurry version of the abs of psi_out
        self.make_abs(self.psi_out,self.blur_temp).wait()
        self.fftplan.execute(self.blur_temp.data,wait_for_finish=True)
        self.blur_convolve(self.blur_temp,self.blur_kernel,self.blur_temp)
        self.fftplan.execute(self.blur_temp.data,inverse=True,wait_for_finish=True)
        
        # do the thresholding procedure and update the support.
        # thresholding and copying to the self.support buffer are in the same kernel
        self.copy_real(self.blur_temp,self.blur_temp_max).wait()
        m = (threshold*cl_array.max(self.blur_temp_max).get()).astype('float32')
        self.support_threshold(self.blur_temp,self.support,m).wait()
        
        # enforce the condition that the support shouldnt expand in size
        if retain_bounds: self.bound_support(self.support,self.support0)
        
        print "updated"
         
    def iteration(self,algorithm,beta=0.8):

        # do a single iteration. algorithm is selectable through keyword

        assert algorithm in ['hio','er'], "real space enforcement algorithm %s is unknown"%algorithm

        # 1. fourier transform the data in psi_in, store the result in psi_fourier
        self.fftplan.execute(self.psi_in.data,data_out=self.psi_fourier.data)
        
        # 2. enforce the fourier constraint by replacing the current-estimated fourier modulus with the measured modulus from the ccd
        self.fourier_constraint(self.psi_fourier,self.modulus,self.psi_fourier)
        
        # 3. inverse fourier transform the new fourier estimate
        self.fftplan.execute(self.psi_fourier.data,data_out=self.psi_out.data,inverse=True)
        
        # 4. enforce the real space constraint. algorithm can be changed based on incoming keyword
        if algorithm == 'hio': self.realspace_constraint_hio(beta,self.support,self.psi_in,self.psi_out,self.psi_in)
        if algorithm == 'er':  self.realspace_constraint_er(self.support,self.psi_out,self.psi_in)
        
    def save(self,savepath,savename,n,save_estimate=True,save_diffraction=False,save_support=False):
        
        if save_to_buffer:
            self.copy_to_buffer.execute(self.queue,(self.cols,self.rows), # opencl stuff
                                   self.save_buffer.data,           # destination
                                   self.psi_in.data,                # source
                                   self.x0, self.y0,
                                   scipy.int32(self.rows),scipy.int32(self.cols),
                                   scipy.int32(trial),scipy.int32(self.N))
        
        if save_to_disk:
            if save_estimate:    io2.save_fits(savepath+'/'+savename+' '+str(iteration)+'.fits',             self.psi_in.get(),          components=['mag','phase'], overwrite=True)
            if save_support:     io2.save_fits(savepath+'/'+savename+' '+str(iteration)+' support.fits',     self.support.get(),         components=['mag'], overwrite=True)
            if save_diffraction: io2.save_fits(savepath+'/'+savename+' '+str(iteration)+' diffraction.fits', speckle(self.psi_in.get()), components=['mag'], overwrite=True)
        
def _do_iterations(reconstruction):
    global iteration
    for iteration in range(iterations):

        if iteration%update_period == 0 and iteration > 0 and shrinkwrap:
            reconstruction.update_support()
            reconstruction.iteration('er')
        else:
            if (iteration+1)%100 != 0:
                reconstruction.iteration('hio')
            else:
                reconstruction.iteration('er')
        if iteration%100 == 0: print "  iteration ",iteration
        
def bound(data,threshold=1e-10,force_to_square=False,pad=0):
    # find the minimally bound non-zero region of the support. useful
    # for storing arrays so that the zero-padding for oversampling is avoided.
    
    data = scipy.where(data > threshold,1,0)
    rows,cols = data.shape
    
    rmin,rmax,cmin,cmax = 0,0,0,0
    
    for row in range(rows):
        if data[row,:].any():
            rmin = row
            break
            
    for row in range(rows):
        if data[rows-row-1,:].any():
            rmax = rows-row
            break
            
    for col in range(cols):
        if data[:,col].any():
            cmin = col
            break
    
    for col in range(cols):
        if data[:,cols-col-1].any():
            cmax = cols-col
            break
        
    if rmin >= pad: rmin += -pad
    else: rmin = 0
    
    if rows-rmax >= pad: rmax += pad
    else: rmax = rows
    
    if cmin >= pad: cmin += -pad
    else: cmin = 0
    
    if cols-cmax >= pad: cmax += pad
    else: cmax = cols
        
    if force_to_square:
        delta_r = rmax-rmin
        delta_c = cmax-cmin
        
        if delta_r%2 == 1:
            delta_r += 1
            if rmax < rows: rmax += 1
            else: rmin += -1
            
        if delta_c%2 == 1:
            delta_c += 1
            if cmax < cols: cmax += 1
            else: cmin += -1
            
        if delta_r > delta_c:
            average_c = (cmax+cmin)/2
            cmin = average_c-delta_r/2
            cmax = average_c+delta_r/2
            
        if delta_c > delta_r:
            average_r = (rmax+rmin)/2
            rmin = average_r-delta_c/2
            rmax = average_r+delta_c/2
            
        if delta_r == delta_c:
            pass
        
    return scipy.array([rmin,rmax,cmin,cmax]).astype('int32')
   
def align_frames(data,align_to=None,method='fft',search=None,use_mag_only=True,return_type='data'):
    
    """ Align a sequence of images for precision averaging.
    
    Required input:
    data -- A 2d or 3d ndarray. probably this needs to be real-valued, not complex.
    
    Optional input:
        align_to -- A 2d array used as the alignment reference. If left as None, data must
                    be 3d and the first frame of data will be the reference.
                
        method -- Alignment method, can be either 'fft' or 'diff'. fft attempts to align images through
                  cross-correlation; 'diff' through method of least differences as an optimization.
                  Default is fft. 'diff' is not yet implemented.
              
        search -- If 'diff' is selected as the method, 'search' is the size of the maximum displacement in
                  each axis that the candidate will be displaced to calculate the diff. So, if the
                  candidate and the reference are off by 3 pixels in y and 1 pixel in x, search needs to
                  be at least 3 to find the correct alignment.
                  
        use_mag_only -- Do alignment using only the magnitude component of data. Default is True
                  
        return_type -- This function is called from multiple places, some of which expect aligned data to come back
                        and some which expect just the alignment coordinates to come back. 'data' returns
                        the aligned data, 'coordinates' returns the alignment coordinates. Default is 'data'.
              
    Returns: an array of shape and dtype identical to data, or a list of alignment coordinates."""
    
    # check types
    assert type(data) == scipy.ndarray,                   "data must be ndarray"
    assert data.ndim in [2,3],                            "data must be 2d or 3d"
    assert type(align_to) in [NoneType,scipy.ndarray],    "align_to must be None or ndarray"
    if type(align_to) == NoneType: assert data.ndim == 3, "with align_to as None, data must be 3d"
    assert method in ['fft','diff'],                      "method must be 'fft' or 'diff'"
    assert return_type in ['data','coordinates'],         "return_type must be 'data' or 'coordinates'"

    coordinates = []

    if method == 'fft':
        if data.ndim == 2:
            if use_mag_only: cc = abs(IDFT(DFT(abs(data))*scipy.conjugate(DFT(align_to))))
            else: cc = abs(IDFT(DFT(data)*scipy.conjugate(DFT(align_to))))
                
            cc_max = cc.argmax()
            rows,cols = data.shape
            max_row,max_col = cc_max/cols,cc_max%cols
            if return_type == 'coordinates': coordinates.append([-max_row,-maxcol])
            if return_type == 'data': data = scipy.roll(scipy.roll(data,-max_row,axis=0),-max_col,axis=1)
           
        if data.ndim == 3:
            if align_to == None: align_to = data[0]
            if use_mag_only: align_to = abs(align_to)
            align_to_f = scipy.conjugate(DFT(align_to))
            rows,cols = align_to.shape
            for n,frame in enumerate(data):
                if use_mag_only: frame = abs(frame)
                cc_max = abs(IDFT(DFT(frame)*align_to_f)).argmax()
                max_row,max_col = cc_max/cols,cc_max%cols
                if return_type == 'coordinates': coordinates.append([-max_row,-max_col])
                if return_type == 'data': data[n] = scipy.roll(scipy.roll(data[n],-max_row,axis=0),-max_col,axis=1)

    if return_type == 'coordinates': return coordinates
    if return_type == 'data': return data

def _example():
    # canonical code to demonstrate library functions
    # allow execution of the following code only when the phasing.py script is invoked directly
    global trial

    if raw_data:
        data,frame_correlations = condition_data(dataname,darkname,dust_plan=dust_plan)
        io2.save_fits('conditioned barker data.fits',data,overwrite=True)
        io2.save_fits('frame correlations barker data.fits',frame_correlations,overwrite=True)
    else:
        data = io2.openfits(dataname)
        
    support = io2.openfits(supportname)
    bounds = bound(support,force_to_square=True,pad=4)
        
    # check sizes
    assert data.ndim == 2, "data has wrong dimensionality"
    assert support.ndim == 2, "support has wrong dimensionality"
    assert data.shape == support.shape, "data and support have different sizes"
    
    # initialize the reconstruction
    N = len(data)
    if where == 'CPU':
        reconstruction = CPUPR(N)
    if where == 'GPU':
        context, queue, device = init_gpu()
        reconstruction = GPUPR(context,queue,device,N,bounds=bounds)
    
    for trial in range(trials):
        
        print "trial ",trial
        
        # load data/make new seed
        sigma = None
        if shrinkwrap: sigma = shrinkwrap_sigma
        reconstruction.load_data(data,support,update_sigma = sigma)
        
        # iterate
        _do_iterations(reconstruction)

        # save result, either to disk or in a memory buffer depending on phasing_parameters
        reconstruction.save(savepath,savename,trial)
        
    # now get the data off the gpu
    cpu_data = reconstruction.save_buffer.get()
    io2.save_fits(savepath+'/'+savename+'.fits', cpu_data, components=['real','imag'], overwrite=True)
    
    # roll the phase to ensure global phase alignment
    cpu_data = roll_phase(cpu_data)
    io2.save_fits(savepath+'/'+savename+' rolled.fits', cpu_data, components=['real','imag'], overwrite=True)
    
    # align the trials
    print cpu_data.shape
    cpu_data = align_frames(cpu_data)
    print cpu_data.shape
    io2.save_fits(savepath+'/'+savename+' aligned.fits', cpu_data, components=['real','imag'], overwrite=True)
    
    # propagate the average and calculate the acutance at each 
        
if __name__== "__main__": _example()

        
        
        
        
    
    
    
