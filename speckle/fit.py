""" A general-purpose library for fitting image and (x,y) data.

Author: Keoki Seu (KASeu@lbl.gov)
""" 

import numpy as np
from scipy.optimize import leastsq

from . import shape, conditioning, wrapping

class OneDimFit():
    """Framework for calculating the fit for a one-dimensional (x,y) set of
        data. This implements the data checking, fitting, and formatting of
        final fit parameters.  This class is expected to be used as a basis for
        a one-dimensional fit.  The functions fit_function() and
        guess_parameters() need to be filled up by the child class.
    """
    def __init__(self, data, mask=None):
        # These parameters should be filled out by the child class.
        self.functional = "f(x) = a*x + b"
        self.params_map ={ 0:"a", 1:"b"}

        # prepare the data
        assert isinstance(data, np.ndarray), "data must be an array"
        assert data.ndim == 2, "data must be two-dimensional"
        assert not np.iscomplexobj(data), "data must not be complex"

        ys, xs = data.shape
        if ys == 2 or xs == 2:
            # small bug: if you have an image where a dimension is 2 then it will think it's 1d data.

            # for 1d fitting, data should be in the shape (npoints, 2)
            if xs == 2:
                self.npoints = ys
            else:
                self.npoints = xs
                data = data.swapaxes(0,1)
    
            self.xvals = data[:, 0]
            self.data = data[:, 1]            
            self.try_again = np.ones_like(self.data)* 1e30
            self.have_image = False
        else:
            # We have an image.
            self.xvals = np.arange(xs)
            self.yvals = np.arange(ys)
            self.npoints = len(self.xvals)*len(self.yvals)
            self.data = data
            self.try_again = np.ones_like(self.data) * 1e30
            self.have_image = True

        if mask is None:
            self.mask = np.ones_like(self.data)
        else:
            assert mask.shape == self.data.shape, "mask and data are different shapes"
            self.mask = np.where(mask, 1, 0)

        self.ys, self.xs = data.shape

    def fit_function(self):
        """ This needs to be filled out by the super() function
        """
        pass

    def guess_parameters(self):
        """ This needs to be filled out by the super() function
        """
        pass

    def residuals(self, params=None):
        """ Calculate the residuals: data - fit_function(x).  params is an
        optional parameter array. If params is not specified, it looks for the
        array in the class.
        """
        if params is not None:
            self.params = params
        return np.nan_to_num(np.ravel((self.data - self.fit_function())*self.mask))

    def fit(self, ransac_params = None):
        """ Fit data using fit_function().  This calls scipy.optimize.leastsq().

        Optionally do a Random sample consensus (ransac) fit. the ransac is an
        algorithm that tries to fit a subset of the data first then tries to
        detect outliers. More details of the ransac algorithm can be found here:
            https://en.wikipedia.org/wiki/RANSAC

        The ransac_params is a list of four parameters (minpts, iterations,
        include_col, min_fit_pts) where:
            minpts - the minimum number of data required to fit the model
            iterations - the number of iterations performed by the algorithm
            include_tol - a threshold value for determining when a datum fits a
                model. All points where the (model - data) < include_tol are
                added as inliers.
            min_fit_pts - the minimum number of data values required to assert
                that a model fits well to data

        In this mode, the mask for the fit (self.mask) is modified with the
        points that are used in the fit.
        """
        try:
            if not isinstance(self.params, np.ndarray):
                self.guess_parameters()
        except AttributeError:
            self.guess_parameters()

        if ransac_params != None:
            self._ransac(ransac_params[0], ransac_params[1], ransac_params[2], ransac_params[3])
            self.guess_parameters()

        optimized, cov_x, infodict, mesg, ier = leastsq(self.residuals, self.params, full_output=1)

        self.params = optimized
        self.final_params = optimized
        self.final_residuals = infodict['fvec']

        self.final_chisq = (self.final_residuals**2).sum()
        self.final_variance = np.var(self.final_residuals)

        self.final_fn_evaulations= infodict['nfev']
        self.final_jacobian = infodict['fjac']

        # https://en.wikipedia.org/wiki/Coefficient_of_determination
        ss_err=(self.final_residuals**2).sum()
        ss_tot=((self.data-self.fit_function().mean())**2).sum()
        self.final_Rsquared = 1 - (ss_err/ss_tot)

        try:
            # calculate the errors the same way that gnuplot uses, namely sqrt(diagonal(cov_x)*chisq/df).
            self.final_errors = np.sqrt(np.diagonal(cov_x))*np.sqrt(self.final_chisq/(self.npoints - len(self.params)))
        except ValueError:
            print "Error: The fit did not converge. Please change initial parameters and try again"
            self.final_errors = np.zeros_like(self.params)

        if ier not in (1,2,3,4):
            # read somewhere that ier = 1, 2, 3, 4 also gives correct solution. http://bugs.debian.org/cgi-bin/bugreport.cgi?bug=474930
            print "Error: No solution found!"
            self.final_errors = np.zeros_like(self.params)
        
        self.final_params_errors = {}
        for k, v in self.params_map.iteritems():
            self.final_params_errors[v] = (optimized[k], self.final_errors[k])

    def format_results(self, header=True, outfile=None):
        """ Format the final, fitting data for printing or writing to disk.

        arguments:
            header - weather or not to print the header.  Defaults to True
            outfile - If an outfile is specified, this function will write the parameters to outfile on disk.

        returns:
            the formatted results string.
        """
        fun = "# fitting functional %s\n" % self.functional
        hdr = "# "
        hdr_e = ""
        outstr = ""
        outstr_e = ""
        for i in range(len(self.final_params_errors)):
            pm = self.params_map
            var_name = pm[i]
            hdr += "%s\t" % var_name
            hdr_e += "error(%s)\t" % var_name
            outstr += "%1.7e\t" % self.final_params_errors[var_name][0]
            outstr_e += "%1.7e\t" % self.final_params_errors[var_name][1]

        hdr_e += "function calls\tchi^2\tR^2\n"
        outstr_e += "%d\t%1.7e\t%1.7f\n" % (self.final_fn_evaulations, self.final_chisq, self.final_Rsquared)

        if header:
            out = fun + hdr + hdr_e + outstr + outstr_e
        else:
            out = outstr + outstr_e

        if outfile is not None:
            with open(outfile, "w") as f:
                f.write(out)

        return out

    def _ransac(self, minpts, iterations, include_tol, min_fit_pts):
        """ Implements the RANdom SAmple Conesensus (RANSAC) iterative algorithm
        for fitting data that contains outliers. The algorithm starts with a
        small subset of data of size n (called inliers) and adds points while
        trying to keep the error minimal.

        more details: https://en.wikipedia.org/wiki/RANSAC

        input:
            minpts - the minimum number of data required to fit the model
            iterations - the number of iterations performed by the algorithm
            include_tol - a threshold value for determining when a datum fits a
                model. All points where the (model - data) < include_tol are
                added as inliers.
            min_fit_pts - the minimum number of data values required to assert
                that a model fits well to data
        """
        import sys
        # This implementation makes use of the self.mask variable to change the fitted parameters.  It does this while respecting the original mask value in the data. At the end of this routine, the mask contains the originally masked out data plus the excluded points (if a consensus is found).
        
        assert(min_fit_pts < self.npoints), "min_fit_pts must be less than data len"

        def get_unique_random_vals(maxval, num):
            """ Grabs unique random values. maxval can be a 2-tuple/list/set of
            max values along each axis. num is the number or values to draw.
            """
            rint = np.random.randint
            if isinstance(maxval, (list, tuple, set)):
                assert len(maxval) == 2, "maxval list/tuple/set must be 2"
                (ys, xs) = maxval
                size = len(maxval)
                assert num <= ys*xs, "The number of unique values requested (%d) is more than the number of elements we can iterate over (%d)" % (num, ys*xs)
            else:
                size = None
                assert num <= maxval, "The number of unique values requested (%d) is more than the number of elements we can iterate over (%d)" % (num, maxval)

            randvals = set()
            while len(randvals) < num:
                remaining = num - len(randvals)
                if size == None:
                    randvals.update( rint(maxval, size=remaining))
                else:
                    randvals.update( zip( rint(ys,size=remaining), rint(xs,size=remaining) ) )

            return randvals

        def guess_and_fit():
            self.guess_parameters()
            return leastsq(self.residuals, self.params)

        def set_mask_with(a_set):
            self.mask = np.zeros_like(initial_mask)
            if self.have_image:
                ys = valid_coordinates[0][list(a_set)]
                xs = valid_coordinates[1][list(a_set)]
                self.mask[ys,xs] = 1
            else:
                self.mask[list(a_set)] = 1
            self.mask *= initial_mask

        def error():
            return np.sqrt(np.sum(self.residuals()**2)/self.mask.sum())

        # keep a copy of the mask so we can replace it at the end.  This function modifies it.
        initial_mask = self.mask.copy()
        # this contains the points that are allowed to be in the fit
        valid_coordinates = np.where(initial_mask == 1)
        
        best_consensus_set = set()
        best_error = 1e30 # large number
        for i in range(iterations):
            # consensus_set - values within valid_coordinates that are good
            consensus_set = get_unique_random_vals(len(valid_coordinates[0]), minpts)
            set_mask_with(consensus_set)
            a = guess_and_fit()

            self.mask = initial_mask
            residuals = self.residuals()


            for candidate_point in np.delete(valid_coordinates, list(consensus_set)):
                if abs(residuals[candidate_point]) < include_tol:
                    consensus_set.add(candidate_point)

            valid = "No"
            this_error = error()
            cs_len = len(consensus_set)
            if cs_len > min_fit_pts:
                set_mask_with(consensus_set)
                a = guess_and_fit()
                if this_error < best_error:
                    best_consensus_set = consensus_set
                    best_error = this_error
                    valid = "Yes"

            sys.stdout.write("\riteration: {0:3d} error: {1:5e} npts: {2:3d}/{3:3d} valid: {4:3}".format(i+1, this_error, cs_len, min_fit_pts, valid))
            sys.stdout.flush()

        print ""

        if len(best_consensus_set) > 0:
            set_mask_with(best_consensus_set)
        else:
            print "RANSAC: not able to find a good set of points, defaulting to original."
            self.mask = initial_mask


class OneDimPeak(OneDimFit):
    """Framework for calculating the fit for a one-dimensional (x,y) set of data
        that has a peak.  This class is the child of the OneDimFit class that
        has a FWHM estimation fitting routine. Depending on the function, the
        FWHM estimated needs to be multipled by a factor to get to the correct
        parameter. This class is expected to be used as a basis for a
        one-dimensional fit of a peak.  The functions fit_function() and
        guess_parameters() need to be filled up by the child class.
    """
    def __init__(self, *args, **kwargs):
        OneDimFit.__init__(self, *args, **kwargs)

    def estimate_fwhm(self):
        """ Estimate a FWHM peak. This returns a reasonably good estimate of the
            FWHM of a 1d peak.
        """
        hm = (self.data.max() - self.data.min())/2
        argmax = self.data.argmax()
        if argmax == len(self.xvals):
            argmax = argmax - 1
        elif argmax == 0:
            argmax = 1

        absdiff = abs(self.data - hm)
        left = self.xvals[absdiff[np.argwhere(self.xvals < argmax)].argmin()]
        right = self.xvals[absdiff[np.argwhere(self.xvals > argmax)].argmin()]

        return abs(left-right)

class TwoDimPeak(OneDimFit):
    """Framework for calculating the fit for a two-dimensional (x,y,z) set of
        data that has a peak.  This class is the child of the OneDimFit class
        that has a FWHM estimation fitting routine for 2d. Depending on the
        function, the FWHM estimated needs to be multipled by a factor to get to
        the correct parameter. This class is expected to be used as a basis for
        a two-dimensional fit of a peak.  The functions fit_function() and
        guess_parameters() need to be filled up by the child class.
    """
    def __init__(self, *args, **kwargs):
        OneDimFit.__init__(self, *args, **kwargs)

    def estimate_fwhm(self):
        # a more accurate estimate can be obtained by doing some sort of linear interpolation in the neighborhood of the left and right solution but this seems like overkill.
        hm = (self.data.max() - self.data.min())/2.0
        yargmax, xargmax = np.unravel_index(self.data.argmax(), self.data.shape)

        lx = abs(self.data[yargmax,:xargmax-1] - hm).argmin()
        rx = abs(self.data[yargmax,xargmax+1:] - hm).argmin()
        ly = abs(self.data[:yargmax-1,xargmax] - hm).argmin()
        ry = abs(self.data[yargmax+1:,xargmax] - hm).argmin()
        return abs(lx - rx), abs(ly-ry)

class TwoDimDonutPeak(OneDimFit):
    """Framework for calculating the fit for a two-dimensional (x,y,z) set of
        data that has a peak and is a donut/ring. An example of this is CoPd
        data collected in transmission geometry. This class is the child of the
        OneDimFit class that has a FWHM estimation fitting routine for 2d rings
        /donuts. Depending on the function, the FWHM estimated needs to be
        multipled by a factor to get to the correct parameter. This class is
        expected to be used as a basis for a two-dimensional fit of a ringed
        peak.  The functions fit_function() and guess_parameters() need to be
        filled up by the child class.
    """
    def __init__(self, *args, **kwargs):
        OneDimFit.__init__(self, *args, **kwargs)

    def estimate_fwhm(self, data):
        """ Estimate a FWHM peak. This returns a reasonably good estimate of the
            FWHM of a 1d peak.
        """
        # I have to reimplement estimate_fwhm for donut data because it's essentially 1d data.  This estimate_fwhm takes an argument of the estimate rather than looking up the data in self.data
        hm = (data.max() - data.min())/2
        argmax = data.argmax()
        xvals = np.arange(len(data))
        if argmax == len(self.xvals):
            argmax = argmax - 1
        elif argmax == 0:
            argmax = 1

        absdiff = abs(data - hm)
        left = xvals[absdiff[np.argwhere(xvals < argmax)].argmin()]
        right = xvals[absdiff[np.argwhere(xvals > argmax)].argmin()]

        return abs(left-right)

class Linear(OneDimFit):
    """ fit a function to a decay exponent.  This fits the function:
        f(x) = a * x + b
    """
    def __init__(self, data, mask=None):
        OneDimFit.__init__(self, data, mask)
        self.functional = "a*x + b"
        self.params_map ={ 0:"a", 1:"b"}

    def fit_function(self):
        a, b = self.params
        return a * self.xvals + b

    def guess_parameters(self):
        self.params = np.zeros(2)
        self.params[0] = ( self.data[0] - self.data[-1] ) * 2 / (self.data[0] + self.data[-1])
        self.params[1] = self.data.mean()

class DecayExpBetaSq(OneDimFit):
    """ fit a function to a (decay exponent with a beta parameter)^2.  This fits the function:
        f(x) = a + b exp(-(x/tf)^beta)^2
    """
    def __init__(self, data, mask=None):
        OneDimFit.__init__(self, data, mask)
        self.functional = "a + b exp(-(t/tf)^beta)^2"
        self.params_map ={ 0:"a", 1:"b", 2:"tf", 3:"beta" }

    def fit_function(self):
        a, b, tf, beta = self.params
        # checks to return high numbers if parameters are getting out of hand.
        if (tf<0 or beta<0): return self.try_again

        return a + b * (np.exp(-1*(self.xvals/tf)**beta)**2)

    def guess_parameters(self):
        self.params = np.zeros(4)
        self.params[0] = self.data[-1]
        self.params[1] = self.data[0] - self.params[0]
        self.params[2] = self.xvals[int(self.npoints/2)]
        self.params[3] = 1.5

class DecayExpBeta(OneDimFit):
    """ fit a function to a decay exponent with a beta parameter.  This fits the function:
        f(x) = a + b exp(-(x/tf)^beta)
    """
    def __init__(self, data, mask=None):
        OneDimFit.__init__(self, data, mask)
        self.functional = "a + b exp(-1*(x/tf)**beta)"
        self.params_map ={ 0:"a", 1:"b", 2:"tf", 3:"beta" }

    def fit_function(self):
        a, b, tf, beta = self.params
        # checks to return high numbers if parameters are getting out of hand.
        if (tf<0 or beta<0): return self.try_again

        return a + b * (np.exp(-1*(self.xvals/tf)**beta))

    def guess_parameters(self):
        self.params = np.zeros(4)
        self.params[0] = self.data[-1]
        self.params[1] = self.data[0] - self.params[0]
        self.params[2] = self.xvals[int(self.npoints/2)]
        self.params[3] = 1.5

class DecayExp(OneDimFit):
    """ fit a function to a decay exponent.  This fits the function:
        f(x) = a + b exp(-(x/tf))
    """
    def __init__(self, data, mask=None):
        OneDimFit.__init__(self, data, mask)
        self.functional = "a + b exp(-1*(x/tf))"
        self.params_map ={ 0:"a", 1:"b", 2:"tf" }

    def fit_function(self):
        a, b, tf = self.params
        # checks to return high numbers if parameters are getting out of hand.
        if ( tf<0 ): return self.try_again

        return a + b * np.exp(-1*(self.xvals/tf))

    def guess_parameters(self):
        self.params = np.zeros(3)
        self.params[0] = self.data[-1]
        self.params[1] = self.data[0] - self.params[0]
        self.params[2] = self.xvals[int(self.npoints/2)]

class Gaussian(OneDimPeak):
    """ fit a function to a 1d Gaussian.  This fits the function:
        f(x) = a exp(-(x-x0)^2/(2w^2)) + shift
    """
    def __init__(self, data, mask=None):
        OneDimPeak.__init__(self, data, mask)
        self.functional = "a exp(-(x-x0)^2/(2*w^2)) + shift"
        self.params_map ={ 0:"a", 1:"x0", 2:"w" , 3:"shift"}

    def fit_function(self):
        a, x0, w, shift = self.params
        if ( w < 0 ): return self.try_again
        return a*shape.gaussian(self.data.shape, (w,), center=(x0,)) + shift

    def guess_parameters(self):
        self.params = np.zeros(4)
        # average the first and last points to try to guess the background
        self.params[3] = (self.data[0] + self.data[-1]) / 2.0
        self.params[0] = self.data.max() - self.params[3]
        self.params[1] = self.xvals[self.data.argmax()]
        
        # width is always some function-dependent multiple of fwhm
        self.params[2] = self.estimate_fwhm()/2.35 #2.35 = 1/(2*sqrt(2*ln(2)))

class Lorentzian(OneDimPeak):
    """ fit a function to a 1d Lorentzian.  This fits the function:
        f(x) = a/( ((x-x0)/w)^2 + 1) + bg
    """
    # from http://mathworld.wolfram.com/LorentzianFunction.html
    def __init__(self, data, mask=None):
        OneDimPeak.__init__(self, data, mask)
        self.functional = "a/( ((x-x0)/w)^2 + 1) + shift"
        self.params_map ={ 0:"a", 1:"x0", 2:"w", 3:"shift"}

    def fit_function(self):
        a, x0, w, shift = self.params
        # checks to return high numbers if parameters are getting out of hand.
        if ( w < 0 ): return self.try_again
        return a*shape.lorentzian(self.data.shape, (w,), center=(x0,)) + shift

    def guess_parameters(self):
        self.params = np.zeros(4)
        # average the first and last points to try to guess the background
        self.params[3] = (self.data[0] + self.data[-1]) / 2.0
        self.params[1] = self.xvals[self.data.argmax()]
        
        # width is always some function-dependent multiple of fwhm
        self.params[2] = self.estimate_fwhm()/1.29 #1.29 = 1/(2*sqrt(sqrt(2)-1))
        self.params[0] = (self.data.max() - self.params[3])*(np.pi*self.params[2])

class LorentzianSq(OneDimPeak):
    """ fit a function to a 1d squared lorentzian.  This fits the function:
        f(x) = a/( ((x-x0)/w)^2 + 1)^2 + bg
    """
    def __init__(self, data, mask=None):
        OneDimPeak.__init__(self, data, mask)
        self.functional = "a/((x-x0)^2 + w^2)^2 + bg"
        self.params_map ={ 0:"a", 1:"x0", 2:"w", 3:"bg"}

    def fit_function(self):
        a, x0, w, bg = self.params
        # checks to return high numbers if parameters are getting out of hand.
        #if ( gam > 0 ): return np.ones_like(self.xvals) * self.try_again
        if w < 0: return self.try_again
        return a*shape.lorentzian(self.data.shape, (w,), center=(x0,))**2 + bg

    def guess_parameters(self):
        self.params = np.zeros(4)
        
        self.params[3] = abs(self.data).min()              # bg
        self.params[0] = self.data.max() - self.params[3]  # scale
        self.params[1] = self.xvals[self.data.argmax()]    # center x0
        
        # width is always some function-dependent multiple of fwhm
        self.params[2] = self.estimate_fwhm()/1.29 #1.29 = 1/(2*sqrt(sqrt(2)-1))
        
class LorentzianSqBlurred(OneDimPeak):
    """ fit a function to a 1d squared lorentzian convolved with a gaussian.  This
    fits the function:
        f(x) = a*convolve( 1/( ((x-x0)/w)^2 + 1)^2, exp(-(x-x0)^2/(2w^2) ) + bg
    """

    def __init__(self, data, mask=None):
        OneDimPeak.__init__(self, data, mask)
        self.functional = "a*convolve(lorentzian(x0,lw)**2,gaussian(gw))+bg"
        self.params_map ={ 0:"a", 1:"x0", 2:"lw", 3:"bg", 4:"gw"}
        
    def fit_function(self):
        a, x0, lw, bg, gw = self.params # gw is gaussian width
        # checks to return high numbers if parameters are getting out of hand.
        #if ( gam > 0 ): return np.ones_like(self.xvals) * self.try_again
        if lw < 0 or gw < 0: return self.try_again

        l = shape.lorentzian(self.data.shape, (lw,), center=(x0,))**2
        g = np.fft.fftshift(shape.gaussian(self.data.shape, (gw,), normalization=1.0))
        f = self._convolve(l,g)

        return a*f+bg
    
    def _convolve(self,x,y):
        return np.fft.ifft(np.fft.fft(x)*np.fft.fft(y)).real

    def guess_parameters(self):
        self.params = np.zeros(5)
        
        self.params[3] = abs(self.data).min()              # bg
        self.params[0] = self.data.max() - self.params[3]  # scale
        self.params[1] = self.xvals[self.data.argmax()]    # center x0
        
        # width is always some function-dependent multiple of fwhm
        self.params[2] = self.estimate_fwhm()/1.29 #1.29 = 1/(2*sqrt(sqrt(2)-1))
        
        # hard to make a case for the convolver width! just set it to 1 and see what happens
        self.params[4] = 4.

class Gaussian2D(TwoDimPeak):
    """ fit a function to a two-dimensional Gaussian.  This fits the function:
        f(x) = a exp(-(x-x0)^2/(2*sigma_x^2) - (y-y0)^2/(2*sigma_y^2)) + shift
    """
    def __init__(self, data, mask=None):
        TwoDimPeak.__init__(self, data, mask)
        self.functional = "a exp(-(x-x0)^2/(2*sigmay^2) - (y-y0)^2/(2*sigmay^2)) + shift"
        self.params_map ={ 0:"a", 1:"x0", 2:"sigmax", 3:"y0", 4:"sigmay", 5:"shift"}

    def fit_function(self):
        a, x0, sigmax, y0, sigmay, shift = self.params
        if ( sigmax < 0 or sigmay < 0): return self.try_again
        return shape.gaussian(self.data.shape, (sigmay, sigmax), center=(y0, x0))*a + shift

    def guess_parameters(self):
        self.params = np.zeros(6)
        sd = self.data
        sp = self.params
        # average the outside corner values
        sp[5] = np.average(np.concatenate((sd[:,0], sd[:,-1], sd[-1,:], sd[0,:])))
        sp[0] = sd.max() - sp[5]
        sp[3], sp[1] = np.unravel_index(sd.argmax(), sd.shape)
        sp[2], sp[4] = self.estimate_fwhm()
        sp[2] = sp[2]/(2*np.sqrt(2*np.log(2)))
        sp[4] = sp[4]/(2*np.sqrt(2*np.log(2)))

class Lorentzian2D(TwoDimPeak):
    """ fit a function to a 2d-lorentzian.  This fits the function:
        f(x) = a/(((x-x0)/wx)^2 + ((y-y0)/wy)^2 + 1) + bg
    """
    def __init__(self, data, mask=None):
        TwoDimPeak.__init__(self, data, mask)
        self.functional = "a / ( ((x-x0)/xw)**2 + ((y-y0)/yw)**2 + 1) + shift "
        self.params_map ={ 0:"a", 1:"x0", 2:"xw", 3:"y0", 4:"yw", 5:"shift"}

    def fit_function(self):
        a, x0, xw, y0, yw, shift = self.params
        if ( xw < 0 or yw < 0): return self.try_again
        return a*shape.lorentzian(self.data.shape, (yw, xw), center=(y0, x0)) + shift

    def guess_parameters(self):
        self.params = np.zeros(6)
        sd = self.data
        sp = self.params
        # average the outside corner values
        sp[5] = np.average(np.concatenate((sd[:,0], sd[:,-1], sd[-1,:], sd[0,:])))
        sp[0] = sd.max() - sp[5]
        sp[3], sp[1] = np.unravel_index(sd.argmax(), sd.shape)
        sp[2], sp[4] = self.estimate_fwhm()
        sp[2] = sp[2]/2
        sp[4] = sp[4]/2

class Lorentzian2DSq(Lorentzian2D):
    """ fit a function to a 2d-lorentzian squared.  This fits the function:
        f(x) = a/(((x-x0)/wx)^2 + ((y-y0)/wy)^2 + 1)^2 + bg
    """
    def __init__(self, data, mask=None):
        Lorentzian2D.__init__(self, data, mask)
        self.functional = "a / ( ((x-x0)/xw)**2 + ((y-y0)/yw)**2 + 1)**2 + shift"
        self.params_map ={ 0:"a", 1:"x0", 2:"xw", 3:"y0", 4:"yw", 5:"shift"}

    def fit_function(self):
        a, x0, xw, y0, yw, shift = self.params
        if ( xw < 0 or yw < 0): return self.try_again
        return a*(shape.lorentzian(self.data.shape, (yw, xw), center=(y0, x0)))**2 + shift

class GaussianDonut2D(TwoDimDonutPeak):
    """ Fit a function to a two-dimensional Gaussian donut.  This fits the a
    radial function:
        f(r) = a exp(-(r-R)^2/(2*sigma_r^2) + shift
    where r is an (yc, xc) center, R is a raidius, and sigma_r is the standard
    deviation.
    """
    def __init__(self, data, mask=None):
        TwoDimDonutPeak.__init__(self, data, mask)
        self.functional = "f(r) = a exp(-(r-R)^2/(2*sigma_r^2) + shift"
        self.params_map ={ 0:"a", 1:"r_x", 2:"r_y", 3:"R", 4:"sigma_r", 5:"shift"}

    def fit_function(self):
        a, rx, ry, R, sigmar, shift = self.params
        if ( sigmar < 0 or R < 0): return self.try_again
        return shift + a*np.exp(-1*(shape.radial(self.data.shape, center=(ry, rx)) -R)**2/(2*sigmar**2))

    def guess_parameters(self):
        self.params = np.zeros(6)
        sd = self.data
        sp = self.params
        # average the outside corner values
        sp[5] = np.average(np.concatenate((sd[:,0], sd[:,-1], sd[-1,:], sd[0,:])))
        sp[0] = sd.max() - sp[5]
        yc, xc = conditioning.find_center(sd, return_type='coords')
        sp[2], sp[1] = yc, xc
        Rmax = min(xc, yc, sd.shape[0] - yc, sd.shape[1] - xc)
        unw_sum = (wrapping.unwrap(self.data, (0, Rmax, (yc,xc) ))).sum(axis=1)
        sp[3] = unw_sum.argmax()
        sp[4] = self.estimate_fwhm(unw_sum)/(2*np.sqrt(2*np.log(2)))

class LorentzianDonut2D(TwoDimDonutPeak):
    """ Fit a function to a two-dimensional Lorentzian donut.  This fits a
    radial function of the form:
        f(r) = a/(((r-R)/rw)^2 + 1) + bg
    where r is an (yc, xc) center, R is a radius, and rw is the std. dev.
    """
    def __init__(self, data, mask=None):
        TwoDimDonutPeak.__init__(self, data, mask)
        self.functional = "f(r) = a/(((r-R)/rw)^2 + 1) + bg"
        self.params_map ={ 0:"a", 1:"r_x", 2:"r_y", 3:"R", 4:"rw", 5:"bg"}

    def fit_function(self):
        a, rx, ry, R, rw, bg = self.params
        if ( rw < 0 or R < 0): return self.try_again
        return bg + a / ( ((shape.radial(self.data.shape,center =(ry,rx))-R)/rw)**2 + 1)

    def guess_parameters(self):
        self.params = np.zeros(6)
        sd = self.data
        sp = self.params
        # average the outside corner values
        sp[5] = np.average(np.concatenate((sd[:,0], sd[:,-1], sd[-1,:], sd[0,:])))
        sp[0] = sd.max() - sp[5]
        yc, xc = conditioning.find_center(sd, return_type='coords')
        sp[2], sp[1] = yc, xc
        Rmax = min(xc, yc, sd.shape[0] - yc, sd.shape[1] - xc)
        unw_sum = (wrapping.unwrap(self.data, (0, Rmax, (yc,xc) ))).sum(axis=1)
        sp[3] = unw_sum.argmax()
        sp[4] = self.estimate_fwhm(unw_sum)/2

class LorentzianDonut2DSq(LorentzianDonut2D):
    """ Fit a function to a two-dimensional (Lorentzian donut)^2.  This fits a
    radial function:
        f(r) = a/(((r-R)/rw)^2 + 1)^2 + bg
    where r is an (yc, xc) center, R is a radius, and wr is the std. dev.
    """
    def __init__(self, data, mask=None):
        LorentzianDonut2D.__init__(self, data, mask)
        self.functional = "f(r) = a/(((r-R)/rw)^2 + 1)^2 + bg"
        self.params_map ={ 0:"a", 1:"r_x", 2:"r_y", 3:"R", 4:"rw", 5:"bg"}

    def fit_function(self):
        a, rx, ry, R, rw, bg = self.params
        if ( rw < 0 or R < 0): return self.try_again
        return bg + a / ( ((shape.radial(self.data.shape,center =(ry,rx))-R)/rw)**2 + 1)**2

def linear(data, mask=None):
    """ fit a function to a line.  This fits the function:
        f(x) = a * x + b
        
    arguments:
        data - Data to fit.  This should be a (N, 2) array of (xvalues, yvalues).
        mask - binary mask that tells the program where the data should be fit.
            This must be the same size as the data.  The default is None.

    returns:
        result - a fit class that contains the final fit.  This object has
            various self descriptive parameters, the most useful is
            result.final_params_errors, which contains a parameter+fit map of
            the final fitted values.
    """
    fit = Linear(data, mask)
    fit.fit()
    return fit

def decay_exp_beta_sq(data, mask=None):
    """ fit a function to a (decay exponent with a beta parameter)^2.  This fits the function:
        f(x) = a + b exp(-(x/tf)^beta)^2

    arguments:
        data - Data to fit.  This should be a (N, 2) array of (xvalues, yvalues).
        mask - binary mask that tells the program where the data should be fit.
            This must be the same size as the data.  The default is None.

    returns:
        result - a fit class that contains the final fit.  This object has
            various self descriptive parameters, the most useful is
            result.final_params_errors, which contains a parameter+fit map of
            the final fitted values.
    """
    fit = DecayExpBetaSq(data, mask)
    fit.fit()
    return fit

def decay_exp_beta(data, mask=None):
    """ fit a function to a decay exponent with a beta parameter.  This fits the function:
        f(x) = a + b exp(-(x/tf)^beta)

    arguments:
        data - Data to fit.  This should be a (N, 2) array of (xvalues, yvalues).
        mask - binary mask that tells the program where the data should be fit.
            This must be the same size as the data.  The default is None.

    returns:
        result - a fit class that contains the final fit.  This object has
            various self descriptive parameters, the most useful is
            result.final_params_errors, which contains a parameter+fit map of
            the final fitted values.
    """
    fit = DecayExpBeta(data, mask)
    fit.fit()
    return fit

def decay_exp(data, mask=None):
    """ fit a function to a decay exponent.  This fits the function:
        f(x) = a + b exp(-(x/tf))

    arguments:
        data - Data to fit.  This should be a (N, 2) array of (xvalues, yvalues).
        mask - binary mask that tells the program where the data should be fit.
            This must be the same size as the data.  The default is None.

    returns:
        result - a fit class that contains the final fit.  This object has
            various self descriptive parameters, the most useful is
            result.final_params_errors, which contains a parameter+fit map of
            the final fitted values.
    """
    fit = DecayExp(data, mask)
    fit.fit()
    return fit

def gaussian(data, mask=None):
    """ fit a function to a Gaussian.  This fits the function:
        f(x) = a exp(-(x-x0)^2/(2w^2)) + shift

    arguments:
        data - Data to fit.  This should be a (N, 2) array of (xvalues, yvalues).
        mask - binary mask that tells the program where the data should be fit.
            This must be the same size as the data.  The default is None.

    returns:
        result - a fit class that contains the final fit.  This object has
            various self descriptive parameters, the most useful is
            result.final_params_errors, which contains a parameter+fit map of
            the final fitted values.
    """
    fit = Gaussian(data, mask)
    fit.fit()
    return fit

def lorentzian(data, mask=None):
    """ fit a function to a Lorentzian.  This fits the function:
        f(x) = a/( ((x-x0)/w)^2 + 1) + bg

    arguments:
        data - Data to fit.  This should be a (N, 2) array of (xvalues, yvalues).
        mask - binary mask that tells the program where the data should be fit.
            This must be the same size as the data.  The default is None.

    returns:
        result - a fit class that contains the final fit.  This object has
            various self descriptive parameters, the most useful is
            result.final_params_errors, which contains a parameter+fit map of
            the final fitted values.
    """
    fit = Lorentzian(data, mask)
    fit.fit()
    return fit

def lorentzian_sq(data, mask=None):
    """ fit a function to a squared lorentzian.  This fits the function:
        f(x) = a/( ((x-x0)/w)^2 + 1)^2 + bg

    arguments:
        data - Data to fit.  This should be a (N, 2) array of (xvalues, yvalues).
        mask - binary mask that tells the program where the data should be fit.
            This must be the same size as the data.  The default is None.

    returns:
        result - a fit class that contains the final fit.  This object has
            various self descriptive parameters, the most useful is
            result.final_params_errors, which contains a parameter+fit map of
            the final fitted values.
    """
    fit = LorentzianSq(data, mask)
    fit.fit()
    return fit

def lorentzian_sq_blurred(data, mask=None):
    """ fit a function to a squared lorentzian convolved with a gaussian. This is
    the most physically plausible lineshape for fitting labyrinth-type speckle
    patterns. This fits the function:

       f(x) = a*convolve( 1/( ((x-x0)/w)^2 + 1)^2, exp(-(x-x0)^2/(2w^2) ) + bg
    
    arguments:
        data - Data to fit.  This should be a (N, 2) array of (xvalues, yvalues).
        mask - binary mask that tells the program where the data should be fit.
            This must be the same size as the data.  The default is None.

    returns:
        result - a fit class that contains the final fit.  This object has
            various self descriptive parameters, the most useful is
            result.final_params_errors, which contains a parameter+fit map of
            the final fitted values.
    """
    fit = LorentzianSqBlurred(data, mask)
    fit.fit()
    return fit

def gaussian_2d(data, mask=None):
    """ fit a function to a two-dimensional Gaussian.  This fits the function:
        f(x) = a exp(-(x-x0)^2/(2*sigma_x^2) - (y-y0)^2/(2*sigma_y^2)) + shift

    arguments:
        data - Data to fit.  This should be a 2-dimensional array.
        mask - binary mask that tells the program where the data should be fit.
            This must be the same size as the data.  The default is None.

    returns:
        result - a fit class that contains the final fit.  This object has
            various self descriptive parameters, the most useful is
            result.final_params_errors, which contains a parameter+fit map of
            the final fitted values.
    """
    fit = Gaussian2D(data, mask)
    fit.fit()
    return fit

def gaussian_donut_2d(data, mask=None):
    """ fit a function to a two-dimensional Gaussian donut/ring. This fits a
    radial function:
        f(r) = a exp(-(r-R)^2/(2*sigma_r^2) + shift
    where r is an (yc, xc) center, R is a radius, and sigma_r is the standard
    deviation.

    arguments:
        data - Data to fit.  This should be a 2-dimensional array.
        mask - binary mask that tells the program where the data should be fit.
            This must be the same size as the data.  The default is None.

    returns:
        result - a fit class that contains the final fit.  This object has
            various self descriptive parameters, the most useful is
            result.final_params_errors, which contains a parameter+fit map of
            the final fitted values.
    """
    fit = GaussianDonut2D(data, mask)
    fit.fit()
    return fit

def lorentzian_donut_2d(data, mask=None):
    """ fit a function to a two-dimensional Lorentzian donut/ring. This fits a
    radial function of the form:
        f(r) = a/(((r-R)/rw)^2 + 1) + bg
    where r is an (yc, xc) center, R is a radius, and rw is the std. dev.

    arguments:
        data - Data to fit.  This should be a 2-dimensional array.
        mask - binary mask that tells the program where the data should be fit.
            This must be the same size as the data.  The default is None.

    returns:
        result - a fit class that contains the final fit.  This object has
            various self descriptive parameters, the most useful is
            result.final_params_errors, which contains a parameter+fit map of
            the final fitted values.
    """
    fit = LorentzianDonut2D(data, mask)
    fit.fit()
    return fit

def lorentzian_donut_2d_sq(data, mask=None):
    """ fit a function to a two-dimensional (Lorentzian donut/ring)^2. This fits a
    radial function of the form:
        f(r) = a/(((r-R)/rw)^2 + 1) + bg
    where r is an (yc, xc) center, R is a radius, and rw is the std. dev.

    arguments:
        data - Data to fit.  This should be a 2-dimensional array.
        mask - binary mask that tells the program where the data should be fit.
            This must be the same size as the data.  The default is None.

    returns:
        result - a fit class that contains the final fit.  This object has
            various self descriptive parameters, the most useful is
            result.final_params_errors, which contains a parameter+fit map of
            the final fitted values.
    """
    fit = LorentzianDonut2DSq(data, mask)
    fit.fit()
    return fit

def lorentzian_2d(data, mask=None):
    """ fit a function to a 2d-lorentzian.  This fits the function:
        f(x) = a/(((x-x0)/wx)^2 + ((y-y0)/wy)^2 + 1) + bg

    arguments:
        data - Data to fit.  This should be a 2-dimensional array.
        mask - binary mask that tells the program where the data should be fit.
            This must be the same size as the data.  The default is None.

    returns:
        result - a fit class that contains the final fit.  This object has
            various self descriptive parameters, the most useful is
            result.final_params_errors, which contains a parameter+fit map of
            the final fitted values.
    """
    fit = Lorentzian2D(data, mask)
    fit.fit()
    return fit

def lorentzian_2d_sq(data, mask=None):
    """ fit a function to a 2d-lorentzian squared.  This fits the function:
        f(x) = a/(((x-x0)/wx)^2 + ((y-y0)/wy)^2 + 1)^2 + bg

    arguments:
        data - Data to fit.  This should be a 2-dimensional array.
        mask - binary mask that tells the program where the data should be fit.
            This must be the same size as the data.  The default is None.

    returns:
        result - a fit class that contains the final fit.  This object has
            various self descriptive parameters, the most useful is
            result.final_params_errors, which contains a parameter+fit map of
            the final fitted values.
    """
    fit = Lorentzian2DSq(data, mask)
    fit.fit()
    return fit

def gaussian_3d(data, mask=None):
    """ fit a function to a three-dimensional Gaussian.  This fits the function:
        f(x) = a exp(-(x-x0)^2/(2*sigma_x^2) - (y-y0)^2/(2*sigma_y^2)) + shift

    arguments:
        data - Data to fit.  This should be a 2-dimensional array.
        mask - binary mask that tells the program where the data should be fit.
            This must be the same size as one frame of the data.  The default is
            None.

    returns:
        A dictionary of fitted results.  The dictionary is indexed by the frame
            number.
    """
    return _3d_fit(data, Gaussian2D, mask)

def lorentzian_3d(data, mask=None):
    """ fit a function to 3d-lorentzian.  This fits the function:
        f(x) = a/(((x-x0)/wx)^2 + ((y-y0)/wy)^2 + 1) + bg

    For 3d data, a 2D Lorentzian is fitted for each frame.

    arguments:
        data - Data to fit.  This should be a 2-dimensional array.
        mask - binary mask that tells the program where the data should be fit.
            This must be the same size as one frame of the data.  The default is
            None.

    returns:
        A dictionary of fitted results.  The dictionary is indexed by the frame
            number.
    """
    return _3d_fit(data, Lorentzian2D, mask)

def lorentzian_3d_sq(data, mask=None):
    """ fit a function to a 2d-lorentzian squared.  This fits the function:
        f(x) = a/(((x-x0)/wx)^2 + ((y-y0)/wy)^2 + 1)^2 + bg

    arguments:
        data - Data to fit.  This should be a 2-dimensional array.
        mask - binary mask that tells the program where the data should be fit.
            This must be the same size as one frame of the data.  The default is
            None.

    returns:
        A dictionary of fitted results.  The dictionary is indexed by the frame
            number.
    """
    return _3d_fit(data, Lorentzian2DSq, mask)

def _3d_fit(data, FitClass, mask=None):
    """ function that will fit a 3d array along the 0-axis.  You provide the
        data (must be 3d) and the FitClass, which is a 2D fit class.
    """
    assert isinstance(data, np.ndarray) and data.ndim == 3, "data must be a three-dimensional numpy array."
    (fr, ys, xs) = data.shape

    result = {}
    for f in range(fr):
        fit = FitClass(data[f], mask)
        fit.fit()
        result[f] = fit
    return result

