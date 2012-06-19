#functional fitting
#	ND
#	arbitrary functions
#	common functions
#		gaussian, lorentzian, error function, exponentials
import numpy as np
from scipy.optimize import leastsq

class OneDimFit():
    def __init__(self, data, mask=None):
        # These parameters should be filled out by the child class.
        self.functional = "f(x) = a*x + b"
        self.params_map ={ 0:"a", 1:"b"}

        # prepare the data
        assert isinstance(data, np.ndarray), "data must be an array"
        assert data.ndim == 2, "data must be two-dimensional"

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
        else:
            # We have an image.
            self.xvals = np.arange(xs)
            self.yvals = np.arange(ys)
            self.npoints = len(self.xvals)*len(self.yvals)
            self.data = data
            self.try_again = np.ones_like(self.data) * 1e30

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
        """ Calculate the residuals: data - fit_function(x).  params is an optional parameter array. If params is not specified, it looks for the array in the class.
        """
        if params is not None:
            self.params = params
        return np.ravel((self.data - self.fit_function())*self.mask)

    def fit(self):
        """ Fit data using fit_function().  This calls scipy.optimize.leastsq().
        """
        try:
            if not isinstance(self.params, np.ndarray):
                self.guess_parameters()
        except AttributeError:
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
            self.final_params_errors = np.sqrt(np.diagonal(cov_x))*np.sqrt(self.final_chisq/(self.npoints - len(self.params)))
        except ValueError:
            print "Error: The fit did not converge. Please change initial parameters and try again"
            self.final_params_errors = np.zeros_like(self.params)

        if ier not in (1,2,3,4):
            # read somewhere that ier = 1, 2, 3, 4 also gives correct solution. http://bugs.debian.org/cgi-bin/bugreport.cgi?bug=474930
            print "Error: No solution found!"
            self.final_params_errors = np.zeros_like(self.params)
        
    def format_results(self, header=True, outfile=None):
        """ Format the final, fitting data for printing or writing to disk.

        arguments:
            header - weather or not to print the header.  Defaults to True
            outfile - If an outfile is specified, this function will write the parameters to outfile on disk.

        returns:
            the formatted results string.
        """
        if len(self.final_params) == len(self.final_params_errors):
            pass

        fun = "# fitting functional %s\n" % self.functional
        hdr = "# "
        hdr_e = ""
        outstr = ""
        outstr_e = ""
        for i in range(len(self.final_params)):
            hdr += "%s\t" % self.params_map[i]
            hdr_e += "error(%s)\t" % self.params_map[i]
            outstr += "%1.7e\t" % self.final_params[i]
            outstr_e += "%1.7e\t" % self.final_params_errors[i]

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

class OneDimPeak(OneDimFit):
    """ Child of the OneDimFit class that has a FWHM estimation fitting routine.
    Depending on the function, the FWHM estimated needs to be multipled by a factor to get to the correct parameter.
    """
    def __init__(self, *args, **kwargs):
        OneDimFit.__init__(self, *args, **kwargs)

    def estimate_fwhm(self):
        # a more accurate estimate can be obtained by doing some sort of linear interpolation in the neighborhood of the left and right solution but this seems like overkill.
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
    """ Child of the OneDimFit class that has a FWHM estimation fitting routine created for two dimensional
    Depending on the function, the FWHM estimated needs to be multipled by a factor to get to the correct parameter.
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

def linear(data):
    """ fit a function to a line.  This fits the function:
        f(x) = a * x + b
        
    arguments:
        data - Data to fit.  This should be a (N, 2) array of (xvalues, yvalues).
        
    returns:
        result - a fit class that contains the final fit.  This object has various
        self-descriptive parameters, the most useful of which are result.final_params
        and result.final_params_errors.
    """
    class Linear(OneDimFit):
        def __init__(self, data):
            OneDimFit.__init__(self, data)
            self.functional = "a*x + b"
            self.params_map ={ 0:"a", 1:"b"}

        def fit_function(self):
            a, b = self.params
            return a * self.xvals + b

        def guess_parameters(self):
            self.params = np.zeros(2)
            self.params[0] = ( self.data[0] - self.data[-1] ) * 2 / (self.data[0] + self.data[-1])
            self.params[1] = self.data.mean()

    fit = Linear(data)
    fit.fit()
    return fit

def decay_exp_beta_sq(data):
    """ fit a function to a (decay exponent with a beta parameter)^2.  This fits the function:
        f(x) = a + b exp(-(x/tf)^beta)^2

    arguments:
        data - Data to fit.  This should be a (N, 2) array of (xvalues, yvalues).

    returns:
        result - a fit class that contains the final fit.  This object has various
        self-descriptive parameters, the most useful of which are result.final_params
        and result.final_params_errors.
    """
    class DecayExpBetaSq(OneDimFit):
        def __init__(self, data):
            OneDimFit.__init__(self, data)
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


    fit = DecayExpBetaSq(data)
    fit.fit()
    return fit

def decay_exp_beta(data):
    """ fit a function to a decay exponent with a beta parameter.  This fits the function:
        f(x) = a + b exp(-(x/tf)^beta)

    arguments:
        data - Data to fit.  This should be a (N, 2) array of (xvalues, yvalues).

    returns:
        result - a fit class that contains the final fit.  This object has various
        self-descriptive parameters, the most useful of which are result.final_params
        and result.final_params_errors.
    """
    class DecayExpBeta(OneDimFit):
        def __init__(self, data):
            OneDimFit.__init__(self, data)
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

    fit = DecayExpBeta(data)
    fit.fit()
    return fit

def decay_exp(data):
    """ fit a function to a decay exponent.  This fits the function:
        f(x) = a + b exp(-(x/tf))

    arguments:
        data - Data to fit.  This should be a (N, 2) array of (xvalues, yvalues).

    returns:
        result - a fit class that contains the final fit.  This object has various
        self-descriptive parameters, the most useful of which are result.final_params
        and result.final_params_errors.
    """
    class DecayExp(OneDimFit):

        def __init__(self, data):
            OneDimFit.__init__(self, data)
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

    fit = DecayExp(data)
    fit.fit()
    return fit

def gaussian(data):
    """ fit a function to a Gaussian.  This fits the function:
        f(x) = a exp(-(x-b)^2/(2c^2)) + shift

    arguments:
        data - Data to fit.  This should be a (N, 2) array of (xvalues, yvalues).

    returns:
        result - a fit class that contains the final fit.  This object has various
        self-descriptive parameters, the most useful of which are result.final_params
        and result_final_params_errors.
    """
    class Gaussian(OneDimPeak):

        def __init__(self, data):
            OneDimPeak.__init__(self, data)
            self.functional = "a exp(-(x-b)^2/(2*c^2)) + shift"
            self.params_map ={ 0:"a", 1:"b", 2:"c" , 3:"shift"}

        def fit_function(self):
            a, b, c, shift = self.params
            if ( c < 0 ): return self.try_again
            return a*np.exp(-(self.xvals-b)**2/(2*c**2)) + shift

        def guess_parameters(self):
            self.params = np.zeros(4)
            # average the first and last points to try to guess the background
            self.params[3] = (self.data[0] + self.data[-1]) / 2.0
            self.params[0] = self.data.max() - self.params[3]
            self.params[1] = self.xvals[self.data.argmax()]
            
            # width is always some function-dependent multiple of fwhm
            self.params[2] = self.estimate_fwhm()/2.35 #2.35 = 1/(2*sqrt(2*ln(2)))

    fit = Gaussian(data)
    fit.fit()
    return fit

def lorentzian(data):
    """ fit a function to a Lorentzian.  This fits the function:
        f(x) = a/((x-x0)^2 + (w/2.0)^2) + shift

    arguments:
        data - Data to fit.  This should be a (N, 2) array of (xvalues, yvalues).

    returns:
        result - a fit class that contains the final fit.  This object has various
        self-descriptive parameters, the most useful of which are result.final_params
        and resul_final_params_errors.
    """
    class Lorentzian(OneDimPeak):
        # from http://mathworld.wolfram.com/LorentzianFunction.html
        def __init__(self, data):
            OneDimPeak.__init__(self, data)
            self.functional = "a/((x-x0)^2 + w^2) + shift"
            self.params_map ={ 0:"a", 1:"x0", 2:"w", 3:"shift"}

        def fit_function(self):
            a, x0, w, shift = self.params
            # checks to return high numbers if parameters are getting out of hand.
            if ( w < 0 ): return self.try_again
            return a/((self.xvals-x0)**2 + w**2) + shift

        def guess_parameters(self):
            self.params = np.zeros(4)
            # average the first and last points to try to guess the background
            self.params[3] = (self.data[0] + self.data[-1]) / 2.0
            self.params[1] = self.xvals[self.data.argmax()]
            
            # width is always some function-dependent multiple of fwhm
            self.params[2] = self.estimate_fwhm()/1.29 #1.29 = 1/(2*sqrt(sqrt(2)-1))
            self.params[0] = (self.data.max() - self.params[3])*(np.pi*self.params[2])

    fit = Lorentzian(data)
    fit.fit()
    return fit

def lorentzian_sq(data):
    """ fit a function to a squared lorentzian.  This fits the function:
        f(x) = a/((x-x0)^2 + w^2)^2 + bg

    arguments:
        data - Data to fit.  This should be a (N, 2) array of (xvalues, yvalues).

    returns:
        result - a fit class that contains the final fit.  This object has various
        self-descriptive parameters, the most useful of which are result.final_params
        and result_final_params_errors.
    """
    class LorentzianSq(OneDimPeak):

        def __init__(self, data):
            OneDimPeak.__init__(self, data)
            self.functional = "a/((x-x0)^2 + w^2)^2 + bg"
            self.params_map ={ 0:"a", 1:"x0", 2:"w", 3:"bg"}

        def fit_function(self):
            a, x0, w, bg = self.params
            # checks to return high numbers if parameters are getting out of hand.
            #if ( gam > 0 ): return np.ones_like(self.xvals) * self.try_again
            return a/((self.xvals-x0)**2./w**2.+1)**2.+bg

        def guess_parameters(self):
            self.params = np.zeros(4)
            
            self.params[3] = abs(self.data).min()              # bg
            self.params[0] = self.data.max() - self.params[3]  # scale
            self.params[1] = self.xvals[self.data.argmax()]    # center x0
            
            # width is always some function-dependent multiple of fwhm
            self.params[2] = self.estimate_fwhm()/1.29 #1.29 = 1/(2*sqrt(sqrt(2)-1))

    fit = LorentzianSq(data)
    fit.fit()
    return fit

def gaussian_2d(data):
    """ fit a function to a two-dimensional Gaussian.  This fits the function:
        f(x) = a exp(-(x-x0)^2/(2*sigma_x^2) - (y-y0)^2/(2*sigma_y^2)) + shift

        result - a fit class that contains the final fit.  This object has various self descriptive parameters, the most useful are result.final_params and resul_final_params_errors.
    """
    class Gaussian2D(TwoDimPeak):
        def __init__(self, data):
            TwoDimPeak.__init__(self, data)
            self.functional = "a exp(-(x-x0)^2/(2*sigmay^2) - (y-y0)^2/(2*sigmay^2)) + shift"
            self.params_map ={ 0:"a", 1:"x0", 2:"sigmax", 3:"y0", 4:"sigmay", 5:"shift"}

        def fit_function(self):
            a, x0, sigmax, y0, sigmay, shift = self.params
            if ( sigmax < 0 or sigmay < 0): return self.try_again
            gauss = lambda val, x0, sx: np.exp(-(self.xvals-x0)**2/(2*sx**2))
            res = np.outer(gauss(self.xvals, x0, sigmax), gauss(self.yvals, y0, sigmay) )
            return res*(a - shift)/res.max() + shift

        def guess_parameters(self):
            self.params = np.zeros(6)
            sd = self.data
            sp = self.params
            # average the outside corner values
            sp[5] = np.average(np.concatenate((sd[:,0], sd[:,-1], sd[-1,:], sd[0,:])))
            sp[0] = sd.max() - sp[5]
            sp[1], sp[3] = np.unravel_index(sd.argmax(), sd.shape)
            sp[2], sp[4] = self.estimate_fwhm()
            sp[2] = sp[2]/(2*np.sqrt(2*np.log(2)))
            sp[4] = sp[4]/(2*np.sqrt(2*np.log(2)))


    fit = Gaussian2D(data)
    fit.fit()
    return fit

