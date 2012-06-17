#functional fitting
#	ND
#	arbitrary functions
#	common functions
#		gaussian, lorentzian, error function, exponentials
import numpy as np
from scipy.optimize import leastsq

class OneDimFit():
    def __init__(self, data):
        self.try_again = 1.0e30
        # These parameters should be filled out by the child class.
        self.functional = "f(x) = a*x + b"
        self.params_map ={ 0:"a", 1:"b"}

        self.data = data

        # prepare the data
        assert isinstance(self.data, np.ndarray), "data must be an array"
        assert self.data.ndim == 2, "data must be two dimensional"
        ys, xs = self.data.shape
        assert ys == 2 or xs == 2, "data must be column data"

        # Data should always be in the shape (npoints, 2)
        if xs == 2:
            self.npoints = ys
        else:
            self.npoints = xs
            self.data = self.data.swapaxes(0,1)
        self.xdata = self.data[:, 0]
        self.ydata = self.data[:, 1]

    def fit_function(self):
        """ This needs to be filled out by the super() function
        """
        pass

    def guess_parameters(self):
        """ This needs to be filled out by the super() function
        """
        pass

    def residuals(self, params=None):
        """ Calculate the residuals: y - fit_function(x).  params is an optional parameter array. If params is not specified, it looks for the array in the class.
        """
        if params is not None:
            self.params = params
        return self.ydata - self.fit_function() 

    def fit(self):
        """ Fit data from (xdata, ydata) using fit_function().  This calls scipy.optimize.leastsq().
        """
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
        ss_tot=((self.ydata-self.fit_function().mean())**2).sum()
        self.final_Rsquared = 1 - (ss_err/ss_tot)

        try:
            # calculate the errors the same way that gnuplot uses, namely sqrt(diagonal(cov_x)*chisq/df).
            self.final_params_errors = np.sqrt(np.diagonal(cov_x))*np.sqrt(self.final_chisq/(self.npoints - len(self.params)))
        except ValueError:
            print "Error: The fit did not converge. Please change parameters and try again"
            self.final_params_errors = np.zeros_like(self.params)

        if ier not in (1,2,3,4):
            # read somewhere that ier = 1, 2, 3, 4 also gives correct solution. http://bugs.debian.org/cgi-bin/bugreport.cgi?bug=474930
            print "Error: No solution found!"
            self.final_params_errors = np.zeros_like(self.params)

    def get_fwhm(self):
        hm    = self.params[0]/2.+self.params[3]
        right = self.xdata[abs(self.ydata[:self.params[1]]-hm).argmin()]
        left  = self.xdata[abs(self.ydata[self.params[1]:]-hm).argmin()]
        self.fwhm = abs(left-right)
        

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
            return a * self.xdata + b

        def guess_parameters(self):
            self.params = np.zeros(2)
            self.params[0] = ( self.ydata[0] - self.ydata[-1] ) * 2 / (self.ydata[0] + self.ydata[-1])
            self.params[1] = self.ydata.mean()

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
            if (tf<0 or beta<0): return np.ones_like(self.xdata) * self.try_again

            return a + b * (np.exp(-1*(self.xdata/tf)**beta)**2)

        def guess_parameters(self):
            self.params = np.zeros(4)
            self.params[0] = self.ydata[-1]
            self.params[1] = self.ydata[0] - self.params[0]
            #print self.npoints/2, self.xdata, len(self.xdata)
            self.params[2] = self.xdata[int(self.npoints/2)]
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
            if (tf<0 or beta<0): return np.ones_like(self.xdata) * self.try_again

            return a + b * (np.exp(-1*(self.xdata/tf)**beta))

        def guess_parameters(self):
            self.params = np.zeros(4)
            self.params[0] = self.ydata[-1]
            self.params[1] = self.ydata[0] - self.params[0]
            #print self.npoints/2, self.xdata, len(self.xdata)
            self.params[2] = self.xdata[int(self.npoints/2)]
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
            if ( tf<0 ): return np.ones_like(self.xdata) * self.try_again

            return a + b * np.exp(-1*(self.xdata/tf))

        def guess_parameters(self):
            self.params = np.zeros(3)
            self.params[0] = self.ydata[-1]
            self.params[1] = self.ydata[0] - self.params[0]
            #print self.npoints/2, self.xdata, len(self.xdata)
            self.params[2] = self.xdata[int(self.npoints/2)]

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
    class Gaussian(OneDimFit):

        def __init__(self, data):
            OneDimFit.__init__(self, data)
            self.functional = "a exp(-(x-b)^2/(2*c^2)) + shift"
            self.params_map ={ 0:"a", 1:"b", 2:"c" , 3:"shift"}

        def fit_function(self):
            a, b, c, shift = self.params
            if ( c > 0 ): return np.ones_like(self.xdata) * self.try_again
            return a*np.exp(-(self.xdata-b)**2/(2*c**2)) + shift

        def guess_parameters(self):
            self.params = np.zeros(4)
            # average the first and last points to try to guess the background
            self.params[3] = (self.ydata[0] + self.ydata[-1]) / 2.0
            self.params[0] = self.ydata.max() - self.params[3]
            self.params[1] = self.xdata[self.ydata.argmax()]
            
            # width is always some function-dependent multiple of fwhm
            self.get_fwhm()
            self.params[2] = self.fwhm/2.35 #2.35 = 1/(2*sqrt(2*ln(2)))

    fit = Gaussian(data)
    fit.fit()
    return fit

def lorentzian(data):
    """ fit a function to a Lorentzian.  This fits the function:
        f(x) = (gam/(2*pi))*a/((x-b)^2 + (gam/2.0)^2) + shift

    arguments:
        data - Data to fit.  This should be a (N, 2) array of (xvalues, yvalues).

    returns:
        result - a fit class that contains the final fit.  This object has various
        self-descriptive parameters, the most useful of which are result.final_params
        and resul_final_params_errors.
    """
    class Lorentzian(OneDimFit):

        def __init__(self, data):
            OneDimFit.__init__(self, data)
            self.functional = "(gam/(2*pi))*a/((x-b)^2 + (gam/2.0)^2) + shift"
            self.params_map ={ 0:"a", 1:"b", 2:"gam", 3:"shift"}

        def fit_function(self):
            a, b, gam, shift = self.params
            # checks to return high numbers if parameters are getting out of hand.
            if ( gam > 0 ): return np.ones_like(self.xdata) * self.try_again
            return (gam/(2*np.pi))*a/((self.xdata-b)**2 + (gam/2.0)**2) + shift

        def guess_parameters(self):
            self.params = np.zeros(4)
            # average the first and last points to try to guess the background
            self.params[3] = (self.ydata[0] + self.ydata[-1]) / 2.0
            self.params[0] = self.ydata.max() - self.params[3]
            self.params[1] = self.xdata[self.ydata.argmax()]
            
            # width is always some function-dependent multiple of fwhm
            self.get_fwhm()
            self.params[2] = self.fwhm/2.

    fit = Lorentzian(data)
    fit.fit()
    return fit

def lorentzian_sq(data):
    """ fit a function to a squared lorentzian.  This fits the function:
        f(x) = a/((x-x0)**2/w**2+1)**2+bg

    arguments:
        data - Data to fit.  This should be a (N, 2) array of (xvalues, yvalues).

    returns:
        result - a fit class that contains the final fit.  This object has various
        self-descriptive parameters, the most useful of which are result.final_params
        and result_final_params_errors.
    """
    class LorentzianSq(OneDimFit):

        def __init__(self, data):
            OneDimFit.__init__(self, data)
            self.functional = "a/((x-x0)**2/w**2+1)**2+bg"
            self.params_map ={ 0:"a", 1:"x0", 2:"w", 3:"bg"}

        def fit_function(self):
            a, x0, w, bg = self.params
            # checks to return high numbers if parameters are getting out of hand.
            #if ( gam > 0 ): return np.ones_like(self.xdata) * self.try_again
            return a/((self.xdata-x0)**2./w**2.+1)**2.+bg

        def guess_parameters(self):
            self.params = np.zeros(4)
            
            self.params[3] = abs(self.ydata).min()              # bg
            self.params[0] = self.ydata.max() - self.params[3]  # scale
            self.params[1] = self.xdata[self.ydata.argmax()]    # center x0
            
            # width is always some function-dependent multiple of fwhm
            self.get_fwhm()
            self.params[2] = self.fwhm/1.29 #1.29 = 1/(2*sqrt(sqrt(2)-1))

    fit = LorentzianSq(data)
    fit.fit()
    return fit

