import numpy, speckle
import matplotlib.pyplot as plt

### this example shows how to fit a set of (fictitious) data to a hypothesized
### functional form, then plot the results. the data is stored in a csv file as
### columns, with column names in the first row. there is no explicit x variable
### (such as time or frame number) in this data.

# open the data using genfromtxt
path_to_csv = '../exampledata/data_to_fit.csv'
data = numpy.genfromtxt(path_to_csv,dtype=numpy.float32,delimiter=',',names=True)

# for each data series, fit to the hypothesized form using speckle.fit, then
# plot both the data and the evaluated fit function. notice that in the absence
# of an explicit x series I create one.
for name in data.dtype.names:
    
    # get the data; put into correct shape
    y_values = data[name]
    x_values = numpy.arange(len(y_values))
    fit_data = numpy.vstack([x_values,y_values]).transpose()
    
    # fit the data using speckle. admire how little work is required!
    # fitted contains a bunch of stuff bundled into a dictionary: fitted.output
    fitted  = speckle.fit.decay_exp_beta(fit_data)
    to_plot = fitted.output['evaluated'] 
    params  = fitted.output['fit_parameters']
    
    print name, params
    
    # plot the data on a loglog scale
    plt.loglog(x_values,y_values,'bo',label='Experiment',markersize=3)
    plt.loglog(x_values,to_plot, 'r-',label='Fitted Data',lw=2)
    plt.legend()
    plt.savefig("fit %s.png"%name)
    plt.clf()
