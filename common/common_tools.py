import numpy as np
import scipy.optimize as opt
import math
import datetime

### Gaussian function
### Args: x-value, normalization, mean, standard deviation
### Returns: value of Gaussian function at x

def gaus(x, N, mu, sigma):
    
    y = N * np.exp(-(x - mu)**2 / (2 * sigma**2)) / np.sqrt(2 * math.pi * sigma**2);
    return(y);



### Fit data to a Gaussian function
### Args: data, number of bins to use, whether to return optimized normalization
### Returns: best fit mean, uncertainty on mean, best fit sigma, uncertainty on sigma,
###          optionally best fit normalization, uncertainty on normalization

def fitGaussian(x, n_bins, return_N=False, weights=[]):

    if (len(weights) == 0):
        counts, bin_edges = np.histogram(x, n_bins);
    else:
        counts, bin_edges = np.histogram(x, n_bins, weights=weights);

    bin_centers = np.array([0.5 * (bin_edges[i+1] + bin_edges[i]) for i in range(len(bin_edges) - 1)]);
    popt, pcov = opt.curve_fit(gaus, bin_centers, counts, p0=[max(counts)/2, np.mean(x), np.std(x)]);
    popt[2] = abs(popt[2]); # Sometimes sigma < 0
    x_N = popt[0];
    x_mean = popt[1];
    x_sigma = popt[2];
    x_N_err = np.sqrt(abs(pcov[0][0]));
    x_mean_err = np.sqrt(abs(pcov[1][1]));
    x_sigma_err = np.sqrt(abs(pcov[2][2]));

    if (return_N):
        return (x_N, x_N_err, x_mean, x_mean_err, x_sigma, x_sigma_err);
    else:
        return (x_mean, x_mean_err, x_sigma, x_sigma_err);



### Integral of Gaussian distribution
### Args: list or array containing x-value, Gaussian mean, Gaussian sigma
### Returns: integral of Gaussian function from -inf to x-value

def gausCDF(arr):
    
    x = arr[0]
    mu = arr[1]
    sigma = arr[2]
    
    z_score = (x - mu) / sigma
    cdf = 0.5 * (1 + math.erf(z_score / math.sqrt(2)))
    return cdf



### -1 times integral of Gaussian distribution (useful for optimization)
### Args: list or array containing x-value, Gaussian mean, Gaussian sigma
### Returns: integral of Gaussian function from -inf to x-value

def negGausCDF(arr):
    return -1 * gausCDF(arr)



### Convert energy from keVnr to keVee
### Args: energy in keVnr
### Returns: energy in keVee

def keVnr_to_keVee(keVnr):
    
    A = 0.17262
    gamma = 1.05
    
    keVee = A* keVnr ^ gamma
    return keVee



### Convert energy from keVee to keVnr
### Args: energy in keVee
### Returns: energy in keVnr

def keVee_to_keVnr(E_ee):
    
    A = 0.17262
    b = 1.05
    
    keVnr = pow(10, (np.log10(E_ee) - np.log10(A))/b)
    
    return keVnr



### Reduce data in a dictionary based on some cuts
### Args: dictionary with all data, boolean array of cuts, dictionary keys to save
### Optional: additional keys and arrays to save to the output dictionary, without any cuts;
###           additional keys and arrays to save to the output dictionary, with cuts;
###           file to save output dictionary
### Output: new dictionary with all cuts and optionally other arrays

def reduceData(data, cut, rqs_to_save, keys_full=None, arrays_full=None, keys_cut=None, arrays_cut=None, save_file=''):
    
    reduced_data = {}    
    for rq in rqs_to_save:
        # We don't care about these RQs anyway, and they might have length 10x the length of the other RQs
        if (rq == 'wsumpod_t0_samples' or rq == 'wsumpod_t2_samples' or rq == 'wsumpod_coincidence'):
            continue
        reduced_data[rq] = data[rq][cut]
        
    if (keys_full != None):
        for k, v in zip(keys_full, arrays_full):
            reduced_data[k] = v[cut]
            
    if (keys_cut != None):
        for k, v in zip(keys_cut, arrays_cut):
            reduced_data[k] = v
            
    if (save_file != ''):
        np.savez(save_file, **reduced_data)
   
    return reduced_data



### Given n dictionaries with the same keys, each key corresponding to a Numpy array,
### combine the dictionaries.
### Args: list of dictionaries, optional ID for the events of each dictionary
### Returns: combined dictionary

def combineData(dictionary_list, ids=None):
    
    df = {}
    for k in dictionary_list[0].keys():
        a = np.array([])
        for d in dictionary_list:
            a = np.append(a, d[k])
        df[k] = a
    
    if (ids != None):

        id_array = np.array([])
        for i in range(len(dictionary_list)):
            d = dictionary_list[i]
            id_temp = np.array([ids[i] for j in range(len(d[d.keys()[0]]))])
            id_array = np.append(id_array, id_temp)
        
        df['id'] = id_array

    return df



### Given luxstamps, return the appropriate g1 and g2.
### Args: array of luxstamps
### Returns: array of g1, array of g2
### See below documents for definition of time bins and values of g1 and g2
### https://docs.google.com/presentation/d/1vsyV6pHdYSy-oU19OYO8g6YAQKeMFNbDnNV6twUaJ30/edit#slide=id.g2430bfb29b_0_75
### https://docs.google.com/document/d/1tAv2BOB6wijrrobjfqrYIQ9umWqN61O2Q_A4T0zlIi8/edit
### https://arxiv.org/pdf/1512.03506.pdf

def getG1G2FromLuxstamp(luxstamps, run=4):
    
    if run ==3:
        g1 = np.full(len(luxstamps), 0.117)
        g2 = np.full(len(luxstamps), 12.1)

    elif run ==4:
    
        epoch = datetime.datetime(2011, 1, 1, 0, 0, 0)
        tb1 = datetime.datetime(2014, 9, 9, 0, 0, 0)
        tb2 = datetime.datetime(2015, 1, 1, 0, 0, 0)
        tb3 = datetime.datetime(2015, 4, 1, 0, 0, 0)
        tb4 = datetime.datetime(2015, 10, 1, 0, 0, 0)

        #g1_tb = [0.117, 0.100435, 0.099738, 0.099068, 0.097741]
        #g2_tb = [12.1, 19.395274, 19.382842, 19.075510, 18.752850]

        g1_tb = [0.117, 0.0994, 0.0995, 0.0989, 0.0974]
        g2_tb = [12.1, 19.56, 19.21, 18.93, 18.64]


        g1 = np.zeros(len(luxstamps))
        g2 = np.zeros(len(luxstamps))

        for j in range(len(luxstamps)):
            t_since_epoch = datetime.timedelta(seconds=(luxstamps[j]/1e8))
            t_event = epoch + t_since_epoch    
            if (t_event < tb1):
                g1[j] = g1_tb[0]
                g2[j] = g2_tb[0]
            elif (t_event < tb2):
                g1[j] = g1_tb[1]
                g2[j] = g2_tb[1]
            elif (t_event < tb3):
                g1[j] = g1_tb[2]
                g2[j] = g2_tb[2]
            elif (t_event < tb4):
                g1[j] = g1_tb[3]
                g2[j] = g2_tb[3]
            else:
                g1[j] = g1_tb[4]
                g2[j] = g2_tb[4]

    return g1, g2
    

# Linear extrapolatation of a curve
# Args: array of x and y points, low value of x at which we extrapolate y (optional),
#       high value of x at which we extrapolate y (optional)
# Returns: new x and y arrays with extrapolated points

def extrapolateCurve(x_arr, y_arr, x_min=None, x_max=None):

    if (x_min != None):
        y_extrap = y_arr[0] + \
            (x_min - x_arr[0]) * (y_arr[0] - y_arr[1]) / (x_arr[0] - x_arr[1])
        x_arr = np.insert(x_arr, 0, x_min)
        y_arr = np.insert(y_arr, 0, y_extrap)

    if (x_max != None):
        y_extrap = y_arr[-1] + \
            (x_max - x_arr[-1]) * (y_arr[-1] - y_arr[-2]) / (x_arr[-1] - x_arr[-2])
        x_arr = np.insert(x_arr, len(x_arr), x_max)
        y_arr = np.insert(y_arr, len(y_arr), y_extrap)

    return x_arr, y_arr


### Apply all the unit transformation
### Args: (all optional) 
###         keys_to_trans: list of keys to transform 
###         trans_factor: list of transformation factor to apply

def applyUnitTransformation(data, keys_to_trans=None, trans_factor=None):

    data_out = {}
    for key in list(data.keys()):
        if key in keys_to_trans:
            index = keys_to_trans.index(key)
            data_out[key] = data[key]*trans_factor[index]
        else:
            data_out[key] = data[key]

    return data_out


import datetime

def luxstampToDate(luxstamp):
    
    epoch = datetime.datetime(2011, 1, 1, 0, 0, 0)
    datestamp = []
    for j in range(len(luxstamp)):
        t_since_epoch = datetime.timedelta(seconds=(luxstamp[j]/1e8))
        t_event = epoch + t_since_epoch
        datestamp.append(t_event)
    
    return np.array(datestamp)




