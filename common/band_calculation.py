import sys
import numpy as np
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import common_tools as tools
import scipy.optimize as opt

from ROOT import TH1D



### Given ER or NR data, compute the ER/NR band.
### Arguments: array of s1c data, array of s2c data, array/list of s1c boundaries,
###            whether to extrapolate to bin edges, whether to include the band widths
### Returns:   s1c and log10(s2c/s1c) points that make up the band,
###            optionally the band width as a function of s1c
### Note: len(band_s1c) = len(band_lq) = len(s1c_bins) - 1
### Note: "lq" = "log quotient"

def getBand(s1c_data, s2c_data, s1c_bins, weight_data=[], return_err=False, return_sigma=False, extrap=False, verbose=False):

    # Set up some useful variables and containers
    s1c_bins = np.array([i for i in s1c_bins], dtype='float')
    band_s1c = 0.5 * (s1c_bins[:-1] + s1c_bins[1:])
    band_lq = np.zeros(len(band_s1c))
    band_lq_err = np.zeros(len(band_s1c))
    band_lq_sigma = np.zeros(len(band_s1c))
    band_lq_sigma_err = np.zeros(len(band_s1c))
    lq_data = np.log10(s2c_data / s1c_data)
    if (len(weight_data) == 0):
        weight_data = np.ones(len(s1c_data))

    # Loop over each s1c bin that the user asked for
    for j in range(len(band_lq)):
    
        # Reduce data to just that s1c bin
        bin_cut = (s1c_data > s1c_bins[j]) * (s1c_data <= s1c_bins[j+1])
        if (len(bin_cut[bin_cut]) == 0 and verbose):
            print('Warning: no data in %.1f < s1c < %.1f' % (s1c_bins[j], s1c_bins[j+1]))
            continue;
        s1c_data_bin = s1c_data[bin_cut]
        s2c_data_bin = s2c_data[bin_cut]
        lq_data_bin = lq_data[bin_cut]
        weight_data_bin = weight_data[bin_cut]

        # Calculate the position of the band median, error of the median, and band width. If there are fewer than 10 events
        # or the Gaussian fit is bad, just use the standard median, its error, and the standard deviation.
        # See http://influentialpoints.com/Training/standard_error_of_median.htm
        if (len(lq_data_bin) <= 10):
            if (verbose):
                print('Warning: sparse data, using mean and std in %.1f < s1c < %.1f' % (s1c_bins[j], s1c_bins[j+1]))
            lq_mean = np.median(lq_data_bin)
            lq_err = 1.2533 * np.std(lq_data_bin) / math.sqrt(len(lq_data_bin))
            lq_sigma = np.std(lq_data_bin)
            h = TH1D("","",100,0,1000)
            for d in lq_data_bin: h.Fill(d)                
            lq_sigma_err = h.GetRMSError()
                
        else:
            try:
                lq_mean, lq_err, lq_sigma, lq_sigma_err = tools.fitGaussian(lq_data_bin, 20, weights=weight_data_bin)
            except (RuntimeError, RuntimeWarning):
                if (verbose):
                    print('Warning: bad fit, using mean and std in %.1f < s1c < %.1f' % (s1c_bins[j], s1c_bins[j+1]))
                lq_mean = np.median(lq_data_bin)
                lq_err = 1.2533 * np.std(lq_data_bin) / math.sqrt(len(lq_data_bin))
                lq_sigma = np.std(lq_data_bin)
                
                h = TH1D("","",100,0,1000)
                for d in lq_data_bin: h.Fill(d)                
                lq_sigma_err = h.GetRMSError()
                
        if (lq_err > 0.2):
            if (verbose):
                print('Warning: bad fit, using mean and std in %.1f < s1c < %.1f' % (s1c_bins[j], s1c_bins[j+1]))
            lq_mean = np.median(lq_data_bin)
            lq_err = 1.2533 * np.std(lq_data_bin) / math.sqrt(len(lq_data_bin))
            lq_sigma = np.std(lq_data_bin)
            h = TH1D("","",100,0,1000)
            for d in lq_data_bin: h.Fill(d)                
            lq_sigma_err = h.GetRMSError()

        band_lq[j] = lq_mean
        band_lq_err[j] = lq_err
        band_lq_sigma[j] = lq_sigma
        band_lq_sigma_err[j] = lq_sigma_err

    # If we want to extrapolate the band to the s1c bin edges, use the same strategy via recursion, but only use the
    # data that is between the bin center and the bin edge.
    if (extrap):
        cut = (s1c_data >= band_s1c[-1]) * (s1c_data <= s1c_bins[-1])
        s1c, lq, err, sigma = getBand(s1c_data[cut], s2c_data[cut], [band_s1c[-1], s1c_bins[-1]], weight_data[cut], True, True)
        band_s1c = np.insert(band_s1c, len(band_s1c), s1c_bins[-1])
        band_lq = np.insert(band_lq, len(band_lq), lq[0])
        band_lq_err = np.insert(band_lq_err, len(band_lq_err), err[0])
        band_lq_sigma = np.insert(band_lq_sigma, len(band_lq_sigma), sigma[0])
        
        cut = (s1c_data >= s1c_bins[0]) * (s1c_data <= band_s1c[0])
        s1c, lq, err, sigma = getBand(s1c_data[cut], s2c_data[cut], [s1c_bins[0], band_s1c[0]], weight_data[cut], True, True)
        band_s1c = np.insert(band_s1c, 0, s1c_bins[0])
        band_lq = np.insert(band_lq, 0, lq[0])
        band_lq_err = np.insert(band_lq_err, 0, err[0])
        band_lq_sigma = np.insert(band_lq_sigma, 0, sigma[0])

    # Return the results
    if (return_err and return_sigma):
        return band_s1c, band_lq, band_lq_err, band_lq_sigma, band_lq_sigma_err
    elif (return_err and not return_sigma):
        return band_s1c, band_lq, band_lq_err
    elif (not return_err and return_sigma):
        return band_s1c, band_lq, band_lq_sigma
    else:
        return band_s1c, band_lq
