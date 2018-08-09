import sys
import numpy as np
import matplotlib as mpl
import matplotlib.dates as md
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from scipy.special import erf
from scipy.optimize import curve_fit
import math


import common_tools as tools



### Parameters to make cuts on the ER and NR data

# For Run 3 tritium, use S2 > 165; 38 < drift < 305; R < 20 cm;
# In paper, the reject events where S2 is truncated by end of event buffer; I don't make that cut.
# Run 3 WIMP search: badArea < 80 phd for goodArea < 630 phd. badArea < 80 + 0.095*(goodArea-630) for goodArea > 630 phd
# Run 3 conversion: Z = 5.6 + 48.72 - 0.1501*drift [cm]
# Run 3 DD paper: S2 > 55 phd for LY measurement; R < 21 cm; badArea > 219 phd; g2 = 11.4; S2 < 5000 phd;
#                 S2_raw > 164 phd from Casey; 80 < drift < 130

# Run 4 DD from Casey: S2_raw > 270; S2 < 8200
# Run 4: use frozen cuts

tdrift_min = {'3': 38., '4': 40.}
tdrift_max = {'3': 305., '4': 300.}

s1_min = {'DD_4': 0,    'CH3T_4': 0,      'C14_4': 0}
s1_max = {'DD_4': 150,  'CH3T_4': 150,    'C14_4': 1000}
s2_min = {'DD_4': 0,    'CH3T_4': 0,      'C14_4': 0}
s2_max = {'DD_4': 8200, 'CH3T_4': 120000, 'C14_4': 120000}

R_cut = {'DD_4': 20, 'CH3T_4': 20, 'C14_4': 20}

field_weight_min = 50
field_weight_max = 700
field_weight_nbins = 100

DD_epoch = 1348 * 24 * 3600 * 1e8
DD_date_bounds = [[0,5], [5,8.4], [8.4,9.6], [9.6,12.6], [12.6,15.5], [15.5,150],
                  [150,194], [194,250], [250,500], [500,514], [514,600], [600,614],
                  [614,615.7], [615.7,617.65], [617.65,622.4], [628.5, 1000]]

WXe = 13.7e-3
#C14_luxstamp_bound = 17829816152662399
C14_luxstamp_low = 17811151587971000 
C14_luxstamp_high = 17818076535270760
C14_Q = 140
CH3T_Q = 18.6
CH3T_Run04_ls_bounds = [[12145132539275712, 12152021754545938],
                        [13101711862286634, 13107861931359714],
                        [14967961084286088, 14974145432728914],
                        [16082399984138552, 16088587192890172],
                        [16836799808488796, 16843110714609792],
                        [17768171107968174, 17774615008543918]]
CH3T_Run03_ls_bounds = [[8212600478558346, 8218506074492283],
                        [8255730697443465, 8260843033793867],
                        [9344297698551568, 9350116928228632],
                        [9353027128954120, 9358842131164324]]


spike_corrections_file = '/global/projecta/projectdirs/lux/shared-analysis/detector-perf/data/spike_count_correction_v10.dat'

PMT_configs_Run04 = [np.array([]) for i in range(8)] + [np.arange(0, 122)]
PMT_removals = [[22], [82], [112], [52,94], [34,4,64], [61,1,31,91,72,12,103], [43,74,14,51,111,92,32,2,62,23,41],
                [101,83,114,54,73]]
for j in range(7, -1, -1):
    previous = PMT_configs_Run04[j+1]
    cut = np.zeros(len(previous))
    for p in PMT_removals[7-j]:
        cut += (previous == p)
    PMT_configs_Run04[j] = np.delete(previous, np.argwhere(cut))

x = PMT_configs_Run04[0]
PMT_configs_Run03 = np.delete(x, np.argwhere((x == 82) + (x == 71) + (x == 11) +
                                             (x == 44) + (x == 104) + (x == 84) + (x == 24)))


########################################################################################################################


### Get the drift time cut for DD data. This cut will be a function of the luxstamp of the event.
### Args: array of drift time, array of luxstamps, any initial cuts that you've already chosen, whether to plot
### Returns: boolean array of DD cuts; True means event is kept

def getDDTimeCut(zr_full, luxstamp_full, init_cuts, plot=False):

    # Days since 2014-09-10 at 00:00:00
    days_full = (luxstamp_full - DD_epoch) / (24 * 3600 * 1e8)

    # Note: instead of the gun position being +/- 4.15 cm, you can also do +/- 23 us
    dmx_time = np.zeros(len(zr_full))
    means = np.zeros(len(DD_date_bounds))
    for k in range(len(DD_date_bounds)):        
        cut_k = init_cuts * (days_full > DD_date_bounds[k][0]) * (days_full < DD_date_bounds[k][1])
        n, bins = np.histogram(zr_full[cut_k], 100)
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        means[k] = bin_centers[np.argmax(n)]
        dmx_time += cut_k * (zr_full > (means[k] - 4.15)) * (zr_full < (means[k] + 4.15))
    
    if (plot):
        fig, axes = plt.subplots(8, 2, figsize=(20,14*8), sharey=True)
        for k in range(len(DD_date_bounds)):
            i = int(k / 2.)
            j = k % 2
            cut_k = init_cuts * (days_full > DD_date_bounds[k][0]) * (days_full < DD_date_bounds[k][1])
            axes[i,j].hist2d(days_full[cut_k], zr_full[cut_k], (200,200),
                norm=LogNorm(), cmap=plt.get_cmap('viridis'))
            axes[i,j].plot([0, 1000], [means[k] - 4.15, means[k] - 4.15], 'k-', lw=5)
            axes[i,j].plot([0, 1000], [means[k] + 4.15, means[k] + 4.15], 'k-', lw=5)
            axes[i,j].set_xlim([min(days_full[cut_k]) - 0.05 * (max(days_full[cut_k]) - min(days_full[cut_k])),
                max(days_full[cut_k])  + 0.05 * (max(days_full[cut_k]) - min(days_full[cut_k]))])
            axes[i,j].set_ylim([10, 50])
            axes[i,j].tick_params(labelsize=25)
            axes[i,j].set_xlabel('Days since 2014-09-10', fontsize=25)
            if (j == 0):
                axes[i,j].set_ylabel('z-position [cm]', fontsize=25)

    dmx_time = dmx_time > 0
    return dmx_time



### Get the square cut for the NR data
### Args: Real x and y numpy arrays
### Returns: array of booleans with the cuts, True means event is kept

def getDataCutDDSquare(xr_raw, yr_raw):
    
    
    eval_edge_1 = (xr_raw - 1) *-12 
    eval_edge_2 = (xr_raw - 9) *-12 
    
    cut_edge_1 = yr_raw > eval_edge_1
    cut_edge_2 = yr_raw < eval_edge_2

    return cut_edge_1 * cut_edge_2
    

### Get all cuts on the ER or NR data
### Args: dictionary with all data, boolean if the data is DD (NR)
### Returns: array of booleans with the cuts, True means event is kept

def getDataCut(data, pt, run, plot=False):

    ### Some useful string variables
    ptr = pt + '_' + str(run)
    rs = str(run)

    ### Load variables
    s1_raw         = data['S1_area_cor']
    s1_max_peak    = data['S1_max_peak_area']
    s1_raw_uncor   = data['S1_area']
    s2_raw         = data['S2_area_cor']
    s2_raw_uncor   = data['S2_area']
   
    xS2_raw        = data['S2_x_raw']
    yS2_raw        = data['S2_y_raw']
    rS2_raw        = np.sqrt(xS2_raw ** 2 + yS2_raw ** 2)
    tdrift_raw     = data['S2_drift']
    xr_raw         = data['x_real']
    yr_raw         = data['y_real']
    rr_raw         = np.sqrt(xr_raw **2 + yr_raw ** 2)
        
    luxstamp_raw   = data['luxstamp']
    field_raw      = data['field']
    chi2_raw       = data['S2_chi2_rec']
    g1_raw, g2_raw = tools.getG1G2FromLuxstamp(luxstamp_raw)
    
    s2_aft_t0      = data['S2_aft_t0_samples']
    s2_aft_t1      = data['S2_aft_t1_samples']
    gaus_fit_sigma = data['S2_gaus_fit_sigma']    
    s2_shape       = (s2_aft_t1 - s2_aft_t0) / gaus_fit_sigma
    chi2_raw       = data['S2_chi2_rec']
    badarea_raw    = data['badarea']
    goodarea_raw   = data['goodarea'] 
    
    ### Create cut object
    full_cut       = np.ones(len(luxstamp_raw), dtype='bool')

    ### Cuts that are different based on particle type, but all particles have the same type of cut
    dmx_s1s2       = (s1_raw > s1_min[ptr]) * (s1_raw < s1_max[ptr]) * (s2_raw > s2_min[ptr]) * (s2_raw < s2_max[ptr])
    dmx_tdrift     = (tdrift_raw > tdrift_min[rs]) * (tdrift_raw < tdrift_max[rs])
    full_cut      *= dmx_s1s2 * dmx_tdrift

    ### Common cuts based on run
    if (run == 4):
              
        dmx_bad_area   = ((goodarea_raw <= 253) * (badarea_raw <= 80)) + \
                           ((goodarea_raw > 253) * (badarea_raw <= (80 * (((pow(10, -2.4)) * goodarea_raw)**0.4668))))
        dmx_bad_area   = dmx_bad_area >= 1
        dmx_raw_area   = s2_raw_uncor > 200
        dmx_s2_width   = gaus_fit_sigma > 35
        dmx_shape1     = (s2_raw_uncor < 1200) * \
                           (s2_shape > (1.2 + (0.3/10000.0) * s2_raw_uncor)) * \
                           (s2_shape < (2.4 + ((3.239212144-2.4)/1200.0) * s2_raw_uncor))
        dmx_shape2     = (s2_raw_uncor >= 1200) * \
                           (s2_shape > (1.2 + (0.3/10000.0) * s2_raw_uncor)) * \
                           (s2_shape < (3.5383 - 2.69535e-4 * s2_raw_uncor + 1.69126e-08 * s2_raw_uncor**2.0))
        dmx_shape3     = (s2_raw > 20000)
        dmx_shape      = dmx_shape1 + dmx_shape2 + dmx_shape3
        dmx_shape      = dmx_shape >= 1
        dmx_chi2       = (s2_raw_uncor > 10000) + \
                           ((s2_raw_uncor > 4200) * (s2_raw_uncor <= 10000) * (chi2_raw < 300)) + \
                           ((s2_raw_uncor <= 4200) * (chi2_raw < (50.0 + (100.0 / 4200.0 * s2_raw_uncor))))
        dmx_chi2       = dmx_chi2 >= 1
        dmx_pmt        = s1_max_peak < (0.10207864 * s1_raw_uncor + 2.94551845)
        #dmx_structure  = s1_prompt_frac > \
        #                   (-2.31131979e-07 * s1_raw_uncor ** 4.0 + 3.19166994e-05 * s1_raw_uncor ** 3.0 + \
        #                   -1.63849839e-03 * s1_raw_uncor ** 2.0 + 3.88637498e-02 * s1_raw_uncor + 3.77155251e-01)
        full_cut      *= dmx_bad_area * dmx_raw_area * dmx_s2_width * dmx_chi2 * dmx_pmt  * dmx_shape
        
    ### Cuts that are specific to a particle type
    if (pt == 'DD' and run == 3):
        dmx_r          = rr_raw < R_cut['{0}_{1}'.format(pt, run)]
        dmx_bad_area   = badarea_raw < 219
        dmx_s2         = s2_raw_uncor > 164
        dmx_drift      = (tdrift_raw > 80.) * (tdrift_raw < 130.)
        dmx_gun        = getDataCutDDSquare(xr_raw, yr_raw)
        full_cut      *= dmx_r * dmx_bad_area * dmx_s2 * dmx_drift * dmx_gun
        
    elif (pt == 'DD' and run == 4):
        zr_raw         = data['z_real']
        
        dmx_r          = rr_raw < R_cut['{0}_{1}'.format(pt, run)]
        dmx_s2         = s2_raw_uncor > 270
        dmx_gun        = getDataCutDDSquare(xr_raw, yr_raw)
        dmx_time       = getDDTimeCut(zr_raw, luxstamp_raw, dmx_tdrift * dmx_r, plot)
        full_cut      *= dmx_r * dmx_s2 * dmx_gun * dmx_time
    elif (pt == 'CH3T' and run == 3):
        E = WXe * (s1_raw / g1_raw + s2_raw / g2_raw)
        dmx_injection  = np.zeros(len(full_cut), dtype='bool')
        for j in range(len(CH3T_Run03_ls_bounds)):
            dmx_injection += ((luxstamp_raw > CH3T_Run03_ls_bounds[j][0]) * (luxstamp_raw < CH3T_Run03_ls_bounds[j][1]))
        dmx_energy     = (E < CH3T_Q)
        dmx_injection  = dmx_injection >= 1
        dmx_bad_area   = ((goodarea_raw <= 630) * (badarea_raw <= 80)) + \
                           ((goodarea_raw > 630) * (badarea_raw <= (80 + 0.095 * (goodarea_raw - 630))))
        dmx_bad_area   = dmx_bad_area >= 1
        dmx_r          = rr_raw < R_cut['{0}_{1}'.format(pt, run)]
        full_cut      *= dmx_r * dmx_energy * dmx_injection * dmx_bad_area
    elif (pt == 'CH3T' and run == 4):
        dmx_r          = rr_raw < R_cut['{0}_{1}'.format(pt, run)]
        E = WXe * (s1_raw / g1_raw + s2_raw / g2_raw)
        dmx_injection  = np.zeros(len(full_cut), dtype='bool')
        for j in range(len(CH3T_Run04_ls_bounds)):
            dmx_injection += ((luxstamp_raw > CH3T_Run04_ls_bounds[j][0]) * (luxstamp_raw < CH3T_Run04_ls_bounds[j][1]))
        dmx_energy     = (E < CH3T_Q)
        dmx_injection  = dmx_injection >= 1
        dmx_r          = rr_raw < R_cut['{0}_{1}'.format(pt, run)]
        full_cut      *= dmx_r * dmx_energy * dmx_injection
    elif (pt == 'C14'):        
        E = WXe * (s1_raw / g1_raw + s2_raw / g2_raw)
        dmx_energy     = (E < C14_Q)
        dmx_injection  = (luxstamp_raw > C14_luxstamp_low) * (luxstamp_raw < C14_luxstamp_high)
        dmx_r          = rr_raw < R_cut['{0}_{1}'.format(pt, run)]
        full_cut      *= dmx_r * dmx_energy * dmx_injection
    else:
        print('Unrecognized pt (' + pt + ') and/or run(' + str(run) + ')')


    return full_cut



