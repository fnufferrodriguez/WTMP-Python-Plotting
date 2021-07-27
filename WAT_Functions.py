'''
Created on 7/16/2021
@author: scott
@organization: Resource Management Associates
@contact: scott@rmanet.com
@note:
'''

import numpy as np
import math
import os
from scipy import interpolate
from scipy.constants import convert_temperature
from sklearn.metrics import mean_absolute_error
import datetime as dt

sat_data_do = [14.60, 14.19, 13.81, 13.44, 13.09, 12.75, 12.43, 12.12, 11.83, 11.55, 11.27, 11.01, 10.76, 10.52, 10.29,
               10.07, 9.85, 9.65, 9.45, 9.26, 9.07, 8.90, 8.72, 8.56, 8.40, 8.24, 8.09, 7.95, 7.81, 7.67, 7.54, 7.41,
               7.28, 7.16, 7.05, 6.93, 6.82, 6.71, 6.61, 6.51, 6.41, 6.31, 6.22, 6.13, 6.04, 5.95]
sat_data_temp = [0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20., 21.,
                 22., 23., 24., 25., 26., 27., 28., 29., 30., 31., 32., 33., 34., 35., 36., 37., 38., 39., 40., 41.,
                 42., 43., 44., 45.]

f_interp = interpolate.interp1d(sat_data_temp, sat_data_do,
                                fill_value=(sat_data_do[0], sat_data_do[-1]), bounds_error=False)

def dt_to_ord(indate):
    '''
    converts datetime objects to ordinal values
    :param indate: datetime object
    :return: ordinal
    '''

    ord = indate.toordinal() + float(indate.hour) / 24. + float(indate.minute) / (24. * 60.)
    return ord

def clean_missing(indata):
    '''
    removes data with -901. flags
    :param indata: array of data to be cleaned
    :return: cleaned data array
    '''

    indata[indata == -901.] = np.nan

    return indata

def clean_computed(indata):
    '''
    removes data with -9999 flags
    :param indata: array of data to be cleaned
    :return: cleaned data array
    '''

    indata[indata == -9999.] = np.nan
    return indata

def do_saturation(temp, diss_ox):
    '''
    calulates dissolved oxygen saturation. uses a series of pre computed DO values interpolated
    :param temp: temperature value
    :param diss_ox: dissolved oxygen
    :return: dissolved oxygen value
    '''

    do_sat = f_interp(temp)
    return diss_ox / do_sat * 100.

def calc_computed_dosat(vtemp, vdo):
    '''
    calculates the computed dissolved saturated oxygen
    :param vtemp: temperature values
    :param vdo: values for dissovled oxy
    :return: DOSat values
    '''

    v = np.zeros_like(vtemp)
    for j in range(len(v)):
        if np.isnan(vtemp[j]) or np.isnan(vdo[j]):
            v[j] = np.nan
        else:
            v[j] = do_saturation(vtemp[j], vdo[j])
    return v

def calc_observed_dosat(ttemp, vtemp, vdo):
    '''
    calc dissolved saturated oxygen for observed data
    :param ttemp: times for data
    :param vtemp: temperature values
    :param vdo: values for dissovled oxy
    :return: time and DOSat values
    '''

    v = np.zeros_like(vtemp)
    for j in range(len(v)):
        if np.isnan(vtemp[j]) or np.isnan(vdo[j]):
            v[j] = np.nan
        else:
            v[j] = do_saturation(vtemp[j], vdo[j])
    return ttemp, v

def convert_obs_depths(obs_depths, model_elevs):
    '''
    calculate observed elevations based on model elevations and obs depths
    :param obs_depths: array of depths for observed data at timestep
    :param model_elevs: array of model elevations at timestep
    :return: array of observed elevations
    '''

    obs_elev = []
    for i, d in enumerate(obs_depths):
        e = []
        topwater_elev = max(model_elevs[i])
        for depth in d:
            e.append(topwater_elev - depth)
        obs_elev.append(np.asarray(e))
    return obs_elev

def get_idx_for_time(time_Array, t_in, offset):
    '''
    finds timestep for date
    :param time_Array: array of time values
    :param t_in: time step
    :param offset: time series offset for ordinal
    :return: timestep index
    '''

    ttmp = t_in.toordinal() + float(t_in.hour) / 24. + float(t_in.minute) / (24. * 60.) - offset
    min_diff = np.min(np.abs(time_Array - ttmp))
    tol = 1. / (24. * 60.)  # 1 minute tolerance
    timestep = np.where((np.abs(time_Array - ttmp) - min_diff) < tol)[0][0]
    if min_diff > 1.:
        print('nearest time step > 1 day away')
        return -1
    return timestep

def get_subplot_config(n_profiles):
    '''
    get subplot configs to figure out numb plots per page and num pages.
    :param n_profiles: number of total plots
    :return: subplot rows, subplot columns
    '''
    plot_per_page = 3.
    # plot_per_page = 4.
    factor = n_profiles / plot_per_page
    if factor < 1:
        return 1, n_profiles
    else:
        return math.ceil(factor), plot_per_page

def nse(simulations, evaluation):
    """Nash-Sutcliffe Efficiency (NSE) as per `Nash and Sutcliffe, 1970
    <https://doi.org/10.1016/0022-1694(70)90255-6>`_.

    :Calculation Details:
        .. math::
           E_{\\text{NSE}} = 1 - \\frac{\\sum_{i=1}^{N}[e_{i}-s_{i}]^2}
           {\\sum_{i=1}^{N}[e_{i}-\\mu(e)]^2}

        where *N* is the length of the *simulations* and *evaluation*
        periods, *e* is the *evaluation* series, *s* is (one of) the
        *simulations* series, and *μ* is the arithmetic mean.

        source: https://pypi.org/project/hydroeval

    """
    nse_ = 1 - (
            np.sum((evaluation - simulations) ** 2, axis=0, dtype=np.float64)
            / np.sum((evaluation - np.mean(evaluation)) ** 2, dtype=np.float64)
    )

    return nse_

def series_stats(t_comp, v_comp, t_obs, v_obs, start_limit=None, end_limit=None, time_series=False,
                 means_only=False, v_obs_min=5):
    '''
    Takes in obs and modeled data and calculates statistics based on match
    :param t_comp: times for computed
    :param v_comp: values for computed
    :param t_obs: times for observed data
    :param v_obs: values for observed data
    :param start_limit: start time to do stats. Allows for subset of data to be calc
    :param end_limit: end time to do stats
    :param time_series: flag to get times from time arrays
    :param means_only: flag to only grab means
    :param v_obs_min: min number of observed values
    :return: stats Mean Bias, MAE, RMSE, NSE, COUNT
    '''
    # will this fail if t_comp is not ascending?
    stime = t_comp[0] if start_limit is None else start_limit
    etime = t_comp[-1] if end_limit is None else end_limit

    if not time_series and stime > etime:
        tmp = etime
        etime = stime
        stime = tmp

    msk = (t_comp >= stime) & (t_comp <= etime)
    t_comp = t_comp[msk]
    v_comp = v_comp[msk]
    msk = (t_obs >= stime) & (t_obs <= etime) & np.isfinite(v_obs)
    t_obs = t_obs[msk]
    v_obs = v_obs[msk]

    mean_diff = 0.
    sq_error = 0.
    count = 0
    if len(v_obs) > v_obs_min and len(v_comp) > v_obs_min:

        if time_series:
            tstamp_comp = np.array([t_comp[j].timestamp() for j in range(len(t_comp))])
            tstamp_obs = np.array([t_obs[j].timestamp() for j in range(len(t_obs))])
        else:
            tstamp_comp = t_comp
            tstamp_obs = t_obs
        f_computed = interpolate.interp1d(tstamp_comp, v_comp, bounds_error=False, fill_value=np.nan)

        comp_matched_to_obs = f_computed(tstamp_obs)
        obs_comp_match_mask = np.isfinite(comp_matched_to_obs)

        v_obs_stats = v_obs[obs_comp_match_mask]
        v_comp_stats = comp_matched_to_obs[obs_comp_match_mask]

        if means_only:
            return {'Obs. Mean': np.mean(v_obs_stats),
                    'Comp. Mean': np.mean(v_comp_stats)}
        else:

            diff = v_comp_stats - v_obs_stats
            count = len(v_comp_stats)
            mean_diff = np.sum(diff) / count
            rmse = np.sqrt(np.sum(diff ** 2) / count)

            nash = nse(v_comp_stats, v_obs_stats)
            try:
                mae = mean_absolute_error(v_obs_stats, v_comp_stats)
            except:
                print('Failed to calculate MAE!')
                mae = np.nan
            return {'Mean Bias': mean_diff, 'MAE': mae, 'RMSE': rmse, 'NSE': nash, 'COUNT': count}
            # return mean_diff,mae,rmse,nash,len(v_obs)
    else:
        return None

def clean_output_dir(dir_name, filetype):
    '''
    removes all files with a prescribed file type from a given directory
    mainly used to erase old images from output directory
    :param dir_name: full path to directory to clean
    :param filetype: file type to delete
    '''

    files_in_directory = os.listdir(dir_name)
    filtered_files = [file for file in files_in_directory if file.endswith(filetype)]
    for file in filtered_files:
        path_to_file = os.path.join(dir_name, file)
        try:
            os.remove(path_to_file)
        except:
            print('Failed to delete', path_to_file)
            print('Continuing..')