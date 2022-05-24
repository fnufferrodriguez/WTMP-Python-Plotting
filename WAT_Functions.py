'''
* Copyright 2022 United States Bureau of Reclamation (USBR).
* United States Department of the Interior
* All Rights Reserved. USBR PROPRIETARY/CONFIDENTIAL.
* Source may not be released without written approval
* from USBR

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

sat_data_do = [14.60, 14.19, 13.81, 13.44, 13.09, 12.75, 12.43, 12.12, 11.83, 11.55, 11.27, 11.01, 10.76, 10.52, 10.29,
               10.07, 9.85, 9.65, 9.45, 9.26, 9.07, 8.90, 8.72, 8.56, 8.40, 8.24, 8.09, 7.95, 7.81, 7.67, 7.54, 7.41,
               7.28, 7.16, 7.05, 6.93, 6.82, 6.71, 6.61, 6.51, 6.41, 6.31, 6.22, 6.13, 6.04, 5.95]
sat_data_temp = [0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20., 21.,
                 22., 23., 24., 25., 26., 27., 28., 29., 30., 31., 32., 33., 34., 35., 36., 37., 38., 39., 40., 41.,
                 42., 43., 44., 45.]

f_interp = interpolate.interp1d(sat_data_temp, sat_data_do,
                                fill_value=(sat_data_do[0], sat_data_do[-1]), bounds_error=False)


def datetime2Ordinal(indate):
    '''
    converts datetime objects to ordinal values
    :param indate: datetime object
    :return: ordinal
    '''

    ord = indate.toordinal() + float(indate.hour) / 24. + float(indate.minute) / (24. * 60.)
    return ord

def cleanMissing(indata):
    '''
    TODO: merge with omit values function?
    removes data with -901. flags
    :param indata: array of data to be cleaned
    :return: cleaned data array
    '''

    indata[indata == -901.] = np.nan

    return indata

def cleanComputed(indata):
    '''
    TODO: merge with omit values function?
    removes data with -9999 flags
    :param indata: array of data to be cleaned
    :return: cleaned data array
    '''

    indata[indata == -9999.] = np.nan
    return indata

def cleanOutputDirectory(dir_name, filetype):
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

def calcDOSaturation(temp, diss_ox):
    '''
    calulates dissolved oxygen saturation. uses a series of pre computed DO values interpolated
    :param temp: temperature value
    :param diss_ox: dissolved oxygen
    :return: dissolved oxygen value
    '''

    do_sat = f_interp(temp)
    return diss_ox / do_sat * 100.

def calcComputedDOSat(vtemp, vdo):
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
            v[j] = calcDOSaturation(vtemp[j], vdo[j])
    return v

def calcObservedDOSat(ttemp, vtemp, vdo):
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
            v[j] = calcDOSaturation(vtemp[j], vdo[j])
    return ttemp, v

# def convertDepths2Elevations(obs_depths, model_elevs):
#     '''
#     calculate observed elevations based on model elevations and obs depths
#     :param obs_depths: array of depths for observed data at timestep
#     :param model_elevs: array of model elevations at timestep
#     :return: array of observed elevations
#     '''
#
#     obs_elev = []
#     for i, d in enumerate(obs_depths):
#         e = []
#         topwater_elev = max(model_elevs[i])
#         for depth in d:
#             e.append(topwater_elev - depth)
#         obs_elev.append(np.asarray(e))
#     return obs_elev

def getIdxForTimestamp(time_Array, t_in, offset):
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
        # print('t_in:', t_in)
        # print('ttmp:', ttmp)
        # print('Available times:', [n for n in time_Array])
        return -1
    return timestep

def getSubplotConfig(n_profiles, plots_per_row):
    '''
    get subplot configs to figure out numb plots per page and num pages.
    :param n_profiles: number of total plots
    :param plots_per_row: number of plots per row
    :return: subplot rows, subplot columns
    '''

    factor = n_profiles / plots_per_row
    if factor < 1:
        return 1, n_profiles
    else:
        return math.ceil(factor), plots_per_row

def matchData(data1, data2):
    '''
    matches two sets of data to have the same length
    if one is shorter than the other, the short one is interpolated
    :param data1: dictionary containing dates and values flags
    :param data2: dictionary containing dates and values flags
    :return: Two dictionaries containing dates and values flags (data1 and data2)
    '''

    #TODO: CHECK THIS FUCNTION

    if 'dates' in data1.keys() and 'dates' in data2.keys():
        y_key = 'dates'
    elif 'depths' in data1.keys() and 'depths' in data2.keys():
        y_key = 'depths'
    elif 'elevations' in data1.keys() and 'elevations' in data2.keys():
        y_key = 'elevations'

    v_1 = data1['values']
    if isinstance(v_1, list):
        v_1 = np.asarray(v_1)

    if y_key == 'dates':
        t_1 = [n.timestamp() for n in data1[y_key]]
    else:
        t_1 = data1[y_key]

    v_2 = data2['values']
    if isinstance(v_2, list):
        v_2 = np.asarray(v_2)

    if y_key == 'dates':
        t_2 = [n.timestamp() for n in data2[y_key]]
    else:
        t_2 = data2[y_key]

    if len(v_1) == 0 or len(v_2) == 0:
        return data1, data2
    if len(v_1) == len(v_2):
        return data1, data2
    elif len(v_1) > len(v_2):
        f_interp = interpolate.interp1d(t_2, v_2, bounds_error=False, fill_value=np.nan)
        v2_interp = f_interp(t_1)
        msk = np.isfinite(v2_interp)
        v_1_msk = v_1[msk]
        v_2_msk = v2_interp[msk]
        data1['values'] = v_1_msk
        data2['values'] = v_2_msk
        data2[y_key] = data1[y_key][msk]
        data1[y_key] = data1[y_key][msk]
        return data1, data2
    elif len(v_2) > len(v_1):
        f_interp = interpolate.interp1d(t_1, v_1, bounds_error=False, fill_value=np.nan)
        v1_interp = f_interp(t_2)
        msk = np.isfinite(v1_interp)
        v_1_msk = v1_interp[msk]
        v_2_msk = np.asarray(v_2)[msk]
        data1['values'] = v_1_msk
        data1[y_key] = data2[y_key][msk]
        data2[y_key] = data2[y_key][msk]
        data2['values'] = v_2_msk
        return data1, data2

def normalize2DElevations(vals, elevations):
    '''
    interpolates reservoir data in order to normalize the list of elevations for W2 runs
    :param vals: list of lists of values at timestamps/elevations
    :param elevations: list of lists of elevations at timestamps
    :return: new values, new elevations
    '''

    newvals = []
    top_elev = np.nanmax([np.nanmax(n) for n in elevations if ~np.all(np.isnan(n))])
    bottom_elev = np.nanmin([np.nanmin(n) for n in elevations if ~np.all(np.isnan(n))])
    new_elevations = np.linspace(bottom_elev, top_elev, elevations.shape[1])
    for vi, v in enumerate(vals):
        valelev_interp = interpolate.interp1d(elevations[vi], v, bounds_error=False, fill_value = np.nan)
        newvals.append(valelev_interp(new_elevations))
    return np.asarray(newvals), np.asarray(new_elevations)

def checkData(dataset, flag='values'):
    '''
    checks datasets to ensure that they are valid for representation. Checks for a given flag, any length and
    that its not all NaNs
    :param dataset: input dataset list, array or dict
    :param flag: flag to get data from dataset if dict
    :return: boolean value on data viability
    '''

    if isinstance(dataset, dict):
        if flag not in dataset.keys():
            return False
        elif len(dataset[flag]) == 0:
            return False
        elif checkAllNaNs(dataset[flag]):
            return False
        else:
            return True
    elif isinstance(dataset, list) or isinstance(dataset, np.ndarray):
        if len(dataset) == 0:
            return False
        elif checkAllNaNs(dataset):
            return False
        else:
            return True
    else:
        return False

def checkAllNaNs(values):
    '''
    checks value sets for all nan values. Some NaN is okay, all is not.
    :param values: list or array of values
    :return: boolean
    '''

    if np.all(np.isnan(values)):
        return True
    else:
        return False

def removeNaNs(data1, data2, flag='values'):
    '''
    removes data points from both datasets where either ones are nans. This way, the nans dont throw off any stat
    analysis
    :param data1: input dataset list, array or dict
    :param data2: input dataset list, array or dict
    :param flag: flag to get data from dataset if dict
    :return:
    '''

    if isinstance(data1, dict):
        d1_msk = np.where(~np.isnan(data1[flag]))
    elif isinstance(data1, list) or isinstance(data1, np.ndarray):
        d1_msk = np.where(~np.isnan(data1))

    if isinstance(data2, dict):
        d2_msk = np.where(~np.isnan(data2[flag]))
    elif isinstance(data2, list) or isinstance(data2, np.ndarray):
        d2_msk = np.where(~np.isnan(data2))

    msk = np.intersect1d(d1_msk, d2_msk)

    if isinstance(data1, dict):
        data1[flag] = np.asarray(data1[flag])[msk]
    elif isinstance(data1, list) or isinstance(data1, np.ndarray):
        data1 = np.asarray(data1)[msk]

    if isinstance(data2, dict):
        data2[flag] = np.asarray(data2[flag])[msk]
    elif isinstance(data2, list) or isinstance(data2, np.ndarray):
        data2 = np.asarray(data2)[msk]

    return data1, data2

def calcMAE(data1, data2):
    '''
    calculates the mean absolute error for two datasets
    :param data1: dataset, list array or dict
    :param data2: dataset, list array or dict
    :return: MAE value
    '''

    data1, data2 = matchData(data1, data2)
    data1, data2 = removeNaNs(data1, data2, flag='values')
    dcheck1 = checkData(data1, flag='values')
    dcheck2 = checkData(data2, flag='values')
    if not dcheck1 or not dcheck2:
        return np.nan
    return mean_absolute_error(data2['values'], data1['values'])

def calcMeanBias(data1, data2):
    '''
    calculates the mean bias for two datasets
    :param data1: dataset, list array or dict
    :param data2: dataset, list array or dict
    :return: Meanbias value
    '''

    data1, data2 = matchData(data1, data2)
    data1, data2 = removeNaNs(data1, data2, flag='values')
    dcheck1 = checkData(data1, flag='values')
    dcheck2 = checkData(data2, flag='values')
    if not dcheck1 or not dcheck2:
        return np.nan
    diff = data1['values'] - data2['values']
    count = len(data1['values'])
    mean_diff = np.sum(diff) / count
    return mean_diff

def calcRMSE(data1, data2):
    '''
    calculates the root mean square error for two datasets
    :param data1: dataset, list array or dict
    :param data2: dataset, list array or dict
    :return: RMSE value
    '''

    data1, data2 = matchData(data1, data2)
    data1, data2 = removeNaNs(data1, data2, flag='values')
    dcheck1 = checkData(data1, flag='values')
    dcheck2 = checkData(data2, flag='values')
    if not dcheck1 or not dcheck2:
        return np.nan
    diff = data1['values'] - data2['values']
    count = len(data1['values'])
    rmse = np.sqrt(np.sum(diff ** 2) / count)
    return rmse

def calcNSE(data1, data2):
    '''
    Nash-Sutcliffe Efficiency (NSE) as per `Nash and Sutcliffe, 1970
    <https://doi.org/10.1016/0022-1694(70)90255-6>`_.

    :Calculation Details:
        .. math::
           E_{\\text{NSE}} = 1 - \\frac{\\sum_{i=1}^{N}[e_{i}-s_{i}]^2}
           {\\sum_{i=1}^{N}[e_{i}-\\mu(e)]^2}

        where *N* is the length of the *simulations* and *evaluation*
        periods, *e* is the *evaluation* series, *s* is (one of) the
        *simulations* series, and *μ* is the arithmetic mean.

        source: https://pypi.org/project/hydroeval
    calculates the NSE for two datasets
    :param data1: dataset, list array or dict
    :param data2: dataset, list array or dict
    :return: NSE value
    '''

    data1, data2 = matchData(data1, data2)
    data1, data2 = removeNaNs(data1, data2, flag='values')
    dcheck1 = checkData(data1, flag='values')
    dcheck2 = checkData(data2, flag='values')
    if not dcheck1 or not dcheck2:
        return np.nan
    # nash = nse(data1['values'], data2['values'])
    ### STEVE
    nse_ = 1 - (
            np.sum((data2['values'] - data1['values']) ** 2, axis=0, dtype=np.float64)
            / np.sum((data2['values'] - np.mean(data2['values'])) ** 2, dtype=np.float64)
               )

    ### MIKE DEAS
    # nse_ = 1 - (
    #         np.sum((data1['values'] - data2['values']) ** 2, axis=0, dtype=np.float64)
    #         / np.sum((data2['values'] - np.mean(data2['values'])) ** 2, dtype=np.float64)
    # )
    if np.isinf(nse_):
        nse_ = np.nan

    return nse_

def getCount(data1):
    '''
    calculates the length of datasets
    :param data1: dataset, list array or dict
    :return: count value
    '''

    dcheck1 = checkData(data1, flag='values')
    if not dcheck1:
        return np.nan
    # return len(data1['values'])
    return len(np.where(~np.isnan(data1['values']))[0])

def calcMean(data1):
    '''
    calculates the mean of datasets
    :param data1: dataset, list array or dict
    :return: mean value
    '''

    dcheck1 = checkData(data1, flag='values')
    if not dcheck1:
        return np.nan
    return(np.nanmean(data1['values']))

def convertObsDepths2Elevations(obs_depths, model_elevs):
    '''
    calculate observed elevations based on model elevations and obs depths
    :param obs_depths: array of depths for observed data at timestep
    :param model_elevs: array of model elevations at timestep
    :return: array of observed elevations
    '''

    obs_elev = []
    for i, d in enumerate(obs_depths):
        e = []
        modeled_elevs = model_elevs[i]
        if len(modeled_elevs) == 0:
            obs_elev.append(np.full(len(d), np.nan)) #make nan boys
        else:
            topwater_elev = max(model_elevs[i])

            for depth in d:
                e.append(topwater_elev - depth)
            obs_elev.append(np.asarray(e))
    return obs_elev

def convertObsElevations2Depths(obs_elevs, model_depths, model_elevations):
    '''
    calculate observed elevations based on model elevations and obs depths
    :param obs_depths: array of depths for observed data at timestep
    :param model_elevs: array of model elevations at timestep
    :return: array of observed elevations
    '''

    obs_depth = []
    for i, e in enumerate(obs_elevs):
        d = []
        modeled_depths = model_depths[i]
        if len(modeled_depths) == 0 or len(model_elevations[i]) == 0:
            obs_depth.append(np.full(len(e), np.nan)) #make nan boys
        else:
            topwater_elev = max(model_elevations[i])

            for elev in e:
                d.append(topwater_elev - elev)
            obs_depth.append(np.asarray(d))
    return obs_depth

def convertTempUnits(values, units):
    '''
    convert temperature units between c and f
    :param values: temp values
    :param units: units string
    :return: converted units
    '''

    if units.lower() in ['f', 'faren', 'degf', 'fahrenheit', 'fahren', 'deg f']:
        values = convert_temperature(values, 'F', 'C')
        return values
    elif units.lower() in ['c', 'cel', 'celsius', 'deg c', 'degc']:
        values = convert_temperature(values, 'C', 'F')
        return values
    else:
        print('Undefined temp units:', units)
        return values

