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
import re
import pickle
import pandas as pd
import datetime as dt

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

def calcDOSaturation(temp, diss_ox, DOSat_Interp):
    '''
    calulates dissolved oxygen saturation. uses a series of pre computed DO values interpolated
    :param temp: temperature value
    :param diss_ox: dissolved oxygen
    :return: dissolved oxygen value
    '''

    do_sat = DOSat_Interp(temp)
    return diss_ox / do_sat * 100.

def calcComputedDOSat(vtemp, vdo, DOSat_Interp):
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
            v[j] = calcDOSaturation(vtemp[j], vdo[j], DOSat_Interp)
    return v

def calcObservedDOSat(ttemp, vtemp, vdo, ):
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
        # valelev_interp = interpolate.interp1d(elevations[vi], v, bounds_error=False, fill_value = np.nan)
        valelev_interp = interpolate.interp1d(elevations[vi], v, bounds_error=False, fill_value ="extrapolate")
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

def filterContourOverTopWater(values, elevations, topwater):
    '''
    takes values for contour reservoir plots and nan's out any values over topwater. Ressim duplicates data
    to the top of the domain instead of cutting it off
    :param values: list of values at each timestep
    :param elevations: elevations to find closest index to top water
    :param topwater: water surface elevation at each timestep
    :return: values with nans
    '''

    for twi, tw in enumerate(topwater):
        elevationtopwateridx = (np.abs(elevations - tw)).argmin()
        values[twi][elevationtopwateridx+1:] = np.nan
    return values

def filterTimestepByYear(timestamps, year):
    '''
    returns only timestamps from the given year. Otherwise, just return all timestamps
    :param timestamps: list of dates
    :param year: target year
    :return:
        timestamps: list of selected timestamps
    '''

    if year == 'ALLYEARS':
        return timestamps
    return [n for n in timestamps if n.year == year]

def replaceflaggedValues(Report, settings, itemset):
    '''
    recursive function to replace flagged values in settings
    :param settings: dict, list or string containing settings, potentially with flags
    :return:
        settings: dict, list or string with flags replaced
    '''

    if isinstance(settings, str):
        if '%%' in settings:
            newval = replaceFlaggedValue(Report, settings, itemset)
            settings = newval
    elif isinstance(settings, dict):
        for key in settings.keys():
            if isinstance(settings[key], dict):
                settings[key] = replaceflaggedValues(Report, settings[key], itemset)
            elif isinstance(settings[key], list):
                new_list = []
                for item in settings[key]:
                    new_list.append(replaceflaggedValues(Report, item, itemset))
                settings[key] = new_list
            else:
                try:
                    if '%%' in settings[key]:
                        newval = replaceFlaggedValue(Report, settings[key], itemset)
                        settings[key] = newval
                except TypeError:
                    continue
    elif isinstance(settings, list):
        for i, item in enumerate(settings):
            if '%%' in item:
                settings[i] = replaceFlaggedValue(Report, item, itemset)

    return settings

def replaceFlaggedValue(Report, value, itemset):
    '''
    replaces strings with flagged values with known paths
    flags are now case insensitive with more intelligent matching. yay.
    needs to use '[1:-1]' for paths, otherwise things like /t in a path C:/trains will be taken literal
    :param value: string potentially containing flagged value
    :return:
        value: string with potential flags replaced
    '''


    if itemset == 'general':
        flagged_values = {'%%region%%': Report.ChapterRegion,
                          '%%observedDir%%': Report.observedDir,
                          '%%startyear%%': str(Report.startYear),
                          '%%endyear%%': str(Report.endYear)
                          }
    elif itemset == 'modelspecific':
        flagged_values = {'%%ModelDSS%%': Report.DSSFile,
                          '%%Fpart%%': Report.alternativeFpart,
                          '%%plugin%%': Report.plugin,
                          '%%modelAltName%%': Report.modelAltName,
                          '%%SimulationName%%': Report.SimulationName,
                          '%%SimulationDir%%': Report.SimulationDir,
                          '%%baseSimulationName%%': Report.baseSimulationName,
                          '%%starttime%%': Report.StartTimeStr,
                          '%%endtime%%': Report.EndTimeStr,
                          '%%LastComputed%%': Report.LastComputed
                          }

    for fv in flagged_values.keys():
        pattern = re.compile(re.escape(fv), re.IGNORECASE)
        value = pattern.sub(repr(flagged_values[fv])[1:-1], value) #this seems weird with [1:-1] but paths wont work otherwise
    return value

def selectContourReachesByID(contoursbyID, ID):
    '''
    selects contour data based on the ID
    :param contoursbyID: dictionary containing all contours
    :param ID: selected ID ('base', 'alt_1')
    :return: list of contour keys that apply to that ID
    '''

    output_contours = {}
    for key in contoursbyID:
        if contoursbyID[key]['ID'] == ID:
            output_contours[key] = contoursbyID[key]
    return output_contours

def selectContourReservoirByID(contoursbyID, ID):
    '''
    returns the correct contour from dictionary based on ID
    :param contoursbyID: dictionary containing several contours
    :param ID: selected ID to find in data
    :return: dictionary for corresponding ID, or empty
    '''

    for key in contoursbyID:
        if contoursbyID[key]['ID'] == ID:
            return contoursbyID[key]
    return {}

def stackContours(contours):
    '''
    stacks data for multiple contour reaches so they appear as a single reach. Adds distances to stay consistent.
    keeps track of the distances in which defined reaches change
    :param contours: dictionary containing reach contour data
    :return:
        output_values: values at each timestep/distance
        output_dates: list of dates for data
        output_distance:distances for each cell center from source
        transitions: names and locations where the reaches change
    '''

    output_values = np.array([])
    output_dates = np.array([])
    output_distance = np.array([])
    transitions = {}
    for contourname in contours.keys():
        contour = contours[contourname]
        if len(output_values) == 0:
            output_values = pickle.loads(pickle.dumps(contour['values'], -1))
        else:
            output_values = np.append(output_values, contour['values'][:, 1:], axis=1)
        if len(output_dates) == 0:
            output_dates = contour['dates']
        if len(output_distance) == 0:
            output_distance = contour['distance']
            transitions[contourname] = 0
        else:
            last_distance = output_distance[-1]
            current_distances = contour['distance'][1:] + last_distance
            output_distance = np.append(output_distance, current_distances)
            transitions[contourname] = current_distances[0]
    return output_values, output_dates, output_distance, transitions

def stackProfileIndicies(exist_data, new_data):
    '''
    takes an existing array of data and adds another array
    for contour plots of several reaches split into different groups
    stacks them together so they function as a single reach
    exist_data is existing array
    new data is data to be added to it
    :param exist_data: dictionary containing existing data
    :param new_data: data to be added to existing data
    :return: modified exist_data
    '''

    for runflag in new_data.keys():
        if runflag not in exist_data.keys():
            exist_data[runflag] = {}
        for itemflag in new_data[runflag]:
            if itemflag not in exist_data[runflag].keys():
                exist_data[runflag][itemflag] = new_data[runflag][itemflag]
            else:
                if isinstance(new_data[runflag][itemflag], list):
                    exist_data[runflag][itemflag] += new_data[runflag][itemflag]
                elif isinstance(new_data[runflag][itemflag], np.ndarray):
                    exist_data[runflag][itemflag] = np.append(exist_data[runflag][itemflag], new_data[runflag][itemflag])
    return exist_data

def getPandasTimeFreq(intervalstring):
    '''
    Reads in the DSS formatted time intervals and translates them to a format pandas.resample() understands
    bases off of the time interval, so 15MIN becomes 15T, or 6MON becomes 6M
    :param intervalstring: DSS interval string such as 1HOUR or 1DAY
    :return: pandas time interval
    '''

    intervalstring = intervalstring.lower()
    if 'min' in intervalstring:
        timeint = intervalstring.replace('min','') + 'T'
        return timeint
    elif 'hour' in intervalstring:
        timeint = intervalstring.replace('hour','') + 'H'
        return timeint
    elif 'day' in intervalstring:
        timeint = intervalstring.replace('day','') + 'D'
        return timeint
    elif 'mon' in intervalstring:
        timeint = intervalstring.replace('mon','') + 'M'
        return timeint
    elif 'week' in intervalstring:
        timeint = intervalstring.replace('week','') + 'W'
        return timeint
    elif 'year' in intervalstring:
        timeint = intervalstring.replace('year','') + 'A'
        return timeint
    else:
        print('Unidentified time interval')
        return 0

def buildTimeSeries(startTime, endTime, interval):
    '''
    builds a regular time series using the start and end time and a given interval
    #TODO: if start time isnt on the hour, but the interval is, change start time to be hourly?
    :param startTime: datetime object
    :param endTime: datetime object
    :param interval: DSS interval
    :return: list of time series dates
    '''

    intervalinfo = getPandasTimeFreq(interval)
    ts = pd.date_range(startTime, endTime, freq=intervalinfo, closed=None)
    ts = np.asarray([t.to_pydatetime() for t in ts])
    return ts
    # try:
    #     # intervalinfo = self.Constants.time_intervals[interval]
    #     interval = intervalinfo[0]
    #     interval_info = intervalinfo[1]
    # except KeyError:
    #     interval, interval_info = self.forceTimeInterval(interval)

    # if interval_info == 'np':
    #     ts = np.arange(startTime, endTime, interval)
    #     ts = np.asarray([t.astype(dt.datetime) for t in ts])
    # elif interval_info == 'pd':
    #     ts = pd.date_range(startTime, endTime, freq=interval, closed=None)
    #     ts = np.asarray([t.to_pydatetime() for t in ts])
    # return ts

def JDateToDatetime(dates, startyear):
    '''
    converts jdate dates to datetime values
    :param dates: list of jdate dates
    :return:
        dtimes: list of dates
        dtime: single date
        dates: original date if unable to convert
    '''

    # first_year_Date = dt.datetime(self.ModelAlt.dt_dates[0].year, 1, 1, 0, 0)
    first_year_Date = dt.datetime(startyear, 1, 1, 0, 0)

    if isinstance(dates, dt.datetime):
        return dates
    elif isinstance(dates, (list, np.ndarray)):
        if isinstance(dates[0], dt.datetime):
            return dates
        dtimes = np.asarray([first_year_Date + dt.timedelta(days=n) for n in dates])
        return dtimes
    elif isinstance(dates, (float, int)):
        dtime = first_year_Date + dt.timedelta(days=dates)
        return dtime
    else:
        return dates

def DatetimeToJDate(dates, time_offset):
    '''
    converts datetime dates to jdate values
    :param dates: list of datetime dates
    :return:
        jdates: list of dates
        jdate: single date
        dates: original date if unable to convert
    '''

    if isinstance(dates, (float, int)):
        return dates
    elif isinstance(dates, (list, np.ndarray)):
        if isinstance(dates[0], (float, int)):
            return dates
        # jdates = np.asarray([(datetime2Ordinal(n) - ModelAlt.t_offset) + 1 for n in dates])
        jdates = np.asarray([(datetime2Ordinal(n) - time_offset) + 1 for n in dates])
        return jdates
    elif isinstance(dates, dt.datetime):
        jdate = (datetime2Ordinal(dates) - time_offset) + 1
        return jdate
    else:
        return dates

def mergeLines(data, settings):
    '''
    reads in mergeline settings and combines time series as defined. returns a single time series based on the
    controller flag. then removes lines if dictated.
    :param data: dictionary containing data
    :param settings: list of settings including mergeline settings
    :return: updated data dictionary
    '''

    removekeys = []
    if 'mergelines' in settings.keys():
        for mergeline in settings['mergelines']:
            dataflags = mergeline['flags']
            if 'controller' in mergeline.keys():
                controller = mergeline['controller']
                if controller not in data.keys():
                    controller = dataflags[0]
            else:
                controller = dataflags[0]
            otherflags = [n for n in dataflags if n != controller]
            if controller not in data.keys():
                print('Mergeline Controller {0} not found in data {1}'.format(controller, data.keys()))
                continue
            flagnotfound = False
            for OF in otherflags:
                if OF not in data.keys():
                    print('Mergeline flag {0} not found in data {1}'.format(OF, data.keys()))
                    flagnotfound = True
            if flagnotfound:
                continue
            if 'math' in mergeline.keys():
                math = mergeline['math'].lower()
            else:
                math = 'add'
            baseunits = data[controller]['units']
            for flag in otherflags:
                if data[flag]['units'] != baseunits:
                    print('WARNING: Attempting to merge lines with differing units')
                    print('{0}: {1} and {2}: {3}'.format(flag, data[flag]['units'], controller, baseunits))
                    print('If incorrect, please modify/append input settings to ensure lines '
                          'are converted prior to merging.')
                data[controller], data[flag] = matchData(data[controller], data[flag])
                if math == 'add':
                    data[controller]['values'] += data[flag]['values']
                elif math == 'multiply':
                    data[controller]['values'] *= data[flag]['values']
                elif math == 'divide':
                    data[controller]['values'] /= data[flag]['values']
                elif math == 'subtract':
                    data[controller]['values'] -= data[flag]['values']
            if 'keeplines' in mergeline.keys():
                if mergeline['keeplines'].lower() == 'false':
                    for flag in otherflags:
                        removekeys.append(flag)
        for flag in removekeys:
            data.pop(flag)
    return data

def filterDataByYear(data, year, extraflag=None):
    '''
    filters data by a given year. Used when splitting by year
    :param data: dictionary containing data to use
    :param year: selected year
    :return:dictionary containing fultered data
    '''
    if year != 'ALLYEARS':
        for flag in data.keys():
            if len(data[flag]['dates']) > 0:
                s_idx, e_idx = getYearlyFilterIdx(data[flag]['dates'], year)
                data[flag]['values'] = data[flag]['values'][s_idx:e_idx+1]
                data[flag]['dates'] = data[flag]['dates'][s_idx:e_idx+1]
                if extraflag != None:
                    data[flag][extraflag] = data[flag][extraflag][s_idx:e_idx+1]
    return data

def getYearlyFilterIdx(dates, year):
    '''
    finds start and end index for a given year for a list of dates
    :param dates: list of datetime objects
    :param year: target year
    :return: start and end index for that year
    '''

    start_date = dates[0]
    end_date = dates[-1]
    e_year_date = dt.datetime(year,12,31,23,59)
    s_year_date = dt.datetime(year,1,1,0,0)
    interval = (dates[1] - start_date).total_seconds()
    if start_date.year == year:
        s_idx = 0
    else:
        s_idx = round(int((s_year_date - start_date).total_seconds() / interval))
    if end_date.year == year:
        e_idx = len(dates)
    else:
        e_idx = round(int((e_year_date - start_date).total_seconds() / interval))

    return s_idx, e_idx

def getMonthlyFilterIdx(dates, month):
    '''
    finds start and end index for a given month for a list of dates
    :param dates: list of datetime objects
    :param month: target month
    :return: start and end index for that year
    '''

    start_date = dates[0]
    end_date = dates[-1]
    s_month_date = dt.datetime(start_date.year, month,1,0,0)
    if month == 12:
        e_month_date = dt.datetime(start_date.year+1,1,1,0,0) - dt.timedelta(seconds=1)
    else:
        e_month_date = dt.datetime(start_date.year,month+1,1,0,0) - dt.timedelta(seconds=1)

    interval = (dates[1] - start_date).total_seconds()
    if start_date.month == month:
        s_idx = 0
    else:
        s_idx = round(int((s_month_date - start_date).total_seconds() / interval))
    if end_date.month == month:
        e_idx = len(dates)
    else:
        e_idx = round(int((e_month_date - start_date).total_seconds() / interval))

    return s_idx, e_idx

def getUnitsList(line_settings):
    '''
    creates a list of units from defined lines in user defined settings
    :param object_settings: currently selected object settings dictionary
    :return: units_list: list of used units
    '''

    units_list = []
    for flag in line_settings.keys():
        units = line_settings[flag]['units']
        if units != None:
            units_list.append(units)
    return units_list

def getUsedIDs(data):
    '''
    Finds all IDs that are actually used by data
    :param data: dictionary for data
    :return: list of IDs
    '''

    IDs = []
    for key in data.keys():
        ID = data[key]['ID']
        if ID not in IDs:
            IDs.append(ID)
    return IDs

def getAllMonthIdx(timestamp_indexes, i):
    '''
    collects all indecies for all years for a given month
    :param timestamp_indexes: list of indicies that fall in every month for every year ([[], [], []])
    :param i: month index (0-11)
    :return: list of indicies for a month for all years
    '''

    out_idx = []
    for yearlist in timestamp_indexes:
        out_idx += yearlist[i]
    return out_idx