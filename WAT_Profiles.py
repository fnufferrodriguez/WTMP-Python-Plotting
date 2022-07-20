'''
* Copyright 2022 United States Bureau of Reclamation (USBR).
* United States Department of the Interior
* All Rights Reserved. USBR PROPRIETARY/CONFIDENTIAL.
* Source may not be released without written approval
* from USBR

Created on 7/15/2021
@author: scott
@organization: Resource Management Associates
@contact: scott@rmanet.com
@note:
'''

import numpy as np
from functools import reduce
from scipy import interpolate

import WAT_Functions as WF
import WAT_Reader as WR
import WAT_Time as WT

def getProfileDates(Line_info, StartTime, EndTime):
    '''
    gets dates from observed text profiles
    :param Line_info: dictionary containing line information, must include filename
    :return: list of times
    '''

    if 'filename' in Line_info.keys(): #Get data from Observed
        times = WR.getTextProfileDates(Line_info['filename'], StartTime, EndTime) #TODO: set up for not observed data??
        return times

    WF.print2stdout('Illegal Dates selection.')
    return []

def getProfileInterpResolution(object_settings, default=30):
    '''
    Gets the resolution value to interpolate profile data over for table stats. default value of 30
    if not defined.
    :param object_settings: currently selected object settings dictionary
    :param default: used if not defined in user settings
    :return: interpolation int value
    '''

    if 'resolution' in object_settings.keys():
        resolution = object_settings['resolution']
    else:
        WF.print2stdout('Resolution not defined. Setting to default values.')
        resolution = default
    return int(resolution)

def getProfileTimestamps(object_settings, StartTime, EndTime):
    '''
    Gets timestamps based off of user settings in XML file and reads/builds them.
    :param object_settings: currently selected object settings dictionary
    :return: list of timestamp values to be plotted
    '''

    if isinstance(object_settings['datessource_flag'], str):
        timestamps = []
        for line in object_settings[object_settings['datakey']]:
            if line['flag'] == object_settings['datessource_flag']:
                timestamps = getProfileDates(line, StartTime, EndTime)
    elif isinstance(object_settings['datessource_flag'], dict): #single date instance..
        timestamps = []
        tstamp_dates = object_settings['datessource']['dates']
        for d in tstamp_dates:
            dfrmt = WT.translateDateFormat(d, 'datetime', None, StartTime, EndTime, None)
            if dfrmt != None:
                timestamps.append(dfrmt)
            else:
                WF.print2stdout('Invalid Timestamp', d)

    if len(timestamps) == 0:
        #if something fails, or not implemented, or theres just no dates in the window, make some up
        timestamps = WT.makeRegularTimesteps(StartTime, EndTime, days=15)

    return np.asarray(timestamps)

def getProfileTimestampYearMonthIndex(object_settings, years):
    '''
    gets month indexes for each year based off timestamps for profile tables
    :param object_settings: dictionary containing settings for current object
    :return: list of lists for years/months and timestamps
    '''

    timestamp_indexes = []
    for year in years:
        year_idx = []
        for mon in range(1,13):
            mon_idx = []
            for ti, timestamp in enumerate(object_settings['timestamps']):
                if timestamp.year == year and timestamp.month == mon:
                    mon_idx.append(ti)
            year_idx.append(mon_idx)
        timestamp_indexes.append(year_idx)
    return timestamp_indexes

def convertDepthsToElevations(data, object_settings, wse_data={}):
    '''
    handles data to convert depths into elevations for observed data
    :param object_settings: dicitonary of user defined settings for current object
    :return: object settings dictionary with updated elevation data
    '''

    #
    for ld in data.keys():
        found_elevs = False
        if data[ld]['elevations'] == []:
            noelev_flag = ld
            wse_data_key = ld + '_wse'
            if wse_data_key in wse_data.keys():
                selected_wse_data = wse_data[wse_data_key]
                if len(selected_wse_data['elevations']) > 0:
                    selected_wse_data = matchProfileTimestamps(data[ld]['times'], selected_wse_data, onflag='elevations')['elevations']
                    found_elevs = True
            # else:
            if not found_elevs:
                for old in data.keys():
                    if len(data[old]['elevations']) > 0:
                        # elev_flag = old
                        selected_wse_data = data[old]['elevations']
                        selected_wse_data = WF.getMaxWSEFromElev(selected_wse_data)
                        found_elevs = True
                        break

            if found_elevs:
                data[noelev_flag]['elevations'] = convertObsDepths2Elevations(data[noelev_flag]['depths'],
                                                                              selected_wse_data)
            else:
                object_settings['usedepth'] = 'true'
    return data, object_settings

def convertElevationsToDepths(data, object_settings, wse_data={}):
    '''
    handles data to convert depths into elevations for observed data
    :param object_settings: dicitonary of user defined settings for current object
    :return: object settings dictionary with updated elevation data
    '''

    for ld in data.keys():
        found_elevs = False
        if data[ld]['elevations'] == []:
            nodepth_flag = ld
            wse_data_key = ld + '_wse'
            if wse_data_key in wse_data.keys():
                selected_wse_data = wse_data[wse_data_key]
                selected_wse_data = matchProfileTimestamps(data[ld]['times'], selected_wse_data, onflag='elevations')['elevations']
                found_elevs = True
            if not found_elevs:
                for old in data.keys():
                    if len(data[old]['elevations']) > 0:
                        # elev_flag = old
                        selected_wse_data = data[old]['elevations']
                        selected_wse_data = WF.getMaxWSEFromElev(selected_wse_data)
                        found_elevs = True
                        break

            if found_elevs:
                data[nodepth_flag]['depths'] = convertObsElevations2Depths(data[nodepth_flag]['elevations'],
                                                                               selected_wse_data)
            else:
                object_settings['usedepth'] = 'false'
    return data, object_settings

def convertObsDepths2Elevations(input_depths, reference_elevs):
    '''
    calculate observed elevations based on model elevations and obs depths
    :param input_depths: array of depths for observed data at timestep
    :param reference_elevs: array of model elevations at timestep
    :return: array of observed elevations
    '''

    obs_elev = []
    if len(reference_elevs) == 0:
        obs_elev.append(np.full(len(input_depths), np.nan)) #make nan boys
    else:
        for i, d in enumerate(input_depths):
            e = []
            topwater_elev = reference_elevs[i]
            for depth in d:
                e.append(topwater_elev - depth)
            obs_elev.append(np.asarray(e))
    return obs_elev

def convertObsElevations2Depths(input_elevs, reference_elevs):
    '''
    calculate observed elevations based on model elevations and obs depths
    :param obs_depths: array of depths for observed data at timestep
    :param model_elevs: array of model elevations at timestep
    :return: array of observed elevations
    '''

    out_depth = []
    if len(reference_elevs) == 0:
        out_depth.append(np.full(len(input_elevs), np.nan)) #make nan boys
    else:
        for i, e in enumerate(input_elevs):
            d = []
            topwater_elev = reference_elevs[i]
            for elev in e:
                d.append(topwater_elev - elev)
            out_depth.append(np.asarray(d))
    return out_depth

def convertProfileDataUnits(object_settings, data, line_settings):
    '''
    converts the units of profile data if unitsystem is defined
    :param object_settings: user defined settings for current object
    :return: object_setting dictionaries with updated units and values
    '''

    if 'unitsystem' not in object_settings.keys():
        WF.print2stdout('Unit system not defined.')
        return data, line_settings
    for flag in data.keys():
        if line_settings[flag]['units'] == None:
            continue
        else:
            profiles = data[flag]['values']
            profileunits = line_settings[flag]['units']
            for pi, profile in enumerate(profiles):
                profile, newunits = WF.convertUnitSystem(profile, profileunits, object_settings['unitsystem'])
                profiles[pi] = profile
            line_settings[flag]['units'] = newunits
    return data, line_settings

def filterProfileData(data, line_settings, object_settings):
    '''
    filters profile data by ylims, xlims, and omitvalues
    :param data: dictionary containing data
    :param line_settings: dictionary contining settings for line
    :param object_settings: dictionary containing settings for object
    :return: filtered profile dictionary
    '''

    xmax = None
    xmin = None
    ymax = None
    ymin = None

    if 'usedepth' in object_settings.keys():
        if object_settings['usedepth'].lower() == 'true':
            yflag = 'depths'
        else:
            yflag = 'elevations'
    else:
        WF.print2stdout('UseDepth flag not set. Cannot filter properly.')
        return data, object_settings

    if 'xlims' in object_settings.keys():
        if 'max' in object_settings['xlims'].keys():
            xmax = float(object_settings['xlims']['max'])
        if 'min' in object_settings['xlims'].keys():
            xmin = float(object_settings['xlims']['min'])

    if 'ylims' in object_settings.keys():
        if 'max' in object_settings['ylims'].keys():
            ymax = float(object_settings['ylims']['max'])
        if 'min' in object_settings['ylims'].keys():
            ymin = float(object_settings['ylims']['min'])

    # Find Index of ALL acceptable values.
    for lineflag in data.keys():
        cur_data = data[lineflag]
        cur_line_settings = line_settings[lineflag]

        current_xmax = xmax
        current_xmin = xmin
        current_ymax = ymax
        current_ymin = ymin
        if 'xlims' in cur_line_settings.keys():
            if 'max' in cur_line_settings['xlims'].keys():
                current_xmax = float(cur_line_settings['xlims']['max'])
            if 'min' in cur_line_settings['xlims'].keys():
                current_xmin = float(cur_line_settings['xlims']['min'])
        if 'ylims' in cur_line_settings.keys():
            if 'max' in cur_line_settings['ylims'].keys():
                current_ymax = float(cur_line_settings['ylims']['max'])
            if 'min' in cur_line_settings['ylims'].keys():
                current_ymin = float(cur_line_settings['ylims']['min'])

        filtbylims = False
        if 'filterbylimits' in cur_line_settings.keys():
            if cur_line_settings['filterbylimits'].lower() == 'true':
                filtbylims = True
        else:
            if 'filterbylimits' in object_settings.keys():
                if object_settings['filterbylimits'].lower() == 'true':
                    filtbylims = True

        if 'omitvalue' in cur_line_settings.keys():
            omitvalue = float(cur_line_settings['omitvalue'])
        else:
            omitvalue = None

        for pi, profile in enumerate(cur_data['values']):
            ydata = cur_data[yflag][pi]

            if current_xmax != None and filtbylims:
                xmax_filt = np.where(profile <= current_xmax)
            else:
                xmax_filt = np.arange(len(profile))

            if current_xmin != None and filtbylims:
                xmin_filt = np.where(profile >= current_xmin)
            else:
                xmin_filt = np.arange(len(profile))

            if current_ymax != None and filtbylims:
                ymax_filt = np.where(ydata <= current_ymax)
            else:
                ymax_filt = np.arange(len(ydata))

            if current_ymin != None and filtbylims:
                ymin_filt = np.where(ydata >= current_ymin)
            else:
                ymin_filt = np.arange(len(ydata))

            if omitvalue != None:
                omitval_filt = np.where(profile != omitvalue)
            else:
                omitval_filt = np.arange(len(profile))

            master_filter = reduce(np.intersect1d, (xmax_filt, xmin_filt, ymax_filt, ymin_filt, omitval_filt))

            data[lineflag]['values'][pi] = profile[master_filter]
            data[lineflag][yflag][pi] = ydata[master_filter]

    return data, object_settings

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

def matchProfileTimestamps(input_timestamps, timeseries_dict, onflag='values'):
    '''
    gets values from time series that allign with profile dates
    :param input_timestamps: profile timesteps in a list
    :param timeseries_dict: dictionary containing times and values
    :param onflag: optional flag incase 'values' is the incorrect flag in timeseries_dict
    :return: new dict with selected values and dates
    '''

    output = {}
    timestamp_idx = WR.getClosestTime(input_timestamps, timeseries_dict['dates'])
    output[onflag] = timeseries_dict[onflag][timestamp_idx]
    output['dates'] = timeseries_dict['dates'][timestamp_idx]
    for key in timeseries_dict.keys():
        if key not in output.keys():
            output[key] = timeseries_dict[key]
    return output

