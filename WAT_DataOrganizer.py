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

import os
import numpy as np
import pandas as pd
import traceback
import pickle

import WAT_Functions as WF
import WAT_Reader as WDR
import WAT_Time as WT
import WAT_Reader as WR

class DataOrganizer(object):

    def __init__(self, Report):
        '''
        class supports getting data from file sources and orgainzing data into memory dictionary for easy recall
        :param Report: self from main Report Generator script
        '''

        self.Report = Report #self from main
        self.intializeMemory()

    def intializeMemory(self):
        '''
        initializes the main dictionary to store data
        '''

        self.Memory = {}

    def buildMemoryKey(self, Data_info):
        '''
        creates uniform name for csv log output for data
        determines how to build the file name from the input type
        :param Data_info: information about line
        :return: name for memory key, or null if can't be determined
        '''

        # very_special_flags = f'{self.Report.SimulationName.replace(" ", "").replace(":", "")}_{self.Report.baseSimulationName.replace(" ", "").replace(":", "")}'
        very_special_flags = WF.sanitizeText(self.Report.SimulationName)

        if 'dss_path' in Data_info.keys(): #Get data from DSS record
            if 'dss_filename' in Data_info.keys():
                outname = WF.sanitizeText(os.path.basename(Data_info['dss_filename'])[:-4])
                dssnamesplit = Data_info['dss_path'].split('/')
                dssname_pick = WF.sanitizeText(f"{dssnamesplit[1]}_{dssnamesplit[2]}_{dssnamesplit[3]}_{dssnamesplit[5]}_{dssnamesplit[-2]}")
                outname = very_special_flags + '_' + outname + '_' + dssname_pick

        elif 'w2_file' in Data_info.keys():
            if 'structurenumbers' in Data_info.keys():
                outname = WF.sanitizeText(os.path.basename(Data_info['w2_file']).split('.')[0])
                if isinstance(Data_info['structurenumbers'], dict):
                    structure_nums = [Data_info['structurenumbers']['structurenumber']]
                elif isinstance(Data_info['structurenumbers'], str):
                    structure_nums = [Data_info['structurenumbers']]
                elif isinstance(Data_info['structurenumbers'], (list, np.ndarray)):
                    structure_nums = Data_info['structurenumbers']
                outname += '_Struct_' + '_'.join(structure_nums) + f'_{very_special_flags}'
            else:
                if 'column' in Data_info.keys():
                    very_special_flags += f'_Colf{Data_info["column"].replace(" ","")}'
                outname = f"{os.path.basename(Data_info['w2_file']).split('.')[0]}_{very_special_flags}"

        elif 'h5file' in Data_info.keys():
            h5name = WF.sanitizeText(os.path.basename(Data_info['h5file']).split('.h5')[0] + 'h5')
            if 'easting' in Data_info.keys() and 'northing' in Data_info.keys():
                outname = 'externalh5_{0}_{1}_{2}_{3}_{4}'.format(h5name, WF.sanitizeText(Data_info['parameter']), Data_info['easting'], Data_info['northing'], very_special_flags)
            elif 'ressimname' in Data_info.keys():
                outname = 'externalh5_{0}_{1}_{2}_{3}'.format(h5name, WF.sanitizeText(Data_info['parameter']), WF.sanitizeText(Data_info['ressimresname']), very_special_flags)

        elif 'easting' in Data_info.keys() and 'northing' in Data_info.keys():
            outname = '{0}_{1}_{2}_{3}'.format(WF.sanitizeText(Data_info['parameter']), Data_info['easting'], Data_info['northing'], very_special_flags)

        elif 'filename' in Data_info.keys(): #Get data from Observed Profile
            outname = WF.sanitizeText(os.path.basename(Data_info['filename']).split('.')[0].replace(' ', '_')) + f'_{very_special_flags}'

        elif 'w2_segment' in Data_info.keys():
            outname = 'W2_{0}_{1}_profile'.format(self.Report.ModelAlt.output_file_name.split('.')[0], Data_info['w2_segment']) + f'_{very_special_flags}'

        elif 'ressimresname' in Data_info.keys():
            outname = '{0}_{1}_{2}_{3}'.format(WF.sanitizeText(os.path.basename(self.Report.ModelAlt.h5fname).split('.')[0] +'_h5'),
                                           WF.sanitizeText(Data_info['parameter']), WF.sanitizeText(Data_info['ressimresname']), very_special_flags)
            if 'target' in Data_info.keys():
                outname += '_trgt'
                if 'parameter' in Data_info['target'].keys():
                    outname += Data_info['target']['parameter'][:4]
                if 'value' in Data_info['target'].keys():
                    outname += Data_info['target']['value']
        else:
            outname = 'NULL'

        return outname[:150]

    #################################################################
    #TimeSeries Functions
    #################################################################

    def updateTimeSeriesDataDictionary(self, data, line_settings, line):
        '''
        organizes line information and places it into a data dictionary
        :param data: dictionary containing line data
        :param line_settings: dictionary containing line settings for all lines
        :param line: dictionary containing line settings for current line
        :return: updated data dictionary
        '''

        dates, values, units = self.getTimeSeries(line, makecopy=False) #TODO: update
        datacheck = False
        if WF.checkData(values):
            datacheck = True
            flag = line['flag']
            if flag in line_settings.keys() or flag in data.keys():
                count = 1
                newflag = flag + '_{0}'.format(count)
                while newflag in data.keys():
                    count += 1
                    newflag = flag + '_{0}'.format(count)
                WF.print2stdout(f'The current flag was {flag}', debug=self.Report.debug)
                flag = newflag
                WF.print2stdout(f'The new flag is {newflag}', debug=self.Report.debug)
            datamem_key = self.buildMemoryKey(line)
            if 'units' in line.keys() and units == None:
                units = line['units']
            line_settings[flag] = {'units': units,
                                   'logoutputfilename': datamem_key}

            data[flag] = {'values': values,
                          'dates': dates}

            for key in line.keys():
                if key not in line_settings[flag].keys():
                    line_settings[flag][key] = line[key]

        return data, line_settings, datacheck

    def getProfileWSE(self, settings, onflag='lines'):
        '''
        gets the Water surface elevation from time series if <wse> flag in line
        WSE is named <flag it was under> + _wse for easy coordination
        :param settings: dictionary containing lines
        :param onflag: optional flag if <lines> is not the main datasource (i.e., datapath)
        :return: new data dictionary with WSE data
        '''

        wse_data = {}
        for dataobject in settings[onflag]:
            if 'wse' in dataobject.keys():
                dates, values, units = self.getTimeSeries(dataobject['wse'], makecopy=False)
                datamem_key = self.buildMemoryKey(dataobject['wse'])
                new_key = dataobject['flag'] + '_wse'
                wse_data[new_key] = {'elevations': values,
                                      'dates': dates,
                                      'units': units,
                                      'logoutputfilename': datamem_key}

        return wse_data

    def filterTimeSeries(self, data, line_settings):
        '''
        filters data read from W2 results files to a target elevation
        :param data: dictionary of data
        :param line_settings: dictionary containing settings for all lines
        :return: modified data dictionary
        '''

        for d in data.keys():
            if 'target_elevation' in line_settings[d].keys():
                target_elevation = float(line_settings[d]['target_elevation'])
                values = data[d]['values']
                if isinstance(values, dict): #w2 results kept in dict with elev
                    for sn in values.keys():
                        targelev_failed = np.where(values[sn]['elevcl'] != target_elevation)
                        data[d]['values'][sn]['q(m3/s)'][targelev_failed] = 0.
                elif isinstance(values, (list, np.ndarray)):
                    if 'elevation' in line_settings[d].keys():
                        if 'flag' not in line_settings[d]['elevation'].keys():
                            line_settings[d]['elevation']['flag'] = line_settings[d]['flag']+'_elev'
                            elev_times, elev_values, elev_units = self.getTimeSeries(line_settings[d]['elevation'])
                            targelev_failed = np.where(elev_values != target_elevation)
                            if len(elev_times) == len(data[d]['values']):
                                data[d]['values'][targelev_failed] = 0.
                            else:
                                WF.print2stdout(f'Values and Elevations in {d} different. Equalizing.', debug=self.Report.debug)
                                mainvalues, elev_data = WF.matchData({'dates': data[d]['dates'], 'values': data[d]['values']},
                                                                     {'dates': elev_times, 'values': elev_values})
                                targelev_failed = np.where(elev_data['values'] != target_elevation)
                                mainvalues['values'][targelev_failed] = 0.
                                data[d]['values'] = mainvalues['values']
                                data[d]['dates'] = mainvalues['dates']

            if 'filters' in line_settings[d].keys():
                for filter in line_settings[d]['filters']:
                    if 'value' not in filter.keys():
                        WF.print2stdout('Value not defined in filter. Not using filter.', debug=self.Report.debug)
                    else:
                        value = float(filter['value'])
                    use_filter_ts = False
                    if np.any([n.lower() in ['w2_file', 'dss_path', 'easting', 'h5file', 'ressimresname'] for n in filter.keys()]):
                        use_filter_ts = True
                        filter_times, filter_values, filter_units = self.getTimeSeries(filter)
                    if use_filter_ts:
                        data_to_filter = filter_values
                    else:
                        data_to_filter = data[d]['values']

                    if 'when' in filter.keys():
                        if filter['when'].lower() == 'under':
                            filtermask = np.where(data_to_filter < value)
                        elif filter['when'].lower() == 'over':
                            filtermask = np.where(data_to_filter > value)
                        elif filter['when'].lower() == 'equals':
                            filtermask = np.where(data_to_filter == value)
                    else:
                        WF.print2stdout('When condition not set in filter. Assuming equals.', debug=self.Report.debug)
                        filtermask = np.where(data_to_filter == value)
                    try:
                        data[d]['values'][filtermask] = np.nan
                    except IndexError:
                        WF.print2stdout('Filter and Data Index not equal. Not Filtering data.', debug=self.Report.debug)
                        WF.print2stdout('Confirm that Data and Filter are on the same timeseries interval', debug=self.Report.debug)

        return data

    def getTimeSeriesDataDictionary(self, settings):
        '''
        Gets profile line data from defined data sources in XML files
        :param settings: currently selected object settings dictionary
        :return: dictionary containing data and information about each data set
        '''

        data = {}
        line_settings = {}
        if 'lines' in settings.keys():
            for line in settings['lines']:
                numtimesused = 0
                if 'flag' not in line.keys():
                    WF.print2stdout('Flag not set for line (Computed/Observed/etc)', debug=self.Report.debug)
                    WF.print2stdout('Not plotting Line:', line, debug=self.Report.debug)
                    continue

                elif line['flag'].lower() == 'computed':
                    for ID in self.Report.accepted_IDs:
                        curline = pickle.loads(pickle.dumps(line, -1))
                        curline = self.Report.configureSettingsForID(ID, curline)
                        if not self.Report.checkModelType(curline):
                            continue
                        curline['numtimesused'] = numtimesused
                        data, line_settings, success = self.updateTimeSeriesDataDictionary(data, line_settings, curline)
                        if success:
                            numtimesused += 1
                else:
                    if self.Report.currentlyloadedID != 'base':
                        line = self.Report.configureSettingsForID('base', line)
                    else:
                        line = WF.replaceflaggedValues(self.Report, line, 'modelspecific')
                    line['numtimesused'] = numtimesused
                    if not self.Report.checkModelType(line):
                        continue
                    data, line_settings, success = self.updateTimeSeriesDataDictionary(data, line_settings, line)
                    if success:
                        numtimesused += 1

        return data, line_settings

    def getStraightLineValue(self, settings):
        '''
        reads settings to get value for straight lines on plots, either by getting value at timestamp or configuring
        timestamp
        :param settings: dictionary of settings for plots
        :return: dictionary with settings for straight lines
        '''

        straightlines = {}
        types_of_straightlines = ['hlines', 'vlines']
        for tosl in types_of_straightlines:
            if tosl in settings.keys():
                straightlines[tosl] = {}
                for line in settings[tosl]:
                    if 'value' in line.keys(): #if defined single value for all plots
                        value = float(line['value'])
                        const_key = f'constant_{value}'
                        if 'timestamps' in settings.keys():
                            values = [value] * len(settings['timestamps'])
                        else:
                            values = [value]
                        straightlines[tosl][const_key] = {'values': values,
                                                          'numtimesused': 0}
                        for key in line.keys():
                            if key not in straightlines[tosl][const_key].keys():
                                straightlines[tosl][const_key][key] = line[key]
                        if 'units' not in straightlines[tosl][const_key].keys():
                            straightlines[tosl][const_key]['units'] = None
                    else:
                        timeserieslines, timeserieslinesettings = self.getTimeSeriesDataDictionary({'lines': [line]})
                        for timeserieslinekey in timeserieslines.keys():
                            values = timeserieslines[timeserieslinekey]['values']
                            dates = timeserieslines[timeserieslinekey]['dates']
                            if 'timestamps' in settings.keys():
                                idx = WR.getClosestTime(settings['timestamps'], dates)
                                straightlines[tosl][timeserieslinekey] = {'values': values[idx]}
                            for key in timeserieslinesettings[timeserieslinekey].keys():
                                if key not in straightlines[tosl][timeserieslinekey].keys():
                                    straightlines[tosl][timeserieslinekey][key] = timeserieslinesettings[timeserieslinekey][key]

        return straightlines

    def getTimeSeries(self, Line_info, makecopy=True):
        '''
        gets time series data from defined sources
        :param Line_info: dictionary of line setttings containing datasources
        :param makecopy: flag that determines if the data is grabbed or just copied
        :return: dates, values, units
        '''

        if 'dss_path' in Line_info.keys(): #Get data from DSS record
            if 'dss_filename' not in Line_info.keys():
                WF.print2stdout('DSS_Filename not set for Line: {0}'.format(Line_info), debug=self.Report.debug)
                return np.array([]), np.array([]), None
            else:
                datamem_key = self.buildMemoryKey(Line_info)
                if datamem_key in self.Memory.keys():
                    WF.print2stdout('Reading {0} from memory'.format(datamem_key), debug=self.Report.debug) #noisy
                    if makecopy:
                        datamementry = pickle.loads(pickle.dumps(self.Memory[datamem_key], -1))
                    else:
                        datamementry = self.Memory[datamem_key]
                    times = datamementry['times']
                    values = datamementry['values']
                    units = datamementry['units']
                else:
                    times, values, units = WDR.readDSSData(Line_info['dss_filename'], Line_info['dss_path'],
                                                           self.Report.StartTime, self.Report.EndTime, self.Report.debug)

                    self.Memory[datamem_key] = {'times': pickle.loads(pickle.dumps(times, -1)),
                                                     'values': pickle.loads(pickle.dumps(values, -1)),
                                                     'units': pickle.loads(pickle.dumps(units, -1))}

                if np.any(values == None):
                    return np.array([]), np.array([]), None
                elif len(values) == 0:
                    return np.array([]), np.array([]), None

        elif 'w2_file' in Line_info.keys():
            if self.Report.plugin.lower() != 'cequalw2':
                return np.array([]), np.array([]), None
            datamem_key = self.buildMemoryKey(Line_info)
            if datamem_key in self.Memory.keys():
                WF.print2stdout('READING {0} FROM MEMORY'.format(datamem_key), debug=self.Report.debug)
                datamementry = pickle.loads(pickle.dumps(self.Memory[datamem_key], -1))
                times = datamementry['times']
                values = datamementry['values']
                units = datamementry['units']

            else:
                if 'structurenumbers' in Line_info.keys():
                    # Ryan Miles: yeah looks like it's str_brX.npt, and X is 1-# of branches (which is defined in the control file)
                    times, values = self.Report.ModelAlt.readStructuredTimeSeries(Line_info['w2_file'], Line_info['structurenumbers'])
                else:
                    times, values = self.Report.ModelAlt.readTimeSeries(Line_info['w2_file'], **Line_info)
                if 'units' in Line_info.keys():
                    units = Line_info['units']
                else:
                    units = None

                self.Memory[datamem_key] = {'times': pickle.loads(pickle.dumps(times, -1)),
                                                 'values': pickle.loads(pickle.dumps(values, -1)),
                                                 'units': pickle.loads(pickle.dumps(units, -1))}

        elif 'h5file' in Line_info.keys():
            datamem_key = self.buildMemoryKey(Line_info)

            if datamem_key in self.Memory.keys():
                WF.print2stdout('READING {0} FROM MEMORY'.format(datamem_key), debug=self.Report.debug)
                datamementry = pickle.loads(pickle.dumps(self.Memory[datamem_key], -1))
                times = datamementry['times']
                values = datamementry['values']
                units = datamementry['units']

            else:
                filename = Line_info['h5file']
                if not os.path.exists(filename):
                    WF.print2stdout('ERROR: H5 file does not exist:', filename, debug=self.Report.debug)
                    return [], [], None
                externalResSim = WDR.ResSim_Results('', '', '', '', self.Report, external=True)
                externalResSim.openH5File(filename)
                externalResSim.load_time() #load time vars from h5
                externalResSim.loadSubdomains()
                times, values = externalResSim.readTimeSeries(Line_info['parameter'],
                                                              float(Line_info['easting']),
                                                              float(Line_info['northing']))
                if 'units' in Line_info.keys():
                    units = Line_info['units']
                else:
                    units = None

                self.Memory[datamem_key] = {'times': pickle.loads(pickle.dumps(times, -1)),
                                                 'values': pickle.loads(pickle.dumps(values, -1)),
                                                 'units': pickle.loads(pickle.dumps(units, -1))}

        elif 'easting' in Line_info.keys() and 'northing' in Line_info.keys():
            datamem_key = self.buildMemoryKey(Line_info)
            if datamem_key in self.Memory.keys():
                WF.print2stdout('READING {0} FROM MEMORY'.format(datamem_key), debug=self.Report.debug)
                datamementry = pickle.loads(pickle.dumps(self.Memory[datamem_key], -1))
                times = datamementry['times']
                values = datamementry['values']
                units = datamementry['units']

            else:
                times, values = self.Report.ModelAlt.readTimeSeries(Line_info['parameter'],
                                                             float(Line_info['easting']),
                                                             float(Line_info['northing']))
                if 'units' in Line_info.keys():
                    units = Line_info['units']
                else:
                    units = None

                self.Memory[datamem_key] = {'times': pickle.loads(pickle.dumps(times, -1)),
                                             'values': pickle.loads(pickle.dumps(values, -1)),
                                             'units': pickle.loads(pickle.dumps(units, -1))}

        elif "ressimresname" in Line_info.keys():
            datamem_key = self.buildMemoryKey(Line_info)
            if datamem_key in self.Memory.keys():
                WF.print2stdout('READING {0} FROM MEMORY'.format(datamem_key), debug=self.Report.debug)
                datamementry = pickle.loads(pickle.dumps(self.Memory[datamem_key], -1))
                times = datamementry['times']
                values = datamementry['values']
                units = datamementry['units']

            else:
                if self.Report.plugin.lower() != 'ressim':
                    WF.print2stdout('Incorrect model type for line using ResSimResName', debug=self.Report.debug)
                    return [], [], None
                if 'target' not in Line_info.keys():
                    WF.print2stdout('No target data for profile target timeseries.', debug=self.Report.debug)
                    WF.print2stdout('Please enter target data including value and parameter.', debug=self.Report.debug)
                    return [], [], None
                if 'parameter' not in Line_info.keys():
                    WF.print2stdout('No parameter for profile target timeseries.', debug=self.Report.debug)
                    WF.print2stdout('Assuming output is elevation.', debug=self.Report.debug)
                    Line_info['parameter'] = 'elevation'

                times, values = self.Report.ModelAlt.getProfileTargetTimeseries(Line_info['ressimresname'],
                                                                                                  Line_info['parameter'],
                                                                                                  Line_info['target'])
                if 'units' in Line_info.keys():
                    units = Line_info['units']
                else:
                    units = None

        else:
            WF.print2stdout('No Data Defined for line', debug=self.Report.debug)
            return np.array([]), np.array([]), None

        if 'omitvalue' in Line_info.keys():
            omitval = float(Line_info['omitvalue'])
            values = WF.NaNOmittedValues(values, omitval, debug=self.Report.debug)
        elif 'omitvalues' in Line_info.keys():
            omitvals = [float(n) for n in Line_info['omitvalues']]
            for omitval in omitvals:
                values = WF.NaNOmittedValues(values, omitval, debug=self.Report.debug)

        if 'interval' in Line_info.keys():
            times, values = WT.changeTimeSeriesInterval(times, values, Line_info, self.Report.ModelAlt.t_offset, self.Report.startYear)


        return times, values, units

    #################################################################
    #Profile Functions
    #################################################################

    def getProfileDataDictionary(self, settings):
        '''
        Gets profile line data from defined data sources in XML files
        :param settings: currently selected object settings dictionary
        :return: dictionary containing data and information about each data set
        '''

        data = {}
        line_settings = {}
        timestamps = settings['timestamps']
        for line in settings[settings['datakey']]:
            numtimesused = 0
            if 'flag' not in line.keys():
                WF.print2stdout('Flag not set for line (Computed/Observed/etc)', debug=self.Report.debug)
                WF.print2stdout('Not plotting Line:', line, debug=self.Report.debug)
                continue
            elif line['flag'].lower() == 'computed':
                # for ID in self.SimulationVariables.keys():
                for ID in self.Report.accepted_IDs:
                    curline = pickle.loads(pickle.dumps(line, -1))
                    curline = self.Report.configureSettingsForID(ID, curline)
                    curline['numtimesused'] = numtimesused
                    curline['ID'] = ID
                    if not self.Report.checkModelType(curline):
                        continue
                    data, line_settings, success = self.updateProfileDataDictionary(data, line_settings, curline, timestamps)
                    if success:
                        numtimesused += 1
            else:
                if self.Report.currentlyloadedID != 'base':
                    line = self.Report.configureSettingsForID('base', line)
                else:
                    line = WF.replaceflaggedValues(self.Report, line, 'modelspecific')
                line['numtimesused'] = numtimesused
                if not self.Report.checkModelType(line):
                    continue
                data, line_settings, success = self.updateProfileDataDictionary(data, line_settings, line, timestamps)
                if success:
                    numtimesused += 1

        return data, line_settings

    def updateProfileDataDictionary(self, data, line_settings, profile, timestamps):
        '''
        organizes line information and places it into a data dictionary
        :param data: dictionary containing data
        :param line_settings: dictionary containing all line settings
        :param profile: dictionary containing line settings
        :param timestamps: list of dates to get data
        :return: updated data dictionary, updated line_settings dictionary
        '''

        vals, elevations, depths, times, flag, units = self.getProfileValues(profile, timestamps) #Test this speed for grabbing all profiles and then choosing
        datacheck = False
        if len(vals) > 0:
            datacheck = True
            datamem_key = self.buildMemoryKey(profile)
            if units == None:
                if 'units' in profile.keys():
                    units = profile['units']
            # else:
            #     units = None

            if profile['flag'] in line_settings.keys() or profile['flag'] in data.keys():
                datakey = '{0}_{1}'.format(profile['flag'], profile['numtimesused'])
            else:
                datakey = profile['flag']

            subset = True
            if isinstance(timestamps, str):
                subset = False

            line_settings[datakey] = {'units': units,
                                      'numtimesused': profile['numtimesused'],
                                      'logoutputfilename': datamem_key,
                                      'subset': subset
                                      }

            data[datakey] = {'values': vals,
                             'elevations': elevations,
                             'depths': depths,
                             'times': times
                             }

            for key in profile.keys():
                if key not in line_settings[datakey].keys():
                    line_settings[datakey][key] = profile[key]

        return data, line_settings, datacheck

    def getProfileValues(self, Profile_info, timesteps):
        '''
        reads in profile data from various sources for profile plots at given timesteps
        attempts to get elevations if possible
        :param Profile_info: dictionary containing settings for line
        :param timesteps: given list of timesteps to extract data at
        :return: values, elevations, depths, flag
        '''

        datamemkey = self.buildMemoryKey(Profile_info)

        if datamemkey in self.Memory.keys():
            dm = pickle.loads(pickle.dumps(self.Memory[datamemkey], -1))
            WF.print2stdout(f'retrieving {datamemkey} profile from datamem', debug=self.Report.debug)
            if isinstance(timesteps, str): #if looking for all
                if dm['subset'] == 'false': #the last time data was grabbed, it was not a subset, aka all
                    return dm['values'], dm['elevations'], dm['depths'], dm['times'], Profile_info['flag'], Profile_info['units']
                else:
                    WF.print2stdout('Incorrect Timesteps in data memory. Re-extracting data for', datamemkey, debug=self.Report.debug)
            elif np.array_equal(timesteps, dm['times']):
                return dm['values'], dm['elevations'], dm['depths'], dm['times'], Profile_info['flag'], dm['units']
            else:
                WF.print2stdout('Incorrect Timesteps in data memory. Re-extracting data for', datamemkey, debug=self.Report.debug)

        if 'filename' in Profile_info.keys(): #Get data from Observed
            filename = Profile_info['filename']
            values, yvals, times = WDR.readTextProfile(filename, timesteps, self.Report.StartTime, self.Report.EndTime)
            if 'y_convention' in Profile_info.keys():
                if Profile_info['y_convention'].lower() == 'depth':
                    return values, [], yvals, times, Profile_info['flag']
                elif Profile_info['y_convention'].lower() == 'elevation':
                    return values, yvals, [], times, Profile_info['flag']
                else:
                    WF.print2stdout('Unknown value for flag y_convention: {0}'.format(Profile_info['y_convention']), debug=self.Report.debug)
                    WF.print2stdout('Please use "depth" or "elevation"', debug=self.Report.debug)
                    WF.print2stdout('Assuming depths...', debug=self.Report.debug)
                    return values, [], yvals, times, Profile_info['flag'], None
            else:
                WF.print2stdout('No value for flag y_convention', debug=self.Report.debug)
                WF.print2stdout('Assuming depths...', debug=self.Report.debug)
                return values, [], yvals, times, Profile_info['flag'], None

        elif 'h5file' in Profile_info.keys() and 'ressimresname' in Profile_info.keys():
            filename = Profile_info['h5file']
            if not os.path.exists(filename):
                WF.print2stdout('ERROR: H5 file does not exist:', filename, debug=self.Report.debug)
                return [], [], [], [], Profile_info['flag']
            externalResSim = WDR.ResSim_Results('', '', '', '', self.Report, external=True)
            externalResSim.openH5File(filename)
            externalResSim.load_time() #load time vars from h5
            externalResSim.loadSubdomains()
            vals, elevations, depths, times = externalResSim.readProfileData(Profile_info['ressimresname'],
                                                                             Profile_info['parameter'], timesteps)
            return vals, elevations, depths, times, Profile_info['flag'], None

        elif 'w2_segment' in Profile_info.keys():
            if self.Report.plugin.lower() != 'cequalw2':
                return [], [], [], [], None
            if 'w2_file' in Profile_info.keys():
                resultsfile = Profile_info['w2_file']
            else:
                resultsfile = None
            vals, elevations, depths, times = self.Report.ModelAlt.readProfileData(Profile_info['w2_segment'], timesteps,
                                                                                   resultsfile=resultsfile)
            times = WT.JDateToDatetime(times, self.Report.startYear)
            # if isinstance(timesteps, str):
            #     vals, elevations = self.Report.Profiles.normalize2DElevations(vals, elevations)
            return vals, elevations, depths, times, Profile_info['flag'], None

        elif 'ressimresname' in Profile_info.keys():
            if self.Report.plugin.lower() != 'ressim':
                return [], [], [], [], None

            vals, elevations, depths, times = self.Report.ModelAlt.readProfileData(Profile_info['ressimresname'],
                                                                                   Profile_info['parameter'], timesteps,
                                                                                   )
            return vals, elevations, depths, times, Profile_info['flag'], None

        WF.print2stdout('No Data Defined for Profile', debug=self.Report.debug)
        WF.print2stdout('Profile:', Profile_info, debug=self.Report.debug)
        return [], [], [], [], None, None

    def getReservoirContourDataDictionary(self, settings):
        '''
        Gets Contour Reservoir data from defined data sources in XML files
        :param settings: currently selected object settings dictionary
        :return: dictionary containing data and information about each data set
        '''

        data = {}
        data_settings = {}
        for datapath in settings['datapaths']:
            for ID in self.Report.accepted_IDs:
                curreach = pickle.loads(pickle.dumps(datapath, -1))
                curreach = self.Report.configureSettingsForID(ID, curreach)
                if not self.Report.checkModelType(curreach):
                    continue
                # object_settings['timestamps'] = 'all'
                values, elevations, depths, dates, flag, units = self.getProfileValues(curreach, 'all')
                topwater = self.getProfileTopWater(curreach, 'all')
                if 'interval' in curreach.keys():
                    dates_change, values = WT.changeTimeSeriesInterval(dates, values, curreach,
                                                                       self.Report.ModelAlt.t_offset,
                                                                       self.Report.startYear)
                    dates_change, topwater = WT.changeTimeSeriesInterval(dates, topwater, curreach,
                                                                         self.Report.ModelAlt.t_offset,
                                                                         self.Report.startYear)
                    dates = dates_change
                if WF.checkData(values):
                    if 'flag' in datapath.keys():
                        flag = datapath['flag']
                    elif 'label' in datapath.keys():
                        flag = datapath['label']
                    else:
                        flag = 'reservoir_{0}'.format(ID)
                    if flag in data.keys():
                        count = 1
                        newflag = flag + '_{0}'.format(count)
                        while newflag in data.keys():
                            count += 1
                            newflag = flag + '_{0}'.format(count)
                        WF.print2stdout('The current flag is {0}'.format(flag), debug=self.Report.debug)
                        flag = newflag
                        WF.print2stdout('The new flag is {0}'.format(newflag), debug=self.Report.debug)
                    datamem_key = self.buildMemoryKey(datapath)

                    if 'units' in datapath.keys():
                        units = datapath['units']
                    else:
                        units = None

                    data_settings[flag] = {'units': units,
                                           'ID': ID,
                                           'logoutputfilename': datamem_key}

                    data[flag] = {'values': values,
                                  'dates': dates,
                                  'elevations': elevations,
                                  'topwater': topwater,
                                  'ID': ID}

                    for key in datapath.keys():
                        if key not in data_settings[flag].keys():
                            data_settings[flag][key] = datapath[key]
        #reset
        self.Report.loadCurrentID('base')
        self.Report.loadCurrentModelAltID('base')
        return data, data_settings

    def getProfileTopWater(self, profile, timesteps):
        '''
        gets topwater elevations for timestamps
        :param profile: dictionary containing settings for line
        :param timesteps: given list of timesteps to extract data at
        :return: values, elevations, depths, flag
        '''

        datamemkey = self.buildMemoryKey(profile)
        if datamemkey in self.Memory.keys():
            dm = pickle.loads(pickle.dumps(self.Memory[datamemkey], -1))
            WF.print2stdout('retrieving profile topwater from datamem', debug=self.Report.debug)
            if isinstance(timesteps, str): #if looking for all
                if dm['subset'] == 'false': #the last time data was grabbed, it was not a subset, aka all
                    return dm['topwater']
                else:
                    WF.print2stdout('Incorrect Timesteps in data memory. Re-extracting data for', datamemkey, debug=self.Report.debug)
            elif np.array_equal(timesteps, dm['times']):
                return dm['topwater']
            else:
                WF.print2stdout('Incorrect Timesteps in data memory. Re-extracting data for', datamemkey, debug=self.Report.debug)

        if 'filename' in profile.keys(): #Get data from Observed
            filename = profile['filename']
            values, yvals, times = WDR.readTextProfile(filename, timesteps, self.Report.StartTime, self.Report.EndTime)
            if 'y_convention' in profile.keys():
                if profile['y_convention'].lower() == 'elevation':
                    return [yval[0] for yval in yvals]
                elif profile['y_convention'].lower() == 'depth':
                    WF.print2stdout('Unable to get topwater from depth.', debug=self.Report.debug)
                    return []
                else:
                    WF.print2stdout('Unknown value for flag y_convention: {0}'.format(profile['y_convention']), debug=self.Report.debug)
                    WF.print2stdout('Please use "elevation"', debug=self.Report.debug)
                    WF.print2stdout('Assuming elevations...', debug=self.Report.debug)
                    return [yval[0] for yval in yvals]
            else:
                WF.print2stdout('No value for flag y_convention', debug=self.Report.debug)
                WF.print2stdout('Assuming elevation...', debug=self.Report.debug)
                return [yval[0] for yval in yvals]

        elif 'h5file' in profile.keys() and 'ressimresname' in profile.keys():
            filename = profile['h5file']
            if not os.path.exists(filename):
                WF.print2stdout('ERROR: H5 file does not exist:', filename, debug=self.Report.debug)
                return []
            externalResSim = WDR.ResSim_Results('', '', '', '', self.Report, external=True)
            externalResSim.openH5File(filename)
            externalResSim.load_time() #load time vars from h5
            externalResSim.loadSubdomains()
            topwater = externalResSim.readProfileTopwater(profile['ressimresname'], timesteps)
            return topwater

        elif 'w2_segment' in profile.keys():
            if self.Report.plugin.lower() != 'cequalw2':
                return []
            topwater = self.Report.ModelAlt.readProfileTopwater(profile['w2_segment'], timesteps)
            return topwater

        elif 'ressimresname' in profile.keys():
            if self.Report.plugin.lower() != 'ressim':
                return []
            topwater = self.Report.ModelAlt.readProfileTopwater(profile['ressimresname'], timesteps)
            return topwater

        WF.print2stdout('No Data Defined for line', debug=self.Report.debug)
        WF.print2stdout('Profile:', profile, debug=self.Report.debug)
        return []

    def commitProfileDataToMemory(self, data, line_settings, object_settings):
        '''
        commits updated data to data memory dictionary that keeps track of data
        :param data: dictionary containing data
        :param line_settings: dictionary containing settings about lines
        :param object_settings:  dicitonary of user defined settings for current object
        '''

        for line in data.keys():
            write = False
            values = pickle.loads(pickle.dumps(data[line]['values'], -1))
            depths = pickle.loads(pickle.dumps(data[line]['depths'], -1))
            elevations = pickle.loads(pickle.dumps(data[line]['elevations'], -1))
            subset = pickle.loads(pickle.dumps(line_settings[line]['subset'], -1))
            datamem_key = line_settings[line]['logoutputfilename']
            if datamem_key not in self.Memory.keys():
                write = True
            else:
                if not np.array_equal(object_settings['timestamps'], self.Memory[datamem_key]['times']):
                    write = True

            if write:
                self.Memory[datamem_key] = {'times': object_settings['timestamps'],
                                             'values': values,
                                             'elevations': elevations,
                                             'depths': depths,
                                             'units': object_settings['plot_units'],
                                             'isprofile': True,
                                             'subset': subset
                                             }

    #################################################################
    #Table Functions
    #################################################################

    def getTableDataDictionary(self, object_settings):
        '''
        Grabs data from time series data to be used in tables
        :param object_settings: currently selected object settings dictionary
        :return: dictionary object containing info from each data source and list of units
        '''

        data = {}
        line_settings = {}
        for dp in object_settings['datapaths']:
            numtimesused = 0
            if 'flag' not in dp.keys():
                WF.print2stdout('Flag not set for line (Computed/Observed/etc)', debug=self.Report.debug)
                WF.print2stdout('Not using Line:', dp, debug=self.Report.debug)
                continue
            elif dp['flag'].lower() == 'computed':
                for ID in self.Report.accepted_IDs:
                    cur_dp = pickle.loads(pickle.dumps(dp, -1))
                    cur_dp = self.Report.configureSettingsForID(ID, cur_dp)
                    cur_dp['numtimesused'] = numtimesused
                    cur_dp['ID'] = ID
                    if not self.Report.checkModelType(cur_dp):
                        continue
                    data, line_settings, success = self.updateTimeSeriesDataDictionary(data, line_settings, cur_dp)
                    if success:
                        numtimesused += 1
            else:
                if self.Report.currentlyloadedID != 'base':
                    dp = self.Report.configureSettingsForID('base', dp)
                else:
                    dp = WF.replaceflaggedValues(self.Report, dp, 'modelspecific')
                dp['numtimesused'] = numtimesused
                if not self.Report.checkModelType(dp):
                    continue
                data, line_settings, success = self.updateTimeSeriesDataDictionary(data, line_settings, dp)
                if success:
                    numtimesused += 1

        return data, line_settings

    #################################################################
    #Contour Functions
    #################################################################

    def getContours(self, settings):
        '''
        Retrieves Contour data from sources
        :param settings: dictionary for object settings
        :return: times, values, units distance. All objects are 1D/2D arrays
        '''

        if 'ressimresname' in settings.keys(): #Ressim subdomain
            datamem_key = self.buildMemoryKey(settings)
            if datamem_key in self.Memory.keys():
                WF.print2stdout('READING {0} FROM MEMORY'.format(datamem_key), debug=self.Report.debug)
                datamem_entry = pickle.loads(pickle.dumps(self.Memory[datamem_key], -1))
                times = datamem_entry['dates']
                values = datamem_entry['values']
                units = datamem_entry['units']
                distance = datamem_entry['distance']

            else:
                checkdomain = self.Report.ModelAlt.checkSubdomain(settings['ressimresname'])
                if not checkdomain:
                    return [], [], [], []
                times, values, distance = self.Report.ModelAlt.readSubdomain(settings['parameter'],
                                                                      settings['ressimresname'])

                if 'units' in settings.keys():
                    units = settings['units']
                else:
                    units = None

                self.Memory[datamem_key] = {'dates': pickle.loads(pickle.dumps(times, -1)),
                                                 'values': pickle.loads(pickle.dumps(values, -1)),
                                                 'units': pickle.loads(pickle.dumps(units, -1)),
                                                 'distance': pickle.loads(pickle.dumps(distance, -1)),
                                                 'iscontour': True}

        elif 'w2_file' in settings.keys():
            datamem_key = self.buildMemoryKey(settings)
            if datamem_key in self.Memory.keys():
                WF.print2stdout('READING {0} FROM MEMORY'.format(datamem_key), debug=self.Report.debug)
                datamementry = pickle.loads(pickle.dumps(self.Memory[datamem_key], -1))
                times = datamementry['dates']
                values = datamementry['values']
                units = datamementry['units']
                distance = datamementry['distance']
            else:
                times, values, distance = self.Report.ModelAlt.readSegment(settings['w2_file'],
                                                                    settings['parameter'])

        if 'interval' in settings.keys():
            times, values = WT.changeTimeSeriesInterval(times, values, settings, self.Report.ModelAlt.t_offset,
                                                         self.Report.startYear)

        return times, values, units, distance

    def getContourDataDictionary(self, settings):
        '''
        Gets Contour line data from defined data sources in XML files
        :param settings: currently selected object settings dictionary
        :return: dictionary containing data and information about each data set
        '''

        data = {}
        data_settings = {}
        for reach in settings['reaches']:
            for ID in self.Report.accepted_IDs:
                curreach = pickle.loads(pickle.dumps(reach, -1))
                curreach = self.Report.configureSettingsForID(ID, curreach)
                if not self.Report.checkModelType(curreach):
                    continue
                dates, values, units, distance = self.getContours(curreach)
                if WF.checkData(values):
                    if 'flag' in reach.keys():
                        flag = reach['flag']
                    elif 'label' in reach.keys():
                        flag = reach['label']
                    else:
                        flag = 'reach_{0}'.format(ID)
                    if flag in data.keys():
                        count = 1
                        newflag = flag + '_{0}'.format(count)
                        while newflag in data.keys():
                            count += 1
                            newflag = flag + '_{0}'.format(count)
                        WF.print2stdout('The current flag is {0}'.format(flag), debug=self.Report.debug)
                        flag = newflag
                        WF.print2stdout('The new flag is {0}'.format(newflag), debug=self.Report.debug)
                    datamem_key = self.buildMemoryKey(reach)

                    if 'units' in reach.keys() and units == None:
                        units = reach['units']

                    if 'y_scalar' in settings.keys():
                        y_scalar = float(settings['y_scalar'])
                        distance *= y_scalar

                    data_settings[flag] = {'units': units,
                                           'distance': distance,
                                           'ID': ID,
                                           'logoutputfilename': datamem_key}

                    data[flag] = {'values': values,
                                  'dates': dates,
                                  'ID': ID}

                    for key in reach.keys():
                        if key not in data_settings[flag].keys():
                            data_settings[flag][key] = reach[key]
        #reset
        self.Report.loadCurrentID('base')
        self.Report.loadCurrentModelAltID('base')
        return data, data_settings

    #################################################################
    #Gate Functions
    #################################################################

    def getGateDataDictionary(self, settings, makecopy=True):
        '''
        Gets profile line data from defined data sources in XML files
        :param settings: currently selected object settings dictionary
        :param makecopy: optional flag that determines if data is grabbed or copied. turn off for speed
        :return: dictionary containing data and information about each data set
        '''

        data = {}
        line_data = {}
        if 'gateops' in settings.keys():
            for gi, gateop in enumerate(settings['gateops']):

                if 'flag' in gateop.keys():
                    if gateop['flag'] not in data.keys():
                        data[gateop['flag']] = {}
                        line_data[gateop['flag']] = {}
                        gateopkey = gateop['flag']
                elif 'label' in gateop.keys():
                    if gateop['label'] not in data.keys():
                        data[gateop['label']] = {}
                        line_data[gateop['flag']] = {}
                        gateopkey = gateop['label']
                else:
                    if 'GATEOP_{0}'.format(gi) not in data.keys():
                        gateopkey = 'GATEOP_{0}'.format(gi)
                        data[gateopkey] = {}
                        line_data[gateopkey] = {}

                data[gateopkey]['gates'] = {}
                line_data[gateopkey]['gates'] = {}
                for gate in gateop['gates']:
                    dates, values, _ = self.getTimeSeries(gate, makecopy=makecopy)
                    if 'flag' in gate.keys():
                        flag = gate['flag']
                    else:
                        flag = 'gate'
                    if flag in data[gateopkey]['gates'].keys():
                        count = 1
                        newflag = flag + '_{0}'.format(count)
                        while newflag in data[gateopkey]['gates'].keys():
                            count += 1
                            newflag = flag + '_{0}'.format(count)
                        flag = newflag
                    datamem_key = self.buildMemoryKey(gate)
                    value_msk = np.where(values==0)
                    values[value_msk] = np.nan
                    if 'flag' in gateop.keys():
                        gategroup = gateop['flag']
                    else:
                        gategroup = 'gategroup_{0}'.format(gi)
                    data[gateopkey]['gates'][flag] = {'values': values,
                                                      'dates': dates}

                    line_data[gateopkey]['gates'][flag] = {'logoutputfilename': datamem_key,
                                                          'gategroup': gategroup}

                    for key in gate.keys():
                        if key not in line_data[gateopkey]['gates'][flag].keys():
                            line_data[gateopkey]['gates'][flag][key] = gate[key]

                for key in gateop.keys():
                    if key not in data[gateopkey].keys():
                        data[gateopkey][key] = gateop[key]
                    if key not in line_data[gateopkey].keys():
                        line_data[gateopkey][key] = gateop[key]

        return data, line_data

    #################################################################
    #Logging Functions
    #################################################################

    def writeDataFiles(self):
        '''
        writes out the data used in figures to csv files for later use and checking
        '''

        for key in self.Memory.keys():
            csv_name = os.path.join(self.Report.CSVPath, '{0}.csv'.format(key))
            try:
                if 'isprofile' in self.Memory[key].keys():
                    if self.Memory[key]['isprofile'] == True:
                        alltimes = self.Memory[key]['times']
                        allvalues = self.Memory[key]['values']
                        alltimes = WF.matcharrays(alltimes, allvalues)
                        allelevs = self.Memory[key]['elevations']
                        alldepths = self.Memory[key]['depths']
                        if len(allelevs) == 0: #elevations may not always fall out
                            allelevs = WF.matcharrays(allelevs, alldepths)
                        units = self.Memory[key]['units']
                        values = WF.getListItems(allvalues)
                        times = WF.getListItems(alltimes)
                        elevs = WF.getListItems(allelevs)
                        depths = WF.getListItems(alldepths)
                        if isinstance(values, (list, np.ndarray)):
                            df = pd.DataFrame({'Dates': times, 'Values ({0})'.format(units): values, 'Elevations': elevs,
                                               'Depths': depths})
                        elif isinstance(values, dict):
                            colvals = {}
                            colvals['Dates'] = times
                            for key in values:
                                colvals[key] = values[key]
                                colvals[key] = elevs[key]
                                colvals[key] = depths[key]
                            df = pd.DataFrame(colvals)
                elif 'iscontour' in self.Memory[key].keys():
                    continue #were not doing this for now, takes ~ 5 seconds per 3yr reach..

                    # if self.Data.Memory[key]['iscontour'] == True:
                    #     alltimes = self.Data.Memory[key]['dates']
                    #     allvalues = self.Data.Memory[key]['values'].T #this gets transposed a few times.. we want distance/date
                    #     alldistance = self.Data.Memory[key]['distance']
                    #     times = WF.matcharrays(alltimes, allvalues)
                    #     distances = WF.matcharrays(alldistance, allvalues)
                    #     values = WF.getListItems(allvalues)
                    #     units = self.Data.Memory[key]['units']
                    #     newstime = time.time()
                    #     df = pd.DataFrame({'Dates': times, 'Values ({0})'.format(units): values, 'Distances': distances,
                    #                        })
                else:
                    allvalues = self.Memory[key]['values']
                    alltimes = self.Memory[key]['times']
                    units = self.Memory[key]['units']
                    values = WF.getListItems(allvalues)
                    times = WF.getListItems(alltimes)
                    if isinstance(values, (list, np.ndarray)):
                        df = pd.DataFrame({'Dates': times, 'Values ({0})'.format(units): values})
                    elif isinstance(values, dict):
                        colvals = {}
                        colvals['Dates'] = times
                        for key in values:
                            colvals[key] = values[key]
                        df = pd.DataFrame(colvals)

                df.to_csv(csv_name, index=False)

            except:
                WF.print2stdout(f'ERROR WRITING CSV FILE {csv_name}')
                WF.print2stdout(traceback.format_exc(), debug=self.Report.debug)

    def scaleValuesByTable(self, data, line_settings):
        '''
        scales values based on table target values. Looks for SPECIFIC values, not ranges.
        :param data: dictionary containing data values
        :param line_settings: dictionary containing line settings
        :return: updated dictionary
        '''

        for d in data.keys():
            if 'scalartable' in line_settings[d].keys() and 'scalefrom' in line_settings[d].keys():
                WF.print2stdout(f'Scalar table found for {d}', debug=self.Report.debug)
                tablevalues = WDR.readScalarTable(line_settings[d]['scalartable'])
                scalefromflag = line_settings[d]['flag']+'_scalefrom'
                scalefrom_dates, scalefrom_values, scalefrom_units = self.getTimeSeries(line_settings[d]['scalefrom'])
                if len(scalefrom_values) != len(data[d]['values']):
                    WF.print2stdout(f'Values and Scaledby in different time intervals. Equalizing..', debug=self.Report.debug)
                    base_data, scaledFrom_data = WF.matchData({'dates': data[d]['dates'], 'values': data[d]['values']},
                                                         {'dates': scalefrom_dates, 'values': scalefrom_values})
                else:
                    base_data = data[d]
                    scaledFrom_data = {'values': scalefrom_values,
                                       'dates': scalefrom_dates,
                                       'units': scalefrom_units}

                #MAKE SURE NPARRAY
                for target in tablevalues.keys():
                    if target in scaledFrom_data['values']:
                        scalar = tablevalues[target]
                        target_i = [i for i, n in enumerate(scaledFrom_data['values']) if n == target]
                        base_data['values'][target_i] *= scalar

                data[d]['values'] = base_data['values']
                data[d][scalefromflag] = scaledFrom_data

        return data

