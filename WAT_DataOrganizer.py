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
import WAT_Profiles as WProfile

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

        very_special_flags = f'{self.Report.SimulationName.replace(" ", "").replace(":", "")}_{self.Report.baseSimulationName}'

        if 'dss_path' in Data_info.keys(): #Get data from DSS record
            if 'dss_filename' in Data_info.keys():
                outname = f"{os.path.basename(Data_info['dss_filename']).split('.')[0]}_" \
                          f"{Data_info['dss_path'].replace('/', '').replace(':', '')}_" \
                          f"{very_special_flags}"
                return outname

        elif 'w2_file' in Data_info.keys():
            if 'structurenumbers' in Data_info.keys():
                outname = '{0}'.format(os.path.basename(Data_info['w2_file']).split('.')[0])
                if isinstance(Data_info['structurenumbers'], dict):
                    structure_nums = [Data_info['structurenumbers']['structurenumber']]
                elif isinstance(Data_info['structurenumbers'], str):
                    structure_nums = [Data_info['structurenumbers']]
                elif isinstance(Data_info['structurenumbers'], (list, np.ndarray)):
                    structure_nums = Data_info['structurenumbers']
                outname += '_Struct_' + '_'.join(structure_nums) + f'_{very_special_flags}'
            else:
                outname = '{0}'.format(os.path.basename(Data_info['w2_file']).split('.')[0]) + f'_{very_special_flags}'
            return outname

        elif 'h5file' in Data_info.keys():
            h5name = os.path.basename(Data_info['h5file']).split('.h5')[0] + '_h5'
            if 'easting' in Data_info.keys() and 'northing' in Data_info.keys():
                outname = 'externalh5_{0}_{1}_{2}_{3}'.format(h5name, Data_info['parameter'], Data_info['easting'], Data_info['northing']) + f'_{very_special_flags}'
                return outname
            elif 'ressimname' in Data_info.keys():
                outname = 'externalh5_{0}_{1}_{2}'.format(h5name, Data_info['parameter'], Data_info['ressimresname']) + f'_{very_special_flags}'
                return outname

        elif 'easting' in Data_info.keys() and 'northing' in Data_info.keys():
            outname = '{0}_{1}_{2}'.format(Data_info['parameter'], Data_info['easting'], Data_info['northing']) + f'_{very_special_flags}'
            return outname

        elif 'filename' in Data_info.keys(): #Get data from Observed Profile
            outname = '{0}'.format(os.path.basename(Data_info['filename']).split('.')[0].replace(' ', '_')) + f'_{very_special_flags}'
            return outname

        elif 'w2_segment' in Data_info.keys():
            outname = 'W2_{0}_{1}_profile'.format(self.Report.ModelAlt.output_file_name.split('.')[0], Data_info['w2_segment']) + f'_{very_special_flags}'
            return outname

        elif 'ressimresname' in Data_info.keys():
            outname = '{0}_{1}_{2}'.format(os.path.basename(self.Report.ModelAlt.h5fname).split('.')[0] +'_h5',
                                           Data_info['parameter'], Data_info['ressimresname']) + f'_{very_special_flags}'

            return outname

        return 'NULL'

    #################################################################
    #TimeSeries Functions
    #################################################################

    def updateTimeSeriesDataDictionary(self, data, line):
        '''
        organizes line information and places it into a data dictionary
        :param data: dictionary containing line data
        :param line: dictionary containing line settings
        :return: updated data dictionary
        '''

        #TODO: split into 2 like profiles
        dates, values, units = self.getTimeSeries(line, makecopy=False) #TODO: update
        if WF.checkData(values):
            flag = line['flag']
            if flag in data.keys():
                count = 1
                newflag = flag + '_{0}'.format(count)
                while newflag in data.keys():
                    count += 1
                    newflag = flag + '_{0}'.format(count)
                flag = newflag
                WF.print2stdout('The new flag is {0}'.format(newflag))
            datamem_key = self.buildMemoryKey(line)
            if 'units' in line.keys() and units != None:
                units = line['units']
            data[flag] = {'values': values,
                          'dates': dates,
                          'units': units,
                          'numtimesused': line['numtimesused'],
                          'logoutputfilename': datamem_key}

            for key in line.keys():
                if key not in data[flag].keys():
                    data[flag][key] = line[key]
        return data

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

    def getTimeSeriesDataDictionary(self, settings):
        '''
        Gets profile line data from defined data sources in XML files
        :param settings: currently selected object settings dictionary
        :return: dictionary containing data and information about each data set
        '''

        data = {}
        if 'lines' in settings.keys():
            for line in settings['lines']:
                numtimesused = 0
                if 'flag' not in line.keys():
                    WF.print2stdout('Flag not set for line (Computed/Observed/etc)')
                    WF.print2stdout('Not plotting Line:', line)
                    continue

                elif line['flag'].lower() == 'computed':
                    for ID in self.Report.accepted_IDs:
                        curline = pickle.loads(pickle.dumps(line, -1))
                        curline = self.Report.configureSettingsForID(ID, curline)
                        if not self.Report.checkModelType(curline):
                            continue
                        curline['numtimesused'] = numtimesused
                        numtimesused += 1
                        data = self.updateTimeSeriesDataDictionary(data, curline)
                else:
                    if self.Report.currentlyloadedID != 'base':
                        line = self.Report.configureSettingsForID('base', line)
                    else:
                        line = WF.replaceflaggedValues(self.Report, line, 'modelspecific')
                    line['numtimesused'] = 0
                    if not self.Report.checkModelType(line):
                        continue
                    data = self.updateTimeSeriesDataDictionary(data, line)

        return data

    def getTimeSeries(self, Line_info, makecopy=True):
        '''
        gets time series data from defined sources
        :param Line_info: dictionary of line setttings containing datasources
        :return: dates, values, units
        '''

        if 'dss_path' in Line_info.keys(): #Get data from DSS record
            if 'dss_filename' not in Line_info.keys():
                WF.print2stdout('DSS_Filename not set for Line: {0}'.format(Line_info))
                return np.array([]), np.array([]), None
            else:
                datamem_key = self.buildMemoryKey(Line_info)
                if datamem_key in self.Memory.keys():
                    # WF.print2stdout('Reading {0} from memory'.format(datamem_key)) #noisy
                    if makecopy:
                        datamementry = pickle.loads(pickle.dumps(self.Memory[datamem_key], -1))
                    else:
                        datamementry = self.Memory[datamem_key]
                    times = datamementry['times']
                    values = datamementry['values']
                    units = datamementry['units']
                else:
                    times, values, units = WDR.readDSSData(Line_info['dss_filename'], Line_info['dss_path'],
                                                           self.Report.StartTime, self.Report.EndTime)

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
                # WF.print2stdout('READING {0} FROM MEMORY'.format(datamem_key))
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
                # WF.print2stdout('READING {0} FROM MEMORY'.format(datamem_key))
                datamementry = pickle.loads(pickle.dumps(self.Memory[datamem_key], -1))
                times = datamementry['times']
                values = datamementry['values']
                units = datamementry['units']

            else:
                filename = Line_info['h5file']
                if not os.path.exists(filename):
                    WF.print2stdout('ERROR: H5 file does not exist:', filename)
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
                # WF.print2stdout('READING {0} FROM MEMORY'.format(datamem_key))
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

        else:
            WF.print2stdout('No Data Defined for line')
            return np.array([]), np.array([]), None

        if 'omitvalue' in Line_info.keys():
            omitval = float(Line_info['omitvalue'])
            values = WF.replaceOmittedValues(values, omitval)

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
                WF.print2stdout('Flag not set for line (Computed/Observed/etc)')
                print('Not plotting Line:', line)
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
                    numtimesused += 1
                    data, line_settings = self.updateProfileDataDictionary(data, line_settings, curline, timestamps)
            else:
                if self.Report.currentlyloadedID != 'base':
                    line = self.Report.configureSettingsForID('base', line)
                else:
                    line = WF.replaceflaggedValues(self.Report, line, 'modelspecific')
                line['numtimesused'] = 0
                if not self.Report.checkModelType(line):
                    continue
                data, line_settings = self.updateProfileDataDictionary(data, line_settings, line, timestamps)

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

        vals, elevations, depths, times, flag = self.getProfileValues(profile, timestamps) #Test this speed for grabbing all profiles and then choosing
        if len(vals) > 0:
            datamem_key = self.buildMemoryKey(profile)
            if 'units' in profile.keys():
                units = profile['units']
            else:
                units = None

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

        return data, line_settings

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
            print('retrieving profile from datamem')
            if isinstance(timesteps, str): #if looking for all
                if dm['subset'] == 'false': #the last time data was grabbed, it was not a subset, aka all
                    return dm['values'], dm['elevations'], dm['depths'], dm['times'], Profile_info['flag']
                else:
                    print('Incorrect Timesteps in data memory. Re-extracting data for', datamemkey)
            elif np.array_equal(timesteps, dm['times']):
                return dm['values'], dm['elevations'], dm['depths'], dm['times'], Profile_info['flag']
            else:
                print('Incorrect Timesteps in data memory. Re-extracting data for', datamemkey)

        if 'filename' in Profile_info.keys(): #Get data from Observed
            filename = Profile_info['filename']
            values, yvals, times = WDR.readTextProfile(filename, timesteps, self.Report.StartTime, self.Report.EndTime)
            if 'y_convention' in Profile_info.keys():
                if Profile_info['y_convention'].lower() == 'depth':
                    return values, [], yvals, times, Profile_info['flag']
                elif Profile_info['y_convention'].lower() == 'elevation':
                    return values, yvals, [], times, Profile_info['flag']
                else:
                    print('Unknown value for flag y_convention: {0}'.format(Profile_info['y_convention']))
                    print('Please use "depth" or "elevation"')
                    print('Assuming depths...')
                    return values, [], yvals, times, Profile_info['flag']
            else:
                print('No value for flag y_convention')
                print('Assuming depths...')
                return values, [], yvals, times, Profile_info['flag']

        elif 'h5file' in Profile_info.keys() and 'ressimresname' in Profile_info.keys():
            filename = Profile_info['h5file']
            if not os.path.exists(filename):
                print('ERROR: H5 file does not exist:', filename)
                return [], [], [], [], Profile_info['flag']
            externalResSim = WDR.ResSim_Results('', '', '', '', self.Report, external=True)
            externalResSim.openH5File(filename)
            externalResSim.load_time() #load time vars from h5
            externalResSim.loadSubdomains()
            vals, elevations, depths, times = externalResSim.readProfileData(Profile_info['ressimresname'],
                                                                             Profile_info['parameter'], timesteps)
            return vals, elevations, depths, times, Profile_info['flag']

        elif 'w2_segment' in Profile_info.keys():
            if self.Report.plugin.lower() != 'cequalw2':
                return [], [], [], [], None
            vals, elevations, depths, times = self.Report.ModelAlt.readProfileData(Profile_info['w2_segment'], timesteps)
            if isinstance(timesteps, str):
                vals, elevations = WProfile.normalize2DElevations(vals, elevations)
            return vals, elevations, depths, times, Profile_info['flag']

        elif 'ressimresname' in Profile_info.keys():
            if self.Report.plugin.lower() != 'ressim':
                return [], [], [], [], None
            vals, elevations, depths, times = self.Report.ModelAlt.readProfileData(Profile_info['ressimresname'],
                                                                            Profile_info['parameter'], timesteps)
            return vals, elevations, depths, times, Profile_info['flag']

        print('No Data Defined for Profile')
        print('Profile:', Profile_info)
        return [], [], [], [], None

    def getReservoirContourDataDictionary(self, settings):
        '''
        Gets Contour Reservoir data from defined data sources in XML files
        :param settings: currently selected object settings dictionary
        :return: dictionary containing data and information about each data set
        '''

        data = {}
        for datapath in settings['datapaths']:
            for ID in self.Report.accepted_IDs:
                curreach = pickle.loads(pickle.dumps(datapath, -1))
                curreach = self.Report.configureSettingsForID(ID, curreach)
                if not self.Report.checkModelType(curreach):
                    continue
                # object_settings['timestamps'] = 'all'
                values, elevations, depths, dates, flag = self.getProfileValues(curreach, 'all')
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
                        flag = newflag
                        print('The new flag is {0}'.format(newflag))
                    datamem_key = self.buildMemoryKey(datapath)

                    if 'units' in datapath.keys():
                        units = datapath['units']
                    else:
                        units = None

                    data[flag] = {'values': values,
                                  'dates': dates,
                                  'units': units,
                                  'elevations': elevations,
                                  'topwater': topwater,
                                  'ID': ID,
                                  'logoutputfilename': datamem_key}

                    for key in datapath.keys():
                        if key not in data[flag].keys():
                            data[flag][key] = datapath[key]
        #reset
        self.Report.loadCurrentID('base')
        self.Report.loadCurrentModelAltID('base')
        return data

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
            print('retrieving profile topwater from datamem')
            if isinstance(timesteps, str): #if looking for all
                if dm['subset'] == 'false': #the last time data was grabbed, it was not a subset, aka all
                    return dm['topwater']
                else:
                    print('Incorrect Timesteps in data memory. Re-extracting data for', datamemkey)
            elif np.array_equal(timesteps, dm['times']):
                return dm['topwater']
            else:
                print('Incorrect Timesteps in data memory. Re-extracting data for', datamemkey)

        if 'filename' in profile.keys(): #Get data from Observed
            filename = profile['filename']
            values, yvals, times = WDR.readTextProfile(filename, timesteps, self.Report.StartTime, self.Report.EndTime)
            if 'y_convention' in profile.keys():
                if profile['y_convention'].lower() == 'elevation':
                    return [yval[0] for yval in yvals]
                elif profile['y_convention'].lower() == 'depth':
                    print('Unable to get topwater from depth.')
                    return []
                else:
                    print('Unknown value for flag y_convention: {0}'.format(profile['y_convention']))
                    print('Please use "elevation"')
                    print('Assuming elevations...')
                    return [yval[0] for yval in yvals]
            else:
                print('No value for flag y_convention')
                print('Assuming elevation...')
                return [yval[0] for yval in yvals]

        elif 'h5file' in profile.keys() and 'ressimresname' in profile.keys():
            filename = profile['h5file']
            if not os.path.exists(filename):
                print('ERROR: H5 file does not exist:', filename)
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

        print('No Data Defined for line')
        print('Profile:', profile)
        return []

    def commitProfileDataToMemory(self, data, line_settings, object_settings):
        '''
        commits updated data to data memory dictionary that keeps track of data
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
        for dp in object_settings['datapaths']:
            numtimesused = 0
            if 'flag' not in dp.keys():
                print('Flag not set for line (Computed/Observed/etc)')
                print('Not using Line:', dp)
                continue
            elif dp['flag'].lower() == 'computed':
                for ID in self.Report.accepted_IDs:
                    cur_dp = pickle.loads(pickle.dumps(dp, -1))
                    cur_dp = self.Report.configureSettingsForID(ID, cur_dp)
                    cur_dp['numtimesused'] = numtimesused
                    cur_dp['ID'] = ID
                    if not self.Report.checkModelType(cur_dp):
                        continue
                    numtimesused += 1
                    data = self.updateTimeSeriesDataDictionary(data, cur_dp)
            else:
                if self.Report.currentlyloadedID != 'base':
                    dp = self.Report.configureSettingsForID('base', dp)
                else:
                    dp = WF.replaceflaggedValues(self.Report, dp, 'modelspecific')
                dp['numtimesused'] = 0
                if not self.Report.checkModelType(dp):
                    continue
                data = self.updateTimeSeriesDataDictionary(data, dp)

        return data

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
                print('READING {0} FROM MEMORY'.format(datamem_key))
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
                print('READING {0} FROM MEMORY'.format(datamem_key))
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
        :param keyval: determines what key to iterate over for data
        :return: dictionary containing data and information about each data set
        '''

        data = {}
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
                        flag = newflag
                        print('The new flag is {0}'.format(newflag))
                    datamem_key = self.buildMemoryKey(reach)

                    if 'units' in reach.keys() and units == None:
                        units = reach['units']

                    if 'y_scalar' in settings.keys():
                        y_scalar = float(settings['y_scalar'])
                        distance *= y_scalar

                    data[flag] = {'values': values,
                                  'dates': dates,
                                  'units': units,
                                  'distance': distance,
                                  'ID': ID,
                                  'logoutputfilename': datamem_key}

                    for key in reach.keys():
                        if key not in data[flag].keys():
                            data[flag][key] = reach[key]
        #reset
        self.Report.loadCurrentID('base')
        self.Report.loadCurrentModelAltID('base')
        return data

    #################################################################
    #Gate Functions
    #################################################################

    def getGateDataDictionary(self, settings, makecopy=True):
        '''
        Gets profile line data from defined data sources in XML files
        :param settings: currently selected object settings dictionary
        :param keyval: determines what key to iterate over for data
        :return: dictionary containing data and information about each data set
        '''

        data = {}
        if 'gateops' in settings.keys():
            for gi, gateop in enumerate(settings['gateops']):

                if 'flag' in gateop.keys():
                    if gateop['flag'] not in data.keys():
                        data[gateop['flag']] = {}
                        gateopkey = gateop['flag']
                elif 'label' in gateop.keys():
                    if gateop['label'] not in data.keys():
                        data[gateop['label']] = {}
                        gateopkey = gateop['label']
                else:
                    if 'GATEOP_{0}'.format(gi) not in data.keys():
                        gateopkey = 'GATEOP_{0}'.format(gi)
                        data[gateopkey] = {}

                data[gateopkey]['gates'] = {}
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
                                                      'dates': dates,
                                                      'logoutputfilename': datamem_key,
                                                      'gategroup': gategroup}

                    for key in gate.keys():
                        if key not in data[gateopkey]['gates'][flag].keys():
                            data[gateopkey]['gates'][flag][key] = gate[key]

                for key in gateop.keys():
                    if key not in data[gateopkey].keys():
                        data[gateopkey][key] = gateop[key]

        return data

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
                print('ERROR WRITING CSV FILE')
                print(traceback.format_exc())
                with open(csv_name, 'w') as inf:
                    inf.write('ERROR WRITING FILE.')

