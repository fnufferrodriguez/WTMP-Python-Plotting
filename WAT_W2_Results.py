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
import sys

import numpy as np
import datetime as dt
import pandas as pd
from scipy.interpolate import interp1d
from collections import Counter
import linecache

import WAT_Functions as WF
import WAT_Time as WT

class W2_Results(object):

    def __init__(self, W2_path, alt_name, alt_Dir, starttime, endtime, Report, interval_min=60):
        '''
        Class Builder init.
        :param W2_path: path to W2 run
        :param alt_name: name of run alternative for pathing, ex: 'Shasta from DSS 14'
        :param alt_Dir: directory of the alternative
        :param starttime: starttime of simulation
        :param endtime: endtime of simulation
        :param interval_min: output time series interval in minutes (60 = 1HOUR, 15 = 4 outputs an hour)
        '''

        self.W2_path = W2_path
        self.alt_name = alt_name #confirm this terminology
        self.run_path = alt_Dir
        self.starttime = starttime
        self.endtime = endtime
        self.Report = Report
        self.interval_min = interval_min #output time series

        control_file_csv = os.path.join(self.run_path, 'w2_con.csv') #this should always be the same, UNTIL IT WASNT
        control_file_npt = os.path.join(self.run_path, 'w2_con.npt') #this should always be the same, UNTIL IT WASNT
        if os.path.exists(control_file_csv):
            self.control_file = control_file_csv
            self.control_file_type = 'csv'
        elif os.path.exists(control_file_npt):
            self.control_file = control_file_npt
            self.control_file_type = 'npt'
        else:
            WF.print2stderr('Unknown or missing W2 control file.')
            sys.exit(1)

        self.readControlFile()
        # dates are output irregular, so we need to build a regular time series to interpolate to
        self.jd_dates, self.dt_dates, self.t_offset = self.buildTimes(self.starttime, self.endtime, self.interval_min)
        if self.control_file_type == 'npt': #turn off, make user input full W2 file instead of trying to build it?
            self.getOutputFileName_NPT() #get the W2 sanctioned output file name convention
        elif self.control_file_type == 'csv':
            self.getOutputFileName_CSV()


    def readControlFile(self):
        '''
        Open control file lines and format them. Control file lines are split into "groups", usually based off of a
        header, and then the values. Control files are split into sections based off of spaces in the control file
        :return: set class variables:
                    self.cf_lines
                    self.line_sections
        '''

        self.cf_lines = self.getControlFileLines(self.control_file)
        if self.control_file_type == 'npt':
            self.line_sections = self.formatCFLines(self.cf_lines)
        else:
            self.line_sections = self.formatCFLines(self.cf_lines)
            # self.line_sections = [] #csv file output contains depths and elevations in the output, which is nice.
            #csv?

    def get_tempprofile_layers(self):
        '''
        gets profile layers from the control file
        :return: set class variables:
                    self.layers
        '''

        self.layers = self.getControlVariable(self.line_sections, 'TSR LAYE', pref_output_type=float)

    def getOutputFileName_NPT(self):
        '''
        gets the name of output files
        :return:
        '''

        # self.output_file_name = self.getControlVariable(self.line_sections, 'TSR FILE')[0]
        output_file_name = self.getControlVariable(self.line_sections, 'TSR FILE')
        if len(output_file_name) > 0:
            self.output_file_name = output_file_name[0]

    def getOutputFileName_CSV(self):
        '''
        gets the name of output files
        :return:
        '''

        output_file_name = self.getControlVariable(self.line_sections, 'TSR')
        if len(output_file_name) > 0:
            self.output_file_name = output_file_name[3]

    def getControlFileLines(self, control_file):
        '''
        reads control file
        :param input_file: full path to control file
        :return: np array of all lines in control file
        '''

        file_read = open(control_file, 'r')
        file_lines = file_read.readlines()
        file_read.close()
        return np.asarray(file_lines)

    def formatCFLines(self, cf_lines):
        '''
        seperates control file lines into sections, based off of spaces in the file. Control files are generally
        formatted like:

        LAYE header1 header2
                  2       5

        TEMP header1 header2
                  5       5

        File then splits both of the sections above into two seperate sections into a list for easier parsing. Sections
        are split based on the spaces between them.

        :param cf_lines: control file lines from self.get_control_file_lines()
        :return: a list of sections
        '''

        sections = []
        small_section = ''
        for line in cf_lines:
            if self.control_file_type == 'csv':
                line = ''.join(list(filter((',').__ne__, list(line))))
            if len(line.strip()) == 0 and len(small_section) == 0:
                continue
            if len(line.strip()) == 0 and len(small_section) != 0:
                sections.append(small_section)
                small_section = ''
            else:
                small_section += line

        return sections

    def getControlVariable(self, lines_sections, variable, pref_output_type=np.str_):
        '''
        Parses the split control file sections from self.format_cf_lines() for a wanted card. Cards usually preface
        headers in the contro file, see docuemntation. For the give example below...

        DLT MAX   DLTMAX  DLTMAX  DLTMAX  DLTMAX  DLTMAX  DLTMAX  DLTMAX  DLTMAX  DLTMAX
                  3600.00

        DLT FRN     DLTF    DLTF    DLTF    DLTF    DLTF    DLTF    DLTF    DLTF    DLTF
                   0.900

        If the user wanted the DLT Max value, they would search 'DLT MAX'. If there is only one instance of the flag,
        then return a single array. Else, output a list of arrays and let the user narrow it down from there.

        :param lines_sections: formatted control line sections from self.format_cf_lines()
        :param variable: intended variable card to find
        :param pref_output_type: preferred variable type for output (i.e. np.float, np.str, etc). Converts all values
                                 in output to intended type. If it fails (aka trying to convert a string value to a
                                 float, return strings instead.
        :return: either list of np arrays for multi output, or a single np.array
        '''

        variable_lines_idx = [i for i, line in enumerate(lines_sections) if variable in line]
        outputs = []
        for var_line_idx in variable_lines_idx:
            cur_otpt = []
            for line in lines_sections[var_line_idx].split('\n')[1:]: #skip header
                if line.strip() == '':
                    break
                sline = line.split()
                for s in sline:
                    cur_otpt.append(s)
            try:
                outputs.append(np.asarray(cur_otpt).astype(pref_output_type))
            except ValueError:
                WF.print2stdout('Array values not able to be converted to {0}'.format(pref_output_type), debug=self.Report.debug)
                WF.print2stdout('Reverting to strings.', debug=self.Report.debug)
                WF.print2stdout('Array:', cur_otpt, debug=self.Report.debug)
                outputs.append(np.asarray(cur_otpt).astype(np.string))
        if len(outputs) == 1:
            return outputs[0]
        return outputs

    def buildTimes(self, start_day, end_day, interval_min):
        '''
        Creates a regular time series. W2 time series are irregular, so we'll create a regular time series for output
        and then interpolate over it. Return a list of jdates (days past the jan 1, starting at 1) and a list of
        datetime values
        :param start_day: start day of the simulation (jdate)
        :param end_day: end day of the simulations (jdate)
        :param interval_min: the desired output time series interval in minutes, aka 60=Hourly, 15=15MIN (4 per hour)
        :return: list of dates in jdate format and datetime format
        '''

        # turns out jdates can be passed into timedelta (decimals) and it works correctly. Just subtract 1 becuase jdates
        # start at 1

        #Get the offset
        year = start_day.year
        t_offset = WT.datetime2Ordinal(dt.datetime(year, 1, 1, 0, 0))
        interval_perc_day = interval_min / (60 * 24)
        start_jdate = (WT.datetime2Ordinal(start_day) - t_offset) + 1
        end_jdate = (WT.datetime2Ordinal(end_day) - t_offset) + 1
        jd_dates = np.arange(start_jdate, end_jdate, interval_perc_day)
        dt_dates = [start_day+dt.timedelta(days=n-1) for n in jd_dates]

        return np.asarray(jd_dates), np.asarray(dt_dates), t_offset

    def readProfileData(self, seg, timesteps, resultsfile=None):
        '''
        Gets the temperature profile values from the output files.
        organizes results into np arrays. Arrays are full of Nan values by default for all possible values, then filled
        in where data applies. Water temps are organized into 2d arrays of dates - layers. This way, a user can index
        one date and get the temperature layers for that time step.
        Output values from W2 model come out in an irregular time series for some reason, so we will take the values,
        then interpolate them and find the interpolated values at the times we are looking for. Hopefully, these will
        still be very close to the values. This is how the pervious output method would put these into DSS.
        Water surface elevations are also needed for calculations of depth and elevation in main plotting script.
        These are the same for all layers at one time step, for valid layers.
        :return: array of water temperatures, elevations and depths
        '''

        if self.control_file_type == 'npt':
            values, elevations, depths, dates = self.readProfileData_NPT(seg, timesteps)
        elif self.control_file_type == 'csv':
            values, elevations, depths, dates = self.readProfileData_CSV(seg, timesteps, resultsfile)

        return values, elevations, depths, dates

    def readProfileData_CSV(self, seg, timesteps, resultsfile=None):
        '''
        output for CSV W2 runs are not always in the spr.csv file. Headers look like this
        Constituent	Julian_day	Depth	Elevation	Seg_111	Elevation	Seg_113
        :param seg:
        :param timesteps:
        :return:
        '''

        if resultsfile == None:
            resultsfile = self.getResultsFile_CSV()
        outputfile = os.path.join(self.run_path, resultsfile)
        if not os.path.exists(outputfile):
            WF.print2stdout(f'Results file {outputfile} does not exist.', debug=self.Report.debug)
            return [], [], [], []
        output = pd.read_csv(outputfile)
        segment_header = f'Seg_{seg}'
        # segment_index = np.where(output.columns.startswith(segment_header))[0] #should only ever be 1
        segment_index = [i for i, col in enumerate(output.columns) if col.startswith(segment_header)] #should only ever be 1
        if len(segment_index) == 0:
            WF.print2stdout(f'ERROR: segment {seg} not found in output file {outputfile}', debug=self.Report.debug)
            return [], [], [], []
        segment_elevation_header = output.columns[segment_index[0]-1]
        segment_value_header = output.columns[segment_index[0]] #stupid segment header can have a space at the end...
        all_jdates = output['Julian_day'].values #csv depths are always the same for all segments output
        all_depths = output['Depth'].values #csv jdates are always the same for all segments output
        all_values = output[segment_value_header].values
        all_elevations = output[segment_elevation_header].values
        unique_dates = np.asarray(list(set(all_jdates)))
        # number_layers = len(np.where(all_jdates == unique_dates[0])[0]) #get number of records for each timestamp.. hopefully these are always the same len?

        values = np.full(len(unique_dates), None)
        elevations = np.full(len(unique_dates), None)
        depths = np.full(len(unique_dates), None)

        for ln, unique_date in enumerate(unique_dates):
            indicies = np.where(all_jdates == unique_date)
            # print((len(indicies[0])))
            values[ln] = all_values[indicies]
            elevations[ln] = all_elevations[indicies]
            depths[ln] = all_depths[indicies]

        if len(unique_dates) == 0:
            WF.print2stdout('No values found in output.', debug=self.Report.debug)
            return [], [], [], []

        if isinstance(timesteps, (list, np.ndarray)):
            select_values = []
            select_elevations = []
            select_depths = []
            select_times = []
            for t, time in enumerate(timesteps):
                timestep = WT.getIdxForTimestamp(unique_dates, time, self.t_offset)
                if timestep > -1:#timestep in model
                    WSE = elevations[timestep] #Meters #get WSE
                    if not WF.checkData(WSE): #if WSE is bad, skip usually first timestep...
                        select_elevations.append(np.array([]))
                        select_depths.append(np.array([]))
                        select_values.append(np.array([]))
                        select_times.append(time)
                        continue

                    select_elevations.append(elevations[timestep][:] * 3.28084)
                    select_depths.append(depths[timestep][:] * 3.28084)
                    select_values.append(values[timestep][:])
                    select_times.append(time)

                else: #if timestep NOT in model, add empties
                    select_values.append(np.array([])) #find WTs
                    select_elevations.append(np.array([]))
                    select_depths.append(np.array([]))
                    select_times.append(time)

            select_values, select_elevations, select_depths = self.matchProfileLengths(select_values, select_elevations, select_depths)
            return select_values, select_elevations, select_depths, unique_dates,
        else:
            return values, elevations, depths, unique_dates


    def readProfileData_NPT(self, seg, timesteps):
        '''
        Reads profile data from NPT file
        :param seg: segment number
        :param timesteps: timesteps to get data for
        :return: profile data
        '''

        self.get_tempprofile_layers() #get the output layers. out at 2m depths

        wt = np.full((len(self.layers), len(self.jd_dates)), np.nan)
        WS_Elev = np.full((len(self.layers), len(self.jd_dates)), np.nan)

        for i in range(1,len(self.layers)+1):
            # WF.print2stdout('{0} of {1}'.format(i, len(self.layers)+1))
            ofn = '{0}_{1}_seg{2}.{3}'.format(self.output_file_name.split('.')[0],
                                              i,
                                              seg,
                                              self.output_file_name.split('.')[1])
            ofn_path = os.path.join(self.run_path, ofn)
            if not os.path.exists(ofn_path):
                WF.print2stdout('File {0} not found'.format(ofn_path))
                continue
            headerline=0
            with open(ofn_path) as ofnf:
                for li, line in enumerate(ofnf):
                    if line.lower().startswith('jday'):
                        headerline=li
                        break

            op_file = pd.read_csv(ofn_path, header=headerline, skip_blank_lines=False)
            op_file.columns = op_file.columns.str.lower()
            if len(op_file['jday']) > 1:
                WT_interp = interp1d(op_file['jday'], op_file['t2(c)'])
                Elev_interp = interp1d(op_file['jday'], op_file['elws(m)'])
                jdate_minmask = np.where(min(op_file['jday']) <= self.jd_dates)
                jdate_maxmask = np.where(max(op_file['jday']) >= self.jd_dates)
                jdate_msk = np.intersect1d(jdate_maxmask, jdate_minmask)
                wt_ts_Vals = np.full(len(self.jd_dates), np.nan)
                wsElev_ts_Vals = np.full(len(self.jd_dates), np.nan)
                wt_ts_Vals[jdate_msk] = WT_interp(self.jd_dates[jdate_msk])
                wsElev_ts_Vals[jdate_msk] = Elev_interp(self.jd_dates[jdate_msk])
                wt[i-1] = wt_ts_Vals
                WS_Elev[i-1] = wsElev_ts_Vals

        wt = np.asarray(wt).T
        WS_Elev = np.asarray(WS_Elev).T

        if isinstance(timesteps, (list, np.ndarray)):
            select_wt = []
            elevations = []
            depths = []
            times = []
            for t, time in enumerate(timesteps):
                e = []
                timestep = WT.getIdxForTimestamp(self.jd_dates, time, self.t_offset)
                if timestep > -1:#timestep in model
                    WSE = WS_Elev[timestep] #Meters #get WSE
                    if not WF.checkData(WSE): #if WSE is bad, skip usually first timestep...
                        elevations.append(np.array([]))
                        depths.append(np.array([]))
                        select_wt.append(np.array([]))
                        times.append(time)
                        continue
                    WSE = WSE[np.where(~np.isnan(WSE))][0] #otherwise find valid
                    WSE_array = np.full((self.layers.shape), WSE)
                    e = (WSE_array - self.layers) * 3.28084

                    select_wt.append(wt[timestep][:]) #find WTs

                else: #if timestep NOT in model, add empties
                    select_wt.append(np.array([])) #find WTs
                    elevations.append(np.array([]))
                    depths.append(np.array([]))
                    times.append(time)
                elevations.append(np.asarray(e)) #then append for timestep
                depths.append(self.layers * 3.28084) #append dpeths
                times.append(time) #get time
            select_wt, elevations, depths = self.matchProfileLengths(select_wt, elevations, depths)

            return select_wt, elevations, depths, np.asarray(times)
        else:
            elevations = (WS_Elev - self.layers) * 3.28084
            return wt, elevations, [], self.dt_dates

    def readProfileTopwater(self, seg, timesteps):
        '''
        gets the WSE for each timestep to filter for profile contour plots
        :param seg: segment number for profile
        :param timesteps: list of timesteps, or 'all'
        :return: list of WSE
        '''

        self.get_tempprofile_layers() #get the output layers. out at 2m depths

        WS_Elev = np.full((len(self.layers), len(self.jd_dates)), np.nan)

        for i in range(1,len(self.layers)+1):
            ofn = '{0}_{1}_seg{2}.{3}'.format(self.output_file_name.split('.')[0],
                                              i,
                                              seg,
                                              self.output_file_name.split('.')[1])
            ofn_path = os.path.join(self.run_path, ofn)
            if not os.path.exists(ofn_path):
                WF.print2stdout('File {0} not found'.format(ofn_path))
                continue
            headerline=0
            with open(ofn_path) as ofnf:
                for li, line in enumerate(ofnf):
                    if line.lower().startswith('jday'):
                        headerline=li
                        break

            op_file = pd.read_csv(ofn_path, header=headerline, skip_blank_lines=False)
            op_file.columns = op_file.columns.str.lower()
            if len(op_file['jday']) > 1:
                Elev_interp = interp1d(op_file['jday'], op_file['elws(m)'])
                jdate_minmask = np.where(min(op_file['jday']) <= self.jd_dates)
                jdate_maxmask = np.where(max(op_file['jday']) >= self.jd_dates)
                jdate_msk = np.intersect1d(jdate_maxmask, jdate_minmask)
                wsElev_ts_Vals = np.full(len(self.jd_dates), np.nan)
                wsElev_ts_Vals[jdate_msk] = Elev_interp(self.jd_dates[jdate_msk])
                WS_Elev[i-1] = wsElev_ts_Vals

        WS_Elev = np.asarray(WS_Elev).T

        if isinstance(timesteps, (list, np.ndarray)):

            WSE_out = []
            for t, time in enumerate(timesteps):
                timestep = WT.getIdxForTimestamp(self.jd_dates, time, self.t_offset)
                if timestep > -1:#timestep in model
                    WSE = WS_Elev[timestep] #Meters #get WSE
                    if not WF.checkData(WSE): #if WSE is bad, skip usually first timestep...
                        WSE_out.append(np.nan)
                        continue
                    WSE = WSE[np.where(~np.isnan(WSE))][0] #otherwise find valid
                    WSE_out.append(WSE)

                else: #if timestep NOT in model, add empties
                    WSE_out.append(np.nan)

            return WSE_out
        else:
            return WS_Elev[:,0] * 3.28084

    def readStructuredTimeSeries(self, output_file_name, structure_nums, skiprows=2):
        """
        output files usually have header with several repeat headers for each structure
         Branch:           1  # of structures:          23  outlet temperatures
        JDAY      T(C)      T(C)      T(C)      T(C)      T(C)      T(C)      T(C)      T(C)      T(C)      T(C)      T(C)      T(C)      T(C)      T(C)      T(C)      T(C)      T(C)      T(C)      T(C)      T(C)      T(C)      T(C)      T(C)   Q(m3/s)   Q(m3/s)   Q(m3/s)   Q(m3/s)   Q(m3/s)   Q(m3/s)   Q(m3/s)   Q(m3/s)   Q(m3/s)   Q(m3/s)   Q(m3/s)   Q(m3/s)   Q(m3/s)   Q(m3/s)   Q(m3/s)   Q(m3/s)   Q(m3/s)   Q(m3/s)   Q(m3/s)   Q(m3/s)   Q(m3/s)   Q(m3/s)   Q(m3/s)    ELEVCL    ELEVCL    ELEVCL    ELEVCL    ELEVCL    ELEVCL    ELEVCL    ELEVCL    ELEVCL    ELEVCL    ELEVCL    ELEVCL    ELEVCL    ELEVCL    ELEVCL    ELEVCL    ELEVCL    ELEVCL    ELEVCL    ELEVCL    ELEVCL    ELEVCL    ELEVCL
        structures are determined by number of each header
        :param output_file_name: name of output file in run path
        :param structure_nums: number values of structures we want to output
        :param skiprows: how many header rows to skip
        :return: dates, values
        """

        ofn_path = os.path.join(self.run_path, output_file_name)
        if not os.path.exists(ofn_path):
            WF.print2stdout(f'File {ofn_path} not found!')
            return [], []

        structure_nums = [int(n) for n in structure_nums]
        values = {}

        with open(ofn_path, 'r') as o:
            for i, line in enumerate(o):
                if i == skiprows-1:
                    headers = line.strip().lower().replace(',','').split()[1:] #skipjdate..
                    break
        header_count = Counter(headers)
        headers = list(set(headers))

        stsf = pd.read_csv(ofn_path, header=skiprows-1, delim_whitespace=True)
        stsf.columns = stsf.columns.str.lower()
        stsf.columns = [n.replace(',','') for n in stsf.columns]
        for structure_num in structure_nums:
            if structure_num < 0:
                structure_num = min(header_count.values()) + structure_num+1 #reverse index the fun way, use min len incase doesnt match for some reason
            if structure_num not in values.keys():
                values[structure_num] = {}
            for header in headers:
                if structure_num == 1:
                    hname = header
                else:
                    hname = header+'.{0}'.format(structure_num-1)
                vals = np.asarray([float(str(n).replace(',','')) for n in stsf[hname].tolist()])
                values[structure_num][header.lower()] = vals

        dates = stsf['jday'].tolist()
        dates = np.asarray([float(str(n).replace(',', '')) for n in dates])

        return dates, values

    def filterByParameter(self, values, line_info):
        '''
        W2 results files have multiple parameters in a single file, so we can return many parameters
        this grabs the parameter defined in the line
        :param values: dictionary of lists of values
        :param line_info: line settings dictionary containing values
        :return: values list, parameter from line settings
        '''

        headerparam = {'flow': 'q(m3/s)',
                       'temperature': 't(c)',
                       'waterlevel': 'elevcl'}

        if 'parameter' not in line_info.keys():
            WF.print2stdout('Parameter not specified.', debug=self.Report.debug)
            WF.print2stdout('Line Info:', line_info, debug=self.Report.debug)
            return values, ''
        new_values = {}
        target_header = headerparam[line_info['parameter']]
        for key in values.keys():
            new_values[key] = values[key][target_header]
        return new_values, line_info['parameter']

    def matchProfileLengths(self, select_val, elevations, depths):
        '''
        Matches lengths for values and elevs. Sometimes values not output at certain elevs
        :param select_val: selected values
        :param elevations: selected elevations
        :param depths: selected depths
        :return: trimmed values, elevations, depths
        '''

        len_val = len(select_val)
        len_elev = len(elevations)
        len_depth = len(depths)
        min_len = min((len_val, len_elev, len_depth))
        return select_val[:min_len], elevations[:min_len], depths[:min_len]

    def readTimeSeries(self, output_file_name, column=1, skiprows=3, **kwargs):
        '''
        get the output time series for W2 at a specified location. Like the temperature profiles, output freq is
        variable so we'll interpolate over a regular time series
        :param output_file_name: full path to the output file.
        :param targetfieldidx: Number of column to grab data from (0 index). Usually date, value.
        :return: np.array of dates, and the values
        '''

        out_vals = np.full(len(self.jd_dates), np.nan)

        try:
            column = int(column)
        except:
            pass

        try:
            skiprows = int(skiprows)
        except:
            pass


        ofn_path = os.path.join(self.run_path, output_file_name)

        if not os.path.exists(ofn_path):
            WF.print2stdout('Data File not found!', ofn_path)
            return [], []

        if isinstance(column, str):
            header = linecache.getline(ofn_path, int(skiprows)).strip().replace(' ','').lower().split(',') #1 indexed, for some reason
            cidx = np.where(np.asarray(header) == column.replace(' ','').lower())[0]
            if len(cidx) > 0:
                column = cidx[0]
            else:
                WF.print2stdout(f'Header {column} not found in file', debug=self.Report.debug)
                return [], []

        dates = []
        values = []
        with open(ofn_path, 'r') as o:
            for i, line in enumerate(o):
                if i >= int(skiprows):
                    sline = line.split(',')
                    if len(sline) == 1: #not csv TODO: figure out this but better..
                        sline = line.split()
                    dates.append(float(sline[0].strip()))
                    # if isinstance(column, int):
                    values.append(float(sline[column].strip()))
                    # elif isinstance(column, str):
                        # header = linecache.getline(ofn_path, int(skiprows)).strip().lower().split() #1 indexed, for some reason
                        # cidx = np.where(np.asarray(header) == column.lower())[0]
                        # values.append(float(sline[cidx].strip()))

        if len(dates) > 1:
            val_interp = interp1d(dates, values)
            for j, jd in enumerate(self.jd_dates):
                try:
                    out_vals[j] = val_interp(jd)
                except ValueError:
                    continue

        return self.dt_dates, out_vals

    def readSegment(self, filename, parameter):
        '''
        Temporary until we figure out how to do W2 contours
        :param filename: file for output
        :param parameter: parameter to get data for
        :return:
        '''
        read_param = self.getParameterFileStr(parameter)
        if read_param == None:
            return [], [], []
        ofn_path = os.path.join(self.run_path, filename)

        output_values = np.array([])
        segments = []
        dates = []
        checkForVar = False
        record_vals = False
        gotvalues = True
        with open(ofn_path, 'r') as otf:
            for line in otf:
                if checkForVar and line.strip() != '':
                    if parameter in line.lower():
                        recordVals = True
                        sline = line.lower().split(parameter).split()
                        month = sline[0]
                        day = sline[1]
                        year = sline[2]
                        time = sline[8]
                        hours = time.split('.')[0]
                        minutes = time.split('.')[1]
                        date = '{0} {1}, {2} {3}:{4}'.format(month, day, year, hours, minutes)
                        dates.append(dt.datetime.strptime(sline[0].strip(), '%B %d, %Y %H:%M'))
                # elif record_vals == True:
                #     if line.startswith(' Layer'):
                #         sline = line.split()
                #         for segnum in sline[2:]:
                #             if segnum not in output_values:

                elif line.startswith(' Model run at'):
                    checkForVar = True


        # otf.split('\n')


    def getParameterFileStr(self, parameter):
        '''
        gets parameter file name based on the input name
        :param parameter: desired parameter
        :return: formatted internal name
        '''

        #input:output
        fileparams = {'temperature': 'Temperature',
                      'density': 'Density',
                      'vertical eddy viscosity': 'Vertical eddy viscosity',
                      'velocity shear stress': 'Velocity shear stress',
                      'internal shear': 'Internal shear',
                      'bottom shear': 'Bottom shear',
                      'longitudinal momentum': 'Longitudinal momentum',
                      'horizontal density gradient': 'Horizontal density gradient',
                      'vertical momentum': 'Vertical momentum',
                      'horizontal pressure gradient': 'Horizontal pressure gradient',
                      'gravity term channel slope': 'Gravity term channel slope',
                      'horizontal velocity': 'Horizontal velocity',
                      'vertical velocity': 'Vertical velocity'}

        if parameter.lower() not in fileparams:
            WF.print2stdout('Parameter {0} not in acceptable parameters.'.format(parameter), debug=self.Report.debug)
            return None
        else:
            return fileparams[parameter.lower()]

    def getResultsFile_CSV(self):
        '''
        Finds the output file name in CSV files for profiles
        Ryan Miles says this is fixed and can be calculated, so here we are.
        Reference email form 10/11/2022 at 10:35
        :return: results file name
        '''

        rootrow = 768

        structureoffsetline = [int(i) for i in self.cf_lines[135].strip().split(',') if i]
        structureoffset = (max(5, max(structureoffsetline))) * 6

        constituentsoffsetline = [int(i) for i in self.cf_lines[21].strip().split(',') if i]
        #Offset = NGC + NSS + NAL + (NBOD * 3) + 32 + NZP + 4
        #NGC, NSS, NAL, NEP, NBOD, NMC, NZP
        NGC = constituentsoffsetline[0]
        NSS = constituentsoffsetline[1]
        NAL = constituentsoffsetline[2]
        NEP = constituentsoffsetline[3]
        NBOD = constituentsoffsetline[4]
        NMC = constituentsoffsetline[5]
        NZP = constituentsoffsetline[6]

        constituentsoffset = NGC + NSS + NAL + (NBOD * 3) + 32 + NZP + 4

        #Offset = Max(5, NEP) * 3
        epiphytonoffset = (max(5, NEP)) * 3

        #Offset = Max(5, NAL) + Max(5, NZP)
        zooplanktonoffset = max(5, NAL) + max(5, NZP)

        #Offset = Max(5, NMC) * 3
        macrophyteoffset = max(5, NMC) * 3

        totaloffset = structureoffset + constituentsoffset + epiphytonoffset + zooplanktonoffset + macrophyteoffset
        inputfile_idx = rootrow + totaloffset
        results_file = self.cf_lines[inputfile_idx-1].strip().split()[0] #excel is 1 idx..
        return results_file