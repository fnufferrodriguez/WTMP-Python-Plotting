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
        self.getOutputFileName() #get the W2 sanctioned output file name convention

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
        # else:
            #csv?

    def get_tempprofile_layers(self):
        '''
        gets profile layers from the control file
        :return: set class variables:
                    self.layers
        '''

        self.layers = self.getControlVariable(self.line_sections, 'TSR LAYE', pref_output_type=float)

    def getOutputFileName(self):
        '''
        gets the name of output files
        :return:
        '''

        self.output_file_name = self.getControlVariable(self.line_sections, 'TSR FILE')[0]

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
                WF.print2stdout('Array values not able to be converted to {0}'.format(pref_output_type))
                WF.print2stdout('Reverting to strings.')
                WF.print2stdout('Array:', cur_otpt)
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

    def readProfileData(self, seg, timesteps):
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
                    e = (WSE_array - self.layers) * 3.28

                    select_wt.append(wt[timestep][:]) #find WTs

                else: #if timestep NOT in model, add empties
                    select_wt.append(np.array([])) #find WTs
                    elevations.append(np.array([]))
                    depths.append(np.array([]))
                    times.append(time)
                elevations.append(np.asarray(e)) #then append for timestep
                depths.append(self.layers * 3.28) #append dpeths
                times.append(time) #get time
            select_wt, elevations, depths = self.matchProfileLengths(select_wt, elevations, depths)

            return select_wt, elevations, depths, np.asarray(times)
        else:
            elevations = (WS_Elev - self.layers) * 3.28
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
            return WS_Elev[:,0] * 3.28

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
            WF.print2stdout('Data File not found!', ofn_path)
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
            WF.print2stdout('Parameter not specified.')
            WF.print2stdout('Line Info:', line_info)
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
            header = linecache.getline(ofn_path, int(skiprows)).strip().lower().split() #1 indexed, for some reason
            cidx = np.where(np.asarray(header) == column.lower())[0]
            if len(cidx) > 0:
                column = cidx[0]

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
        :param filename:
        :param parameter:
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


        WF.print2stdout('stp')

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
            WF.print2stdout('Parameter {0} not in acceptable parameters.'.format(parameter))
            return None
        else:
            return fileparams[parameter.lower()]
