'''
Created on 7/14/2021
@author: scott
@organization: Resource Management Associates
@contact: scott@rmanet.com
@note: Class containing functions to read and handle CeQual-W2 results text files
documentation for W2 output is here: \\raid01\QA\WAT\W2 Documentation\W2 Documentation - V422
'''

import os
import numpy as np
import pandas as pd
import datetime as dt
from scipy.interpolate import interp1d

class W2_Results(object):

    def __init__(self, W2_path, region, alt_name, interval_min=60):
        '''
        Class Builder init.
        :param W2_path: path to W2 run
        :param region: name of region. determines the output segment and path, ex: 'shasta'
        :param alt_name: name of run alternative for pathing, ex: 'Shasta from DSS 14'
        :param interval_min: output time series interval in minutes (60 = 1HOUR, 15 = 4 outputs an hour)
        '''

        self.W2_path = W2_path
        self.region = region
        self.alt_name = alt_name #confirm this terminology
        self.interval_min = interval_min #output time series
        self.run_path = os.path.join(self.W2_path, 'CeQual-W2', region.capitalize(), alt_name)
        self.control_file = os.path.join(self.run_path, 'w2_con.npt') #this should always be the same
        self.read_control_file()
        self.get_time_info()
        # dates are output irregular, so we need to build a regular time series to interpolate to
        self.jd_dates, self.dt_dates = self.build_times(self.start_day, self.end_day, self.year, self.interval_min)

    def get_time_info(self):
        '''
        Read control file and grab the TIME line. Should be the only line with TIME in the header "card"
        :return: sets class variables:
                    self.start_day
                    self.end_day
                    self.year
        '''
        time_lines = self.get_control_variable(self.line_sections, 'TIME', pref_output_type=np.float)
        self.start_day = float(time_lines[0])
        self.end_day = float(time_lines[1])
        self.year = float(time_lines[2])

    def read_control_file(self):
        '''
        Open control file lines and format them. Control file lines are split into "groups", usually based off of a
        header, and then the values. Control files are split into sections based off of spaces in the control file
        :return: set class variables:
                    self.cf_lines
                    self.line_sections
        '''
        self.cf_lines = self.get_control_file_lines(self.control_file)
        self.line_sections = self.format_cf_lines(self.cf_lines)

    def get_tempprofile_layers(self):
        '''
        gets profile layers from the control file
        :return: set class variables:
                    self.layers
        '''

        self.layers = self.get_control_variable(self.line_sections, 'TSR LAYE', pref_output_type=np.float)

    def get_outputfile_name(self):
        '''
        gets the name of output files
        :return:
        '''

        self.output_file_name = self.get_control_variable(self.line_sections, 'TSR FILE')[0]

    def get_control_file_lines(self, control_file):
        '''
        reads control file
        :param input_file: full path to control file
        :return: np array of all lines in control file
        '''

        file_read = open(control_file, 'r')
        file_lines = file_read.readlines()
        file_read.close()
        return np.asarray(file_lines)

    def format_cf_lines(self, cf_lines):
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

    def get_control_variable(self, lines_sections, variable, pref_output_type=np.str):
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
                print('Array values not able to be converted to {0}'.format(pref_output_type))
                print('Reverting to strings.')
                print('Array:', cur_otpt)
                outputs.append(np.asarray(cur_otpt).astype(np.string))
        if len(outputs) == 1:
            return outputs[0]
        return outputs

    def build_times(self, start_day, end_day, year, interval_min):
        '''
        Creates a regular time series. W2 time series are irregular, so we'll create a regular time series for output
        and then interpolate over it. Return a list of jdates (days past the jan 1, starting at 1) and a list of
        datetime values
        :param start_day: start day of the simulation (jdate)
        :param end_day: end day of the simulations (jdate)
        :param year: year of simulation
        :param interval_min: the desired output time series interval in minutes, aka 60=Hourly, 15=15MIN (4 per hour)
        :return: list of dates in jdate format and datetime format
        '''

        # turns out jdates can be passed into timedelta (decimals) and it works correctly. Just subtract 1 becuase jdates
        # start at 1
        dt_start_day = dt.datetime(int(year),1,1,0,0) + dt.timedelta(days=start_day-1)
        interval_perc_day = interval_min / (60 * 24)
        jd_dates= np.arange(start_day, end_day, interval_perc_day)
        dt_dates = [dt_start_day+dt.timedelta(days=n-1) for n in jd_dates]

        return jd_dates, dt_dates

    def get_TemperatureProfile(self):
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
        :return:
        '''

        #not sure where this is defined for W2.. for future get this from control file?
        if self.region.lower() == 'shasta':
            seg = 77
        elif self.region.lower() == 'keswick':
            seg = 32

        self.get_tempprofile_layers() #get the output layers. out at 2m depths
        self.get_outputfile_name() #get the W2 sanctioned output file name convention

        wt = np.full((len(self.jd_dates), len(self.layers)), np.nan)
        WS_Elev = np.full((len(self.jd_dates), len(self.layers)), np.nan)

        for i in range(1,len(self.layers)+1):
            print('{0} out of {1}'.format(i, len(self.layers)))
            ofn = '{0}_{1}_seg{2}.{3}'.format(self.output_file_name.split('.')[0],
                                              i,
                                              seg,
                                              self.output_file_name.split('.')[1])
            ofn_path = os.path.join(self.run_path, ofn)
            op_file = pd.read_csv(ofn_path, header=0)
            if len(op_file['JDAY']) > 1:
                WT_interp = interp1d(op_file['JDAY'], op_file['T2(C)'])
                Elev_interp = interp1d(op_file['JDAY'], op_file['ELWS(m)'])
                for j, jd in enumerate(self.jd_dates):
                    try:
                        wt[j][i] = WT_interp(jd)
                        WS_Elev[j][i] = Elev_interp(jd)
                    except ValueError:
                        continue

        return self.dt_dates, wt, WS_Elev

    def get_Timeseries(self, output_file_name, targetfieldidx=1):
        '''
        get the output time series for W2 at a specified location. Like the temperature profiles, output freq is
        variable so we'll interpolate over a regular time series
        :param output_file_name: full path to the output file.
        :param targetfieldidx: Number of column to grab data from (0 index). Usually date, value.
        :return: np.array of dates, and the values
        '''

        out_vals = np.full(len(self.jd_dates), np.nan)

        ofn_path = os.path.join(self.run_path, output_file_name)
        dates = []
        values = []
        skiplines = 3 #not sure if this is always true?
        with open(ofn_path, 'r') as o:
            for i, line in enumerate(o):
                if i >= skiplines:
                    sline = line.split(',')
                    dates.append(float(sline[0].strip()))
                    values.append(float(sline[targetfieldidx].strip()))

        if len(dates) > 1:
            val_interp = interp1d(dates, values)
            for j, jd in enumerate(self.jd_dates):
                try:
                    out_vals[j] = val_interp(jd)
                except ValueError:
                    continue

        return self.dt_dates, out_vals

if __name__ == '__main__':
    path = r"\\wattest\C\WAT\USBR_FrameworkTest_r3\runs\Shasta-Keswick_W2\2014"
    region = 'shasta'
    alt_name = 'Shasta from DSS 14'
    trw2 = W2_Results(path, region, alt_name)
    dates, water_temps, elevations = trw2.get_TemperatureProfile()
    dates, water_temps2 = trw2.get_Timeseries('two_77.opt')
    print('done')