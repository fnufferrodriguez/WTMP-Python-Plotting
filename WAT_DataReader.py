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
import os, sys
import numpy as np
import pandas as pd
import datetime as dt
import h5py
from scipy.interpolate import interp1d
from collections import Counter
from pydsstools.heclib.dss import HecDss
# from pydsstools.heclib.utils import dss_logging
# dss_logging.config(level="None")
import xml.etree.ElementTree as ET
import linecache
import WAT_Functions as WF
import pendulum

def definedVarCheck(Block, flags):
    '''
    confirms that all flags are contained in the given block, aka check for headers in XML
    :param Block: xml with flags
    :param flags: flags that we need for it to be valid
    :return: True if contains headers, False if not
    '''

    tags = [n.tag for n in list(Block)]
    for flag in flags:
        if flag not in tags:
            return False
    return True

def readSimulationFile(simulation_name, studyfolder, iscomp=False):
    '''
       Read the right csv file, and determine what region you are working with.
       Simulation CSV files are named after the simulation, and consist of plugin, model alter name, and then region(s)
       :param simulation_name: name of simulation to find file
       :param studyfolder: full path to study folder
       :returns: dictionary containing information from file
    '''

    if iscomp:
        simulation_file = os.path.join(studyfolder, 'reports', '{0}_comparison.csv'.format(simulation_name.replace(' ', '_')))
    else:
        simulation_file = os.path.join(studyfolder, 'reports', '{0}.csv'.format(simulation_name.replace(' ', '_')))
    print('Attempting to read {0}'.format(simulation_file))
    if not os.path.exists(simulation_file):
        print('ERROR: no Simulation CSV file {0}.csv for simulation: {1}'.format(simulation_name.replace(' ', '_'), simulation_name))
        sys.exit()
    sim_info = {}
    with open(simulation_file, 'r') as sf:
        for i, line in enumerate(sf):
            sline = line.strip().split(',')
            sim_info[i] = {'deffile': sline[-1].strip()} #comparison reports always put xml last
            sline = sline[:-1]
            sim_info[i]['plugins'] = []
            sim_info[i]['modelaltnames'] = []
            for si, s in enumerate(sline):
                if si % 2 == 0: #even
                    sim_info[i]['plugins'].append(s.strip())
                else: #odd
                    sim_info[i]['modelaltnames'].append(s.strip())
    return sim_info

def readGraphicsDefaults(GD_file):
    '''
    reads graphics default file and iterates through to form dictionary
    :param GD_file: graphics default file path
    :return: dictionary with settings
    '''

    tree = ET.parse(GD_file)
    root = tree.getroot()
    gd_reportObjects = root.findall('ReportObject')
    reportObjects = iterateGraphicsDefaults(gd_reportObjects, 'Type')

    return reportObjects

def readDefaultLineStyle(linefile):
    '''
    reads default linestyle file and iterates through to form dictionary
    :param linefile: linestyle default file path
    :return: dictionary with settings
    '''

    tree = ET.parse(linefile)
    root = tree.getroot()
    def_lineTypes = root.findall('LineType')
    lineTypes = iterateGraphicsDefaults(def_lineTypes, 'Name')
    return lineTypes

def readChapterDefFile(CD_file):
    '''
    reads chapter definitions file
    :param CD_file: xml of read chapter definitions file
    :return:
    '''

    Chapters = []
    tree = ET.parse(CD_file)
    root = tree.getroot()
    for chapter in root:
        check = definedVarCheck(chapter, ['Name', 'Region', 'Sections'])
        if check:
            ChapterDef = {}
            chap_name = chapter.find('Name').text
            chap_region = chapter.find('Region').text
            ChapterDef['name'] = chap_name
            ChapterDef['region'] = chap_region
            ChapterDef['sections'] = []
            cd_sections = chapter.findall('Sections/Section')
            for section in cd_sections:
                section_objects = {}
                try:
                    section_objects['header'] = section.find('Header').text
                except:
                    section_objects['header'] = ''
                section_objects['objects'] = []
                sec_objects = section.findall('Object')
                objects = iterateChapterDefintions(sec_objects)
                section_objects['objects'] = (objects)
                ChapterDef['sections'].append(section_objects)
        Chapters.append(ChapterDef)

    return Chapters


def readDSSData(dss_file, pathname, startdate, enddate):
    '''
    calls pydsstools from https://github.com/gyanz/pydsstools to read dss data
    :param dss_file: full path to DSS file
    :param pathname: path to read in DSS file
    :param startdate: start date datetime object
    :param enddate: end date datetime object
    :return: time array and values array and units

    Example from documentation
    # dss_file = "example.dss"
    # pathname = "/REGULAR/TIMESERIES/FLOW//1HOUR/Ex1/"
    # startDate = "15JUL2019 19:00:00"
    # endDate = "15AUG2019 19:00:00"
    '''

    startDatestr = startdate.strftime('%d%b%Y %H:%M:%S')
    endDatestr = enddate.strftime('%d%b%Y %H:%M:%S')

    if not os.path.exists(dss_file):
        print('DSS file not found!', dss_file)
        return [], [], None

    fid = HecDss.Open(dss_file)
    ts = fid.read_ts(pathname,window=(startDatestr,endDatestr),regular=True,trim_missing=False)
    if ts.empty: #if empty, it must be the path or time window. DSS record must exist
        print('Invalid Timeseries record path of {0} or time window of {1} - {2}'.format(pathname, startDatestr, endDatestr))
        print('Please check these parameters and rerun.')
        return [], [], None
    values = np.array(ts.values)
    values[ts.nodata] = np.nan

    made_ts = False
    if ts.dtype == 'Regular TimeSeries':
        interval_seconds = ts.interval
        times = []
        current_time = startdate
        while current_time <= enddate:
            times.append(current_time)
            current_time += dt.timedelta(seconds=interval_seconds)
        if len(times) == len(values):
            made_ts = True

    if not made_ts:
        times = np.asarray(ts.pytimes)
        print('Irregular DSS detected with {0} in {1}'.format(pathname, dss_file))
        print('Recommend changing to regular time series for speed increases.')
    else:
        times = np.asarray(times)

    units = ts.units

    return times, values, units

def formatPyDSSToolsDates(datestring):
    try:
        ts_stime = dt.datetime.strptime(datestring, '%d%b%Y %H:%M:%S')
    except ValueError:
        ts_stime_splt = datestring.split(' ')
        if ts_stime_splt[1][:2] == '24':
            ts_stime_splt[1] = '00' + ts_stime_splt[1][2:]
            datestring = ' '.join(ts_stime_splt)
            ts_stime = dt.datetime.strptime(datestring, '%d%b%Y %H:%M:%S')
            ts_stime += dt.timedelta(days=1)
    return ts_stime


def readW2ResultsFile(output_file_name, jd_dates, run_path, targetfieldidx=1):
    '''
    reads W2 output text files. files are a bit specialized so the targetfieldidx is variable and allows input
    :param output_file_name: name of file
    :param jd_dates: list of jdates. W2 output is in the form of jdates
    :param run_path: path of W2 run
    :param targetfieldidx: index of which row in CSV to grab
    :return: list of list of arrays
    '''

    out_vals = np.full(len(jd_dates), np.nan)
    ofn_path = os.path.join(run_path, output_file_name)
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
    for j, jd in enumerate(jd_dates):
        try:
            out_vals[j] = val_interp(jd)
        except ValueError:
            continue

    return out_vals

def readTextProfile(observed_data_filename, timestamps):
    '''
    reads in observed data files and returns values for Temperature Profiles
    :param observed_data_filename: file name
    :param timestamps: list of selected timestamps
    :return: returns values, depths and times
    '''

    t = []
    wt = []
    d = []
    t_profile = []
    wt_profile = []
    d_profile = []
    last_dtstr = ''
    last_dt = dt.datetime(1933, 10, 15)
    hold_dt = dt.datetime(1933, 10, 15) #https://www.onthisday.com/date/1933/october/15 sorry Steve
    if not os.path.exists(observed_data_filename):
        print('Observed data at {0} does not exist.'.format(observed_data_filename))
        return [], [], []
    with open(observed_data_filename, 'r') as odf:
        for j, line in enumerate(odf):
            if j == 0:
                continue
            sline = line.split(',')
            dt_str = sline[0]
            if dt_str == last_dtstr:
                dt_tmp = last_dt
            else:
                # dt_tmp = dateutil.parser.parse(dt_str)
                dt_tmp = pendulum.parse(dt_str, strict=False).replace(tzinfo=None)
                last_dtstr = dt_str
                last_dt = dt_tmp
            # if (dt_tmp.year != hold_dt.year or dt_tmp.month != hold_dt.month or dt_tmp.day != hold_dt.day): #if its a new date
            if (dt_tmp != hold_dt): #if its a new date
                if len(t_profile) != 0 and len(wt_profile) != 0 and len(d_profile) != 0:
                    t.append(np.array(t_profile))
                    wt.append(np.array(wt_profile))
                    d.append(np.array(d_profile))
                t_profile = []
                wt_profile = []
                d_profile = []
            else:
                if float(sline[2]) not in d_profile:
                    t_profile.append(dt_tmp)
                    wt_profile.append(float(sline[1]))
                    d_profile.append(float(sline[2]))
            hold_dt = dt_tmp

    if len(t_profile) != 0 and len(wt_profile) != 0 and len(d_profile) != 0:
        t.append(np.array(t_profile))
        wt.append(np.array(wt_profile))
        d.append(np.array(d_profile))

    cti = getClosestTime(timestamps, [n[0] for n in t])
    wtn = []
    dn = []
    ts = []

    for ci, i in enumerate(cti):
        if i != None:
            wtn.append(np.asarray(wt[i]))
            dn.append(np.asarray(d[i]))
            ts.append(timestamps[ci])
        else:
            wtn.append(np.array([]))
            dn.append(np.array([]))
            ts.append(timestamps[ci])

    return wtn, dn, np.asarray(ts)

def getTextProfileDates(observed_data_filename, starttime, endtime):
    '''
    reads profile text files and extracts dates
    :param observed_data_filename: filename of obs data
    :param starttime: start time for data
    :param endtime: end time for data
    :return: list of dates between start and end dates
    '''

    t = []
    if not os.path.exists(observed_data_filename):
        return t
    with open(observed_data_filename, 'r') as odf:
        for j, line in enumerate(odf):
            if j == 0:
                continue
            sline = line.split(',')
            dt_str = sline[0]
            # dt_tmp = dateutil.parser.parse(dt_str) #trying this to see if it will work
            dt_tmp = pendulum.parse(dt_str, strict=False).replace(tzinfo=None) #trying this to see if it will work
            if starttime <= dt_tmp <= endtime: #get time window
                if dt_tmp not in t:
                    t.append(dt_tmp)
    return np.asarray(t)

def getClosestTime(timestamps, dates):
    '''
    gets timestamp closest to given timestamps for profile plots
    #TODO: set some limit?
    :param timestamps: list of target timestamps
    :param dates: dates in file
    :return:
    '''

    cdi = [] #closest date index
    for timestamp in timestamps:
        closestDateidx = None
        # closestDateDist = 9999999999999999999999 #seconds, so this needs to be sorta big.
        closestDateDist = 86400 #1 day in seconds
        for i, date in enumerate(dates):
            timedelt = abs((timestamp-date).total_seconds())
            if timedelt < closestDateDist:
                closestDateidx = i
                closestDateDist = timedelt

        cdi.append(closestDateidx)
    return cdi

def getchildren(root, returnkeyless=False):
    '''
    recursive function that will read through settings and break down into smaller components. Forms dict relationships
    if simple value- label. Forms lists of several of the same flag within a flag.
    :param root: section of XML to be read and config
    :param returnkeyless: returns dict if false, list if true
    :return: dictionary processed values
    '''

    children = {}
    if len(root) == 0:
        try:
            if len(root.text.strip()) == 0:
                children[root.tag.lower()] = None
            else:
                children[root.tag.lower()] = root.text.strip()
        except:
            children[root.tag.lower()] = root.text
    else:
        children[root.tag.lower()] = []
        for subroot in root: #figure out if we have a list or dict..
            if Counter([n.tag for n in root])[subroot.tag] != 1:
                break
            if len(subroot.text.strip()) != 0: #empty meaning another def, aka "line" in "Lines"
                children[root.tag.lower()] = {}
                break
        for subroot in root:
            if isinstance(children[root.tag.lower()], list):
                children[root.tag.lower()].append(getchildren(subroot, returnkeyless=True))
            elif isinstance(children[root.tag.lower()], dict):
                children[root.tag.lower()].update(getchildren(subroot))

    if returnkeyless:
        children = children[root.tag.lower()]
    return children

def iterateGraphicsDefaults(root, main_key):
    '''
    iterates through the graphics default file to get settings
    :param root: section of XML to be read
    :param main_key: Name of main id for xml
    :return: dictionary with settings
    '''

    out = {}
    for cr in root:
        key = cr.find(main_key).text.lower()
        out[key.lower()] = {}
        for child in cr:
            if child.tag == main_key:
                continue
            else:
                out[key.lower()][child.tag.lower()] = getchildren(child, returnkeyless=True)
    return out

def iterateChapterDefintions(root):
    '''
    iterates through chapter definition file
    :param root: section of xml to be read
    :return: list of settings
    '''

    out = []
    for cr in root:
        keylist = {}
        for child in cr:
            keylist[child.tag.lower()] = getchildren(child, returnkeyless=True)
        out.append(keylist)
    return out

class W2_Results(object):

    def __init__(self, W2_path, alt_name, alt_Dir, starttime, endtime, interval_min=60):
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
        self.interval_min = interval_min #output time series
        self.control_file = os.path.join(self.run_path, 'w2_con.npt') #this should always be the same
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
        self.line_sections = self.formatCFLines(self.cf_lines)

    def get_tempprofile_layers(self):
        '''
        gets profile layers from the control file
        :return: set class variables:
                    self.layers
        '''

        self.layers = self.getControlVariable(self.line_sections, 'TSR LAYE', pref_output_type=np.float)

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

    # def getControlVariable(self, lines_sections, variable, pref_output_type=np.str):
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
                print('Array values not able to be converted to {0}'.format(pref_output_type))
                print('Reverting to strings.')
                print('Array:', cur_otpt)
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
        t_offset = WF.datetime2Ordinal(dt.datetime(year, 1, 1, 0, 0))
        interval_perc_day = interval_min / (60 * 24)
        start_jdate = (WF.datetime2Ordinal(start_day) - t_offset) + 1
        end_jdate = (WF.datetime2Ordinal(end_day) - t_offset) + 1
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

        unique_times = timesteps
        print('Num unique times:', len(unique_times))

        self.get_tempprofile_layers() #get the output layers. out at 2m depths
        # self.getOutputFileName() #get the W2 sanctioned output file name convention

        # wt = np.full((len(self.jd_dates), len(self.layers)), np.nan)
        # WS_Elev = np.full((len(self.jd_dates), len(self.layers)), np.nan)

        # wt = []
        # WS_Elev = []
        wt = np.full((len(self.layers), len(self.jd_dates)), np.nan)
        WS_Elev = np.full((len(self.layers), len(self.jd_dates)), np.nan)

        for i in range(1,len(self.layers)+1):
            # print('{0} of {1}'.format(i, len(self.layers)+1))
            ofn = '{0}_{1}_seg{2}.{3}'.format(self.output_file_name.split('.')[0],
                                              i,
                                              seg,
                                              self.output_file_name.split('.')[1])
            ofn_path = os.path.join(self.run_path, ofn)
            if not os.path.exists(ofn_path):
                print('File {0} not found'.format(ofn_path))
                continue
            op_file = pd.read_csv(ofn_path, header=0)
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
                # wt.append(wt_ts_Vals)
                # WS_Elev.append(wsElev_ts_Vals)
                # wt.append(WT_interp(self.jd_dates[jdate_msk]))
                # WS_Elev.append(Elev_interp(self.jd_dates[jdate_msk]))
                # for j, jd in enumerate(self.jd_dates):
                #     try:
                #         wt[j][i-1] = WT_interp(jd)
                #         WS_Elev[j][i-1] = Elev_interp(jd)
                #     except ValueError:
                #         continue

        wt = np.asarray(wt).T
        WS_Elev = np.asarray(WS_Elev).T

        select_wt = []
        elevations = []
        depths = []
        times = []
        for t, time in enumerate(unique_times):
            # print('{0} of {1}'.format(t, len(unique_times)))
            e = []
            timestep = WF.getIdxForTimestamp(self.jd_dates, time, self.t_offset)
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
                # for depth in self.layers: #get all depths
                #     e.append((WSE - depth) * 3.28) #conv to feet
                # select_wt[t] = wt[timestep][:] #find WTs
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
        # return select_wt, elevations, np.asarray(depths), np.asarray(times)

        return select_wt, elevations, depths, np.asarray(times)

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
            print('Data File not found!', ofn_path)
            return [], []
        if isinstance(structure_nums, dict):
            structure_nums = [structure_nums['structurenumber']]
        elif isinstance(structure_nums, str):
            structure_nums = [structure_nums]
        structure_nums = [int(n) for n in structure_nums]
        values = {}

        with open(ofn_path, 'r') as o:
            for i, line in enumerate(o):
                if i == skiprows-1:
                    headers = line.strip().lower().replace(',','').split()[1:] #skipjdate..
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
                # vals = [float(n) for n in stsf[hname].tolist()]
                values[structure_num][header.lower()] = vals

        dates = stsf['jday'].tolist()
        dates = np.asarray([float(str(n).replace(',', '')) for n in dates])
        # dates = [float(n) for n in dates]

        return dates, values

    def filterByParameter(self, values, Line_info):
        '''
        W2 results files have multiple parameters in a single file, so we can return many parameters
        this grabs the parameter defined in the line
        :param values: dictionary of lists of values
        :param Line_info: line settings dictionary containing values
        :return: values list, parameter from line settings
        '''

        headerparam = {'flow': 'q(m3/s)',
                       'temperature': 't(c)',
                       'waterlevel': 'elevcl'}

        if 'parameter' not in Line_info.keys():
            print('Parameter not specified.')
            print('Line Info:', Line_info)
            return values, ''
        new_values = {}
        target_header = headerparam[Line_info['parameter']]
        for key in values.keys():
            new_values[key] = values[key][target_header]
        return new_values, Line_info['parameter']

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

    # def readTimeSeries(self, output_file_name, column=1, skiprows=3, **args):
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
        except ValueError:
            pass #keep as string

        ofn_path = os.path.join(self.run_path, output_file_name)

        if not os.path.exists(ofn_path):
            print('Data File not found!', ofn_path)
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
                    if isinstance(column, int):
                        values.append(float(sline[column].strip()))
                    elif isinstance(column, str):
                        header = linecache.getline(ofn_path, int(skiprows)-1).strip().lower().split()
                        cidx = np.where(np.asarray(header) == column.lower())[0]
                        values.append(float(sline[cidx].strip()))

        if len(dates) > 1:
            val_interp = interp1d(dates, values)
            for j, jd in enumerate(self.jd_dates):
                try:
                    out_vals[j] = val_interp(jd)
                except ValueError:
                    continue

        return self.dt_dates, out_vals

    def readSegment(self, filename, parameter):
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


        print('stp')

        # otf.split('\n')


    def getParameterFileStr(self, parameter):
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
            print('Parameter {0} not in acceptable parameters.'.format(parameter))
            return None
        else:
            return fileparams[parameter.lower()]


class ResSim_Results(object):

    def __init__(self, simulationPath, alternativeName, starttime, endtime):
        '''
        Class Builder init.
        :param simulationPath: full path to ResSim simulation
        :param alternativeName: Name of selected Ressim Alternative run
        :param starttime: start time datetime object
        :param endtime: end time datetime object
        '''

        self.simulationPath = simulationPath
        self.alternativeName = alternativeName.replace(':', '_')
        self.starttime = starttime
        self.endtime = endtime

        self.getH5File()
        self.load_time() #load time vars from h5
        self.loadSubdomains()

    def getH5File(self):
        '''
        build h5 file name and open file
        :return: class variable
                    self.h
        '''

        h5filefrmt = self.alternativeName.replace(' ', '_')
        self.h5fname = os.path.join(self.simulationPath, 'rss', h5filefrmt + '.h5')
        self.h = h5py.File(self.h5fname, 'r')

    def load_time(self):
        '''
        Loads times from H5 file
        :return: set class variables
                    self.tstr - time string
                    self.t - times list
                    self.nt - num times
                    self.t_offset - time offset
        '''

        self.tstr = self.h['Results/Subdomains/Time Date Stamp']
        tstr0 = (self.tstr[0]).decode("utf-8")
        ttmp = self.h['Results/Subdomains/Time']
        jd_dates = ttmp[:]
        try:
            ttmp = dt.datetime.strptime(tstr0, '%Y-%m-%d, %H:%M')
        except ValueError:
            tstrtmp = tstr0.replace('24:00', '23:00')
            ttmp = dt.datetime.strptime(tstrtmp, '%Y-%m-%d, %H:%M')
            ttmp += dt.timedelta(hours=1)
        dt_dates = []
        t_offset = ttmp.toordinal() + float(ttmp.hour) / 24. + float(ttmp.minute) / (24. * 60.)
        for t in jd_dates:
            dt_dates.append(ttmp + dt.timedelta(days=t))
        self.nt = len(dt_dates)

        self.dt_dates = np.asarray(dt_dates)
        self.jd_dates = np.asarray(jd_dates)
        self.t_offset = t_offset

    def loadSubdomains(self):
        '''
        creates a dictionary of all subdomains in the H5 file and grabes their XY coordinates for later reference
        :return: set class variables
                    self.subdomains
        '''

        self.subdomains = {}
        group = self.h['Geometry/Subdomains']
        for subdomain in group:
            dataset = self.h['Geometry/Subdomains/{0}/Cell Center Coordinate'.format(subdomain)]
            ncells = (np.shape(dataset))[0]
            x = np.array([dataset[i][0] for i in range(ncells)])
            y = np.array([dataset[i][1] for i in range(ncells)])
            self.subdomains[subdomain] = {'x': x, 'y': y}

    def readProfileData(self, resname, metric, times):
        '''
        reads Ressim profile data from model
        :param resname: name of reservoir in h5 file
        :param metric: metric of data to be extracted
        :param times: timesteps to grab
        :return: vals, elevations, depths
        '''

        self.loadElevation(alt_subdomain_name=resname)
        unique_times = [n for n in times]

        vals = []
        elevations = []
        depths = []
        times = []
        # print('UNIQUE TIMES:', unique_times)
        for j, time_in in enumerate(unique_times):
            timestep = WF.getIdxForTimestamp(self.jd_dates, time_in, self.t_offset)
            if timestep == -1:
                depths.append(np.asarray([]))
                elevations.append(np.asarray([]))
                vals.append(np.asarray([]))
                times.append(time_in)
                # continue
            else:
            # print('finding time for', time_in)
                self.loadResults(time_in, metric.lower(), alt_subdomain_name=resname)
                ktop = self.getTopLayer(timestep) #get waterlevel top layer to know where to grab data from
                v_el = self.vals[:ktop + 1]
                el = self.elev[:ktop + 1]
                d_step = []
                e_step = []
                v_step = []
                for ei, e in enumerate(el):
                    d_step.append(np.max(el) - e)
                    e_step.append(e)
                    v_step.append(v_el[ei])
                depths.append(np.asarray(d_step))
                elevations.append(np.asarray(e_step))
                vals.append(np.asarray(v_step))
                times.append(time_in)
        return vals, elevations, depths, np.asarray(times)

    def getTopLayer(self, timestep_index):
        '''
        grabs the top active layer of water for a given timestep
        :param timestep_index: timestep index to grab data at
        :return: returns index for top layer of water column
        '''

        elev = self.elev_ts[timestep_index] #elevations at a timestep
        for k in range(len(self.elev) - 1): #for each cell..
            cell_z = self.elev[k]  # layer midpoint
            cell_z1 = self.elev[k + 1]  # layer above midpoint
            top_of_cell_z = 0.5 * (cell_z + cell_z1)
            if elev < top_of_cell_z:
                break
        return k

    def loadElevation(self, alt_subdomain_name=None):
        '''
        loads elevations from the H5 file
        :param alt_subdomain_name: alternate field if the domain is not class defined subdomain name
        :return: set class variables
                    self.ncells - number of cells
                    self.elev - elevation time series for profile
                    self.elev_ts - elevation time series
        '''

        this_subdomain = self.subdomain_name if alt_subdomain_name is None else alt_subdomain_name
        cell_center_xy = self.h['Geometry/Subdomains/' + this_subdomain + '/Cell Center Coordinate']
        self.ncells = (np.shape(cell_center_xy))[0]
        # self.elev = np.array([cell_center_xy[i][2] for i in range(self.ncells)])
        self.elev = np.array(cell_center_xy[:self.ncells, 2])
        elev_ts = self.h['Results/Subdomains/' + this_subdomain + '/Water Surface Elevation']
        # self.elev_ts = np.array([elev_ts[i] for i in range(self.nt)])
        self.elev_ts = np.array(elev_ts[:self.nt])


    def loadResults(self, t_in, metrc, alt_subdomain_name=None):
        '''
        loads results for a specific time step from h5 file
        :param t_in: time in datetime object
        :param metrc: metric to get data from
        :param alt_subdomain_name: alt field if data is grabbed from a different location than the default
        :return: set class vairables
                self.t_data - timestep
                self.vals - array of values
        '''

        this_subdomain = self.subdomain_name if alt_subdomain_name is None else alt_subdomain_name

        timestep = WF.getIdxForTimestamp(self.jd_dates, t_in, self.t_offset) #get timestep index for current date
        if timestep == -1:
            print('should never be here..')
        self.t_data = t_in
        if metrc.lower() == 'temperature':
            metric_name = 'Water Temperature'
            try:
                v = self.h['Results/Subdomains/' + this_subdomain + '/' + metric_name]
            except KeyError:
                raise KeyError('WQ Simulation does not have results for metric: {0}'.format(metric_name))
            vals = np.array([v[timestep][i] for i in range(self.ncells)])
        elif metrc == 'diss_oxy' or metrc.lower() == 'do':
            metric_name = 'Dissolved Oxygen'
            try:
                v = self.h['Results/Subdomains/' + this_subdomain + '/' + metric_name]
            except KeyError:
                raise KeyError('WQ Simulation does not have results for metric: {0}'.format(metric_name))
            vals = np.array([v[timestep][i] for i in range(self.ncells)])
        elif metrc.lower() == 'do_sat':
            metric_name = 'Water Temperature'
            vtmp = self.h['Results/Subdomains/' + this_subdomain + '/' + metric_name]
            vt = np.array([vtmp[timestep][i] for i in range(self.ncells)])
            metric_name = 'Dissolved Oxygen'
            vtmp = self.h['Results/Subdomains/' + this_subdomain + '/' + metric_name]
            vdo = np.array([vtmp[timestep][i] for i in range(self.ncells)])
            vals = WF.calcComputedDOSat(vt, vdo)
        self.vals = vals

    def readTimeSeries(self, metric, x, y):
        '''
        Gets Time series values from Ressim H5 files.
        :param metric: metric of data
        :param x: X coordinate of the observed data to be passed into self.find_computed_station_cell(xy) to find
                    location of modeled data
        :param y: Y coordinate of the observed data to be passed into self.find_computed_station_cell(xy) to find
                    location of modeled data
        :return: times and values arrays for selected metric and time window
        '''

        i, subdomain_name = self.findComputedStationCell(x, y)

        if metric.lower() == 'flow':
            dataset_name = 'Cell flow'
            dataset = self.h['Results/Subdomains/{0}/{1}'.format(subdomain_name, dataset_name)]
            v = np.array(dataset[:, i])
            v = WF.cleanComputed(v)
        elif metric.lower() == 'elevation':
            dataset_name = 'Water Surface Elevation'
            dataset = self.h['Results/Subdomains/{0}/{1}'.format(subdomain_name, dataset_name)]
            v = np.array(dataset[:])
            v = WF.cleanComputed(v)
        elif metric.lower() == 'temperature':
            dataset_name = 'Water Temperature'
            dataset = self.h['Results/Subdomains/{0}/{1}'.format(subdomain_name, dataset_name)]
            v = np.array(dataset[:, i])
            v = WF.cleanComputed(v)
        elif metric.lower() == 'do':
            dataset_name = 'Dissolved Oxygen'
            dataset = self.h['Results/Subdomains/{0}/{1}'.format(subdomain_name, dataset_name)]
            v = np.array(dataset[:, i])
            v = WF.cleanComputed(v)
        elif metric.lower() == 'do_sat':
            dataset_name = 'Water Temperature'
            dataset = self.h['Results/Subdomains/{0}/{1}'.format(subdomain_name, dataset_name)]
            vt = np.array(dataset[:, i])
            dataset_name = 'Dissolved Oxygen'
            dataset = self.h['Results/Subdomains/{0}/{1}'.format(subdomain_name, dataset_name)]
            vdo = np.array(dataset[:, i])
            vt = WF.cleanComputed(vt)
            vdo = WF.cleanComputed(vdo)
            v = WF.calcComputedDOSat(vt, vdo)

        if not hasattr(self, 't_computed'):
            self.loadComputedTime()
        istart = 0
        iend = -1
        return self.t_computed[istart:iend], v[istart:iend]

    def readSubdomain(self, metric, subdomain_name):

        if metric.lower() == 'flow':
            dataset_name = 'Cell flow'
            dataset = self.h['Results/Subdomains/{0}/{1}'.format(subdomain_name, dataset_name)]
            v = np.array(dataset[:])
            v = WF.cleanComputed(v)
        elif metric.lower() == 'elevation':
            dataset_name = 'Water Surface Elevation'
            dataset = self.h['Results/Subdomains/{0}/{1}'.format(subdomain_name, dataset_name)]
            v = np.array(dataset[:])
            v = WF.cleanComputed(v)
        elif metric.lower() == 'temperature':
            dataset_name = 'Water Temperature'
            dataset = self.h['Results/Subdomains/{0}/{1}'.format(subdomain_name, dataset_name)]
            v = np.array(dataset[:])
            v = WF.cleanComputed(v)
        elif metric.lower() == 'do':
            dataset_name = 'Dissolved Oxygen'
            dataset = self.h['Results/Subdomains/{0}/{1}'.format(subdomain_name, dataset_name)]
            v = np.array(dataset[:])
            v = WF.cleanComputed(v)
        elif metric.lower() == 'do_sat':
            dataset_name = 'Water Temperature'
            dataset = self.h['Results/Subdomains/{0}/{1}'.format(subdomain_name, dataset_name)]
            vt = np.array(dataset[:])
            dataset_name = 'Dissolved Oxygen'
            dataset = self.h['Results/Subdomains/{0}/{1}'.format(subdomain_name, dataset_name)]
            vdo = np.array(dataset[:])
            vt = WF.cleanComputed(vt)
            vdo = WF.cleanComputed(vdo)
            v = WF.calcComputedDOSat(vt, vdo)

        distance = self.calcSubdomainDistances(subdomain_name)
        #add a value at the start and end to compensate for the start and end values
        v = np.insert(v, 0, v.T[:][0], 1)
        v = np.insert(v, -1, v.T[:][-1], 1)

        if not hasattr(self, 't_computed'):
            self.loadComputedTime()
        istart = 0
        iend = -1
        return self.t_computed[istart:iend], v[istart:iend], distance

    def readModelTimeseriesData(self, data, metric):
        '''
        function to wrangle data and be universably 'callable' from main script
        :param data: dictionary containing information about data records
        :param metric: not needed for W2, but still passed in for Ressim results so a single function can be called
        :return: arrays of dates and values
        '''

        x = data['easting']
        y = data['northing']
        dates, vals = self.get_Timeseries(metric, xy=[x, y])
        return dates, vals

    def calcSubdomainDistances(self, subdomain):

        cell_center_xy = self.h['Geometry/Subdomains/{0}/Cell Center Coordinate'.format(subdomain)]
        firstpoint = cell_center_xy[0]
        distance = []
        if 'Geometry/Subdomains/{0}/Cell Length'.format(subdomain) in self.h.keys():
            distance.append(0)
            cell_lengths = self.h['Geometry/Subdomains/{0}/Cell Length'.format(subdomain)]
            for cli, celllen in enumerate(cell_lengths):
                if cli == 0:
                    distance.append(celllen.item()/2)# distance from edge to first cell center
                else:
                    #half the len of current cell, half the len of last cell to get distnace between cell centers
                    #then add on the distance we've calc'd
                    distance.append(celllen.item()/2 + cell_lengths[cli-1].item()/2 + distance[-1])
                if cli == len(cell_lengths)-1:
                    #add the distance from the last cell center to the edge
                    distance.append(celllen.item()/2 + distance[-1])
            distance = np.asarray(distance)
        else:
            for cell in cell_center_xy:
                d = np.sqrt( (cell[0] - firstpoint[0])**2 +  (cell[1] - firstpoint[1])**2)
                distance.append(d)
            #get roughly half the distance between first two cells. This is the closest we can get to half the cell distance
            first_distance_diff_half = (distance[1] - distance[0]) / 2
            #shift all distance so the first instance is now the cell center of the first point
            distance = np.asarray(distance) + first_distance_diff_half
            #then add 0 to the start, so it starts at 0
            distance = np.insert(distance, 0, 0)
            #then do the same for the backend
            last_distance_diff_half = (distance[-1] - distance[-2]) / 2
            distance = np.append(distance, distance[-1] + last_distance_diff_half)

        return distance

    def findComputedStationCell(self, easting, northing):
        '''
        finds subdomains that are closest to observed station coordinates
        TODO: add some kind of tolerance or max distance?
        :param xy: XY coordinates for observed station
        :return: cell index and subdomain information closest to observed data
        '''


        nearest_dist = 1e6
        for subdomain, sd_data in self.subdomains.items():
            x = sd_data['x']
            y = sd_data['y']
            dist = np.sqrt((x - easting) * (x - easting) + (y - northing) * (y - northing))
            min_dist = np.min(dist)
            if min_dist < nearest_dist:
                min_cell = np.argmin(dist)
                data_index = min_cell
                data_subdomain = subdomain
                nearest_dist = min_dist
        return data_index, data_subdomain

    def loadComputedTime(self):
        '''
        loads computed time values, replacing 24 hr date values with 0000 the next day
        grabs all values instead of user defined, if none are defined
        TODO: is this still needed? require user input.
        :return: sets class variables
                    self.t_computed - list of times used in computation
        '''

        tstr = self.h['Results/Subdomains/Time Date Stamp']
        tstr0 = (tstr[0]).decode("utf-8")
        tstr1 = (tstr[1]).decode("utf-8")
        ttmp = self.h['Results/Subdomains/Time']
        nt = len(ttmp)
        try:
            ttmp0 = dt.datetime.strptime(tstr0, '%Y-%m-%d, %H:%M')
        except ValueError:
            tstrtmp = tstr0.replace('24:00', '23:00')
            ttmp0 = dt.datetime.strptime(tstrtmp, '%Y-%m-%d, %H:%M')
            ttmp0 += dt.timedelta(hours=1)
        try:
            ttmp1 = dt.datetime.strptime(tstr1, '%Y-%m-%d, %H:%M')
        except ValueError:
            tstrtmp = tstr1.replace('24:00', '23:00')
            ttmp1 = dt.datetime.strptime(tstrtmp, '%Y-%m-%d, %H:%M')
            ttmp1 += dt.timedelta(hours=1)
        delta_t = ttmp1 - ttmp0
        self.t_computed = []
        for j in range(nt):
            self.t_computed.append(ttmp0 + j * delta_t)
        self.t_computed = np.array(self.t_computed)

if __name__ == '__main__':
    #TODO: for debugging, remove.

    # graphicsDefaultfile = r"D:\Work2021\USBR\RessimXMLTest\Shasta_W2_BUZZ.xml"
    # results = ReadChapterDefFile(graphicsDefaultfile)
    # for n in results:
    #     print(n)
    # graphicsDefaultfile = r"D:\Work2021\USBR\RessimXMLTest\Graphics_Defaults_Beta_v5.xml"
    # results = read_GraphicsDefaults(graphicsDefaultfile)
    # print(results)
    print('debug.')
