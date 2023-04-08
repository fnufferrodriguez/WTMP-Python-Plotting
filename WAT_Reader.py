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
import datetime as dt
import pandas as pd
from scipy.interpolate import interp1d
from pydsstools.heclib.dss import HecDss
from collections import Counter
import xml.etree.ElementTree as ET
import pendulum
import traceback

import WAT_Functions as WF

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

def readSimulationFile(simulationfile):
    '''
    Read the right csv file, and determine what region you are working with.
    Simulation CSV files are named after the simulation, and consist of plugin, model alter name, and then region(s)
    :param simulation_name: name of simulation to find file
    :param studyfolder: full path to study folder
    :returns: dictionary containing information from file
    '''

    # if iscomp:
    #     simulation_file = os.path.join(studyfolder, 'reports', '{0}_comparison.csv'.format(simulation_name.replace(' ', '_')))
    # else:
    #     simulation_file = os.path.join(studyfolder, 'reports', '{0}.csv'.format(simulation_name.replace(' ', '_')))
    WF.print2stdout('Attempting to read {0}'.format(simulationfile))
    if not os.path.exists(simulationfile):
        WF.print2stderr(f'Could not find CSV file: {simulationfile}')
        WF.print2stderr(f'Please create {simulationfile} in the Reports Directory and run report again.')
        sys.exit(1)
    sim_info = {}
    with open(simulationfile, 'r') as sf:
        for i, line in enumerate(sf):
            if len(line.strip()) > 0:
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

def findTargetinChapterDefFile(flags, chapter, default=''):
    targettext = default
    grouptext_flags = flags
    for flag in grouptext_flags:
        findtext = chapter.find(flag)
        if isinstance(findtext, ET.Element):
            targettext = findtext.text
            break
    return targettext

def readBCPathsMap(bcpathsmapfile):
    bcpathsmap = readFormattedTable_Pandas(bcpathsmapfile)
    return bcpathsmap


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
        # check = definedVarCheck(chapter, ['Name', 'Region', 'Sections'])
        # if check:
        ChapterDef = {}
        try: chap_name = chapter.find('Name').text
        except: chap_name = ''
        try: chap_region = chapter.find('Region').text
        except: chap_region = ''
        ChapterDef['name'] = chap_name
        ChapterDef['region'] = chap_region
        ChapterDef['sections'] = []

        # grouptext = ''
        # grouptext_flags = ['text', 'Text', 'TEXT']
        # for flag in grouptext_flags:
        #     findtext = chapter.find(flag)
        #     if isinstance(findtext, ET.Element):
        #         grouptext = findtext.text
        #         break
        # ChapterDef['grouptext'] = grouptext
        grouptext_flags = ['text', 'Text', 'TEXT']
        ChapterDef['grouptext'] = findTargetinChapterDefFile(grouptext_flags, chapter)

        # resolution = 'high'
        # resolution_flags = ['resolution', 'Resolution', 'RESOLUTION']
        # for flag in resolution_flags:
        #     findtext = chapter.find(flag)
        #     if isinstance(findtext, ET.Element):
        #         resolution = findtext.text
        #         break
        # ChapterDef['resolution'] = resolution
        resolution_flags = ['resolution', 'Resolution', 'RESOLUTION']
        ChapterDef['resolution'] = findTargetinChapterDefFile(resolution_flags, chapter, default='high')

        # debug = 'false'
        # debug_flags = ['debug', 'Debug', 'DEBUG']
        # for flag in debug_flags:
        #     findtext = chapter.find(flag)
        #     if isinstance(findtext, ET.Element):
        #         debug = findtext.text
        #         break
        # ChapterDef['debug'] = debug
        debug_flags = ['debug', 'Debug', 'DEBUG']
        ChapterDef['debug'] = findTargetinChapterDefFile(debug_flags, chapter, default='false')

        # collections_flags = ['isensemble', 'Isensemble', 'ISENSEMBLE'] #TODO: is this final flag??
        # ChapterDef['isensemble'] = findTargetinChapterDefFile(collections_flags, chapter, default='false')

        forecastiteration_flags = ['forecastiteration', 'Forecastiteration', 'ForecastIteration', 'FORECASTITERATION'] #TODO: is this final flag??
        ChapterDef['forecastiteration'] = findTargetinChapterDefFile(forecastiteration_flags, chapter, default='false')

        cd_sections = chapter.findall('Sections/Section')
        for section in cd_sections:
            section_objects = {}

            # headertext = ''
            # headerflags = ['header', 'Header', 'HEADER']
            # for flag in headerflags:
            #     findtext = section.find(flag)
            #     if isinstance(findtext, ET.Element):
            #         headertext = findtext.text
            #         break
            # section_objects['header'] = headertext
            headerflags = ['header', 'Header', 'HEADER']
            section_objects['header'] = findTargetinChapterDefFile(headerflags, section)

            # summaryflags = ['summary', 'Summary', 'SUMMARY']
            # section_objects['summary'] = findTargetinChapterDefFile(summaryflags, section)

            section_objects['objects'] = []
            sec_objects = section.findall('Object')
            section_objects['objects'] = iterateChapterDefintions(sec_objects)
            ChapterDef['sections'].append(section_objects)

        Chapters.append(ChapterDef)

    return Chapters

def readCollectionsDSSData(dss_file, pathname, collectionIDs, startdate, enddate, debug):
    try:
        if os.path.exists(dss_file):
            fid = HecDss.Open(dss_file)
            if pathname.split('/')[4] != '*': #make sure date field is blank
                pns = pathname.split('/')
                pns[4] = '*'
                pathname = '/'.join(pns)
            collection_pn = fid.getPathnameList(pathname)
            collection_pn = list(set(['/'.join(WF.ReplaceListAtIdx(n.split('/'), 4, '*')) for n in collection_pn]))
            WF.print2stdout(f'Found {len(collection_pn)} records in collection for {pathname}', debug=debug)
            if len(collection_pn) == 0:
                WF.print2stdout(f'No records in collection for {pathname}', debug=debug)
                fid.close()
                return [], [], None, []
            if collectionIDs == 'all':
                collectionIDs = [int(n.split('/')[6].split('|')[0].replace('C:', '')) for n in collection_pn]
            else:
                collectionIDs = [int(n) for n in collectionIDs]

            collectionIDs.sort()

            for i, ID in enumerate(collectionIDs):
                if len(str(ID)) != 6:
                    ID = str(ID).zfill(6) #DSS conventions want 6 spaces
                CID_pathname_fpart = pathname.split('/')[6].replace('*|', f'C:{ID}|')
                CID_pathname_split = pathname.split('/')
                CID_pathname_split[6] = CID_pathname_fpart
                CID_pathname = '/'.join(CID_pathname_split)
                WF.print2stdout(f'Currently working on {ID}', debug=debug)
                ts = fid.read_ts(CID_pathname, window=(startdate, enddate), regular=True, trim_missing=True)
                values = np.asarray(ts.values, dtype=np.float64)
                if i == 0: #set vars like times and units that are always the same for all collection
                    times = np.array(ts.pytimes)
                    units = ts.units
                    collection_values = {}
                    # collection_values = np.full((len(collection_pn), len(times)), np.nan) #initialize array
                # collection_values[int(ID)-1] = values #Todo: shelf for now
                collection_values[int(ID)] = values
            fid.close()
            return times, collection_values, units, collectionIDs
        else:
            WF.print2stdout(f'DSS file {dss_file} not found.', debug=True)
            return [], [], None, []
    except:
        WF.print2stdout(f'Unable to get data from {dss_file} {pathname}')
        WF.print2stdout(traceback.format_exc(), debug=debug)
        return [], [], None, []

def readDSSData(dss_file, pathname, startdate, enddate, debug):
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
        WF.print2stdout('DSS file not found!', dss_file, debug=debug)
        return [], [], None

    fid = HecDss.Open(dss_file)
    ts = fid.read_ts(pathname,window=(startDatestr,endDatestr),regular=True,trim_missing=False)
    if ts.empty: #if empty, it must be the path or time window. DSS record must exist
        WF.print2stdout('Invalid Timeseries record path of {0} or time window of {1} - {2}'.format(pathname, startDatestr, endDatestr), debug=debug)
        WF.print2stdout('Please check these parameters and rerun.', debug=debug)
        return [], [], None
    values = np.array(ts.values)
    values[ts.nodata] = np.nan

    made_ts = False
    if ts.dtype == 'Regular TimeSeries':
        interval_seconds = ts.interval
        times = []
        # current_time = startdate
        current_time = ts.startPyDateTime
        end_time = ts.endDateTime #bugged where the pydatetime shows start time..
        try:
            end_pytime = dt.datetime.strptime(end_time, '%d%b%Y %H:%M:%S')
        except ValueError:
            end_time_repl = end_time.replace(' 24:', ' 23:')
            end_pytime = dt.datetime.strptime(end_time_repl, '%d%b%Y %H:%M:%S')
            end_pytime += dt.timedelta(hours=1)
        while current_time <= end_pytime:
            times.append(current_time)
            current_time += dt.timedelta(seconds=interval_seconds)
        if len(times) == len(values):
            made_ts = True

    if not made_ts:
        times = np.asarray(ts.pytimes)
        WF.print2stdout('Irregular DSS detected with {0} in {1}'.format(pathname, dss_file), debug=debug)
        WF.print2stdout('Recommend changing to regular time series for speed increases.', debug=debug)
    else:
        times = np.asarray(times)

    units = ts.units

    return times, values, units

def formatPyDSSToolsDates(datestring):
    '''
    takes in date strings and converts it to the correct form to be read by PYDSSTOOLS
    fixes 2400 issues too
    :param datestring: date string or datetime object
    :return: formatted datetring
    '''

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

def readFormattedTable_Pandas(filename):
    if os.path.exists(filename):
        ext = filename.split('.')[-1]
        if ext.lower() == 'csv':
            df = pd.read_csv(filename)
        df.dropna(inplace=True) #sometimes theres extra rows..
        return df
    else:
        WF.print2stdout(f'{filename} not found.')
        return pd.DataFrame()

def readTextProfile(observed_data_filename, timestamps, starttime=None, endtime=None):
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
        WF.print2stdout('Observed data at {0} does not exist.'.format(observed_data_filename))
        return [], [], []
    with open(observed_data_filename, 'r') as odf:
        for j, line in enumerate(odf):
            if j == 0:
                headers = line.strip().split(',')
                continue
            sline = line.split(',')
            dt_str = sline[0]
            if dt_str == last_dtstr:
                dt_tmp = last_dt
            else:
                dt_tmp = pendulum.parse(dt_str, strict=False).replace(tzinfo=None)
                last_dtstr = dt_str
                last_dt = dt_tmp
            if starttime > dt_tmp:
                continue
            if endtime < dt_tmp:
                break
            # if (dt_tmp.year != hold_dt.year or dt_tmp.month != hold_dt.month or dt_tmp.day != hold_dt.day): #if its a new date
            if (dt_tmp != hold_dt): #if its a new date
                if len(t_profile) != 0 and len(wt_profile) != 0 and len(d_profile) != 0:
                    t.append(np.array(t_profile))
                    wt.append(np.array(wt_profile))
                    d.append(np.array(d_profile))
                t_profile = [dt_tmp]
                wt_profile = [float(sline[1])]
                d_profile = [float(sline[2])]
            else:
                # if float(sline[2]) not in d_profile:
                t_profile.append(dt_tmp)
                wt_profile.append(float(sline[1]))
                d_profile.append(float(sline[2]))
            hold_dt = dt_tmp

    if len(t_profile) != 0 and len(wt_profile) != 0 and len(d_profile) != 0:
        t.append(np.array(t_profile))
        wt.append(np.array(wt_profile))
        d.append(np.array(d_profile))

    if isinstance(timestamps, (list, np.ndarray)):
        wtn = []
        dn = []
        ts = []
        if len(t) > 0:
            cti = getClosestProfileTime(timestamps, [n[0] for n in t])

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
    else:
        return wt, d, np.asarray(t)

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
            if dt_str not in t:
                t.append(dt_str)
    t_frmt = []
    for tdate in t:
        dt_tmp = pendulum.parse(tdate, strict=False).replace(tzinfo=None) #trying this to see if it will work
        if starttime <= dt_tmp <= endtime: #get time window
            if dt_tmp not in t_frmt:
                t_frmt.append(dt_tmp)

    return np.asarray(t_frmt)

def getClosestProfileTime(timestamps, dates):
    '''
    gets timestamp closest to given timestamps for profile plots
    :param timestamps: list of target timestamps
    :param dates: dates in file
    :return:
    '''

    cdi = []
    for timestamp in timestamps:
        cloz_dict = {
            abs(timestamp.timestamp() - date.timestamp()) : di
            for di, date in enumerate(dates)}
        res = cloz_dict[min(cloz_dict.keys())]
        if abs(timestamp.timestamp() - dates[res].timestamp()) > 86400: #seconds in a day
            cdi.append(None)
        else:
            cdi.append(res)
    return cdi

def getClosestTime(timestamps, dates):
    '''
    gets timestamp closest to given timestamps for profile plots
    :param timestamps: list of target timestamps
    :param dates: dates in file
    :return:
    '''

    cdi = []
    if len(dates) > 0:
        t0 = dates[0]
    else:
        return []
    if len(dates) > 1:
        t_interval = dates[1] - t0 #timedelta
    t_interval_seconds = t_interval.total_seconds()
    for timestamp in timestamps:
        ts_diff = timestamp - t0
        index = int(round(ts_diff.total_seconds() / t_interval_seconds))
        cdi.append(index)
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
        if len(Counter([n.tag.lower() for n in root])) > 1: #if the amount of diff subroots > 1
            children[root.tag.lower()] = {}
        else: #if there is only 1 subroot
            if len(root.text.strip()) == 0: #if the text len of root is 0, we have subitems
                if len([n.tag.lower() for n in root]) > 1: #if we have more than 1 of the same subroot
                    children[root.tag.lower()] = []
                else: #otherwise, we have a single dictionary
                    subroot = root[0]
                    #if the subroots is just the root, but singular, aka lines -> line, reaches -> reach
                    if subroot.tag.lower() == root.tag.lower()[:-1] or subroot.tag.lower() == root.tag.lower()[:-2]:
                        children[root.tag.lower()] = []
                    else:
                        children[root.tag.lower()] = {}
            else:
                children[root.tag.lower()] = []

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

def readSimulationsCSV(Report):
    '''
    reads the Simulation file and gets the region info
    :return: class variable
                self.SimulationCSV
    '''
    simulation_file = os.path.join(Report.studyDir, 'reports', '{0}.csv'.format(Report.SimulationVariables[Report.base_id]['baseSimulationName'].replace(' ', '_')))
    Report.SimulationCSV = readSimulationFile(simulation_file)

def readSimulationInfo(Report, simulationInfoFile):
    '''
    reads sim info XML file and organizes paths and variables into a list for iteration
    :param simulationInfoFile: full path to simulation information XML file from WAT
    :return: class variables:
                self.Simulations
                self.reportType
                self.studyDir
                self.observedData
    '''

    WF.checkExists(simulationInfoFile)

    Report.Simulations = []
    tree = ET.parse(simulationInfoFile)
    root = tree.getroot()

    Report.reportType = root.find('ReportType').text
    Report.studyDir = root.find('Study/Directory').text
    Report.observedDir = root.find('Study/ObservedData').text
    Report.installDir = root.find('Study/InstallDirectory').text #TODO: update
    Report.outputDir = root.find('Study/WriteDirectory').text #TODO: update

    Report.iscomp = False
    Report.isforecast = False
    if Report.reportType == 'alternativecomparison':
        Report.iscomp = True
    elif Report.reportType == 'forecast':
        Report.isforecast = True

    SimRoot = root.find('Simulations')
    for simulation in SimRoot:
        simulationInfo = {'name': simulation.find('Name').text,
                          'basename': simulation.find('BaseName').text,
                          'directory': simulation.find('Directory').text,
                          'dssfile': simulation.find('DSSFile').text,
                          'starttime': simulation.find('StartTime').text,
                          'endtime': simulation.find('EndTime').text,
                          'lastcomputed': simulation.find('LastComputed').text
                          }

        if Report.isforecast:
            iterations = getchildren(simulation.find('iterations'), returnkeyless=True)
            simulationInfo['iterations'] = iterations

        try:
            simulationInfo['ID'] = simulation.find('ID').text
        except AttributeError:
            simulationInfo['ID'] = Report.base_id


        modelAlternatives = []
        for modelAlt in simulation.find('ModelAlternatives'):
            modelAlternatives.append({'name': modelAlt.find('Name').text,
                                      'program': modelAlt.find('Program').text,
                                      'fpart': modelAlt.find('FPart').text,
                                      'directory': modelAlt.find('Directory').text})

        simulationInfo['modelalternatives'] = modelAlternatives
        Report.Simulations.append(simulationInfo)

def readGraphicsDefaultFile(Report):
    '''
    sets up path for graphics default file in study and reads the xml
    :return: class variable
                self.graphicsDefault
    '''

    graphicsDefaultfile = os.path.join(Report.studyDir, 'reports', 'Graphics_Defaults.xml')
    WF.checkExists(graphicsDefaultfile)
    # graphicsDefaultfile = os.path.join(self.default_dir, 'Graphics_Defaults.xml') #TODO: implement with build
    Report.graphicsDefault = readGraphicsDefaults(graphicsDefaultfile)

def readDefinitionsFile(Report, simorder):
    '''
    reads the chapter definitions file defined in the plugin csv file for a specified simulation
    :param simorder: simulation dictionary object
    :return: class variable
                self.ChapterDefinitions
    '''

    ChapterDefinitionsFile = os.path.join(Report.studyDir, 'reports', simorder['deffile'])
    WF.checkExists(ChapterDefinitionsFile)
    Report.ChapterDefinitions = readChapterDefFile(ChapterDefinitionsFile)

def readComparisonSimulationsCSV(Report):
    '''
    Reads in the simulation CSV but for comparison plots. Comparison plots have '_comparison' appended to the end of them,
    but are built in general the same as regular Simulation CSV files.
    :return:
    '''
    simulation_file = os.path.join(Report.studyDir, 'reports', '{0}_comparison.csv'.format(Report.SimulationVariables[Report.base_id]['baseSimulationName'].replace(' ', '_')))
    Report.SimulationCSV = readSimulationFile(simulation_file)

def readForecastSimulationsCSV(Report):
    '''
    Reads in the simulation CSV but for Forecast plots. Forecast plots have '_forecast' appended to the end of them,
    but are built in general the same as regular Simulation CSV files.
    :return:
    '''

    simulation_file = os.path.join(Report.studyDir, 'reports', '{0}_forecast.csv'.format(Report.SimulationVariables[Report.base_id]['baseSimulationName'].replace(' ', '_')))
    Report.SimulationCSV = readSimulationFile(simulation_file)

def readTemplate(Report, templatefilename):
    '''
    read template file
    :param Report: main report class object
    :param templatefilename: name of template file
    :return: formated dictionary for template file
    '''

    templatefile = os.path.join(Report.studyDir, 'reports', templatefilename)
    tree = ET.parse(templatefile)
    root = tree.getroot()
    templateObjects = root.findall('Object')
    reportObjects = iterateGraphicsDefaults(templateObjects, 'Type')
    return reportObjects

def readScalarTable(scalartablepath):
    '''
    reads in scalar tables. headers ignored but not necessary
    format must be
    name, scaled by, scalar
    OR
    scaled by, scalar
    so ex
    1out, 363, .34
    OR
    363, .43
    :param scalartablepath: path to table file
    :return: dictionary [scaledbyval: scalar]
    '''

    scalars = {}
    scalartable = np.genfromtxt(scalartablepath, delimiter=',', dtype=None, encoding="utf8")
    for line in scalartable:
        if len(line) == 2:
            try:
                line_flt = [float(n) for n in line]
                scalars[line_flt[0]] = line_flt[1]
            except:
                #probably header line
                continue
        elif len(line) > 2:
            try:
                line_flt = [float(n) for n in line[1:3]]
                scalars[line_flt[0]] = line_flt[1]
            except:
                #probably header line
                continue
        else:
            WF.print2stderr(f'Scalar table {scalartablepath} formatted incorrectly. Should be [target, scalar]')
            WF.print2stderr('Now exiting')
            sys.exit(1)

    return scalars
