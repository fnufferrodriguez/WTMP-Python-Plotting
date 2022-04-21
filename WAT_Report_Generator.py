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

VERSIONNUMBER = '4.0'

import datetime as dt
import os
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import is_color_like
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import calendar
from dateutil.relativedelta import relativedelta
import re
from collections import Counter
import shutil
from scipy import interpolate
from functools import reduce
import pickle
import pendulum
import itertools
import traceback

import WAT_DataReader as WDR
import WAT_Functions as WF
import WAT_XML_Utils as XML_Utils


class MakeAutomatedReport(object):
    '''
    class to organize data and generate XML file for Jasper processing in conjunction with WAT. Takes in a simulation
    information file output from WAT and develops the report from there.
    '''

    def __init__(self, simulationInfoFile, batdir):
        '''
        organizes input data and generates XML report
        :param simulationInfoFile: full path to simulation information XML file output from WAT.
        '''
        self.printVersion()
        self.simulationInfoFile = simulationInfoFile
        self.WriteLog = True #TODO we're testing this.
        self.batdir = batdir
        self.readSimulationInfo(simulationInfoFile) #read file output by WAT
        # self.EnsureDefaultFiles() #TODO: turn this back on for copying
        self.definePaths()
        self.cleanOutputDirs()
        self.defineUnits()
        self.defineMonths()
        self.defineTimeIntervals()
        self.defineDefaultColors()
        self.readGraphicsDefaultFile() #read graphical component defaults
        self.readDefaultLineStylesFile()
        self.buildLogFile()
        self.initializeDataMemory()
        if self.reportType == 'single': #Eventually be able to do comparison reports, put that here
            for simulation in self.Simulations:
                print('Running Simulation:', simulation)
                self.initSimulationDict()
                self.setSimulationVariables(simulation)
                self.loadCurrentID('base') #load the data for the current sim, we do 1 at a time here..
                self.defineStartEndYears()
                self.readSimulationsCSV() #read to determine order/sims/regions in report
                self.cleanOutputDirs()
                self.initializeXML()
                self.writeXMLIntroduction()
                for simorder in self.SimulationCSV.keys():
                    self.setSimulationCSVVars(self.SimulationCSV[simorder])
                    self.readDefinitionsFile(self.SimulationCSV[simorder])
                    self.loadModelAlts(self.SimulationCSV[simorder])
                    self.loadCurrentModelAltID('base')
                    self.addSimLogEntry()
                    self.writeChapter()
                    self.fixXMLModelIntroduction(simorder)
                self.XML.writeReportEnd()
                self.equalizeLog()
        elif self.reportType == 'alternativecomparison':
            self.initSimulationDict()
            for simulation in self.Simulations:
                self.setSimulationVariables(simulation)
            self.loadCurrentID('base') #load the data for the current sim, we do 1 at a time here..
            self.setMultiRunStartEndYears() #find the start and end time
            self.defineStartEndYears() #format the years correctly after theyre set
            self.readComparisonSimulationsCSV() #read to determine order/sims/regions in report
            self.cleanOutputDirs()
            self.initializeXML()
            self.writeXMLIntroduction()
            for simorder in self.SimulationCSV.keys():
                self.setSimulationCSVVars(self.SimulationCSV[simorder])
                self.readDefinitionsFile(self.SimulationCSV[simorder])
                self.loadModelAlts(self.SimulationCSV[simorder])
                self.loadCurrentModelAltID('base')
                self.addSimLogEntry()
                self.writeChapter()
                self.fixXMLModelIntroduction(simorder)
            self.XML.writeReportEnd()
            self.equalizeLog()
        else:
            print('UNKNOWN REPORT TYPE:', self.reportType)
            sys.exit()
        self.writeLogFile()
        self.writeDataFiles()

    def addLogEntry(self, keysvalues, isdata=False):
        '''
        adds an entry to the log file. If data, add an entry for all lists.
        :param keysvalues: dictionary containing values and headers
        :param isdata: if True, adds blanks for all data rows to keep things consistent
        '''

        for key in keysvalues.keys():
            self.Log[key].append(keysvalues[key])
        if isdata:
            allkeys = ['type', 'name', 'function', 'description', 'value', 'units',
                       'value_start_date', 'value_end_date', 'logoutputfilename']
            for key in allkeys:
                if key not in keysvalues.keys():
                    self.Log[key].append('')

    def addSimLogEntry(self):
        '''
        adds entries for a simulation with relevenat metadata
        '''

        for ID in self.accepted_IDs:
            print('ID:', ID)
            print('Simvars:', self.SimulationVariables[ID])
            self.Log['observed_data_path'].append(self.observedDir)
            self.Log['start_time'].append(self.SimulationVariables[ID]['StartTimeStr'])
            self.Log['end_time'].append(self.SimulationVariables[ID]['EndTimeStr'])
            self.Log['compute_time'].append(self.SimulationVariables[ID]['LastComputed'])
            self.Log['program'].append(self.SimulationVariables[ID]['plugin'])
            self.Log['alternative_name'].append(self.SimulationVariables[ID]['modelAltName'])
            self.Log['fpart'].append(self.SimulationVariables[ID]['alternativeFpart'])
            self.Log['program_directory'].append(self.SimulationVariables[ID]['alternativeDirectory'])

    def defineMonths(self):
        '''
        defines month 3 letter codes for table labels, and reference dicts for months and numbers (aka Jan: 1)
        :return: class variables
                    self.month2num
                    self.num2month
                    self.mo_str_3
        '''

        self.month2num = {month.lower(): index for index, month in enumerate(calendar.month_abbr) if month}
        self.num2month = {index: month.lower() for index, month in enumerate(calendar.month_abbr) if month}
        self.mo_str_3 = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    def defineStartEndYears(self):
        '''
        defines start and end years for the simulation so they can be replaced by flagged values.
        end dates that end on the first of the year with no min seconds (aka Dec 31 @ 24:00) have their end
        years set to be the year prior, as its not fair to really call them that next year
        self.years is a list of all years used
        self.years_str is a string representation of the years, either as a single year, or range, aka 2003-2005
        :return: class variables
                    self.startYear
                    self.endYear
                    self.years
                    self.years_str
        '''

        tw_start = self.StartTime
        tw_end = self.EndTime
        if tw_end == dt.datetime(tw_end.year, 1, 1, 0, 0):
            tw_end += dt.timedelta(seconds=-1) #if its this day just go back

        self.startYear = tw_start.year
        self.endYear = tw_end.year
        if self.startYear == self.endYear:
            self.years_str = str(self.startYear)
            self.years = [self.startYear]
        else:
            self.years = range(tw_start.year, tw_end.year+1)
            self.years_str = "{0}-{1}".format(self.startYear, self.endYear)

    def defineTimeIntervals(self):
        '''
        sets up a dictionary for DSS intervals and their associated timedelta amount for setting up regular
        interval time series arrays
        :return: class variable
                    self.time_intervals
        '''

        self.time_intervals = {'1MIN': [dt.timedelta(minutes=1),'np'],
                               '2MIN': [dt.timedelta(minutes=2),'np'],
                               '5MIN': [dt.timedelta(minutes=5),'np'],
                               '6MIN': [dt.timedelta(minutes=6),'np'],
                               '10MIN': [dt.timedelta(minutes=10),'np'],
                               '12MIN': [dt.timedelta(minutes=12),'np'],
                               '15MIN': [dt.timedelta(minutes=15),'np'],
                               '30MIN': [dt.timedelta(minutes=30),'np'],
                               '1HOUR': [dt.timedelta(hours=1),'np'],
                               '2HOUR': [dt.timedelta(hours=2),'np'],
                               '3HOUR': [dt.timedelta(hours=3),'np'],
                               '4HOUR': [dt.timedelta(hours=4),'np'],
                               '5HOUR': [dt.timedelta(hours=5),'np'],
                               '6HOUR': [dt.timedelta(hours=6),'np'],
                               '1DAY': [dt.timedelta(days=1),'np'],
                               '1MON': ['1M', 'pd'],
                               '2MON': ['2M', 'pd'],
                               '6MON': ['6M', 'pd'],
                               '1YEAR': ['1Y', 'pd']}


    def defineDefaultColors(self):
        '''
        sets up a list of default colors to use in the event that colors are not set up in the graphics default file
        for a line
        Color Changes based off of bureau of Rec Identification program pdf
        :return: class variable
                    self.def_colors
        '''

        self.def_colors = ['#003E51', '#FF671F', '#007396', '#215732', '#C69214', '#4C12A1', '#DDCBA4', '#9A3324']
        #                     blue       orange    l blue     green      mustard     purple     tan      dark red

    def defineUnits(self):
        '''
        creates dictionary with units for vars for labels
        #TODO: expand this
        :return: set class variable
                    self.units
        '''

        self.units = {'temperature': {'metric':'c', 'english':'f'},
                      'temp': {'metric':'c', 'english':'f'},
                      'do_sat': {'metric': '%', 'english': '%'},
                      'flow': {'metric': 'm3/s', 'english': 'cfs'},
                      'storage': {'metric': 'm3', 'english': 'af'},
                      'stor': {'metric': 'm3', 'english': 'af'},
                      'elevation': {'metric': 'm', 'english': 'ft'},
                      'elev': {'metric': 'm', 'english': 'ft'},
                      'ec':  {'metric': 'us/cm', 'english': 'us/cm'},
                      'electrical conductivity': {'metric': 'us/cm', 'english': 'us/cm'},
                      'salinity': {'metric': 'psu', 'english': 'psu'},
                      'sal': {'metric': 'psu', 'english': 'psu'},
                      }

    def definePaths(self):
        '''
        defines run specific paths
        used to contain more paths, but not needed. Consider moving.
        :return: set class variables
                    self.images_path
        '''

        self.images_path = os.path.join(self.studyDir, 'reports', 'Images')
        self.CSVPath = os.path.join(self.studyDir, 'reports', 'CSVData')
        self.default_dir = os.path.join(os.path.split(self.batdir)[0], 'Default')

    def makeTimeSeriesPlot(self, object_settings):
        '''
        takes in object settings to build time series plot and write to XML
        :param object_settings: currently selected object settings dictionary
        :return: creates png in images dir and writes to XML file
        '''

        print('\n################################')
        print('Now making TimeSeries Plot.')
        print('################################\n')

        default_settings = self.loadDefaultPlotObject('timeseriesplot') #get default TS plot items
        object_settings = self.replaceDefaults(default_settings, object_settings) #overwrite the defaults with chapter file

        object_settings['split_by_year'], object_settings['years'], object_settings['yearstr'] = self.getPlotYears(object_settings)

        object_settings = self.confirm_axis(object_settings)

        for yi, year in enumerate(object_settings['years']):
            cur_obj_settings = pickle.loads(pickle.dumps(object_settings, -1))
            if object_settings['split_by_year']:
                yearstr = str(year)
            else:
                yearstr = object_settings['yearstr']
            cur_obj_settings = self.updateFlaggedValues(cur_obj_settings, '%%year%%', yearstr)

            if len(cur_obj_settings['axs']) == 1:
                figsize=(12, 6)
                pageformat = 'half'
            else:
                figsize=(12,18)
                pageformat = 'full'

            axis_weight = []
            for ax_settings in cur_obj_settings['axs']:
                if 'weight' in ax_settings.keys():
                    axis_weight.append(float(ax_settings['weight']))
                else:
                    axis_weight.append(1)

            if 'sharex' not in cur_obj_settings:
                cur_obj_settings['sharex'] == 'false'

            if cur_obj_settings['sharex'].lower() == 'true':
                fig, axes = plt.subplots(nrows=len(cur_obj_settings['axs']), sharex=True, figsize=figsize,
                                         gridspec_kw={'height_ratios': axis_weight})
            else:
                fig, axes = plt.subplots(ncols=1, nrows=len(cur_obj_settings['axs']), figsize=figsize,
                                         gridspec_kw={'height_ratios': axis_weight})

            plt.subplots_adjust(wspace=0, hspace=0)

            legend_left = False
            left_sided_axes = []
            legend_right = False
            right_sided_axes = []


            for axi, ax_settings in enumerate(cur_obj_settings['axs']):

                ax_settings = self.copyObjectSettingsToAxSetting(ax_settings, cur_obj_settings, ignore=['axs'])

                if len(cur_obj_settings['axs']) == 1:
                    ax = axes
                else:
                    ax = axes[axi]

                ax_settings = self.setTimeSeriesXlims(ax_settings, yearstr, object_settings['years'])

                ### Make Twin axis ###
                _usetwinx = False
                if 'twinx' in ax_settings.keys():
                    if ax_settings['twinx'].lower() == 'true':
                        _usetwinx = True

                if _usetwinx:
                    ax2 = ax.twinx()

                unitslist = []
                unitslist2 = []
                linedata = self.getTimeseriesData(ax_settings)
                linedata = self.mergeLines(linedata, ax_settings)
                ax_settings = self.configureSettingsForID('base', ax_settings)
                gatedata = self.getGateData(ax_settings)
                linedata = self.filterDataByYear(linedata, year)
                linedata = self.correctDuplicateLabels(linedata)
                for gateop in gatedata.keys():
                    gatedata[gateop]['gates'] = self.filterDataByYear(gatedata[gateop]['gates'], year)

                if 'relative' in ax_settings.keys():
                    if ax_settings['relative'].lower() == 'true':
                        RelativeMasterSet, RelativeLineSettings = self.getRelativeMasterSet(linedata)
                        if 'unitsystem' in ax_settings.keys():
                            RelativeMasterSet, RelativeLineSettings['units'] = self.convertUnitSystem(RelativeMasterSet,
                                                                                                      RelativeLineSettings['units'],
                                                                                                      ax_settings['unitsystem'])
                # LINE DATA #
                for line in linedata:
                    curline = linedata[line]
                    parameter, ax_settings['param_count'] = self.getParameterCount(curline, ax_settings)
                    i = ax_settings['param_count'][parameter]

                    values = curline['values']
                    dates = curline['dates']
                    units = curline['units']

                    if units == None:
                        if parameter != None:
                            try:
                                units = self.units[parameter]
                            except KeyError:
                                units = None

                    if isinstance(units, dict):
                        if 'unitsystem' in ax_settings.keys():
                            units = units[ax_settings['unitsystem'].lower()]
                        else:
                            units = None

                    if 'unitsystem' in ax_settings.keys():
                        values, units = self.convertUnitSystem(values, units, ax_settings['unitsystem'])

                    chkvals = WF.checkData(values)
                    if not chkvals:
                        print('Invalid Data settings for line:', line)
                        continue

                    if 'dateformat' in ax_settings.keys():
                        if ax_settings['dateformat'].lower() == 'jdate':
                            if isinstance(dates[0], dt.datetime):
                                dates = self.DatetimeToJDate(dates)
                        elif ax_settings['dateformat'].lower() == 'datetime':
                            if isinstance(dates[0], (int,float)):
                                dates = self.JDateToDatetime(dates)

                    line_settings = self.getDefaultLineSettings(curline, parameter, i)
                    line_settings = self.fixDuplicateColors(line_settings) #used the line, used param, then double up so subtract 1


                    if 'zorder' not in line_settings.keys():
                        line_settings['zorder'] = 4

                    if 'label' not in line_settings.keys():
                        line_settings['label'] = ''

                    if 'filterbylimits' not in line_settings.keys():
                        line_settings['filterbylimits'] = 'true' #set default

                    if line_settings['filterbylimits'].lower() == 'true':
                        if 'xlims' in object_settings.keys():
                            dates, values = self.limitXdata(dates, values, cur_obj_settings['xlims'])
                        if 'ylims' in object_settings.keys():
                            dates, values = self.limitYdata(dates, values, cur_obj_settings['ylims'])

                    curax = ax
                    axis2 = False
                    if _usetwinx:
                        if 'yaxis' in line_settings.keys():
                            if line_settings['yaxis'].lower() == 'right':
                                curax = ax2
                                axis2 = True

                    if units != '' and units != None:
                        if axis2:
                            unitslist2.append(units)
                        else:
                            unitslist.append(units)

                    if 'relative' in ax_settings:
                        if ax_settings['relative'].lower() == 'true':
                            if RelativeLineSettings['interval'] != None:
                                dates, values = self.changeTimeSeriesInterval(dates, values, RelativeLineSettings)
                            values = values/RelativeMasterSet

                    if line_settings['drawline'].lower() == 'true' and line_settings['drawpoints'].lower() == 'true':
                        curax.plot(dates, values, label=line_settings['label'], c=line_settings['linecolor'],
                                   lw=line_settings['linewidth'], ls=line_settings['linestylepattern'],
                                   marker=line_settings['symboltype'], markerfacecolor=line_settings['pointfillcolor'],
                                   markeredgecolor=line_settings['pointlinecolor'], markersize=float(line_settings['symbolsize']),
                                   markevery=int(line_settings['numptsskip']), zorder=float(line_settings['zorder']),
                                   alpha=float(line_settings['alpha']))

                    elif line_settings['drawline'].lower() == 'true':
                        curax.plot(dates, values, label=line_settings['label'], c=line_settings['linecolor'],
                                   lw=line_settings['linewidth'], ls=line_settings['linestylepattern'],
                                   zorder=float(line_settings['zorder']),
                                   alpha=float(line_settings['alpha']))

                    elif line_settings['drawpoints'].lower() == 'true':
                        curax.scatter(dates[::int(line_settings['numptsskip'])], values[::int(line_settings['numptsskip'])],
                                      marker=line_settings['symboltype'], facecolor=line_settings['pointfillcolor'],
                                      edgecolor=line_settings['pointlinecolor'], s=float(line_settings['symbolsize']),
                                      label=line_settings['label'], zorder=float(line_settings['zorder']),
                                      alpha=float(line_settings['alpha']))


                    self.addLogEntry({'type': line_settings['label'] + '_TimeSeries' if line_settings['label'] != '' else 'Timeseries',
                                      'name': self.ChapterRegion+'_'+yearstr,
                                      'description': ax_settings['description'],
                                      'units': units,
                                      'value_start_date': self.translateDateFormat(dates[0], 'datetime', '').strftime('%d %b %Y'),
                                      'value_end_date': self.translateDateFormat(dates[-1], 'datetime', '').strftime('%d %b %Y'),
                                      'logoutputfilename': curline['logoutputfilename']
                                      },
                                     isdata=True)
                # GATE DATA #
                if 'gatespacing' in ax_settings.keys():
                    gatespacing = float(ax_settings['gatespacing'])
                else:
                    gatespacing = 3
                gate_placement = gatespacing

                gategroup_labels = []
                gatelabels_positions = []
                gateop_rev = list(gatedata.keys())
                if len(gateop_rev) > 1:
                    gateop_rev.reverse()
                for ggi, gateop in enumerate(gateop_rev):
                    # gate_placement += ggi*gatespacing
                    gate_count = 0 #keep track of gate number in group
                    if 'label' in gatedata[gateop]:
                        gategroup_labels.append(gatedata[gateop]['label'].replace('\\n', '\n'))
                    elif 'flag' in gatedata[gateop]:
                        gategroup_labels.append(gatedata[gateop]['flag'].replace('\\n', '\n'))
                    else:
                        gategroup_labels.append(gateop)

                    gatelines_positions = []
                    for gate in gatedata[gateop]['gates'].keys():

                        curgate = gatedata[gateop]['gates'][gate]
                        values = curgate['values']
                        dates = curgate['dates']

                        if 'dateformat' in ax_settings.keys():
                            if ax_settings['dateformat'].lower() == 'jdate':
                                if isinstance(dates[0], dt.datetime):
                                    dates = self.DatetimeToJDate(dates)
                            elif ax_settings['dateformat'].lower() == 'datetime':
                                if isinstance(dates[0], (int,float)):
                                    dates = self.JDateToDatetime(dates)

                        gate_line_settings = self.getDefaultGateLineSettings(curgate, gate_count)

                        if 'zorder' not in gate_line_settings.keys():
                            gate_line_settings['zorder'] = 4

                        if 'label' not in gate_line_settings.keys():
                            gate_line_settings['label'] = '{0}_{1}'.format(gateop['label'], gate_count)

                        if 'filterbylimits' not in gate_line_settings.keys():
                            gate_line_settings['filterbylimits'] = 'true' #set default

                        if gate_line_settings['filterbylimits'].lower() == 'true':
                            if 'xlims' in ax_settings.keys():
                                dates, values = self.limitXdata(dates, values, ax_settings['xlims'])

                        if 'placement' in gate_line_settings.keys():
                            line_placement = float(gate_line_settings['placement'])
                        else:
                            line_placement = gate_placement

                        values *= line_placement
                        gatelines_positions.append(line_placement)

                        curax = ax
                        if _usetwinx:
                            if 'xaxis' in line_settings.keys():
                                if 'xaxis'.lower() == 'right':
                                    curax = ax2

                        if gate_line_settings['drawline'].lower() == 'true' and gate_line_settings['drawpoints'].lower() == 'true':
                            curax.plot(dates, values, label=gate_line_settings['label'], c=gate_line_settings['linecolor'],
                                       lw=gate_line_settings['linewidth'], ls=gate_line_settings['linestylepattern'],
                                       marker=gate_line_settings['symboltype'], markerfacecolor=gate_line_settings['pointfillcolor'],
                                       markeredgecolor=gate_line_settings['pointlinecolor'], markersize=float(gate_line_settings['symbolsize']),
                                       markevery=int(gate_line_settings['numptsskip']), zorder=float(gate_line_settings['zorder']),
                                       alpha=float(gate_line_settings['alpha']))

                        elif gate_line_settings['drawline'].lower() == 'true':
                            curax.plot(dates, values, label=gate_line_settings['label'], c=gate_line_settings['linecolor'],
                                       lw=gate_line_settings['linewidth'], ls=gate_line_settings['linestylepattern'],
                                       zorder=float(gate_line_settings['zorder']),
                                       alpha=float(gate_line_settings['alpha']))

                        elif gate_line_settings['drawpoints'].lower() == 'true':
                            curax.scatter(dates[::int(gate_line_settings['numptsskip'])], values[::int(gate_line_settings['numptsskip'])],
                                          marker=gate_line_settings['symboltype'], facecolor=gate_line_settings['pointfillcolor'],
                                          edgecolor=gate_line_settings['pointlinecolor'], s=float(gate_line_settings['symbolsize']),
                                          label=gate_line_settings['label'], zorder=float(gate_line_settings['zorder']),
                                          alpha=float(gate_line_settings['alpha']))

                        gate_count += 1 #keep track of gate number in group
                        gate_placement += 1 #keep track of gate palcement in space
                        self.addLogEntry({'type': gate_line_settings['label'] + '_GateTimeSeries' if gate_line_settings['label'] != '' else 'GateTimeseries',
                                          'name': self.ChapterRegion+'_'+yearstr,
                                          'description': ax_settings['description'],
                                          'units': 'BINARY',
                                          'value_start_date': self.translateDateFormat(dates[0], 'datetime', '').strftime('%d %b %Y'),
                                          'value_end_date': self.translateDateFormat(dates[-1], 'datetime', '').strftime('%d %b %Y'),
                                          'logoutputfilename': curgate['logoutputfilename']
                                          },
                                         isdata=True)

                    gatelabels_positions.append(np.average(gatelines_positions))
                    gate_placement += gatespacing

                if 'operationlines' in ax_settings.keys():
                    operationtimes = self.getGateOperationTimes(gatedata)
                    axs_to_add_line = [ax]
                    if 'allaxis' in ax_settings['operationlines'].keys():
                        if ax_settings['operationlines']['allaxis'].lower() == 'true':
                            axs_to_add_line = axes

                    opline_settings = self.getDefaultStraightLineSettings(ax_settings['operationlines'])

                    for ax_to_add_line in axs_to_add_line:
                        for operationTime in operationtimes:
                            if 'dateformat' in ax_settings.keys():
                                if ax_settings['dateformat'].lower() == 'jdate':
                                    if isinstance(operationTime, dt.datetime):
                                        operationTime = self.DatetimeToJDate(operationTime)
                                    elif isinstance(operationTime, str):
                                        try:
                                            operationTime = float(operationTime)
                                        except:
                                            operationTime = self.translateDateFormat(operationTime, 'datetime', '')
                                            operationTime = self.DatetimeToJDate(operationTime)
                                elif ax_settings['dateformat'].lower() == 'datetime':
                                    if isinstance(operationTime, (int,float)):
                                        operationTime = self.JDateToDatetime(operationTime)
                                    elif isinstance(operationTime, str):
                                        operationTime = self.translateDateFormat(operationTime, 'datetime', '')
                            else:
                                operationTime = self.translateDateFormat(operationTime, 'datetime', '')

                            if 'zorder' not in opline_settings.keys():
                                opline_settings['zorder'] = 3

                            ax_to_add_line.axvline(operationTime, c=opline_settings['linecolor'],
                                                   lw=opline_settings['linewidth'], ls=opline_settings['linestylepattern'],
                                                   zorder=float(opline_settings['zorder']),
                                                       alpha=float(opline_settings['alpha']))

                ### VERTICAL LINES ###
                if 'vlines' in ax_settings.keys():
                    for vline in ax_settings['vlines']:
                        vline_settings = self.getDefaultStraightLineSettings(vline)
                        try:
                            vline_settings['value'] = float(vline_settings['value'])
                        except:
                            vline_settings['value'] = self.translateDateFormat(vline_settings['value'], 'datetime', '')
                        if 'dateformat' in ax_settings.keys():
                            if ax_settings['dateformat'].lower() == 'jdate':
                                if isinstance(vline_settings['value'], dt.datetime):
                                    vline_settings['value'] = self.DatetimeToJDate(vline_settings['value'])
                                elif isinstance(vline_settings['value'], str):
                                    try:
                                        vline_settings['value'] = float(vline_settings['value'])
                                    except:
                                        vline_settings['value'] = self.translateDateFormat(vline_settings['value'], 'datetime', '')
                                        vline_settings['value'] = self.DatetimeToJDate(vline_settings['value'])
                            elif ax_settings['dateformat'].lower() == 'datetime':
                                if isinstance(vline_settings['value'], (int,float)):
                                    vline_settings['value'] = self.JDateToDatetime(vline_settings['value'])
                                elif isinstance(vline_settings['value'], str):
                                    vline_settings['value'] = self.translateDateFormat(vline_settings['value'], 'datetime', '')
                        else:
                            vline_settings['value'] = self.translateDateFormat(vline_settings['value'], 'datetime', '')

                        if 'label' not in vline_settings.keys():
                            vline_settings['label'] = None
                        if 'zorder' not in vline_settings.keys():
                            vline_settings['zorder'] = 3

                        ax.axvline(vline_settings['value'], label=vline_settings['label'], c=vline_settings['linecolor'],
                                   lw=vline_settings['linewidth'], ls=vline_settings['linestylepattern'],
                                   zorder=float(vline_settings['zorder']),
                                   alpha=float(vline_settings['alpha']))
                            
                ### Horizontal LINES ###
                if 'hlines' in ax_settings.keys():
                    for hline in ax_settings['hlines']:
                        hline_settings = self.getDefaultStraightLineSettings(hline)
                        if 'label' not in hline_settings.keys():
                            hline_settings['label'] = None
                        if 'zorder' not in hline_settings.keys():
                            hline_settings['zorder'] = 3
                        hline_settings['value'] = float(hline_settings['value'])

                        ax.axhline(hline_settings['value'], label=hline_settings['label'], c=hline_settings['linecolor'],
                                   lw=hline_settings['linewidth'], ls=hline_settings['linestylepattern'],
                                   zorder=float(hline_settings['zorder']),
                                   alpha=float(hline_settings['alpha']))

                plotunits = self.getPlotUnits(unitslist, ax_settings)
                plotunits2 = self.getPlotUnits(unitslist2, ax_settings)
                ax_settings = self.updateFlaggedValues(ax_settings, '%%units%%', plotunits)
                ax_settings = self.updateFlaggedValues(ax_settings, '%%units2%%', plotunits2)

                if axi == 0:
                    if 'title' in ax_settings.keys():
                        if 'titlesize' in ax_settings.keys():
                            titlesize = float(ax_settings['titlesize'])
                        elif 'fontsize' in ax_settings.keys():
                            titlesize = float(ax_settings['fontsize'])
                        else:
                            titlesize = 15
                        ax.set_title(ax_settings['title'], fontsize=titlesize)

                if 'gridlines' in ax_settings.keys():
                    if ax_settings['gridlines'].lower() == 'true':
                        ax.grid(True)

                if 'ylabel' in ax_settings.keys():
                    if 'ylabelsize' in ax_settings.keys():
                        ylabsize = float(ax_settings['ylabelsize'])
                    elif 'fontsize' in ax_settings.keys():
                        ylabsize = float(ax_settings['fontsize'])
                    else:
                        ylabsize = 12
                    ax.set_ylabel(ax_settings['ylabel'].replace("\\n", "\n"), fontsize=ylabsize)

                if 'xlabel' in ax_settings.keys():
                    if 'xlabelsize' in ax_settings.keys():
                        xlabsize = float(ax_settings['xlabelsize'])
                    elif 'fontsize' in ax_settings.keys():
                        xlabsize = float(ax_settings['fontsize'])
                    else:
                        xlabsize = 12
                    ax.set_xlabel(ax_settings['xlabel'].replace("\\n", "\n"), fontsize=xlabsize)

                if 'legend' in ax_settings.keys():
                    if ax_settings['legend'].lower() == 'true':
                        if 'legendsize' in ax_settings.keys():
                            legsize = float(ax_settings['legendsize'])
                        elif 'fontsize' in ax_settings.keys():
                            legsize = float(ax_settings['fontsize'])
                        else:
                            legsize = 12
                        if ax_settings['legend_outside'].lower() == 'true':
                            if _usetwinx:
                                legend_left = True
                                left_sided_axes.append(ax)
                                left_offset = ax.get_window_extent().x0 / ax.get_window_extent().width
                                ax.legend(loc='center left', bbox_to_anchor=(-left_offset, 0.5), ncol=1,fontsize=legsize)
                            else:
                                legend_right = True
                                right_sided_axes.append(ax)
                                ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1,fontsize=legsize)
                        else:
                            ax.legend(fontsize=legsize)

                self.formatDateXAxis(ax, ax_settings)

                if 'ylims' in ax_settings.keys():
                    if 'min' in ax_settings['ylims']:
                        ax.set_ylim(bottom=float(ax_settings['ylims']['min']))
                    else:
                        if len(gatedata.keys()) != 0:
                            ax.set_ylim(bottom=0)
                    if 'max' in ax_settings['ylims']:
                        ax.set_ylim(top=float(ax_settings['ylims']['max']))
                    else:
                        if len(gatedata.keys()) != 0:
                            ax.set_ylim(top=gate_placement)
                else:
                    if len(gatedata.keys()) != 0:
                        ax.set_ylim(bottom=0, top=gate_placement)

                if 'xticksize' in ax_settings.keys():
                    xticksize = float(ax_settings['xticksize'])
                elif 'fontsize' in ax_settings.keys():
                    xticksize = float(ax_settings['fontsize'])
                else:
                    xticksize = 10
                ax.tick_params(axis='x', labelsize=xticksize)

                if 'yticksize' in ax_settings.keys():
                    yticksize = float(ax_settings['yticksize'])
                elif 'fontsize' in ax_settings.keys():
                    yticksize = float(ax_settings['fontsize'])
                else:
                    yticksize = 10
                ax.tick_params(axis='y', labelsize=yticksize)

                if len(gatelabels_positions) > 0:
                    ax.set_yticks(gatelabels_positions)
                    ax.set_yticklabels(gategroup_labels, rotation=90, va='center', ha='center')
                    ax.tick_params(axis='both', which='both',color='w')

                if _usetwinx:
                    if 'ylabel2' in ax_settings.keys():
                        if 'ylabelsize2' in ax_settings.keys():
                            ylabsize2 = float(ax_settings['ylabelsize2'])
                        elif 'fontsize' in cur_obj_settings.keys():
                            ylabsize2 = float(ax_settings['fontsize'])
                        else:
                            ylabsize2 = 12
                        ax2.set_ylabel(ax_settings['ylabel2'].replace("\\n", "\n"), fontsize=ylabsize2)

                    if 'ylims2' in ax_settings.keys():
                        if 'min' in ax_settings['ylims2']:
                            ax2.set_ylim(bottom=float(ax_settings['ylims2']['min']))
                        if 'max' in ax_settings['ylims2']:
                            ax2.set_ylim(top=float(ax_settings['ylims2']['max']))

                    if 'yticksize2' in ax_settings.keys():
                        yticksize2 = float(ax_settings['yticksize2'])
                    elif 'fontsize' in ax_settings.keys():
                        yticksize2 = float(ax_settings['fontsize'])
                    else:
                        yticksize2 = 10
                    ax2.tick_params(axis='y', labelsize=yticksize2)

                    if 'legend2' in ax_settings.keys():
                        if ax_settings['legend2'].lower() == 'true':
                            if 'legendsize' in ax_settings.keys():
                                legsize = float(ax_settings['legendsize'])
                            elif 'legendsize2' in ax_settings.keys():
                                legsize = float(ax_settings['legendsize'])
                            elif 'fontsize' in ax_settings.keys():
                                legsize = float(ax_settings['fontsize'])
                            else:
                                legsize = 12
                            if ax_settings['legend_outside'].lower() == 'true':
                                legend_right = True
                                right_sided_axes.append(ax2)
                                right_offset = ax.get_window_extent().x0 / ax.get_window_extent().width
                                ax2.legend(loc='center left', bbox_to_anchor=(1+right_offset/2, 0.5), ncol=1,fontsize=legsize)
                            else:
                                ax2.legend(fontsize=legsize)

                    ax2.grid(False)
                    ax.set_zorder(ax2.get_zorder()+1) #axis called second will always be on top unless this
                    ax.patch.set_visible(False)

            plt.gcf().canvas.draw() #refresh so we can get legend stuff
            left_mod = 0
            for lax in left_sided_axes:
                lax_leg = lax.get_legend()
                lax_leg.get_window_extent()
                # lax_leg_width_ratio = lax_leg.get_window_extent().width / lax.get_window_extent().width
                lax_leg_width_ratio = lax_leg.get_window_extent().x1 / lax.get_window_extent().width
                if lax_leg_width_ratio > left_mod:
                    left_mod = lax_leg_width_ratio

            right_mod = 0
            for rax in right_sided_axes:
                rax_leg = rax.get_legend()
                rax_leg.get_window_extent()
                rax_leg_width_ratio = rax_leg.get_window_extent().width / rax.get_window_extent().width
                if rax_leg_width_ratio > right_mod:
                    right_mod = rax_leg_width_ratio

            basefigname = os.path.join(self.images_path, 'TimeSeriesPlot' + '_' + self.ChapterRegion.replace(' ','_')
                                       + '_' + yearstr)
            exists = True
            tempnum = 1
            tfn = basefigname
            while exists:
                if os.path.exists(tfn + '.png'):
                    tfn = basefigname + '_{0}'.format(tempnum)
                    tempnum += 1
                else:
                    exists = False
            figname = tfn + '.png'
            plt.savefig(figname, bbox_inches='tight')
            plt.close('all')

            if pageformat == 'half':
                self.XML.writeHalfPagePlot(os.path.basename(figname), cur_obj_settings['description'])
            if pageformat == 'full':
                self.XML.writeFullPagePlot(os.path.basename(figname), cur_obj_settings['description'])
            
            
    def makeProfileStatisticsTable(self, object_settings):
        '''
        Makes a table to compute stats based off of profile lines. Data is interpolated over a series of points
        determined by the user
        :param object_settings: currently selected object settings dictionary
        :return: writes table to XML
        '''
        print('\n################################')
        print('Now making Profile Stats Table.')
        print('################################\n')

        default_settings = self.loadDefaultPlotObject('profilestatisticstable')
        object_settings = self.replaceDefaults(default_settings, object_settings)
        object_settings['datakey'] = 'datapaths'

        ################# Get timestamps #################
        object_settings['datessource_flag'] = self.getDateSourceFlag(object_settings)
        object_settings['timestamps'] = self.getProfileTimestamps(object_settings)

        ################# Get units #################
        object_settings['plot_parameter'] = self.getPlotParameter(object_settings)

        ################# Get data #################
        data, line_settings = self.getProfileData(object_settings)
        line_settings = self.correctDuplicateLabels(line_settings)
        table_blueprint = pickle.loads(pickle.dumps(object_settings, -1))

        object_settings = self.configureSettingsForID('base', object_settings)

        ################ reformat data ###################
        data, object_settings = self.convertDepthsToElevations(data, object_settings)

        ################# Get plot units #################
        data, line_settings = self.convertProfileDataUnits(object_settings, data, line_settings)
        object_settings['units_list'] = self.getUnitsList(line_settings)
        object_settings['plot_units'] = self.getPlotUnits(object_settings['units_list'], object_settings)
        object_settings = self.updateFlaggedValues(object_settings, '%%units%%', object_settings['plot_units'])
        # table_blueprint = pickle.loads(pickle.dumps(object_settings, -1))

        table_blueprint = self.updateFlaggedValues(table_blueprint, '%%units%%', object_settings['plot_units'])

        self.commitProfileDataToMemory(data, line_settings, object_settings)
        data, object_settings = self.filterProfileData(data, line_settings, object_settings)

        object_settings['resolution'] = self.getProfileInterpResolution(object_settings)

        object_settings['split_by_year'], object_settings['years'], object_settings['yearstr'] = self.getPlotYears(object_settings)

        yrheaders, yrheaders_i = self.buildHeadersByTimestamps(object_settings['timestamps'], self.years)
        yrheaders = self.convertHeaderFormats(yrheaders, object_settings)
        if not object_settings['split_by_year']: #if we dont want to split by year, just make a big ass list
            yrheaders = [list(itertools.chain.from_iterable(yrheaders))]
            yrheaders_i = [list(itertools.chain.from_iterable(yrheaders_i))]
        for yi, year in enumerate(self.years):
            yearstr = object_settings['yearstr'][yi]
            object_desc = self.updateFlaggedValues(object_settings['description'], '%%year%%', yearstr)
            if self.iscomp:
                self.XML.writeDateControlledTableStart(object_desc, 'Statistics')
            else:
                self.XML.writeTableStart(object_desc, 'Statistics')
            for yhi, yrheader in enumerate(yrheaders[yi]):
                if self.iscomp:
                    #if a comparison, write the date column. Otherwise, this will be our header
                    self.XML.writeDateColumn(yrheader)
                header_i = yrheaders_i[yi][yhi]
                headings, rows = self.buildProfileStatsTable(table_blueprint, yrheaders[yi][yhi], line_settings)
                for hi,heading in enumerate(headings):
                    frmt_rows = []
                    for row in rows:
                        s_row = row.split('|')
                        rowname = s_row[0]
                        row_val = s_row[hi+1]
                        if '%%' in row_val:

                            stats_data = self.formatStatsProfileLineData(row_val, data, object_settings['resolution'],
                                                                         object_settings['usedepth'], header_i)
                            row_val, stat = self.getStatsLine(row_val, stats_data)
                            self.addLogEntry({'type': 'ProfileTableStatistic',
                                              'name': ' '.join([self.ChapterRegion, heading, stat]),
                                              'description': object_desc,
                                              'value': row_val,
                                              'function': stat,
                                              'units': object_settings['plot_units'],
                                              'value_start_date': yrheader,
                                              'value_end_date': yrheader,
                                              'logoutputfilename': ', '.join([line_settings[flag]['logoutputfilename'] for flag in line_settings])
                                              },
                                             isdata=True)

                        frmt_rows.append('{0}|{1}'.format(rowname, row_val))
                    self.XML.writeTableColumn(heading, frmt_rows)
                if self.iscomp:
                    self.XML.writeDateColumnEnd()
            self.XML.writeTableEnd()

    def makeProfilePlot(self, object_settings):
        '''
        takes in object settings to build profile plot and write to XML
        :param object_settings: currently selected object settings dictionary
        :return: creates png in images dir and writes to XML file
        '''

        print('\n################################')
        print('Now making Profile Plot.')
        print('################################\n')

        default_settings = self.loadDefaultPlotObject('profileplot')
        object_settings = self.replaceDefaults(default_settings, object_settings)
        object_settings['datakey'] = 'lines'

        obj_desc = self.updateFlaggedValues(object_settings['description'], '%%year%%', self.years_str)
        self.XML.writeProfilePlotStart(obj_desc)

        ################# Get timestamps #################
        object_settings['datessource_flag'] = self.getDateSourceFlag(object_settings)
        object_settings['timestamps'] = self.getProfileTimestamps(object_settings)

        ################# Get units #################
        object_settings['plot_parameter'] = self.getPlotParameter(object_settings)

        ################# Get data #################
        data, line_settings = self.getProfileData(object_settings)
        line_settings = self.correctDuplicateLabels(line_settings)

        object_settings = self.configureSettingsForID('base', object_settings)
        gatedata = self.getGateData(object_settings)

        ################# Get plot units #################
        data, line_settings = self.convertProfileDataUnits(object_settings, data, line_settings)
        object_settings['units_list'] = self.getUnitsList(line_settings)
        object_settings['plot_units'] = self.getPlotUnits(object_settings['units_list'], object_settings)
        object_settings = self.updateFlaggedValues(object_settings, '%%units%%', object_settings['plot_units'])

        ################ convert Elevs ################
        data, object_settings = self.convertDepthsToElevations(data, object_settings)

        self.commitProfileDataToMemory(data, line_settings, object_settings)
        linedata, object_settings = self.filterProfileData(data, line_settings, object_settings)

        object_settings['split_by_year'], object_settings['years'], object_settings['yearstr'] = self.getPlotYears(object_settings)

        ################ Build Plots ################
        for yi, year in enumerate(object_settings['years']):
            if object_settings['split_by_year']:
                yearstr = str(year)
            else:
                yearstr = object_settings['yearstr']

            t_stmps = self.filterTimestepByYear(object_settings['timestamps'], year)

            prof_indices = [np.where(object_settings['timestamps'] == n)[0][0] for n in t_stmps]
            n = int(object_settings['profilesperrow']) * int(object_settings['rowsperpage']) #Get number of plots on page
            page_indices = [prof_indices[i * n:(i + 1) * n] for i in range((len(prof_indices) + n - 1) // n)]
            cur_obj_settings = pickle.loads(pickle.dumps(object_settings, -1))
            cur_obj_settings = self.updateFlaggedValues(cur_obj_settings, '%%year%%', yearstr) #TODO: reudce the settings

            for page_i, pgi in enumerate(page_indices):

                subplot_rows, subplot_cols = WF.getSubplotConfig(len(pgi), int(cur_obj_settings['profilesperrow']))
                n_nrow_active = np.ceil(len(pgi) / subplot_cols)

                fig, axs = plt.subplots(nrows=int(object_settings['rowsperpage']), ncols=int(object_settings['profilesperrow']), figsize=(7,10))

                for i in range(n):

                    current_row = i // int(object_settings['rowsperpage'])
                    current_col = i % int(object_settings['rowsperpage'])
                    ax = axs[current_row, current_col]
                    if i+1 > len(pgi):
                        ax.axis('off')
                        continue
                    else:
                        j = pgi[i]

                    if object_settings['usedepth'].lower() == 'true':
                        ax.invert_yaxis()

                    for li, line in enumerate(data.keys()):

                        try:
                            values = data[line]['values'][j]
                            if len(values) == 0:
                                print('No values for {0} on {1}'.format(line, object_settings['timestamps'][j]))
                                continue
                            msk = np.where(~np.isnan(values))
                            values = values[msk]
                        except IndexError:
                            print('No values for {0} on {1}'.format(line, object_settings['timestamps'][j]))
                            continue

                        try:
                            if object_settings['usedepth'].lower() == 'true':
                                levels = data[line]['depths'][j][msk]
                            else:
                                levels = data[line]['elevations'][j][msk]
                            if not WF.checkData(levels):
                                print('Non Viable depths/elevations for {0} on {1}'.format(line, object_settings['timestamps'][j]))
                                continue
                        except IndexError:
                            print('Non Viable depths/elevations for {0} on {1}'.format(line, object_settings['timestamps'][j]))
                            continue

                        if not WF.checkData(values):
                            continue

                        current_ls = line_settings[line] #we have all the settings we need..
                        current_ls = self.getDefaultLineSettings(current_ls, object_settings['plot_parameter'], li)
                        current_ls = self.fixDuplicateColors(current_ls) #used the line, used param, then double up so subtract 1

                        if current_ls['drawline'].lower() == 'true' and current_ls['drawpoints'].lower() == 'true':
                            ax.plot(values, levels, label=current_ls['label'], c=current_ls['linecolor'],
                                    lw=current_ls['linewidth'], ls=current_ls['linestylepattern'],
                                    marker=current_ls['symboltype'], markerfacecolor=current_ls['pointfillcolor'],
                                    markeredgecolor=current_ls['pointlinecolor'], markersize=float(current_ls['symbolsize']),
                                    markevery=int(current_ls['numptsskip']), zorder=int(current_ls['zorder']),
                                    alpha=float(current_ls['alpha']))

                        elif current_ls['drawline'].lower() == 'true':
                            ax.plot(values, levels, label=current_ls['label'], c=current_ls['linecolor'],
                                    lw=current_ls['linewidth'], ls=current_ls['linestylepattern'],
                                    zorder=int(current_ls['zorder']),
                                    alpha=float(current_ls['alpha']))

                        elif current_ls['drawpoints'].lower() == 'true':
                            ax.scatter(values[::int(current_ls['numptsskip'])], levels[::int(current_ls['numptsskip'])],
                                       marker=current_ls['symboltype'], facecolor=current_ls['pointfillcolor'],
                                       edgecolor=current_ls['pointlinecolor'], s=float(current_ls['symbolsize']),
                                       label=current_ls['label'], zorder=int(current_ls['zorder']),
                                       alpha=float(current_ls['alpha']))

                    ### HLINES ###
                    if 'hlines' in cur_obj_settings.keys():
                        for hline_settings in cur_obj_settings['hlines']:
                            if 'value' in hline_settings.keys():
                                value = float(hline_settings['value'])
                                units = None
                            else:
                                dates, values, units = self.getTimeSeries(hline_settings)
                                hline_idx = np.where(object_settings['timestamps'][j] == dates)
                                value = values[hline_idx]

                            if 'parameter' in hline_settings:
                                if object_settings['usedepth'].lower() == 'true':
                                    if hline_settings['parameter'].lower() == 'elevation':
                                        value = 0 #top of the water, should always be 0
                                elif object_settings['usedepth'].lower() == 'false':
                                    if hline_settings['parameter'].lower() == 'depth':
                                        valueconv = self.convertDepthsToElevations({'hline': {'depths': [value],
                                                                                              'elevations': []}})
                                        value = valueconv['hline']['elevation'][0]

                            #currently cant convert these units..
                            # if units != None:
                            #     valueconv, units = self.convertUnitSystem(value, units, object_settings['unitsystem'])
                            #     value = valueconv[0]

                            ### instead, use scalar to be manual
                            if 'scalar' in hline_settings.keys():
                                value *= float(hline_settings['scalar'])

                            hline_settings = self.getDefaultStraightLineSettings(hline_settings)
                            if 'label' not in hline_settings.keys():
                                hline_settings['label'] = None
                            if 'zorder' not in hline_settings.keys():
                                hline_settings['zorder'] = 3

                            ax.axhline(value, label=hline_settings['label'], c=hline_settings['linecolor'],
                                       lw=hline_settings['linewidth'], ls=hline_settings['linestylepattern'],
                                       zorder=float(hline_settings['zorder']),
                                       alpha=float(hline_settings['alpha']))

                    ### VERTICAL LINES ###
                    if 'vlines' in cur_obj_settings.keys():
                        for vline in cur_obj_settings['vlines']:
                            vline_settings = self.getDefaultStraightLineSettings(vline)
                            if 'value' in vline_settings.keys():
                                value = float(vline_settings['value'])
                                units = None
                            else:
                                dates, values, units = self.getTimeSeries(vline_settings)
                                vline_idx = np.where(object_settings['timestamps'][j] == dates)
                                value = values[vline_idx]

                            if 'label' not in vline_settings.keys():
                                vline_settings['label'] = None
                            if 'zorder' not in vline_settings.keys():
                                vline_settings['zorder'] = 3

                            ax.axvline(value, label=vline_settings['label'], c=vline_settings['linecolor'],
                                       lw=vline_settings['linewidth'], ls=vline_settings['linestylepattern'],
                                       zorder=float(vline_settings['zorder']),
                                       alpha=float(vline_settings['alpha']))

                    if 'xlims' in object_settings.keys():
                        if 'min' in object_settings['xlims']:
                            ax.set_xlim(left=float(object_settings['xlims']['min']))
                        if 'max' in object_settings['xlims']:
                            ax.set_xlim(right=float(object_settings['xlims']['max']))
                    if 'ylims' in object_settings.keys():
                        if 'min' in object_settings['ylims']:
                            ax.set_ylim(bottom=float(object_settings['ylims']['min']))
                        if 'max' in object_settings['ylims']:
                            ax.set_ylim(top=float(object_settings['ylims']['max']))

                    ### GATES ###
                    # gategroups = {}
                    gateconfig = {}
                    if len(gatedata.keys()) > 0:

                        for ggi, gategroup in enumerate(gatedata.keys()):
                            gatetop = None
                            gatebottom = None
                            gatemiddle = None
                            gateop_has_value = False
                            gate_count = 0 #keep track of gate number in group
                            numgates = len(gatedata[gategroup]['gates'])
                            gatepoint_xpositions = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1],numgates+2)[1:-1] #+2 for start and end

                            cur_gateop = gatedata[gategroup]

                            if 'top' in cur_gateop.keys():
                                gatetop = float(cur_gateop['top'])

                            if 'bottom' in cur_gateop.keys():
                                gatebottom = float(cur_gateop['bottom'])

                            if 'middle' in cur_gateop.keys():
                                gatemiddle = float(cur_gateop['middle'])
                            elif gatetop != None and gatebottom != None:
                                gatemiddle = np.mean([gatetop, gatebottom])

                            if gatetop == None and gatebottom != None and gatemiddle == None: #bottom no top/middle
                                gatetop = gatebottom + float(object_settings['defaultgateenvelope'])
                                gatemiddle = gatebottom + float(object_settings['defaultgateenvelope'])/2
                            elif gatetop != None and gatebottom == None and gatemiddle == None: #top no bottom/middle
                                gatebottom = gatetop - float(object_settings['defaultgateenvelope'])
                                gatemiddle = gatetop - float(object_settings['defaultgateenvelope'])/2
                            elif gatetop == None and gatebottom == None and gatemiddle != None: #only middle
                                gatebottom = gatemiddle - float(object_settings['defaultgateenvelope'])/2
                                gatetop = gatemiddle + float(object_settings['defaultgateenvelope'])/2

                            for gate in cur_gateop['gates'].keys():

                                curgate = cur_gateop['gates'][gate]

                                values = curgate['values']
                                dates = curgate['dates']

                                if 'dateformat' in cur_obj_settings.keys():
                                    if cur_obj_settings['dateformat'].lower() == 'datetime':
                                        if isinstance(dates[0], (int,float)):
                                            dates = self.JDateToDatetime(dates)

                                gatemsk = np.where(object_settings['timestamps'][j] == dates)
                                value = values[gatemsk][0]
                                xpos = gatepoint_xpositions[gate_count]
                                if gategroup not in gateconfig.keys():
                                    gateconfig[gategroup] = {gate: value}
                                else:
                                    gateconfig[gategroup][gate] = value


                                if not np.isnan(value):
                                    gateop_has_value = True
                                    showgate = False
                                    if 'showgates' in cur_gateop.keys():
                                        if cur_gateop['showgates'].lower() == 'true':
                                            showgate = True

                                    if showgate:
                                        ax.scatter(xpos, gatemiddle, edgecolor='black', facecolor='black', marker='o')

                                gate_count += 1 #keep track of gate number in group
                                self.addLogEntry({'type': gate + '_GateTimeSeries' if gate != '' else 'GateTimeseries',
                                                  'name': self.ChapterRegion+'_'+yearstr,
                                                  'description': cur_obj_settings['description'],
                                                  'units': 'BINARY',
                                                  'value_start_date': self.translateDateFormat(dates[0], 'datetime', '').strftime('%d %b %Y'),
                                                  'value_end_date': self.translateDateFormat(dates[-1], 'datetime', '').strftime('%d %b %Y'),
                                                  'logoutputfilename': curgate['logoutputfilename']
                                                  },
                                                 isdata=True)

                            if 'color' in cur_gateop.keys():
                                color = cur_gateop['color']
                                default_color = self.def_colors[ggi]
                                color = self.confirmColor(color, default_color)
                            if 'top' in cur_gateop.keys():
                                ax.axhline(gatetop, color=color, zorder=-7)
                            if 'bottom' in cur_gateop.keys():
                                ax.axhline(gatebottom, color=color, zorder=-7)
                            if 'middle' in cur_gateop.keys():
                                ax.axhline(gatemiddle, color=color, zorder=-7)

                            if gateop_has_value:
                                ax.axhspan(gatebottom,gatetop, alpha=0.5, color=color, zorder=-8)

                    show_xlabel, show_ylabel = self.getPlotLabelMasks(i, len(pgi), subplot_cols)

                    if cur_obj_settings['gridlines'].lower() == 'true':
                        ax.grid(zorder=-9)

                    if show_ylabel:
                        if 'ylabel' in cur_obj_settings.keys():
                            if 'ylabelsize' in object_settings.keys():
                                ylabsize = float(object_settings['ylabelsize'])
                            elif 'fontsize' in object_settings.keys():
                                ylabsize = float(object_settings['fontsize'])
                            else:
                                ylabsize = 12
                            ax.set_ylabel(cur_obj_settings['ylabel'], fontsize=ylabsize)

                    if show_xlabel:
                        if 'xlabel' in cur_obj_settings.keys():
                            if 'xlabelsize' in object_settings.keys():
                                xlabsize = float(object_settings['xlabelsize'])
                            elif 'fontsize' in object_settings.keys():
                                xlabsize = float(object_settings['fontsize'])
                            else:
                                xlabsize = 12
                            if 'bottomtext' in cur_obj_settings:
                                labelpad = 12
                            else:
                                labelpad = 0
                            ax.set_xlabel(cur_obj_settings['xlabel'], fontsize=xlabsize, labelpad=labelpad)

                    if 'xticksize' in object_settings.keys():
                        xticksize = float(object_settings['xticksize'])
                    elif 'fontsize' in object_settings.keys():
                        xticksize = float(object_settings['fontsize'])
                    else:
                        xticksize = 10
                    ax.tick_params(axis='x', labelsize=xticksize)

                    if 'yticksize' in object_settings.keys():
                        yticksize = float(object_settings['yticksize'])
                    elif 'fontsize' in object_settings.keys():
                        yticksize = float(object_settings['fontsize'])
                    else:
                        yticksize = 10
                    ax.tick_params(axis='y', labelsize=yticksize)

                    cur_timestamp = object_settings['timestamps'][j]
                    if 'dateformat' in object_settings:
                        if object_settings['dateformat'].lower() == 'datetime':
                            cur_timestamp = self.translateDateFormat(cur_timestamp, 'datetime', '')
                            ttl_str = cur_timestamp.strftime('%d %b %Y')
                        elif object_settings['dateformat'].lower() == 'jdate':
                            cur_timestamp = self.translateDateFormat(cur_timestamp, 'jdate', '')
                            ttl_str = str(cur_timestamp)
                        else:
                            ttl_str = cur_timestamp
                    else:
                        ttl_str = object_settings['timestamps'][j].strftime('%d %b %Y') #should get set to datetime anyways..

                    if 'datetext' in cur_obj_settings.keys():
                        if cur_obj_settings['datetext'].lower() == 'true':
                            xbufr = 0.05
                            ybufr = 0.05
                            xl = ax.get_xlim()
                            yl = ax.get_ylim()
                            xtext = xl[0] + xbufr * (xl[1] - xl[0])
                            ytext = yl[1] - ybufr * (yl[1] - yl[0])
                            ax.text(xtext, ytext, ttl_str, ha='left', va='top', size=10, #TODO: make this variable
                                    bbox=dict(boxstyle='round',facecolor='w', alpha=0.35),
                                    zorder=10)

                    if 'bottomtext' in cur_obj_settings.keys():
                        bottomtext_str = []
                        for text in cur_obj_settings['bottomtext']:
                            if text.lower() == 'date':
                                bottomtext_str.append(object_settings['timestamps'][j].strftime('%m/%d/%Y'))
                            elif text.lower() == 'gateconfiguration':
                                gateconfignum = self.getGateConfigurationDays(gateconfig, gatedata, object_settings['timestamps'][j])
                                bottomtext_str.append(str(gateconfignum))
                            elif text.lower() == 'gateblend':
                                gateblendnum = self.getGateBlendDays(gateconfig, gatedata, object_settings['timestamps'][j])
                                bottomtext_str.append(str(gateblendnum))
                            else:
                                bottomtext_str.append(text)
                        bottomtext = ', '.join(bottomtext_str)
                        # plt.text(0.02, -0.2, bottomtext, fontsize=8, color='red', transform=ax.transFigure)
                        # if show_xlabel:
                        #     ax.annotate(bottomtext, xy=(0.02, -40), fontsize=6, color='red', xycoords='axes points')
                        # else:
                        ax.annotate(bottomtext, xy=(0.02, -22), fontsize=6, color='red', xycoords='axes points')

                plt.tight_layout()


                if 'legend' in cur_obj_settings.keys():
                    if cur_obj_settings['legend'].lower() == 'true':
                        leg_labels = []
                        leg_handles = []
                        for ax in axs:
                            for subax in ax:
                                leg_handle_labels = subax.get_legend_handles_labels()
                                for li in range(len(leg_handle_labels[0])):
                                    if leg_handle_labels[1][li] not in leg_labels:
                                        leg_labels.append(leg_handle_labels[1][li])
                                        leg_handles.append(leg_handle_labels[0][li])

                        if len(leg_labels) > 0:
                            if 'legendsize' in cur_obj_settings.keys():
                                legsize = float(cur_obj_settings['legendsize'])
                            elif 'fontsize' in cur_obj_settings.keys():
                                legsize = float(cur_obj_settings['fontsize'])
                            else:
                                legsize = 12

                            ncolumns = 3

                            # n_legends_row = np.ceil(len(linedata.keys()) / ncolumns) * .65
                            n_legends_row = np.ceil(len(leg_handles) / ncolumns) * .65
                            if n_legends_row < 1:
                                n_legends_row = 1

                            plt.subplots_adjust(bottom=.1*n_legends_row)
                            fig_ratio = (axs[int(n_nrow_active)-1,0].bbox.extents[1] - (fig.bbox.height * (.1025 * n_legends_row))) / fig.bbox.height

                            plt.legend(bbox_to_anchor=(.5,fig_ratio), loc="lower center", fontsize=legsize,
                                       bbox_transform=fig.transFigure, ncol=ncolumns, handles=leg_handles,
                                       labels=leg_labels)


                figname = 'ProfilePlot_{0}_{1}_{2}_{3}_{4}.png'.format(self.ChapterName, yearstr,
                                                                       object_settings['plot_parameter'], self.plugin,
                                                                       page_i)

                plt.savefig(os.path.join(self.images_path, figname))
                plt.close('all')

                ################################################

                description = '{0}: {1} of {2}'.format(cur_obj_settings['description'], page_i+1, len(page_indices))
                self.XML.writeProfilePlotFigure(figname, description)

                self.addLogEntry({'type': 'ProfilePlot',
                                  'name': self.ChapterRegion,
                                  'description': description,
                                  'units': object_settings['plot_units'],
                                  'value_start_date': self.translateDateFormat(object_settings['timestamps'][pgi[0]],
                                                                               'datetime', '').strftime('%d %b %Y'),
                                  'value_end_date': self.translateDateFormat(object_settings['timestamps'][pgi[-1]],
                                                                             'datetime', '').strftime('%d %b %Y'),
                                  'logoutputfilename': ', '.join([line_settings[flag]['logoutputfilename'] for flag in line_settings])
                                  },
                                 isdata=True)

        self.XML.writeProfilePlotEnd()

    def makeErrorStatisticsTable(self, object_settings):
        '''
        takes in object settings to build error stats table and write to XML
        :param object_settings: currently selected object settings dictionary
        :return: writes to XML file
        '''

        print('\n################################')
        print('Now making Error Stats table.')
        print('################################\n')

        default_settings = self.loadDefaultPlotObject('errorstatisticstable')
        object_settings = self.replaceDefaults(default_settings, object_settings)

        object_settings['split_by_year'], object_settings['years'], object_settings['yearstr'] = self.getPlotYears(object_settings)

        data = self.getTableData(object_settings)
        data = self.mergeLines(data, object_settings)

        headings, rows = self.buildTable(object_settings, object_settings['split_by_year'], data)

        object_settings = self.configureSettingsForID('base', object_settings)

        object_settings['units_list'] = self.getUnitsList(data)
        object_settings['plot_units'] = self.getPlotUnits(object_settings['units_list'], object_settings)

        object_settings = self.updateFlaggedValues(object_settings, '%%units%%', object_settings['plot_units'])
        rows = self.updateFlaggedValues(rows, '%%units%%', object_settings['plot_units'])
        headings = self.updateFlaggedValues(headings, '%%units%%', object_settings['plot_units'])

        data = self.filterTableData(data, object_settings)
        data = self.correctTableUnits(data, object_settings)

        if 'description' in object_settings.keys():
            desc = object_settings['description']
        else:
            desc = ''

        self.XML.writeTableStart(desc, 'Statistics')
        for i, yearheader in enumerate(headings):
            year = yearheader[0]
            header = yearheader[1]
            frmt_rows = []
            for row in rows:
                s_row = row.split('|')
                rowname = s_row[0]
                row_val = s_row[i+1]
                if '%%' in row_val:
                    rowdata, sr_month = self.getStatsLineData(row_val, data, year=year)
                    if len(rowdata) == 0:
                        row_val = None
                    else:
                        row_val, stat = self.getStatsLine(row_val, rowdata)

                        data_start_date, data_end_date = self.getTableDates(year, object_settings)
                        self.addLogEntry({'type': 'Statistic',
                                          'name': ' '.join([self.ChapterRegion, header, stat]),
                                          'description': desc,
                                          'value': row_val,
                                          'function': stat,
                                          'units': object_settings['plot_units'],
                                          'value_start_date': self.translateDateFormat(data_start_date, 'datetime', ''),
                                          'value_end_date': self.translateDateFormat(data_end_date, 'datetime', ''),
                                          'logoutputfilename': ', '.join([data[flag]['logoutputfilename'] for flag in data])
                                          },
                                         isdata=True)

                header = '' if header == None else header
                frmt_rows.append('{0}|{1}'.format(rowname, row_val))
            self.XML.writeTableColumn(header, frmt_rows)
        self.XML.writeTableEnd()

    def makeMonthlyStatisticsTable(self, object_settings):
        '''
        takes in object settings to build monthly stats table and write to XML
        :param object_settings: currently selected object settings dictionary
        :return: writes to XML file
        '''

        print('\n################################')
        print('Now making Monthly Stats Table.')
        print('################################\n')

        default_settings = self.loadDefaultPlotObject('monthlystatisticstable')
        object_settings = self.replaceDefaults(default_settings, object_settings)
        object_settings['split_by_year'], object_settings['years'], object_settings['yearstr'] = self.getPlotYears(object_settings)

        data = self.getTableData(object_settings)
        data = self.mergeLines(data, object_settings)

        headings, rows = self.buildTable(object_settings, object_settings['split_by_year'], data)

        object_settings= self.configureSettingsForID('base', object_settings)
        object_settings['units_list'] = self.getUnitsList(data)
        object_settings['plot_units'] = self.getPlotUnits(object_settings['units_list'], object_settings)

        object_settings = self.updateFlaggedValues(object_settings, '%%units%%', object_settings['plot_units'])

        data = self.filterTableData(data, object_settings)
        data = self.correctTableUnits(data, object_settings)

        self.XML.writeTableStart(object_settings['description'], 'Month')
        for i, yearheader in enumerate(headings):
            year = yearheader[0]
            header = yearheader[1]
            frmt_rows = []
            for row in rows:
                s_row = row.split('|')
                rowname = s_row[0]
                row_val = s_row[i+1]
                if '%%' in row_val:
                    rowdata, sr_month = self.getStatsLineData(row_val, data, year=year)
                    if len(rowdata) == 0:
                        row_val = None
                    else:

                        row_val, stat = self.getStatsLine(row_val, rowdata)

                        data_start_date, data_end_date = self.getTableDates(year, object_settings, month=sr_month)
                        self.addLogEntry({'type': 'Statistic',
                                          'name': ' '.join([self.ChapterRegion, header, stat]),
                                          'description': object_settings['description'],
                                          'value': row_val,
                                          'units': object_settings['units_list'],
                                          'function': stat,
                                          'value_start_date': self.translateDateFormat(data_start_date, 'datetime', ''),
                                          'value_end_date': self.translateDateFormat(data_end_date, 'datetime', ''),
                                          'logoutputfilename': ', '.join([data[flag]['logoutputfilename'] for flag in data])
                                          },
                                         isdata=True)

                header = '' if header == None else header
                frmt_rows.append('{0}|{1}'.format(rowname, row_val))
            self.XML.writeTableColumn(header, frmt_rows)
        self.XML.writeTableEnd()

    def makeContourPlot(self, object_settings):
        '''
        takes in object settings to build contour plot and write to XML
        :param object_settings: currently selected object settings dictionary
        :return: creates png in images dir and writes to XML file
        '''

        print('\n################################')
        print('Now making Contour Plot.')
        print('################################\n')

        default_settings = self.loadDefaultPlotObject('contourplot') #get default TS plot items
        object_settings = self.replaceDefaults(default_settings, object_settings) #overwrite the defaults with chapter file

        object_settings['split_by_year'], object_settings['years'], object_settings['yearstr'] = self.getPlotYears(object_settings)

        for yi, year in enumerate(object_settings['years']):
            cur_obj_settings = pickle.loads(pickle.dumps(object_settings, -1))
            if object_settings['split_by_year']:
                yearstr = str(year)
            else:
                yearstr = object_settings['yearstr']

            cur_obj_settings = self.setTimeSeriesXlims(cur_obj_settings, yearstr, object_settings['years'])

            #NOTES
            #Data structure:
            #2D array of dates[distance from source]
            # array of dates corresponding to the number of the first D of array above
            # supplementary array for distances corrsponding to the second D of array above
            #ex
            #[[1,2,3,5],[2,3,4,2],[5,3,2,5]] #values per date at a distance
            #[01jan2016, 04Feb2016, 23May2016] #dates
            #[0, 19, 25, 35] #distances

            contoursbyID = self.getContourData(cur_obj_settings)
            contoursbyID = self.filterDataByYear(contoursbyID, year)
            selectedContourIDs = self.getUsedIDs(contoursbyID)

            if len(selectedContourIDs) == 1:
                figsize=(12, 6)
                pageformat = 'half'
            else:
                figsize=(12,12)
                pageformat = 'full'

            if pageformat == 'full':
                height_ratios = []
                for i in range(len(selectedContourIDs)):
                    if i == len(selectedContourIDs)-1:
                        height_ratios.append(1)
                    else:
                        height_ratios.append(.75)
                fig, axes = plt.subplots(ncols=1, nrows=len(selectedContourIDs), sharex=True, figsize=figsize,
                                         gridspec_kw={'height_ratios':height_ratios})
            else:
                fig, axes = plt.subplots(ncols=1, nrows=1, figsize=figsize,
                                         )

            for IDi, ID in enumerate(selectedContourIDs):
                contour_settings = pickle.loads(pickle.dumps(cur_obj_settings, -1))
                contour_settings = self.configureSettingsForID(ID, contour_settings)
                # contours = contoursbyID[ID]
                contours = self.selectContoursByID(contoursbyID, ID)
                values, dates, distance, transitions = self.stackContours(contours)
                if len(selectedContourIDs) == 1:
                    axes = [axes]

                ax = axes[IDi]

                for contourname in contours:
                    contour = contours[contourname]
                    parameter, contour_settings['param_count'] = self.getParameterCount(contour, contour_settings)

                if 'units' in contour_settings.keys():
                    units = contour_settings['units']
                else:
                    if 'parameter' in contour_settings.keys():
                        parameter = contour_settings['parameter']
                    else:
                        parameter = ''
                        top_count = 0
                        for key in contour_settings['param_count'].keys():
                            if contour_settings['param_count'][key] > top_count:
                                parameter = key
                    try:
                        units = self.units[parameter]
                    except KeyError:
                        units = None

                if isinstance(units, dict):
                    if 'unitsystem' in contour_settings.keys():
                        units = units[contour_settings['unitsystem'].lower()]
                    else:
                        units = None

                if 'unitsystem' in contour_settings.keys():
                    values, units = self.convertUnitSystem(values, units, contour_settings['unitsystem']) #TODO: confirm

                chkvals = WF.checkData(values)
                if not chkvals:
                    print('Invalid Data settings for contour plot year {0}'.format(year))
                    continue

                if 'dateformat' in contour_settings.keys():
                    if contour_settings['dateformat'].lower() == 'jdate':
                        if isinstance(dates[0], dt.datetime):
                            dates = self.DatetimeToJDate(dates)
                    elif contour_settings['dateformat'].lower() == 'datetime':
                        if isinstance(dates[0], (int,float)):
                            dates = self.JDateToDatetime(dates)

                if 'label' not in contour_settings.keys():
                    contour_settings['label'] = ''

                if 'description' not in contour_settings.keys():
                    contour_settings['description'] = ''

                contour_settings = self.getDefaultContourSettings(contour_settings)

                if 'filterbylimits' not in contour_settings.keys():
                    contour_settings['filterbylimits'] = 'true' #set default

                if contour_settings['filterbylimits'].lower() == 'true':
                    if 'xlims' in contour_settings.keys():
                        dates, values = self.limitXdata(dates, values, contour_settings['xlims'])
                    if 'ylims' in contour_settings.keys():
                        dates, values = self.limitYdata(dates, values, contour_settings['ylims'], baseOn=distance)

                if 'min' in contour_settings['colorbar']:
                    vmin = float(contour_settings['colorbar']['min'])
                else:
                    vmin = np.nanmin(values)
                if 'max' in contour_settings['colorbar']:
                    vmax = float(contour_settings['colorbar']['max'])
                else:
                    vmax = np.nanmax(values)

                contr = ax.contourf(dates, distance, values.T, cmap=contour_settings['colorbar']['colormap'],
                                    vmin=vmin, vmax=vmax,
                                    levels=np.linspace(vmin, vmax, int(contour_settings['colorbar']['bins'])), #add one to get the desired number..
                                    extend='both') #the .T transposes the array so dates on bottom TODO:make extend variable
                ax.invert_yaxis()

                self.addLogEntry({'type': contour_settings['label'] + '_ContourPlot' if contour_settings['label'] != '' else 'ContourPlot',
                                  'name': self.ChapterRegion+'_'+yearstr,
                                  'description': contour_settings['description'],
                                  'units': units,
                                  'value_start_date': self.translateDateFormat(dates[0], 'datetime', '').strftime('%d %b %Y'),
                                  'value_end_date': self.translateDateFormat(dates[-1], 'datetime', '').strftime('%d %b %Y'),
                                  'logoutputfilename': contour['logoutputfilename']
                                  },
                                 isdata=True)

                contour_settings = self.updateFlaggedValues(contour_settings, '%%units%%', units)

                if 'contourlines' in contour_settings.keys():
                    for contourline in contour_settings['contourlines']:
                        if 'value' in contourline.keys():
                            val = float(contourline['value'])
                        else:
                            print('No Value set for contour line.')
                            continue
                        contourline = self.getDefaultContourLineSettings(contourline)
                        cs = ax.contour(contr, levels=[val], linewidths=[float(contourline['linewidth'])], colors=[contourline['linecolor']],
                                        linestyles=contourline['linestylepattern'], alpha=float(contourline['alpha']))
                        if 'contourlinetext' in contourline.keys():
                            if contourline['contourlinetext'].lower() == 'true':
                                ax.clabel(cs, inline_spacing=contourline['inline_spacing'],
                                          fontsize=contourline['fontsize'], inline=contourline['text_inline'])
                        if 'show_in_legend' in contourline.keys():
                            if contourline['show_in_legend'].lower() == 'true':
                                if 'label' in contourline.keys():
                                    label = contourline['label']
                                else:
                                    label = str(val)
                                cl_leg = ax.plot([], [], c=contourline['linecolor'], ls=contourline['linestylepattern'],
                                                 alpha=float(contourline['alpha']), lw=float(contourline['linewidth']),
                                                 label=label)

                ### VERTICAL LINES ###
                if 'vlines' in contour_settings.keys():
                    for vline in contour_settings['vlines']:
                        vline_settings = self.getDefaultStraightLineSettings(vline)
                        try:
                            vline_settings['value'] = float(vline_settings['value'])
                        except:
                            vline_settings['value'] = self.translateDateFormat(vline_settings['value'], 'datetime', '')
                        if 'dateformat' in contour_settings.keys():
                            if contour_settings['dateformat'].lower() == 'jdate':
                                if isinstance(vline_settings['value'], dt.datetime):
                                    vline_settings['value'] = self.DatetimeToJDate(vline_settings['value'])
                                elif isinstance(vline_settings['value'], str):
                                    try:
                                        vline_settings['value'] = float(vline_settings['value'])
                                    except:
                                        vline_settings['value'] = self.translateDateFormat(vline_settings['value'], 'datetime', '')
                                        vline_settings['value'] = self.DatetimeToJDate(vline_settings['value'])
                            elif contour_settings['dateformat'].lower() == 'datetime':
                                if isinstance(vline_settings['value'], (int,float)):
                                    vline_settings['value'] = self.JDateToDatetime(vline_settings['value'])
                                elif isinstance(vline_settings['value'], str):
                                    vline_settings['value'] = self.translateDateFormat(vline_settings['value'], 'datetime', '')
                        else:
                            vline_settings['value'] = self.translateDateFormat(vline_settings['value'], 'datetime', '')

                        if 'label' not in vline_settings.keys():
                            vline_settings['label'] = None
                        if 'zorder' not in vline_settings.keys():
                            vline_settings['zorder'] = 3

                        ax.axvline(vline_settings['value'], label=vline_settings['label'], c=vline_settings['linecolor'],
                                   lw=vline_settings['linewidth'], ls=vline_settings['linestylepattern'],
                                   zorder=float(vline_settings['zorder']),
                                   alpha=float(vline_settings['alpha']))

                ### Horizontal LINES ###
                if 'hlines' in contour_settings.keys():
                    for hline in contour_settings['hlines']:
                        hline_settings = self.getDefaultStraightLineSettings(hline)
                        if 'label' not in hline_settings.keys():
                            hline_settings['label'] = None
                        if 'zorder' not in hline_settings.keys():
                            hline_settings['zorder'] = 3
                        hline_settings['value'] = float(hline_settings['value'])

                        ax.axhline(hline_settings['value'], label=hline_settings['label'], c=hline_settings['linecolor'],
                                   lw=hline_settings['linewidth'], ls=hline_settings['linestylepattern'],
                                   zorder=float(hline_settings['zorder']),
                                   alpha=float(hline_settings['alpha']))

                if 'transitions' in contour_settings.keys():
                    for transkey in transitions.keys():
                        transition_start = transitions[transkey]
                        trans_name = None
                        hline = self.getDefaultStraightLineSettings(contour_settings['transitions'])

                        linecolor = self.prioritizeKey(contours[transkey], hline, 'linecolor')
                        linestylepattern = self.prioritizeKey(contours[transkey], hline, 'linestylepattern')
                        alpha = self.prioritizeKey(contours[transkey], hline, 'alpha')
                        linewidth = self.prioritizeKey(contours[transkey], hline, 'linewidth')

                        ax.axhline(y=transition_start, c=linecolor, ls=linestylepattern, alpha=float(alpha),
                                   lw=float(linewidth))
                        if 'name' in contour_settings['transitions'].keys():
                            trans_flag = contour_settings['transitions']['name'].lower() #blue:pink:white:pink:blue
                            text_settings = self.getDefaultTextSettings(contour_settings['transitions'])

                            if trans_flag in contours[transkey].keys():
                                trans_name = contours[transkey][trans_flag]
                            if trans_name != None:

                                trans_y_ratio = abs(1.0 - (transition_start / max(ax.get_ylim()) + .01)) #dont let user touch this

                                fontcolor = self.prioritizeKey(contours[transkey], text_settings, 'fontcolor')
                                fontsize = self.prioritizeKey(contours[transkey], text_settings, 'fontsize')
                                horizontalalignment = self.prioritizeKey(contours[transkey], text_settings, 'horizontalalignment')
                                text_x_pos = self.prioritizeKey(contours[transkey], text_settings, 'text_x_pos', backup=0.001)

                                ax.text(float(text_x_pos), trans_y_ratio, trans_name, c=fontcolor, size=float(fontsize),
                                        transform=ax.transAxes, horizontalalignment=horizontalalignment,
                                        verticalalignment='top')
                if self.iscomp:
                    if 'modeltext' in contour_settings.keys():
                        modeltext = contour_settings['modeltext']
                    else:
                        modeltext = self.SimulationName
                    plt.text(1.02, 0.5, modeltext, fontsize=12, transform=ax.transAxes, verticalalignment='center', horizontalalignment='center', rotation='vertical')


                if 'gridlines' in contour_settings.keys():
                    if contour_settings['gridlines'].lower() == 'true':
                        ax.grid(True)

                if 'ylabel' in contour_settings.keys():
                    if 'ylabelsize' in contour_settings.keys():
                        ylabsize = float(contour_settings['ylabelsize'])
                    elif 'fontsize' in contour_settings.keys():
                        ylabsize = float(contour_settings['fontsize'])
                    else:
                        ylabsize = 12
                    ax.set_ylabel(contour_settings['ylabel'], fontsize=ylabsize)

                if 'ylims' in contour_settings.keys():
                    if 'min' in contour_settings['ylims']:
                        ax.set_ylim(bottom=float(contour_settings['ylims']['min']))
                    if 'max' in contour_settings['ylims']:
                        ax.set_ylim(top=float(contour_settings['ylims']['max']))

                if 'xticksize' in contour_settings.keys():
                    xticksize = float(contour_settings['xticksize'])
                elif 'fontsize' in contour_settings.keys():
                    xticksize = float(contour_settings['fontsize'])
                else:
                    xticksize = 10
                ax.tick_params(axis='x', labelsize=xticksize)

                if 'yticksize' in contour_settings.keys():
                    yticksize = float(contour_settings['yticksize'])
                elif 'fontsize' in contour_settings.keys():
                    yticksize = float(contour_settings['fontsize'])
                else:
                    yticksize = 10
                ax.tick_params(axis='y', labelsize=yticksize)

            # #stuff to call once per plot
            self.configureSettingsForID('base', cur_obj_settings)
            cur_obj_settings = self.updateFlaggedValues(cur_obj_settings, '%%units%%', units)

            if 'title' in cur_obj_settings.keys():
                if 'titlesize' in cur_obj_settings.keys():
                    titlesize = float(object_settings['titlesize'])
                elif 'fontsize' in cur_obj_settings.keys():
                    titlesize = float(object_settings['fontsize'])
                else:
                    titlesize = 15
                axes[0].set_title(cur_obj_settings['title'], fontsize=titlesize)

            if 'xlabel' in cur_obj_settings.keys():
                if 'xlabelsize' in cur_obj_settings.keys():
                    xlabsize = float(cur_obj_settings['xlabelsize'])
                elif 'fontsize' in cur_obj_settings.keys():
                    xlabsize = float(cur_obj_settings['fontsize'])
                else:
                    xlabsize = 12
                axes[-1].set_xlabel(cur_obj_settings['xlabel'], fontsize=xlabsize)

            self.formatDateXAxis(axes[-1], cur_obj_settings)

            if 'legend' in cur_obj_settings.keys():
                if cur_obj_settings['legend'].lower() == 'true':
                    if 'legendsize' in cur_obj_settings.keys():
                        legsize = float(cur_obj_settings['legendsize'])
                    elif 'fontsize' in cur_obj_settings.keys():
                        legsize = float(cur_obj_settings['fontsize'])
                    else:
                        legsize = 12
                    if len(axes[0].get_legend_handles_labels()[0]) > 0:
                        plt.legend(fontsize=legsize)

            cbar = plt.colorbar(contr, ax=axes[-1], orientation='horizontal', aspect=50.)
            locs = np.linspace(vmin, vmax, int(cur_obj_settings['colorbar']['bins']))[::int(cur_obj_settings['colorbar']['skipticks'])]
            cbar.set_ticks(locs)
            cbar.set_ticklabels(locs.round(2))
            if 'label' in cur_obj_settings['colorbar']:
                if 'labelsize' in cur_obj_settings['colorbar'].keys():
                    labsize = float(cur_obj_settings['colorbar']['labelsize'])
                elif 'fontsize' in cur_obj_settings['colorbar'].keys():
                    labsize = float(cur_obj_settings['colorbar']['fontsize'])
                else:
                    labsize = 12
                cbar.set_label(cur_obj_settings['colorbar']['label'], fontsize=labsize)

            # plt.tight_layout()
            plt.subplots_adjust(hspace=0.05)

            if 'description' not in cur_obj_settings.keys():
                cur_obj_settings['description'] = ''

            basefigname = os.path.join(self.images_path, 'ContourPlot' + '_' + self.ChapterRegion.replace(' ','_')
                                       + '_' + yearstr)
            exists = True
            tempnum = 1
            tfn = basefigname
            while exists:
                if os.path.exists(tfn + '.png'):
                    tfn = basefigname + '_{0}'.format(tempnum)
                    tempnum += 1
                else:
                    exists = False
            figname = tfn + '.png'
            plt.savefig(figname, bbox_inches='tight')
            plt.close('all')

            if pageformat == 'full':
                self.XML.writeFullPagePlot(os.path.basename(figname), cur_obj_settings['description'])
            elif pageformat == 'half':
                self.XML.writeHalfPagePlot(os.path.basename(figname), cur_obj_settings['description'])


    def makeBuzzPlot(self, object_settings):
        '''
        takes in object settings to build buzzplots and write to XML
        :param object_settings: currently selected object settings dictionary
        :return: creates png in images dir and writes to XML file
        '''

        print('\n################################')
        print('Now making Buzz Plot.')
        print('################################\n')

        default_settings = self.loadDefaultPlotObject('buzzplot')
        object_settings = self.replaceDefaults(default_settings, object_settings)
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot()

        ### Make Twin axis ###
        if 'twinx' in object_settings.keys():
            if object_settings['twinx'].lower() == 'true':
                _usetwinx = True
        if 'twiny' in object_settings.keys():
            if object_settings['twiny'].lower() == 'true':
                _usetwiny = True

        if _usetwinx:
            ax2 = ax.twinx()

            if 'ylabel2' in object_settings.keys():
                if 'ylabelsize2' in object_settings.keys():
                    ylabsize2 = float(object_settings['ylabelsize2'])
                elif 'fontsize' in object_settings.keys():
                    ylabsize2 = float(object_settings['fontsize'])
                else:
                    ylabsize2 = 12
                ax2.set_ylabel(object_settings['ylabel2'], fontsize=ylabsize2)

            if 'ylims2' in object_settings.keys():
                if 'min' in object_settings['ylims2']:
                    ax2.set_ylim(bottom=float(object_settings['ylims2']['min']))
                if 'max' in object_settings['ylims2']:
                    ax2.set_ylim(top=float(object_settings['ylims2']['max']))

            if 'yticksize2' in object_settings.keys():
                yticksize2 = float(object_settings['yticksize2'])
            elif 'fontsize' in object_settings.keys():
                yticksize2 = float(object_settings['fontsize'])
            else:
                yticksize2 = 10
            ax2.tick_params(axis='y', labelsize=yticksize2)

            ax2.grid(False)
            ax.set_zorder(ax2.get_zorder()+1) #axis called second will always be on top unless this
            ax.patch.set_visible(False)

        ### Grid Lines ###
        if 'gridlines' in object_settings.keys():
            if object_settings['gridlines'].lower() == 'true':
                ax.grid(zorder=-1)

        ### Plot Lines ###
        stackplots = {}
        data = self.getTimeseriesData(object_settings)
        data = self.mergeLines(data, object_settings)

        object_settings = self.configureSettingsForID('base', object_settings)

        for i, line in enumerate(data.keys()):
            curline = data[line]
            values = curline['values']
            dates = curline['dates']

            if 'target' in line.keys():
                values = self.buzzTargetSum(dates, values, float(line['target']))
            else:
                if isinstance(values, dict):
                    if len(values.keys()) == 1: #single struct station, like leakage..
                        values = values[list(values.keys())[0]]
                        values = self.pickByParameter(values, line)
                    else:
                        print('Too many values to iterate. check if line should have a target, or an incorrect number')
                        print('of structures are defined.')
                        print('Line:', line)
                        continue

            chkvals = WF.checkData(values)
            if not chkvals:
                print('Invalid Data settings for line:', line)
                continue

            if 'dateformat' in object_settings.keys():
                if object_settings['dateformat'].lower() == 'jdate':
                    if isinstance(dates[0], dt.datetime):
                        dates = self.DatetimeToJDate(dates)
                elif object_settings['dateformat'].lower() == 'datetime':
                    if isinstance(dates[0], (float, int)):
                        dates = self.JDateToDatetime(dates)
                else:
                    if isinstance(dates[0], (float, int)):
                        dates = self.JDateToDatetime(dates)

            if 'unitsystem' in object_settings.keys():
                values, units = self.convertUnitSystem(values, units, object_settings['unitsystem'])

            if 'parameter' in line.keys():
                line_settings = self.getDefaultLineSettings(line, line['parameter'], i)
            else:
                line_settings = self.getDefaultLineSettings(line, None, i)

            if 'scalar' in line.keys():
                try:
                    scalar = float(line['scalar'])
                    values = scalar * values
                except ValueError:
                    print('Invalid Scalar. {0}'.format(line['scalar']))
                    continue

            if 'filterbylimits' not in line_settings.keys():
                line_settings['filterbylimits'] = 'true' #set default

            if line_settings['filterbylimits'].lower() == 'true':
                if _usetwinx:
                    if 'xaxis' in line.keys():
                        if line['xaxis'].lower() == 'left':
                            xaxis = 'left'
                        else:
                            xaxis = 'right'
                    else:
                        xaxis = 'left'
                else:
                    xaxis = 'left'

                if xaxis == 'left':
                    if 'xlims' in object_settings.keys():
                        dates, values = self.limitXdata(dates, values, object_settings['xlims'])
                    if 'ylims' in object_settings.keys():
                        dates, values = self.limitYdata(dates, values, object_settings['ylims'])
                else:
                    if 'xlims2' in object_settings.keys():
                        dates, values = self.limitXdata(dates, values, object_settings['xlims'])
                    if 'ylims' in object_settings.keys():
                        dates, values = self.limitYdata(dates, values, object_settings['ylims2'])

            if 'linetype' in line.keys():
                if line['linetype'].lower() == 'stacked': #stacked plots need to be added at the end..
                    if _usetwinx:
                        if 'xaxis' in line.keys():
                            axis = line['xaxis'].lower()
                    else:
                        axis = 'left' #if not twinx, then only can use left

                    if axis not in stackplots.keys(): #left or right
                        stackplots[axis] = []
                    stackplots[axis].append({'values': values,
                                             'dates': dates,
                                             'label': line_settings['label'],
                                             'color': line_settings['linecolor']})

            else: #otherwise we're normal lines
                if _usetwinx:
                    if 'xaxis' in line_settings:
                        if line['xaxis'].lower() == 'left':
                            curax = ax
                        else:
                            curax = ax2
                    else:
                        curax = ax

                if 'zorder' not in line_settings.keys():
                    line_settings['zorder'] = 4

                if line_settings['drawline'].lower() == 'true' and line_settings['drawpoints'].lower() == 'true':
                    curax.plot(dates, values, label=line_settings['label'], c=line_settings['linecolor'],
                               lw=line_settings['linewidth'], ls=line_settings['linestylepattern'],
                               marker=line_settings['symboltype'], markerfacecolor=line_settings['pointfillcolor'],
                               markeredgecolor=line_settings['pointlinecolor'], markersize=float(line_settings['symbolsize']),
                               markevery=int(line_settings['numptsskip']), zorder=int(line_settings['zorder']),
                               alpha=float(line_settings['alpha']))

                elif line_settings['drawline'].lower() == 'true':
                    curax.plot(dates, values, label=line_settings['label'], c=line_settings['linecolor'],
                               lw=line_settings['linewidth'], ls=line_settings['linestylepattern'],
                               zorder=int(line_settings['zorder']),
                               alpha=float(line_settings['alpha']))

                elif line_settings['drawpoints'].lower() == 'true':
                    curax.scatter(dates[::int(line_settings['numptsskip'])], values[::int(line_settings['numptsskip'])],
                                  marker=line_settings['symboltype'], facecolor=line_settings['pointfillcolor'],
                                  edgecolor=line_settings['pointlinecolor'], s=float(line_settings['symbolsize']),
                                  label=line_settings['label'], zorder=int(line_settings['zorder']),
                                  alpha=float(line_settings['alpha']))

                self.addLogEntry({'type': line_settings['label'] + '_BuzzPlot' if line_settings['label'] != '' else 'BuzzPlot',
                                  'name': self.ChapterRegion,
                                  'description': object_settings['description'],
                                  'units': units,
                                  'value_start_date': self.translateDateFormat(dates[0], 'datetime', '').strftime('%d %b %Y'),
                                  'value_end_date': self.translateDateFormat(dates[-1], 'datetime', '').strftime('%d %b %Y'),
                                  'logoutputfilename': line['logoutputfilename']
                                  },
                                 isdata=True)

        for stackplot_ax in stackplots.keys():
            if stackplot_ax == 'left':
                curax = ax
            elif stackplot_ax == 'right':
                curax = ax2

            sps = stackplots[stackplot_ax]
            values = [n['values'] for n in sps]
            dates = [n['dates'] for n in sps]
            labels = [n['label'] for n in sps]
            colors = [n['color'] for n in sps]

            matched_dates = list(set(dates[0]).intersection(*dates)) #find dates that ALL dates have.
            matched_dates.sort()

            #now filter values associated with dates not in this list
            for di, datelist in enumerate(dates):
                mask_date_idx = [ni for ni, date in enumerate(datelist) if date in matched_dates]
                values[di] = np.asarray(values[di])[mask_date_idx]

            curax.stackplot(matched_dates, values, labels=labels, colors=colors, zorder=2)

        plt.title(object_settings['title'])

        if 'ylabel' in object_settings.keys():
            if 'ylabelsize' in object_settings.keys():
                ylabelsize = float(object_settings['ylabelsize'])
            elif 'fontsize' in object_settings.keys():
                ylabelsize = float(object_settings['fontsize'])
            else:
                ylabelsize = 12
            ax.set_ylabel(object_settings['ylabel'], fontsize=ylabelsize)

        if 'yticksize' in object_settings.keys():
            yticksize = float(object_settings['yticksize'])
            yticksize = float(object_settings['fontsize'])
        else:
            yticksize = 10
        ax.tick_params(axis='y', labelsize=yticksize)

        if 'xlabel' in object_settings.keys():
            if 'xlabelsize' in object_settings.keys():
                xlabelsize = float(object_settings['xlabelsize'])
            elif 'fontsize' in object_settings.keys():
                xlabelsize = float(object_settings['fontsize'])
            else:
                xlabelsize = 12
            ax.set_xlabel(object_settings['xlabel'], fontsize=xlabelsize)

        if 'xticksize' in object_settings.keys():
            xticksize = float(object_settings['xticksize'])
        elif 'fontsize' in object_settings.keys():
            xticksize = float(object_settings['fontsize'])
        else:
            xticksize = 10
        ax.tick_params(axis='x', labelsize=xticksize)

        ### Add everything to legend ###
        if object_settings['legend'].lower() == 'true':
            lines, labels = ax.get_legend_handles_labels()
            if _usetwinx:
                lines2, labels2 = ax2.get_legend_handles_labels()
                leg_lines = lines + lines2
                leg_labels = labels + labels2
            else:
                leg_lines = lines
                leg_labels = labels

            if 'legendsize' in object_settings.keys():
                legsize = float(object_settings['legendsize'])
            elif 'fontsize' in object_settings.keys():
                legsize = float(object_settings['fontsize'])
            else:
                legsize = 12

            ax.legend(leg_lines, leg_labels, fontsize=legsize).set_zorder(102) #always on top)

        ### Xaxis formatting ###
        self.formatDateXAxis(ax, object_settings)

        if 'ylims' in object_settings.keys():
            if 'min' in object_settings['ylims']:
                ax.set_ylim(bottom=float(object_settings['ylims']['min']))
            if 'max' in object_settings['ylims2']:
                ax.set_ylim(top=float(object_settings['ylims']['max']))

        if _usetwiny:
            ax2 = ax.twiny()
            if 'xlabel2' in object_settings.keys():
                if 'xlabelsize2' in object_settings.keys():
                    xlabelsize2 = float(object_settings['xlabelsize2'])
                elif 'fontsize' in object_settings.keys():
                    xlabelsize2 = float(object_settings['fontsize'])
                else:
                    xlabelsize2 = 12
                ax2.set_xlabel(object_settings['xlabel2'], fontsize=xlabelsize2)

            self.formatDateXAxis(ax2, object_settings, twin=True)

            if 'xticksize2' in object_settings.keys():
                xticksize2 = float(object_settings['xticksize2'])
            elif 'fontsize' in object_settings.keys():
                xticksize2 = float(object_settings['fontsize'])
            else:
                xticksize2 = 10
            ax2.tick_params(axis='x', labelsize=xticksize2)

            ax2.grid(False)

        ax.set_axisbelow(True) #lets legend be on top.

        basefigname = os.path.join(self.images_path, 'BuzzPlot' + '_' + self.ChapterRegion.replace(' ','_'))
        exists = True
        tempnum = 1
        tfn = basefigname
        while exists:
            if os.path.exists(tfn + '.png'):
                tfn = basefigname + '_{0}'.format(tempnum)
                tempnum += 1
            else:
                exists = False
        figname = tfn + '.png'
        plt.savefig(figname, bbox_inches='tight')
        plt.close('all')

        self.XML.writeTimeSeriesPlot(os.path.basename(figname), object_settings['description'])

    def readSimulationInfo(self, simulationInfoFile):
        '''
        reads sim info XML file and organizes paths and variables into a list for iteration
        :param simulationInfoFile: full path to simulation information XML file from WAT
        :return: class variables:
                    self.Simulations
                    self.reportType
                    self.studyDir
                    self.observedData
        '''

        self.Simulations = []
        tree = ET.parse(simulationInfoFile)
        root = tree.getroot()

        self.reportType = root.find('ReportType').text
        self.studyDir = root.find('Study/Directory').text
        self.observedDir = root.find('Study/ObservedData').text

        if self.reportType == 'alternativecomparison':
            self.iscomp = True
        else:
            self.iscomp = False

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

            try:
                simulationInfo['ID'] = simulation.find('ID').text
            except AttributeError:
                simulationInfo['ID'] = 'base'


            modelAlternatives = []
            for modelAlt in simulation.find('ModelAlternatives'):
                modelAlternatives.append({'name': modelAlt.find('Name').text,
                                          'program': modelAlt.find('Program').text,
                                          'fpart': modelAlt.find('FPart').text,
                                          'directory': modelAlt.find('Directory').text})

            simulationInfo['modelalternatives'] = modelAlternatives
            self.Simulations.append(simulationInfo)

    def readSimulationsCSV(self):
        '''
        reads the Simulation file and gets the region info
        :return: class variable
                    self.SimulationCSV
        '''

        self.SimulationCSV = WDR.readSimulationFile(self.baseSimulationName, self.studyDir)

    def readComparisonSimulationsCSV(self):
        '''
        Reads in the simulation CSV but for comparison plots. Comparison plots have '_comparison' appended to the end of them,
        but are built in general the same as regular Simulation CSV files.
        :return:
        '''
        self.SimulationCSV = WDR.readSimulationFile(self.SimulationVariables['base']['baseSimulationName'], self.studyDir, iscomp=self.iscomp)

    def recordTimeSeriesData(self, data, line):
        '''
        organizes line information and places it into a data dictionary
        :param data: dictionary containing line data
        :param line: dictionary containing line settings
        :return: updated data dictionary
        '''

        #TODO: split into 2 like profiles
        dates, values, units = self.getTimeSeries(line)
        if WF.checkData(values):
            flag = line['flag']
            if flag in data.keys():
                count = 1
                newflag = flag + '_{0}'.format(count)
                while newflag in data.keys():
                    count += 1
                    newflag = flag + '_{0}'.format(count)
                flag = newflag
                print('The new flag is {0}'.format(newflag))
            datamem_key = self.buildDataMemoryKey(line)
            if 'units' in line.keys() and units == None:
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

    def recordProfileData(self, data, line_settings, line, timestamps):
        '''
        organizes line information and places it into a data dictionary
        :param data: dictionary containing data
        :param line_settings: dictionary containing all line settings
        :param line: dictionary containing line settings
        :param timestamps: list of dates to get data
        :return: updated data dictionary, updated line_settings dictionary
        '''

        vals, elevations, depths, times, flag = self.getProfileValues(line, timestamps) #Test this speed for grabbing all profiles and then choosing
        if len(vals) > 0:
            datamem_key = self.buildDataMemoryKey(line)
            if 'units' in line.keys():
                units = line['units']
            else:
                units = None

            if line['flag'] in line_settings.keys() or line['flag'] in data.keys():
                datakey = '{0}_{1}'.format(line['flag'], line['numtimesused'])
            else:
                datakey = line['flag']

            line_settings[datakey]  = {'units': units,
                                      'numtimesused': line['numtimesused'],
                                      'logoutputfilename': datamem_key
                                       }

            data[datakey] = {'values': vals,
                             'elevations': elevations,
                             'depths': depths,
                             'times': times
                             }

            for key in line.keys():
                if key not in line_settings[datakey].keys():
                    line_settings[datakey][key] = line[key]

        return data, line_settings

    def readGraphicsDefaultFile(self):
        '''
        sets up path for graphics default file in study and reads the xml
        :return: class variable
                    self.graphicsDefault
        '''

        graphicsDefaultfile = os.path.join(self.studyDir, 'reports', 'Graphics_Defaults.xml')
        # graphicsDefaultfile = os.path.join(self.default_dir, 'Graphics_Defaults.xml') #TODO: implement with build
        self.graphicsDefault = WDR.readGraphicsDefaults(graphicsDefaultfile)

    def readDefaultLineStylesFile(self):
        '''
        sets up path for default line styles file and reads the xml
        :return: class variable
                    self.defaultLineStyles
        '''

        defaultLinesFile = os.path.join(self.studyDir, 'reports', 'defaultLineStyles.xml')
        # defaultLinesFile = os.path.join(self.default_dir, 'defaultLineStyles.xml') #TODO: implement with build
        self.defaultLineStyles = WDR.readDefaultLineStyle(defaultLinesFile)

    def readDefinitionsFile(self, simorder):
        '''
        reads the chapter definitions file defined in the plugin csv file for a specified simulation
        :param simorder: simulation dictionary object
        :return: class variable
                    self.ChapterDefinitions
        '''

        ChapterDefinitionsFile = os.path.join(self.studyDir, 'reports', simorder['deffile'])
        self.ChapterDefinitions = WDR.readChapterDefFile(ChapterDefinitionsFile)

    def setSimulationDateTimes(self, ID):
        '''
        sets the simulation start time and dates from string format. If timestamp says 24:00, converts it to be correct
        Datetime format of the next day at 00:00
        :return: class varables
                    self.StartTime
                    self.EndTime
        '''

        StartTimeStr = self.SimulationVariables[ID]['StartTimeStr']
        EndTimeStr = self.SimulationVariables[ID]['EndTimeStr']

        if '24:00' in StartTimeStr:
            tstrtmp = StartTimeStr.replace('24:00', '23:00')
            StartTime = dt.datetime.strptime(tstrtmp, '%d %B %Y, %H:%M')
            StartTime += dt.timedelta(hours=1)
        else:
            StartTime = dt.datetime.strptime(StartTimeStr, '%d %B %Y, %H:%M')
        self.SimulationVariables[ID]['StartTime'] = StartTime

        if '24:00' in EndTimeStr:
            tstrtmp = EndTimeStr.replace('24:00', '23:00')
            EndTime = dt.datetime.strptime(tstrtmp, '%d %B %Y, %H:%M')
            EndTime += dt.timedelta(hours=1)
        else:
            EndTime = dt.datetime.strptime(EndTimeStr, '%d %B %Y, %H:%M')
        self.SimulationVariables[ID]['EndTime'] = EndTime

    def setSimulationCSVVars(self, simlist):
        '''
        set variables pertaining to a specified simulation
        :param simlist: dictionary of specified simulation
        :return: class variables
                    self.plugin
                    self.modelAltName
                    self.defFile
        '''

        self.plugins = simlist['plugins']
        self.modelAltNames = simlist['modelaltnames']
        self.defFile = simlist['deffile']

    def setSimulationVariables(self, simulation):
        '''
        sets various class variables for selected variable
        sets simulation dates and times
        :param simulation: simulation dictionary object for specified simulation
        :return: class variables
                    self.Data_Memory
                    self.SimulationName
                    self.baseSimulationName
                    self.simulationDir
                    self.DSSFile
                    self.StartTimeStr
                    self.EndTimeStr
                    self.LastComputed
                    self.ModelAlternatives
        '''

        # self.Data_Memory = {}
        ID = simulation['ID']
        self.SimulationVariables[ID] = {}
        self.SimulationVariables[ID]['SimulationName'] = simulation['name']
        self.SimulationVariables[ID]['baseSimulationName'] = simulation['basename']
        self.SimulationVariables[ID]['simulationDir'] = simulation['directory']
        self.SimulationVariables[ID]['DSSFile'] = simulation['dssfile']
        self.SimulationVariables[ID]['StartTimeStr'] = simulation['starttime']
        self.SimulationVariables[ID]['EndTimeStr'] = simulation['endtime']
        self.SimulationVariables[ID]['LastComputed'] = simulation['lastcomputed']
        self.SimulationVariables[ID]['ModelAlternatives'] = simulation['modelalternatives']
        self.setSimulationDateTimes(ID)

    def setTimeSeriesXlims(self, cur_obj_settings, yearstr, years):
        '''
        gets the xlimits for time series. This can be dependent on year, so needs to be looped over.
        :param cur_obj_settings: current plotting object settings dictionary
        :param yearstr: current year string
        :param years: list of years
        :return: updated cur_obj_settings dict
        '''

        if 'ALLYEARS' not in years:
            cur_obj_settings = self.updateFlaggedValues(cur_obj_settings, '%%year%%', yearstr)
        else:
            if 'xlims' in cur_obj_settings.keys():
                if 'min' in cur_obj_settings['xlims']:
                    cur_obj_settings['xlims']['min'] = self.updateFlaggedValues(cur_obj_settings['xlims']['min'],
                                                                                '%%year%%', str(self.years[0]))
                if 'max' in cur_obj_settings['xlims']:
                    cur_obj_settings['xlims']['max'] = self.updateFlaggedValues(cur_obj_settings['xlims']['max'],
                                                                                '%%year%%', str(self.years[-1]))
            cur_obj_settings = self.updateFlaggedValues(cur_obj_settings, '%%year%%', yearstr)

        return cur_obj_settings

    def selectContoursByID(self, contoursbyID, ID):
        output_contours = {}
        for key in contoursbyID:
            if contoursbyID[key]['ID'] == ID:
                output_contours[key] = contoursbyID[key]
        return output_contours

    def stackContours(self, contours):
        output_values = np.array([])
        output_dates = np.array([])
        output_distance = np.array([])
        transitions = {}
        for contourname in contours.keys():
            contour = contours[contourname]
            if len(output_values) == 0:
                output_values = pickle.loads(pickle.dumps(contour['values'], -1))
            else:
                output_values = np.append(output_values, contour['values'][:,1:], axis=1)
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

    def setMultiRunStartEndYears(self):
        for simID in self.SimulationVariables.keys():
            if self.SimulationVariables[simID]['StartTime'] > self.StartTime:
                self.StartTime = self.SimulationVariables[simID]['StartTime']
            if self.SimulationVariables[simID]['EndTime'] < self.EndTime:
                self.EndTime = self.SimulationVariables[simID]['EndTime']
        print('Start and End time set to {0} - {1}'.format(self.StartTime, self.EndTime))

    def getPandasTimeFreq(self, intervalstring):
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

    def getDefaultLineSettings(self, LineSettings, param, i):
        '''
        gets line settings and adds missing needed settings with defaults. Then translates java style inputs to
        python commands. Gets colors and styles.
        :param LineSettings: dictionary object containing settings and flags for lines/points
        :param param: parameter of data in order to grab default
        :param i: number of line on the plot in order to get the right sequential color
        :return:
            LineSettings: dictionary containing keys describing how the line/points are drawn
        '''

        LineSettings = self.getDrawFlags(LineSettings)
        if LineSettings['drawline'] == 'true':
            if param in self.defaultLineStyles.keys():
                if i >= len(self.defaultLineStyles[param]['lines']):
                    i = i - len(self.defaultLineStyles[param]['lines'])
                default_lines = self.defaultLineStyles[param]['lines'][i]
                for key in default_lines.keys():
                    if key not in LineSettings.keys():
                        LineSettings[key] = default_lines[key]

            default_default_lines = self.getDefaultDefaultLineStyles(i)
            for key in default_default_lines.keys():
                if key not in LineSettings.keys():
                    LineSettings[key] = default_default_lines[key]

            LineSettings['linecolor'] = self.confirmColor(LineSettings['linecolor'], default_default_lines['linecolor'])
            LineSettings = self.translateLineStylePatterns(LineSettings)

        if LineSettings['drawpoints'] == 'true':
            if param in self.defaultLineStyles.keys():
                if i >= len(self.defaultLineStyles[param]['lines']):
                    i = i - len(self.defaultLineStyles[param]['lines'])
                default_lines = self.defaultLineStyles[param]['lines'][i]
                for key in default_lines.keys():
                    if key not in LineSettings.keys():
                        LineSettings[key] = default_lines[key]

            default_default_points = self.getDefaultDefaultPointStyles(i)

            for key in default_default_points.keys():
                if key not in LineSettings.keys():
                    LineSettings[key] = default_default_points[key]

            LineSettings['pointfillcolor'] = self.confirmColor(LineSettings['pointfillcolor'], default_default_points['pointfillcolor'])
            LineSettings['pointlinecolor'] = self.confirmColor(LineSettings['pointlinecolor'], default_default_points['pointlinecolor'])

            try:
                if int(LineSettings['numptsskip']) == 0:
                    LineSettings['numptsskip'] = 1
            except ValueError:
                print('Invalid setting for numptsskip.', LineSettings['numptsskip'])
                print('defaulting to 25')
                LineSettings['numptsskip'] = 25

            LineSettings = self.translatePointStylePatterns(LineSettings)

        return LineSettings

    def getDefaultGateLineSettings(self, GateLineSettings, i):
        '''
        gets line settings and adds missing needed settings with defaults. Then translates java style inputs to
        python commands. Gets colors and styles.
        :param LineSettings: dictionary object containing settings and flags for lines/points
        :param param: parameter of data in order to grab default
        :param i: number of line on the plot in order to get the right sequential color
        :return:
            LineSettings: dictionary containing keys describing how the line/points are drawn
        '''

        GateLineSettings = self.getDrawFlags(GateLineSettings)
        if GateLineSettings['drawline'] == 'true':
            default_default_lines = self.getDefaultDefaultLineStyles(i)
            for key in default_default_lines.keys():
                if key not in GateLineSettings.keys():
                    GateLineSettings[key] = default_default_lines[key]

            GateLineSettings = self.translateLineStylePatterns(GateLineSettings)
            GateLineSettings['linecolor'] = self.confirmColor(GateLineSettings['linecolor'], default_default_lines['linecolor'])


        if GateLineSettings['drawpoints'] == 'true':
            default_default_points = self.getDefaultDefaultPointStyles(i)
            for key in default_default_points.keys():
                if key not in GateLineSettings.keys():
                    GateLineSettings[key] = default_default_points[key]
            try:
                if int(GateLineSettings['numptsskip']) == 0:
                    GateLineSettings['numptsskip'] = 1
            except ValueError:
                print('Invalid setting for numptsskip.', GateLineSettings['numptsskip'])
                print('defaulting to 25')
                GateLineSettings['numptsskip'] = 25

            GateLineSettings['pointlinecolor'] = self.confirmColor(GateLineSettings['pointlinecolor'], default_default_lines['pointlinecolor'])
            GateLineSettings['pointfillcolor'] = self.confirmColor(GateLineSettings['pointfillcolor'], default_default_lines['pointfillcolor'])
            GateLineSettings = self.translatePointStylePatterns(GateLineSettings)

        return GateLineSettings

    def getDefaultStraightLineSettings(self, LineSettings):
        '''
        gets line settings and adds missing needed settings with defaults. Then translates java style inputs to
        python commands. Gets colors and styles.
        :param LineSettings: dictionary object containing settings and flags for lines/points
        :param param: parameter of data in order to grab default
        :return:
            LineSettings: dictionary containing keys describing how the line/points are drawn
        '''

        default_default_lines = self.getDefaultDefaultLineStyles(0)
        default_default_lines['linecolor'] = 'black' #don't need different colors by default..
        for key in default_default_lines.keys():
            if key not in LineSettings.keys():
                LineSettings[key] = default_default_lines[key]

        LineSettings = self.translateLineStylePatterns(LineSettings)
        LineSettings['linecolor'] = self.confirmColor(LineSettings['linecolor'], default_default_lines['linecolor'])

        return LineSettings

    def getDefaultTextSettings(self, TextSettings):
        '''
        gets text settings and adds missing needed settings with defaults. Then translates java style inputs to
        python commands. Gets colors and styles.
        :param TextSettings: dictionary object containing settings and flags for text
        :return:
            LineSettings: dictionary containing keys describing how the line/points are drawn
        '''

        default_default_text = self.getDefaultDefaultTextStyles()
        for key in default_default_text.keys():
            if key not in TextSettings.keys():
                TextSettings[key] = default_default_text[key]

        TextSettings['fontcolor'] = self.confirmColor(TextSettings['fontcolor'], default_default_text['fontcolor'])

        return TextSettings

    def getDefaultContourLineSettings(self, contour_settings):
        default_contour_settings = {'linecolor': 'grey',
                                    'linewidth':1,
                                    'linestylepattern':'solid',
                                    'alpha': 1,
                                    'contourlinetext': 'false',
                                    'fontsize': 10,
                                    'text_inline': 'true',
                                    'inline_spacing': 10,
                                    'legend': 'false'}

        for key in default_contour_settings.keys():
            if key not in contour_settings:
                contour_settings[key] = default_contour_settings[key]
                if key == 'text_inline':
                    if contour_settings[key].lower() == 'true':
                        contour_settings[key] = True
                    else:
                        contour_settings[key] = False

        contour_settings = self.translateLineStylePatterns(contour_settings)

        return contour_settings

    def getDrawFlags(self, LineSettings):
        '''
        reads line settings dictionary to look for defined settings of lines or points to determine if either or both
        should be drawn. If nothing is explicitly stated, then draw lines with default settings.
        :param LineSettings: dictionary object containing settings and flags for lines/points
        :return:
            LineSettings: dictionary containing keys describing how the line/points are drawn
        '''

        #unless explicitly stated, look for key identifiers to draw lines or not
        LineVars = ['linecolor', 'linestylepattern', 'linewidth']
        PointVars = ['pointfillcolor', 'pointlinecolor', 'symboltype', 'symbolsize', 'numptsskip', 'markersize']

        if 'drawline' not in LineSettings.keys():
            for var in LineVars:
                if var in LineSettings.keys():
                    LineSettings['drawline'] = 'true'
                    break
            if 'drawline' not in LineSettings.keys():
                LineSettings['drawline'] = 'false'

        if 'drawpoints' not in LineSettings.keys():
            for var in PointVars:
                if var in LineSettings.keys():
                    LineSettings['drawpoints'] = 'true'
                    break
            if 'drawpoints' not in LineSettings.keys():
                LineSettings['drawpoints'] = 'false'

        if LineSettings['drawpoints'] == 'false' and LineSettings['drawline'] == 'false':
            LineSettings['drawline'] = 'true' #gotta do something..

        return LineSettings

    def getDefaultContourSettings(self, object_settings):
        defaultColormap = mpl.cm.get_cmap('jet')
        default_colorbar_settings = {'colormap': defaultColormap,
                                     'bins':10,
                                     'skipticks':1}

        if 'colorbar' in object_settings.keys():
            if 'colormap' in object_settings['colorbar'].keys():
                try:
                    usercolormap = mpl.cm.get_cmap(object_settings['colorbar']['colormap'])
                    object_settings['colormap'] = usercolormap
                except ValueError:
                    print('User selected invalid colormap:', object_settings['colorbar']['colormap'])
                    print('Tip: make sure capitalization is correct!')
                    print('Defaulting to Jet.')
                    object_settings['colormap'] = defaultColormap
        else:
            object_settings['colorbar'] = {}

        for key in default_colorbar_settings.keys():
            if key not in object_settings['colorbar']:
                object_settings['colorbar'][key] = default_colorbar_settings[key]

        return object_settings

    def getDefaultDefaultLineStyles(self, i):
        '''
        creates a default line style based off of the number line and default colors
        used if param is undefined or not in defaults file
        :param i: count of line on the plot
        :return: dictionary with line settings
        '''

        if i >= len(self.def_colors):
            i = i - len(self.def_colors)
        return {'linewidth': 2, 'linecolor': self.def_colors[i], 'linestylepattern': 'solid', 'alpha': 1.0}

    def getDefaultDefaultTextStyles(self):
        '''
        creates a default line style based off of the number line and default colors
        used if param is undefined or not in defaults file
        :return: dictionary with line settings
        '''

        return {'fontsize': 9, 'fontcolor': 'black', 'alpha': 1.0, 'horizontalalignment': 'left'}

    def getDefaultDefaultPointStyles(self, i):
        '''
        creates a default point style based off of the number points and default colors
        used if param is undefined or not in defaults file
        :param i: count of points on the plot
        :return: dictionary with point settings
        '''

        if i >= len(self.def_colors):
            i = i - len(self.def_colors)
        return {'pointfillcolor': self.def_colors[i], 'pointlinecolor': self.def_colors[i], 'symboltype': 1,
                'symbolsize': 5, 'numptsskip': 0, 'alpha': 1.0}

    def getLineSettings(self, LineSettings, Flag):
        '''
        gets the correct line settings for the selected flag
        :param LineSettings: dictionary of settings
        :param Flag: selected flag to match line
        :return: deep copy of line
        '''

        for line in LineSettings:
            if Flag == line['flag']:
                return pickle.loads(pickle.dumps(line, -1))

    def getPlotLabelMasks(self, idx, nprofiles, cols):
        '''
        Get plot label masks
        :param idx: page index
        :param nprofiles: number of profiles
        :param cols: number of columns
        :return: boolean fields for plotting
        '''

        if idx >= nprofiles - cols:
            add_xlabel = True
        else:
            add_xlabel = False
        if idx % cols == 0:
            add_ylabel = True
        else:
            add_ylabel = False

        return add_xlabel, add_ylabel

    def getTimeSeries(self, Line_info):
        '''
        gets time series data from defined sources
        :param Line_info: dictionary of line setttings containing datasources
        :return: dates, values, units
        '''

        if 'dss_path' in Line_info.keys(): #Get data from DSS record
            if 'dss_filename' not in Line_info.keys():
                print('DSS_Filename not set for Line: {0}'.format(Line_info))
                return np.array([]), np.array([]), None
            else:
                datamem_key = self.buildDataMemoryKey(Line_info)
                if datamem_key in self.Data_Memory.keys():
                    print('Reading {0} from memory'.format(datamem_key))
                    datamementry = pickle.loads(pickle.dumps(self.Data_Memory[datamem_key], -1))
                    times = datamementry['times']
                    values = datamementry['values']
                    units = datamementry['units']
                else:
                    times, values, units = WDR.readDSSData(Line_info['dss_filename'], Line_info['dss_path'],
                                                           self.StartTime, self.EndTime)

                    self.Data_Memory[datamem_key] = {'times': pickle.loads(pickle.dumps(times, -1)),
                                                     'values': pickle.loads(pickle.dumps(values, -1)),
                                                     'units': pickle.loads(pickle.dumps(units, -1))}

                if np.any(values == None):
                    return np.array([]), np.array([]), None
                elif len(values) == 0:
                    return np.array([]), np.array([]), None

        elif 'w2_file' in Line_info.keys():
            if self.plugin.lower() != 'cequalw2':
                return np.array([]), np.array([]), None
            datamem_key = self.buildDataMemoryKey(Line_info)
            if datamem_key in self.Data_Memory.keys():
                print('READING {0} FROM MEMORY'.format(datamem_key))
                datamementry = pickle.loads(pickle.dumps(self.Data_Memory[datamem_key], -1))
                times = datamementry['times']
                values = datamementry['values']
                units = datamementry['units']


            else:
                if 'structurenumbers' in Line_info.keys():
                    # Ryan Miles: yeah looks like it's str_brX.npt, and X is 1-# of branches (which is defined in the control file)
                    times, values = self.ModelAlt.readStructuredTimeSeries(Line_info['w2_file'], Line_info['structurenumbers'])
                else:
                    times, values = self.ModelAlt.readTimeSeries(Line_info['w2_file'], **Line_info)
                if 'units' in Line_info.keys():
                    units = Line_info['units']
                else:
                    units = None

                self.Data_Memory[datamem_key] = {'times': pickle.loads(pickle.dumps(times, -1)),
                                                 'values': pickle.loads(pickle.dumps(values, -1)),
                                                 'units': pickle.loads(pickle.dumps(units, -1))}

        elif 'easting' in Line_info.keys() and 'northing' in Line_info.keys():
            datamem_key = self.buildDataMemoryKey(Line_info)
            if datamem_key in self.Data_Memory.keys():
                print('READING {0} FROM MEMORY'.format(datamem_key))
                datamementry = pickle.loads(pickle.dumps(self.Data_Memory[datamem_key], -1))
                times = datamementry['times']
                values = datamementry['values']
                units = datamementry['units']

            else:
                times, values = self.ModelAlt.readTimeSeries(Line_info['parameter'],
                                                             float(Line_info['easting']),
                                                             float(Line_info['northing']))
                if 'units' in Line_info.keys():
                    units = Line_info['units']
                else:
                    units = None

                self.Data_Memory[datamem_key] = {'times': pickle.loads(pickle.dumps(times, -1)),
                                                 'values': pickle.loads(pickle.dumps(values, -1)),
                                                 'units': pickle.loads(pickle.dumps(units, -1))}

        else:
            print('No Data Defined for line')
            return np.array([]), np.array([]), None

        if 'omitvalue' in Line_info.keys():
            omitval = float(Line_info['omitvalue'])
            values = self.replaceOmittedValues(values, omitval)

        if 'interval' in Line_info.keys():
            times, values = self.changeTimeSeriesInterval(times, values, Line_info)

        return times, values, units

    def getTimeIntervalSeconds(self, interval):
        '''
        converts a given time interval into seconds
        :param interval: DSS interval (ex: 6MIN)
        :return: time interval in seconds
        '''

        interval = interval.lower()
        if 'min' in interval:
            timeint = int(interval.replace('min','')) * 60 #convert to sec
            return timeint
        elif 'hour' in interval:
            timeint = int(interval.replace('hour','')) * 3600 #convert to sec
            return timeint
        elif 'day' in interval:
            timeint = int(interval.replace('day','')) * 86400 #convert to sec
            return timeint
        elif 'mon' in interval:
            timeint = int(interval.replace('mon','')) * 2.628e+6 #convert to sec
            return timeint
        elif 'year' in interval:
            timeint = int(interval.replace('year','')) * 3.154e+7 #convert to sec
            return timeint
        else:
            print('Unidentified time interval')
            return 0

    def getTimeInterval(self, times):
        '''
        attempts to find out the time interval of the time series by finding the most common time interval
        :param times: list of times
        :return:
        '''

        t_ints = []
        for i, t in enumerate(times):
            if i == 0: #skip 1
                last_time = t
            else:
                t_ints.append(t - last_time)

        return self.getMostCommon(t_ints)

    def getMostCommon(self, listvars):
        '''
        gets most common instance of a var in a list
        :param listvars: list of variables
        :return: value that is most common in the list
        '''

        occurence_count = Counter(listvars)
        most_common_interval = occurence_count.most_common(1)[0][0]
        return most_common_interval

    def getProfileValues(self, Line_info, timesteps):
        '''
        reads in profile data from various sources for profile plots at given timesteps
        attempts to get elevations if possible
        :param Line_info: dictionary containing settings for line
        :param timesteps: given list of timesteps to extract data at
        :return: values, elevations, depths, flag
        '''

        datamemkey = self.buildDataMemoryKey(Line_info)

        if datamemkey in self.Data_Memory.keys():
            dm = pickle.loads(pickle.dumps(self.Data_Memory[datamemkey], -1))
            print('retrieving profile from datamem')
            return dm['values'], dm['elevations'], dm['depths'], dm['times'], Line_info['flag']

        elif 'filename' in Line_info.keys(): #Get data from Observed
            filename = Line_info['filename']
            values, depths, times = WDR.readTextProfile(filename, timesteps)
            return values, [], depths, times, Line_info['flag']

        elif 'w2_segment' in Line_info.keys():
            if self.plugin.lower() != 'cequalw2':
                return [], [], [], [], None
            vals, elevations, depths, times = self.ModelAlt.readProfileData(Line_info['w2_segment'], timesteps)
            return vals, elevations, depths, times, Line_info['flag']

        elif 'ressimresname' in Line_info.keys():
            if self.plugin.lower() != 'ressim':
                return [], [], [], [], None
            vals, elevations, depths, times = self.ModelAlt.readProfileData(Line_info['ressimresname'],
                                                                            Line_info['parameter'], timesteps)
            return vals, elevations, depths, times, Line_info['flag']

        print('No Data Defined for line')
        print('Line:', Line_info)
        return [], [], [], [], None

    def getProfileData(self, object_settings):
        '''
        Gets profile line data from defined data sources in XML files
        :param object_settings: currently selected object settings dictionary
        :param keyval: determines what key to iterate over for data
        :return: dictionary containing data and information about each data set
        '''

        data = {}
        line_settings = {}
        timestamps = object_settings['timestamps']
        for line in object_settings[object_settings['datakey']]:
            numtimesused = 0
            if 'flag' not in line.keys():
                print('Flag not set for line (Computed/Observed/etc)')
                print('Not plotting Line:', line)
                continue
            elif line['flag'].lower() == 'computed':
                # for ID in self.SimulationVariables.keys():
                for ID in self.accepted_IDs:
                    curline = pickle.loads(pickle.dumps(line, -1))
                    curline = self.configureSettingsForID(ID, curline)
                    curline['numtimesused'] = numtimesused
                    curline['ID'] = ID
                    if not self.checkModelType(curline):
                        continue
                    numtimesused += 1
                    data, line_settings = self.recordProfileData(data, line_settings, curline, timestamps)
            else:
                if self.currentlyloadedID != 'base':
                    line = self.configureSettingsForID('base', line)
                else:
                    line = self.replaceflaggedValues(line, 'modelspecific')
                line['numtimesused'] = 0
                if not self.checkModelType(line):
                    continue
                data, line_settings = self.recordProfileData(data, line_settings, line, timestamps)

        return data, line_settings

    def getTimeseriesData(self, object_settings):
        '''
        Gets profile line data from defined data sources in XML files
        :param object_settings: currently selected object settings dictionary
        :param keyval: determines what key to iterate over for data
        :return: dictionary containing data and information about each data set
        '''

        data = {}
        if 'lines' in object_settings.keys():
            for line in object_settings['lines']:
                numtimesused = 0
                if 'flag' not in line.keys():
                    print('Flag not set for line (Computed/Observed/etc)')
                    print('Not plotting Line:', line)
                    continue
                elif line['flag'].lower() == 'computed':
                    for ID in self.accepted_IDs:
                        curline = pickle.loads(pickle.dumps(line, -1))
                        curline = self.configureSettingsForID(ID, curline)
                        if not self.checkModelType(curline):
                            continue
                        curline['numtimesused'] = numtimesused
                        numtimesused += 1
                        data = self.recordTimeSeriesData(data, curline)
                else:
                    if self.currentlyloadedID != 'base':
                        line = self.configureSettingsForID('base', line)
                    else:
                        line = self.replaceflaggedValues(line, 'modelspecific')
                    line['numtimesused'] = 0
                    if not self.checkModelType(line):
                        continue
                    data = self.recordTimeSeriesData(data, line)
        return data

    def getGateData(self, object_settings):
        '''
        Gets profile line data from defined data sources in XML files
        :param object_settings: currently selected object settings dictionary
        :param keyval: determines what key to iterate over for data
        :return: dictionary containing data and information about each data set
        '''

        data = {}
        if 'gateops' in object_settings.keys():
            for gi, gateop in enumerate(object_settings['gateops']):

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
                    dates, values, _ = self.getTimeSeries(gate)
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
                    datamem_key = self.buildDataMemoryKey(gate)
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

    def getProfileDates(self, Line_info):
        '''
        gets dates from observed text profiles
        :param Line_info: dictionary containing line information, must include filename
        :return: list of times
        '''

        if 'filename' in Line_info.keys(): #Get data from Observed
            times = WDR.getTextProfileDates(Line_info['filename'], self.StartTime, self.EndTime) #TODO: set up for not observed data??
            return times

        print('Illegal Dates selection. ')
        return []

    def getPlotUnits(self, unitslist, object_settings):
        '''
        gets units for the plot. Either looks at data already plotted units, or if there are no defined units
        in the plotted data, look for a parameter flag
        :param object_settings: dictionary with plot settings
        :return: string units value
        '''

        if 'parameter' in object_settings.keys():
            try:
                plotunits = self.units[object_settings['parameter'].lower()]
                if isinstance(plotunits, dict):
                    if 'unitsystem' in object_settings.keys():
                        plotunits = plotunits[object_settings['unitsystem'].lower()]
                    else:
                        plotunits = plotunits['metric']
            except KeyError:
                plotunits = ''

        elif len(unitslist) > 0:
            plotunits = self.getMostCommon(unitslist)

        else:
            print('No units defined.')
            plotunits = ''

        plotunits = self.translateUnits(plotunits)
        return plotunits

    def getTableDates(self, year, object_settings, month='None'):
        '''
        gets start and end dates from lines in tables for logging
        :param year: selected year int or 'all' string
        :param object_settings: dictionary of item setting
        :param month: selected month (for monthly table) or None
        :return: start and end date
        '''

        xmin = 'NONE'
        xmax = 'NONE'
        if 'xlims' in object_settings.keys():
            if 'min' in object_settings['xlims'].keys():
                xmin = self.translateDateFormat(object_settings['xlims']['min'], 'datetime', self.StartTime)
                xmin = xmin.strftime('%d %b %Y')
            if 'max' in object_settings['xlims'].keys():
                xmax = self.translateDateFormat(object_settings['xlims']['max'], 'datetime', self.EndTime)
                xmax = xmax.strftime('%d %b %Y')

        if xmin != 'NONE':
            start_date = xmin
        elif year == self.startYear:
            start_date = self.StartTime.strftime('%d %b %Y')
        else:
            if str(year).lower() == 'all':
                start_date = '01 Jan {0}'.format(self.startYear)
            else:
                start_date = '01 Jan {0}'.format(year)

        if xmax != 'NONE':
            start_date = xmax
        elif year == self.endYear:
            end_date = self.EndTime.strftime('%d %b %Y')
        else:
            if str(year).lower() == 'all':
                end_date = '31 Dec {0}'.format(self.endYear)
            else:
                end_date = '31 Dec {0}'.format(year)
            if month != 'None':
                try:
                    month = int(month)
                except ValueError:
                    month = self.month2num

                try:
                    start_date = dt.datetime.strptime(start_date, '%d %b %Y').replace(month=month).strftime('%d %b %Y')
                except ValueError:
                    start_date = dt.datetime.strptime(start_date, '%d %b %Y')
                    start_date = start_date.replace(day=1)
                    start_date = start_date.replace(month=month+1)
                    start_date -= dt.timedelta(days=1)
                    start_date = start_date.strftime('%d %b %Y')
                try:
                    end_date = dt.datetime.strptime(end_date, '%d %b %Y').replace(month=month).strftime('%d %b %Y')
                except ValueError:
                    end_date = dt.datetime.strptime(end_date, '%d %b %Y')
                    end_date = end_date.replace(day=1)
                    end_date = end_date.replace(month=month+1)
                    end_date -= dt.timedelta(days=1)
                    end_date = end_date.strftime('%d %b %Y')

        return start_date, end_date

    def getTableData(self, object_settings):
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
                for ID in self.accepted_IDs:
                    cur_dp = pickle.loads(pickle.dumps(dp, -1))
                    cur_dp = self.configureSettingsForID(ID, cur_dp)
                    cur_dp['numtimesused'] = numtimesused
                    cur_dp['ID'] = ID
                    if not self.checkModelType(cur_dp):
                        continue
                    numtimesused += 1
                    data = self.recordTimeSeriesData(data, cur_dp)
            else:
                if self.currentlyloadedID != 'base':
                    dp = self.configureSettingsForID('base', dp)
                else:
                    dp = self.replaceflaggedValues(dp, 'modelspecific')
                dp['numtimesused'] = 0
                if not self.checkModelType(dp):
                    continue
                data = self.recordTimeSeriesData(data, dp)

        return data

    def getListItems(self, listvals):
        '''
        recursive function to convert lists of lists into single lists for logging
        :param listvals: value object
        :return: list of values
        '''

        if isinstance(listvals, (list, np.ndarray)):
            outvalues = []
            for item in listvals:
                if isinstance(item, (list, np.ndarray)):
                    vals = self.getListItems(item)
                    for v in vals:
                        outvalues.append(v)
                else:
                    return listvals #we just have a list of values, so we're good! return list
        elif isinstance(listvals, dict):
            outvalues = self.getListItemsFromDict(listvals)
        return outvalues

    def getListItemsFromDict(self, indict):
        '''
        recursive function to convert dictionary of lists into single dictionary for logging. Keys are determined
        using original keys
        :param indict: value dictionary object
        :return: dictionary of values
        '''

        outdict = {}
        for key in indict:
            if isinstance(indict[key], dict):
                returndict = self.getListItemsFromDict(indict[key])
                returndict = {'{0}_{1}'.format(key, newkey): returndict[newkey] for newkey in returndict}
                for key in returndict.keys():
                    outdict[key] = returndict[key]
            elif isinstance(indict[key], (list, np.ndarray)):
                outdict[key] = indict[key]
        return outdict

    def getContourData(self, object_settings):
        '''
        Gets Contour line data from defined data sources in XML files
        :param object_settings: currently selected object settings dictionary
        :param keyval: determines what key to iterate over for data
        :return: dictionary containing data and information about each data set
        '''

        data = {}
        for reach in object_settings['reaches']:
            for ID in self.accepted_IDs:
                curreach = pickle.loads(pickle.dumps(reach, -1))
                curreach = self.configureSettingsForID(ID, curreach)
                if not self.checkModelType(curreach):
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
                    datamem_key = self.buildDataMemoryKey(reach)

                    if 'units' in reach.keys() and units == None:
                        units = reach['units']

                    if 'y_scalar' in object_settings.keys():
                        y_scalar = float(object_settings['y_scalar'])
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
        self.loadCurrentID('base')
        self.loadCurrentModelAltID('base')
        return data

    def getContours(self, object_settings):

        if 'ressimresname' in object_settings.keys(): #Ressim subdomain
            datamem_key = self.buildDataMemoryKey(object_settings)
            if datamem_key in self.Data_Memory.keys():
                print('READING {0} FROM MEMORY'.format(datamem_key))
                datamem_entry = pickle.loads(pickle.dumps(self.Data_Memory[datamem_key], -1))
                times = datamem_entry['dates']
                values = datamem_entry['values']
                units = datamem_entry['units']
                distance = datamem_entry['distance']

            else:
                checkdomain = self.ModelAlt.checkSubdomain(object_settings['ressimresname'])
                if not checkdomain:
                    return [], [], [], []
                times, values, distance = self.ModelAlt.readSubdomain(object_settings['parameter'],
                                                                      object_settings['ressimresname'])

                if 'units' in object_settings.keys():
                    units = object_settings['units']
                else:
                    units = None

                self.Data_Memory[datamem_key] = {'dates': pickle.loads(pickle.dumps(times, -1)),
                                                 'values': pickle.loads(pickle.dumps(values, -1)),
                                                 'units': pickle.loads(pickle.dumps(units, -1)),
                                                 'distance': pickle.loads(pickle.dumps(distance, -1)),
                                                 'iscontour': True}

        elif 'w2_file' in object_settings.keys():
            datamem_key = self.buildDataMemoryKey(object_settings)
            if datamem_key in self.Data_Memory.keys():
                print('READING {0} FROM MEMORY'.format(datamem_key))
                datamementry = pickle.loads(pickle.dumps(self.Data_Memory[datamem_key], -1))
                times = datamementry['dates']
                values = datamementry['values']
                units = datamementry['units']
                distance = datamementry['distance']
            else:
                times, values, distance = self.ModelAlt.readSegment(object_settings['w2_file'],
                                                                    object_settings['parameter'])

        if 'interval' in object_settings.keys():
            times, values = self.changeTimeSeriesInterval(times, values, object_settings)

        return times, values, units, distance

    def getDateSourceFlag(self, object_settings):
        '''
        Gets the datesource from object settings
        :param object_settings: currently selected object settings dictionary
        :return: datessource_flag
        '''

        if 'datessource' in object_settings.keys():
            datessource_flag = object_settings['datessource'] #determine how you want to get dates? either flag or list
        else:
            datessource_flag = [] #let it make timesteps

        return datessource_flag

    def getProfileTimestamps(self, object_settings):
        '''
        Gets timestamps based off of user settings in XML file and reads/builds them.
        :param object_settings: currently selected object settings dictionary
        :return: list of timestamp values to be plotted
        '''

        if isinstance(object_settings['datessource_flag'], str):
            for line in object_settings[object_settings['datakey']]:
                if line['flag'] == object_settings['datessource_flag']:
                    timestamps = self.getProfileDates(line)
        elif isinstance(object_settings['datessource_flag'], dict): #single date instance..
            timestamps = object_settings['datessource']['date']
        elif isinstance(object_settings['datessource_flag'], list):
            timestamps = []
            for d in object_settings['datessource']:
                dfrmt = self.translateDateFormat(d, 'datetime', None)
                if dfrmt != None:
                    timestamps.append(dfrmt)
                else:
                    print('Invalid Timestamp', d)

        if len(timestamps) == 0:
            #if something fails, or not implemented, or theres just no dates in the window, make some up
            print('No TimeSteps found.. Making regular timesteps..')
            timestamps = self.makeRegularTimesteps(days=15)

        return np.asarray(timestamps)

    def getPlotParameter(self, object_settings):
        '''
        Gets the plot parameter based on user settings. If explicitly stated, uses that. Otherwise, looks at the
        defined parameters in the linedata and grabs the most common one.
        :param object_settings: currently selected object settings dictionary
        :return: plot parameter if possible, otherwise None
        '''

        if 'parameter' in object_settings.keys():
            plot_parameter = object_settings['parameter']
        else:
            params = []
            for line in object_settings['lines']:
                if 'parameter' in line.keys():
                    params.append(line['parameter'])
            if len(list(set(params))) == 1:
                plot_parameter = params[0]
            else:
                plot_parameter = None
        return plot_parameter

    def getProfileInterpResolution(self, object_settings, default=30):
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
            print('Resolution not defined. Setting to default values.')
            resolution = default
        return int(resolution)

    def getPlotYears(self, object_settings):
        '''
        formats years settings for plots/tables. Figures out years used, if split by year, and year strings
        years is set to "ALLYEARS" if not split by year. This tells other parts of script to include all.
        :param object_settings: currently selected object settings dictionary
        :return:
            split_by_year: boolean if plots/tables should be split up year to year or all at once
            years: list of years that are used
            yearstr: list of years as strings, or set of years (ex: 2013-2016)
        '''

        split_by_year = False
        yearstr = ''
        if 'splitbyyear' in object_settings.keys():
            if object_settings['splitbyyear'].lower() == 'true':
                split_by_year = True
                years = self.years
                yearstr = [str(year) for year in years]
        if not split_by_year:
            yearstr = self.years_str
            years = ['ALLYEARS']

        return split_by_year, years, yearstr

    def getParameterCount(self, line, object_settings):
        '''
        Returns parameter used in dataset and keeps a running total of used parameters in dataset
        :param line: current line in linedata dataset
        :param object_settings: currently selected object settings dictionary
        :return:
            param: current parameter if available else None
            param_count: running count of used parameters
        '''

        if 'param_count' not in object_settings.keys():
            param_count = {}
        else:
            param_count = object_settings['param_count']

        if 'parameter' in line.keys():
            param = line['parameter'].lower()
        else:
            param = None
        if param not in param_count.keys():
            param_count[param] = 0
        else:
            param_count[param] += 1

        return param, param_count

    def getUnitsList(self, line_settings):
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

    def getUsedIDs(self, contoursbyID):
        IDs = []
        for key in contoursbyID.keys():
            ID = contoursbyID[key]['ID']
            if ID not in IDs:
                IDs.append(ID)
        return IDs

    def getRowHeader(self, row_val, line_settings):
        header = []
        rv_split = row_val.replace('%', '').split('.')[1:]
        for rv in rv_split:
            if rv in line_settings.keys():
                if 'label' in line_settings[rv]:
                    header.append(line_settings[rv]['label'])
        header = ', '.join(header)
        return header

    def getGateConfigurationDays(self, gateconfig, gatedata, timestamp):
        time_interval = None
        operation_idx = []
        for gateop in gatedata.keys():
            for gi, gate in enumerate(gatedata[gateop]['gates'].keys()):
                curgate = gatedata[gateop]['gates'][gate]
                datamsk = np.where(curgate['dates'] == timestamp)
                if len(datamsk) == 0:
                    continue
                else:
                    datamsk = datamsk[0][0] + 1
                if time_interval == None:
                    time_interval = curgate['dates'][1] - curgate['dates'][0]
                if np.isnan(gateconfig[gateop][gate]):
                    correct_ops_idx = np.where(curgate['values'][:datamsk] != 1)[0].tolist()
                else:
                    correct_ops_idx = np.where(curgate['values'][:datamsk] == 1)[0].tolist()

                operation_idx.append(correct_ops_idx)

        correct_config = len(set.intersection(*map(set,operation_idx)))
        return round((time_interval * correct_config).days + ((time_interval * correct_config).seconds / 86400), 3)

    def getGateBlendDays(self, gateconfig, gatedata, timestamp):
        time_interval = None
        gateconfig_activegateop = {}
        alldays_activeop = {}
        for gateop in gateconfig.keys():
            gateconfig_activegateop[gateop] = False
            for gate in gateconfig[gateop].keys():
                if not np.isnan(gateconfig[gateop][gate]):
                    gateconfig_activegateop[gateop] = True
                    break
            for gateop in gatedata.keys():
                for gate in gatedata[gateop]['gates'].keys():
                    curgate = gatedata[gateop]['gates'][gate]
                    if curgate['gategroup'] == gateop:
                        datamsk = np.where(curgate['dates'] == timestamp)
                        if len(datamsk) == 0:
                            continue
                        else:
                            datamsk = datamsk[0][0]+1
                        if gateop not in alldays_activeop.keys():
                            alldays_activeop[gateop] = np.full(datamsk, False)
                        idx_active = np.where(~np.isnan(curgate['values'][:datamsk]))
                        alldays_activeop[gateop][idx_active] = True
                        if time_interval == None:
                            time_interval = curgate['dates'][1] - curgate['dates'][0]

        idx_count = 0
        for i in range(datamsk):
            valid = True #assume all is well
            for gateop in alldays_activeop.keys(): #for each gate on this timestamp
                if alldays_activeop[gateop][i] != gateconfig_activegateop[gateop]: #if the gate operation is not what it should be
                    valid = False #thenwe cannot count
            if valid:
                idx_count += 1

        return round((time_interval * idx_count).days + ((time_interval * idx_count).seconds / 86400), 3)

    def getRelativeMasterSet(self, linedata):
        #add all thje data together. then we cna use this when plotting it to get %
        #TODO: deal with irregular intervals
        intervals = {}
        biggest_interval = None
        type = 'INST-VAL'
        for line in linedata.keys():
            if 'interval' in linedata[line].keys():
                td = self.getTimeInterval(linedata[line]['dates'])
                if linedata[line]['interval'].upper() not in intervals.keys():
                    intervals[linedata[line]['interval'].upper()] = td
                if biggest_interval == None:
                    biggest_interval = linedata[line]['interval'].upper()
                    if 'type' in linedata[line].keys():
                        type = linedata[line]['type'].upper()
                else:
                    if td > intervals[biggest_interval]:
                        biggest_interval = linedata[line]['interval'].upper()
                        if linedata[line]['type'] in line.keys():
                            type = linedata[line]['type'].upper()

        RelativeLineSettings = {'interval': biggest_interval,
                                'type': type}
        RelativeMasterSet = []
        units = []
        for li, line in enumerate(linedata.keys()):
            curline = pickle.loads(pickle.dumps(linedata[line], -1))
            curline['values'], curline['units'] = self.convertUnitSystem(curline['values'], curline['units'], 'metric') #just make everything metric..
            units.append(curline['units'])
            if li == 0:
                if biggest_interval != None:
                    _, RelativeMasterSet = self.changeTimeSeriesInterval(curline['dates'], curline['values'], RelativeLineSettings)
                else:
                    RelativeMasterSet = curline['values']
            else:
                if biggest_interval != None:
                    curline['interval'] = biggest_interval
                    curline['type'] = type
                    _, newvals = self.changeTimeSeriesInterval(curline['dates'], curline['values'], RelativeLineSettings)
                    RelativeMasterSet += newvals
                else:
                    RelativeMasterSet += curline['values']

        RelativeLineSettings['units'] = self.getMostCommon(units)

        return RelativeMasterSet, RelativeLineSettings

    def getGateOperationTimes(self, gatedata):
        operationTimes = []
        for gateop in gatedata.keys():
            gateop_ops = np.array([])
            for gi, gate in enumerate(gatedata[gateop]['gates']):
                curgate = gatedata[gateop]['gates'][gate]
                gateop_ops = np.append(gateop_ops, np.where(~np.isnan(curgate['values'])))#TODO: make sure gate dates are same length?
            gateop_ops = list(set(gateop_ops.tolist()))
            for gateop_op in gateop_ops: #idx where data IS valid, i.e. A gate is in
                if gateop_op != 0 and gateop_op != len(curgate['values'])-1:
                    if gateop_op + 1 not in gateop_ops or gateop_op - 1 not in gateop_ops:#gate was in, then taken out
                        operationTimes.append(curgate['dates'][int(gateop_op)])

        return operationTimes


    def replaceDefaults(self, default_settings, object_settings):
        '''
        makes deep copies of default and defined settings so no settings are accidentally carried over
        replaces flagged values (%%) with easily identified variables
        iterates through settings and replaces all default settings with defined settings
        :param default_settings: default object settings dictionary
        :param object_settings: user defined settings dictionary
        :return:
            default_settings: dictionary of user and default settings
        '''

        default_settings = pickle.loads(pickle.dumps(self.replaceflaggedValues(default_settings, 'general'), -1))
        object_settings = pickle.loads(pickle.dumps(self.replaceflaggedValues(object_settings, 'general'), -1))

        for key in object_settings.keys():
            if key not in default_settings.keys(): #if defaults doesnt have key
                default_settings[key] = object_settings[key]
            elif default_settings[key] == None: #if defaults has key, but is none
                default_settings[key] = object_settings[key]
            elif isinstance(object_settings[key], list): #if settings is a list, aka rows or lines
                # if key.lower() == 'rows': #if the default has rows defined, just overwrite them.
                if key in default_settings.keys():
                    default_settings[key] = object_settings[key]
                elif key.lower() not in default_settings.keys():
                    default_settings[key] = object_settings[key] #if the defaults dont have anything defined, fill it in
                # else:
                #     for item in object_settings[key]:
                #         if isinstance(item, dict):
                #             if 'flag' in item.keys(): #if we flag line
                #                 flag_match = False
                #                 for defaultitem in default_settings[key]:
                #                     if 'flag' in defaultitem.keys():
                #                         if defaultitem['flag'].lower() == item['flag'].lower(): #matching flags!
                #                             flag_match = True
                #                             for subkey in item.keys(): #for each settings defined, overwrite
                #                                 defaultitem[subkey] = item[subkey]
                #                 if not flag_match:
                #                     default_settings[key].append(item)
                #         if isinstance(item, str):
                #             default_settings[key] = object_settings[key] #replace string with list, ex datessource
                #             break
            else:
                default_settings[key] = object_settings[key]

        return default_settings

    def replaceflaggedValues(self, settings, itemset):
        '''
        recursive function to replace flagged values in settings
        :param settings: dict, list or string containing settings, potentially with flags
        :return:
            settings: dict, list or string with flags replaced
        '''

        if isinstance(settings, str):
            if '%%' in settings:
                newval = self.replaceFlaggedValue(settings, itemset)
                settings = newval
        elif isinstance(settings, dict):
            for key in settings.keys():
                if isinstance(settings[key], dict):
                    settings[key] = self.replaceflaggedValues(settings[key], itemset)
                elif isinstance(settings[key], list):
                    new_list = []
                    for item in settings[key]:
                        new_list.append(self.replaceflaggedValues(item, itemset))
                    settings[key] = new_list
                else:
                    try:
                        if '%%' in settings[key]:
                            newval = self.replaceFlaggedValue(settings[key], itemset)
                            settings[key] = newval
                    except TypeError:
                        continue
        elif isinstance(settings, list):
            for i, item in enumerate(settings):
                if '%%' in item:
                    settings[i] = self.replaceFlaggedValue(item, itemset)

        return settings

    def replaceFlaggedValue(self, value, itemset):
        '''
        replaces strings with flagged values with known paths
        flags are now case insensitive with more intelligent matching. yay.
        needs to use '[1:-1]' for paths, otherwise things like /t in a path C:/trains will be taken literal
        :param value: string potentially containing flagged value
        :return:
            value: string with potential flags replaced
        '''


        if itemset == 'general':
            flagged_values = {'%%region%%': self.ChapterRegion,
                              '%%observedDir%%': self.observedDir,
                              '%%startyear%%': str(self.startYear),
                              '%%endyear%%': str(self.endYear)
                              }
        elif itemset == 'modelspecific':
            flagged_values = {'%%ModelDSS%%': self.DSSFile,
                              '%%Fpart%%': self.alternativeFpart,
                              '%%plugin%%': self.plugin,
                              '%%modelAltName%%': self.modelAltName,
                              '%%SimulationName%%': self.SimulationName,
                              '%%SimulationDir%%': self.SimulationDir,
                              '%%baseSimulationName%%': self.baseSimulationName,
                              '%%starttime%%': self.StartTimeStr,
                              '%%endtime%%': self.EndTimeStr,
                              '%%LastComputed%%': self.LastComputed
                              }

        for fv in flagged_values.keys():
            pattern = re.compile(re.escape(fv), re.IGNORECASE)
            value = pattern.sub(repr(flagged_values[fv])[1:-1], value) #this seems weird with [1:-1] but paths wont work otherwise
        return value

    def replaceOmittedValues(self, values, omitval):
        '''
        replaces a specified value in time series. Can be variable depending on data source (-99999, 0, 100, etc)
        :param values: array of values
        :param omitval: value to be omitted
        :return: new values
        '''

        if isinstance(values, dict):
            new_values = {}
            for key in values:
                new_values[key] = self.replaceOmittedValues(values[key], omitval)
        else:
            o_msk = np.where(values == omitval)
            values[o_msk] = np.nan
            new_values = np.asarray(values)
            print('Omitted {0} values of {1}'.format(len(o_msk[0]), omitval))
        return new_values

    def translateLineStylePatterns(self, LineSettings):
        '''
        translates java line style patterns to python friendly commands.
        :param LineSettings: dictionary containing keys describing how the line/points are drawn
        :return:
            LineSettings: dictionary containing keys describing how the line/points are drawn
        '''

        #java|python
        linestylesdict = {'dash': 'dashed',
                          'dash dot': 'dashdot',
                          'dash dot-dot': (0, (3, 5, 1, 5, 1, 5)), #this one doesnt get a string name?
                          'dot': 'dotted',
                          'solid': 'solid'}

        if 'linestylepattern' in LineSettings.keys():
            if LineSettings['linestylepattern'].lower() in linestylesdict.values(): #existing python values
                LineSettings['linestylepattern'] = LineSettings['linestylepattern'].lower() #use python but lower it
            else:
                try:
                    LineSettings['linestylepattern'] = linestylesdict[LineSettings['linestylepattern'].lower()]
                except KeyError:
                    print('Invalid lineStylePattern:', LineSettings['linestylepattern'])
                    print('Defaulting to Solid.')
                    LineSettings['linestylepattern'] = 'solid'
        else:
            print('lineStylePattern undefined for line. Using solid')
            LineSettings['linestylepattern'] = 'solid'

        return LineSettings

    def translatePointStylePatterns(self, LineSettings):
        '''
        translates java point style patterns to python friendly commands.
        :param LineSettings: dictionary containing keys describing how the line/points are drawn
        :return:
            LineSettings: dictionary containing keys describing how the line/points are drawn
        '''

        #java|python
        #https://matplotlib.org/stable/api/markers_api.html#module-matplotlib.markers
        pointstylesdict = {1: 's', #square
                           2: 'o', #circle
                           3: '^', #triangle up
                           4: 'v', #triangle down
                           5: 'D', #diamond
                           6: '*' #star
                           }

        if 'symboltype' in LineSettings.keys():
            if LineSettings['symboltype'] in pointstylesdict.values(): #existing python values
                LineSettings['symboltype'] = LineSettings['symboltype'] #needs to be case sensitive..
            else:
                try:
                    LineSettings['symboltype'] = pointstylesdict[int(LineSettings['symboltype'])]
                except:
                    print('Invalid Symboltype:', LineSettings['symboltype'])
                    print('Defaulting to Square.')
                    LineSettings['symboltype'] = 's'

        else:
            print('Symbol not defined. Defaulting to Square.')
            LineSettings['symboltype'] = 's'

        return LineSettings

    def translateDateFormat(self, lim, dateformat, fallback):
        '''
        translates date formats between datetime and jdate, as desired
        :param lim: limit value, either int or datetime
        :param dateformat: desired date format, either 'datetime' or 'jdate'
        :param fallback: if setting translation fails, use backup, usually starttime or endtime
        :return:
            lim: original limit, if translate fails
            lim_fmrt: translated limit
        '''

        if dateformat.lower() == 'datetime': #if want datetime
            if isinstance(lim, dt.datetime):
                return lim
            else:
                try:
                    lim_frmt = pendulum.parse(lim, strict=False).replace(tzinfo=None)#try simple date formatting.
                    if not self.StartTime <= lim_frmt <= self.EndTime: #check for false negative
                        raise IndexError
                    return lim_frmt
                except IndexError:
                    print('Xlim of {0} not between start and endtime {1} - {2}'.format(lim_frmt, self.StartTime,
                                                                                       self.EndTime))
                except:
                    print('Error Reading Limit: {0} as a dt.datetime object.'.format(lim))
                    print('If this is wrong, try format: Apr 2014 1 12:00')

                print('Trying as Jdate..')
                try:
                    lim_frmt = float(lim)
                    lim_frmt = self.JDateToDatetime(lim_frmt)
                    print('JDate {0} as {1} Accepted!'.format(lim, lim_frmt))
                    return lim_frmt
                except:
                    print('Limit value of {0} also invalid as jdate.'.format(lim))

                if fallback != None and fallback != '':
                    print('Setting to fallback {0}.'.format(fallback))
                else:
                    print('Setting to fallback.')
                return fallback

        elif dateformat.lower() == 'jdate':
            try:
                return float(lim)
            except:
                print('Error Reading Limit: {0} as a jdate.'.format(lim))
                print('If this is wrong, try format: 180')
                print('Trying as Datetime..')
                if isinstance(lim, (dt.datetime, str)):
                    try:
                        if isinstance(lim, str):
                            lim_frmt = pendulum.parse(lim, strict=False).replace(tzinfo=None)
                            print('Datetime {0} as {1} Accepted!'.format(lim, lim_frmt))
                        else:
                            lim_frmt = lim
                        print('converting to jdate..')
                        lim_frmt = self.DatetimeToJDate(lim_frmt)
                        print('Converted to jdate!', lim_frmt)
                        return lim_frmt
                    except:
                        print('Error Reading Limit: {0} as a dt.datetime object.'.format(lim))
                        print('If this is wrong, try format: Apr 2014 1 12:00')

                    fallback = self.DatetimeToJDate(fallback)

                    if fallback != None and fallback != '':
                        print('Setting to fallback {0}.'.format(fallback))
                    else:
                        print('Setting to fallback.')
                    return fallback

    def translateUnits(self, units):
        '''
        translates possible units to better known flags for consistancy in the script and conversion purposes
        :param units: units string
        :return: units string
        '''

        units_conversion = {'f': ['f', 'faren', 'degf', 'fahrenheit', 'fahren', 'deg f'],
                            'c': ['c', 'cel', 'celsius', 'deg c', 'degc'],
                            'm3/s': ['m3/s', 'm3s', 'metercubedpersecond', 'cms'],
                            'cfs': ['cfs', 'cubicftpersecond', 'f3/s', 'f3s'],
                            'm': ['m', 'meters', 'mtrs'],
                            'ft': ['ft', 'feet'],
                            'm3': ['m3', 'meters cubed', 'meters3', 'meterscubed', 'meters-cubed'],
                            'af': ['af', 'acrefeet', 'acre-feet', 'acfeet', 'acft', 'ac-ft', 'ac/ft'],
                            'm/s': ['mps', 'm/s', 'meterspersecond', 'm/second'],
                            'ft/s': ['ft/s', 'fps', 'feetpersecond', 'feet/s']}

        if units != None:
            for key in units_conversion.keys():
                if units.lower() in units_conversion[key]:
                    return key

        print('Units Undefined:', units)
        return units

    def filterDataByYear(self, data, year):
        if year != 'ALLYEARS':
            for flag in data.keys():
                years = np.array([n.year for n in data[flag]['dates']])
                msk = np.where(years==year)

                data[flag]['values'] = data[flag]['values'][msk]
                data[flag]['dates'] = data[flag]['dates'][msk]
        return data

    def formatDateXAxis(self, curax, object_settings, twin=False):
        '''
        formats the xaxis to be jdate or datetime and sets up xlimits. also sets up secondary xaxis
        :param curax: current plot axis
        :param object_settings: dictionary of settings
        :param twin: if true, will configure top axis
        :return: sets xlimits for axis
        '''

        if twin:
            if 'xlims2' in object_settings.keys():
                xlims_flag = 'xlims2'
            else:
                print('Using Same Xlims for top and bottom.')
                xlims_flag = 'xlims'
            dateformat_flag = 'dateformat2'
        else:
            xlims_flag = 'xlims'
            dateformat_flag = 'dateformat'

        if dateformat_flag in object_settings.keys():
            dateformat = object_settings[dateformat_flag].lower()
        else:
            print('Dateformat flag not set. Defaulting to datetime..')
            dateformat = 'datetime'

        if xlims_flag in object_settings.keys():
            xlims = object_settings[xlims_flag]#should be min max flags in here

            if 'min' in xlims.keys():
                min = xlims['min']
            else:
                if dateformat == 'datetime':
                    min = self.StartTime
                elif dateformat == 'jdate':
                    min = self.DatetimeToJDate(self.StartTime)
                else:
                    #we've done everything we can at this point..
                    min = self.StartTime

            if 'max' in xlims.keys():
                max = xlims['max']
            else:
                if dateformat == 'datetime':
                    max = self.EndTime
                elif dateformat == 'jdate':
                    max = self.DatetimeToJDate(self.EndTime)
                else:
                    #we've done everything we can at this point..
                    max = self.StartTime

            min = self.translateDateFormat(min, dateformat, self.StartTime)
            max = self.translateDateFormat(max, dateformat, self.EndTime)

            curax.set_xlim(left=min, right=max)

        else:
            print('No Xlims flag set for {0}'.format(xlims_flag))
            print('Not setting Xlims.')

    def getStatsLineData(self, row, data_dict, year='ALL'):
        '''
        takes rows for tables and replaces flags with the correct data, computing stat analysis if needed
        :param row: row section string
        :param data_dict: dictionary of data that could be used
        :param year: selected year, or 'ALL'
        :return: new row value
        '''

        data = {}

        rrow = row.replace('%%', '')
        s_row = rrow.split('.')
        sr_month = ''
        curflag = None
        for sr in s_row:
            if sr in data_dict.keys():
                curflag = sr
                curvalues = np.array(data_dict[sr]['values'])
                curdates = np.array(data_dict[sr]['dates'])
                data[curflag] = {'values': curvalues, 'dates': curdates}
            else:
                if '=' in sr:
                    sr_spl = sr.split('=')
                    if sr_spl[0].lower() == 'month':
                        sr_month = sr_spl[1]
                        try:
                            sr_month = int(sr_month)
                        except ValueError:
                            try:
                                sr_month = self.month2num[sr_month.lower()]
                            except KeyError:
                                print('Invalid Entry for {0}'.format(sr))
                                print('Try using interger values or 3 letter monthly code.')
                                print('Ex: MONTH=1 or MONTH=JAN')
                                continue
                        if curflag == None:
                            print('Invalid Table row for {0}'.format(row))
                            print('Data Key not contained within {0}'.format(data_dict.keys()))
                            print('Please check Datapaths in the XML file, or modify the rows to have the correct flags'
                                  ' for the data present')
                            return data, ''
                        months = np.array([n.month for n in curdates])
                        msk = np.where(months==sr_month)

                        data[curflag]['values'] = curvalues[msk]
                        data[curflag]['dates'] = curdates[msk]


        if year != 'ALL':
            for flag in data.keys():
                years = np.array([n.year for n in data[flag]['dates']])
                msk = np.where(years==year)

                data[flag]['values'] = data[flag]['values'][msk]
                data[flag]['dates'] = data[flag]['dates'][msk]

        return data, sr_month

    def formatStatsProfileLineData(self, row, data_dict, resolution, usedepth, index):
        '''
        formats Profile line statistics for table using user inputs
        finds the highest and lowest overlapping profile points and uses them as end points, then interpolates
        :param row: Row line from inputs. String seperated by '|' and using flags surrounded by '%%'
        :param data_dict: dictionary containing available line data to be used
        :param resolution: number of values to interpolate to. this way each dataset has values at the same levels
                            and there is enough data to do stats over.
        :param usedepth: string bool for using depth or elevation fields
        :param index: date index for profile to use
        :return:
            out_data: dictionary containing values and depths/elevations
        '''

        rrow = row.replace('%%', '')
        s_row = rrow.split('.')
        flags = []
        out_data = {}
        for sr in s_row:
            if sr in data_dict.keys():
                flags.append(sr)
        top = None
        bottom = None
        for flag in flags:
            #get elevs
            if usedepth.lower() == 'true':
                depths = data_dict[flag]['depths'][index]
                if len(depths) > 0:
                    top_depth = np.min(depths)
                    bottom_depth = np.max(depths)
                    #find limits comparing flags so we can be sure to interpolate over the same data
                    if top == None:
                        top = top_depth
                    else:
                        if top_depth > top:
                            top = top_depth

                    if bottom == None:
                        bottom = bottom_depth
                    else:
                        if bottom_depth < bottom:
                            bottom = bottom_depth

            else:
                elevs = data_dict[flag]['elevations'][index]
                if len(elevs) > 0:
                    top_elev = np.max(elevs)
                    bottom_elev = np.min(elevs)
                    #find limits comparing flags so we can be sure to interpolate over the same data
                    if top == None:
                        top = top_elev
                    else:
                        if top_elev < top:
                            top = top_elev

                    if bottom == None:
                        bottom = bottom_elev
                    else:
                        if bottom_elev > bottom:
                            bottom = bottom_elev

        if usedepth.lower() == 'true':
            #build elev profiles
            output_interp_depths = np.arange(top, bottom, (bottom-top)/float(resolution))
        else:
            output_interp_elevations = np.arange(bottom, top, (top-bottom)/float(resolution))

        for flag in flags:
            out_data[flag] = {}
            #interpolate over all values and then get interp values

            if len(data_dict[flag]['values'][index]) < 2:
                print('Insufficient data points with current bounds for {0}'.format(flag))
                out_data[flag]['values'] = []
                out_data[flag]['depths'] = []
                out_data[flag]['elevations'] = []
            else:
                if usedepth.lower() == 'true':
                    f_interp = interpolate.interp1d(data_dict[flag]['depths'][index], data_dict[flag]['values'][index], fill_value='extrapolate')
                    out_data[flag]['depths'] = output_interp_depths
                    out_data[flag]['values'] = f_interp(output_interp_depths)
                else:
                    f_interp = interpolate.interp1d(data_dict[flag]['elevations'][index], data_dict[flag]['values'][index], fill_value='extrapolate')
                    out_data[flag]['elevations'] = output_interp_elevations
                    out_data[flag]['values'] = f_interp(output_interp_elevations)

        return out_data

    def getStatsLine(self, row, data):
        '''
        takes rows for tables and replaces flags with the correct data, computing stat analysis if needed
        :param row: row section string
        :param data_dict: dictionary of data that could be used
        :return:
            out_stat: stat value
            stat: string name for stat
        '''

        flags = list(data.keys())

        if 'Computed' in flags:
            flag1 = 'Computed'
            if len(flags) >= 2:
                flag2 = [n for n in flags if n != flag1][0] #not computed
        else:
            flag1 = flags[0]
            if len(flags) >= 2:
                flag2 = flags[1]

        if row.lower().startswith('%%meanbias'):
            out_stat = WF.calcMeanBias(data[flag1], data[flag2])
            stat = 'meanbias'
        elif row.lower().startswith('%%mae'):
            out_stat = WF.calcMAE(data[flag1], data[flag2])
            stat = 'mae'
        elif row.lower().startswith('%%rmse'):
            out_stat = WF.calcRMSE(data[flag1], data[flag2])
            stat = 'rmse'
        elif row.lower().startswith('%%nse'):
            out_stat = WF.calcNSE(data[flag1], data[flag2])
            stat = 'nse'
        elif row.lower().startswith('%%count'):
            out_stat = WF.getCount(data[flag1])
            stat = 'count'
        elif row.lower().startswith('%%mean'):
            out_stat = WF.calcMean(data[flag1])
            stat = 'mean'
        else:
            if '%%' in row:
                print('Unable to convert flag in row', row)
            return row, ''

        return out_stat, stat

    def buildTable(self, object_settings, split_by_year, data):
        headers = {}
        rows = {}
        outputyears = [n for n in self.years] #this is usually a range or ALLYEARS
        outputyears.append('ALL') #do this last
        for year in outputyears:
            headers[year] = []
            rows[year] = {}
            for ri, row in enumerate(object_settings['rows']):
                rows[year][ri] = []

        for i, header in enumerate(object_settings['headers']):
            if isinstance(object_settings['headers'], dict):
                header = object_settings['headers']['header'] #single headers come as dict objs TODO fix this eventually...
            curheader = pickle.loads(pickle.dumps(header, -1))
            if self.iscomp: #comp run
                isused = False
                for datakey in data.keys():
                    if '%%{0}%%'.format(data[datakey]['flag']) in curheader: #found data specific flag
                        isused = True
                        if 'ID' in data[datakey].keys():
                            ID = data[datakey]['ID']
                            tmpheader = self.configureSettingsForID(ID, curheader)
                        else:
                            tmpheader = pickle.loads(pickle.dumps(curheader, -1))
                        tmpheader = tmpheader.replace('%%{0}%%'.format(data[datakey]['flag']), '')
                        if split_by_year and '%%year%%' in curheader:
                            for year in self.years:
                                headers[year].append(tmpheader)
                                for ri, row in enumerate(object_settings['rows']):
                                    srow = row.split('|')[1:][i]
                                    rows[year][ri].append(srow.replace(data[datakey]['flag'], datakey))
                        else:
                            headers['ALL'].append(tmpheader)
                            for ri, row in enumerate(object_settings['rows']):
                                srow = row.split('|')[1:][i]
                                rows['ALL'][ri].append(srow.replace(data[datakey]['flag'], datakey))

                if not isused: #if a header doesnt get used, probably something observed and not needing replacing.
                    if split_by_year and '%%year%%' in curheader:
                        for year in self.years:
                            headers[year].append(curheader)
                            for ri, row in enumerate(object_settings['rows']):
                                srow = row.split('|')[1:][i]
                                rows[year][ri].append(srow)

                    else:
                        headers['ALL'].append(curheader)
                        for ri, row in enumerate(object_settings['rows']):
                            srow = row.split('|')[1:][i]
                            rows['ALL'][ri].append(srow)

            else: #single run
                if split_by_year and '%%year%%' in curheader:
                    for year in self.years:
                        headers[year].append(curheader)
                        for ri, row in enumerate(object_settings['rows']):
                            srow = row.split('|')[1:][i]
                            rows[year][ri].append(srow)

                else:
                    headers['ALL'].append(curheader)
                    for ri, row in enumerate(object_settings['rows']):
                        srow = row.split('|')[1:][i]
                        rows['ALL'][ri].append(srow)
        

        organizedheaders = []
        organizedrows = []
        for row in object_settings['rows']:
            organizedrows.append(row.split('|')[0])
        for year in outputyears:
            yrstr = str(year) if split_by_year and year != 'ALL' else self.years_str
            for hdr in headers[year]:
                organizedheaders.append([year, self.updateFlaggedValues(hdr, '%%year%%', yrstr)])
            for ri in rows[year].keys():
                for rw in rows[year][ri]:
                    organizedrows[ri] += '|{0}'.format(rw)

        return organizedheaders, organizedrows

    def buildProfileStatsTable(self, object_settings, timestamp, data):
        headers = []
        rows = []
        for ri, row in enumerate(object_settings['rows']):
            rows.append(row.split('|')[0])

        if self.iscomp: #comp run
            for i, header in enumerate(object_settings['headers']):
                if isinstance(object_settings['headers'], dict):
                    header = object_settings['headers']['header'] #single headers come as dict objs TODO fix this eventually...
                curheader = pickle.loads(pickle.dumps(header, -1))
                for datakey in data.keys():
                    if '%%{0}%%'.format(data[datakey]['flag']) in curheader: #found data specific flag
                        if 'ID' in data[datakey].keys():
                            ID = data[datakey]['ID']
                            tmpheader = self.configureSettingsForID(ID, curheader)
                        else:
                            tmpheader = pickle.loads(pickle.dumps(curheader, -1))
                        tmpheader = tmpheader.replace('%%{0}%%'.format(data[datakey]['flag']), '')
                        headers.append(tmpheader)
                        for ri, row in enumerate(object_settings['rows']):
                            srow = row.split('|')[1:][i]
                            rows[ri] += '|{0}'.format(srow.replace(data[datakey]['flag'], datakey))

        else: #single run
            headers = [timestamp]
            rows = object_settings['rows']

        return headers, rows


    def buildHeadersByYear(self, object_settings, years, split_by_year):
        '''
        if split by year is selected, and a header has %%year%% flag, iterate through and create a new header for
        each year and header
        :param object_settings: dictionary of settings
        :param years: list of years
        :param split_by_year: boolean if splitting up by year or not
        :return: list of headers
        '''

        headings = []
        header_by_year = []
        yearstr = object_settings['yearstr']
        for i, header in enumerate(object_settings['headers']):
            if isinstance(object_settings['headers'], dict):
                header = object_settings['headers']['header'] #single headers come as dict objs TODO fix this eventually...
            if '%%year%%' in header:
                if split_by_year:
                    header_by_year.append(header)
                else:
                    headings.append(['ALL', self.updateFlaggedValues(header, '%%year%%', yearstr)])
            else:
                if len(header_by_year) > 0:
                    for yi, year in enumerate(years):
                        for yrhd in header_by_year:
                            headings.append([year, self.updateFlaggedValues(yrhd, '%%year%%', yearstr[yi])])
                    header_by_year = []
                headings.append(['ALL', header])
        if len(header_by_year) > 0:
            for yi, year in enumerate(years):
                for yrhd in header_by_year:
                    headings.append([year, self.updateFlaggedValues(yrhd, '%%year%%', yearstr[yi])])
        return headings

    def buildTimeSeries(self, startTime, endTime, interval):
        '''
        builds a regular time series using the start and end time and a given interval
        #TODO: if start time isnt on the hour, but the interval is, change start time to be hourly?
        :param startTime: datetime object
        :param endTime: datetime object
        :param interval: DSS interval
        :return: list of time series dates
        '''

        try:
            intervalinfo = self.time_intervals[interval]
            interval = intervalinfo[0]
            interval_info = intervalinfo[1]
        except KeyError:
            interval_info = 'np'

        if interval_info == 'np':
            ts = np.arange(startTime, endTime, interval)
            ts = np.asarray([t.astype(dt.datetime) for t in ts])
        elif interval_info == 'pd':
            ts = pd.date_range(startTime, endTime, freq=interval, closed=None)
            ts = np.asarray([t.to_pydatetime() for t in ts])
        return ts

    def buildFileName(self, Line_info):
        '''
        creates uniform name for csv log output for data
        :param Line_info: dictionary containing line values
        :return: file name
        '''

        MemKey = self.buildDataMemoryKey(Line_info)
        if MemKey == 'Null':
            return MemKey
        else:
            return MemKey + '.csv'

    def buildDataMemoryKey(self, Line_info):
        '''
        creates uniform name for csv log output for data
        determines how to build the file name from the input type
        :param Line_info: information about line
        :return: name for memory key, or null if can't be determined
        '''

        if 'dss_path' in Line_info.keys(): #Get data from DSS record
            if 'dss_filename' in Line_info.keys():
                outname = '{0}_{1}'.format(os.path.basename(Line_info['dss_filename']).split('.')[0],
                                        Line_info['dss_path'].replace('/', '').replace(':', ''))
                return outname

        elif 'w2_file' in Line_info.keys():
            if 'structurenumbers' in Line_info.keys():
                outname = '{0}'.format(os.path.basename(Line_info['w2_file']).split('.')[0])
                if isinstance(Line_info['structurenumbers'], dict):
                    structure_nums = [Line_info['structurenumbers']['structurenumber']]
                elif isinstance(Line_info['structurenumbers'], str):
                    structure_nums = [Line_info['structurenumbers']]
                elif isinstance(Line_info['structurenumbers'], (list, np.ndarray)):
                    structure_nums = Line_info['structurenumbers']
                outname += '_Struct_' + '_'.join(structure_nums)
            else:
                outname = '{0}'.format(os.path.basename(Line_info['w2_file']).split('.')[0])
            return outname

        elif 'easting' in Line_info.keys() and 'northing' in Line_info.keys():
            outname = '{0}_{1}_{2}'.format(Line_info['parameter'], Line_info['easting'], Line_info['northing'])
            return outname

        elif 'filename' in Line_info.keys(): #Get data from Observed Profile
            outname = '{0}'.format(os.path.basename(Line_info['filename']).split('.')[0].replace(' ', '_'))
            return outname

        elif 'w2_segment' in Line_info.keys():
            outname = 'W2_{0}_{1}_profile'.format(self.ModelAlt.output_file_name.split('.')[0], Line_info['w2_segment'])
            return outname

        elif 'ressimresname' in Line_info.keys():
            outname = '{0}_{1}_{2}'.format(os.path.basename(self.ModelAlt.h5fname).split('.')[0]+'_h5',
                                               Line_info['parameter'], Line_info['ressimresname'])
            return outname

        return 'NULL'

    def buildHeadersByTimestamps(self, timestamps, years):
        '''
        build headers for profile line stat tables by timestamp
        convert to Datetime, no matter what. We can convert back..
        Filter by year, using year input. If ALLYEARS, no data is filtered.
        :param timestamps: list of available timesteps
        :param year: used to filter down to the year, or if ALLYEARS, allow all years
        :return: list of headers
        '''

        headers = []
        headers_i = []

        for year in years:
            h = []
            hi = []
            for ti, timestamp in enumerate(timestamps):
                if isinstance(timestamp, dt.datetime):
                    if year == timestamp.year:
                        h.append(timestamp)
                        hi.append(ti)

                elif isinstance(timestamp, float):
                    ts_dt = self.JDateToDatetime(timestamp)
                    if year == ts_dt.year:
                        h.append(str(timestamp))
                        hi.append(ti)
            headers.append(h)
            headers_i.append(hi)

        return headers, headers_i

    def equalizeLog(self):
        '''
        ensure that all arrays are the same length with a '' character
        :return: append self.Log object
        '''

        longest_array_len = 0
        for key in self.Log.keys():
            if len(self.Log[key]) > longest_array_len:
                longest_array_len = len(self.Log[key])
        for key in self.Log.keys():
            if len(self.Log[key]) < longest_array_len:
                num_entries = longest_array_len - len(self.Log[key])
                for i in range(num_entries):
                    self.Log[key].append('')

    def buildLogFile(self):
        '''
        builts the log dictionary for conisistent dictionary values
        '''

        self.Log = {'type': [], 'name': [], 'description': [], 'value': [], 'units': [], 'observed_data_path': [],
                    'start_time': [], 'end_time': [], 'compute_time': [], 'program': [], 'alternative_name': [],
                    'fpart': [], 'program_directory': [], 'region': [], 'value_start_date': [], 'value_end_date': [],
                    'function': [], 'logoutputfilename': []}

    def limitXdata(self, dates, values, xlims):
        '''
        if the filterbylimits flag is true, filters out values outside of the xlimits
        :param dates: list of dates
        :param values: list of values
        :param xlims: dictionary of xlims, containing potentially min and/or max
        :return: filtered dates and values
        '''

        if isinstance(dates[0], (int, float)):
            wantedformat = 'jdate'
        elif isinstance(dates[0], dt.datetime):
            wantedformat = 'datetime'
        if 'min' in xlims.keys():
            min = self.translateDateFormat(xlims['min'], wantedformat, self.StartTime)
            for i, d in enumerate(dates):
                if min > d:
                    values[i] = np.nan #exclude
        if 'max' in xlims.keys():
            max = self.translateDateFormat(xlims['max'], wantedformat, self.EndTime)
            for i, d in enumerate(dates):
                if max < d:
                    values[i] = np.nan #exclude

        return dates, values

    def limitYdata(self, dates, values, ylims):
        '''
        if the filterbylimits flag is true, filters out values outside of the ylimits
        :param dates: list of dates
        :param values: list of values
        :param ylims: dictionary of ylims, containing potentially min and/or max
        :return: filtered dates and values
        '''

        if 'min' in ylims.keys():
            for i, v in enumerate(values):
                if float(ylims['min']) > v:
                    values[i] = np.nan #exclude
        if 'max' in ylims.keys():
            for i, v in enumerate(values):
                if float(ylims['max']) < v:
                    values[i] = np.nan #exclude

        return dates, values

    def writeXMLIntroduction(self):
        '''
        writes the intro section for XML file. Creates a line in the intro for each model used
        '''

        self.XML.writeIntroStart()
        for model in self.SimulationCSV.keys():
            # self.XML.writeIntroLine(self.SimulationCSV[model]['plugin'])
            self.XML.writeIntroLine('%%REPLACEINTRO_{0}%%'.format(model))
        self.XML.writeIntroEnd()

    def writeChapter(self):
        '''
        writes each chapter defined in the simulation CSV file to the XML file. Generates plots and figures
        :return: class variables
                    self.ChapterName
                    self.ChapterRegion
        '''

        for Chapter in self.ChapterDefinitions:
            self.ChapterName = Chapter['name']
            self.ChapterRegion = Chapter['region']
            self.addLogEntry({'region': self.ChapterRegion})
            self.XML.writeChapterStart(self.ChapterName)
            for section in Chapter['sections']:
                section_header = section['header']
                self.XML.writeSectionHeader(section_header)
                for object in section['objects']:
                    objtype = object['type'].lower()
                    if objtype == 'timeseriesplot':
                        self.makeTimeSeriesPlot(object)
                    elif objtype == 'profileplot':
                        self.makeProfilePlot(object)
                    elif objtype == 'errorstatisticstable':
                        self.makeErrorStatisticsTable(object)
                    elif objtype == 'monthlystatisticstable':
                        self.makeMonthlyStatisticsTable(object)
                    elif objtype == 'buzzplot':
                        self.makeBuzzPlot(object)
                    elif objtype == 'profilestatisticstable':
                        self.makeProfileStatisticsTable(object)
                    elif objtype == 'contourplot':
                        self.makeContourPlot(object)
                    else:
                        print('Section Type {0} not identified.'.format(objtype))
                        print('Skipping Section..')
                self.XML.writeSectionHeaderEnd()
            print('\n################################')
            print('Chapter Complete.')
            print('################################\n')
            self.XML.writeChapterEnd()

    def writeLogFile(self):
        '''
        Writes out logfile data to csv file in report dir
        '''

        df = pd.DataFrame({'observed_data_path': self.Log['observed_data_path'],
                           'start_time': self.Log['start_time'],
                           'end_time': self.Log['end_time'],
                           'compute_time': self.Log['compute_time'],
                           'program': self.Log['program'],
                           'region': self.Log['region'],
                           'alternative_name': self.Log['alternative_name'],
                           'fpart': self.Log['fpart'],
                           'program_directory': self.Log['program_directory'],
                           'type': self.Log['type'],
                           'name': self.Log['name'],
                           'description': self.Log['description'],
                           'function': self.Log['function'],
                           'value': self.Log['value'],
                           'units': self.Log['units'],
                           'value_start_date': self.Log['value_start_date'],
                           'value_end_date': self.Log['value_end_date'],
                           'CSVOutputFilename': self.Log['logoutputfilename']})

        df.to_csv(os.path.join(self.images_path, 'Log.csv'), index=False)

    def writeDataFiles(self):
        '''
        writes out the data used in figures to csv files for later use and checking
        '''

        for key in self.Data_Memory.keys():
            csv_name = os.path.join(self.CSVPath, '{0}.csv'.format(key))
            try:
                if 'isprofile' in self.Data_Memory[key].keys():
                    if self.Data_Memory[key]['isprofile'] == True:
                        alltimes = self.Data_Memory[key]['times']
                        allvalues = self.Data_Memory[key]['values']
                        alltimes = self.matcharrays(alltimes, allvalues)
                        allelevs = self.Data_Memory[key]['elevations']
                        alldepths = self.Data_Memory[key]['depths']
                        if len(allelevs) == 0: #elevations may not always fall out
                            allelevs = self.matcharrays(allelevs, alldepths)
                        units = self.Data_Memory[key]['units']
                        values = self.getListItems(allvalues)
                        times = self.getListItems(alltimes)
                        elevs = self.getListItems(allelevs)
                        depths = self.getListItems(alldepths)
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
                elif 'iscontour' in self.Data_Memory[key].keys():
                    continue #were not doing this for now, takes ~ 5 seconds per 3yr reach..
                    if self.Data_Memory[key]['iscontour'] == True:
                        alltimes = self.Data_Memory[key]['dates']
                        allvalues = self.Data_Memory[key]['values'].T #this gets transposed a few times.. we want distance/date
                        alldistance = self.Data_Memory[key]['distance']
                        times = self.matcharrays(alltimes, allvalues)
                        distances = self.matcharrays(alldistance, allvalues)
                        values = self.getListItems(allvalues)
                        units = self.Data_Memory[key]['units']
                        newstime = time.time()
                        df = pd.DataFrame({'Dates': times, 'Values ({0})'.format(units): values, 'Distances': distances,
                                           })
                else:
                    allvalues = self.Data_Memory[key]['values']
                    alltimes = self.Data_Memory[key]['times']
                    units = self.Data_Memory[key]['units']
                    values = self.getListItems(allvalues)
                    times = self.getListItems(alltimes)
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

    def matcharrays(self, array1, array2):
        '''
        iterative recursive function that aims to line up arrays of different lengths. Takes in variable input so that
        if there are lists of lists with a single date (aka profiles), alligns those so each elevation value has a date
        assigned to it for easy output
        :param array1: np.array or list of values, generally values
        :param array2: np.array or list of values, generally dates
        :return: array1 with correct length
        '''

        if isinstance(array1, (list, np.ndarray)) and isinstance(array2, (list, np.ndarray)):
            if len(np.asarray(array1).shape) < len(np.asarray(array2).shape):
                new_array1 = np.array([])
                for i, ar2 in enumerate(array2):
                    new_array1 = np.append(new_array1, np.asarray([array1[i]] * len(ar2)))
                return new_array1
            #if both are lists..
            elif len(array1) < len(array2):
                '''
                either ['Date1', 'Date2'], ['1,2,3'] OR ['Date1'], [1,2,3] OR ['DATE1'], [[1,2,3], [1,2,3,4]]
                 OR ['Date1', 'Date2'], [[1,2,3], [2,4,5],[6,
                 or [], [1,2,3]
                scenario 1: shouldnt ever happen
                scenario 2: do Date1 for each item in array2
                scenario 3: do date1 for each value in each subarray in 2 '''

                if len(array1) == 1: #solo date
                    new_array1 = []
                    for subarray2 in array2:
                        new_array1.append(self.matcharrays(array1[0], subarray2))
                    return new_array1
                elif len(array1) == 0: #no data
                    new_array1 = []
                    for subarray2 in array2:
                        new_array1.append(self.matcharrays('', subarray2))
                    return new_array1

                else:
                    print('ERROR') #If the Len of the arrays are offset, then there should only ever be 1 date
            elif len(array1) == len(array2):
                new_array1 = []
                for i, subarray1 in enumerate(array1):
                    new_array1.append(self.matcharrays(subarray1, array2[i]))
                return new_array1
            else:
                print('Array 1 is bigger than array2')
                print(len(array1))
                print(len(array2))
                new_array1 = []
                for i in range(len(array2)):
                    new_array1.append(array1[i])
                return new_array1

        #GOAL LOOP
        elif isinstance(array1, (str, dt.datetime, int, float)) and isinstance(array2, (list, np.ndarray)):
            # array1 is a single value, array2 is a list of values
            new_array1 = []
            for subarray2 in array2:
                if isinstance(subarray2, (list, np.ndarray)):
                    new_array1.append(self.matcharrays(array1, subarray2))
                else:
                    new_array1.append(array1)
            return new_array1

        else:
            return array1

    def mergeLines(self, data, settings):
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
                    data[controller], data[flag] = WF.matchData(data[controller], data[flag])
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

    def cleanOutputDirs(self):
        '''
        cleans the images output directory, so pngs from old reports aren't mistakenly
        added to new reports. Creates directory if it doesn't exist.
        '''

        if not os.path.exists(self.images_path):
            os.makedirs(self.images_path)
        if not os.path.exists(self.CSVPath):
            os.makedirs(self.CSVPath)

        WF.cleanOutputDirectory(self.images_path, '.png')
        WF.cleanOutputDirectory(self.images_path, '.csv')
        WF.cleanOutputDirectory(self.CSVPath, '.csv')

    def loadModelAlts(self, simCSVAlt):
        '''
        Loads info for specified model alts. Loads correct model plugin class from WDR
        :param simCSVAlt: simulation alt dict object from self.simulation class
        :return: class variables
                self.alternativeFpart
                self.alternativeDirectory
                self.ModelAlt - WDR class that is plugin specific
        '''
        self.accepted_IDs = []
        for ID in self.SimulationVariables.keys():
            approved_modelalts = [modelalt for modelalt in self.SimulationVariables[ID]['ModelAlternatives']
                                  if modelalt['name'] in simCSVAlt['modelaltnames'] and
                                  modelalt['program'] in simCSVAlt['plugins']]
            # if len(approved_modelalts) == 0:
            #
            # else:
            if len(approved_modelalts) > 0:
                approved_modelalt = approved_modelalts[0]
                print('Added {0} for ID {1}'.format(approved_modelalt['program'], ID))
                self.SimulationVariables[ID]['alternativeFpart'] = approved_modelalt['fpart']
                self.SimulationVariables[ID]['alternativeDirectory'] = approved_modelalt['directory']
                self.SimulationVariables[ID]['modelAltName'] = approved_modelalt['name']
                self.SimulationVariables[ID]['plugin'] = approved_modelalt['program']

                if self.SimulationVariables[ID]['plugin'].lower() == "ressim":
                    self.SimulationVariables[ID]['ModelAlt'] = WDR.ResSim_Results(self.SimulationVariables[ID]['simulationDir'],
                                                                                  self.SimulationVariables[ID]['alternativeFpart'],
                                                                                  self.StartTime, self.EndTime)
                elif self.SimulationVariables[ID]['plugin'].lower() == 'cequalw2':
                    self.SimulationVariables[ID]['ModelAlt'] = WDR.W2_Results(self.SimulationVariables[ID]['simulationDir'],
                                                                              self.SimulationVariables[ID]['modelAltName'],
                                                                              self.SimulationVariables[ID]['alternativeDirectory'],
                                                                              self.StartTime, self.EndTime)
                else:
                    self.SimulationVariables[ID]['ModelAlt'] == 'unknown'
                self.accepted_IDs.append(ID)

        if len(self.accepted_IDs) == 0:
            if self.iscomp:
                csv_file_name = '{0}_comparison.csv'.format(self.baseSimulationName.replace(' ', '_'))
            else:
                csv_file_name = '{0}.csv'.format(self.baseSimulationName.replace(' ', '_'))
            print('Incompatible input information from the WAT XML output file ({0})\nand Simulation CSV file ({1})'.format(self.simulationInfoFile, csv_file_name))
            print('Please Confirm inputs and run again.')
            if self.iscomp:
                print('If comparison plot, ensure that all model alts are in {0}'.format(csv_file_name))
                print('Example line: ResSim, TmpNFlo, CeQualW2, Shasta from DSS 15, Shasta_ResSim_TCD_comparison.XML')
            print('Now Exiting...')
            sys.exit()

    def loadCurrentID(self, ID):
        self.currentlyloadedID = ID
        self.SimulationName = self.SimulationVariables[ID]['SimulationName']
        self.baseSimulationName = self.SimulationVariables[ID]['baseSimulationName']
        self.SimulationDir = self.SimulationVariables[ID]['simulationDir']
        self.DSSFile = self.SimulationVariables[ID]['DSSFile']
        self.StartTimeStr = self.SimulationVariables[ID]['StartTimeStr']
        self.EndTimeStr = self.SimulationVariables[ID]['EndTimeStr']
        self.LastComputed = self.SimulationVariables[ID]['LastComputed']
        self.ModelAlternatives = self.SimulationVariables[ID]['ModelAlternatives']
        self.StartTime = self.SimulationVariables[ID]['StartTime']
        self.EndTime = self.SimulationVariables[ID]['EndTime']

    def loadCurrentModelAltID(self, ID):
        self.alternativeFpart = self.SimulationVariables[ID]['alternativeFpart']
        self.alternativeDirectory = self.SimulationVariables[ID]['alternativeDirectory']
        self.modelAltName = self.SimulationVariables[ID]['modelAltName']
        self.plugin = self.SimulationVariables[ID]['plugin']
        self.ModelAlt = self.SimulationVariables[ID]['ModelAlt']
        # print('Model {0} Loaded'.format(ID))

    def initializeXML(self):
        '''
        creates a new version of the template XML file, initiates the XML class and writes the cover page
        :return: sets class variables
                    self.XML
        '''

        new_xml = os.path.join(self.studyDir, 'reports', 'Datasources', 'USBRAutomatedReportOutput.xml') #required name for file
        self.XML = XML_Utils.XMLReport(new_xml)
        self.XML.writeCover('DRAFT Temperature Validation Summary Report')

    def initializeDataMemory(self):
        self.Data_Memory = {}

    def initSimulationDict(self):
        self.SimulationVariables = {}

    def ensureDefaultFiles(self):
        '''
        Copies Reporting files into the main study if they dont exist. Allows for "default" reports out of the gate
        Checks if the default directory exists in the install. if not continue
        then look for files ending in specific extensions in the default dir and check for them in the destination
        '''

        if not os.path.exists(self.default_dir):
            print('ERROR finding default files at {0}'.format(self.default_dir))
            print('No default files copied.')
            return

        default_copy_dir = os.path.join(os.path.split(self.batdir)[0], 'Default', '_COPY', 'Reports')
        to_dir = os.path.join(self.studyDir, 'Reports')
        filetypes = ['.xml', '.csv', '.jrxml', '.png']
        self.copyDefaultFiles(default_copy_dir, to_dir, filetypes)

        default_copy_dir = os.path.join(os.path.split(self.batdir)[0], 'Default', '_COPY', 'Reports.DataSources')
        to_dir = os.path.join(self.studyDir, 'Reports', 'DataSources')
        filetypes = ['.xml']
        self.copyDefaultFiles(default_copy_dir, to_dir, filetypes)

    def copyDefaultFiles(self, fromdir, todir, filetypes):
        '''
        looks for all files of given extensions in a default directory, and if they dont exist in an output dir, copy
        them. If the fromdir doesnt exist, return.
        :param fromdir: default directory where the files live
        :param todir: directory to copy files too, usually self.study_dir/reports
        :param filetypes: list of filetype extensions of files to copy
        '''

        if not os.path.exists(fromdir):
            print('ERROR finding default files at {0}'.format(fromdir))
            print('No default files copied.')
            return

        files_in_directory = os.listdir(fromdir)
        for filetype in filetypes:
            filtered_files = [file for file in files_in_directory if file.endswith(filetype)]
            for filtfile in filtered_files:
                old_file_path = os.path.join(fromdir, filtfile)
                new_file_path = os.path.join(todir, filtfile)
                if not os.path.exists(new_file_path):
                    shutil.copyfile(old_file_path, new_file_path)
                    print('Successfully copied to', new_file_path)

    def copyObjectSettingsToAxSetting(self, to_dict, from_dict, ignore=[]):
        for key in from_dict.keys():
            if key not in ignore:
                if key not in to_dict.keys():
                    to_dict[key] = from_dict[key]
        return to_dict

    def correctDuplicateLabels(self, linedata):
        for line in linedata.keys():
            if 'label' in linedata[line].keys():
                curlabel = linedata[line]['label']
                lineidx = linedata[line]['numtimesused']
                if lineidx > 0: #leave the first guy alone..
                    for otherline in linedata.keys():
                        if otherline != line:
                            if linedata[otherline]['label'] == curlabel:
                                linedata[line]['label'] = '{0} {1}'.format(curlabel, lineidx) #append the number
        return linedata

    def correctTableUnits(self, data, object_settings):
        for datapath in data.keys():
            values = data[datapath]['values']
            units = data[datapath]['units']
            if 'parameter' in data[datapath].keys():
                units = self.configureUnits(object_settings, data[datapath]['parameter'], units)
            if 'unitsystem' in object_settings.keys():
                data[datapath]['values'], data[datapath]['units'] = self.convertUnitSystem(values, units, object_settings['unitsystem'])

        return data

    def loadDefaultPlotObject(self, plotobject):
        '''
        loads the graphic default options.
        :param plotobject: string specifying the default graphics object
        :return:
            plot_info: dict of object settings
        '''

        plot_info = pickle.loads(pickle.dumps(self.graphicsDefault[plotobject], -1))
        return plot_info

    def getLineModelType(self, Line_info):
        # plugin = self.SimulationVariables[ID]['plugin'].lower()
        model_specific_vars = {'ressimresname': 'ressim',
                               'xy': 'ressim',
                               'w2_segment': 'cequalw2',
                               'w2_file': 'cequalw2'}
        for var, ident in model_specific_vars.items():
            # if var in Line_info.keys() and ident == plugin:
            if var in Line_info.keys():
                return ident
            # elif var in Line_info and ident != plugin:
            #     return ident
        return 'undefined' #no id either way..

    def DatetimeToJDate(self, dates):
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
            jdates = np.asarray([(WF.datetime2Ordinal(n) - self.ModelAlt.t_offset) + 1 for n in dates])
            return jdates
        elif isinstance(dates, dt.datetime):
            jdate = (WF.datetime2Ordinal(dates) - self.ModelAlt.t_offset) + 1
            return jdate
        else:
            return dates

    def JDateToDatetime(self, dates):
        '''
        converts jdate dates to datetime values
        :param dates: list of jdate dates
        :return:
            dtimes: list of dates
            dtime: single date
            dates: original date if unable to convert
        '''

        first_year_Date = dt.datetime(self.ModelAlt.dt_dates[0].year, 1, 1, 0, 0)

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

    def printVersion(self):
        print('VERSION:', VERSIONNUMBER)

    def pickByParameter(self, values, line):
        '''
        some data (W2) has multiple parameters coming from a single results file, we can't know which one we want at
        the moment. This grabs the right parameter based on input
        :param values: dictionary of values
        :param line: dictionary of line settings
        :return:
            values: list of values
        '''

        w2_param_dict = {'temperature': 't(c)',
                         'elevation': 'elevcl',
                         'flow': 'q(m3/s)'}

        if 'parameter' not in line.keys():
            print("Parameter not set for line.")
            print("using the first set of values, {0}".format(list(values.keys())[0]))
            return values[list(values.keys())[0]]
        else:
            if line['parameter'].lower() not in w2_param_dict.keys():
                print('Parameter {0} not found in dict in pickByParameter(). {1}'.format(line['parameter'].lower(), w2_param_dict.keys()))
                print("using the first set of values, {0}".format(list(values.keys())[0]))
                return values[list(values.keys())[0]]
            else:
                p = line['parameter'].lower()
                param_key = w2_param_dict[p]
                return values[param_key]

    def prioritizeKey(self, firstchoice, secondchoice, key, backup=None):
        if key in firstchoice:
            return firstchoice[key]
        elif key in secondchoice:
            return secondchoice[key]
        else:
            return backup

    def buzzTargetSum(self, dates, values, target):
        '''
        finds buzzplot targets defined and returns the flow sums
        :param dates: list of dates
        :param values: list of dicts of values @ structures
        :param target: target value
        :return: sum of values
        '''

        sum_vals = []
        for i, d in enumerate(dates):
            sum = 0
            for sn in values.keys():
                if values[sn]['elevcl'][i] == target:
                    sum += values[sn]['q(m3/s)'][i]
            sum_vals.append(sum)
        return np.asarray(sum_vals)

    def convertUnitSystem(self, values, units, target_unitsystem):
        '''
        converts unit systems if defined english/metric
        :param values: list of values
        :param units: units of select values
        :param target_unitsystem: unit system to convert to
        :return:
            values: either converted units if successful, or original values if unsuccessful
            units: original units if unsuccessful
            new_units: new converted units if successful
        '''

        #Following is the SOURCE units, then the conversion to units listed above
        conversion = {'m3/s': 35.314666213,
                      'cfs': 0.0283168469997284,
                      'm': 3.28084,
                      'ft': 0.3048,
                      'm3': 0.000810714,
                      'af': 1233.48}

        units = self.translateUnits(units)

        english_units = {self.units[key]['metric']: self.units[key]['english'] for key in self.units.keys()}
        metric_units = {v: k for k, v in english_units.items()}

        if units == None:
            print('Units undefined.')
            return values, units

        if target_unitsystem.lower() == 'english':
            if units.lower() in english_units.keys():
                new_units = english_units[units.lower()]
                print('Converting {0} to {1}'.format(units, new_units))
            elif units.lower() in english_units.values():
                print('Values already in target unit system. {0} {1}'.format(units, target_unitsystem))
                return values, units
            else:
                print('Units not found in definitions. Not Converting.')
                return values, units

        elif target_unitsystem.lower() == 'metric':
            if units.lower() in metric_units.keys():
                new_units = metric_units[units.lower()]
                print('Converting {0} to {1}'.format(units, new_units))
            elif units.lower() in metric_units.values():
                print('Values already in target unit system. {0} {1}'.format(units, target_unitsystem))
                return values, units
            else:
                print('Units not found in definitions. Not Converting.')
                return values, units

        else:
            print('Target Unit System undefined.', target_unitsystem)
            print('Try english or metric')
            return values, units

        if units == new_units:
            print('data already in target unit system.')
            return values, units

        if units.lower() in ['c', 'f']:
            values = WF.convertTempUnits(values, units)
        elif units.lower() in conversion.keys():
            conversion_factor = conversion[units.lower()]
            values *= conversion_factor
        elif new_units.lower() in conversion.keys():
            conversion_factor = 1/conversion[units.lower()]
            values *= conversion_factor
        else:
            print('Undefined Units conversion for units {0}.'.format(units))
            print('No Conversions taking place.')
            return values, units

        return values, new_units

    def convertHeaderFormats(self, headers, object_settings):
        '''
        converts the formats of headers for profile line data tables to the correct format
        if the dateformat is selected, returns a formatted string.
        if Jdate, the string of the float value is used
        Datetime if not specified
        :param headers: list of datetime objects for headers
        :param object_settings: user defined settings for current object
        :return: list of new headers
        '''

        if 'dateformat' not in object_settings.keys():
            object_settings['dateformat'] = 'datetime'

        new_headers = []
        for headeryear in headers:
            nh = []
            for header in headeryear:
                if object_settings['dateformat'].lower() == 'datetime':
                    header = self.translateDateFormat(header, 'datetime', '')
                    header = header.strftime('%d%b%Y')
                elif object_settings['dateformat'].lower() == 'jdate':
                    header = self.translateDateFormat(header, 'jdate', '')
                    header = str(header)
                nh.append(header)
            new_headers.append(nh)

        return new_headers

    def convertProfileDataUnits(self, object_settings, data, line_settings):
        '''
        converts the units of profile data if unitsystem is defined
        :param object_settings: user defined settings for current object
        :return: object_setting dictionaries with updated units and values
        '''

        if 'unitsystem' not in object_settings.keys():
            print('Unit system not defined.')
            return data, line_settings
        for flag in data.keys():
            if line_settings[flag]['units'] == None:
                continue
            else:
                profiles = data[flag]['values']
                profileunits = line_settings[flag]['units']
                for pi, profile in enumerate(profiles):
                    profile, newunits = self.convertUnitSystem(profile, profileunits, object_settings['unitsystem'])
                    profiles[pi] = profile
                line_settings[flag]['units'] = newunits
        return data, line_settings

    def buildRowsByYear(self, object_settings, years, split_by_year):
        '''
        if split by year is selected, and a header has %%year%% flag, iterate through and create a new row object for
        each year and header
        :param object_settings: dictionary of settings
        :param years: list of years
        :param split_by_year: boolean if splitting up by year or not
        :return:
            rows: list of newly built rows
        '''

        rows = []
        rows_by_year = []
        for i, row in enumerate(object_settings['rows']):
            if isinstance(object_settings['rows'], dict):
                row = object_settings['rows']['row'] #single headers come as dict objs TODO fix this eventually...
            srow = row.split('|')
            r = [srow[0]] #<Row>Jan|%%MEAN.Computed.MONTH=JAN%%|%%MEAN.Observed.MONTH=JAN%%</Row>
            for si, sr in enumerate(srow[1:]):
                if isinstance(object_settings['headers'][si], dict):
                    header = object_settings['headers'][si]['header'] #single headers come as dict objs TODO fix this eventually...
                else:
                    header = object_settings['headers'][si]
                if '%%year%%' in header:
                    if split_by_year:
                        rows_by_year.append(sr)
                    else:
                        r.append(sr)
                else:
                    if len(rows_by_year) > 0:
                        for year in years:
                            for yrrow in rows_by_year:
                                r.append(yrrow)
                        rows_by_year = []
                    r.append(sr)
            if len(rows_by_year) > 0:
                for year in years:
                    for yrrow in rows_by_year:
                        r.append(yrrow)
                rows_by_year = []
            rows.append('|'.join(r))
        return rows

    def filterTimestepByYear(self, timestamps, year):
        '''
        returns only timestamps from the given year. Otherwise, just return all timestamps
        :param timestamps: list of dates
        :param year: target year
        :return:
            timestamps: original date values
            list of selected timestamps
        '''

        if year == 'ALLYEARS':
            return timestamps
        return [n for n in timestamps if n.year == year]

    def fixXMLModelIntroduction(self, simorder):
        outstr = '{0}:'.format(self.ChapterRegion)
        # for cnt, (ID, simvar) in enumerate(self.SimulationVariables.items()):
        for cnt, ID in enumerate(self.accepted_IDs):
            if cnt > 0:
                outstr += ','
            outstr += ' {0}'.format(self.SimulationVariables[ID]['plugin'])
            # outstr += ' {0}-{1}'.format(simvar['plugin'], simvar['SimulationName'])
        self.XML.replaceinXML('%%REPLACEINTRO_{0}%%'.format(simorder), outstr)

    def fixDuplicateColors(self, line_settings):
        '''
        when doing comparison runs, we can end up with multiple runs with the same lines set
        settings can be set to a list of colors, like linecolors instead of linecolor.
        finds the correct index for each line ,or chooses a default color
        :param line_settings:
        :return:
        '''
        lineusedcount = line_settings['numtimesused']
        if lineusedcount > len(self.def_colors):
            defcol_idx = lineusedcount%len(self.def_colors)
        else:
            defcol_idx = lineusedcount
        if line_settings['drawline'].lower() == 'true':
            if lineusedcount > 0: #if more than one, the color specified is already used. Use a new color..
                if 'linecolors' in line_settings.keys():
                    if lineusedcount > len(line_settings['linecolors']):
                        lc_idx = lineusedcount%len(line_settings['linecolors'])
                    else:
                        lc_idx = lineusedcount
                    line_settings['linecolor'] = line_settings['linecolors'][lc_idx]
                else:
                    line_settings['linecolor'] = self.def_colors[defcol_idx]
            else: #case where first line, but linecolor isnt defined, but linecolorS is
                  #so it used default color INSTEAD of the desired colro...
                if 'linecolors' in line_settings.keys():
                    line_settings['linecolor'] = line_settings['linecolors'][0]

        if line_settings['drawpoints'].lower() == 'true':
            if lineusedcount > 0: #if more than one, the color specified is already used. Use a new color..
                if 'pointfillcolors' in line_settings.keys():
                    if isinstance(line_settings['pointfillcolors'], dict):
                        line_settings['pointfillcolors'] = [line_settings['pointfillcolors']['pointfillcolor']]
                    # pfc_idx = copy.copy(lineusedcount_idx)
                    if lineusedcount > len(line_settings['pointfillcolors']):
                        pfc_idx = lineusedcount%len(line_settings['pointfillcolors'])
                    else:
                        pfc_idx = lineusedcount
                    line_settings['pointfillcolor'] = line_settings['pointfillcolors'][pfc_idx]
                if 'pointlinecolors' in line_settings.keys():
                    if isinstance(line_settings['pointlinecolors'], dict):
                        line_settings['pointlinecolors'] = [line_settings['pointlinecolors']['pointlinecolor']]
                    if lineusedcount > len(line_settings['pointlinecolors']):
                        plc_idx = lineusedcount%len(line_settings['pointlinecolors'])
                    else:
                        plc_idx = lineusedcount
                    line_settings['pointlinecolor'] = line_settings['pointlinecolors'][plc_idx]

                if 'pointfillcolor' not in line_settings.keys():
                    if 'pointlinecolor' in line_settings.keys():
                        line_settings['pointfillcolor'] = line_settings['pointlinecolor']
                    else:
                        line_settings['pointfillcolor'] = self.def_colors[defcol_idx]

                if 'pointlinecolor' not in line_settings.keys():
                    if 'pointfillcolor' in line_settings.keys():
                        line_settings['pointlinecolor'] = line_settings['pointfillcolor']
                    else:
                        line_settings['pointlinecolor'] = self.def_colors[defcol_idx]

            else: #case where first line, but linecolor isnt defined, so it used default color...
                if 'pointfillcolors' in line_settings.keys():
                    if isinstance(line_settings['pointfillcolors'], dict):
                        line_settings['pointfillcolors'] = [line_settings['pointfillcolors']['pointfillcolor']]
                    line_settings['pointfillcolor'] = line_settings['pointfillcolors'][0]
                if 'pointlinecolors' in line_settings.keys():
                    if isinstance(line_settings['pointlinecolors'], dict):
                        line_settings['pointlinecolors'] = [line_settings['pointlinecolors']['pointlinecolor']]
                    line_settings['pointlinecolor'] = line_settings['pointlinecolors'][0]

        return line_settings

    def filterProfileData(self, data, line_settings, object_settings):
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
            print('UseDepth flag not set. Cannot filter properly.')
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

    def filterTableData(self, data, object_settings):

        xmax = None
        xmin = None
        ymax = None
        ymin = None

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
            line = data[lineflag]
            values = line['values']
            dates = line['dates']

            filtbylims = False
            if 'filterbylimits' in line.keys():
                if line['filterbylimits'].lower() == 'true':
                    filtbylims = True
            else:
                if 'filterbylimits' in object_settings.keys():
                    if object_settings['filterbylimits'].lower() == 'true':
                        filtbylims = True

            if 'omitvalue' in line.keys():
                omitvalue = float(line['omitvalue'])
            else:
                omitvalue = None

            if xmax != None and filtbylims:
                xmax_filt = np.where(values <= xmax)
            else:
                xmax_filt = np.arange(len(values))

            if xmin != None and filtbylims:
                xmin_filt = np.where(values >= xmin)
            else:
                xmin_filt = np.arange(len(values))

            if ymax != None and filtbylims:
                ymax_filt = np.where(dates <= ymax)
            else:
                ymax_filt = np.arange(len(dates))

            if ymin != None and filtbylims:
                ymin_filt = np.where(dates >= ymin)
            else:
                ymin_filt = np.arange(len(dates))

            if omitvalue != None:
                omitval_filt = np.where(values != omitvalue)
            else:
                omitval_filt = np.arange(len(values))

            master_filter = reduce(np.intersect1d, (xmax_filt, xmin_filt, ymax_filt, ymin_filt, omitval_filt))

            data[lineflag]['values'] = values[master_filter]
            data[lineflag]['dates'] = dates[master_filter]

        return data

    def updateFlaggedValues(self, settings, flaggedvalue, replacevalue):
        '''
        iterates and updates specific flagged values with a replacement value
        :param settings: dictionary, list or str settings
        :param flaggedvalue: flagged value to look for and replace
        :param replacevalue: value to replace flagged value with
        :return: updated settings
        '''

        if isinstance(settings, list):
            new_list = []
            for item in settings:
                item = self.updateFlaggedValues(item, flaggedvalue, replacevalue)
                new_list.append(item)
            return new_list

        if isinstance(settings, np.ndarray):
            new_list = []
            for item in settings:
                item = self.updateFlaggedValues(item, flaggedvalue, replacevalue)
                new_list.append(item)
            return np.asarray(new_list, dtype=settings.dtype)

        elif isinstance(settings, dict):
            for key in settings.keys():
                settings[key] = self.updateFlaggedValues(settings[key], flaggedvalue, replacevalue)
            return settings

        elif isinstance(settings, str):
            pattern = re.compile(re.escape(flaggedvalue), re.IGNORECASE)
            settings = pattern.sub(repr(replacevalue)[1:-1], settings) #this seems weird with [1:-1] but paths wont work otherwise
            return settings

        else:
            #this gets REALLY noisy.
            #lots is set up to not be replaceable, so uncomment at your own risk
            # print('Cannot set {0}'.format(flaggedvalue))
            # print('Input Not recognized type', settings)
            return settings

    def changeTimeSeriesInterval(self, times, values, Line_info):
        '''
        changes time series of time series data. If type is defined, use that to average data. default is INST-VAL
        :param times: list of times
        :param values: list of values
        :param Line_info: settings dictionary for line
        :return: new times and values
        '''

        convert_to_jdate = False

        if isinstance(times[0], (int, float)): #check for jdate, this is easier in dt..
            times = self.JDateToDatetime(times)
            convert_to_jdate = True

        if 'type' in Line_info.keys() and 'interval' not in Line_info.keys():
            print('Defined Type but no interval..')
            if convert_to_jdate:
                return self.DatetimeToJDate(times), values
            else:
                return times, values

        # INST-CUM, INST-VAL, PER-AVER, PER-CUM)
        if 'type' in Line_info:
            avgtype = Line_info['type'].upper()
        else:
            avgtype = 'INST-VAL'
            # avgtype = 'PER-AVER'

        if isinstance(values, dict):
            new_values = {}
            for key in values:
                new_times, new_values[key] = self.changeTimeSeriesInterval(times, values[key], Line_info)
        else:

            if 'interval' in Line_info:
                interval = Line_info['interval'].upper()
                pd_interval = self.getPandasTimeFreq(interval)
            else:
                print('No time interval Defined.')
                return times, values

            if avgtype == 'INST-VAL':
                #at the point in time, find intervals and use values
                if len(values.shape) == 1:
                    df = pd.DataFrame({'times': times, 'values': values})
                    df = df.set_index('times')
                    df = df.resample(pd_interval, origin='end_day').asfreq().fillna(method='bfill')
                    new_values = df['values'].to_numpy()
                    new_times = df.index.to_pydatetime()
                elif len(values.shape) == 2:
                    tvals = values.T #transpose so now were [distances, times]
                    new_values = []
                    for i in range(tvals.shape[0]):#for each depth profile..
                        df = pd.DataFrame({'times': times, 'values': tvals[i]})
                        df = df.set_index('times')
                        df = df.resample(pd_interval, origin='end_day').asfreq().fillna(method='bfill')
                        new_values.append(df['values'].to_numpy())
                        new_times = df.index.to_pydatetime()
                    new_values = np.asarray(new_values).T #transpose back..

            elif avgtype == 'INST-CUM':
                if len(values.shape) == 1:
                    df = pd.DataFrame({'times': times, 'values': values})
                    df = df.set_index('times')
                    df = df.cumsum(skipna=True).resample(pd_interval, origin='end_day').asfreq().fillna(method='bfill')
                    new_values = df['values'].to_numpy()
                    new_times = df.index.to_pydatetime()
                elif len(values.shape) == 2:
                    tvals = values.T #transpose so now were [distances, times]
                    new_values = []
                    for i in range(tvals.shape[0]):#for each depth profile..
                        df = pd.DataFrame({'times': times, 'values': tvals[i]})
                        df = df.set_index('times')
                        df = df.cumsum(skipna=True).resample(pd_interval, origin='end_day').asfreq().fillna(method='bfill')
                        new_values.append(df['values'].to_numpy())
                        new_times = df.index.to_pydatetime()
                    new_values = np.asarray(new_values).T #transpose back..

            elif avgtype == 'PER-AVER':
                #average over the period
                if len(values.shape) == 1:
                    df = pd.DataFrame({'times': times, 'values': values})
                    df = df.set_index('times')
                    df = df.resample(pd_interval, origin='end_day').mean().fillna(method='bfill')
                    new_values = df['values'].to_numpy()
                    new_times = df.index.to_pydatetime()
                elif len(values.shape) == 2:
                    tvals = values.T #transpose so now were [distances, times]
                    new_values = []
                    for i in range(tvals.shape[0]):#for each depth profile..
                        df = pd.DataFrame({'times': times, 'values': tvals[i]})
                        df = df.set_index('times')
                        df = df.resample(pd_interval, origin='end_day').mean().fillna(method='bfill')
                        new_values.append(df['values'].to_numpy())
                        new_times = df.index.to_pydatetime()
                    new_values = np.asarray(new_values).T #transpose back..

            elif avgtype == 'PER-CUM':
                #cum over the period
                if len(values.shape) == 1:
                    df = pd.DataFrame({'times': times, 'values': values})
                    df = df.set_index('times')
                    df = df.resample(pd_interval, origin='end_day').sum().fillna(method='bfill')
                    new_values = df['values'].to_numpy()
                    new_times = df.index.to_pydatetime()
                elif len(values.shape) == 2:
                    tvals = values.T #transpose so now were [distances, times]
                    new_values = []
                    for i in range(tvals.shape[0]):#for each depth profile..
                        df = pd.DataFrame({'times': times, 'values': tvals[i]})
                        df = df.set_index('times')
                        df = df.resample(pd_interval, origin='end_day').sum().fillna(method='bfill')
                        new_values.append(df['values'].to_numpy())
                        new_times = df.index.to_pydatetime()
                    new_values = np.asarray(new_values).T #transpose back..

            else:
                print('INVALID INPUT TYPE DETECTED', avgtype)
                return times, values

        if convert_to_jdate:
            return self.DatetimeToJDate(new_times), np.asarray(new_values)
        else:
            return new_times, np.asarray(new_values)

    def checkModelType(self, line_info):
        modeltype = self.getLineModelType(line_info)
        if modeltype == 'undefined':
            return True
        if modeltype.lower() != self.plugin.lower():
            return False
        return True

    def makeRegularTimesteps(self, days=15):
        '''
        makes regular time series for profile plots if there are no times defined
        :param days: day interval
        :return: timestep list
        '''

        timesteps = []
        print('No Timesteps found. Setting to Regular interval')
        cur_date = self.StartTime
        while cur_date < self.EndTime:
            timesteps.append(cur_date)
            cur_date += dt.timedelta(days=days)
        return np.asarray(timesteps)

    def convertDepthsToElevations(self, data, object_settings):
        '''
        handles data to convert depths into elevations for observed data
        :param object_settings: dicitonary of user defined settings for current object
        :return: object settings dictionary with updated elevation data
        '''

        elev_flag = 'NOVALID'
        if object_settings['usedepth'].lower() == 'false':
            for ld in data.keys():
                if data[ld]['elevations'] == []:
                    noelev_flag = ld
                    for old in data.keys():
                        if len(data[old]['elevations']) > 0:
                            elev_flag = old
                            break

                    if elev_flag != 'NOVALID':
                        data[noelev_flag]['elevations'] = WF.convertObsDepths2Elevations(data[noelev_flag]['depths'],
                                                                                         data[elev_flag]['elevations'])
                    else:
                        object_settings['usedepth'] = 'true'
        return data, object_settings

    def commitProfileDataToMemory(self, data, line_settings, object_settings):
        '''
        commits updated data to data memory dictionary that keeps track of data
        :param object_settings:  dicitonary of user defined settings for current object
        '''
        for line in data.keys():
            values = pickle.loads(pickle.dumps(data[line]['values'], -1))
            depths = pickle.loads(pickle.dumps(data[line]['depths'], -1))
            elevations = pickle.loads(pickle.dumps(data[line]['elevations'], -1))
            datamem_key = line_settings[line]['logoutputfilename']
            if datamem_key not in self.Data_Memory.keys():
                self.Data_Memory[datamem_key] = {'times': object_settings['timestamps'],
                                                 'values': values,
                                                 'elevations': elevations,
                                                 'depths': depths,
                                                 'units': object_settings['plot_units'],
                                                 'isprofile': True}

    def commitContourDataToMemory(self, values, dates, distance, units, datamem_key):
        '''
        commits updated data to data memory dictionary that keeps track of data
        :param object_settings:  dicitonary of user defined settings for current object
        '''

        self.Data_Memory[datamem_key] = {'times': dates,
                                         'values': values,
                                         'distance': distance,
                                         'units': units,
                                         'iscontour': True}

    def configureUnits(self, object_settings, parameter, units):
        '''
        configure units from line settings
        :param object_settings:  dicitonary of user defined settings for current object
        :param line: current line settings
        :param units: current units of line
        :return: units
        '''

        if units == None:
            try:
                units = self.units[parameter.lower()]
            except KeyError:
                units = None

        if isinstance(units, dict):
            if 'unitsystem' in object_settings.keys():
                units = units[object_settings['unitsystem'].lower()]
            else:
                units = None
        return units

    def configureSettingsForID(self, ID, settings):
        self.loadCurrentID(ID)
        self.loadCurrentModelAltID(ID)
        settings = self.replaceflaggedValues(settings, 'modelspecific')
        return settings

    def confirmColor(self, user_color, default_color):

        if not is_color_like(user_color):
            if not is_color_like(user_color.replace(' ', '')):
                print('Invalid pointfillcolor with {0}'.format(user_color))
                print('Replacing with default color')
                return default_color
            else:
                print('Misspelling in pointfillcolor with {0}'.format(user_color))
                print('Replacing with {0}'.format(user_color.replace(' ', '')))
                return user_color.replace(' ', '')
        else:
            return user_color

    def confirm_axis(self, object_settings):
        if 'axs' not in object_settings.keys():
            object_settings['axs'] = [{}] #empty axis object
        return object_settings

if __name__ == '__main__':
    rundir = sys.argv[0]
    simInfoFile = sys.argv[1]
    # import cProfile
    # ar = cProfile.run('MakeAutomatedReport(simInfoFile, rundir)')
    MakeAutomatedReport(simInfoFile, rundir)

