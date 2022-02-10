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

import datetime as dt
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import calendar
import re
from collections import Counter
import shutil
from scipy import interpolate
from functools import reduce
import pickle
import pendulum

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
        if self.reportType == 'single': #Eventually be able to do comparison reports, put that here
            for simulation in self.Simulations:
                print('Running Simulation:', simulation)
                self.setSimulationVariables(simulation)
                self.defineStartEndYears()
                self.readSimulationsCSV() #read to determine order/sims/regions in report
                self.cleanOutputDirs()
                self.initializeXML()
                self.writeXMLIntroduction()
                for simorder in self.SimulationCSV.keys():
                    self.setSimulationCSVVars(self.SimulationCSV[simorder])
                    self.readDefinitionsFile(self.SimulationCSV[simorder])
                    self.loadModelAlt(self.SimulationCSV[simorder])
                    self.addSimLogEntry()
                    self.writeChapter()
                self.XML.writeReportEnd()
                self.equalizeLog()
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

        self.Log['observed_data_path'].append(self.observedDir)
        self.Log['start_time'].append(self.StartTimeStr)
        self.Log['end_time'].append(self.EndTimeStr)
        self.Log['compute_time'].append(self.LastComputed)
        self.Log['program'].append(self.plugin)
        self.Log['alternative_name'].append(self.modelAltName)
        self.Log['fpart'].append(self.alternativeFpart)
        self.Log['program_directory'].append(self.alternativeDirectory)

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

        object_settings = self.updateFlaggedValues(object_settings, '%%year%%', self.years_str)

        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot()
        unitslist = []
        for line in object_settings['lines']:

            parameter, object_settings['param_count'] = self.getParameterCount(line, object_settings)
            i = object_settings['param_count'][parameter]

            line['logoutputfilename'] = self.buildFileName(line)

            dates, values, units = self.getTimeSeries(line)

            if units == None:
                if parameter != None:
                    try:
                        units = self.units[parameter]
                    except KeyError:
                        units = None

            if isinstance(units, dict):
                if 'unitsystem' in object_settings.keys():
                    units = units[object_settings['unitsystem'].lower()]
                else:
                    units = None

            if 'unitsystem' in object_settings.keys():
                values, units = self.convertUnitSystem(values, units, object_settings['unitsystem'])

            chkvals = WF.checkData(values)
            if not chkvals:
                print('Invalid Data settings for line:', line)
                continue

            if 'dateformat' in object_settings.keys():
                if object_settings['dateformat'].lower() == 'jdate':
                    if isinstance(dates[0], dt.datetime):
                        dates = self.DatetimeToJDate(dates)
                elif object_settings['dateformat'].lower() == 'datetime':
                    if isinstance(dates[0], (int,float)):
                        dates = self.JDateToDatetime(dates)

            if units != '' and units != None:
                unitslist.append(units)

            line_settings = self.getDefaultLineSettings(line, parameter, i)

            if 'zorder' not in line_settings.keys():
                line_settings['zorder'] = 4

            if 'label' not in line_settings.keys():
                line_settings['label'] = ''

            if 'filterbylimits' not in line_settings.keys():
                line_settings['filterbylimits'] = 'true' #set default

            if line_settings['filterbylimits'].lower() == 'true':
                if 'xlims' in object_settings.keys():
                    dates, values = self.limitXdata(dates, values, object_settings['xlims'])
                if 'ylims' in object_settings.keys():
                    dates, values = self.limitYdata(dates, values, object_settings['ylims'])

            if line_settings['drawline'].lower() == 'true' and line_settings['drawpoints'].lower() == 'true':
                ax.plot(dates, values, label=line_settings['label'], c=line_settings['linecolor'],
                        lw=line_settings['linewidth'], ls=line_settings['linestylepattern'],
                        marker=line_settings['symboltype'], markerfacecolor=line_settings['pointfillcolor'],
                        markeredgecolor=line_settings['pointlinecolor'], markersize=float(line_settings['symbolsize']),
                        markevery=int(line_settings['numptsskip']), zorder=float(line_settings['zorder']),
                        alpha=float(line_settings['alpha']))

            elif line_settings['drawline'].lower() == 'true':
                ax.plot(dates, values, label=line_settings['label'], c=line_settings['linecolor'],
                        lw=line_settings['linewidth'], ls=line_settings['linestylepattern'],
                        zorder=float(line_settings['zorder']),
                        alpha=float(line_settings['alpha']))

            elif line_settings['drawpoints'].lower() == 'true':
                ax.scatter(dates[::int(line_settings['numptsskip'])], values[::int(line_settings['numptsskip'])],
                           marker=line_settings['symboltype'], facecolor=line_settings['pointfillcolor'],
                           edgecolor=line_settings['pointlinecolor'], s=float(line_settings['symbolsize']),
                           label=line_settings['label'], zorder=float(line_settings['zorder']),
                           alpha=float(line_settings['alpha']))


            self.addLogEntry({'type': line_settings['label'] + '_TimeSeries' if line_settings['label'] != '' else 'Timeseries',
                              'name': self.ChapterRegion,
                              'description': object_settings['description'],
                              'units': units,
                              'value_start_date': self.translateDateFormat(dates[0], 'datetime', '').strftime('%d %b %Y'),
                              'value_end_date': self.translateDateFormat(dates[-1], 'datetime', '').strftime('%d %b %Y'),
                              'logoutputfilename': line['logoutputfilename']
                              },
                             isdata=True)

        object_settings['units_list'] = unitslist
        plotunits = self.getPlotUnits(object_settings)
        object_settings = self.updateFlaggedValues(object_settings, '%%units%%', plotunits)

        if 'title' in object_settings.keys():
            if 'titlesize' in object_settings.keys():
                titlesize = float(object_settings['titlesize'])
            elif 'fontsize' in object_settings.keys():
                titlesize = float(object_settings['fontsize'])
            else:
                titlesize = 15
            plt.title(object_settings['title'], fontsize=titlesize)

        if 'gridlines' in object_settings.keys():
            if object_settings['gridlines'].lower() == 'true':
                plt.grid(True)

        if 'ylabel' in object_settings.keys():
            if 'ylabelsize' in object_settings.keys():
                ylabsize = float(object_settings['ylabelsize'])
            elif 'fontsize' in object_settings.keys():
                ylabsize = float(object_settings['fontsize'])
            else:
                ylabsize = 12
            plt.ylabel(object_settings['ylabel'], fontsize=ylabsize)

        if 'xlabel' in object_settings.keys():
            if 'xlabelsize' in object_settings.keys():
                xlabsize = float(object_settings['xlabelsize'])
            elif 'fontsize' in object_settings.keys():
                xlabsize = float(object_settings['fontsize'])
            else:
                xlabsize = 12
            plt.xlabel(object_settings['xlabel'], fontsize=xlabsize)

        if 'legend' in object_settings.keys():
            if object_settings['legend'].lower() == 'true':
                if 'legendsize' in object_settings.keys():
                    legsize = float(object_settings['legendsize'])
                elif 'fontsize' in object_settings.keys():
                    legsize = float(object_settings['fontsize'])
                else:
                    legsize = 12
                plt.legend(fontsize=legsize)

        self.formatDateXAxis(ax, object_settings)

        if 'ylims' in object_settings.keys():
            if 'min' in object_settings['ylims']:
                ax.set_ylim(bottom=float(object_settings['ylims']['min']))
            if 'max' in object_settings['ylims']:
                ax.set_ylim(top=float(object_settings['ylims']['max']))

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

        basefigname = os.path.join(self.images_path, 'TimeSeriesPlot' + '_' + self.ChapterRegion.replace(' ','_'))
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
        object_settings['usedepth'] = 'true'

        ################# Get timestamps #################
        object_settings['datessource_flag'] = self.getDateSourceFlag(object_settings)
        object_settings['timestamps'] = self.getProfileTimestamps(object_settings)

        ################# Get units #################
        object_settings['plot_parameter'] = self.getPlotParameter(object_settings)

        ################# Get data #################
        object_settings['linedata'] = self.getProfileData(object_settings)

        ################# Get plot units #################
        object_settings = self.convertProfileDataUnits(object_settings)
        object_settings['units_list'] = self.getUnitsList(object_settings)
        object_settings['plot_units'] = self.getPlotUnits(object_settings)

        object_settings = self.updateFlaggedValues(object_settings, '%%units%%', object_settings['plot_units'])

        object_settings['resolution'] = self.getProfileInterpResolution(object_settings)
        object_settings = self.filterProfileData(object_settings)


        object_settings['split_by_year'], object_settings['years'], object_settings['yearstr'] = self.getPlotYears(object_settings)

        #start data loop
        for yi, year in enumerate(object_settings['years']):
            if object_settings['split_by_year']:
                yearstr = str(year)
            else:
                yearstr = object_settings['yearstr']


            object_desc = self.updateFlaggedValues(object_settings['description'], '%%year%%', yearstr)
            self.XML.writeTableStart(object_desc, 'Statistics')
            yrheaders = self.buildHeadersByTimestamps(object_settings['timestamps'], year=year)
            yrheaders = self.convertHeaderFormats(yrheaders, object_settings)

            for i, header in enumerate(yrheaders):
                frmt_rows = []
                for row in object_settings['rows']:
                    s_row = row.split('|')
                    rowname = s_row[0]
                    row_val = s_row[1]
                    if '%%' in row_val:
                        data = self.formatStatsProfileLineData(row, object_settings['linedata'],
                                                               object_settings['resolution'], i)
                        row_val, stat = self.getStatsLine(row_val, data)
                        self.addLogEntry({'type': 'ProfileTableStatistic',
                                          'name': ' '.join([self.ChapterRegion, header, stat]),
                                          'description': object_desc,
                                          'value': row_val,
                                          'function': stat,
                                          'units': object_settings['plot_units'],
                                          'value_start_date': header,
                                          'value_end_date': header,
                                          'logoutputfilename': ', '.join([object_settings['linedata'][flag]['logoutputfilename'] for flag in object_settings['linedata']])
                                          },
                                         isdata=True)
                    header = '' if header == None else header
                    frmt_rows.append('{0}|{1}'.format(rowname, row_val))
                self.XML.writeTableColumn(header, frmt_rows)
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
        object_settings['linedata'] = self.getProfileData(object_settings)

        ################# Get plot units #################
        object_settings = self.convertProfileDataUnits(object_settings)
        object_settings['units_list'] = self.getUnitsList(object_settings)
        object_settings['plot_units'] = self.getPlotUnits(object_settings)
        object_settings = self.updateFlaggedValues(object_settings, '%%units%%', object_settings['plot_units'])

        ################ convert Elevs ################
        object_settings = self.convertDepthsToElevations(object_settings)

        self.commitProfileDataToMemory(object_settings)
        object_settings = self.filterProfileData(object_settings)

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

            for page_i, pgi in enumerate(page_indices):

                subplot_rows, subplot_cols = WF.getSubplotConfig(len(pgi), int(cur_obj_settings['profilesperrow']))
                n_nrow_active = np.ceil(len(pgi) / subplot_cols)
                fig = plt.figure(figsize=(7, 1 + 3 * n_nrow_active))

                current_object_settings = self.updateFlaggedValues(cur_obj_settings, '%%year%%', yearstr) #TODO: reudce the settings

                for i, j in enumerate(pgi):
                    ax = fig.add_subplot(int(subplot_rows), int(subplot_cols), i + 1)

                    if object_settings['usedepth'].lower() == 'true':
                        ax.invert_yaxis()

                    for li, line in enumerate(object_settings['linedata'].keys()):

                        try:
                            values = object_settings['linedata'][line]['values'][j]
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
                                levels = object_settings['linedata'][line]['depths'][j][msk]
                            else:
                                levels = object_settings['linedata'][line]['elevations'][j][msk]
                            if not WF.checkData(levels):
                                print('Non Viable depths/elevations for {0} on {1}'.format(line, object_settings['timestamps'][j]))
                                continue
                        except IndexError:
                            print('Non Viable depths/elevations for {0} on {1}'.format(line, object_settings['timestamps'][j]))
                            continue

                        if not WF.checkData(values):
                            continue

                        current_ls = self.getLineSettings(object_settings['lines'], line)
                        current_ls = self.getDefaultLineSettings(current_ls, object_settings['plot_parameter'], li)

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

                    show_xlabel, show_ylabel = self.getPlotLabelMasks(i, len(pgi), subplot_cols)

                    if current_object_settings['gridlines'].lower() == 'true':
                        ax.grid(zorder=0)

                    if show_ylabel:
                        if 'ylabel' in current_object_settings.keys():
                            if 'ylabelsize' in object_settings.keys():
                                ylabsize = float(object_settings['ylabelsize'])
                            elif 'fontsize' in object_settings.keys():
                                ylabsize = float(object_settings['fontsize'])
                            else:
                                ylabsize = 12
                            ax.set_ylabel(current_object_settings['ylabel'], fontsize=ylabsize)

                    if show_xlabel:
                        if 'xlabel' in current_object_settings.keys():
                            if 'xlabelsize' in object_settings.keys():
                                xlabsize = float(object_settings['xlabelsize'])
                            elif 'fontsize' in object_settings.keys():
                                xlabsize = float(object_settings['fontsize'])
                            else:
                                xlabsize = 12
                            ax.set_xlabel(current_object_settings['xlabel'], fontsize=xlabsize)

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

                    xbufr = 0.05
                    ybufr = 0.05
                    xl = ax.get_xlim()
                    yl = ax.get_ylim()
                    xtext = xl[0] + xbufr * (xl[1] - xl[0])
                    ytext = yl[1] - ybufr * (yl[1] - yl[0])
                    ax.text(xtext, ytext, ttl_str, ha='left', va='top', size=10, #TODO: make this variable
                            bbox=dict(boxstyle='round',facecolor='w', alpha=0.35),
                            zorder=10)

                plt.tight_layout()

                if 'legend' in current_object_settings.keys():
                    if current_object_settings['legend'].lower() == 'true':
                        if len(ax.get_legend_handles_labels()[1]) > 0:
                            if 'legendsize' in current_object_settings.keys():
                                legsize = float(current_object_settings['legendsize'])
                            elif 'fontsize' in current_object_settings.keys():
                                legsize = float(current_object_settings['fontsize'])
                            else:
                                legsize = 12

                            ncolumns = 3

                            n_legends_row = np.ceil(len(object_settings['linedata'].keys()) / ncolumns) * .65
                            if n_legends_row < 1:
                                n_legends_row = 1
                            plt.subplots_adjust(bottom=(.3/n_nrow_active)*n_legends_row)
                            plt.legend(bbox_to_anchor=(.5,0), loc="lower center", fontsize=legsize,
                                       bbox_transform=fig.transFigure, ncol=ncolumns)

                # plt.tight_layout()
                figname = 'ProfilePlot_{0}_{1}_{2}_{3}_{4}.png'.format(self.ChapterName, yearstr,
                                                                       object_settings['plot_parameter'], self.plugin,
                                                                       page_i)

                # plt.savefig(os.path.join(self.images_path, figname), dpi=600)
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
                                  'logoutputfilename': ', '.join([object_settings['linedata'][flag]['logoutputfilename'] for flag in object_settings['linedata']])
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

        object_settings['unitslist'] = []

        object_settings['data'], object_settings['units_list'] = self.getTableData(object_settings)

        plotunits = self.getPlotUnits(object_settings)
        object_settings = self.updateFlaggedValues(object_settings, '%%units%%', plotunits)

        headings = self.buildHeadersByYear(object_settings, object_settings['years'], object_settings['split_by_year'])
        rows = self.buildRowsByYear(object_settings, object_settings['years'], object_settings['split_by_year'])

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
                    rowdata, sr_month = self.getStatsLineData(row_val, object_settings['data'], year=year)
                    row_val, stat = self.getStatsLine(row_val, rowdata)

                    data_start_date, data_end_date = self.getTableDates(year, object_settings)
                    self.addLogEntry({'type': 'Statistic',
                                      'name': ' '.join([self.ChapterRegion, header, stat]),
                                      'description': desc,
                                      'value': row_val,
                                      'function': stat,
                                      'units': plotunits,
                                      'value_start_date': self.translateDateFormat(data_start_date, 'datetime', ''),
                                      'value_end_date': self.translateDateFormat(data_end_date, 'datetime', ''),
                                      'logoutputfilename': ', '.join([object_settings['data'][flag]['logoutputfilename'] for flag in object_settings['data']])
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

        object_settings['unitslist'] = []

        object_settings['data'], object_settings['units_list'] = self.getTableData(object_settings)

        plotunits = self.getPlotUnits(object_settings)
        object_settings = self.updateFlaggedValues(object_settings, '%%units%%', plotunits)

        headings = self.buildHeadersByYear(object_settings, object_settings['years'], object_settings['split_by_year'])
        rows = self.buildRowsByYear(object_settings, object_settings['years'], object_settings['split_by_year'])

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
                    rowdata, sr_month = self.getStatsLineData(row_val, object_settings['data'], year=year)

                    row_val, stat = self.getStatsLine(row_val, rowdata)

                    data_start_date, data_end_date = self.getTableDates(year, object_settings, month=sr_month)
                    self.addLogEntry({'type': 'Statistic',
                                      'name': ' '.join([self.ChapterRegion, header, stat]),
                                      'description': object_settings['description'],
                                      'value': row_val,
                                      'units': plotunits,
                                      'function': stat,
                                      'value_start_date': self.translateDateFormat(data_start_date, 'datetime', ''),
                                      'value_end_date': self.translateDateFormat(data_end_date, 'datetime', ''),
                                      'logoutputfilename': ', '.join([object_settings['data'][flag]['logoutputfilename'] for flag in object_settings['data']])
                                      },
                                     isdata=True)

                header = '' if header == None else header
                frmt_rows.append('{0}|{1}'.format(rowname, row_val))
            self.XML.writeTableColumn(header, frmt_rows)
        self.XML.writeTableEnd()

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
        for i, line in enumerate(object_settings['lines']):
            line['logoutputfilename'] = self.buildFileName(line)
            dates, values, units = self.getTimeSeries(line)

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

    def setSimulationDateTimes(self):
        '''
        sets the simulation start time and dates from string format. If timestamp says 24:00, converts it to be correct
        Datetime format of the next day at 00:00
        :return: class varables
                    self.StartTime
                    self.EndTime
        '''

        if '24:00' in self.StartTimeStr:
            tstrtmp = (self.StartTimeStr).replace('24:00', '23:00')
            self.StartTime = dt.datetime.strptime(tstrtmp, '%d %B %Y, %H:%M')
            self.StartTime += dt.timedelta(hours=1)
        else:
            self.StartTime = dt.datetime.strptime(self.StartTimeStr, '%d %B %Y, %H:%M')

        if '24:00' in self.EndTimeStr:
            tstrtmp = (self.EndTimeStr).replace('24:00', '23:00')
            self.EndTime = dt.datetime.strptime(tstrtmp, '%d %B %Y, %H:%M')
            self.EndTime += dt.timedelta(hours=1)
        else:
            self.EndTime = dt.datetime.strptime(self.EndTimeStr, '%d %B %Y, %H:%M')

    def setSimulationCSVVars(self, simlist):
        '''
        set variables pertaining to a specified simulation
        :param simlist: dictionary of specified simulation
        :return: class variables
                    self.plugin
                    self.modelAltName
                    self.defFile
        '''

        self.plugin = simlist['plugin']
        self.modelAltName = simlist['modelaltname']
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

        self.Data_Memory = {}
        self.SimulationName = simulation['name']
        self.baseSimulationName = simulation['basename']
        self.simulationDir = simulation['directory']
        self.DSSFile = simulation['dssfile']
        self.StartTimeStr = simulation['starttime']
        self.EndTimeStr = simulation['endtime']
        self.LastComputed = simulation['lastcomputed']
        self.ModelAlternatives = simulation['modelalternatives']
        self.setSimulationDateTimes()

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

        # if param != 'unknown':
        #     param = param.lower()
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

            LineSettings = self.translateLineStylePatterns(LineSettings) #TODO: convert colors?

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
            try:
                if int(LineSettings['numptsskip']) == 0:
                    LineSettings['numptsskip'] = 1
            except ValueError:
                print('Invalid setting for numptsskip.', LineSettings['numptsskip'])
                print('defaulting to 25')
                LineSettings['numptsskip'] = 25

            LineSettings = self.translatePointStylePatterns(LineSettings) #TODO: convert colors?

        return LineSettings

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

    def getDefaultDefaultLineStyles(self, i):
        '''
        creates a default line style based off of the number line and default colors
        used if param is undefined or not in defaults file
        :param i: count of line on the plot
        :return: dictionary with line settings
        '''

        if i >= len(self.def_colors):
            i = i - len(self.def_colors)
        return {'linewidth': 2, 'linecolor': self.def_colors[i], 'linestyle': 'solid', 'alpha': 1.0}

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
                return [], [], None
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
                    return [], [], None
                elif len(values) == 0:
                    return [], [], None

        elif 'w2_file' in Line_info.keys():
            datamem_key = self.buildDataMemoryKey(Line_info)
            if datamem_key in self.Data_Memory.keys():
                print('READING {0} FROM MEMORY'.format(datamem_key))
                datamementry = pickle.loads(pickle.dumps(self.Data_Memory[datamem_key], -1))
                times = datamementry['times']
                values = datamementry['values']
                units = datamementry['units']
                # times = pickle.loads(pickle.dumps(self.Data_Memory[datamem_key]['times'], -1))
                # values = pickle.loads(pickle.dumps(self.Data_Memory[datamem_key]['values'], -1))
                # units = pickle.loads(pickle.dumps(self.Data_Memory[datamem_key]['units'], -1))

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
                # times = pickle.loads(pickle.dumps(self.Data_Memory[datamem_key]['times'], -1))
                # values = pickle.loads(pickle.dumps(self.Data_Memory[datamem_key]['values'], -1))
                # units = pickle.loads(pickle.dumps(self.Data_Memory[datamem_key]['units'], -1))
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
            return [], [], None

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

        if 'filename' in Line_info.keys(): #Get data from Observed
            filename = Line_info['filename']
            values, depths, times = WDR.readTextProfile(filename, timesteps)
            # times = WDR.getTextProfileDates(filename, self.StartTime, self.EndTime)
            return values, [], depths, times, Line_info['flag']

        elif 'w2_segment' in Line_info.keys():
            vals, elevations, depths, times = self.ModelAlt.readProfileData(Line_info['w2_segment'], timesteps)
            return vals, elevations, depths, times, Line_info['flag']

        elif 'ressimresname' in Line_info.keys():
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

        linedata = {}
        for line in object_settings[object_settings['datakey']]:
            vals, elevations, depths, times, flag = self.getProfileValues(line, object_settings['timestamps']) #Test this speed for grabbing all profiles and then choosing
            datamem_key = self.buildDataMemoryKey(line)
            if 'units' in line.keys():
                units = line['units']
            else:
                units = None
            linedata[flag] = {'values': vals,
                              'elevations': elevations,
                              'depths': depths,
                              'times': times,
                              'units': units,
                              'logoutputfilename': datamem_key}

            for key in line.keys():
                if key not in linedata[flag].keys():
                    linedata[flag][key] = line[key]

        return linedata

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

    def getPlotUnits(self, object_settings):
        '''
        gets units for the plot. Either looks at data already plotted units, or if there are no defined units
        in the plotted data, look for a parameter flag
        :param object_settings: dictionary with plot settings
        :return: string units value
        '''

        if len(object_settings['units_list']) > 0:
            plotunits = self.getMostCommon(object_settings['units_list'])
        elif 'parameter' in object_settings.keys():
            try:
                plotunits = self.units[object_settings['parameter'].lower()]
                if isinstance(plotunits, dict):
                    if 'unitsystem' in object_settings.keys():
                        plotunits = plotunits[object_settings['unitsystem'].lower()]
                    else:
                        plotunits = plotunits['metric']
            except KeyError:
                plotunits = ''
        else:
            print('No units defined.')
            plotunits = ''

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
        units_list = []
        for dp in object_settings['datapaths']:
            dp['logoutputfilename'] = self.buildFileName(dp)
            dates, values, units = self.getTimeSeries(dp)

            units = self.configureUnits(object_settings, dp, units)

            if 'unitsystem' in object_settings.keys():
                values, units = self.convertUnitSystem(values, units, object_settings['unitsystem'])

            if 'filterbylimits' not in dp.keys():
                dp['filterbylimits'] = 'true' #set default

            if dp['filterbylimits'].lower() == 'true':
                if 'xlims' in object_settings.keys():
                    dates, values = self.limitXdata(dates, values, object_settings['xlims'])
                if 'ylims' in object_settings.keys():
                    dates, values = self.limitYdata(dates, values, object_settings['ylims'])

            if units != None:
                units_list.append(units)

            data[dp['flag']] = {'dates': dates,
                                'values': values,
                                'units': units,
                                'logoutputfilename': dp['logoutputfilename']}
        return data, units_list

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

    def getUnitsList(self, object_settings):
        '''
        creates a list of units from defined lines in user defined settings
        :param object_settings: currently selected object settings dictionary
        :return: units_list: list of used units
        '''

        units_list = []
        for flag in object_settings['linedata'].keys():
            units = object_settings['linedata'][flag]['units']
            if units != None:
                units_list.append(units)
        return units_list

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

        default_settings = pickle.loads(pickle.dumps(self.replaceflaggedValues(default_settings), -1))
        object_settings = pickle.loads(pickle.dumps(self.replaceflaggedValues(object_settings), -1))

        for key in object_settings.keys():
            if key not in default_settings.keys(): #if defaults doesnt have key
                default_settings[key] = object_settings[key]
            elif default_settings[key] == None: #if defaults has key, but is none
                default_settings[key] = object_settings[key]
            elif isinstance(object_settings[key], list): #if settings is a list, aka rows or lines
                if key.lower() == 'rows': #if the default has rows defined, just overwrite them.
                    if key in default_settings.keys():
                        default_settings[key] = object_settings[key]
                elif key.lower() not in default_settings.keys():
                    default_settings[key] = object_settings[key] #if the defaults dont have anything defined, fill it in
                else:
                    for item in object_settings[key]:
                        if isinstance(item, dict):
                            if 'flag' in item.keys(): #if we flag line
                                flag_match = False
                                for defaultitem in default_settings[key]:
                                    if 'flag' in defaultitem.keys():
                                        if defaultitem['flag'].lower() == item['flag'].lower(): #matching flags!
                                            flag_match = True
                                            for subkey in item.keys(): #for each settings defined, overwrite
                                                defaultitem[subkey] = item[subkey]
                                if not flag_match:
                                    default_settings[key].append(item)
                        if isinstance(item, str):
                            default_settings[key] = object_settings[key] #replace string with list, ex datessource
                            break
            else:
                default_settings[key] = object_settings[key]

        return default_settings

    def replaceflaggedValues(self, settings):
        '''
        recursive function to replace flagged values in settings
        :param settings: dict, list or string containing settings, potentially with flags
        :return:
            settings: dict, list or string with flags replaced
        '''

        if isinstance(settings, str):
            if '%%' in settings:
                newval = self.replaceFlaggedValue(settings)
                settings = newval
        elif isinstance(settings, dict):
            for key in settings.keys():
                if settings[key] == None:
                    continue
                elif isinstance(settings[key], dict):
                    settings[key] = self.replaceflaggedValues(settings[key])
                elif isinstance(settings[key], list):
                    new_list = []
                    for item in settings[key]:
                        new_list.append(self.replaceflaggedValues(item))
                    settings[key] = new_list
                else:
                    if '%%' in settings[key]:
                        newval = self.replaceFlaggedValue(settings[key])
                        settings[key] = newval
        elif isinstance(settings, list):
            for i, item in enumerate(settings):
                if '%%' in item:
                    settings[i] = self.replaceFlaggedValue(item)

        return settings

    def replaceFlaggedValue(self, value):
        '''
        replaces strings with flagged values with known paths
        flags are now case insensitive with more intelligent matching. yay.
        needs to use '[1:-1]' for paths, otherwise things like /t in a path C:/trains will be taken literal
        :param value: string potentially containing flagged value
        :return:
            value: string with potential flags replaced
        '''

        flagged_values = {'%%ModelDSS%%': self.DSSFile,
                          '%%region%%': self.ChapterRegion,
                          '%%Fpart%%': self.alternativeFpart,
                          '%%plugin%%': self.plugin,
                          '%%modelAltName%%': self.modelAltName,
                          '%%SimulationName%%': self.SimulationName,
                          '%%SimulationDir%%': self.simulationDir,
                          '%%baseSimulationName%%': self.baseSimulationName,
                          '%%starttime%%': self.StartTimeStr,
                          '%%endtime%%': self.EndTimeStr,
                          '%%LastComputed%%': self.LastComputed,
                          '%%observedDir%%': self.observedDir,
                          '%%startyear%%': str(self.startYear),
                          '%%endyear%%': str(self.endYear)
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
                    # lim_frmt = dateutil.parser.parse(lim) #try simple date formatting.
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
                if isinstance(lim, dt.datetime):
                    try:
                        lim_frmt = pendulum.parse(lim, strict=False).replace(tzinfo=None)
                        print('Datetime {0} as {1} Accepted!'.format(lim, lim_frmt))
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
        # flags = []
        for sr in s_row:
            if sr in data_dict.keys():
                curflag = sr
                curvalues = np.array(data_dict[sr]['values'])
                curdates = np.array(data_dict[sr]['dates'])
                data[curflag] = {'values': curvalues, 'dates': curdates}
                # flags.append(sr)
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

                        # msk = [i for i, n in enumerate(curdates) if n.month == sr_month]
                        months = [n.month for n in curdates]
                        msk = np.where(months==sr_month)
                        data[curflag]['values'] = curvalues[msk]
                        data[curflag]['dates'] = curdates[msk]

        if year != 'ALL':
            for flag in data.keys():
                msk = [i for i, n in enumerate(data[flag]['dates']) if n.year == year]
                data[flag]['values'] = np.asarray(data[flag]['values'])[msk]
                data[flag]['dates'] = np.asarray(data[flag]['dates'])[msk]

        return data, sr_month

    def formatStatsProfileLineData(self, row, data_dict, resolution, index):
        '''
        formats Profile line statistics for table using user inputs
        finds the highest and lowest overlapping profile points and uses them as end points, then interpolates
        :param row: Row line from inputs. String seperated by '|' and using flags surrounded by '%%'
        :param data_dict: dictionary containing available line data to be used
        :param resolution: number of values to interpolate to. this way each dataset has values at the same levels
                            and there is enough data to do stats over.
        :param index: date index for profile to use
        :return:
            out_data: dictionary containing values and depths/elevations
        '''

        # data = pickle.loads(pickle.dumps(data_dict, -1))
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
        #build elev profiles
        output_interp_elevations = np.arange(top, bottom, (bottom-top)/float(resolution))

        for flag in flags:
            out_data[flag] = {}
            #interpolate over all values and then get interp values
            # try:
            f_interp = interpolate.interp1d(data_dict[flag]['depths'][index], data_dict[flag]['values'][index], fill_value='extrapolate')
            out_data[flag]['depths'] = output_interp_elevations
            out_data[flag]['values'] = f_interp(output_interp_elevations)
            # except ValueError:
            #     print('Cannot interpolate depths for', flag)
            #     out_data[flag]['depths'] = output_interp_elevations
            #     out_data[flag]['values'] = np.full(len(output_interp_elevations), np.nan)

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

        flags = [n for n in data.keys()]

        if row.lower().startswith('%%meanbias'):
            out_stat = WF.calcMeanBias(data[flags[0]], data[flags[1]])
            stat = 'meanbias'
        elif row.lower().startswith('%%mae'):
            out_stat = WF.calcMAE(data[flags[0]], data[flags[1]])
            stat = 'mae'
        elif row.lower().startswith('%%rmse'):
            out_stat = WF.calcRMSE(data[flags[0]], data[flags[1]])
            stat = 'rmse'
        elif row.lower().startswith('%%nse'):
            out_stat = WF.calcNSE(data[flags[0]], data[flags[1]])
            stat = 'nse'
        elif row.lower().startswith('%%count'):
            out_stat = WF.getCount(data[flags[0]])
            stat = 'count'
        elif row.lower().startswith('%%mean'):
            out_stat = WF.calcMean(data[flags[0]])
            stat = 'mean'
        else:
            if '%%' in row:
                print('Unable to convert flag in row', row)
            return row, ''

        return out_stat, stat

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
            outname = '{0}'.format(self.ModelAlt.output_file_name.split('.')[0])
            return outname

        elif 'ressimresname' in Line_info.keys():
            outname = '{0}_{1}_{2}'.format(os.path.basename(self.ModelAlt.h5fname).split('.')[0]+'_h5',
                                               Line_info['parameter'], Line_info['ressimresname'])
            return outname

        return 'NULL'

    def buildHeadersByTimestamps(self, timestamps, year='ALLYEARS'):
        '''
        build headers for profile line stat tables by timestamp
        convert to Datetime, no matter what. We can convert back..
        Filter by year, using year input. If ALLYEARS, no data is filtered.
        :param timestamps: list of available timesteps
        :param year: used to filter down to the year, or if ALLYEARS, allow all years
        :return: list of headers
        '''

        headers = []
        for timestamp in timestamps:

            if isinstance(timestamp, dt.datetime):
                if year != 'ALLYEARS':
                    if year == timestamp.year:
                        headers.append(timestamp)
                else:
                    headers.append(timestamp)

            elif isinstance(timestamp, float):
                ts_dt = self.JDateToDatetime(timestamp)
                if year != 'ALLYEARS':
                    if year == ts_dt.year:
                        headers.append(str(timestamp))
                else:
                    headers.append(str(timestamp))
        return headers

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
            self.XML.writeIntroLine(self.SimulationCSV[model]['plugin'])
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
                    else:
                        print('Section Type {0} not identified.'.format(section['type']))
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

        # df = pd.DataFrame(data, columns=columns)
        df.to_csv(os.path.join(self.images_path, 'Log.csv'), index=False)

    def writeDataFiles(self):
        '''
        writes out the data used in figures to csv files for later use and checking
        '''

        for key in self.Data_Memory.keys():
            csv_name = os.path.join(self.CSVPath, '{0}.csv'.format(key))
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
            #if both are lists..
            if len(array1) < len(array2):
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
                for i in enumerate(array2):
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

    def loadModelAlt(self, simCSVAlt):
        '''
        Loads info for specified model alts. Loads correct model plugin class from WDR
        :param simCSVAlt: simulation alt dict object from self.simulation class
        :return: class variables
                self.alternativeFpart
                self.alternativeDirectory
                self.ModelAlt - WDR class that is plugin specific
        '''

        approved_modelalts = [modelalt for modelalt in self.ModelAlternatives if modelalt['name'] == simCSVAlt['modelaltname'] and
                              modelalt['program'] == simCSVAlt['plugin']]
        if len(approved_modelalts) == 0:
            print('Incompatible input information from the WAT XML output file ({0})\nand Simulation CSV file ({1})'.format(self.simulationInfoFile, '{0}.csv'.format(self.baseSimulationName.replace(' ', '_'))))
            print('Please Confirm inputs and run again.')
            print('Now Exiting...')
            sys.exit()
        else:
            approved_modelalt = approved_modelalts[0]
            self.alternativeFpart = approved_modelalt['fpart']
            self.alternativeDirectory = approved_modelalt['directory']

        if self.plugin.lower() == "ressim":
            self.ModelAlt = WDR.ResSim_Results(self.simulationDir, self.alternativeFpart, self.StartTime, self.EndTime)
        elif self.plugin.lower() == 'cequalw2':
            self.ModelAlt = WDR.W2_Results(self.simulationDir, self.modelAltName, self.alternativeDirectory, self.StartTime, self.EndTime)

    def initializeXML(self):
        '''
        creates a new version of the template XML file, initiates the XML class and writes the cover page
        :return: sets class variables
                    self.XML
        '''

        # new_xml = 'USBRAutomatedReportOutput.xml' #required name for file
        new_xml = os.path.join(self.studyDir, 'reports', 'Datasources', 'USBRAutomatedReportOutput.xml') #required name for file
        self.XML = XML_Utils.XMLReport(new_xml)
        self.XML.writeCover('DRAFT Temperature Validation Summary Report')

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

    def loadDefaultPlotObject(self, plotobject):
        '''
        loads the graphic default options.
        :param plotobject: string specifying the default graphics object
        :return:
            plot_info: dict of object settings
        '''

        plot_info = pickle.loads(pickle.dumps(self.graphicsDefault[plotobject], -1))
        return plot_info

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
            # print('Dates already in JDate form.')
            return dates
        elif isinstance(dates, (list, np.ndarray)):
            if isinstance(dates[0], (float, int)):
                # print('Dates already in JDate form.')
                return dates
            jdates = np.asarray([(WF.datetime2Ordinal(n) - self.ModelAlt.t_offset) + 1 for n in dates])
            return jdates
        elif isinstance(dates, dt.datetime):
            jdate = (WF.datetime2Ordinal(dates) - self.ModelAlt.t_offset) + 1
            return jdate
        else:
            # print('Unable to convert type {0} to JDates'.format(type(dates)))
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
            # print('Dates already in Datetime form.')
            return dates
        elif isinstance(dates, (list, np.ndarray)):
            if isinstance(dates[0], dt.datetime):
                # print('Dates already in Datetime form.')
                return dates
            dtimes = np.asarray([first_year_Date + dt.timedelta(days=n) for n in dates])
            return dtimes
        elif isinstance(dates, (float, int)):
            dtime = first_year_Date + dt.timedelta(days=dates)
            return dtime
        else:
            # print('Unable to convert type {0} to datetime'.format(type(dates)))
            return dates

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
        for header in headers:
            if object_settings['dateformat'].lower() == 'datetime':
                header = self.translateDateFormat(header, 'datetime', '')
                header = header.strftime('%d%b%Y')
            elif object_settings['dateformat'].lower() == 'jdate':
                header = self.translateDateFormat(header, 'jdate', '')
                header = str(header)
            new_headers.append(header)

        return new_headers

    def convertProfileDataUnits(self, object_settings):
        '''
        converts the units of profile data if unitsystem is defined
        :param object_settings: user defined settings for current object
        :return: object_setting dictionaries with updated units and values
        '''

        if 'unitsystem' not in object_settings.keys():
            print('Unit system not defined.')
            return object_settings
        for flag in object_settings['linedata'].keys():
            if object_settings['linedata'][flag]['units'] == None:
                continue
            else:
                profiles = object_settings['linedata'][flag]['values']
                profileunits = object_settings['linedata'][flag]['units']
                for pi, profile in enumerate(profiles):
                    profile, newunits = self.convertUnitSystem(profile, profileunits, object_settings['unitsystem'])
                    profiles[pi] = profile
                object_settings['linedata'][flag]['units'] = newunits
        return object_settings

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

    def filterProfileData(self, object_settings):
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
            return object_settings

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
        for lineflag in object_settings['linedata'].keys():
            line = object_settings['linedata'][lineflag]

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

            for pi, profile in enumerate(line['values']):
                ydata = line[yflag][pi]

                if xmax != None and filtbylims:
                    xmax_filt = np.where(profile <= xmax)
                else:
                    xmax_filt = np.arange(len(profile))

                if xmin != None and filtbylims:
                    xmin_filt = np.where(profile >= xmin)
                else:
                    xmin_filt = np.arange(len(profile))

                if ymax != None and filtbylims:
                    ymax_filt = np.where(ydata <= ymax)
                else:
                    ymax_filt = np.arange(len(ydata))

                if ymin != None and filtbylims:
                    ymin_filt = np.where(ydata >= ymin)
                else:
                    ymin_filt = np.arange(len(ydata))

                if omitvalue != None:
                    omitval_filt = np.where(profile != omitvalue)
                else:
                    omitval_filt = np.arange(len(profile))

                master_filter = reduce(np.intersect1d, (xmax_filt, xmin_filt, ymax_filt, ymin_filt, omitval_filt))

                object_settings['linedata'][lineflag]['values'][pi] = profile[master_filter]
                object_settings['linedata'][lineflag][yflag][pi] = ydata[master_filter]

        return object_settings


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
                df = pd.DataFrame({'times': times, 'values': values})
                df = df.set_index('times')
                df = df.resample(pd_interval, origin='end_day').asfreq().fillna(method='bfill')
                new_values = df['values'].to_numpy()
                new_times = df.index.to_pydatetime()

            elif avgtype == 'INST-CUM':
                df = pd.DataFrame({'times': times, 'values': values})
                df = df.set_index('times')
                df = df.cumsum(skipna=True).resample(pd_interval, origin='end_day').asfreq().fillna(method='bfill')
                new_values = df['values'].to_numpy()
                new_times = df.index.to_pydatetime()


            elif avgtype == 'PER-AVER':
                #average over the period
                df = pd.DataFrame({'times': times, 'values': values})
                df = df.set_index('times')
                df = df.resample(pd_interval, origin='end_day').mean().fillna(method='bfill')
                new_values = df['values'].to_numpy()
                new_times = df.index.to_pydatetime()

            elif avgtype == 'PER-CUM':
                #cum over the period
                df = pd.DataFrame({'times': times, 'values': values})
                df = df.set_index('times')
                df = df.resample(pd_interval, origin='end_day').sum().fillna(method='bfill')
                new_values = df['values'].to_numpy()
                new_times = df.index.to_pydatetime()

            else:
                print('INVALID INPUT TYPE DETECTED', avgtype)
                return times, values

        if convert_to_jdate:
            return self.DatetimeToJDate(new_times), np.asarray(new_values)
        else:
            return new_times, np.asarray(new_values)

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

    def convertDepthsToElevations(self, object_settings):
        '''
        handles data to convert depths into elevations for observed data
        :param object_settings: dicitonary of user defined settings for current object
        :return: object settings dictionary with updated elevation data
        '''

        elev_flag = 'NOVALID'
        if object_settings['usedepth'].lower() == 'false':
            for ld in object_settings['linedata'].keys():
                if object_settings['linedata'][ld]['elevations'] == []:
                    noelev_flag = ld
                    for old in object_settings['linedata'].keys():
                        if len(object_settings['linedata'][old]['elevations']) > 0:
                            elev_flag = old
                            break

                    if elev_flag != 'NOVALID':
                        object_settings['linedata'][noelev_flag]['elevations'] = WF.convertObsDepths2Elevations(object_settings['linedata'][noelev_flag]['depths'],
                                                                                                       object_settings['linedata'][elev_flag]['elevations'])
                    else:
                        object_settings['usedepth'] = 'true'
        return object_settings

    def commitProfileDataToMemory(self, object_settings):
        '''
        commits updated data to data memory dictionary that keeps track of data
        :param object_settings:  dicitonary of user defined settings for current object
        '''
        copied_object_settings = pickle.loads(pickle.dumps(object_settings, -1))
        for line in copied_object_settings['linedata'].keys():
            values = copied_object_settings['linedata'][line]['values']
            depths = copied_object_settings['linedata'][line]['depths']
            elevations = copied_object_settings['linedata'][line]['elevations']
            datamem_key = copied_object_settings['linedata'][line]['logoutputfilename']
            # self.Data_Memory[datamem_key] = {'times': pickle.loads(pickle.dumps(object_settings['timestamps'], -1)),
            #                                  'values': pickle.loads(pickle.dumps(values, -1)),
            #                                  'elevations': pickle.loads(pickle.dumps(elevations, -1)),
            #                                  'depths': pickle.loads(pickle.dumps(depths, -1)),
            #                                  'units': object_settings['plot_units'],
            #                                  'isprofile': True}
            self.Data_Memory[datamem_key] = {'times': copied_object_settings['timestamps'],
                                             'values': values,
                                             'elevations': elevations,
                                             'depths': depths,
                                             'units': copied_object_settings['plot_units'],
                                             'isprofile': True}

    def configureUnits(self, object_settings, line, units):
        '''
        configure units from line settings
        :param object_settings:  dicitonary of user defined settings for current object
        :param line: current line settings
        :param units: current units of line
        :return: units
        '''

        if units == None:
            if 'parameter' in line.keys():
                try:
                    units = self.units[line['parameter'].lower()]
                except KeyError:
                    units = None

        if isinstance(units, dict):
            if 'unitsystem' in object_settings.keys():
                units = units[object_settings['unitsystem'].lower()]
            else:
                units = None
        return units

if __name__ == '__main__':
    rundir = sys.argv[0]
    simInfoFile = sys.argv[1]
    import cProfile
    ar = cProfile.run('MakeAutomatedReport(simInfoFile, rundir)')
    # MakeAutomatedReport(simInfoFile, rundir)

