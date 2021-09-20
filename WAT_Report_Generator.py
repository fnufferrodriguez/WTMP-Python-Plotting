'''
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
import copy
import calendar
import dateutil.parser
import re
from collections import Counter
import shutil

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

        self.batdir = batdir
        self.ReadSimulationInfo(simulationInfoFile) #read file output by WAT
        # self.EnsureDefaultFiles() #TODO: turn this back on for copying
        self.DefinePaths()
        self.DefineUnits()
        self.DefineMonths()
        self.DefineTimeIntervals()
        self.DefineDefaultColors()
        self.ReadGraphicsDefaultFile() #read graphical component defaults
        self.ReadLinesstylesDefaultFile()
        if self.reportType == 'single': #Eventually be able to do comparison reports, put that here
            for simulation in self.Simulations:
                # print('SIMULATION:', simulation)
                self.SetSimulationVariables(simulation)
                self.DefineStartEndYears()
                self.ReadSimulationsCSV() #read to determine order/sims/regions in report
                self.cleanOutputDirs()
                self.initializeXML()
                self.writeXMLIntroduction()
                for simorder in self.SimulationCSV.keys():
                    self.SetSimulationCSVVars(self.SimulationCSV[simorder])
                    self.ReadDefinitionsFile(self.SimulationCSV[simorder])
                    self.LoadModelAlt(self.SimulationCSV[simorder])
                    self.WriteChapter()
                self.XML.writeReportEnd()

    def DefineMonths(self):
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

    def DefineStartEndYears(self):
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

    def DefineTimeIntervals(self):
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

    def DefineDefaultColors(self):
        '''
        sets up a list of default colors to use in the event that colors are not set up in the graphics default file
        for a line
        :return: class variable
                    self.def_colors
        '''

        self.def_colors = ['darkgreen', 'red', 'blue', 'orange', 'darkcyan', 'darkmagenta', 'gray', 'black']

    def DefineUnits(self):
        '''
        creates dictionary with units for vars for labels
        #TODO: expand this
        :return: set class variable
                    self.units
        '''

        self.units = {'temperature': {'metric':'c', 'english':'f'},
                      'do_sat': '%',
                      'flow': {'metric': 'm3/s', 'english': 'cfs'},
                      'storage': {'metric': 'm3', 'english': 'af'}}

    def DefinePaths(self):
        '''
        defines run specific paths
        used to contain more paths, but not needed. Consider moving.
        :return: set class variables
                    self.images_path
        '''

        self.images_path = os.path.join(self.studyDir, 'reports', 'Images')
        self.default_dir = os.path.join(os.path.split(self.batdir)[0], 'Default')

    def MakeTimeSeriesPlot(self, object_settings):
        '''
        takes in object settings to build time series plot and write to XML
        #TODO: pull common settings out into their own funcitons so we dont have to look at them
        :param object_settings: currently selected object settings dictionary
        :return: creates png in images dir and writes to XML file
        '''

        default_settings = self.load_defaultPlotObject('timeseriesplot') #get default TS plot items
        object_settings = self.replaceDefaults(default_settings, object_settings) #overwrite the defaults with chapter file

        object_settings = self.updateFlaggedValues(object_settings, '%%year%%', self.years_str)

        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot()
        param_count = {}
        unitslist = []
        for line in object_settings['lines']:
            if 'parameter' in line.keys():
                param = line['parameter']
            else:
                param = None
            if param not in param_count.keys():
                param_count[param] = 0
            else:
                param_count[param] += 1
            i = param_count[param]

            dates, values, units = self.getTimeSeries(line)

            if units == None:
                if param != None:
                    try:
                        units = self.units[param.lower()]
                    except KeyError:
                        units = None

            if isinstance(units, dict):
                if 'unitsystem' in object_settings.keys():
                    units = units[object_settings['unitsystem'].lower()]
                else:
                    units = None

            if 'unitsystem' in object_settings.keys():
                values, units = self.convertUnitSystem(values, units, object_settings['unitsystem'])

            chkvals = WF.check_data(values)
            if not chkvals:
                print('Invalid Data settings for line:', line)
                continue

            if 'dateformat' in object_settings.keys():
                if object_settings['dateformat'].lower() == 'jdate':
                    if isinstance(dates[0], dt.datetime):
                        dates = self.DatetimeToJDate(dates)
                elif object_settings['dateformat'].lower() == 'datetime':
                    if isinstance(dates[0], float) or isinstance(dates[0], int):
                        dates = self.JDateToDatetime(dates)

            if units != '' and units != None:
                unitslist.append(units)

            line_settings = self.getLineDefaultSettings(line, param, i)

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
                        markevery=int(line_settings['numptsskip']), zorder=float(line_settings['zorder']))

            elif line_settings['drawline'].lower() == 'true':
                ax.plot(dates, values, label=line_settings['label'], c=line_settings['linecolor'],
                        lw=line_settings['linewidth'], ls=line_settings['linestylepattern'],
                        zorder=float(line_settings['zorder']))

            elif line_settings['drawpoints'].lower() == 'true':
                ax.scatter(dates[::int(line_settings['numptsskip'])], values[::int(line_settings['numptsskip'])],
                           marker=line_settings['symboltype'], facecolor=line_settings['pointfillcolor'],
                           edgecolor=line_settings['pointlinecolor'], s=float(line_settings['symbolsize']),
                           label=line_settings['label'], zorder=float(line_settings['zorder']))

        plotunits = self.getPlotUnits(unitslist, object_settings)
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

    def MakeProfilePlot(self, object_settings):
        '''
        takes in object settings to build profile plot and write to XML
        #TODO: pull common settings out into their own funcitons so we dont have to look at them
        #TODO: figure out way to change or control units on the axis?
        #TODO: figure out way to convert profiles that are essentially unitless
        :param object_settings: currently selected object settings dictionary
        :return: creates png in images dir and writes to XML file
        '''

        default_settings = self.load_defaultPlotObject('profileplot')
        object_settings = self.replaceDefaults(default_settings, object_settings)
        obj_desc = self.updateFlaggedValues(object_settings['description'], '%%year%%', self.years_str)
        self.XML.writeProfilePlotStart(obj_desc)

        linedata = {}
        ################# Get timestamps #################
        timestamps = []
        if 'datessource' in object_settings.keys():
            datessource_flag = object_settings['datessource'] #determine how you want to get dates? either flag or list
        else:
            datessource_flag = [] #let it make timesteps

        if isinstance(datessource_flag, str):
            for line in object_settings['lines']:
                if line['flag'] == datessource_flag:
                    timestamps = self.getProfileDates(line)
        elif isinstance(datessource_flag, dict): #single date instance..
            timestamps = object_settings['datessource']['date']
        elif isinstance(datessource_flag, list):
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

        ################# Get units #################
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

        if 'units' in object_settings.keys():
            plot_units = object_settings['units'].lower()
        elif plot_parameter != None:
            plot_units = self.units[plot_parameter.lower()]
            if isinstance(plot_units, dict):
                if 'unitsystem' in object_settings.keys():
                    plot_units = plot_units[object_settings['unitsystem'].lower()]
                else:
                    plot_units = ''
        else:
            plot_units = ''
        object_settings = self.updateFlaggedValues(object_settings, '%%units%%', plot_units)

        ################# Get data #################
        #do now incase no elevs, so we can convert
        for line in object_settings['lines']:
            vals, elevations, depths, flag = self.getProfileData(line, timestamps) #Test this speed for grabbing all profiles and then choosing

            linedata[flag] = {'values': vals,
                              'elevations': elevations,
                              'depths': depths}

        ################ convert Elevs ################
        elev_flag = 'NOVALID'
        if object_settings['usedepth'].lower() == 'false':
            for ld in linedata.keys():
                if linedata[ld]['elevations'] == []:
                    noelev_flag = ld
                    for old in linedata.keys():
                        if len(linedata[old]['elevations']) > 0:
                            elev_flag = old
                            break

                    if elev_flag != 'NOVALID':
                        linedata[noelev_flag]['elevations'] = WF.convert_obs_depths(linedata[noelev_flag]['depths'],
                                                                                    linedata[elev_flag]['elevations'])
                    else:
                        object_settings['usedepth'] = 'true'

        split_by_year = False
        if 'splitbyyear' in object_settings.keys():
            if object_settings['splitbyyear'].lower() == 'true':
                split_by_year = True
                years = self.years
        if not split_by_year:
            yearstr = self.years_str
            years = ['ALLYEARS']

        ################ Build Plots ################
        for yi, year in enumerate(years):
            if split_by_year:
                yearstr = str(year)

            t_stmps = self.filterTimestepByYear(timestamps, year)
            prof_indices = list(range(len(t_stmps)))
            n = int(object_settings['profilesperrow']) * int(object_settings['rowsperpage']) #Get number of plots on page
            page_indices = [prof_indices[i * n:(i + 1) * n] for i in range((len(prof_indices) + n - 1) // n)]
            cur_obj_settings = copy.deepcopy(object_settings)

            for page_i, pgi in enumerate(page_indices):

                subplot_rows, subplot_cols = WF.get_subplot_config(len(pgi), int(cur_obj_settings['profilesperrow']))
                n_nrow_active = np.ceil(len(pgi) / subplot_cols)
                fig = plt.figure(figsize=(7, 1 + 3 * n_nrow_active))

                current_object_settings = self.updateFlaggedValues(cur_obj_settings, '%%year%%', yearstr)

                for i, j in enumerate(pgi):
                    ax = fig.add_subplot(int(subplot_rows), int(subplot_cols), i + 1)

                    if object_settings['usedepth'].lower() == 'true':
                        ax.invert_yaxis()

                    for li, line in enumerate(linedata.keys()):
                        try:
                            values = linedata[line]['values'][j]
                            msk = np.where(~np.isnan(values))
                            values = values[msk]
                        except IndexError:
                            print('No Data. Skipping line...')
                            continue

                        if object_settings['usedepth'].lower() == 'true':
                            levels = linedata[line]['depths'][j][msk]
                        else:
                            levels = linedata[line]['elevations'][j][msk]

                        if not WF.check_data(values):
                            continue

                        current_ls = self.getLineSettings(object_settings['lines'], line)
                        current_ls = self.getLineDefaultSettings(current_ls, plot_parameter, li)

                        if current_ls['drawline'].lower() == 'true' and current_ls['drawpoints'].lower() == 'true':
                            ax.plot(values, levels, label=current_ls['label'], c=current_ls['linecolor'],
                                    lw=current_ls['linewidth'], ls=current_ls['linestylepattern'],
                                    marker=current_ls['symboltype'], markerfacecolor=current_ls['pointfillcolor'],
                                    markeredgecolor=current_ls['pointlinecolor'], markersize=float(current_ls['symbolsize']),
                                    markevery=int(current_ls['numptsskip']))

                        elif current_ls['drawline'].lower() == 'true':
                            ax.plot(values, levels, label=current_ls['label'], c=current_ls['linecolor'],
                                    lw=current_ls['linewidth'], ls=current_ls['linestylepattern'])

                        elif current_ls['drawpoints'].lower() == 'true':
                            ax.scatter(values[::int(current_ls['numptsskip'])], levels[::int(current_ls['numptsskip'])],
                                       marker=current_ls['symboltype'], facecolor=current_ls['pointfillcolor'],
                                       edgecolor=current_ls['pointlinecolor'], s=float(current_ls['symbolsize']),
                                       label=current_ls['label'])

                    show_legend, show_xlabel, show_ylabel = self.getPlotLabelMasks(i, len(pgi), subplot_cols)

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

                    if show_legend:
                        if 'legend' in current_object_settings.keys():
                            if current_object_settings['legend'].lower() == 'true':
                                if 'legendsize' in current_object_settings.keys():
                                    legsize = float(current_object_settings['legendsize'])
                                elif 'fontsize' in current_object_settings.keys():
                                    legsize = float(current_object_settings['fontsize'])
                                else:
                                    legsize = 12
                                ax.legend(loc='lower right', fontsize=legsize)

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

                    ttl_str = dt.datetime.strftime(t_stmps[j], '%d %b %Y')
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
                figname = 'ProfilePlot_{0}_{1}_{2}_{3}_{4}.png'.format(self.ChapterName, yearstr, plot_parameter, self.plugin, page_i)

                plt.savefig(os.path.join(self.images_path, figname), dpi=600)
                plt.close('all')

                ################################################
                # if not split_by_year:
                #     description = '{0} {1}: {2} of {3}'.format(current_object_settings['description'], yearstr, page_i+1, len(page_indices))
                # else:
                # description = '{0}: {1} of {2}'.format(current_object_settings['description'], page_i+1, len(page_indices))
                description = '{0}: {1} of {2}'.format(cur_obj_settings['description'], page_i+1, len(page_indices))
                self.XML.writeProfilePlotFigure(figname, description)

        self.XML.writeProfilePlotEnd()

    def MakeErrorStatisticsTable(self, object_settings):
        '''
        takes in object settings to build error stats table and write to XML
        :param object_settings: currently selected object settings dictionary
        :return: writes to XML file
        '''

        default_settings = self.load_defaultPlotObject('errorstatisticstable')
        object_settings = self.replaceDefaults(default_settings, object_settings)
        datapaths = object_settings['datapaths']

        years = self.years

        split_by_year = False
        if 'splitbyyear' in object_settings.keys():
            if object_settings['splitbyyear'].lower() == 'true':
                split_by_year = True
        if not split_by_year:
            years = [self.years_str]

        data = {}
        unitslist = []
        for dp in datapaths:
            dates, values, units = self.getTimeSeries(dp)

            if units == None:
                if 'parameter' in dp.keys():
                    try:
                        units = self.units[dp['parameter'].lower()]
                    except KeyError:
                        units = None

            if isinstance(units, dict):
                if 'unitsystem' in object_settings.keys():
                    units = units[object_settings['unitsystem'].lower()]
                else:
                    units = None

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
                unitslist.append(units)

            data[dp['flag']] = {'dates': dates,
                                'values': values,
                                'units': units}

        plotunits = self.getPlotUnits(unitslist, object_settings)
        object_settings = self.updateFlaggedValues(object_settings, '%%units%%', plotunits)

        headings = self.buildHeadersByYear(object_settings, years, split_by_year)
        rows = self.buildRowsByYear(object_settings, years, split_by_year)

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
                    row_val = self.formatStatsLine(row_val, data, year=year)
                header = '' if header == None else header
                frmt_rows.append('{0}|{1}'.format(rowname, row_val))
            self.XML.writeTableColumn(header, frmt_rows)
        self.XML.writeTableEnd()

    def MakeMonthlyStatisticsTable(self, object_settings):
        '''
        takes in object settings to build monthly stats table and write to XML
        :param object_settings: currently selected object settings dictionary
        :return: writes to XML file
        '''

        default_settings = self.load_defaultPlotObject('monthlystatisticstable')
        object_settings = self.replaceDefaults(default_settings, object_settings)
        datapaths = object_settings['datapaths']

        years = self.years

        split_by_year = False
        if 'splitbyyear' in object_settings.keys():
            if object_settings['splitbyyear'].lower() == 'true':
                split_by_year = True
        if not split_by_year:
            if len(years) == 1:
                years = [str(n) for n in years]
            else:
                years = ['{0}-{1}'.format(years[0], years[-1])]

        headings = self.buildHeadersByYear(object_settings, years, split_by_year)
        rows = self.buildRowsByYear(object_settings, years, split_by_year)

        data = {}
        unitslist = []
        for dp in datapaths:
            dates, values, units = self.getTimeSeries(dp)

            if units == None:
                if 'parameter' in dp.keys():
                    try:
                        units = self.units[dp['parameter'].lower()]
                    except KeyError:
                        units = None

            if isinstance(units, dict):
                if 'unitsystem' in object_settings.keys():
                    units = units[object_settings['unitsystem'].lower()]
                else:
                    units = None

            if 'unitsystem' in object_settings.keys():
                values, units = self.convertUnitSystem(values, units, object_settings['unitsystem'])

            if units != None:
                unitslist.append(units)

            if 'filterbylimits' not in dp.keys():
                dp['filterbylimits'] = 'true' #set default

            if dp['filterbylimits'].lower() == 'true':
                if 'xlims' in object_settings.keys():
                    dates, values = self.limitXdata(dates, values, object_settings['xlims'])
                if 'ylims' in object_settings.keys():
                    dates, values = self.limitYdata(dates, values, object_settings['ylims'])

            data[dp['flag']] = {'dates':dates,
                                'values': values,
                                'units': units}

        plotunits = self.getPlotUnits(unitslist, object_settings)
        object_settings = self.updateFlaggedValues(object_settings, '%%units%%', plotunits)

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
                    row_val = self.formatStatsLine(row_val, data, year=year)
                header = '' if header == None else header
                frmt_rows.append('{0}|{1}'.format(rowname, row_val))
            self.XML.writeTableColumn(header, frmt_rows)
        self.XML.writeTableEnd()

    def MakeBuzzPlot(self, object_settings):
        '''
        takes in object settings to build buzzplots and write to XML
        :param object_settings: currently selected object settings dictionary
        :return: creates png in images dir and writes to XML file
        '''

        default_settings = self.load_defaultPlotObject('buzzplot')
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
            dates, values, units = self.getTimeSeries(line)

            if 'target' in line.keys():
                values = self.BuzzTargetSum(dates, values, float(line['target']))
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

            chkvals = WF.check_data(values)
            if not chkvals:
                print('Invalid Data settings for line:', line)
                continue

            if 'dateformat' in object_settings.keys():
                if object_settings['dateformat'].lower() == 'jdate':
                    if isinstance(dates[0], dt.datetime):
                        dates = self.DatetimeToJDate(dates)
                elif object_settings['dateformat'].lower() == 'datetime':
                    if isinstance(dates[0], float) or isinstance(dates[0], int):
                        dates = self.JDateToDatetime(dates)
                else:
                    if isinstance(dates[0], float) or isinstance(dates[0], int):
                        dates = self.JDateToDatetime(dates)

            if 'unitsystem' in object_settings.keys():
                values, units = self.convertUnitSystem(values, units, object_settings['unitsystem'])

            if 'parameter' in line.keys():
                line_settings = self.getLineDefaultSettings(line, line['parameter'], i)
            else:
                line_settings = self.getLineDefaultSettings(line, None, i)

            if 'scalar' in line.keys():
                try:
                    scalar = float(line['scalar'])
                    values = scalar * np.asarray(values)
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
                               markevery=int(line_settings['numptsskip']), zorder=int(line_settings['zorder']))

                elif line_settings['drawline'].lower() == 'true':
                    curax.plot(dates, values, label=line_settings['label'], c=line_settings['linecolor'],
                               lw=line_settings['linewidth'], ls=line_settings['linestylepattern'],
                               zorder=int(line_settings['zorder']))

                elif line_settings['drawpoints'].lower() == 'true':
                    curax.scatter(dates[::int(line_settings['numptsskip'])], values[::int(line_settings['numptsskip'])],
                                  marker=line_settings['symboltype'], facecolor=line_settings['pointfillcolor'],
                                  edgecolor=line_settings['pointlinecolor'], s=float(line_settings['symbolsize']),
                                  label=line_settings['label'], zorder=int(line_settings['zorder']))

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
                values[di] = list(np.asarray(values[di])[mask_date_idx])

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
        elif 'fontsize' in object_settings.keys():
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

    def ReadSimulationInfo(self, simulationInfoFile):
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

    def ReadSimulationsCSV(self):
        '''
        reads the Simulation file and gets the region info
        :return: class variable
                    self.SimulationCSV
        '''
        self.SimulationCSV = WDR.ReadSimulationFile(self.baseSimulationName, self.studyDir)

    def ReadGraphicsDefaultFile(self):
        '''
        sets up path for graphics default file in study and reads the xml
        :return: class variable
                    self.graphicsDefault
        '''

        graphicsDefaultfile = os.path.join(self.studyDir, 'reports', 'Graphics_Defaults.xml')
        # graphicsDefaultfile = os.path.join(self.default_dir, 'Graphics_Defaults.xml') #TODO: implement with build
        self.graphicsDefault = WDR.ReadGraphicsDefaults(graphicsDefaultfile)

    def ReadLinesstylesDefaultFile(self):
        '''
        sets up path for default line styles file and reads the xml
        :return: class variable
                    self.defaultLineStyles
        '''

        defaultLinesFile = os.path.join(self.studyDir, 'reports', 'defaultLineStyles.xml')
        # defaultLinesFile = os.path.join(self.default_dir, 'defaultLineStyles.xml') #TODO: implement with build
        self.defaultLineStyles = WDR.ReadDefaultLineStyle(defaultLinesFile)

    def ReadDefinitionsFile(self, simorder):
        '''
        reads the chapter definitions file defined in the plugin csv file for a specified simulation
        :param simorder: simulation dictionary object
        :return: class variable
                    self.ChapterDefinitions
        '''

        ChapterDefinitionsFile = os.path.join(self.studyDir, 'reports', simorder['deffile'])
        self.ChapterDefinitions = WDR.ReadChapterDefFile(ChapterDefinitionsFile)

    def SetSimulationDateTimes(self):
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

    def SetSimulationCSVVars(self, simlist):
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

    def SetSimulationVariables(self, simulation):
        '''
        sets various class variables for selected variable
        sets simulation dates and times
        :param simulation: simulation dictionary object for specified simulation
        :return: class variables
                    self.SimulationName
                    self.baseSimulationName
                    self.simulationDir
                    self.DSSFile
                    self.StartTimeStr
                    self.EndTimeStr
                    self.LastComputed
                    self.ModelAlternatives
        '''

        self.SimulationName = simulation['name']
        self.baseSimulationName = simulation['basename']
        self.simulationDir = simulation['directory']
        self.DSSFile = simulation['dssfile']
        self.StartTimeStr = simulation['starttime']
        self.EndTimeStr = simulation['endtime']
        self.LastComputed = simulation['lastcomputed']
        self.ModelAlternatives = simulation['modelalternatives']
        self.SetSimulationDateTimes()

    def getLineDefaultSettings(self, LineSettings, param, i):
        '''
        gets line settings and adds missing needed settings with defaults. Then translates java style inputs to
        python commands. Gets colors and styles.
        :param LineSettings: dictionary object containing settings and flags for lines/points
        :param param: parameter of data in order to grab default
        :param i: number of line on the plot in order to get the right sequential color
        :return:
            LineSettings: dictionary containing keys describing how the line/points are drawn
        '''

        if param != None:
            param = param.lower()
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
        PointVars = ['pointfillcolor', 'pointlinecolor', 'symboltype', 'symbolsize', 'numptsskip']

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
        return {'linewidth': 2, 'linecolor': self.def_colors[i], 'linestyle': 'solid'}

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
                'symbolsize': 5, 'numptsskip': 0}

    def getLineSettings(self, LineSettings, Flag):
        '''
        gets the correct line settings for the selected flag
        :param LineSettings: dictionary of settings
        :param Flag: selected flag to match line
        :return: deep copy of line
        '''

        for line in LineSettings:
            if Flag == line['flag']:
                return copy.deepcopy(line)

    def getPlotLabelMasks(self, idx, nprofiles, cols):
        '''
        Get plot label masks
        :param idx: page index
        :param nprofiles: number of profiles
        :param cols: number of columns
        :return: boolean fields for plotting
        '''

        if idx == cols - 1:
            add_legend = True
        else:
            add_legend = False
        if idx >= nprofiles - cols:
            add_xlabel = True
        else:
            add_xlabel = False
        if idx % cols == 0:
            add_ylabel = True
        else:
            add_ylabel = False
        return add_legend, add_xlabel, add_ylabel

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
                times, values, units = WDR.ReadDSSData(Line_info['dss_filename'], Line_info['dss_path'],
                                                       self.StartTime, self.EndTime)

                if np.any(values == None):
                    return [], [], None
                elif len(values) == 0:
                    return [], [], None

        elif 'w2_file' in Line_info.keys():
            if 'structurenumbers' in Line_info.keys():
                # Ryan Miles: yeah looks like it's str_brX.npt, and X is 1-# of branches (which is defined in the control file)
                times, values = self.ModelAlt.readStructuredTimeSeries(Line_info['w2_file'], Line_info['structurenumbers'])
                # values, parameter = self.ModelAlt.filterByParameter(values, Line_info) #we need all params...
            else:
                times, values = self.ModelAlt.readTimeSeries(Line_info['w2_file'], **Line_info)
            if 'units' in Line_info.keys():
                units = Line_info['units']
            else:
                units = None
                # units = self.units[Line_info['parameter'].lower()]

        elif 'xy' in Line_info.keys():
            times, values = self.ModelAlt.readTimeSeries(Line_info['parameter'], Line_info['xy'])
            units = self.units[Line_info['parameter'].lower()]

        else:
            print('No Data Defined for line')
            return [], [], None

        if 'omitvalue' in Line_info.keys():
            omitval = float(Line_info['omitvalue'])
            values = self.replaceOmittedValues(values, omitval)

        if len(values) == 0:
            return [], [], None
        else:
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

        # occurence_count = Counter(t_ints)
        # most_common_interval = occurence_count.most_common(1)[0][0]
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

    def getProfileData(self, Line_info, timesteps):
        '''
        reads in profile data from various sources for profile plots at given timesteps
        attempts to get elevations if possible
        :param Line_info: dictionary containing settings for line
        :param timesteps: given list of timesteps to extract data at
        :return: values, elevations, depths, flag
        '''

        if 'filename' in Line_info.keys(): #Get data from Observed
            filename = Line_info['filename']
            values, depths = WDR.ReadTextProfile(filename, timesteps)
            return values, [], depths, Line_info['flag']

        elif 'w2_segment' in Line_info.keys():
            vals, elevations, depths = self.ModelAlt.readProfileData(Line_info['w2_segment'], timesteps)
            return vals, elevations, depths, Line_info['flag']

        elif 'ressimresname' in Line_info.keys():
            vals, elevations, depths = self.ModelAlt.readProfileData(Line_info['ressimresname'],
                                                                     Line_info['parameter'], timesteps)
            return vals, elevations, depths, Line_info['flag']

        print('No Data Defined for line')
        print('Line:', Line_info)
        return [], [], [], None

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
        :param unitslist: list of units plotted
        :param object_settings: dictionary with plot settings
        :return: string units value
        '''

        if len(unitslist) > 0:
            plotunits = self.getMostCommon(unitslist)
        elif 'parameter' in object_settings.keys() and len(unitslist) == 0:
            try:
                plotunits = self.units[object_settings['parameter'].lower()]
                if isinstance(plotunits, dict):
                    if 'unitsystem' in object_settings.keys():
                        plotunits = plotunits[object_settings['unitsystem'].lower()]
                    else:
                        plotunits = ''
            except KeyError:
                plotunits = ''
        else:
            print('No units defined.')
            plotunits = ''
        return plotunits

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

        default_settings = copy.deepcopy(self.replaceflaggedValues(default_settings))
        object_settings = copy.deepcopy(self.replaceflaggedValues(object_settings))

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
        if isinstance(values, dict):
            new_values = {}
            for key in values:
                new_values[key] = self.replaceOmittedValues(values[key], omitval)
        else:
            o_msk = np.where(values==omitval)
            values[o_msk] = np.nan
            new_values = values
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
            if not isinstance(lim, dt.datetime):
                try:
                    lim_frmt = dateutil.parser.parse(lim) #try simple date formatting.
                    if not self.StartTime <= lim_frmt <= self.EndTime: #check for false negative
                        print('Xlim of {0} not between start and endtime {1} - {2}'.format(lim_frmt, self.StartTime,
                                                                                           self.EndTime))
                        raise Exception
                except:
                    print('Error Reading Limit: {0} as a dt.datetime object.'.format(lim))
                    print('If this is wrong, try format: Apr 2014 1 12:00')
                    print('Trying as Jdate..')
                    try:
                        lim_frmt = float(lim)
                        lim_frmt = self.JDateToDatetime(lim_frmt)
                    except:
                        print('Limit value of {0} also invalid as jdate.'.format(lim))
                        print('Setting to fallback {0}.'.format(fallback))
                        lim_frmt = fallback

                return lim_frmt

        elif dateformat.lower() == 'jdate':
            try:
                lim_frmt = float(lim) #try simple date formatting.
            except:
                print('Error Reading Limit: {0} as a jdate.'.format(lim))
                print('If this is wrong, try format: 180')
                print('Trying as Datetime..')
                try:
                    if not isinstance(lim, dt.datetime):
                        lim_frmt = dateutil.parser.parse(lim)
                    lim_frmt = self.DatetimeToJDate(lim_frmt)
                except:
                    print('Limit value of {0} also invalid as datetime.'.format(lim))
                    print('Setting to fallback {0}.'.format(fallback))
                    lim_frmt = self.DatetimeToJDate(fallback)
            return lim_frmt

        else:
            print('Invalid dateformat of {0}'.format(dateformat))
            print('Using fallback in dt form.')
            lim_frmt = fallback
            return lim_frmt

        return lim

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
            print('set Xlims at', min, max)

        else:
            print('No Xlims flag set for {0}'.format(xlims_flag))
            print('Not setting Xlims.')

    def formatStatsLine(self, row, data_dict, year='ALL'):
        '''
        takes rows for tables and replaces flags with the correct data, computing stat analysis if needed
        :param row: row section string
        :param data_dict: dictionary of data that could be used
        :param year: selected year, or 'ALL'
        :return: new row value
        '''

        data = copy.deepcopy(data_dict)
        rrow = row.replace('%%', '')
        s_row = rrow.split('.')
        flags = []
        for sr in s_row:
            if sr in data_dict.keys():
                curflag = sr
                curvalues = np.array(data[sr]['values'])
                curdates = np.array(data[sr]['dates'])
                flags.append(curflag)
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
                        msk = [i for i, n in enumerate(curdates) if n.month == sr_month]
                        data[curflag]['values'] = curvalues[msk]
                        data[curflag]['dates'] = curdates[msk]

        if year != 'ALL':
            for flag in flags:
                msk = [i for i, n in enumerate(data[flag]['dates']) if n.year == year]
                data[flag]['values'] = np.asarray(data[flag]['values'])[msk]
                data[flag]['dates'] = np.asarray(data[flag]['dates'])[msk]

        if row.lower().startswith('%%meanbias'):
            return WF.meanbias(data[flags[0]], data[flags[1]])
        elif row.lower().startswith('%%mae'):
            return WF.MAE(data[flags[0]], data[flags[1]])
        elif row.lower().startswith('%%rmse'):
            return WF.RMSE(data[flags[0]], data[flags[1]])
        elif row.lower().startswith('%%nse'):
            return WF.NSE(data[flags[0]], data[flags[1]])
        elif row.lower().startswith('%%count'):
            return WF.COUNT(data[flags[0]])
        elif row.lower().startswith('%%mean'):
            return WF.MEAN(data[flags[0]])
        else:
            if '%%' in row:
                print('Unable to convert flag in row', row)
            return row

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
            # r = []
        return rows

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
        for i, header in enumerate(object_settings['headers']):
            if isinstance(object_settings['headers'], dict):
                header = object_settings['headers']['header'] #single headers come as dict objs TODO fix this eventually...
            if '%%year%%' in header:
                if split_by_year:
                    header_by_year.append(header)
                else:
                    headings.append(['ALL', self.updateFlaggedValues(header, '%%year%%', str(years[0]))])
            else:
                if len(header_by_year) > 0:
                    for year in years:
                        for yrhd in header_by_year:
                            headings.append([year, self.updateFlaggedValues(yrhd, '%%year%%', str(year))])
                    header_by_year = []
                headings.append(['ALL', header])
        if len(header_by_year) > 0:
            for year in years:
                for yrhd in header_by_year:
                    headings.append([year, self.updateFlaggedValues(yrhd, '%%year%%', str(year))])
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
            ts = [t.astype(dt.datetime) for t in ts]
        elif interval_info == 'pd':
            ts = pd.date_range(startTime, endTime, freq=interval, closed=None)
            ts = [t.to_pydatetime() for t in ts]
        return ts

    def limitXdata(self, dates, values, xlims):
        '''
        if the filterbylimits flag is true, filters out values outside of the xlimits
        :param dates: list of dates
        :param values: list of values
        :param xlims: dictionary of xlims, containing potentially min and/or max
        :return: filtered dates and values
        '''

        if isinstance(dates[0], int) or isinstance(dates[0], float):
            wantedformat = 'jdate'
        elif isinstance(dates[0], dt.datetime):
            wantedformat = 'datetime'
        if 'min' in xlims.keys():
            #ensure we have dt, dss dates should be... #TODO: make sure values are DT
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

    def WriteChapter(self):
        '''
        writes each chapter defined in the simulation CSV file to the XML file. Generates plots and figures
        :return: class variables
                    self.ChapterName
                    self.ChapterRegion
        '''

        for Chapter in self.ChapterDefinitions:
            self.ChapterName = Chapter['name']
            self.ChapterRegion = Chapter['region']
            self.XML.writeChapterStart(self.ChapterName)
            for section in Chapter['sections']:
                section_header = section['header']
                self.XML.writeSectionHeader(section_header)
                for object in section['objects']:
                    if object['type'] == 'TimeSeriesPlot':
                        self.MakeTimeSeriesPlot(object)
                    elif object['type'] == 'ProfilePlot':
                        self.MakeProfilePlot(object)
                    elif object['type'] == 'ErrorStatisticsTable':
                        self.MakeErrorStatisticsTable(object)
                    elif object['type'] == 'MonthlyStatisticsTable':
                        self.MakeMonthlyStatisticsTable(object)
                    elif object['type'] == 'BuzzPlot':
                        self.MakeBuzzPlot(object)
                    else:
                        print('Section Type {0} not identified.'.format(section['type']))
                        print('Skipping Section..')
                self.XML.writeSectionHeaderEnd()
            self.XML.writeChapterEnd()

    def cleanOutputDirs(self):
        '''
        cleans the images output directory, so pngs from old reports aren't mistakenly
        added to new reports. Creates directory if it doesn't exist.
        '''

        if not os.path.exists(self.images_path):
            os.makedirs(self.images_path)

        WF.clean_output_dir(self.images_path, '.png')

    def LoadModelAlt(self, simCSVAlt):
        '''
        Loads info for specified model alts. Loads correct model plugin class from WDR
        :param simCSVAlt: simulation alt dict object from self.simulation class
        :return: class variables
                self.alternativeFpart
                self.alternativeDirectory
                self.ModelAlt - WDR class that is plugin specific
        '''

        for modelalt in self.ModelAlternatives:
            if modelalt['name'] == simCSVAlt['modelaltname'] and modelalt['program'] == simCSVAlt['plugin']:
                self.alternativeFpart = modelalt['fpart']
                self.alternativeDirectory = modelalt['directory']
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

    def EnsureDefaultFiles(self):
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

    def load_defaultPlotObject(self, plotobject):
        '''
        loads the graphic default options. Uses deepcopy so residual settings are not carried over
        :param plotobject: string specifying the default graphics object
        :return:
            plot_info: dict of object settings
        '''

        plot_info = copy.deepcopy(self.graphicsDefault[plotobject])
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

        if isinstance(dates, list) or isinstance(dates, np.ndarray):
            jdates = [(WF.dt_to_ord(n) - self.ModelAlt.t_offset) + 1 for n in dates]
            return jdates
        elif isinstance(dates, dt.datetime):
            jdate = (WF.dt_to_ord(dates) - self.ModelAlt.t_offset) + 1
            return jdate
        else:
            print('Unable to convert to JDates')
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
        if isinstance(dates, list) or isinstance(dates, np.ndarray):
            dtimes = [first_year_Date + dt.timedelta(days=n) for n in dates]
            return dtimes
        elif isinstance(dates, float) or isinstance(dates, int):
            dtime = first_year_Date + dt.timedelta(days=dates)
            return dtime
        else:
            print('Unable to convert to datetime')
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

    def BuzzTargetSum(self, dates, values, target):
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
        return sum_vals

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

        english_units = {'m3/s':'cfs',
                         'm': 'ft',
                         'm3': 'af',
                         'c': 'f'}
        metric_units = {v: k for k, v in english_units.items()}

        if units == None:
            return values, units

        units = self.translateUnits(units)

        #Following is the SOURCE units, then the conversion to units listed above
        conversion = {'m3/s': 35.314666213,
                      'cfs': 0.0283168469997284,
                      'm': 3.28084,
                      'ft': 0.3048,
                      'm3': 0.000810714,
                      'af': 1233.48}

        if target_unitsystem.lower() == 'english':
            if units.lower() in english_units.keys():
                new_units = english_units[units.lower()]
            elif units.lower() in english_units.values():
                print('Values already in target unit system. {0} {1}'.format(units, target_unitsystem))
                return values, units
            else:
                print('Units not found in definitions. Not Converting.')
                return values, units
        elif target_unitsystem.lower() == 'metric':
            if units.lower() in metric_units.keys():
                new_units = metric_units[units.lower()]
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

        if units.lower() in ['c', 'f']:
            values = WF.convertTempUnits(values, units)
        elif units.lower() in conversion.keys():
            conversion_factor = conversion[units.lower()]
            values = np.asarray(values) * conversion_factor
        elif new_units.lower() in conversion.keys():
            conversion_factor = 1/conversion[units.lower()]
            values = np.asarray(values) * conversion_factor
        else:
            print('Undefined Units conversion for units {0}.'.format(units))
            print('No Conversions taking place.')
            return values, units

        return values, new_units

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
            # r = []
        return rows

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
        for i, header in enumerate(object_settings['headers']):
            if isinstance(object_settings['headers'], dict):
                header = object_settings['headers']['header'] #single headers come as dict objs TODO fix this eventually...
            if '%%year%%' in header:
                if split_by_year:
                    header_by_year.append(header)
                else:
                    headings.append(['ALL', self.updateFlaggedValues(header, '%%year%%', str(years[0]))])
            else:
                if len(header_by_year) > 0:
                    for year in years:
                        for yrhd in header_by_year:
                            headings.append([year, self.updateFlaggedValues(yrhd, '%%year%%', str(year))])
                    header_by_year = []
                headings.append(['ALL', header])
        if len(header_by_year) > 0:
            for year in years:
                for yrhd in header_by_year:
                    headings.append([year, self.updateFlaggedValues(yrhd, '%%year%%', str(year))])
        return headings

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

        elif isinstance(settings, dict):
            for key in settings.keys():
                settings[key] = self.updateFlaggedValues(settings[key], flaggedvalue, replacevalue)
            return settings

        elif isinstance(settings, str):
            pattern = re.compile(re.escape(flaggedvalue), re.IGNORECASE)
            settings = pattern.sub(repr(replacevalue)[1:-1], settings) #this seems weird with [1:-1] but paths wont work otherwise
            return settings

        else:
            print('Input Not recognized type', settings)
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
        if isinstance(times[0], (int, float)): #chcek for jdate, this is easier in dt..
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

        if 'interval' in Line_info:
            interval = Line_info['interval'].upper()
        else:
            interval = self.getTimeInterval(times)

        input_interval = self.getTimeInterval(times)

        new_times = self.buildTimeSeries(times[0], times[-1], interval)
        new_times_interval = self.getTimeInterval(new_times)

        # if isinstance(new_times_interval, (int, float)):
        #     if new_times_interval < input_interval:
        #         print('New interval is smaller than the old interval.')
        #         print('Currently not supporting this.')
        #         return times, values
        if isinstance(new_times_interval, dt.timedelta):
            if new_times_interval.total_seconds() < input_interval.total_seconds():
                print('New interval is smaller than the old interval.')
                print('Currently not supporting this.')
                if convert_to_jdate:
                    return self.DatetimeToJDate(times), values
                else:
                    return times, values
        else:
            print('Error defining new interval types')
            if convert_to_jdate:
                return self.DatetimeToJDate(times), values
            else:
                return times, values

        if new_times_interval == input_interval: #no change..
            print('Input interval matches data interval.')
            if convert_to_jdate:
                return self.DatetimeToJDate(times), values
            else:
                return times, values

        if isinstance(values, dict):
            new_values = {}
            for key in values:
                new_times, new_values[key] = self.changeTimeSeriesInterval(times, values[key], Line_info)
        else:
            values = np.asarray(values)
            if avgtype == 'INST-VAL':
                #at the point in time, find intervals and use values
                new_values = []
                for t in new_times:
                    ti = np.where(np.asarray(times) == t)[0]
                    if len(ti) == 1:
                        new_values.append(np.asarray(values)[ti][0])
                    elif len(ti) == 0:
                        #missing date, could be missing data or irreg?
                        new_values.append(np.NaN)
                    else:
                        print('Multiple date idx found??')
                        new_values.append(np.NaN)
                        continue
                # return new_times, new_values

            elif avgtype == 'INST-CUM':
                if interval == input_interval:
                    new_times = copy.deepcopy(times)
                new_values = []
                for t in new_times:
                    ti = [i for i, n in enumerate(times) if n <= t]
                    cum_val = np.sum(values[ti])
                    new_values.append(cum_val)

                # return new_times, new_values

            elif avgtype == 'PER-AVER':
                #average over the period
                new_values = []
                interval = self.getTimeIntervalSeconds(interval)
                for t in new_times:
                    t2 = t + dt.timedelta(seconds=interval)
                    date_idx = [i for i, n in enumerate(times) if t <= n < t2]
                    new_values.append(np.mean(values[date_idx]))
                # return new_times, new_values

            elif avgtype == 'PER-CUM':
                #cum over the perio
                new_values = []
                interval = self.getTimeIntervalSeconds(interval)
                for t in new_times:
                    t2 = t + dt.timedelta(seconds=interval)
                    date_idx = [i for i, n in enumerate(times) if t <= n < t2]
                    new_values.append(np.sum(values[date_idx]))
                # return new_times, new_values

        if convert_to_jdate:
            return self.DatetimeToJDate(new_times), new_values
        else:
            return new_times, new_values


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
        return timesteps

if __name__ == '__main__':
    rundir = sys.argv[0]
    simInfoFile = sys.argv[1]
    # simInfoFile = r"\\wattest\C\WAT\USBR_FrameworkTest_r3_singlescript\reports\ResSim-val2013.xml"

    ar = MakeAutomatedReport(simInfoFile, rundir)

