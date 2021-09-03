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
import xml.etree.ElementTree as ET
import copy
import calendar
import dateutil.parser
from collections import Counter
from dateutil.relativedelta import relativedelta

import WAT_DataReader as WDR
import WAT_Functions as WF
import WAT_XML_Utils as XML_Utils


class MakeAutomatedReport(object):
    '''
    class to organize data and generate XML file for Jasper processing in conjunction with WAT. Takes in a simulation
    information file output from WAT and develops the report from there.
    '''

    def __init__(self, simulationInfoFile):
        '''
        organizes input data and generates XML report
        :param simulationInfoFile: full path to simulation information XML file output from WAT.
        '''

        self.ReadSimulationInfo(simulationInfoFile) #read file output by WAT
        self.ReadGraphicsDefaultFile() #read graphical component defaults
        self.ReadLinesstylesDefaultFile()
        self.DefineUnits()
        self.DefinePaths()
        self.DefineMonths()
        self.DefineTimeIntervals()
        self.DefineDefaultColors()
        if self.reportType == 'single': #Eventually be able to do comparison reports, put that here
            for simulation in self.Simulations:
                # print('SIMULATION:', simulation)
                self.setSimulationVariables(simulation)
                self.DefineStartEndYears()
                self.ReadSimulationsCSV() #read to determine order/sims/regions in report
                self.cleanOutputDirs()
                self.initializeXML()
                self.writeXMLIntroduction()
                for simorder in self.SimulationCSV.keys():
                    self.SetSimulationCSVVars(self.SimulationCSV[simorder])
                    self.ReadDefinitionsFile(simorder)
                    self.LoadModelAlt(self.SimulationCSV[simorder])
                    self.WriteChapter()
                self.XML.writeReportEnd()

    def ReadSimulationInfo(self, simulationInfoFile):
        '''
        reads in the simulation information XML file from WAT and organizes data into two dictionaries
        One for model alternatives, using the simulation name as the main key
        Another for overall simulation information, using a simlation name as the main key
        Other universal vars for the entire WAT project and intended report type
        :param simulationInfoFile: full path to simulation information XML file from WAT
        :return: class variables:
                    self.modelAlternatives
                    self.simulationInfo
                    self.reportType
                    self.studyDir
                    self.observedData
        '''

        # self.modelAlternatives = {}
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

    def cleanOutputDirs(self):
        '''
        cleans the images output directory, so pngs from old reports aren't mistakenly
        added to new reports. Creates directory if it doesn't exist.
        '''

        if not os.path.exists(self.images_path):
            os.makedirs(self.images_path)

        WF.clean_output_dir(self.images_path, '.png')

    def LoadModelAlt(self, simCSVAlt):
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
        copys a new version of the template XML file, initiates the XML class and writes the cover page
        :return: sets class variables
                    self.XML
        '''
        new_xml = 'USBRAutomatedReportOutput.xml' #required name for file
        self.XML = XML_Utils.XMLReport(new_xml)
        self.XML.writeCover('DRAFT Temperature Validation Summary Report')

    def writeXMLIntroduction(self):
        self.XML.writeIntroStart()
        for model in self.SimulationCSV.keys():
            self.XML.writeIntroLine(self.SimulationCSV[model]['plugin'])
        self.XML.writeIntroEnd()

    def getModelAltForRegion(self, order):
        '''
        Reads the region information to find and define new parameters. If the model alt and plugin does not
        exist in the WAM simulated, skip that model alt
        :param order: number of region in region information file
        :return: sets class variables:
                    self.modelAltName
                    self.region
                    self.plugin
                 Boolean value of model identification status
        '''

        self.modelAltName = self.regInfo[order]['modelaltname'].strip()
        self.region = self.regInfo[order]['region'].strip()
        self.plugin = self.regInfo[order]['plugin'].strip()
        _model_alt_set = False
        for model_alt in self.modelAlternatives[self.simulationName].keys():
            cur_MA_name = model_alt
            cur_MA_plugin = self.modelAlternatives[self.simulationName][model_alt]['program']
            if cur_MA_name == self.modelAltName and cur_MA_plugin == self.plugin:
                self.alternativeFpart = self.modelAlternatives[self.simulationName][model_alt]['fpart']
                _model_alt_set = True
                break

        if _model_alt_set:
            print('Model Alternative Found:', self.modelAltName)
            return True
        else:
            print('Model Alternative NOT found:', self.modelAltName, self.region, self.plugin)
            return False

    def DefineMonths(self):
        '''
        defines month 3 letter codes for table labels
        :return: sets class variables
                    self.mo_str_3
        '''
        self.month2num = {month.lower(): index for index, month in enumerate(calendar.month_abbr) if month}
        self.num2month = {index: month.lower() for index, month in enumerate(calendar.month_abbr) if month}
        self.mo_str_3 = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    def DefineStartEndYears(self):
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
        self.time_intervals = {'1MIN': dt.timedelta(minutes=1),
                               '2MIN': dt.timedelta(minutes=2),
                               '5MIN': dt.timedelta(minutes=5),
                               '6MIN': dt.timedelta(minutes=6),
                               '10MIN': dt.timedelta(minutes=10),
                               '12MIN': dt.timedelta(minutes=12),
                               '15MIN': dt.timedelta(minutes=15),
                               '30MIN': dt.timedelta(minutes=30),
                               '1HOUR': dt.timedelta(hours=1),
                               '2HOUR': dt.timedelta(hours=2),
                               '3HOUR': dt.timedelta(hours=3),
                               '4HOUR': dt.timedelta(hours=4),
                               '5HOUR': dt.timedelta(hours=5),
                               '6HOUR': dt.timedelta(hours=6),
                               '1DAY': dt.timedelta(days=1),
                               '1MON': relativedelta(months=1),
                               '2MON': relativedelta(months=2),
                               '6MON': relativedelta(months=6),
                               '1YEAR': relativedelta(years=1)}

    def DefineDefaultColors(self):
        self.def_colors = ['darkgreen', 'red', 'blue', 'orange', 'darkcyan', 'darkmagenta', 'gray', 'black']

    def DefineUnits(self):
        '''
        creates dictionary with units for vars for labels
        :return: set class variable
                    self.units
        '''

        WQ_metrics = ['temperature', 'do', 'do_sat']
        WQ_units = ['C', 'mg/L', '%']
        metrics = ['flow', ] + WQ_metrics
        metric_units = ['m3/s', ] + WQ_units
        self.units = dict(zip(metrics, metric_units))

    def DefinePaths(self):
        '''
        defines run specific paths
        :return: set class variables
                    self.images_path
                    self.ProfileStations_meta_file
                    self.TimeSeries_Stations_meta_file
        '''

        self.images_path = os.path.join(self.studyDir, 'reports', 'Images')
        self.ProfileStations_meta_file = os.path.join(self.studyDir, 'reports', "Profile_stations.txt")
        self.TimeSeries_Stations_meta_file = os.path.join(self.studyDir, 'reports', "TS_stations.txt")

    def setSimulationDateTimes(self):
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


    def ReadSimulationsCSV(self):
        '''
        finds the correct Simulation file and gets the region info
        :return: sets class variable
                    self.SimInfo
        '''
        self.SimulationCSV = WDR.ReadSimulationFile(self.baseSimulationName, self.studyDir)

    def SetSimulationCSVVars(self, simlist):
        self.plugin = simlist['plugin']
        self.modelAltName = simlist['modelaltname']
        self.defFile = simlist['deffile']

    def ReadGraphicsDefaultFile(self):
        graphicsDefaultfile = os.path.join(self.studyDir, 'reports', 'Graphics_Defaults_Beta_v6.xml') #TODO: finalize the path here, should it live in study?
        self.graphicsDefault = WDR.read_GraphicsDefaults(graphicsDefaultfile)

    def ReadLinesstylesDefaultFile(self):
        defaultLinesFile = os.path.join(self.studyDir, 'reports', 'defaultLineStyles_Beta.xml')
        self.defaultLineStyles = WDR.read_DefaultLineStyle(defaultLinesFile)

    def ReadDefinitionsFile(self, order):
        ChapterDefinitionsFile = os.path.join(self.studyDir, 'reports', self.SimulationCSV[order]['deffile'])
        self.ChapterDefinitions = WDR.read_ChapterDefFile(ChapterDefinitionsFile)

    def setSimulationVariables(self, simulation):
        self.SimulationName = simulation['name']
        self.baseSimulationName = simulation['basename']
        self.simulationDir = simulation['directory']
        self.DSSFile = simulation['dssfile']
        self.StartTimeStr = simulation['starttime']
        self.EndTimeStr = simulation['endtime']
        self.LastComputed = simulation['lastcomputed']
        self.ModelAlternatives = simulation['modelalternatives']
        self.setSimulationDateTimes()

    def WriteChapter(self):
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


    def load_defaultPlotObject(self, plotobject):
        plot_info = copy.deepcopy(self.graphicsDefault[plotobject])
        return plot_info

    def getLineDefaultSettings(self, LineSettings, param, i):
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

            default_default_lines = self.get_DefaultDefaultLineStyles(i)
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

            default_default_points = self.get_DefaultDefaultPointStyles(i)

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

    def translateLineStylePatterns(self, LineSettings):
        #java|python
        linestylesdict = {'dash': 'dashed',
                          'dash dot': 'dashdot',
                          'dash dot-dot': (0, (3, 5, 1, 5, 1, 5)),
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

    def get_DefaultDefaultLineStyles(self, i):
        if i >= len(self.def_colors):
            i = i - len(self.def_colors)
        return {'linewidth': 2, 'linecolor': self.def_colors[i], 'linestyle': 'solid'}

    def get_DefaultDefaultPointStyles(self, i):
        if i >= len(self.def_colors):
            i = i - len(self.def_colors)
        return {'pointfillcolor': self.def_colors[i], 'pointlinecolor': self.def_colors[i], 'symboltype': 1,
                'symbolsize': 5, 'numptsskip': 0}

    def replaceDefaults(self, default_settings, object_settings):
        #replace flagged values
        #replace defaults

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
        if isinstance(settings, str):
            if '$$' in settings:
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
                    if '$$' in settings[key]:
                        newval = self.replaceFlaggedValue(settings[key])
                        settings[key] = newval
        elif isinstance(settings, list):
            for i, item in enumerate(settings):
                if '$$' in item:
                    settings[i] = self.replaceFlaggedValue(item)

        return settings

    def replaceFlaggedValue(self, value):
        flagged_values = {'$$ModelDSS$$': self.DSSFile,
                          '$$region$$': self.ChapterRegion,
                          '$$Fpart$$': self.alternativeFpart,
                          '$$plugin$$': self.plugin,
                          '$$modelAltName$$': self.modelAltName,
                          '$$SimulationName$$': self.SimulationName,
                          '$$baseSimulationName$$': self.baseSimulationName,
                          '$$starttime$$': self.StartTimeStr,
                          '$$endtime$$': self.EndTimeStr,
                          '$$LastComputed$$': self.LastComputed,
                          '$$observedDir$$': self.observedDir,
                          '$$startyear$$': str(self.StartTime.year),
                          '$$endyear$$': str(self.EndTime.year)
                          }

        for fv in flagged_values.keys():
            value = value.replace(fv, flagged_values[fv])
        return value


    def MakeTimeSeriesPlot(self, object_settings):
        default_settings = self.load_defaultPlotObject('timeseriesplot') #get default TS plot items
        object_settings = self.replaceDefaults(default_settings, object_settings) #overwrite the defaults with chapter file

        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot()
        param_count = {}

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

            line_settings = self.getLineDefaultSettings(line, param, i)

            if 'zorder' not in line_settings.keys():
                line_settings['zorder'] = 4

            if 'label' not in line_settings.keys():
                line_settings['label'] = ''

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
        These plots need to go through the following steps:
        get the times for plots - either from available obs data, or defined dates from the chapter XML, or just a set
                                  interval in the start and end dates
        determine the plot parameter - get consensus from inputs?
        split plots up
        '''
        self.XML.writeProfilePlotStart(object_settings['description'])
        default_settings = self.load_defaultPlotObject('profileplot')
        object_settings = self.replaceDefaults(default_settings, object_settings)

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
        if plot_parameter.lower() in self.units.keys():
            units = self.units[plot_parameter.lower()]
        else:
            units = None
        object_settings = self.updateFlaggedValues(object_settings, '$$units$$', units)

        ################# Get data #################
        #do now incase no elevs, so we can convert
        for line in object_settings['lines']:
            vals, elevations, depths, flag = self.getProfileData(line, timestamps) #Test this speed for grabbing all profiles and then choosing

            linedata[flag] = {'values':vals,
                              'elevations':elevations,
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

            for page_i, pgi in enumerate(page_indices):

                subplot_rows, subplot_cols = WF.get_subplot_config(len(pgi))
                n_nrow_active = np.ceil(len(pgi) / subplot_cols)
                fig = plt.figure(figsize=(7, 1 + 3 * n_nrow_active))

                current_object_settings = self.updateFlaggedValues(object_settings, '$$year$$', yearstr)

                for i, j in enumerate(pgi):
                    ax = fig.add_subplot(int(subplot_rows), int(subplot_cols), i + 1)

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

                    show_legend, show_xlabel, show_ylabel = self.get_plot_label_masks(i, len(pgi), subplot_cols)

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
                if split_by_year:
                    description = '{0} {1}: {2} of {3}'.format(current_object_settings['description'], yearstr, page_i+1, len(page_indices))
                else:
                    description = '{0}: {1} of {2}'.format(current_object_settings['description'], page_i+1, len(page_indices))
                self.XML.writeProfilePlotFigure(figname, description)

        self.XML.writeProfilePlotEnd()

    def MakeErrorStatisticsTable(self, section_settings):
        default_settings = self.load_defaultPlotObject('errorstatisticstable')
        object_settings = self.replaceDefaults(default_settings, section_settings)
        datapaths = object_settings['datapaths']

        if 'parameter' in object_settings.keys():
            if object_settings['parameter'].lower() in self.units.keys():
                units = self.units[object_settings['parameter'].lower()]
                object_settings = self.updateFlaggedValues(object_settings, '$$units$$', units)
            else:
                print('param not in units', object_settings['parameter'], self.units.keys())
        else:
            print('no param')

        years = self.years

        split_by_year = False
        if 'splitbyyear' in object_settings.keys():
            if object_settings['splitbyyear'].lower() == 'true':
                split_by_year = True
        if not split_by_year:
            years = [self.years_str]

        headings = self.buildHeadersByYear(object_settings, years, split_by_year)
        rows = self.buildRowsByYear(object_settings, years)

        data = {}
        for dp in datapaths:
            dates, values, units = self.getTimeSeries(dp)
            if units == None:
                if 'parameter' in dp.keys():
                    if dp['parameter'].lower() in self.units.keys():
                        units = self.units[dp['parameter'].lower()]
                        object_settings = self.updateFlaggedValues(object_settings, '$$units$$', units)

            data[dp['flag']] = {'dates':dates,
                                'values': values,
                                'units': units}

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
                if '$$' in row_val:
                    row_val = self.formatStatsLine(row_val, data, year=year)
                header = '' if header == None else header
                frmt_rows.append('{0}|{1}'.format(rowname, row_val))
            self.XML.writeTableColumn(header, frmt_rows)
        self.XML.writeTableEnd()

    def MakeMonthlyStatisticsTable(self, section_settings):
        default_settings = self.load_defaultPlotObject('monthlystatisticstable')
        object_settings = self.replaceDefaults(default_settings, section_settings)
        datapaths = object_settings['datapaths']

        if 'parameter' in object_settings.keys():
            if object_settings['parameter'].lower() in self.units.keys():
                units = self.units[object_settings['parameter'].lower()]
                object_settings = self.updateFlaggedValues(object_settings, '$$units$$', units)
                print('units', units)
            else:
                print('param not in units', object_settings['parameter'], self.units.keys())
        else:
            print('no param')

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
        rows = self.buildRowsByYear(object_settings, years)

        data = {}
        for dp in datapaths:
            dates, values, units = self.getTimeSeries(dp)
            if units == None:
                if 'parameter' in dp.keys():
                    if dp['parameter'].lower() in self.units.keys():
                        units = self.units[dp['parameter'].lower()]

            data[dp['flag']] = {'dates':dates,
                                'values': values,
                                'units': units}

        self.XML.writeTableStart(object_settings['description'], 'Month')
        for i, yearheader in enumerate(headings):
            year = yearheader[0]
            header = yearheader[1]
            frmt_rows = []
            for row in rows:
                s_row = row.split('|')
                rowname = s_row[0]
                row_val = s_row[i+1]
                if '$$' in row_val:
                    row_val = self.formatStatsLine(row_val, data, year=year)
                header = '' if header == None else header
                frmt_rows.append('{0}|{1}'.format(rowname, row_val))
            self.XML.writeTableColumn(header, frmt_rows)
        self.XML.writeTableEnd()

    def MakeBuzzPlot(self, section_settings):
        default_settings = self.load_defaultPlotObject('buzzplot')
        object_settings = self.replaceDefaults(default_settings, section_settings)
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

    def formatDateXAxis(self, curax, object_settings, twin=False):
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
            print('Dateformat flag not set. Defualting to datetime..')
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

    def translateDateFormat(self, lim, dateformat, fallback):
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

    def DatetimeToJDate(self, dates):
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
        sum_vals = []
        for i, d in enumerate(dates):
            sum = 0
            for sn in values.keys():
                if values[sn]['elevcl'][i] == target:
                    sum += values[sn]['q(m3/s)'][i]
            sum_vals.append(sum)
        return sum_vals

    def convertUnitSystem(self, values, units, target_unitsystem):
        english_units = {'m3/s':'cfs',
                         'm': 'ft',
                         'm3': 'af',
                         'c': 'f'}
        metric_units = {v: k for k, v in english_units.items()}

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

    def buildRowsByYear(self, object_settings, years):
        rows = []
        if 'rows' in object_settings.keys():
            if isinstance(object_settings['rows'], dict): #single row
                object_settings['rows'] = [object_settings['rows']['row']]
            if isinstance(object_settings['rows'], list):
                for row in object_settings['rows']:
                    srow = row.split('|')
                    r = [srow[0]] #make list, add the row name
                    for sr in srow[1:]:
                        for year in years:
                            r.append(sr)
                    rows.append('|'.join(r))
        return rows

    def buildHeadersByYear(self, object_settings, years, split_by_year):
        headings = []
        header_by_year = []
        for i, header in enumerate(object_settings['headers']):
            if isinstance(object_settings['headers'], dict):
                header = object_settings['headers']['header'] #single headers come as dict objs TODO fix this eventually...
            if '$$year$$' in header:
                if split_by_year:
                    header_by_year.append(header)
                else:
                    headings.append(['ALL', self.updateFlaggedValues(header, '$$year$$', str(years[0]))])
            else:
                if len(header_by_year) > 0:
                    for year in years:
                        for yrhd in header_by_year:
                            headings.append([year, self.updateFlaggedValues(yrhd, '$$year$$', str(year))])
                    header_by_year = []
                headings.append(['ALL', header])
        if len(header_by_year) > 0:
            for year in years:
                for yrhd in header_by_year:
                    headings.append([year, self.updateFlaggedValues(yrhd, '$$year$$', str(year))])
        return headings

    def filterTimestepByYear(self, timestamps, year):
        if year == 'ALLYEARS':
            return timestamps
        return [n for n in timestamps if n.year == year]

    def formatStatsLine(self, row, data_dict, year='ALL'):
        data = copy.deepcopy(data_dict)
        rrow = row.replace('$$', '')
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

        if row.lower().startswith('$$meanbias'):
            return WF.meanbias(data[flags[0]], data[flags[1]])
        elif row.lower().startswith('$$mae'):
            return WF.MAE(data[flags[0]], data[flags[1]])
        elif row.lower().startswith('$$rmse'):
            return WF.RMSE(data[flags[0]], data[flags[1]])
        elif row.lower().startswith('$$nse'):
            return WF.NSE(data[flags[0]], data[flags[1]])
        elif row.lower().startswith('$$count'):
            return WF.COUNT(data[flags[0]])
        elif row.lower().startswith('$$mean'):
            return WF.MEAN(data[flags[0]])
        else:
            if '$$' in row:
                print('Unable to convert flag in row', row)
            return row

    def getLineSettings(self, LineSettings, Flag):
        for line in LineSettings:
            if Flag == line['flag']:
                return copy.deepcopy(line)

    def updateFlaggedValues(self, settings, flaggedvalue, replacevalue):
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
            settings = settings.replace(flaggedvalue, replacevalue)
            return settings

        else:
            print('Input Not recognized type', settings)
            return settings

    def get_plot_label_masks(self, idx, nprofiles, cols):
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
        if 'dss_path' in Line_info.keys(): #Get data from DSS record
            if 'dss_filename' not in Line_info.keys():
                print('DSS_Filename not set for Line: {0}'.format(Line_info))
                return [], [], None
            else:
                times, values, units = WDR.readDSSData(Line_info['dss_filename'], Line_info['dss_path'],
                                                       self.StartTime, self.EndTime)

                if np.any(values == None):
                    return [], [], None
                elif len(values) == 0:
                    return [], [], None

                values = WF.convertTempUnits(values, units)

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
                units = self.units[Line_info['parameter'].lower()]

        elif 'xy' in Line_info.keys():
            times, values = self.ModelAlt.readTimeSeries(Line_info['parameter'], Line_info['xy'])
            units = self.units[Line_info['parameter'].lower()]

        else:
            print('No Data Defined for line')
            return [], [], None

        if len(values) == 0:
            return [], [], None
        else:
            times, values = self.changeTimeSeriesInterval(times, values, Line_info)
            return times, values, units

    def changeTimeSeriesInterval(self, times, values, Line_info):
        # INST-CUM, INST-VAL, PER-AVER, PER-CUM)
        if 'type' in Line_info:
            type = Line_info['type'].upper()
        else:
            type = 'INST-VAL'

        if 'interval' in Line_info:
            interval = Line_info['interval'].upper()
        else:
            interval = self.getTimeInterval(times)

        input_interval = self.getTimeInterval(times)

        if type == 'INST-VAL':
            #at the point in time, find intervals and use values
            if interval == input_interval: #no change..
                return times, values
            # TODO: What if time interval is lower than existing ts?
            new_times = self.buildTimeSeries(times[0], times[-1], interval)
            new_values = []
            for t in new_times:
                ti = np.where(np.asarray(times)==t)[0]
                if len(ti) == 1:
                    new_values.append(values[ti])
                elif len(ti) == 0:
                    #missing date, could be missing data or irreg?
                    new_values.append(None)
                else:
                    print('Multiple date idx found??')
                    new_values.append(None)
                    continue
            return new_times, new_values

        elif type == 'INST-CUM':
            if interval != input_interval:
                new_times = self.buildTimeSeries(times[0], times[-1], interval)
            else:
                new_times = copy.deepcopy(times)
            new_values = []
            for t in new_times:
                ti = [i for i, n in enumerate(times) if n <= t]
                cum_val = np.sum(values[ti])
                new_values.append(cum_val)

            return new_times, new_values

        elif type == 'PER-AVER':
            #average over the period
            if interval == input_interval: #no change..
                return times, values
            # TODO: What if time interval is lower than existing ts?
            new_times = self.buildTimeSeries(times[0], times[-1], interval)
            new_values = []
            interval = self.getTimeIntervalSeconds(interval)
            for t in new_times:
                t2 = t + dt.timedelta(seconds=interval)
                date_idx = [i for i, n in enumerate(times) if t <= n < t2]
                new_values.append(np.mean(values[date_idx]))
            return new_times, new_values

        elif type == 'PER-CUM':
            #cum over the perio
            if interval == input_interval: #no change..
                return times, values
                # TODO: What if time interval is lower than existing ts?
            new_times = self.buildTimeSeries(times[0], times[-1], interval)
            new_values = []
            interval = self.getTimeIntervalSeconds(interval)
            for t in new_times:
                t2 = t + dt.timedelta(seconds=interval)
                date_idx = [i for i, n in enumerate(times) if t <= n < t2]
                new_values.append(np.sum(values[date_idx]))
            return new_times, new_values

    def buildTimeSeries(self, startTime, endTime, interval):
        interval = self.time_intervals[interval]
        ts = np.arange(startTime, endTime, interval)
        ts = [t.astype(dt.datetime) for t in ts]
        return ts

    def getTimeIntervalSeconds(self, interval):
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
        t_ints = []
        for i, t in enumerate(times):
            if i == 0: #skip 1
                last_time = t
            else:
                t_ints.append(t - last_time)

        occurence_count = Counter(t_ints)
        most_common_interval = occurence_count.most_common(1)[0][0]
        return most_common_interval


    def getProfileData(self, Line_info, timesteps):

        if 'filename' in Line_info.keys(): #Get data from Observed
            # times, values, depths = WDR.readTextProfile(os.path.join(self.observedDir, Line_info['FileName']), self.StartTime, self.EndTime)
            filename = os.path.join(self.observedDir, Line_info['filename'])
            values, depths = WDR.readTextProfile(filename, timesteps)
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
        if 'filename' in Line_info.keys(): #Get data from Observed
            times = WDR.getTextProfileDates(os.path.join(self.observedDir, Line_info['filename']), self.StartTime, self.EndTime) #TODO: set up for not observed data??
            return times

        print('Illegal Dates selection. ')
        return []

    def makeRegularTimesteps(self, days=15):
        timesteps = []
        print('No Timesteps found. Setting to Regular interval')
        cur_date = self.StartTime
        while cur_date < self.EndTime:
            timesteps.append(cur_date)
            cur_date += dt.timedelta(days=days)
        return timesteps

if __name__ == '__main__':
    simInfoFile = sys.argv[1]
    # simInfoFile = r"\\wattest\C\WAT\USBR_FrameworkTest_r3_singlescript\reports\ResSim-val2013.xml"

    ar = MakeAutomatedReport(simInfoFile)

