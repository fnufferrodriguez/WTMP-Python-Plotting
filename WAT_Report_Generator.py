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

VERSIONNUMBER = '5.3.5b'

import os
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pickle
import itertools
import traceback
import time

import WAT_Reader as WR
import WAT_Functions as WF
import WAT_XML_Utils as WXMLU
import WAT_Logger as WL
import WAT_Constants as WC
import WAT_Time as WT
import WAT_Defaults as WD
import WAT_DataOrganizer as WDO
import WAT_ResSim_Results as WRSS
import WAT_W2_Results as WW2
import WAT_Profiles as WProfile
import WAT_Tables as WTable
import WAT_Plots as WPlot
import WAT_Gates as WGates

import warnings
warnings.filterwarnings("always")

mpl.rcParams['axes.autolimit_mode'] = 'round_numbers'
mpl.use("Agg")

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
        WF.printVersion(VERSIONNUMBER)
        self.simulationInfoFile = simulationInfoFile
        self.WriteLog = True #TODO we're testing this.
        self.batdir = batdir
        WR.readSimulationInfo(self, simulationInfoFile) #read file output by WAT
        self.definePaths()
        self.Constants = WC.WAT_Constants()
        self.cleanOutputDirs()
        WF.checkJasperFiles(self.studyDir, self.installDir)
        WR.readGraphicsDefaultFile(self) #read graphical component defaults
        self.defaultLineStyles = WD.readDefaultLineStylesFile(self)
        self.WAT_log = WL.WAT_Logger(self)
        self.modelOrder = 0

        if self.reportType == 'single': #Eventually be able to do comparison reports, put that here
            for simulation in self.Simulations:
                WF.print2stdout('Running Simulation:', simulation)
                self.initSimulationDict()
                self.setSimulationVariables(simulation)
                self.loadCurrentID('base') #load the data for the current sim, we do 1 at a time here..
                WF.checkExists(self.SimulationDir)
                WT.defineStartEndYears(self)
                WR.readSimulationsCSV(self) #read to determine order/sims/regions in report
                self.cleanOutputDirs()
                self.initializeXML()
                self.writeXMLIntroduction()
                for simorder in self.SimulationCSV.keys():
                    self.setSimulationCSVVars(self.SimulationCSV[simorder])
                    WR.readDefinitionsFile(self, self.SimulationCSV[simorder])
                    self.loadModelAlts(self.SimulationCSV[simorder])
                    self.initializeDataOrganizer()
                    self.loadCurrentModelAltID('base')
                    self.WAT_log.addSimLogEntry(self.accepted_IDs, self.SimulationVariables, self.observedDir)
                    self.writeChapter()
                    self.appendXMLModelIntroduction(simorder)
                    self.Data.writeDataFiles()
                self.fixXMLModelIntroduction()
                self.XML.writeReportEnd()
                self.WAT_log.equalizeLog()
        elif self.reportType == 'alternativecomparison':
            self.initSimulationDict()
            for simulation in self.Simulations:
                self.setSimulationVariables(simulation)
                WF.checkExists(simulation['directory'])
            self.loadCurrentID('base') #load the data for the current sim, we do 1 at a time here..
            WT.setMultiRunStartEndYears(self) #find the start and end time
            WT.defineStartEndYears(self) #format the years correctly after theyre set
            WR.readComparisonSimulationsCSV(self) #read to determine order/sims/regions in report
            self.cleanOutputDirs()
            self.initializeXML()
            self.writeXMLIntroduction()
            for simorder in self.SimulationCSV.keys():
                self.setSimulationCSVVars(self.SimulationCSV[simorder])
                WR.readDefinitionsFile(self, self.SimulationCSV[simorder])
                self.loadModelAlts(self.SimulationCSV[simorder])
                self.initializeDataOrganizer()
                self.loadCurrentModelAltID('base')
                self.WAT_log.addSimLogEntry(self.accepted_IDs, self.SimulationVariables, self.observedDir)
                self.writeChapter()
                self.appendXMLModelIntroduction(simorder)
                self.Data.writeDataFiles()
            self.fixXMLModelIntroduction()
            self.XML.writeReportEnd()
            self.WAT_log.equalizeLog()
        else:
            WF.print2stderr('UNKNOWN REPORT TYPE:', self.reportType)
            sys.exit(1)
        self.WAT_log.writeLogFile(self.images_path)
        self.Data.writeDataFiles()

    def definePaths(self):
        '''
        defines run specific paths
        used to contain more paths, but not needed. Consider moving.
        :return: set class variables
                    self.images_path
        '''

        # self.images_path = os.path.join(self.studyDir, 'reports', 'Images')
        self.images_path = os.path.join(self.outputDir, 'Images')
        if not os.path.exists(self.images_path):
            try:
                os.makedirs(self.images_path)
                WF.print2stdout(f'{self.images_path} created!')
            except:
                WF.print2stderr(f'Unable to make {self.images_path}')
                sys.exit(1)

        # self.CSVPath = os.path.join(self.studyDir, 'reports', 'CSVData')
        self.CSVPath = os.path.join(self.outputDir, 'CSVData') #TODO: update

        if not os.path.exists(self.CSVPath):
            try:
                os.makedirs(self.CSVPath)
                WF.print2stdout(f'{self.CSVPath} created!')
            except:
                WF.print2stderr(f'Unable to make {self.CSVPath}')
                sys.exit(1)

    def makeTimeSeriesPlot(self, object_settings):
        '''
        takes in object settings to build time series plot and write to XML
        :param object_settings: currently selected object settings dictionary
        :return: creates png in images dir and writes to XML file
        '''

        WF.print2stdout('\n################################')
        WF.print2stdout('Now making TimeSeries Plot.')
        WF.print2stdout('################################\n')

        objectstarttime = time.time()

        self.Plots = WPlot.Plots(self)

        default_settings = self.loadDefaultPlotObject('timeseriesplot') #get default TS plot items
        object_settings = WF.replaceDefaults(self, default_settings, object_settings) #overwrite the defaults with chapter file

        if 'template' in object_settings.keys():
            template_settings = WR.readTemplate(self, object_settings['template'])
            if object_settings['type'].lower() in template_settings.keys():
                object_settings = WF.replaceDefaults(self, template_settings[object_settings['type'].lower()], object_settings)

        object_settings['split_by_year'], object_settings['years'], object_settings['yearstr'] = WF.getObjectYears(self, object_settings)

        object_settings = self.Plots.confirmAxis(object_settings)

        object_settings['years'], object_settings['yearstr'] = WF.organizePlotYears(object_settings)

        for yi, year in enumerate(object_settings['years']):
            cur_obj_settings = pickle.loads(pickle.dumps(object_settings, -1))
            yearstr = object_settings['yearstr'][yi]

            cur_obj_settings = WF.updateFlaggedValues(cur_obj_settings, '%%year%%', yearstr)

            if len(cur_obj_settings['axs']) == 1:
                figsize=(12, 6)
                pageformat = 'half'
            else:
                figsize=(12,14)
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

            left_sided_axes = []
            right_sided_axes = []
            useAx = []
            for axi, ax_settings in enumerate(cur_obj_settings['axs']):

                ax_settings = WF.copyKeysBetweenDicts(ax_settings, cur_obj_settings, ignore=['axs'])

                if len(cur_obj_settings['axs']) == 1:
                    ax = axes
                else:
                    ax = axes[axi]

                ax_settings = self.Plots.setTimeSeriesXlims(ax_settings, yearstr, object_settings['years'])
                self.Plots.setInitialXlims(ax, year)

                ### Make Twin axis ###
                _usetwinx = False
                if 'twinx' in ax_settings.keys():
                    if ax_settings['twinx'].lower() == 'true':
                        _usetwinx = True

                _usetwiny = False
                if 'twiny' in object_settings.keys():
                    if object_settings['twiny'].lower() == 'true':
                        _usetwiny = True

                if _usetwinx:
                    ax2 = ax.twinx()

                unitslist = []
                unitslist2 = []
                stackplots = {}
                linedata, line_settings = self.Data.getTimeSeriesDataDictionary(ax_settings)
                linedata = self.Data.filterByTargetElev(linedata, line_settings)
                linedata = self.Data.scaleValuesByTable(linedata, line_settings)
                linedata = WF.mergeLines(linedata, line_settings, ax_settings)
                ax_settings = self.configureSettingsForID('base', ax_settings)
                gatedata, gate_settings = self.Data.getGateDataDictionary(ax_settings, makecopy=False)
                linedata = WF.filterDataByYear(linedata, year)
                line_settings = WF.correctDuplicateLabels(line_settings)
                straightlines = self.Data.getStraightLineValue(ax_settings)

                for gateop in gatedata.keys():
                    gatedata[gateop]['gates'] = WF.filterDataByYear(gatedata[gateop]['gates'], year)

                if 'relative' in ax_settings.keys():
                    if ax_settings['relative'].lower() == 'true':
                        RelativeMasterSet, RelativeLineSettings = self.Plots.getRelativeMasterSet(linedata, line_settings)
                        if 'unitsystem' in ax_settings.keys():
                            RelativeMasterSet, RelativeLineSettings['units'] = WF.convertUnitSystem(RelativeMasterSet,
                                                                                                    RelativeLineSettings['units'],
                                                                                                    ax_settings['unitsystem'])
                # LINE DATA #
                for line in linedata:
                    curline = linedata[line]
                    curline_settings = line_settings[line]
                    parameter, ax_settings['param_count'] = WF.getParameterCount(curline_settings, ax_settings)
                    i = ax_settings['param_count'][parameter]

                    values = curline['values']
                    dates = curline['dates']
                    units = curline_settings['units']

                    values = WF.ValueSum(dates, values) #check for dict values and add them all together

                    isstack = False
                    if 'linetype' in curline_settings.keys():
                        if curline_settings['linetype'].lower() == 'stacked': #stacked plots need to be added at the end..
                            if _usetwinx:
                                if 'yaxis' in curline_settings.keys():
                                    axis = curline_settings['yaxis'].lower()
                            else:
                                axis = 'left' #if not twinx, then only can use left

                            isstack = True

                    if units == None:
                        if parameter != None:
                            try:
                                units = self.Constants.units[parameter]
                            except KeyError:
                                units = None

                    if isinstance(units, dict):
                        if 'unitsystem' in ax_settings.keys():
                            units = units[ax_settings['unitsystem'].lower()]
                        else:
                            units = None

                    if 'unitsystem' in ax_settings.keys():
                        values, units = WF.convertUnitSystem(values, units, ax_settings['unitsystem'])

                    chkvals = WF.checkData(values)
                    if not chkvals:
                        WF.print2stdout('Invalid Data settings for line:', line, debug=self.debug)
                        continue

                    dates = WT.JDateToDatetime(dates, self.startYear)

                    if 'elevation_storage_area_file' in curline_settings.keys():
                        values = WF.calculateStorageFromElevation(values, curline_settings)

                    if 'scalar' in curline_settings.keys():
                        try:
                            scalar = float(curline_settings['scalar'])
                            values = scalar * values
                        except ValueError:
                            WF.print2stdout('Invalid Scalar. {0}'.format(curline_settings['scalar']), debug=self.debug)
                            continue

                    line_draw_settings = WD.getDefaultLineSettings(self.defaultLineStyles, curline_settings, parameter, i, debug=self.debug)
                    line_draw_settings = WF.fixDuplicateColors(line_draw_settings) #used the line, used param, then double up so subtract 1

                    if 'zorder' not in line_draw_settings.keys():
                        line_draw_settings['zorder'] = 4

                    if 'label' not in line_draw_settings.keys():
                        line_draw_settings['label'] = ''

                    curax = ax
                    axis2 = False
                    if _usetwinx:
                        if 'yaxis' in line_draw_settings.keys():
                            if line_draw_settings['yaxis'].lower() == 'right':
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
                                dates, values = WT.changeTimeSeriesInterval(dates, values, RelativeLineSettings,
                                                                              self.ModelAlt.t_offset,
                                                                              self.startYear)
                            values = values/RelativeMasterSet

                    if isstack:
                        if axis not in stackplots.keys(): #left or right
                            stackplots[axis] = []
                        stackplots[axis].append({'values': values,
                                                 'dates': dates,
                                                 'label': line_draw_settings['label'],
                                                 'color': line_draw_settings['linecolor']})

                    else:

                        if line_draw_settings['drawline'].lower() == 'true' and line_draw_settings['drawpoints'].lower() == 'true':
                            self.Plots.plotLinesAndPoints(dates, values, curax, line_draw_settings)

                        elif line_draw_settings['drawline'].lower() == 'true':
                            self.Plots.plotLines(dates, values, curax, line_draw_settings)

                        elif line_draw_settings['drawpoints'].lower() == 'true':
                            self.Plots.plotPoints(dates, values, curax, line_draw_settings)


                        self.WAT_log.addLogEntry({'type': line_draw_settings['label'] + '_TimeSeries' if line_draw_settings['label'] != '' else 'Timeseries',
                                                  'name': self.ChapterRegion+'_'+yearstr,
                                                  'description': ax_settings['description'],
                                                  'units': units,
                                                  'value_start_date': WT.translateDateFormat(dates[0], 'datetime', '',
                                                                                             self.StartTime, self.EndTime,
                                                                                             self.ModelAlt.t_offset, debug=self.debug).strftime('%d %b %Y'),
                                                  'value_end_date': WT.translateDateFormat(dates[-1], 'datetime', '',
                                                                                           self.StartTime, self.EndTime,
                                                                                           self.ModelAlt.t_offset, debug=self.debug).strftime('%d %b %Y'),
                                                  'logoutputfilename': line_draw_settings['logoutputfilename']
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

                    gateline_pos = {}
                    line_pos = gatespacing
                    for gateop in gateop_rev[::-1]:
                        gate_line_settings = gatedata[gateop]
                        for gate in list(gate_line_settings['gates'].keys())[::-1]:
                            gateline_pos[f"{gateop}_{gate}"] = line_pos
                            line_pos += 1
                        line_pos += gatespacing

                    for ggi, gateop in enumerate(gateop_rev):
                        # gate_placement += ggi*gatespacing
                        gate_count = 0 #keep track of gate number in group
                        if 'label' in gate_settings[gateop]:
                            gategroup_labels.append(gate_settings[gateop]['label'].replace('\\n', '\n'))
                        elif 'flag' in gate_settings[gateop]:
                            gategroup_labels.append(gate_settings[gateop]['flag'].replace('\\n', '\n'))
                        else:
                            gategroup_labels.append(gateop)

                        gatelines_positions = []
                        for gate in gatedata[gateop]['gates'].keys():

                            curgate = gatedata[gateop]['gates'][gate]
                            curgate_settings = gate_settings[gateop]['gates'][gate]
                            values = curgate['values']
                            dates = curgate['dates']

                            if len(dates) > 0:

                                dates = WT.JDateToDatetime(dates, self.startYear)

                                gate_line_settings = WD.getDefaultGateLineSettings(curgate_settings, gate_count, debug=self.debug)

                                if 'zorder' not in gate_line_settings.keys():
                                    gate_line_settings['zorder'] = 4

                                if 'label' not in gate_line_settings.keys():
                                    gate_line_settings['label'] = '{0}_{1}'.format(gateop, gate_count)

                                line_placement = gateline_pos[f"{gateop}_{gate}"]
                                gatelines_positions.append(line_placement)
                                gatevalues = line_placement * values

                                curax = ax
                                if _usetwinx:
                                    if 'xaxis' in line_settings.keys():
                                        if 'xaxis'.lower() == 'right':
                                            curax = ax2

                                if gate_line_settings['drawline'].lower() == 'true' and gate_line_settings['drawpoints'].lower() == 'true':
                                    self.Plots.plotLinesAndPoints(dates, gatevalues, curax, gate_line_settings)

                                elif gate_line_settings['drawline'].lower() == 'true':
                                    self.Plots.plotLines(dates, gatevalues, curax, gate_line_settings)

                                elif gate_line_settings['drawpoints'].lower() == 'true':
                                    self.Plots.plotPoints(dates, gatevalues, curax, gate_line_settings)

                                gate_count += 1 #keep track of gate number in group
                                gate_placement += 1 #keep track of gate palcement in space
                                self.WAT_log.addLogEntry({'type': gate_line_settings['label'] + '_GateTimeSeries' if gate_line_settings['label'] != '' else 'GateTimeseries',
                                                          'name': self.ChapterRegion+'_'+yearstr,
                                                          'description': ax_settings['description'],
                                                          'units': 'BINARY',
                                                          'value_start_date': WT.translateDateFormat(dates[0], 'datetime', '',
                                                                                                     self.StartTime, self.EndTime,
                                                                                                     self.ModelAlt.t_offset, debug=self.debug).strftime('%d %b %Y'),
                                                          'value_end_date': WT.translateDateFormat(dates[-1], 'datetime', '',
                                                                                                   self.StartTime, self.EndTime,
                                                                                                   self.ModelAlt.t_offset, debug=self.debug).strftime('%d %b %Y'),
                                                          'logoutputfilename': curgate_settings['logoutputfilename']
                                                          },
                                                         isdata=True)

                        gatelabels_positions.append(np.average(gatelines_positions))
                        gate_placement += gatespacing

                if 'operationlines' in ax_settings.keys():
                    operationtimes = WF.getGateOperationTimes(gatedata)
                    axs_to_add_line = [ax]
                    if 'allaxis' in ax_settings['operationlines'].keys():
                        if ax_settings['operationlines']['allaxis'].lower() == 'true':
                            axs_to_add_line = axes

                    opline_settings = WD.getDefaultStraightLineSettings(ax_settings['operationlines'], self.debug)

                    for ax_to_add_line in axs_to_add_line:
                        for operationTime in operationtimes:
                            operationTime = WT.translateDateFormat(operationTime, 'datetime', '',
                                                                       self.StartTime, self.EndTime,
                                                                       self.ModelAlt.t_offset, debug=self.debug)

                            if 'zorder' not in opline_settings.keys():
                                opline_settings['zorder'] = 3

                            ax_to_add_line.axvline(operationTime, c=opline_settings['linecolor'],
                                                   lw=opline_settings['linewidth'], ls=opline_settings['linestylepattern'],
                                                   zorder=float(opline_settings['zorder']),
                                                       alpha=float(opline_settings['alpha']))

                for stackplot_ax in stackplots.keys():
                    if stackplot_ax == 'left':
                        curax = ax
                    elif stackplot_ax == 'right':
                        curax = ax2

                    sps = stackplots[stackplot_ax]
                    stackvalues = [n['values'] for n in sps]
                    stackdates = [n['dates'] for n in sps]
                    stacklabels = [n['label'] for n in sps]
                    stackcolors = [n['color'] for n in sps]

                    matched_dates = list(set(stackdates[0]).intersection(*stackdates)) #find dates that ALL dates have.
                    matched_dates.sort()

                    if len(matched_dates) == 0:
                        WF.print2stdout('Mismatching dates for stack plot.', debug=self.debug)
                        WF.print2stdout('Please check inputs are on the date interval and time stamps are on the same hours.', debug=self.debug)

                    #now filter values associated with dates not in this list
                    for di, datelist in enumerate(stackdates):
                        mask_date_idx = [ni for ni, date in enumerate(datelist) if date in matched_dates]
                        stackvalues[di] = np.asarray(stackvalues[di])[mask_date_idx]

                    curax.stackplot(matched_dates, stackvalues, labels=stacklabels, colors=stackcolors, zorder=2)


                ### VERTICAL LINES ###
                self.Plots.plotVerticalLines(straightlines, ax, cur_obj_settings, isdate=True)

                ### Horizontal LINES ###
                self.Plots.plotHorizontalLines(straightlines, ax, cur_obj_settings)

                plotunits = WF.getPlotUnits(unitslist, ax_settings)
                plotunits2 = WF.getPlotUnits(unitslist2, ax_settings)
                ax_settings = WF.updateFlaggedValues(ax_settings, '%%units%%', WF.formatUnitsStrings(plotunits))
                ax_settings = WF.updateFlaggedValues(ax_settings, '%%units2%%', WF.formatUnitsStrings(plotunits2))

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

                        handles, labels = ax.get_legend_handles_labels()

                        plot_blank = True
                        if _usetwinx:
                            if len(handles) > 0:
                                if 'useblanklegendentry' in ax_settings.keys():
                                    if ax_settings['useblanklegendentry'].lower() == 'false':
                                        plot_blank = False
                                if plot_blank:
                                    empty_handle, = ax.plot([],[],color="w", alpha=0.0)
                                    handles.append(empty_handle)
                                    labels.append('')
                            ax2_handles, ax2_labels = ax2.get_legend_handles_labels()
                            handles += ax2_handles
                            labels += ax2_labels
                            right_sided_axes.append(ax)
                            right_offset = ax.get_window_extent().x0 / ax.get_window_extent().width

                        if 'numlegendcolumns' in ax_settings:
                            numcols = int(ax_settings['numlegendcolumns'])
                        else:
                            numcols = 1

                        if ax_settings['legend_outside'].lower() == 'true': #TODO: calibrate the offset
                            if _usetwinx:
                                ax.legend(handles=handles, labels=labels, loc='center left', bbox_to_anchor=(1+right_offset/2, 0.5), ncol=numcols,fontsize=legsize)

                            else:
                                # right_sided_axes.append(ax)
                                ax.legend(handles=handles, labels=labels, loc='center left', bbox_to_anchor=(1, 0.5), ncol=numcols,fontsize=legsize)
                        else:
                            ax.legend(handles=handles, labels=labels, fontsize=legsize, ncol=numcols)

                ############# xticks and lims #############

                useplot = self.Plots.formatDateXAxis(ax, ax_settings)
                if not useplot:
                    useAx.append(False)
                else:
                    useAx.append(True)

                xmin, xmax = ax.get_xlim()

                xmin = mpl.dates.num2date(xmin)
                xmax = mpl.dates.num2date(xmax)

                if 'xticks' in ax_settings.keys():
                    xtick_settings = ax_settings['xticks']
                else:
                    xtick_settings = {}
                self.Plots.formatTimeSeriesXticks(ax, xtick_settings, ax_settings)

                ax.set_xlim(left=xmin)
                ax.set_xlim(right=xmax)

                if _usetwiny:
                    ax2y = ax.twiny()
                    ax2y.set_xlim(left=xmin)
                    ax2y.set_xlim(right=xmax)
                    useplot = self.Plots.formatDateXAxis(ax2y, object_settings, twin=True)
                    xmin, xmax = ax2y.get_xlim()

                    xmin = mpl.dates.num2date(xmin)
                    xmax = mpl.dates.num2date(xmax)

                    if 'xticks2' in ax_settings.keys():
                        xtick_settings = ax_settings['xticks2']
                    else:
                        xtick_settings = {}
                    if 'copybottom' in xtick_settings.keys():
                        if xtick_settings['copybottom'].lower() == 'true':
                            ax2y.set_xticks(ax.get_xticks())

                    self.Plots.formatTimeSeriesXticks(ax2y, xtick_settings, ax_settings, dateformatflag='dateformat2')

                    ax2y.set_xlim(left=xmin)
                    ax2y.set_xlim(right=xmax)

                    ax2y.grid(False)

                ############# yticks and lims #############
                self.Plots.formatYTicks(ax, ax_settings, gatedata, gate_placement)

                if _usetwinx:
                    self.Plots.fixEmptyYAxis(ax, ax2)

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

                    self.Plots.formatYTicks(ax2, ax_settings, gatedata, gate_placement, axis='right')

                    if len(ax.get_lines()) > 0:
                        ax2.grid(False)
                    else:
                        ax2.grid(True)
                    ax.set_zorder(ax2.get_zorder()+1) #axis called second will always be on top unless this
                    ax.patch.set_visible(False)

            if not any(useAx):
                print(f'Plot for {year} not included due to xlimits.')
                plt.close("all")
                continue

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

            plt.tight_layout()

            if 'spacebetweenaxis' in object_settings.keys():
                if object_settings['spacebetweenaxis'].lower() != 'true':
                    plt.subplots_adjust(wspace=0, hspace=0)
            else:
                plt.subplots_adjust(wspace=0, hspace=0)

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
            # plt.tight_layout()
            # plt.savefig(figname)
            if self.highres:
                plt.savefig(figname, dpi=300)
            else:
                plt.savefig(figname)
            # plt.savefig(figname, bbox_inches='tight')
            plt.close('all')

            if pageformat == 'half':
                self.XML.writeHalfPagePlot(os.path.basename(figname), cur_obj_settings['description'])
            if pageformat == 'full':
                self.XML.writeFullPagePlot(os.path.basename(figname), cur_obj_settings['description'])

        WF.print2stdout(f'Timeseries Plot took {time.time() - objectstarttime} seconds.')

    def makeProfileStatisticsTable(self, object_settings):
        '''
        Makes a table to compute stats based off of profile lines. Data is interpolated over a series of points
        determined by the user
        :param object_settings: currently selected object settings dictionary
        :return: writes table to XML
        '''

        WF.print2stdout('\n################################')
        WF.print2stdout('Now making Profile Stats Table.')
        WF.print2stdout('################################\n')

        objectstarttime = time.time()

        self.Tables = WTable.Tables(self)
        self.Profiles = WProfile.Profiles(self)

        default_settings = self.loadDefaultPlotObject('profilestatisticstable')
        object_settings = WF.replaceDefaults(self, default_settings, object_settings)

        if 'template' in object_settings.keys():
            template_settings = WR.readTemplate(self, object_settings['template'])
            if object_settings['type'].lower() in template_settings.keys():
                object_settings = WF.replaceDefaults(self, template_settings[object_settings['type'].lower()], object_settings)

        object_settings['datakey'] = 'datapaths'

        ################# Get timestamps #################
        object_settings['datessource_flag'] = WF.getDateSourceFlag(object_settings)
        object_settings['timestamps'] = self.Profiles.getProfileTimestamps(object_settings, self.StartTime, self.EndTime)

        ################# Get units #################
        object_settings['plot_parameter'] = self.getPlotParameter(object_settings)

        ################# Get data #################
        data, line_settings = self.Data.getProfileDataDictionary(object_settings)
        line_settings = WF.correctDuplicateLabels(line_settings)
        table_blueprint = pickle.loads(pickle.dumps(object_settings, -1))

        object_settings = self.configureSettingsForID('base', object_settings)

        ################ convert yflags ################
        if object_settings['usedepth'].lower() == 'false':
            wse_data = self.Data.getProfileWSE(object_settings, onflag='datapaths')
            data, object_settings = self.Profiles.convertDepthsToElevations(data, object_settings, wse_data=wse_data)
        elif object_settings['usedepth'].lower() == 'true':
            wse_data = self.Data.getProfileWSE(object_settings, onflag='datapaths')
            data, object_settings = self.Profiles.convertElevationsToDepths(data, object_settings, wse_data=wse_data)

        ################# Get plot units #################
        data, line_settings = self.Profiles.convertProfileDataUnits(object_settings, data, line_settings)
        object_settings['units_list'] = WF.getUnitsList(line_settings)
        object_settings['plot_units'] = WF.getPlotUnits(object_settings['units_list'], object_settings)
        object_settings = WF.updateFlaggedValues(object_settings, '%%units%%', WF.formatUnitsStrings(object_settings['plot_units'], format='external'))

        table_blueprint = WF.updateFlaggedValues(table_blueprint, '%%units%%', WF.formatUnitsStrings(object_settings['plot_units'], format='external'))

        self.Data.commitProfileDataToMemory(data, line_settings, object_settings)
        data, object_settings = self.Profiles.filterProfileData(data, line_settings, object_settings)

        object_settings['resolution'] = self.Profiles.getProfileInterpResolution(object_settings)

        object_settings['split_by_year'], object_settings['years'], object_settings['yearstr'] = WF.getObjectYears(self, object_settings, allowIncludeAllYears=False)

        yrheaders, yrheaders_i = self.Tables.buildHeadersByTimestamps(object_settings['timestamps'], self.years)
        yrheaders = self.Tables.convertHeaderFormats(yrheaders, object_settings)



        if not object_settings['split_by_year']: #if we dont want to split by year, just make a big ass list
            yrheaders = [list(itertools.chain.from_iterable(yrheaders))]
            yrheaders_i = [list(itertools.chain.from_iterable(yrheaders_i))]
        for yi, yrheader_group in enumerate(yrheaders):

            yearstr = object_settings['yearstr'][yi]
            table_constructor = {}

            if len(yrheader_group) == 0:
                WF.print2stdout('No data for', yearstr, debug=self.debug)
                continue

            object_desc = WF.updateFlaggedValues(object_settings['description'], '%%year%%', yearstr)

            for yhi, yrheader in enumerate(yrheader_group):
                header_i = yrheaders_i[yi][yhi]
                headings, rows = self.Tables.buildProfileStatsTable(table_blueprint, yrheader, line_settings)
                for hi,heading in enumerate(headings):
                    tcnum = len(table_constructor.keys())
                    table_constructor[tcnum] = {}
                    if self.iscomp:
                        table_constructor[tcnum]['datecolumn'] = yrheader
                    frmt_rows = []
                    threshold_colors = np.full(len(rows), None)
                    for ri, row in enumerate(rows):
                        s_row = row.split('|')
                        rowname = s_row[0]
                        row_val = s_row[hi+1]
                        if '%%' in row_val:

                            stats_data = self.Tables.formatStatsProfileLineData(row_val, data, object_settings['resolution'],
                                                                                object_settings['usedepth'], header_i)
                            row_val, stat = self.Tables.getStatsLine(row_val, stats_data)
                            if not np.isnan(row_val) and row_val != None:
                                thresholdsettings = self.Tables.matchThresholdToStat(stat, object_settings)
                                for thresh in thresholdsettings:
                                    if thresh['colorwhen'] == 'under':
                                        if row_val < thresh['value']:
                                            threshold_colors[ri] = thresh['color']
                                    elif thresh['colorwhen'] == 'over':
                                        if row_val > thresh['value']:
                                            threshold_colors[ri] = thresh['color']

                            self.WAT_log.addLogEntry({'type': 'ProfileTableStatistic',
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
                        numberFormat = self.Tables.matchNumberFormatByStat(stat, object_settings)
                        frmt_rows.append('{0}|{1}'.format(rowname, WF.formatNumbers(row_val, numberFormat)))
                    table_constructor[tcnum]['rows'] = frmt_rows
                    table_constructor[tcnum]['thresholdcolors'] = threshold_colors
                    table_constructor[tcnum]['header'] = heading

            keeptable = False
            keepall = True
            keepcolumn = {}
            for row_num in table_constructor.keys():
                constructor = table_constructor[row_num]
                rows = constructor['rows']
                header = constructor['header']
                for row in rows:
                    srow = row.split('|')
                    if header not in keepcolumn.keys():
                        keepcolumn[header] = False
                        if srow[1].lower() not in ['nan', '-', 'none']:
                            keepcolumn[header] = True

            for key in keepcolumn.keys():
                if keepcolumn[key] == True:
                    keeptable = True
                else:
                    keepall = False

            if keeptable: #quick check if we're even writing a table..
                if not keepall:
                    new_table_constructor = {}
                    for row_num in table_constructor.keys():
                        constructor = table_constructor[row_num]
                        header = constructor['header']
                        if keepcolumn[header] == True:
                            new_table_constructor[row_num] = constructor
                else:
                    new_table_constructor = table_constructor

                #THEN write table
                if self.iscomp:
                    self.XML.writeDateControlledTableStart(object_desc, 'Statistics')
                else:
                    self.XML.writeTableStart(object_desc, 'Statistics')

                self.Tables.writeTable(new_table_constructor)
                if not keepall:
                    self.Tables.writeMissingTableItemsWarning(object_desc)
            else:
                self.Tables.writeMissingTableWarning(object_desc)
                WF.print2stdout('No values found for table. Not writing table.', debug=self.debug)

        WF.print2stdout(f'Profile Stat Table took {time.time() - objectstarttime} seconds.')

    def makeProfilePlot(self, object_settings):
        '''
        takes in object settings to build profile plot and write to XML
        :param object_settings: currently selected object settings dictionary
        :return: creates png in images dir and writes to XML file
        '''

        WF.print2stdout('\n################################')
        WF.print2stdout('Now making Profile Plot.')
        WF.print2stdout('################################\n')

        objectstarttime = time.time()

        self.Plots = WPlot.Plots(self)
        self.Profiles = WProfile.Profiles(self)

        default_settings = self.loadDefaultPlotObject('profileplot')
        object_settings = WF.replaceDefaults(self, default_settings, object_settings)

        if 'template' in object_settings.keys():
            template_settings = WR.readTemplate(self, object_settings['template'])
            if object_settings['type'].lower() in template_settings.keys():
                object_settings = WF.replaceDefaults(self, template_settings[object_settings['type'].lower()], object_settings)

        object_settings['datakey'] = 'lines'

        obj_desc = WF.updateFlaggedValues(object_settings['description'], '%%year%%', self.years_str)
        self.XML.writeProfilePlotStart(obj_desc)

        ################# Get timestamps #################
        object_settings['datessource_flag'] = WF.getDateSourceFlag(object_settings)
        object_settings['timestamps'] = self.Profiles.getProfileTimestamps(object_settings, self.StartTime, self.EndTime)

        ################# Get units #################
        object_settings['plot_parameter'] = self.getPlotParameter(object_settings)

        ################# Get data #################
        data, line_settings = self.Data.getProfileDataDictionary(object_settings)
        straightlines = self.Data.getStraightLineValue(object_settings)

        line_settings = WF.correctDuplicateLabels(line_settings)

        object_settings = self.configureSettingsForID('base', object_settings)
        gatedata, gate_settings = self.Data.getGateDataDictionary(object_settings, makecopy=False)

        ################# Get plot units #################
        data, line_settings = self.Profiles.convertProfileDataUnits(object_settings, data, line_settings)
        object_settings['units_list'] = WF.getUnitsList(line_settings)
        object_settings['plot_units'] = WF.getPlotUnits(object_settings['units_list'], object_settings)
        object_settings = WF.updateFlaggedValues(object_settings, '%%units%%', WF.formatUnitsStrings(object_settings['plot_units']))

        ################ convert yflags ################
        if object_settings['usedepth'].lower() == 'false':
            wse_data = self.Data.getProfileWSE(object_settings)
            data, object_settings = self.Profiles.convertDepthsToElevations(data, object_settings, wse_data=wse_data)
        elif object_settings['usedepth'].lower() == 'true':
            wse_data = self.Data.getProfileWSE(object_settings)
            data, object_settings = self.Profiles.convertElevationsToDepths(data, object_settings, wse_data=wse_data)

        self.Data.commitProfileDataToMemory(data, line_settings, object_settings)
        linedata, object_settings = self.Profiles.filterProfileData(data, line_settings, object_settings)

        object_settings['split_by_year'], object_settings['years'], object_settings['yearstr'] = WF.getObjectYears(self, object_settings, allowIncludeAllYears=False)

        ################ Build Plots ################
        for yi, year in enumerate(object_settings['years']):
            yearstr = object_settings['yearstr'][yi]
            # if object_settings['split_by_year']:
            #     yearstr = str(year)
            # else:
            #     yearstr = self.years_str

            t_stmps = WT.filterTimestepByYear(object_settings['timestamps'], year)

            prof_indices = [np.where(object_settings['timestamps'] == n)[0][0] for n in t_stmps]
            n = int(object_settings['profilesperrow']) * int(object_settings['rowsperpage']) #Get number of plots on page
            page_indices = [prof_indices[i * n:(i + 1) * n] for i in range((len(prof_indices) + n - 1) // n)]
            cur_obj_settings = pickle.loads(pickle.dumps(object_settings, -1))
            cur_obj_settings = WF.updateFlaggedValues(cur_obj_settings, '%%year%%', yearstr) #TODO: reudce the settings

            for page_i, pgi in enumerate(page_indices):

                subplot_rows, subplot_cols = WF.getSubplotConfig(len(pgi), int(cur_obj_settings['profilesperrow']))
                n_nrow_active = np.ceil(len(pgi) / subplot_cols)

                fig, axs = plt.subplots(nrows=int(object_settings['rowsperpage']), ncols=int(object_settings['profilesperrow']), figsize=(9,10))

                for i in range(n):

                    current_row = i // int(object_settings['profilesperrow'])
                    current_col = i % int(object_settings['profilesperrow'])

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
                                WF.print2stdout('No values for {0} on {1}'.format(line, object_settings['timestamps'][j]), debug=self.debug)
                                continue
                            msk = np.where(~np.isnan(values))
                            values = values[msk]
                        except IndexError:
                            WF.print2stdout('No values for {0} on {1}'.format(line, object_settings['timestamps'][j]), debug=self.debug)
                            continue

                        try:
                            if object_settings['usedepth'].lower() == 'true':
                                levels = data[line]['depths'][j][msk]
                            else:
                                levels = data[line]['elevations'][j][msk]
                            if not WF.checkData(levels):
                                WF.print2stdout('Non Viable depths/elevations for {0} on {1}'.format(line, object_settings['timestamps'][j]), debug=self.debug)
                                continue
                        except IndexError:
                            WF.print2stdout('Non Viable depths/elevations for {0} on {1}'.format(line, object_settings['timestamps'][j]), debug=self.debug)
                            continue

                        if not WF.checkData(values):
                            continue

                        current_ls = line_settings[line] #we have all the settings we need..
                        current_ls = WD.getDefaultLineSettings(self.defaultLineStyles, current_ls, object_settings['plot_parameter'], li, debug=self.debug)
                        current_ls = WF.fixDuplicateColors(current_ls) #used the line, used param, then double up so subtract 1

                        if current_ls['drawline'].lower() == 'true' and current_ls['drawpoints'].lower() == 'true':
                            self.Plots.plotLinesAndPoints(values, levels, ax, current_ls)

                        elif current_ls['drawline'].lower() == 'true':
                            self.Plots.plotLines(values, levels, ax, current_ls)

                        elif current_ls['drawpoints'].lower() == 'true':
                            self.Plots.plotPoints(values, levels, ax, current_ls)

                    ### HLINES ###
                    self.Plots.plotHorizontalLines(straightlines, ax, cur_obj_settings, timestamp_index=j)

                    ### VERTICAL LINES ###
                    self.Plots.plotVerticalLines(straightlines, ax, cur_obj_settings, timestamp_index=j, isdate=False)

                    show_xlabel, show_ylabel = self.getPlotLabelMasks(i, len(pgi), subplot_cols)

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

                    show_xticks = True
                    if 'xlims' in object_settings.keys() and not show_xlabel:
                        if all([x in object_settings['xlims'].keys() for x in ['min', 'max']]):
                            show_xticks = False

                    self.Plots.formatXTicks(ax, object_settings)

                    if not show_xticks:
                        ax.set_xticklabels([])
                        ax.tick_params(axis='x', which='both', bottom=False)

                    show_yticks = True
                    if 'ylims' in object_settings.keys() and not show_ylabel:
                        if all([x in object_settings['ylims'].keys() for x in ['min', 'max']]):
                            show_yticks = False

                    self.Plots.formatYTicks(ax, object_settings)

                    if not show_yticks:
                        ax.set_yticklabels([])
                        ax.tick_params(axis='y', which='both', left=False)

                    if cur_obj_settings['gridlines'].lower() == 'true':
                        ax.grid(zorder=-9)

                    ### GATES ###
                    # gategroups = {}
                    gateconfig = {}
                    if len(gatedata.keys()) > 0:
                        gatemsk = None
                        for ggi, gategroup in enumerate(gatedata.keys()):
                            gatetop = None
                            gatebottom = None
                            gatemiddle = None
                            gateop_has_value = False
                            gate_count = 0 #keep track of gate number in group
                            numgates = len(gatedata[gategroup]['gates'])
                            gatepoint_xpositions = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1],numgates+2)[1:-1] #+2 for start and end

                            cur_gateop = gatedata[gategroup]
                            cur_gateop_settings = gate_settings[gategroup]

                            if 'top' in cur_gateop_settings.keys():
                                gatetop = float(cur_gateop_settings['top'])

                            if 'bottom' in cur_gateop_settings.keys():
                                gatebottom = float(cur_gateop_settings['bottom'])

                            if 'middle' in cur_gateop_settings.keys():
                                gatemiddle = float(cur_gateop_settings['middle'])
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
                                curgate_settings = cur_gateop_settings['gates'][gate]

                                values = curgate['values']
                                dates = curgate['dates']

                                if 'dateformat' in cur_obj_settings.keys():
                                    if cur_obj_settings['dateformat'].lower() == 'datetime':
                                        if isinstance(dates[0], (int,float)):
                                            dates = WT.JDateToDatetime(dates, self.startYear)

                                if gatemsk == None:
                                    gatemsk = WR.getClosestTime([object_settings['timestamps'][j]], dates)
                                if len(gatemsk) == 0:
                                    value = np.nan
                                else:
                                    value = values[gatemsk[0]]

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
                                    elif 'showgates' in cur_obj_settings.keys():
                                        if cur_obj_settings['showgates'].lower() == 'true':
                                            showgate = True

                                    if showgate:
                                        ax.scatter(xpos, gatemiddle, edgecolor='black', facecolor='black', marker='o')

                                gate_count += 1 #keep track of gate number in group
                                self.WAT_log.addLogEntry({'type': gate + '_GateTimeSeries' if gate != '' else 'GateTimeseries',
                                                          'name': self.ChapterRegion+'_'+yearstr,
                                                          'description': cur_obj_settings['description'],
                                                          'units': 'BINARY',
                                                          'value_start_date': WT.translateDateFormat(dates[0], 'datetime', '',
                                                                                                     self.StartTime, self.EndTime,
                                                                                                     self.ModelAlt.t_offset, debug=self.debug).strftime('%d %b %Y'),
                                                          'value_end_date': WT.translateDateFormat(dates[-1], 'datetime', '',
                                                                                                   self.StartTime, self.EndTime,
                                                                                                   self.ModelAlt.t_offset, debug=self.debug).strftime('%d %b %Y'),
                                                          'logoutputfilename': curgate_settings['logoutputfilename']
                                                          },
                                                         isdata=True)

                            if 'color' in cur_gateop_settings.keys():
                                color = cur_gateop_settings['color']
                                default_color = self.Constants.def_colors[ggi]
                                color = WF.confirmColor(color, default_color, debug=self.debug)
                            if 'top' in cur_gateop_settings.keys():
                                ax.axhline(gatetop, color=color, zorder=-7)
                            if 'bottom' in cur_gateop_settings.keys():
                                ax.axhline(gatebottom, color=color, zorder=-7)
                            if 'middle' in cur_gateop_settings.keys():
                                ax.axhline(gatemiddle, color=color, zorder=-7)

                            if gateop_has_value:
                                ax.axhspan(gatebottom,gatetop, alpha=0.5, color=color, zorder=-8)

                    cur_timestamp = object_settings['timestamps'][j]
                    if 'dateformat' in object_settings:
                        if object_settings['dateformat'].lower() == 'datetime':
                            cur_timestamp = WT.translateDateFormat(cur_timestamp, 'datetime', '',
                                                                   self.StartTime, self.EndTime,
                                                                   self.ModelAlt.t_offset, debug=self.debug)
                            ttl_str = cur_timestamp.strftime('%d %b %Y')
                        elif object_settings['dateformat'].lower() == 'jdate':
                            cur_timestamp = WT.translateDateFormat(cur_timestamp, 'jdate', '',
                                                                   self.StartTime, self.EndTime,
                                                                   self.ModelAlt.t_offset, debug=self.debug)
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
                                gateconfignum = WGates.getGateConfigurationDays(gateconfig, gatedata, object_settings['timestamps'][j])
                                # bottomtext_str.append(str(gateconfignum))
                                bottomtext_str.append(str('{num:,.{digits}f}'.format(num=gateconfignum, digits=3)))
                            elif text.lower() == 'gateblend':
                                gateblendnum = WGates.getGateBlendDays(gateconfig, gatedata, object_settings['timestamps'][j])
                                # bottomtext_str.append(str(gateblendnum))
                                bottomtext_str.append(str('{num:,.{digits}f}'.format(num=gateblendnum, digits=3)))
                            else:
                                bottomtext_str.append(text)
                        bottomtext = ', '.join(bottomtext_str)

                        if show_xticks:
                            bottomtext_y = -25
                        else:
                            bottomtext_y = -10
                        if 'bottomtextfontsize' in cur_obj_settings.keys():
                            bottomtextfontsize = float(cur_obj_settings['bottomtextfontsize'])
                        else:
                            bottomtextfontsize = 6
                        if 'bottomtextfontcolor' in cur_obj_settings.keys():
                            bottomtextfontcolor = WF.confirmColor(cur_obj_settings['bottomtextfontcolor'], 'red', debug= self.debug)
                        else:
                            bottomtextfontcolor = 'red'
                        ax.annotate(bottomtext, xy=(0.02, bottomtext_y), fontsize=bottomtextfontsize, color=bottomtextfontcolor, xycoords='axes points')

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
                            # fig_ratio = (axs[int(n_nrow_active)-1,0].bbox.extents[1] - (fig.bbox.height * (.1025 * n_legends_row))) / fig.bbox.height
                            fig_ratio = (axs[int(n_nrow_active)-1,0].bbox.extents[1] - (fig.bbox.height * .055)) / fig.bbox.height

                            # plt.legend(bbox_to_anchor=(.5,fig_ratio), loc="lower center", fontsize=legsize,
                            plt.legend(bbox_to_anchor=(.5,fig_ratio), loc="upper center", fontsize=legsize,
                                       bbox_transform=fig.transFigure, ncol=ncolumns, handles=leg_handles,
                                       labels=leg_labels)


                basefigname = 'ProfilePlot_{0}_{1}_{2}_{3}_{4}'.format(self.ChapterName, yearstr,
                                                                       object_settings['plot_parameter'], self.plugin,
                                                                       page_i)

                exists = True
                tempnum = 1
                tfn = basefigname
                while exists:
                    if os.path.exists(os.path.join(self.images_path, tfn + '.png')):
                        tfn = basefigname + '_{0}'.format(tempnum)
                        tempnum += 1
                    else:
                        exists = False

                figname = tfn + '.png'

                if self.highres:
                    plt.savefig(os.path.join(self.images_path, figname), dpi=300)
                else:
                    plt.savefig(os.path.join(self.images_path, figname))

                # plt.savefig(os.path.join(self.images_path, figname))
                plt.close('all')

                ################################################

                description = '{0}: {1} of {2}'.format(cur_obj_settings['description'], page_i+1, len(page_indices))
                self.XML.writeProfilePlotFigure(figname, description)

                self.WAT_log.addLogEntry({'type': 'ProfilePlot',
                                          'name': self.ChapterRegion,
                                          'description': description,
                                          'units': object_settings['plot_units'],
                                          'value_start_date': WT.translateDateFormat(object_settings['timestamps'][pgi[0]],
                                                                                     'datetime', '',
                                                                                     self.StartTime, self.EndTime,
                                                                                     self.ModelAlt.t_offset, debug=self.debug).strftime('%d %b %Y'),
                                          'value_end_date': WT.translateDateFormat(object_settings['timestamps'][pgi[-1]],
                                                                                   'datetime', '',
                                                                                   self.StartTime, self.EndTime,
                                                                                   self.ModelAlt.t_offset, debug=self.debug).strftime('%d %b %Y'),
                                          'logoutputfilename': ', '.join([line_settings[flag]['logoutputfilename'] for flag in line_settings])
                                          },
                                         isdata=True)

        self.XML.writeProfilePlotEnd()

        WF.print2stdout(f'Profile Plot took {time.time() - objectstarttime} seconds.')

    def makeErrorStatisticsTable(self, object_settings):
        '''
        takes in object settings to build error stats table and write to XML
        :param object_settings: currently selected object settings dictionary
        :return: writes to XML file
        '''

        WF.print2stdout('\n################################')
        WF.print2stdout('Now making Error Stats table.')
        WF.print2stdout('################################\n')

        objectstarttime = time.time()

        self.Tables = WTable.Tables(self)

        default_settings = self.loadDefaultPlotObject('errorstatisticstable')
        object_settings = WF.replaceDefaults(self, default_settings, object_settings)

        if 'template' in object_settings.keys():
            template_settings = WR.readTemplate(self, object_settings['template'])
            if object_settings['type'].lower() in template_settings.keys():
                object_settings = WF.replaceDefaults(self, template_settings[object_settings['type'].lower()], object_settings)

        object_settings['split_by_year'], object_settings['years'], object_settings['yearstr'] = WF.getObjectYears(self, object_settings)

        data, data_settings = self.Data.getTableDataDictionary(object_settings)
        data = WF.mergeLines(data, data_settings, object_settings)

        object_settings = self.Tables.replaceComparisonSettings(object_settings, self.iscomp)

        headings, rows = self.Tables.buildErrorStatsTable(object_settings, data_settings)

        object_settings = self.configureSettingsForID('base', object_settings)

        object_settings['units_list'] = WF.getUnitsList(data_settings)
        object_settings['plot_units'] = WF.getPlotUnits(object_settings['units_list'], object_settings)

        object_settings = WF.updateFlaggedValues(object_settings, '%%units%%', WF.formatUnitsStrings(object_settings['plot_units'], format='external'))
        rows = WF.updateFlaggedValues(rows, '%%units%%', WF.formatUnitsStrings(object_settings['plot_units'], format='external'))
        headings = WF.updateFlaggedValues(headings, '%%units%%', WF.formatUnitsStrings(object_settings['plot_units'], format='external'))

        data = self.Tables.filterTableData(data, object_settings)
        data = self.Tables.correctTableUnits(data, data_settings, object_settings)

        if 'description' in object_settings.keys():
            desc = object_settings['description']
        else:
            desc = ''

        desc = WF.updateFlaggedValues(desc, '%%year%%', object_settings['yearstr'])

        table_constructor = {}

        for yi, year in enumerate(object_settings['years']): #iterate years. If comp, thats the date header.

            yearheader = object_settings['yearstr'][yi]
            for hi, header in enumerate(headings):
                tcnum = len(table_constructor.keys())
                table_constructor[tcnum] = {}
                if self.iscomp:
                    table_constructor[tcnum]['datecolumn'] = yearheader

                header_frmt = header.replace('%%year%%', yearheader)
                frmt_rows = []
                threshold_colors = np.full(len(rows), None)
                for ri, row in enumerate(rows):
                    s_row = row.split('|')
                    rowname = s_row[0]
                    row_val = s_row[hi+1]
                    stat = None
                    if '%%' in row_val:
                        rowdata, sr_month = self.Tables.getStatsLineData(row_val, data, year=year)
                        if len(rowdata) == 0:
                            row_val = None
                            stat = None
                        else:
                            row_val, stat = self.Tables.getStatsLine(row_val, rowdata)
                            if not np.isnan(row_val) and row_val != None:
                                thresholdsettings = self.Tables.matchThresholdToStat(stat, object_settings)
                                for thresh in thresholdsettings:
                                    if thresh['colorwhen'] == 'under':
                                        if row_val < thresh['value']:
                                            threshold_colors[ri] = thresh['color']
                                    elif thresh['colorwhen'] == 'over':
                                        if row_val > thresh['value']:
                                            threshold_colors[ri] = thresh['color']
                            data_start_date, data_end_date = self.Tables.getTableDates(year, object_settings)
                            self.WAT_log.addLogEntry({'type': 'Statistic',
                                                      'name': ' '.join([self.ChapterRegion, header_frmt, stat]),
                                                      'description': desc,
                                                      'value': row_val,
                                                      'function': stat,
                                                      'units': object_settings['plot_units'],
                                                      'value_start_date': WT.translateDateFormat(data_start_date, 'datetime', '',
                                                                                                 self.StartTime, self.EndTime,
                                                                                                 self.ModelAlt.t_offset, debug=self.debug),
                                                      'value_end_date': WT.translateDateFormat(data_end_date, 'datetime', '',
                                                                                               self.StartTime, self.EndTime,
                                                                                               self.ModelAlt.t_offset, debug=self.debug),
                                                      'logoutputfilename': ', '.join([data_settings[flag]['logoutputfilename'] for flag in data_settings])
                                                      },
                                                     isdata=True)

                    header_frmt = '' if header_frmt == None else header_frmt
                    numberFormat = self.Tables.matchNumberFormatByStat(stat, object_settings)
                    frmt_rows.append('{0}|{1}'.format(rowname, WF.formatNumbers(row_val, numberFormat)))
                table_constructor[tcnum]['rows'] = frmt_rows
                table_constructor[tcnum]['thresholdcolors'] = threshold_colors
                table_constructor[tcnum]['header'] = header_frmt

        keeptable = False
        keepall = True
        keepcolumn = {}
        for row_num in table_constructor.keys():
            constructor = table_constructor[row_num]
            rows = constructor['rows']
            header = constructor['header']
            for row in rows:
                srow = row.split('|')
                if header not in keepcolumn.keys():
                    keepcolumn[header] = False
                if srow[1].lower() not in ['nan', '-', 'none']:
                    keepcolumn[header] = True

        for key in keepcolumn.keys():
            if keepcolumn[key] == True:
                keeptable = True
            else:
                keepall = False

        if keeptable: #quick check if we're even writing a table..
            if not keepall:
                new_table_constructor = {}
                for row_num in table_constructor.keys():
                    constructor = table_constructor[row_num]
                    header = constructor['header']
                    if keepcolumn[header] == True:
                        new_table_constructor[row_num] = constructor
            else:
                new_table_constructor = table_constructor

            #THEN write table
            if self.iscomp:
                self.XML.writeDateControlledTableStart(desc, 'Year')
            else:
                self.XML.writeTableStart(desc, 'Year')

            self.Tables.writeTable(new_table_constructor)
            if not keepall:
                self.Tables.writeMissingTableItemsWarning(desc)
        else:
            self.Tables.writeMissingTableWarning(desc)
            WF.print2stdout('No values found for table. Not writing table.', debug=self.debug)


        WF.print2stdout(f'Error Stats table took {time.time() - objectstarttime} seconds.')

    def makeMonthlyStatisticsTable(self, object_settings):
        '''
        takes in object settings to build monthly stats table and write to XML
        :param object_settings: currently selected object settings dictionary
        :return: writes to XML file
        '''

        WF.print2stdout('\n################################')
        WF.print2stdout('Now making Monthly Stats Table.')
        WF.print2stdout('################################\n')

        objectstarttime = time.time()

        self.Tables = WTable.Tables(self)

        default_settings = self.loadDefaultPlotObject('monthlystatisticstable')
        object_settings = WF.replaceDefaults(self, default_settings, object_settings)

        if 'template' in object_settings.keys():
            template_settings = WR.readTemplate(self, object_settings['template'])
            if object_settings['type'].lower() in template_settings.keys():
                object_settings = WF.replaceDefaults(self, template_settings[object_settings['type'].lower()], object_settings)

        object_settings['split_by_year'], object_settings['years'], object_settings['yearstr'] = WF.getObjectYears(self, object_settings)

        data, data_settings = self.Data.getTableDataDictionary(object_settings)
        data = WF.mergeLines(data, data_settings, object_settings)

        object_settings = self.Tables.replaceComparisonSettings(object_settings, self.iscomp)

        headings, rows = self.Tables.buildMonthlyStatsTable(object_settings, data_settings)

        object_settings= self.configureSettingsForID('base', object_settings)
        object_settings['units_list'] = WF.getUnitsList(data_settings)
        object_settings['plot_units'] = WF.getPlotUnits(object_settings['units_list'], object_settings)

        object_settings = WF.updateFlaggedValues(object_settings, '%%units%%', WF.formatUnitsStrings(object_settings['plot_units'], format='external'))

        data = self.Tables.filterTableData(data, object_settings)
        data = self.Tables.correctTableUnits(data, data_settings, object_settings)

        thresholds = self.Tables.formatThreshold(object_settings)

        if 'description' in object_settings.keys():
            desc = object_settings['description']
        else:
            desc = ''
        desc = WF.updateFlaggedValues(desc, '%%year%%', object_settings['yearstr'])

        table_constructor = {}

        for yi, year in enumerate(object_settings['years']): #iterate years. If comp, thats the date header.

            yearheader = object_settings['yearstr'][yi]
            for hi, header in enumerate(headings):
                tcnum = len(table_constructor.keys())
                table_constructor[tcnum] = {}
                if self.iscomp:
                    table_constructor[tcnum]['datecolumn'] = yearheader

                header_frmt = header.replace('%%year%%', yearheader)

                frmt_rows = []
                threshold_colors = np.full(len(rows), None)
                for ri, row in enumerate(rows):
                    s_row = row.split('|')
                    rowname = s_row[0]
                    row_val = s_row[hi+1]
                    stat = None
                    if '%%' in row_val:
                        rowdata, sr_month = self.Tables.getStatsLineData(row_val, data, year=year)
                        if len(rowdata) == 0:
                            row_val = None
                            stat = None
                        else:
                            row_val, stat = self.Tables.getStatsLine(row_val, rowdata)
                            if not np.isnan(row_val) and row_val != None:
                                for thresh in thresholds:
                                    if thresh['colorwhen'] == 'under':
                                        if row_val < thresh['value']:
                                            threshold_colors[ri] = thresh['color']
                                    elif thresh['colorwhen'] == 'over':
                                        if row_val > thresh['value']:
                                            threshold_colors[ri] = thresh['color']
                            data_start_date, data_end_date = self.Tables.getTableDates(year, object_settings)
                            self.WAT_log.addLogEntry({'type': 'Statistic',
                                                      'name': ' '.join([self.ChapterRegion, header_frmt, stat]),
                                                      'description': desc,
                                                      'value': row_val,
                                                      'function': stat,
                                                      'units': object_settings['plot_units'],
                                                      'value_start_date': WT.translateDateFormat(data_start_date, 'datetime', '',
                                                                                                 self.StartTime, self.EndTime,
                                                                                                 self.ModelAlt.t_offset, debug=self.debug),
                                                      'value_end_date': WT.translateDateFormat(data_end_date, 'datetime', '',
                                                                                               self.StartTime, self.EndTime,
                                                                                               self.ModelAlt.t_offset, debug=self.debug),
                                                      'logoutputfilename': ', '.join([data_settings[flag]['logoutputfilename'] for flag in data_settings])
                                                      },
                                                     isdata=True)

                    header_frmt = '' if header_frmt == None else header_frmt
                    numberFormat = self.Tables.matchNumberFormatByStat(stat, object_settings)
                    frmt_rows.append('{0}|{1}'.format(rowname, WF.formatNumbers(row_val, numberFormat)))
                table_constructor[tcnum]['rows'] = frmt_rows
                table_constructor[tcnum]['thresholdcolors'] = threshold_colors
                table_constructor[tcnum]['header'] = header_frmt

        keeptable = False
        keepall = True
        keepcolumn = {}
        for row_num in table_constructor.keys():
            constructor = table_constructor[row_num]
            rows = constructor['rows']
            header = constructor['header']
            for row in rows:
                srow = row.split('|')
                if header not in keepcolumn.keys():
                    keepcolumn[header] = False
                if srow[1].lower() not in ['nan', '-', 'none']:
                    keepcolumn[header] = True

        for key in keepcolumn.keys():
            if keepcolumn[key] == True:
                keeptable = True
            else:
                keepall = False

        if keeptable: #quick check if we're even writing a table..
            if not keepall:
                new_table_constructor = {}
                for row_num in table_constructor.keys():
                    constructor = table_constructor[row_num]
                    header = constructor['header']
                    if keepcolumn[header] == True:
                        new_table_constructor[row_num] = constructor
            else:
                new_table_constructor = table_constructor

            #THEN write table
            if self.iscomp:
                self.XML.writeDateControlledTableStart(desc, 'Month')
            else:
                self.XML.writeTableStart(desc, 'Month')

            self.Tables.writeTable(new_table_constructor)
            if not keepall:
                self.Tables.writeMissingTableItemsWarning(desc)
        else:
            self.Tables.writeMissingTableWarning(desc)
            WF.print2stdout('No values found for table. Not writing table.', debug=self.debug)

        WF.print2stdout(f'Monthly Stat Table took {time.time() - objectstarttime} seconds.')

    def makeSingleStatisticTable(self, object_settings):
        '''
        takes in object settings to build Single Statistic table and write to XML
        :param object_settings: currently selected object settings dictionary
        :return: creates png in images dir and writes to XML file
        '''

        WF.print2stdout('\n################################')
        WF.print2stdout('Now making Single Statistic Table.')
        WF.print2stdout('################################\n')

        objectstarttime = time.time()

        self.Tables = WTable.Tables(self)

        default_settings = self.loadDefaultPlotObject('singlestatistictable') #get default SingleStat items
        object_settings = WF.replaceDefaults(self, default_settings, object_settings) #overwrite the defaults with chapter file

        if 'template' in object_settings.keys():
            template_settings = WR.readTemplate(self, object_settings['template'])
            if object_settings['type'].lower() in template_settings.keys():
                object_settings = WF.replaceDefaults(self, template_settings[object_settings['type'].lower()], object_settings)

        object_settings['split_by_year'], object_settings['years'], object_settings['yearstr'] = WF.getObjectYears(self, object_settings)

        data, data_settings = self.Data.getTableDataDictionary(object_settings)
        data = WF.mergeLines(data, data_settings, object_settings)

        object_settings['units_list'] = WF.getUnitsList(data_settings)
        object_settings['plot_units'] = WF.getPlotUnits(object_settings['units_list'], object_settings)

        object_settings = WF.updateFlaggedValues(object_settings, '%%units%%', WF.formatUnitsStrings(object_settings['plot_units'], format='external'))

        data = self.Tables.filterTableData(data, object_settings)
        data = self.Tables.correctTableUnits(data, data_settings, object_settings)

        object_settings = self.Tables.replaceComparisonSettings(object_settings, self.iscomp)

        thresholds = self.Tables.formatThreshold(object_settings)

        object_settings_blueprint = pickle.loads(pickle.dumps(object_settings, -1))

        headings, rows = self.Tables.buildSingleStatTable(object_settings_blueprint, data_settings)
        object_settings = self.configureSettingsForID('base', object_settings)

        if 'description' in object_settings.keys():
            desc = object_settings['description']
        else:
            desc = ''

        desc = WF.updateFlaggedValues(desc, '%%year%%', object_settings['yearstr'])

        table_constructor = {}

        if self.iscomp:
            datecolumns = self.Constants.mo_str_3
        else:
            datecolumns = [''] #if not comp run we dont need date headings, months will be in headings

        for mi, month in enumerate(datecolumns):
            if len(headings) == 0:
                WF.print2stdout('No headings for table.', debug=self.debug)
                # self.XML.writeDateColumn(month)
            for i, header in enumerate(headings):
                tcnum = len(table_constructor.keys())
                table_constructor[tcnum] = {}
                if self.iscomp:  #if a comparison, write the date column.
                    table_constructor[tcnum]['datecolumn'] = month
                frmt_rows = []
                threshold_colors = np.full(len(rows), None)
                for ri, row in enumerate(rows):
                    s_row = row.split('|')
                    rowname = s_row[0]
                    if rowname == 'ALLYEARS':
                        rowname = 'All'
                    year = object_settings['years'][ri]
                    row_val = s_row[i+(len(headings)*mi)+1]
                    stat = None
                    if '%%' in row_val:
                        rowdata, sr_month = self.Tables.getStatsLineData(row_val, data, year=year)
                        if len(rowdata) == 0:
                            row_val = None
                        else:
                            row_val, stat = self.Tables.getStatsLine(row_val, rowdata)
                            if np.isnan(row_val):
                                row_val = '-'
                            else:
                                for thresh in thresholds:
                                    if thresh['colorwhen'] == 'under':
                                        if row_val < thresh['value']:
                                            threshold_colors[ri] = thresh['color']
                                    elif thresh['colorwhen'] == 'over':
                                        if row_val > thresh['value']:
                                            threshold_colors[ri] = thresh['color']

                            data_start_date, data_end_date = self.Tables.getTableDates(year, object_settings_blueprint, month=sr_month)
                            self.WAT_log.addLogEntry({'type': 'Statistic',
                                                      'name': ' '.join([self.ChapterRegion, header, stat]),
                                                      'description': object_settings_blueprint['description'],
                                                      'value': row_val,
                                                      'units': object_settings_blueprint['units_list'],
                                                      'function': stat,
                                                      'value_start_date': WT.translateDateFormat(data_start_date, 'datetime', '',
                                                                                                 self.StartTime, self.EndTime,
                                                                                                 self.ModelAlt.t_offset, debug=self.debug),
                                                      'value_end_date': WT.translateDateFormat(data_end_date, 'datetime', '',
                                                                                               self.StartTime, self.EndTime,
                                                                                               self.ModelAlt.t_offset, debug=self.debug),
                                                      'logoutputfilename': ', '.join([data_settings[flag]['logoutputfilename'] for flag in data_settings])
                                                      },
                                                     isdata=True)

                    header = '' if header == None else header
                    numberFormat = self.Tables.matchNumberFormatByStat(stat, object_settings_blueprint)
                    frmt_rows.append('{0}|{1}'.format(rowname, WF.formatNumbers(row_val, numberFormat)))
                table_constructor[tcnum]['rows'] = frmt_rows
                table_constructor[tcnum]['thresholdcolors'] = threshold_colors
                table_constructor[tcnum]['header'] = header

        #check for entire rows/columns that can be sniped
        keeptable = False
        keepall = True
        keepheader = {}
        for row_num in table_constructor.keys():
            constructor = table_constructor[row_num]
            rows = constructor['rows']
            for row in rows:
                srow = row.split('|')
                if srow[0] not in keepheader.keys():
                    keepheader[srow[0]] = False
                if srow[1].lower() not in ['nan', '-', 'none']:
                    keepheader[srow[0]] = True

        for key in keepheader.keys():
            if keepheader[key] == True:
                keeptable = True
            else:
                keepall = False

        if keeptable: #quick check if we're even writing a table..
            if not keepall:
                for row_num in table_constructor.keys():
                    constructor = table_constructor[row_num]
                    rows = constructor['rows']
                    new_rows = []
                    new_thresh = []
                    for row in rows:
                        srow = row.split('|')
                        if keepheader[srow[0]] == True:
                            new_rows.append(row)
                            new_thresh.append(constructor['thresholdcolors'][ri])
                    table_constructor[row_num]['rows'] = new_rows
                    table_constructor[row_num]['thresholdcolors'] = new_thresh

            #THEN write table
            if self.iscomp:
                self.XML.writeDateControlledTableStart(desc, 'Year')
            else:
                self.XML.writeNarrowTableStart(desc, 'Year')

            self.Tables.writeTable(table_constructor)
            if not keepall:
                self.Tables.writeMissingTableItemsWarning(desc)
        else:
            self.Tables.writeMissingTableWarning(desc)
            WF.print2stdout('No values found for table. Not writing table.', debug=self.debug)

        WF.print2stdout(f'Single Stat Table took {time.time() - objectstarttime} seconds.')

    def makeSingleStatisticProfileTable(self, object_settings):
        '''
        takes in object settings to build Single Statistic profile table and write to XML
        :param object_settings: currently selected object settings dictionary
        :return: creates png in images dir and writes to XML file
        '''

        WF.print2stdout('\n################################')
        WF.print2stdout('Now making Single Statistic Profile Table.')
        WF.print2stdout('################################\n')

        objectstarttime = time.time()

        self.Tables = WTable.Tables(self)
        self.Profiles = WProfile.Profiles(self)

        default_settings = self.loadDefaultPlotObject('singlestatisticprofiletable') #get default SingleStat items
        object_settings = WF.replaceDefaults(self, default_settings, object_settings) #overwrite the defaults with chapter file

        if 'template' in object_settings.keys():
            template_settings = WR.readTemplate(self, object_settings['template'])
            if object_settings['type'].lower() in template_settings.keys():
                object_settings = WF.replaceDefaults(self, template_settings[object_settings['type'].lower()], object_settings)

        object_settings['datakey'] = 'datapaths'

        ################# Get timestamps #################
        object_settings['datessource_flag'] = WF.getDateSourceFlag(object_settings)
        object_settings['timestamps'] = self.Profiles.getProfileTimestamps(object_settings, self.StartTime, self.EndTime)
        object_settings['timestamp_index'] = self.Profiles.getProfileTimestampYearMonthIndex(object_settings, self.years)

        object_settings['split_by_year'], object_settings['years'], object_settings['yearstr'] = WF.getObjectYears(self, object_settings)

        data, line_settings = self.Data.getProfileDataDictionary(object_settings)
        line_settings = WF.correctDuplicateLabels(line_settings)

        object_settings = self.Tables.replaceComparisonSettings(object_settings, self.iscomp)

        ################ convert yflags ################
        if object_settings['usedepth'].lower() == 'false':
            wse_data = self.Data.getProfileWSE(object_settings, onflag='datapaths')
            data, object_settings = self.Profiles.convertDepthsToElevations(data, object_settings, wse_data=wse_data)
        elif object_settings['usedepth'].lower() == 'true':
            wse_data = self.Data.getProfileWSE(object_settings, onflag='datapaths')
            data, object_settings = self.Profiles.convertElevationsToDepths(data, object_settings, wse_data=wse_data)

        # object_settings= self.configureSettingsForID('base', object_settings) #will turn on for comparison plot later
        ################# Get plot units #################
        data, line_settings = self.Profiles.convertProfileDataUnits(object_settings, data, line_settings)
        object_settings['units_list'] = WF.getUnitsList(line_settings)
        object_settings['plot_units'] = WF.getPlotUnits(object_settings['units_list'], object_settings)

        object_settings = WF.updateFlaggedValues(object_settings, '%%units%%', WF.formatUnitsStrings(object_settings['plot_units'], format='external'))

        self.Data.commitProfileDataToMemory(data, line_settings, object_settings)

        data, object_settings = self.Profiles.filterProfileData(data, line_settings, object_settings)

        object_settings['resolution'] = self.Profiles.getProfileInterpResolution(object_settings)

        thresholds = self.Tables.formatThreshold(object_settings)

        object_settings_blueprint = pickle.loads(pickle.dumps(object_settings, -1))

        headings, rows = self.Tables.buildSingleStatTable(object_settings_blueprint, line_settings)
        object_settings = self.configureSettingsForID('base', object_settings)

        if 'description' in object_settings.keys():
            desc = object_settings['description']
        else:
            desc = ''

        desc = WF.updateFlaggedValues(desc, '%%year%%', object_settings['yearstr'])

        table_constructor = {}

        if self.iscomp:
            datecolumns = self.Constants.mo_str_3
        else:
            datecolumns = [''] #if not comp run we dont need date headings, months will be in headings

        for mi, month in enumerate(datecolumns):
            if len(headings) == 0:
                WF.print2stdout('No headings for table.', debug=self.debug)
            for i, header in enumerate(headings):
                tcnum = len(table_constructor.keys())
                table_constructor[tcnum] = {}
                if self.iscomp:
                    table_constructor[tcnum]['datecolumn'] = month
                frmt_rows = []
                threshold_colors = np.full(len(rows), None)
                for ri, row in enumerate(rows):
                    s_row = row.split('|')
                    rowname = s_row[0]
                    if rowname == 'ALLYEARS':
                        rowname = 'All'
                    year = object_settings['years'][ri]
                    row_val = s_row[i+(len(headings)*mi)+1]

                    stat = None
                    if '%%' in row_val:
                        rowval_stats = {}
                        if year == 'ALLYEARS':
                            if self.iscomp:
                                data_idx = WF.getAllMonthIdx(object_settings_blueprint['timestamp_index'], mi)
                            else:
                                data_idx = WF.getAllMonthIdx(object_settings_blueprint['timestamp_index'], i)
                        else:
                            if self.iscomp:
                                data_idx = object_settings_blueprint['timestamp_index'][ri][mi]
                            else:
                                data_idx = object_settings_blueprint['timestamp_index'][ri][i]
                        for di in data_idx:
                            stats_data = self.Tables.formatStatsProfileLineData(row_val, data, object_settings_blueprint['resolution'],
                                                                           object_settings_blueprint['usedepth'], di)
                            rowval_stats = self.Profiles.stackProfileIndicies(rowval_stats, stats_data)

                        row_val, stat = self.Tables.getStatsLine(row_val, rowval_stats)
                        if np.isnan(row_val):
                            row_val = '-'
                        else:
                            for thresh in thresholds:
                                if thresh['colorwhen'] == 'under':
                                    if row_val < thresh['value']:
                                        threshold_colors[ri] = thresh['color']
                                elif thresh['colorwhen'] == 'over':
                                    if row_val > thresh['value']:
                                        threshold_colors[ri] = thresh['color']
                        if self.iscomp:
                            data_start_date, data_end_date = self.Tables.getTableDates(year, object_settings_blueprint, month=month)
                        else:
                            data_start_date, data_end_date = self.Tables.getTableDates(year, object_settings_blueprint, month=header)
                        self.WAT_log.addLogEntry({'type': 'ProfileTableStatistic',
                                                  'name': ' '.join([self.ChapterRegion, header, stat]),
                                                  'description': object_settings_blueprint['description'],
                                                  'value': row_val,
                                                  'function': stat,
                                                  'units': object_settings_blueprint['plot_units'],
                                                  'value_start_date': data_start_date,
                                                  'value_end_date': data_end_date,
                                                  'logoutputfilename': ', '.join([line_settings[flag]['logoutputfilename'] for flag in line_settings])
                                                  },
                                                 isdata=True)

                    header = '' if header == None else header
                    numberFormat = self.Tables.matchNumberFormatByStat(stat, object_settings_blueprint)
                    frmt_rows.append('{0}|{1}'.format(rowname, WF.formatNumbers(row_val, numberFormat)))
                table_constructor[tcnum]['rows'] = frmt_rows
                table_constructor[tcnum]['thresholdcolors'] = threshold_colors
                table_constructor[tcnum]['header'] = header

        keeptable = False
        keepall = True
        keepheader = {}
        for row_num in table_constructor.keys():
            constructor = table_constructor[row_num]
            rows = constructor['rows']
            for row in rows:
                srow = row.split('|')
                if srow[0] not in keepheader.keys():
                    keepheader[srow[0]] = False
                if srow[1].lower() not in ['nan', '-', 'none']:
                    keepheader[srow[0]] = True

        for key in keepheader.keys():
            if keepheader[key] == True:
                keeptable = True
            else:
                keepall = False

        if keeptable: #quick check if we're even writing a table..
            if not keepall:
                for row_num in table_constructor.keys():
                    constructor = table_constructor[row_num]
                    rows = constructor['rows']
                    new_rows = []
                    new_thresh = []
                    for r, row in enumerate(rows):
                        srow = row.split('|')
                        if keepheader[srow[0]] == True:
                            new_rows.append(row)
                            new_thresh.append(constructor['thresholdcolors'][ri])
                    table_constructor[row_num]['rows'] = new_rows
                    table_constructor[row_num]['thresholdcolors'] = new_thresh

            #THEN write table
            if self.iscomp:
                self.XML.writeDateControlledTableStart(desc, 'Year')
            else:
                self.XML.writeNarrowTableStart(desc, 'Year')

            self.Tables.writeTable(table_constructor)
            if not keepall:
                self.Tables.writeMissingTableItemsWarning(desc)

        else:
            self.Tables.writeMissingTableWarning(desc)
            WF.print2stdout('No values found for table. Not writing table.', debug=self.debug)

        WF.print2stdout(f'Single Profile Stat Table took {time.time() - objectstarttime} seconds.')

    def makeContourPlot(self, object_settings):
        '''
        takes in object settings to build contour plot and write to XML
        :param object_settings: currently selected object settings dictionary
        :return: creates png in images dir and writes to XML file
        '''

        WF.print2stdout('\n################################')
        WF.print2stdout('Now making Contour Plot.')
        WF.print2stdout('################################\n')

        objectstarttime = time.time()

        self.Plots = WPlot.Plots(self)

        default_settings = self.loadDefaultPlotObject('contourplot') #get default TS plot items
        object_settings = WF.replaceDefaults(self, default_settings, object_settings) #overwrite the defaults with chapter file

        if 'template' in object_settings.keys():
            template_settings = WR.readTemplate(self, object_settings['template'])
            if object_settings['type'].lower() in template_settings.keys():
                object_settings = WF.replaceDefaults(self, template_settings[object_settings['type'].lower()], object_settings)

        object_settings['split_by_year'], object_settings['years'], object_settings['yearstr'] = WF.getObjectYears(self, object_settings)

        object_settings['years'], object_settings['yearstr'] = WF.organizePlotYears(object_settings)

        for yi, year in enumerate(object_settings['years']):
            useAx = []
            cur_obj_settings = pickle.loads(pickle.dumps(object_settings, -1))

            yearstr = object_settings['yearstr'][yi]

            cur_obj_settings = self.Plots.setTimeSeriesXlims(cur_obj_settings, yearstr, object_settings['years'])

            #NOTES
            #Data structure:
            #2D array of dates[distance from source]
            # array of dates corresponding to the number of the first D of array above
            # supplementary array for distances corrsponding to the second D of array above
            #ex
            #[[1,2,3,5],[2,3,4,2],[5,3,2,5]] #values per date at a distance
            #[01jan2016, 04Feb2016, 23May2016] #dates
            #[0, 19, 25, 35] #distances

            contoursbyID, contoursbyID_settings = self.Data.getContourDataDictionary(cur_obj_settings)
            contoursbyID = WF.filterDataByYear(contoursbyID, year)
            selectedContourIDs = WF.getUsedIDs(contoursbyID_settings)

            straightlines = self.Data.getStraightLineValue(cur_obj_settings)

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
                contour_plot_settings = pickle.loads(pickle.dumps(cur_obj_settings, -1))
                contour_plot_settings = self.configureSettingsForID(ID, contour_plot_settings)
                contours = WF.selectContourByID(contoursbyID, ID)
                contour_settings = WF.selectContourByID(contoursbyID_settings, ID)
                values, dates, distance, transitions = WF.stackContours(contours, contoursbyID_settings)
                if len(selectedContourIDs) == 1:
                    axes = [axes]

                ax = axes[IDi]

                for contourname in contour_settings:
                    c_set = contour_settings[contourname]
                    parameter, contour_plot_settings['param_count'] = WF.getParameterCount(c_set, contour_plot_settings)

                if 'units' in contour_plot_settings.keys():
                    units = contour_plot_settings['units']
                else:
                    if 'parameter' in contour_plot_settings.keys():
                        parameter = contour_plot_settings['parameter']
                    else:
                        parameter = ''
                        top_count = 0
                        for key in contour_plot_settings['param_count'].keys():
                            if contour_plot_settings['param_count'][key] > top_count:
                                parameter = key
                    try:
                        units = self.Constants.units[parameter]
                    except KeyError:
                        units = None

                if isinstance(units, dict):
                    if 'unitsystem' in contour_plot_settings.keys():
                        units = units[contour_plot_settings['unitsystem'].lower()]
                    else:
                        units = None

                if 'unitsystem' in contour_plot_settings.keys():
                    values, units = WF.convertUnitSystem(values, units, contour_plot_settings['unitsystem']) #TODO: confirm

                chkvals = WF.checkData(values)
                if not chkvals:
                    WF.print2stdout('Invalid Data settings for contour plot year {0}'.format(year), debug=self.debug)
                    continue

                dates = WT.JDateToDatetime(dates, self.startYear)

                if 'label' not in contour_plot_settings.keys():
                    contour_plot_settings['label'] = ''

                if 'description' not in contour_plot_settings.keys():
                    contour_plot_settings['description'] = ''

                contour_plot_settings = WD.getDefaultContourSettings(contour_plot_settings, debug=self.debug)

                if 'min' in contour_plot_settings['colorbar']:
                    vmin = float(contour_plot_settings['colorbar']['min'])
                else:
                    vmin = np.nanmin(values)
                if 'max' in contour_plot_settings['colorbar']:
                    vmax = float(contour_plot_settings['colorbar']['max'])
                else:
                    vmax = np.nanmax(values)

                contr = ax.contourf(dates, distance, values.T, cmap=contour_plot_settings['colorbar']['colormap'],
                                    vmin=vmin, vmax=vmax,
                                    levels=np.linspace(vmin, vmax, int(contour_plot_settings['colorbar']['bins'])), #add one to get the desired number..
                                    extend='both') #the .T transposes the array so dates on bottom TODO:make extend variable
                # ax.invert_yaxis()

                self.WAT_log.addLogEntry({'type': contour_plot_settings['label'] + '_ContourPlot' if contour_plot_settings['label'] != '' else 'ContourPlot',
                                          'name': self.ChapterRegion+'_'+yearstr,
                                          'description': contour_plot_settings['description'],
                                          'units': units,
                                          'value_start_date': WT.translateDateFormat(dates[0], 'datetime', '',
                                                                                     self.StartTime, self.EndTime,
                                                                                     self.ModelAlt.t_offset, debug=self.debug).strftime('%d %b %Y'),
                                          'value_end_date': WT.translateDateFormat(dates[-1], 'datetime', '',
                                                                                   self.StartTime, self.EndTime,
                                                                                   self.ModelAlt.t_offset, debug=self.debug).strftime('%d %b %Y'),
                                          'logoutputfilename': 'NA'
                                          },
                                         isdata=True)

                contour_plot_settings = WF.updateFlaggedValues(contour_plot_settings, '%%units%%', WF.formatUnitsStrings(units))

                if 'contourlines' in contour_plot_settings.keys():
                    for contourline in contour_plot_settings['contourlines']:
                        if 'value' in contourline.keys():
                            val = float(contourline['value'])
                        else:
                            WF.print2stdout('No Value set for contour line.', debug=self.debug)
                            continue
                        contourline = WD.getDefaultContourLineSettings(contourline)
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
                self.Plots.plotVerticalLines(straightlines, ax, contour_plot_settings, isdate=True)

                ### Horizontal LINES ###
                self.Plots.plotHorizontalLines(straightlines, ax, contour_plot_settings)

                if self.iscomp:
                    if 'modeltext' in contour_plot_settings.keys():
                        modeltext = contour_plot_settings['modeltext']
                    else:
                        modeltext = self.SimulationName
                    plt.text(1.02, 0.5, modeltext, fontsize=12, transform=ax.transAxes, verticalalignment='center',
                             horizontalalignment='center', rotation='vertical')

                if 'gridlines' in contour_plot_settings.keys():
                    if contour_plot_settings['gridlines'].lower() == 'true':
                        ax.grid(True)

                if 'ylabel' in contour_plot_settings.keys():
                    if 'ylabelsize' in contour_plot_settings.keys():
                        ylabsize = float(contour_plot_settings['ylabelsize'])
                    elif 'fontsize' in contour_plot_settings.keys():
                        ylabsize = float(contour_plot_settings['fontsize'])
                    else:
                        ylabsize = 12
                    ax.set_ylabel(contour_plot_settings['ylabel'], fontsize=ylabsize)


                xmin, xmax = ax.get_xlim()
                # if contour_settings['dateformat'].lower() == 'datetime':
                xmin = mpl.dates.num2date(xmin)
                xmax = mpl.dates.num2date(xmax)
                if 'xticks' in contour_plot_settings.keys():
                    xtick_settings = contour_plot_settings['xticks']
                else:
                    xtick_settings = {}
                self.Plots.formatTimeSeriesXticks(ax, xtick_settings, contour_plot_settings)

                ax.set_xlim(left=xmin)
                ax.set_xlim(right=xmax)

                self.Plots.formatYTicks(ax, contour_plot_settings)

                if 'transitions' in contour_plot_settings.keys():
                    for transkey in transitions.keys():
                        transition_start = transitions[transkey]
                        trans_name = None
                        hline = WD.getDefaultStraightLineSettings(contour_plot_settings['transitions'], self.debug)

                        linecolor = WF.prioritizeKey(contours[transkey], hline, 'linecolor')
                        linestylepattern = WF.prioritizeKey(contours[transkey], hline, 'linestylepattern')
                        alpha = WF.prioritizeKey(contours[transkey], hline, 'alpha')
                        linewidth = WF.prioritizeKey(contours[transkey], hline, 'linewidth')

                        ax.axhline(y=transition_start, c=linecolor, ls=linestylepattern, alpha=float(alpha),
                                   lw=float(linewidth))
                        if 'name' in contour_plot_settings['transitions'].keys():
                            trans_flag = contour_plot_settings['transitions']['name'].lower() #blue:pink:white:pink:blue
                            text_settings = WD.getDefaultTextSettings(contour_plot_settings['transitions'], self.debug)

                            if trans_flag in contour_settings[transkey].keys():
                                trans_name = contour_settings[transkey][trans_flag]
                            if trans_name != None:
                                if ax.get_ylim()[0] <= transition_start <= ax.get_ylim()[1]:
                                    trans_y_value = transition_start + ((ax.get_ylim()[1] - ax.get_ylim()[0]) * .01)

                                    fontcolor = WF.prioritizeKey(contour_settings[transkey], text_settings, 'fontcolor')
                                    fontsize = WF.prioritizeKey(contour_settings[transkey], text_settings, 'fontsize')
                                    horizontalalignment = WF.prioritizeKey(contour_settings[transkey], text_settings, 'horizontalalignment')
                                    text_x_pos = WF.prioritizeKey(contour_settings[transkey], text_settings, 'text_x_pos', backup=0.001)
                                    trans_x_value = mpl.dates.num2date(ax.get_xlim()[0] + ((ax.get_xlim()[1] - ax.get_xlim()[0]) * float(text_x_pos)))

                                    ax.text(trans_x_value, trans_y_value, trans_name, c=fontcolor, size=float(fontsize),
                                            horizontalalignment=horizontalalignment,
                                            verticalalignment='top')

                ax.invert_yaxis()

            # #stuff to call once per plot
            self.configureSettingsForID('base', cur_obj_settings)
            cur_obj_settings = WF.updateFlaggedValues(cur_obj_settings, '%%units%%', WF.formatUnitsStrings(units))

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

            useplot = self.Plots.formatDateXAxis(axes[-1], cur_obj_settings)
            if not useplot:
                useAx.append(False)
            else:
                useAx.append(True)

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

            if not any(useAx):
                print(f'Plot for {year} not included due to xlimits.')
                plt.close("all")
                continue

            cbar = plt.colorbar(contr, ax=axes[-1], orientation='horizontal', aspect=50.)
            locs = np.linspace(vmin, vmax, int(contour_plot_settings['colorbar']['numticks']))
            cbar.set_ticks(locs)
            cbar.set_ticklabels(locs.round(2))
            if 'label' in contour_plot_settings['colorbar']:
                if 'labelsize' in contour_plot_settings['colorbar'].keys():
                    labsize = float(contour_plot_settings['colorbar']['labelsize'])
                elif 'fontsize' in contour_plot_settings['colorbar'].keys():
                    labsize = float(cur_obj_settings['colorbar']['fontsize'])
                else:
                    labsize = 12
                cbar.set_label(contour_plot_settings['colorbar']['label'], fontsize=labsize)

            plt.tight_layout()
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
            # plt.savefig(figname, bbox_inches='tight')
            if self.highres:
                plt.savefig(figname, dpi=300)
            else:
                plt.savefig(figname)
            plt.close('all')

            if pageformat == 'full':
                self.XML.writeFullPagePlot(os.path.basename(figname), cur_obj_settings['description'])
            elif pageformat == 'half':
                self.XML.writeHalfPagePlot(os.path.basename(figname), cur_obj_settings['description'])

        WF.print2stdout(f'Contour Plot took {time.time() - objectstarttime} seconds.')

    def makeReservoirContourPlot(self, object_settings):
        '''
        takes in object settings to build reservoir contour plot and write to XML
        :param object_settings: currently selected object settings dictionary
        :return: creates png in images dir and writes to XML file
        '''

        WF.print2stdout('\n################################')
        WF.print2stdout('Now making Reservoir Contour Plot.')
        WF.print2stdout('################################\n')

        objectstarttime = time.time()

        self.Plots = WPlot.Plots(self)

        default_settings = self.loadDefaultPlotObject('contourplot') #get default TS plot items
        object_settings = WF.replaceDefaults(self, default_settings, object_settings) #overwrite the defaults with chapter file

        if 'template' in object_settings.keys():
            template_settings = WR.readTemplate(self, object_settings['template'])
            if object_settings['type'].lower() in template_settings.keys():
                object_settings = WF.replaceDefaults(self, template_settings[object_settings['type'].lower()], object_settings)

        object_settings['datakey'] = 'datapaths'

        object_settings['split_by_year'], object_settings['years'], object_settings['yearstr'] = WF.getObjectYears(self, object_settings)

        object_settings['years'], object_settings['yearstr'] = WF.organizePlotYears(object_settings)

        for yi, year in enumerate(object_settings['years']):
            useAx = []
            cur_obj_settings = pickle.loads(pickle.dumps(object_settings, -1))
            yearstr = object_settings['yearstr'][yi]

            cur_obj_settings = self.Plots.setTimeSeriesXlims(cur_obj_settings, yearstr, object_settings['years'])

            #NOTES
            #Data structure:
            #2D array of dates[distance from source]
            # array of dates corresponding to the number of the first D of array above
            # supplementary array for distances corrsponding to the second D of array above
            #ex
            #[[1,2,3,5],[2,3,4,2],[5,3,2,5]] #values per date
            #[01jan2016, 04Feb2016, 23May2016] #dates
            #[0, 19, 25, 35] #elevations
            #[0, 19, 25, 35] #top water elevations

            contoursbyID, contoursbyID_settings = self.Data.getReservoirContourDataDictionary(cur_obj_settings)
            contoursbyID = WF.filterDataByYear(contoursbyID, year, extraflag='topwater')
            selectedContourIDs = WF.getUsedIDs(contoursbyID)
            straightlines = self.Data.getStraightLineValue(cur_obj_settings)

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
                contour_plot_settings = pickle.loads(pickle.dumps(cur_obj_settings, -1))
                contour_plot_settings = self.configureSettingsForID(ID, contour_plot_settings)
                contours = WF.selectContourByID(contoursbyID, ID)
                contours_settings = WF.selectContourByID(contoursbyID_settings, ID)
                if len(contours.keys()) > 1:
                    WF.print2stdout(f'Too many reservoir keys defined. Using the first, {list(contours.keys())[0]}', debug=self.debug)
                elif len(contours.keys()) == 0:
                    WF.print2stdout(f'No reservoir keys defined.', debug=self.debug)
                    return
                contour = contours[list(contours.keys())[0]]
                contour_settings = contours_settings[list(contours.keys())[0]]

                values = contour['values']
                elevations = contour['elevations']
                dates = contour['dates']
                topwater = contour['topwater']

                if len(selectedContourIDs) == 1:
                    axes = [axes]

                ax = axes[IDi]

                parameter, contour_plot_settings['param_count'] = WF.getParameterCount(contour_settings, contour_plot_settings)

                if 'units' in contour_plot_settings.keys():
                    units = contour_settings['units']
                else:
                    if 'parameter' in contour_plot_settings.keys():
                        parameter = contour_plot_settings['parameter']
                    else:
                        parameter = ''
                        top_count = 0
                        for key in contour_plot_settings['param_count'].keys():
                            if contour_plot_settings['param_count'][key] > top_count:
                                parameter = key
                    try:
                        units = self.Constants.units[parameter]
                    except KeyError:
                        units = None

                if isinstance(units, dict):
                    if 'unitsystem' in contour_plot_settings.keys():
                        units = units[contour_plot_settings['unitsystem'].lower()]
                    else:
                        units = None

                if 'unitsystem' in contour_plot_settings.keys():
                    values, units = WF.convertUnitSystem(values, units, contour_plot_settings['unitsystem']) #TODO: confirm

                chkvals = WF.checkData(values)
                if not chkvals:
                    WF.print2stdout('Invalid Data settings for contour plot year {0}'.format(year), debug=self.debug)
                    continue

                dates = WT.JDateToDatetime(dates, self.startYear)

                if 'label' not in contour_plot_settings.keys():
                    contour_plot_settings['label'] = ''

                if 'description' not in contour_plot_settings.keys():
                    contour_plot_settings['description'] = ''

                contour_plot_settings = WD.getDefaultContourSettings(contour_plot_settings, debug=self.debug)

                if 'min' in contour_plot_settings['colorbar']:
                    vmin = float(contour_plot_settings['colorbar']['min'])
                else:
                    vmin = np.nanmin(values)
                if 'max' in contour_plot_settings['colorbar']:
                    vmax = float(contour_plot_settings['colorbar']['max'])
                else:
                    vmax = np.nanmax(values)

                values = WF.filterContourOverTopWater(values, elevations, topwater)

                contr = ax.contourf(dates, elevations, values.T, cmap=contour_plot_settings['colorbar']['colormap'],
                                    vmin=vmin, vmax=vmax,
                                    levels=np.linspace(vmin, vmax, int(contour_plot_settings['colorbar']['bins'])), #add one to get the desired number..
                                    extend='both') #the .T transposes the array so dates on bottom TODO:make extend variable

                # ax.plot(dates, topwater, c='red') #debug topwater
                self.WAT_log.addLogEntry({'type': contour_plot_settings['label'] + '_ContourPlot' if contour_plot_settings['label'] != '' else 'ContourPlot',
                                          'name': self.ChapterRegion+'_'+yearstr,
                                          'description': contour_plot_settings['description'],
                                          'units': units,
                                          'value_start_date': WT.translateDateFormat(dates[0], 'datetime', '',
                                                                                     self.StartTime, self.EndTime,
                                                                                     self.ModelAlt.t_offset, debug=self.debug).strftime('%d %b %Y'),
                                          'value_end_date': WT.translateDateFormat(dates[-1], 'datetime', '',
                                                                                   self.StartTime, self.EndTime,
                                                                                   self.ModelAlt.t_offset, debug=self.debug).strftime('%d %b %Y'),
                                          'logoutputfilename': 'NA'
                                          },
                                         isdata=True)

                contour_plot_settings = WF.updateFlaggedValues(contour_plot_settings, '%%units%%', WF.formatUnitsStrings(units))

                if 'contourlines' in contour_plot_settings.keys():
                    for contourline in contour_plot_settings['contourlines']:
                        if 'value' in contourline.keys():
                            val = float(contourline['value'])
                        else:
                            WF.print2stdout('No Value set for contour line.', debug=self.debug)
                            continue
                        contourline = WD.getDefaultContourLineSettings(contourline)
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
                self.Plots.plotVerticalLines(straightlines, ax, contour_plot_settings, isdate=True)

                ### Horizontal LINES ###
                self.Plots.plotHorizontalLines(straightlines, ax, contour_plot_settings)

                if self.iscomp:
                    if 'modeltext' in contour_plot_settings.keys():
                        modeltext = contour_plot_settings['modeltext']
                    else:
                        modeltext = self.SimulationName
                    plt.text(1.02, 0.5, modeltext, fontsize=12, transform=ax.transAxes, verticalalignment='center',
                             horizontalalignment='center', rotation='vertical')


                if 'gridlines' in contour_plot_settings.keys():
                    if contour_plot_settings['gridlines'].lower() == 'true':
                        ax.grid(True)

                if 'ylabel' in contour_plot_settings.keys():
                    if 'ylabelsize' in contour_plot_settings.keys():
                        ylabsize = float(contour_plot_settings['ylabelsize'])
                    elif 'fontsize' in contour_plot_settings.keys():
                        ylabsize = float(contour_plot_settings['fontsize'])
                    else:
                        ylabsize = 12
                    ax.set_ylabel(contour_plot_settings['ylabel'], fontsize=ylabsize)

                ############# xticks and lims #############

                useplot = self.Plots.formatDateXAxis(ax, contour_plot_settings)
                if not useplot:
                    useAx.append(False)
                else:
                    useAx.append(True)
                xmin, xmax = ax.get_xlim()

                xmin = mpl.dates.num2date(xmin)
                xmax = mpl.dates.num2date(xmax)

                if 'xticks' in contour_plot_settings.keys():
                    xtick_settings = contour_plot_settings['xticks']
                else:
                    xtick_settings = {}
                self.Plots.formatTimeSeriesXticks(ax, xtick_settings, contour_plot_settings)

                ax.set_xlim(left=xmin)
                ax.set_xlim(right=xmax)

                ############# yticks and lims #############
                self.Plots.formatYTicks(ax, contour_plot_settings)

            # #stuff to call once per plot
            self.configureSettingsForID('base', cur_obj_settings)
            cur_obj_settings = WF.updateFlaggedValues(cur_obj_settings, '%%units%%', WF.formatUnitsStrings(units))

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

            useplot = self.Plots.formatDateXAxis(axes[-1], cur_obj_settings)
            if not useplot:
                useAx.append(False)
            else:
                useAx.append(True)

            if not any(useAx):
                print(f'Plot for {year} not included due to xlimits.')
                plt.close("all")
                continue

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
            locs = np.linspace(vmin, vmax, int(contour_plot_settings['colorbar']['numticks']))
            cbar.set_ticks(locs)
            cbar.set_ticklabels(locs.round(2))
            if 'label' in contour_plot_settings['colorbar']:
                if 'labelsize' in contour_plot_settings['colorbar'].keys():
                    labsize = float(contour_plot_settings['colorbar']['labelsize'])
                elif 'fontsize' in cur_obj_settings['colorbar'].keys():
                    labsize = float(cur_obj_settings['colorbar']['fontsize'])
                else:
                    labsize = 12
                cbar.set_label(contour_plot_settings['colorbar']['label'], fontsize=labsize)

            plt.tight_layout()
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
            # plt.savefig(figname, bbox_inches='tight')
            # plt.savefig(figname)
            if self.highres:
                plt.savefig(figname, dpi=300)
            else:
                plt.savefig(figname)
            plt.close('all')

            if pageformat == 'full':
                self.XML.writeFullPagePlot(os.path.basename(figname), cur_obj_settings['description'])
            elif pageformat == 'half':
                self.XML.writeHalfPagePlot(os.path.basename(figname), cur_obj_settings['description'])

        WF.print2stdout(f'Reservoir Contour Plot took {time.time() - objectstarttime} seconds.')

    def makeTextBox(self, object_settings):
        '''
        Makes a text box object in the report
        :param object_settings: currently selected object settings dictionary
        :return:
        '''

        objectstarttime = time.time()

        if 'text' not in object_settings.keys():
            WF.print2stdout('Failed to input textbox contents using <text> flag.', debug=self.debug)

        self.XML.writeTextBox(object_settings['text'])

        WF.print2stdout(f'Text box took {time.time() - objectstarttime} seconds.')

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
                    self.Data.Memory
                    self.SimulationName
                    self.baseSimulationName
                    self.simulationDir
                    self.DSSFile
                    self.StartTimeStr
                    self.EndTimeStr
                    self.LastComputed
                    self.ModelAlternatives
        '''

        # self.Data.Memory = {}
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
        WT.setSimulationDateTimes(self, ID)

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
            self.ChapterText = Chapter['grouptext']
            self.ChapterResolution = Chapter['resolution']
            self.debug_boolean = Chapter['debug']

            self.debug = False
            if self.debug_boolean.lower() == 'true':
                self.debug = True
                WF.print2stdout('Verbose mode activated!')
            else:
                WF.print2stdout('Quiet mode activated.')

            self.highres = True #default
            if self.ChapterResolution.lower() == 'high':
                self.highres = True
                WF.print2stdout('Running High Res Mode!')
            elif self.ChapterResolution.lower() == 'low':
                self.highres = False
                WF.print2stdout('Running Low Res Mode!')

            self.WAT_log.addLogEntry({'region': self.ChapterRegion})
            self.XML.writeChapterStart(self.ChapterName, self.ChapterText)
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
                    elif objtype == 'profilestatisticstable':
                        self.makeProfileStatisticsTable(object)
                    elif objtype == 'contourplot':
                        self.makeContourPlot(object)
                    elif objtype == 'reservoircontourplot':
                        self.makeReservoirContourPlot(object)
                    elif objtype == 'singlestatistictable':
                        self.makeSingleStatisticTable(object)
                    elif objtype == 'singlestatisticprofiletable':
                        self.makeSingleStatisticProfileTable(object)
                    elif objtype == 'textbox':
                        self.makeTextBox(object)
                    else:
                        WF.print2stdout('Section Type {0} not identified.'.format(objtype))
                        WF.print2stdout('Skipping Section..')
                self.XML.writeSectionHeaderEnd()
            WF.print2stdout('\n################################')
            WF.print2stdout('Chapter Complete.')
            WF.print2stdout('################################\n')
            self.XML.writeChapterEnd()

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

            if len(approved_modelalts) > 0:
                approved_modelalt = approved_modelalts[0]
                WF.print2stdout('Added {0} for ID {1}'.format(approved_modelalt['program'], ID))
                self.SimulationVariables[ID]['alternativeFpart'] = approved_modelalt['fpart']
                self.SimulationVariables[ID]['alternativeDirectory'] = approved_modelalt['directory']
                self.SimulationVariables[ID]['modelAltName'] = approved_modelalt['name']
                self.SimulationVariables[ID]['plugin'] = approved_modelalt['program']

                if self.SimulationVariables[ID]['plugin'].lower() == "ressim":
                    self.SimulationVariables[ID]['ModelAlt'] = WRSS.ResSim_Results(self.SimulationVariables[ID]['simulationDir'],
                                                                                  self.SimulationVariables[ID]['alternativeFpart'],
                                                                                  self.StartTime, self.EndTime, self)
                elif self.SimulationVariables[ID]['plugin'].lower() == 'cequalw2':
                    self.SimulationVariables[ID]['ModelAlt'] = WW2.W2_Results(self.SimulationVariables[ID]['simulationDir'],
                                                                              self.SimulationVariables[ID]['modelAltName'],
                                                                              self.SimulationVariables[ID]['alternativeDirectory'],
                                                                              self.StartTime, self.EndTime, self)
                else:
                    self.SimulationVariables[ID]['ModelAlt'] == 'unknown'
                self.accepted_IDs.append(ID)

        if len(self.accepted_IDs) == 0:
            if self.iscomp:
                csv_file_name = '{0}_comparison.csv'.format(self.baseSimulationName.replace(' ', '_'))
            else:
                csv_file_name = '{0}.csv'.format(self.baseSimulationName.replace(' ', '_'))
            WF.print2stderr('Incompatible input information from the WAT XML output file ({0})\nand Simulation CSV file ({1})'.format(self.simulationInfoFile, csv_file_name))
            WF.print2stderr('Please Confirm inputs and run again.')
            if self.iscomp:
                WF.print2stderr('If comparison plot, ensure that all model alts are in {0}'.format(csv_file_name))
                WF.print2stderr('Example line: ResSim, TmpNFlo, CeQualW2, Shasta from DSS 15, Shasta_ResSim_TCD_comparison.XML')
            WF.print2stderr('Now Exiting...')
            sys.exit(1)

    def loadCurrentID(self, ID):
        '''
        loads model specific settings for a given ID
        :param ID: selected ID, such as 'base' or 'alt_1'
        '''

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
        '''
        loads model alternative specific settings for a given ID
        :param ID: selected ID, such as 'base' or 'alt_1'
        '''

        self.alternativeFpart = self.SimulationVariables[ID]['alternativeFpart']
        self.alternativeDirectory = self.SimulationVariables[ID]['alternativeDirectory']
        self.modelAltName = self.SimulationVariables[ID]['modelAltName']
        self.plugin = self.SimulationVariables[ID]['plugin']
        self.ModelAlt = self.SimulationVariables[ID]['ModelAlt']
        # WF.print2stdout('Model {0} Loaded'.format(ID), debug=self.debug) #noisy

    def initializeXML(self):
        '''
        creates a new version of the template XML file, initiates the XML class and writes the cover page
        :return: sets class variables
                    self.XML
        '''

        # new_xml = os.path.join(self.studyDir, 'reports', 'Datasources', 'USBRAutomatedReportOutput.xml') #required name for file
        new_xml = os.path.join(self.outputDir, 'Datasources', 'USBRAutomatedReportOutput.xml') #required name for file
        print(f'Creating new XML at {new_xml}')

        self.XML = WXMLU.XMLReport(new_xml)
        self.XML.writeCover('DRAFT Temperature Validation Summary Report')

    def initializeDataOrganizer(self):
        '''
        create Data_Memory dictionary
        '''

        self.Data = WDO.DataOrganizer(self)

    def initSimulationDict(self):
        '''
        create simulationVariables dictionary
        '''

        self.SimulationVariables = {}

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
        '''
        attempts to figure out what model the data is for based off the model source as an attempt to filter out data
        retrieval attempts
        :param Line_info: dictionary of settings for line
        :return: model plugin name if possible
        '''

        for var, ident in self.Constants.model_specific_vars.items():
            if var in Line_info.keys():
                return ident

        return 'undefined' #no id either way..

    def appendXMLModelIntroduction(self, simorder):
        '''
        Fixes intro in XML that shows what models are used for each region.
        Updates a flag with used models.
        :param simorder: number of simulation file
        :return:
        '''

        modelstrs = []
        for Chapter in self.ChapterDefinitions:
            chapname = Chapter['name']
            outstr = f'<Model ModelOrder="%%modelOrder%%" >{chapname}:'
            for cnt, ID in enumerate(self.accepted_IDs):
                if cnt > 0:
                    outstr += ','
                outstr += ' {0}'.format(self.SimulationVariables[ID]['plugin'])
            outstr += '</Model>\n'
            modelstrs.append(outstr)

        lastline = '%%REPLACEINTRO_{0}%%'.format(simorder)
        for i, ms in enumerate(modelstrs):
            tmpstr = ms.replace('%%modelOrder%%', str(self.modelOrder))
            self.XML.insertAfter(lastline, tmpstr)
            self.modelOrder += 1
            lastline = tmpstr

    def fixXMLModelIntroduction(self):
        self.XML.removeLine('%%REPLACEINTRO_')

    def checkModelType(self, line_info):
        '''
         checks to see if current data path configuration is congruent with currently loaded model ID.
        :param line_info: selected line or datapath
        :return: boolean
        '''

        modeltype = self.getLineModelType(line_info)
        if modeltype == 'undefined':
            return True
        if modeltype.lower() != self.plugin.lower():
            return False
        return True

    def configureSettingsForID(self, ID, settings):
        '''
        loads settings for selected run ID. Mainly for comparison plots. The replaces model specific flags in settings
        using loaded variables
        :param ID: selected ID, aka 'base' or 'alt_1'
        :param settings: dictionary of settings possibly containing flags to replace
        :return: settings with updated flags. also flags such as self.baseSimulationName are updated to current ID
        '''

        self.loadCurrentID(ID)
        self.loadCurrentModelAltID(ID)
        settings = WF.replaceflaggedValues(self, settings, 'modelspecific')
        return settings

if __name__ == '__main__':

    if '--version' in sys.argv:
        WF.print2stdout(VERSIONNUMBER)
        sys.exit(0)
    else:
        rundir = sys.argv[0]
        simInfoFile = sys.argv[1]

        # import cProfile
        # ar = cProfile.run('MakeAutomatedReport(simInfoFile, rundir)')

        try:
            MakeAutomatedReport(simInfoFile, rundir)
        except:
            WF.print2stderr(traceback.format_exc())
            sys.exit(1)

