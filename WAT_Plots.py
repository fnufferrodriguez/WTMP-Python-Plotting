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

import pickle
import datetime as dt
import numpy as np
import matplotlib as mpl

import WAT_Functions as WF
import WAT_Time as WT
import WAT_Profiles as WProfile
import WAT_Defaults as WD
import WAT_Reader as WR

class Plots(object):

    def __init__(self, Report):
        '''
        class that controls plotting functions
        :param Report: self from main Report Generator script
        '''

        self.Report = Report

    def confirmAxis(self, object_settings):
        '''
        Checks for an axis item in object settings. If not, make one.
        :param object_settings: dictionary containing settings for current object
        :return: object settings but with empty axis.
        '''

        if 'axs' not in object_settings.keys():
            object_settings['axs'] = [{}] #empty axis object
        return object_settings

    def setTimeSeriesXlims(self, cur_obj_settings, yearstr, years):
        '''
        gets the xlimits for time series. This can be dependent on year, so needs to be looped over.
        :param cur_obj_settings: current plotting object settings dictionary
        :param yearstr: current year string
        :param years: list of years
        :return: updated cur_obj_settings dict
        '''

        if 'ALLYEARS' not in years:
            cur_obj_settings = WF.updateFlaggedValues(cur_obj_settings, '%%year%%', yearstr)
        else:
            if 'xlims' in cur_obj_settings.keys():
                if 'min' in cur_obj_settings['xlims']:
                    cur_obj_settings['xlims']['min'] = WF.updateFlaggedValues(cur_obj_settings['xlims']['min'],
                                                                              '%%year%%', str(self.Report.years[0]))
                if 'max' in cur_obj_settings['xlims']:
                    cur_obj_settings['xlims']['max'] = WF.updateFlaggedValues(cur_obj_settings['xlims']['max'],
                                                                              '%%year%%', str(self.Report.years[-1]))
            cur_obj_settings = WF.updateFlaggedValues(cur_obj_settings, '%%year%%', yearstr)

        return cur_obj_settings

    def getRelativeMasterSet(self, linedata):
        '''
        organizes data and gets it on the same interval
        :param linedata: dictionary containing data
        :return: set of data on same interavl and units, settings for relative lines
        '''

        #add all thje data together. then we cna use this when plotting it to get %
        #TODO: deal with irregular intervals
        intervals = {}
        biggest_interval = None
        type = 'INST-VAL'
        for line in linedata.keys():
            if 'interval' in linedata[line].keys():
                td = WF.getTimeInterval(linedata[line]['dates'])
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
            curline['values'], curline['units'] = WF.convertUnitSystem(curline['values'], curline['units'], 'metric') #just make everything metric..
            units.append(curline['units'])
            if li == 0:
                if biggest_interval != None:
                    _, RelativeMasterSet = WT.changeTimeSeriesInterval(curline['dates'], curline['values'],
                                                                                RelativeLineSettings,
                                                                                self.Report.ModelAlt.t_offset,
                                                                                self.Report.startYear)
                else:
                    RelativeMasterSet = curline['values']
            else:
                if biggest_interval != None:
                    curline['interval'] = biggest_interval
                    curline['type'] = type
                    _, newvals = WT.changeTimeSeriesInterval(curline['dates'], curline['values'],
                                                              RelativeLineSettings,
                                                              self.Report.ModelAlt.t_offset,
                                                              self.Report.startYear)
                    RelativeMasterSet += newvals
                else:
                    RelativeMasterSet += curline['values']

        RelativeLineSettings['units'] = WF.getMostCommon(units)

        return RelativeMasterSet, RelativeLineSettings

    def plotLinesAndPoints(self, x, y, curaxis, settings):
        '''
        plots lines and points on line on given axis
        :param x: data list for x axis (dates or otherwise)
        :param y: data list for y axis
        :param curaxis: current axis object
        :param settings: dictionary containing settings for plot object
        '''

        curaxis.plot(x, y, label=settings['label'], c=settings['linecolor'],
                     lw=settings['linewidth'], ls=settings['linestylepattern'],
                     marker=settings['symboltype'], markerfacecolor=settings['pointfillcolor'],
                     markeredgecolor=settings['pointlinecolor'], markersize=float(settings['symbolsize']),
                     markevery=int(settings['numptsskip']), zorder=float(settings['zorder']),
                     alpha=float(settings['alpha']))

    def plotLines(self, x, y, curaxis, settings):
        '''
        plots lines on line on given axis
        :param x: data list for x axis (dates or otherwise)
        :param y: data list for y axis
        :param curaxis: current axis object
        :param settings: dictionary containing settings for plot object
        '''

        curaxis.plot(x, y, label=settings['label'], c=settings['linecolor'],
                     lw=settings['linewidth'], ls=settings['linestylepattern'],
                     zorder=float(settings['zorder']),
                     alpha=float(settings['alpha']))

    def plotPoints(self, x, y, curaxis, settings):
        '''
        plots points on line on given axis
        :param x: data list for x axis (dates or otherwise)
        :param y: data list for y axis
        :param curaxis: current axis object
        :param settings: dictionary containing settings for plot object
        '''

        curaxis.scatter(x[::int(settings['numptsskip'])], y[::int(settings['numptsskip'])],
                        marker=settings['symboltype'], facecolor=settings['pointfillcolor'],
                        edgecolor=settings['pointlinecolor'], s=float(settings['symbolsize']),
                        label=settings['label'], zorder=float(settings['zorder']),
                        alpha=float(settings['alpha']))

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
                WF.print2stdout('Using Same Xlims for top and bottom.')
                xlims_flag = 'xlims'
            dateformat_flag = 'dateformat2'
        else:
            xlims_flag = 'xlims'
            dateformat_flag = 'dateformat'

        if dateformat_flag in object_settings.keys():
            dateformat = object_settings[dateformat_flag].lower()
        else:
            WF.print2stdout('Dateformat flag not set. Defaulting to datetime..')
            dateformat = 'datetime'

        if xlims_flag in object_settings.keys():
            xlims = object_settings[xlims_flag]#should be min max flags in here

            if 'min' in xlims.keys():
                min = xlims['min']
            else:
                if dateformat == 'datetime':
                    min = self.Report.StartTime
                elif dateformat == 'jdate':
                    min = WT.DatetimeToJDate(self.Report.StartTime, self.Report.ModelAlt.t_offset)
                else:
                    #we've done everything we can at this point..
                    min = self.Report.StartTime

            if 'max' in xlims.keys():
                max = xlims['max']
            else:
                if dateformat == 'datetime':
                    max = self.Report.EndTime
                elif dateformat == 'jdate':
                    max = WT.DatetimeToJDate(self.Report.EndTime, self.Report.ModelAlt.t_offset)
                else:
                    #we've done everything we can at this point..
                    max = self.Report.StartTime

            min = WT.translateDateFormat(min, dateformat, self.Report.StartTime, self.Report.StartTime, self.Report.EndTime,
                                         self.Report.ModelAlt.t_offset)
            max = WT.translateDateFormat(max, dateformat, self.Report.EndTime, self.Report.StartTime,
                                         self.Report.EndTime, self.Report.ModelAlt.t_offset)

            curax.set_xlim(left=min, right=max)

        else:
            WF.print2stdout('No Xlims flag set for {0}'.format(xlims_flag))
            WF.print2stdout('Not setting Xlims.')

    def formatTickLabels(self, ticks, ticksettings):
        '''
        changes ticks based on input settings
        :param ticks: existing ticks
        :param ticksettings: dictionary contianing settings for ticks
        :return: formatted ticks
        '''

        newticklabels = []
        allticksinteger = False
        if np.all([float(tick).is_integer() if isinstance(tick, (int, float)) else False for tick in ticks]):
            allticksinteger=True

        for tick in ticks:
            if isinstance(tick, (int, float)):

                if 'numdecimals' in ticksettings.keys():
                    numdecimals = int(ticksettings['numdecimals'])
                elif allticksinteger:
                    numdecimals = 0
                else:
                    numdecimals = 2

                newticklabels.append('{num:,.{digits}f}'.format(num=tick, digits=numdecimals))

            elif isinstance(tick, dt.datetime):
                if 'datetimeformat' in ticksettings.keys():
                    datetimeformat = ticksettings['datetimeformat']
                else:
                    datetimeformat = '%m/%d/%Y'
                tick_str = tick.strftime(datetimeformat)
                newticklabels.append(tick_str)

            else:
                newticklabels.append(str(tick))

        return newticklabels

    def formatTimeSeriesXticks(self, curax, xtick_settings, axis_settings, dateformatflag='dateformat'):
        '''
        applies tick settings to Xaxis for time series
        :param curax: current axis object
        :param xtick_settings: dictionary containing settings for ticks
        :param axis_settings: dictionary containing settings for ticks
        :param dateformatflag: flag to get dateformat from settings
        '''

        xmin, xmax = curax.get_xlim()

        if axis_settings[dateformatflag].lower() == 'datetime':
            xmin = mpl.dates.num2date(xmin)
            xmax = mpl.dates.num2date(xmax)

        if 'fontsize' in xtick_settings.keys():
            xticksize = float(xtick_settings['fontsize'])
        elif 'fontsize' in axis_settings.keys():
            xticksize = float(axis_settings['fontsize'])
        else:
            xticksize = 10

        if 'rotation' in xtick_settings.keys():
            rotation = float(xtick_settings['rotation'])
        else:
            rotation = 0

        curax.tick_params(axis='x', labelsize=xticksize, rotation=rotation)

        if 'onmonths' in xtick_settings.keys():
            if isinstance(xtick_settings['onmonths'], dict):
                xtick_settings['onmonths'] = [xtick_settings['onmonths']['month']]
            bymonthday = [1]

            if 'ondays' in xtick_settings.keys():
                if isinstance(xtick_settings['ondays'], dict):
                    xtick_settings['ondays'] = [xtick_settings['ondays']['day']]
                bymonthday = [int(n) for n in xtick_settings['ondays']]

            try:
                locator = mpl.dates.MonthLocator([int(n) for n in xtick_settings['onmonths']], bymonthday=bymonthday)
            except ValueError:
                WF.print2stdout('Invalid month values. Please use integer representation of Months (aka 1, 3, 5, etc...)')
                formatted_months = [self.Report.Constants.month2num[n.lower()] for n in xtick_settings['onmonths']]
                locator = mpl.dates.MonthLocator(formatted_months, bymonthday=bymonthday)

            curax.xaxis.set_major_locator(locator)
            if axis_settings[dateformatflag].lower() == 'datetime':
                if 'datetimeformat' in xtick_settings.keys():
                    datetimeformat = xtick_settings['datetimeformat']
                else:
                    datetimeformat = '%b/%Y'
                fmt = mpl.dates.DateFormatter(datetimeformat)
                curax.xaxis.set_major_formatter(fmt)

        elif 'ondays' in xtick_settings.keys():
            if isinstance(xtick_settings['ondays'], dict):
                xtick_settings['ondays'] = [xtick_settings['ondays']['day']]

            locator = mpl.dates.DayLocator([int(n) for n in xtick_settings['ondays']])
            curax.xaxis.set_major_locator(locator)
            if axis_settings[dateformatflag].lower() == 'datetime':
                if 'datetimeformat' in xtick_settings.keys():
                    datetimeformat = xtick_settings['datetimeformat']
                else:
                    datetimeformat = '%m/%d/%Y'
                fmt = mpl.dates.DateFormatter(datetimeformat)
                curax.xaxis.set_major_formatter(fmt)

        elif 'spacing' in xtick_settings.keys():
            xtickspacing = xtick_settings['spacing']
            if axis_settings[dateformatflag].lower() == 'jdate':
                if float(xtickspacing).is_integer():
                    xtickspacing = int(xtickspacing)
                else:
                    xtickspacing = float(xtickspacing)
                newxticks = np.arange(xmin, (xmax+xtickspacing), xtickspacing)
            elif axis_settings[dateformatflag].lower() == 'datetime':
                dt_xmin = WT.JDateToDatetime(xmin, self.Report.startYear) #do everything on datetime, and we can convert later
                dt_xmax = WT.JDateToDatetime(xmax, self.Report.startYear) #do everything on datetime, and we can convert later
                newxticks = WT.buildTimeSeries(dt_xmin.replace(tzinfo=None), dt_xmax.replace(tzinfo=None), xtickspacing)

            newxticklabels = Plots.formatTickLabels(newxticks, xtick_settings)
            curax.set_xticks(newxticks)
            curax.set_xticklabels(newxticklabels)

    def formatYTicks(self, ax, ax_settings, gatedata={}, gate_placement=10, axis='left'):

        if axis == 'left':
            ylimflag = 'ylims'
            yticksflag = 'yticks'
        else:
            ylimflag = 'ylims2'
            yticksflag = 'yticks2'

        ymin, ymax = ax.get_ylim()
        if ylimflag in ax_settings.keys():
            if 'min' in ax_settings[ylimflag]:
                ymin = float(ax_settings[ylimflag]['min'])

            if 'max' in ax_settings[ylimflag]:
                ymax = float(ax_settings[ylimflag]['max'])

        if len(gatedata.keys()) != 0:
            ymax = gate_placement
            ymin = 0

        ax.set_ylim(bottom=ymin)
        ax.set_ylim(top=ymax)

        if yticksflag in ax_settings.keys():
            ytick_settings = ax_settings[yticksflag]
            if 'fontsize' in ytick_settings.keys():
                yticksize = float(ytick_settings['fontsize'])
            elif 'fontsize' in ax_settings.keys():
                yticksize = float(ax_settings['fontsize'])
            else:
                yticksize = 10
            ax.tick_params(axis='y', labelsize=yticksize)

            if 'spacing' in ytick_settings.keys():
                ytickspacing = ytick_settings['spacing']

                if float(ytickspacing).is_integer():
                    ytickspacing = int(ytickspacing)
                else:
                    ytickspacing = float(ytickspacing)

                if 'ylims' not in ax_settings.keys():
                    ymax = int(np.ceil(ymax))
                    ymin = int(np.floor(ymin))
                else:
                    if 'min' not in ax_settings[ylimflag].keys():
                        ymin = int(np.floor(ymin))
                    if 'max' not in ax_settings[ylimflag].keys():
                        ymax = int(np.ceil(ymax))

                newyticks = np.arange(ymin, (ymax+ytickspacing), ytickspacing)
                newyticklabels = self.formatTickLabels(newyticks, ytick_settings)
                ax.set_yticks(newyticks)
                ax.set_yticklabels(newyticklabels)

                ax.set_ylim(bottom=min(newyticks))
                ax.set_ylim(top=max(newyticks))
                return

        newyticklabels = self.formatTickLabels(ax.get_yticks(), {})
        ax.set_yticks(ax.get_yticks())
        ax.set_yticklabels(newyticklabels)
        ax.set_ylim(bottom=ymin)
        ax.set_ylim(top=ymax)

    def formatXTicks(self, ax, ax_settings, axis='bottom'):

        if axis == 'bottom':
            xlimflag = 'xlims'
            xticksflag = 'xticks'
        else:
            xlimflag = 'xlims2'
            xticksflag = 'xticks2'

        xmin, xmax = ax.get_xlim()
        if 'xlims' in ax_settings.keys():
            if 'min' in ax_settings[xlimflag]:
                xmin = float(ax_settings[xlimflag]['min'])
            if 'max' in ax_settings[xlimflag]:
                xmax = float(ax_settings[xlimflag]['max'])

        if xticksflag in ax_settings.keys():
            xtick_settings = ax_settings[xticksflag]
            if 'fontsize' in xtick_settings.keys():
                xticksize = float(xtick_settings['fontsize'])
            elif 'fontsize' in ax_settings.keys():
                xticksize = float(ax_settings['fontsize'])
            else:
                xticksize = 10
            ax.tick_params(axis='x', labelsize=xticksize)

            if 'spacing' in xtick_settings.keys():
                xtickspacing = xtick_settings['spacing']
                if float(xtickspacing).is_integer():
                    xtickspacing = int(xtickspacing)
                else:
                    xtickspacing = float(xtickspacing)
                newxticks = np.arange(xmin, (xmax+xtickspacing), xtickspacing)
                newxticklabels = self.formatTickLabels(newxticks, xtick_settings)
                # ax.xaxis.set_major_locator(mticker.FixedLocator(newxticks))
                ax.set_xticks(newxticks)
                ax.set_xticklabels(newxticklabels)

        ax.set_xlim(left=xmin)
        ax.set_xlim(right=xmax)

    def plotHorizontalLines(self, object_settings, ax, timestamp_index=None):
        '''
        organizes given data and plots horizontal lines on a given axis
        :param object_settings: dictionary potentionally containing <hlines> field
        :param ax: current axis
        :param timestamp_index: optional datetime index if hline is a timeseries
        '''

        if 'hlines' in object_settings.keys():
            for hline_settings in object_settings['hlines']:
                if 'value' in hline_settings.keys():
                    value = float(hline_settings['value'])
                    units = None
                else:
                    dates, values, units = self.Report.Data.getTimeSeries(hline_settings, makecopy=False)
                    if timestamp_index != None:
                        hline_idx = WR.getClosestTime([object_settings['timestamps'][timestamp_index]], dates)
                    else:
                        return
                    if len(hline_idx) == 0:
                        value = np.nan
                    else:
                        value = values[hline_idx[0]]

                if 'parameter' in hline_settings:
                    if object_settings['usedepth'].lower() == 'true':
                        if hline_settings['parameter'].lower() == 'elevation':
                            value = 0 #top of the water, should always be 0
                    elif object_settings['usedepth'].lower() == 'false':
                        if hline_settings['parameter'].lower() == 'depth':
                            valueconv = WProfile.convertDepthsToElevations({'hline': {'depths': [value],
                                                                                      'elevations': []}})
                            value = valueconv['hline']['elevation'][0]

                #currently cant convert these units..
                # if units != None:
                #     valueconv, units = WF.convertUnitSystem(value, units, object_settings['unitsystem'])
                #     value = valueconv[0]

                ### instead, use scalar to be manual
                if 'scalar' in hline_settings.keys():
                    value *= float(hline_settings['scalar'])

                hline_settings = WD.getDefaultStraightLineSettings(hline_settings)
                if 'label' not in hline_settings.keys():
                    hline_settings['label'] = None
                if 'zorder' not in hline_settings.keys():
                    hline_settings['zorder'] = 3

                ax.axhline(value, label=hline_settings['label'], c=hline_settings['linecolor'],
                           lw=hline_settings['linewidth'], ls=hline_settings['linestylepattern'],
                           zorder=float(hline_settings['zorder']),
                           alpha=float(hline_settings['alpha']))

    def plotVerticalLines(self, object_settings, ax, isdate=False, timestamp_index=None):
        '''
        organizes given data and plots vertical lines on a given axis
        :param object_settings: dictionary potentionally containing <vlines> field
        :param ax: current axis
        :param isdate: optional if input value is a date, in which case we may need to format it
        :param timestamp_index: optional datetime index if vline is a timeseries
        '''

        if 'vlines' in object_settings.keys():
            for vline in object_settings['vlines']:
                vline_settings = WD.getDefaultStraightLineSettings(vline)
                if 'value' in vline_settings.keys():
                    try:
                        value = float(vline_settings['value'])
                    except:
                        value = WT.translateDateFormat(vline_settings['value'], 'datetime', '',
                                                                         self.Report.StartTime, self.Report.EndTime,
                                                                         self.Report.ModelAlt.t_offset)
                else:
                    dates, values, units = self.Report.Data.getTimeSeries(vline_settings, makecopy=False)
                    if timestamp_index != None:
                        vline_idx = WR.getClosestTime([object_settings['timestamps'][timestamp_index]], dates)
                    else:
                        return
                    if len(vline_idx) == 0:
                        value = np.nan
                    else:
                        value = values[vline_idx[0]]

                if isdate:
                    if 'dateformat' in object_settings.keys():
                        if object_settings['dateformat'].lower() == 'jdate':
                            if isinstance(value, dt.datetime):
                                vline_settings['value'] = WT.DatetimeToJDate(value, self.Report.ModelAlt.t_offset)
                            elif isinstance(value, str):
                                try:
                                    value = float(value)
                                except:
                                    value = WT.translateDateFormat(value, 'datetime', '',
                                                                     self.Report.StartTime, self.Report.EndTime,
                                                                     self.Report.ModelAlt.t_offset)
                                    value = WT.DatetimeToJDate(value, self.Report.ModelAlt.t_offset)
                        elif object_settings['dateformat'].lower() == 'datetime':
                            if isinstance(value, (int,float)):
                                value = WT.JDateToDatetime(value, self.Report.startYear)
                            elif isinstance(value, str):
                                value = WT.translateDateFormat(value, 'datetime', '',
                                                                 self.Report.StartTime, self.Report.EndTime,
                                                                 self.Report.ModelAlt.t_offset)
                    else:
                        value = WT.translateDateFormat(value, 'datetime', '',
                                                         self.Report.StartTime, self.Report.EndTime,
                                                         self.Report.ModelAlt.t_offset)

                if 'label' not in vline_settings.keys():
                    vline_settings['label'] = None
                if 'zorder' not in vline_settings.keys():
                    vline_settings['zorder'] = 3

                ax.axvline(value, label=vline_settings['label'], c=vline_settings['linecolor'],
                           lw=vline_settings['linewidth'], ls=vline_settings['linestylepattern'],
                           zorder=float(vline_settings['zorder']),
                           alpha=float(vline_settings['alpha']))

    def fixEmptyYAxis(self, ax, ax2):
        right_handles, right_labels = ax.get_legend_handles_labels()
        left_handles, left_labels = ax2.get_legend_handles_labels()
        if len(right_handles) == 0:
            ax.set_yticks([])
            ax.set_yticklabels([])
        if len(left_handles) == 0:
            ax2.set_yticks([])
            ax2.set_yticklabels([])

def translateLineStylePatterns(LineSettings):
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
                WF.print2stdout('Invalid lineStylePattern:', LineSettings['linestylepattern'])
                WF.print2stdout('Defaulting to Solid.')
                LineSettings['linestylepattern'] = 'solid'
    else:
        WF.print2stdout('lineStylePattern undefined for line. Using solid')
        LineSettings['linestylepattern'] = 'solid'

    return LineSettings

def translatePointStylePatterns(LineSettings):
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
                WF.print2stdout('Invalid Symboltype:', LineSettings['symboltype'])
                WF.print2stdout('Defaulting to Square.')
                LineSettings['symboltype'] = 's'

    else:
        WF.print2stdout('Symbol not defined. Defaulting to Square.')
        LineSettings['symboltype'] = 's'

    return LineSettings


