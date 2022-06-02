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

class Plots(object):

    def __init__(self, Report):
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
        curaxis.plot(x, y, label=settings['label'], c=settings['linecolor'],
                     lw=settings['linewidth'], ls=settings['linestylepattern'],
                     marker=settings['symboltype'], markerfacecolor=settings['pointfillcolor'],
                     markeredgecolor=settings['pointlinecolor'], markersize=float(settings['symbolsize']),
                     markevery=int(settings['numptsskip']), zorder=float(settings['zorder']),
                     alpha=float(settings['alpha']))

    def plotLines(self, x, y, curaxis, settings):
        curaxis.plot(x, y, label=settings['label'], c=settings['linecolor'],
                     lw=settings['linewidth'], ls=settings['linestylepattern'],
                     zorder=float(settings['zorder']),
                     alpha=float(settings['alpha']))

    def plotPoints(self, x, y, curaxis, settings):
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
        for tick in ticks:
            if isinstance(tick, (np.float64, int, float)):

                if 'numdecimals' in ticksettings.keys():
                    numdecimals = int(ticksettings['numdecimals'])
                else:
                    numdecimals = 1

                if numdecimals == 0:
                    newticklabel = int(round(tick, 0))
                    newticklabels.append(str(newticklabel))
                else:
                    newticklabel = round(tick, numdecimals)
                    if numdecimals == 1: #fix numbers like 10.0
                        if str(newticklabel).split('.')[1].startswith('0'):
                            newticklabel = str(newticklabel).split('.')[0]
                            newticklabels.append(newticklabel)
                        else:
                            newticklabels.append(str(newticklabel))

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
                if '.' in xtickspacing:
                    xtickspacing = float(xtickspacing)
                else:
                    xtickspacing = int(xtickspacing)
                newxticks = np.arange(xmin, (xmax+xtickspacing), xtickspacing)
            elif axis_settings[dateformatflag].lower() == 'datetime':
                dt_xmin = WT.JDateToDatetime(xmin, self.Report.startYear) #do everything on datetime, and we can convert later
                dt_xmax = WT.JDateToDatetime(xmax, self.Report.startYear) #do everything on datetime, and we can convert later
                newxticks = WT.buildTimeSeries(dt_xmin.replace(tzinfo=None), dt_xmax.replace(tzinfo=None), xtickspacing)

            newxticklabels = Plots.formatTickLabels(newxticks, xtick_settings)
            curax.set_xticks(newxticks)
            curax.set_xticklabels(newxticklabels)


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


