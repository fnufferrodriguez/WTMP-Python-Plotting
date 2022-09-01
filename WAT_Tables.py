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
import pickle
import numpy as np
from functools import reduce
from matplotlib.colors import to_hex
from scipy import interpolate

import WAT_Functions as WF
import WAT_Time as WT

class Tables(object):

    def __init__(self, Report):
        '''
        Class to control table objects
        :param Report: self class from main Report Generator script
        '''
        self.Report = Report

    def buildHeadersByTimestamps(self, timestamps, years):
        '''
        build headers for profile line stat tables by timestamp
        convert to Datetime, no matter what. We can convert back..
        Filter by year, using year input. If ALLYEARS, no data is filtered.
        :param timestamps: list of available timesteps
        :param years: used to filter down to the year, or if ALLYEARS, allow all years
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
                    ts_dt = WT.JDateToDatetime(timestamp)
                    if year == ts_dt.year:
                        h.append(str(timestamp))
                        hi.append(ti)
            headers.append(h)
            headers_i.append(hi)

        return headers, headers_i

    def buildTable(self, object_settings, split_by_year, data_settings):
        '''
        builds a table header and rows
        :param object_settings: dictionary containing settings for object
        :param split_by_year: boolean flag to determine if table is for all years, or split up
        :return: headers and rows lists
        '''

        headers = {}
        rows = {}
        outputyears = object_settings['years'] #this is usually a range or ALLYEARS

        for year in outputyears:
            headers[year] = []
            rows[year] = {}
            for ri, row in enumerate(object_settings['rows']):
                rows[year][ri] = []

        # conditions
        # -single run all years
        # -single run split by year
        # -single run split by year and include all years
        # -comp run all years
        # -comp run split by year
        # -comp run split by year and include all years

        #comp run but header == '%%year%%'
        #comp run but header has %%computed%% in it

        for i, header in enumerate(object_settings['headers']):
            curheader = pickle.loads(pickle.dumps(header, -1))
            if self.Report.iscomp: #comp run
                isused = False
                for datakey in data_settings.keys():
                    if '%%{0}%%'.format(data_settings[datakey]['flag']) in curheader: #found data specific flag
                        isused = True
                        if 'ID' in data_settings[datakey].keys():
                            ID = data_settings[datakey]['ID']
                            tmpheader = self.Report.configureSettingsForID(ID, curheader)
                        else:
                            tmpheader = pickle.loads(pickle.dumps(curheader, -1))
                        tmpheader = tmpheader.replace('%%{0}%%'.format(data_settings[datakey]['flag']), '')
                        if tmpheader == '%%year%%': #find a situation where the header is just the year
                            tmpheader = self.Report.SimulationName
                        if split_by_year:
                            for year in outputyears:
                                if year == 'ALLYEARS':
                                    headers[year].append(tmpheader.replace('%%year%%', self.Report.years_str))
                                else:
                                    headers[year].append(tmpheader)
                                for ri, row in enumerate(object_settings['rows']):
                                    srow = row.split('|')[1:][i]
                                    rows[year][ri].append(srow.replace(data_settings[datakey]['flag'], datakey))
                        else:
                            headers[year].append(tmpheader.replace('%%year%%', self.Report.years_str))
                            for ri, row in enumerate(object_settings['rows']):
                                srow = row.split('|')[1:][i]
                                rows['ALLYEARS'][ri].append(srow.replace(data_settings[datakey]['flag'], datakey))

                if not isused: #if a header doesnt get used, probably something observed and not needing replacing. OR custom headers for computed data..
                    if split_by_year:
                        for year in outputyears:
                            if year == 'ALLYEARS':
                                headers[year].append(curheader.replace('%%year%%', self.Report.years_str))
                            else:
                                headers[year].append(curheader)
                            for ri, row in enumerate(object_settings['rows']):
                                srow = row.split('|')[1:]
                                if i > len(srow)-1:
                                    WF.print2stdout(f'Too many headers for defined rows. Not using data for {header}.')
                                    rows[year][ri].append(None)
                                else:
                                    rows[year][ri].append(srow[i])

                    else:
                        headers[year].append(curheader.replace('%%year%%', self.Report.years_str))
                        for ri, row in enumerate(object_settings['rows']):
                            srow = row.split('|')[1:][i]
                            rows['ALLYEARS'][ri].append(srow.replace(data_settings[datakey]['flag'], datakey))

            else: #single run

                if split_by_year:
                    for year in outputyears:
                        if year == 'ALLYEARS':
                            headers[year].append(curheader.replace('%%year%%', self.Report.years_str))
                        else:
                            headers[year].append(curheader)
                        for ri, row in enumerate(object_settings['rows']):
                            srow = row.split('|')[1:][i]
                            rows[year][ri].append(srow)
                else:
                    headers[year].append(curheader.replace('%%year%%', self.Report.years_str))
                    for ri, row in enumerate(object_settings['rows']):
                        srow = row.split('|')[1:][i]
                        rows['ALLYEARS'][ri].append(srow)

        organizedheaders = []
        organizedrows = []
        for row in object_settings['rows']:
            organizedrows.append(row.split('|')[0])
        for year in outputyears:
            yrstr = str(year) if split_by_year and year != 'ALLYEARS' else self.Report.years_str
            for hdr in headers[year]:
                organizedheaders.append([year, WF.updateFlaggedValues(hdr, '%%year%%', yrstr)])
            for ri in rows[year].keys():
                for rw in rows[year][ri]:
                    organizedrows[ri] += '|{0}'.format(rw)

        return organizedheaders, organizedrows

    def buildSingleStatTable(self, object_settings, data):
        '''
        builds headings and rows for single stat table
        :param object_settings: dictionary containing settings for table
        :param data: dictionary contianing data
        :return: headings and rows used to build tables
        '''

        headers = []
        rows = []
        months = [n for n in self.Report.Constants.mo_str_3]
        stat = object_settings['statistic']
        datakeys = list(data.keys())

        if stat in ['mean', 'count']:
            numflagsneeded = 1
        else:
            numflagsneeded = 2

        if self.Report.iscomp:
            if 'headers' in object_settings.keys():
                hdrs = object_settings['headers']
                for curheader in hdrs:
                    isused = False
                    for datakey in datakeys:
                        if '%%{0}%%'.format(data[datakey]['flag']) in curheader:
                            if 'ID' in data[datakey].keys():
                                ID = data[datakey]['ID']
                                tmpheader = self.Report.configureSettingsForID(ID, curheader)
                            else:
                                tmpheader = pickle.loads(pickle.dumps(curheader, -1))
                            tmpheader = tmpheader.replace('%%{0}%%'.format(data[datakey]['flag']), '')
                            headers.append(tmpheader)
                            isused = True
                    if not isused:
                        if '%%' in curheader: #check for unused flags.. if theyre there, move on
                            continue
                        else:
                            headers.append(curheader)

            else:
                for datakey in datakeys:
                    if 'label' in data[datakey].keys():
                        if numflagsneeded == 2:
                            if data[datakey]['flag'].lower() != 'computed':
                                headers.append(data[datakey]['label'])
                        else:
                            headers.append(data[datakey]['label'])
                    else:
                        headers.append(datakey)
        else:
            headers = months

        computed_keys = []
        observed_keys = []
        for datakey in datakeys:
            if data[datakey]['flag'].lower() == 'computed': #get the easy ones
                computed_keys.append(datakey)
            elif data[datakey]['flag'].lower() == 'observed':
                observed_keys.append(datakey)
        if len(computed_keys) == 0 and len(observed_keys) > 0: #if none are computed and some are observed, assume the rest computed
            for datakey in datakeys:
                if data[datakey]['flag'].lower() not in observed_keys:
                    computed_keys.append(datakey)
        if len(observed_keys) == 0 and len(computed_keys) > 0: #if some are computed and none are observed, assume the rest computed
            for datakey in datakeys:
                if data[datakey]['flag'].lower() not in computed_keys:
                    observed_keys.append(datakey)


        for year in object_settings['years']:
            row = f'{year}'
            for month in months:
                if numflagsneeded == 1:
                    for datakey in computed_keys:
                        row += f'|%%{stat}.{datakey}.MONTH={month.upper()}%%'
                else:
                    for cflag in computed_keys:
                        for oflag in observed_keys:
                            row += f'|%%{stat}.{cflag}.MONTH={month.upper()}.{data[oflag]["flag"]}.MONTH={month.upper()}%%'
            rows.append(row)

        return headers, rows


    def buildProfileStatsTable(self, object_settings, timestamp, data):
        '''
        builds a table header and rows for profile statistics
        :param object_settings: dictionary containing settings
        :param timestamp: current timestamp to configure table for
        :param data: dictionary contianing data
        :return: list of headers, list of rows
        '''

        headers = []
        rows = []
        for ri, row in enumerate(object_settings['rows']):
            rows.append(row.split('|')[0])

        if self.Report.iscomp: #comp run
            for i, header in enumerate(object_settings['headers']):
                curheader = pickle.loads(pickle.dumps(header, -1))
                for datakey in data.keys():
                    if '%%{0}%%'.format(data[datakey]['flag']) in curheader: #found data specific flag
                        if 'ID' in data[datakey].keys():
                            ID = data[datakey]['ID']
                            tmpheader = self.Report.configureSettingsForID(ID, curheader)
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

    def filterTableData(self, data, object_settings):
        '''
        filters out data through xlims, ylims and omitvalues.
        :param data: dictionary of data
        :param object_settings: settings for current object containing limits
        :return: filtered data dictionary
        '''

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

    def correctTableUnits(self, data, data_settings, object_settings):
        '''
        converts units for table data
        :param data: dictionary containing data
        :param data_settings: data metadata
        :param object_settings: settings for plot/table
        :return: data with updated units
        '''

        for datapath in data.keys():
            values = data[datapath]['values']
            units = data_settings[datapath]['units']
            if 'parameter' in data_settings[datapath].keys():
                units = WF.configureUnits(object_settings, data_settings[datapath]['parameter'], units)
            if 'unitsystem' in object_settings.keys():
                data[datapath]['values'], data_settings[datapath]['units'] = WF.convertUnitSystem(values, units, object_settings['unitsystem'])

        return data

    def getStatsLineData(self, row, data_dict, year='ALLYEARS'):
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
                                sr_month = self.Report.Constants.month2num[sr_month.lower()]
                            except KeyError:
                                WF.print2stdout('Invalid Entry for {0}'.format(sr))
                                WF.print2stdout('Try using interger values or 3 letter monthly code.')
                                WF.print2stdout('Ex: MONTH=1 or MONTH=JAN')
                                continue
                        if curflag == None:
                            WF.print2stdout('Invalid Table row for {0}'.format(row))
                            WF.print2stdout('Data Key not contained within {0}'.format(data_dict.keys()))
                            WF.print2stdout('Please check Datapaths in the XML file, or modify the rows to have the correct flags'
                                  ' for the data present')
                            return data, ''

                        newvals = np.array([])
                        newdates = np.array([])
                        if year != 'ALLYEARS':
                            year_loops = [year]
                        else:
                            year_loops = self.Report.years
                        if len(curdates) > 0:
                            for yearloop in year_loops:
                                s_idx, e_idx = WF.getYearlyFilterIdx(curdates, yearloop)
                                yearvals = curvalues[s_idx:e_idx+1]
                                yeardates = curdates[s_idx:e_idx+1]

                                if len(yeardates) > 0:
                                    s_idx, e_idx = WF.getMonthlyFilterIdx(yeardates, sr_month)

                                    newvals = np.append(newvals, yearvals[s_idx:e_idx+1])
                                    newdates = np.append(newdates, yeardates[s_idx:e_idx+1])

                        data[curflag]['values'] = newvals
                        data[curflag]['dates'] = newdates

        if year != 'ALLYEARS':
            for flag in data.keys():
                if len(data[flag]['dates']) == 0:
                    continue
                s_idx, e_idx = WF.getYearlyFilterIdx(data[flag]['dates'], year)
                data[flag]['values'] = data[flag]['values'][s_idx:e_idx+1]
                data[flag]['dates'] = data[flag]['dates'][s_idx:e_idx+1]

        return data, sr_month

    def getStatsLine(self, row, data):
        '''
        takes rows for tables and replaces flags with the correct data, computing stat analysis if needed
        :param row: row section string
        :param data: dictionary of data that could be used
        :return:
            out_stat: stat value
            stat: string name for stat
        '''

        stat_flag_Req = {'%%meanbias': 2,
                         '%%mae': 2,
                         '%%rmse': 2,
                         '%%nse': 2,
                         '%%count': 1,
                         '%%mean': 1}

        numFlagsReqd=2 #start with most restrictive..
        for key in stat_flag_Req.keys():
            if row.lower().startswith(key):
                numFlagsReqd = stat_flag_Req[key]
                break

        flags = list(data.keys())

        if len(flags) > 0:
            getdata = True
            if 'Computed' in flags:
                flag1 = 'Computed'
                if numFlagsReqd == 2:
                    if len(flags) >= 2:
                        if 'Observed' in flags:
                            flag2 = 'Observed'
                        else:
                            flag2 = [n for n in flags if n != flag1][0] #not computed
                    else:
                        getdata=False
            else:
                flag1 = flags[0]
                if numFlagsReqd == 2:
                    if len(flags) >= 2:
                        flag2 = flags[1]
                    else:
                        getdata = False
        else:
            getdata = False

        out_stat = np.nan

        for key in data.keys():
            if len(data[key]) == 0:
                WF.print2stdout('Insufficient data.')
                getdata = False

        if row.lower().startswith('%%meanbias'):
            if getdata:
                out_stat = WF.calcMeanBias(data[flag1], data[flag2])
            stat = 'meanbias'
        elif row.lower().startswith('%%mae'):
            if getdata:
                out_stat = WF.calcMAE(data[flag1], data[flag2])
            stat = 'mae'
        elif row.lower().startswith('%%rmse'):
            if getdata:
                out_stat = WF.calcRMSE(data[flag1], data[flag2])
            stat = 'rmse'
        elif row.lower().startswith('%%nse'):
            if getdata:
                out_stat = WF.calcNSE(data[flag1], data[flag2])
            stat = 'nse'
        elif row.lower().startswith('%%count'):
            if getdata:
                out_stat = WF.getCount(data[flag1])
            stat = 'count'
        elif row.lower().startswith('%%mean'):
            if getdata:
                out_stat = WF.calcMean(data[flag1])
            stat = 'mean'
        else:
            if '%%' in row:
                WF.print2stdout('Unable to convert flag in row', row)
            return row, ''

        return out_stat, stat

    def matchThresholdToStat(self, stat, object_settings):
        '''
        matches prescribed threshold values to statistic value
        :param stat: string name of stat
        :param object_settings: dictionary containing object settings
        :return:
        '''
        thresholds = []
        if 'tablecolors' in object_settings.keys():
            for tablecolor in object_settings['tablecolors']:
                if 'statistic' in tablecolor.keys():
                    if stat.lower() == tablecolor['statistic'].lower():
                        thresholds += self.formatThreshold(tablecolor)
                else:
                    thresholds += self.formatThreshold(tablecolor)
        return thresholds

    def matchNumberFormatByStat(self, stat, settings):
        '''
        matches how numbers should be formatted by what stat they are using user settings in tables
        :param stat: current statistical value
        :param settings: dictionary of settings to parse
        :return: formatting
        '''

        numberFormats_default = []
        numberFormats_statspec = []

        if 'numberformats' in settings.keys():
            for numberformat in settings['numberformats']:
                if 'stats' in numberformat:
                    if stat.lower() in [n.lower() for n in numberformat['stats']]:
                        numberFormats_statspec.append(numberformat)
                else:
                    numberFormats_default.append(numberformat)
        if isinstance(stat, str):
            if stat.lower() == 'count':
                if len(numberFormats_statspec) == 0:
                    numberFormats_statspec.append({'decimalplaces': 0})

        if len(numberFormats_statspec) > 0:
            return numberFormats_statspec
        else:
            return numberFormats_default

    def formatThreshold(self, object_settings):
        '''
        organizes settings for thresholds for stat tables. Fills in missing values with defaults
        :param object_settings: dictionary containing settings for thresholds
        :return: list of dictionary objects for each threshold
        '''

        default_color = '#a6a6a6' #default, grey
        default_colorwhen = 'under' #default
        accepted_threshold_conditions = ['under', 'over']
        thresholds = []

        if 'thresholds' in object_settings.keys():
            for threshold in object_settings['thresholds']:

                if 'value' in threshold.keys():
                    threshold_value = float(threshold['value'])
                else:
                    continue #dont record this threshold

                if 'color' in threshold.keys():
                    threshold_color = self.formatThresholdColor(threshold['color'], default=default_color)
                else:
                    threshold_color = default_color

                if 'colorwhen' in threshold.keys():
                    if any([n.lower() == threshold['colorwhen'].lower() for n in accepted_threshold_conditions]):
                        threshold_colorwhen = threshold['colorwhen'].lower()
                    else:
                        WF.print2stdout(f"Invalid threshold setting {threshold['colorwhen']}")
                        WF.print2stdout(f'Please select value in {accepted_threshold_conditions}')
                else:
                    threshold_colorwhen = default_colorwhen

                thresholds.append({'value': threshold_value, 'color': threshold_color, 'colorwhen': threshold_colorwhen})

        return thresholds

    def formatThresholdColor(self, in_color, default='#a6a6a6'):
        '''
        formats input color to either turn to hex or use default
        :param in_color: string to test if legit
        :param default: color to use if in_color fails
        :return:hex color
        '''

        threshold_color = default
        if in_color.startswith('#'):
            threshold_color = in_color
        else:
            try:
                threshold_color = to_hex(in_color)
            except ValueError:
                WF.print2stdout(f'Invalid color of {in_color}')

        return threshold_color

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
                xmin = WT.translateDateFormat(object_settings['xlims']['min'], 'datetime', self.StartTime,
                                              self.StartTime, self.EndTime,
                                              self.ModelAlt.t_offset)
                xmin = xmin.strftime('%d %b %Y')
            if 'max' in object_settings['xlims'].keys():
                xmax = WT.translateDateFormat(object_settings['xlims']['max'], 'datetime', self.EndTime,
                                              self.StartTime, self.EndTime,
                                              self.ModelAlt.t_offset)
                xmax = xmax.strftime('%d %b %Y')

        if xmin != 'NONE':
            start_date = xmin
        elif year == self.Report.startYear:
            start_date = self.Report.StartTime.strftime('%d %b %Y')
        else:
            if str(year).lower() == 'allyears':
                start_date = '01 Jan {0}'.format(self.Report.startYear)
            else:
                start_date = '01 Jan {0}'.format(year)

        if xmax != 'NONE':
            start_date = xmax
        elif year == self.Report.endYear:
            end_date = self.Report.EndTime.strftime('%d %b %Y')
        else:
            if str(year).lower() == 'allyears':
                end_date = '31 Dec {0}'.format(self.Report.endYear)
            else:
                end_date = '31 Dec {0}'.format(year)
            if month != 'None':
                try:
                    month = int(month)
                except ValueError:
                    month = self.Report.Constants.month2num[month.lower()]

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
                    header = WT.translateDateFormat(header, 'datetime', '',
                                                    self.Report.StartTime, self.Report.EndTime,
                                                    self.Report.ModelAlt.t_offset)
                    header = header.strftime('%d%b%Y')
                elif object_settings['dateformat'].lower() == 'jdate':
                    header = WT.translateDateFormat(header, 'jdate', '',
                                                    self.Report.StartTime, self.Report.EndTime,
                                                    self.Report.ModelAlt.t_offset)
                    header = str(header)
                nh.append(header)
            new_headers.append(nh)

        return new_headers

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

        if bottom != None and top != None:
            if usedepth.lower() == 'true':
                #build elev profiles
                output_interp_depths = np.arange(top, bottom, (bottom-top)/float(resolution))
            else:
                output_interp_elevations = np.arange(bottom, top, (top-bottom)/float(resolution))
        else:
            output_interp_elevations = []
            output_interp_depths = []

        for flag in flags:
            out_data[flag] = {}
            #interpolate over all values and then get interp values

            if len(data_dict[flag]['values'][index]) < 2:
                WF.print2stdout('Insufficient data points with current bounds for {0}'.format(flag))
                out_data[flag]['values'] = []
                out_data[flag]['depths'] = []
                out_data[flag]['elevations'] = []
                continue
            elif usedepth.lower() == 'true':
                if len(output_interp_depths) == 0:
                    WF.print2stdout(f'Insufficient depth points for row {flag} in {row}')
                    out_data[flag]['values'] = []
                    out_data[flag]['depths'] = []
                    out_data[flag]['elevations'] = []
                    continue
            elif usedepth.lower() == 'false':
                if len(output_interp_elevations) == 0:
                    WF.print2stdout(f'Insufficient elevation points for row {flag} in {row}')
                    out_data[flag]['values'] = []
                    out_data[flag]['depths'] = []
                    out_data[flag]['elevations'] = []
                    continue
            # else:
            if usedepth.lower() == 'true':
                f_interp = interpolate.interp1d(data_dict[flag]['depths'][index], data_dict[flag]['values'][index], fill_value='extrapolate')
                out_data[flag]['depths'] = output_interp_depths
                out_data[flag]['values'] = f_interp(output_interp_depths)
            else:
                f_interp = interpolate.interp1d(data_dict[flag]['elevations'][index], data_dict[flag]['values'][index], fill_value='extrapolate')
                out_data[flag]['elevations'] = output_interp_elevations
                out_data[flag]['values'] = f_interp(output_interp_elevations)

        return out_data

    def replaceComparisonSettings(self, object_settings, iscomp):
        '''
        replaces normal settings with comparison settings if found to be comparison plot
        :param object_settings: dictionary containing settings
        :param iscomp: boolean, is comparison run or not
        :return: settings with replaced settings
        '''

        replace_flags = {'comparisonheaders': 'headers'}
        replaced_defaults = object_settings['replaced_defaults']
        if iscomp:
            for comparisonflag in replace_flags.keys():
                normalflag = replace_flags[comparisonflag]
                if comparisonflag in object_settings.keys():
                    if normalflag in replaced_defaults:
                        continue
                    object_settings[normalflag] = object_settings[comparisonflag]

        return object_settings

    def configureHeadingsGroups(self, headings):
        '''
        finds indicies of groupings for headers
        :param headings: list of headings
        :return: indicies of headers
        '''

        headings_groups = []
        for h in headings:
            if h[0] not in headings_groups:
                headings_groups.append(h[0])
        headings_i = [[] for n in headings_groups]
        for hgi, hgroup in enumerate(headings_groups):
            for hi, h in enumerate(headings):
                if str(h[0]) == str(hgroup):
                    headings_i[hgi].append(hi)
        return headings_i