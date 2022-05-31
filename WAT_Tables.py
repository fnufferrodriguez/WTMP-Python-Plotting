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

class Tables(object):

    def __init__(self, Report):
        self.Report = Report

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
                    ts_dt = self.Report.JDateToDatetime(timestamp)
                    if year == ts_dt.year:
                        h.append(str(timestamp))
                        hi.append(ti)
            headers.append(h)
            headers_i.append(hi)

        return headers, headers_i

    def buildTable(self, object_settings, split_by_year, data):
        '''
        builds a table header and rows
        :param object_settings: dictionary containing settings for object
        :param split_by_year: boolean flag to determine if table is for all years, or split up
        :param data: dictionary contining data
        :return: headers and rows lists
        '''

        headers = {}
        rows = {}
        outputyears = [n for n in self.Report.years] #this is usually a range or ALLYEARS
        outputyears.append('ALL') #do this last
        for year in outputyears:
            headers[year] = []
            rows[year] = {}
            for ri, row in enumerate(object_settings['rows']):
                rows[year][ri] = []

        for i, header in enumerate(object_settings['headers']):
            curheader = pickle.loads(pickle.dumps(header, -1))
            if self.Report.iscomp: #comp run
                isused = False
                for datakey in data.keys():
                    if '%%{0}%%'.format(data[datakey]['flag']) in curheader: #found data specific flag
                        isused = True
                        if 'ID' in data[datakey].keys():
                            ID = data[datakey]['ID']
                            tmpheader = self.Report.configureSettingsForID(ID, curheader)
                        else:
                            tmpheader = pickle.loads(pickle.dumps(curheader, -1))
                        tmpheader = tmpheader.replace('%%{0}%%'.format(data[datakey]['flag']), '')
                        if split_by_year and '%%year%%' in curheader:
                            for year in self.Report.years:
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
                        for year in self.Report.years:
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
                    for year in self.Report.years:
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
            yrstr = str(year) if split_by_year and year != 'ALL' else self.Report.years_str
            for hdr in headers[year]:
                organizedheaders.append([year, self.Report.updateFlaggedValues(hdr, '%%year%%', yrstr)])
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

        rows = []
        headers = [n for n in self.Report.Constants.mo_str_3]
        stat = object_settings['statistic']
        if stat in ['mean', 'count']:
            numflagsneeded = 1
        else:
            numflagsneeded = 2
        datakeys = list(data.keys())
        if len(data.keys()) < 2 and numflagsneeded != 1:
            print('\nWARNING: Insufficient amount of datapaths defined.')
            print(f'Need 2 datapaths to compute statistics for {stat}')
            print('Resulting table will not generate correctly.')
            for year in object_settings['years']:
                row = f'{year}'
                for month in self.Report.Constants.mo_str_3:
                    row += '|-'
                rows.append(row)
            if 'includeallyears' in object_settings.keys():
                if object_settings['includeallyears'].lower() == 'true':
                    row = 'All'
                    for month in self.Report.Constants.mo_str_3:
                        row += '|-'
                    rows.append(row)
        elif len(data.keys()) > 2:
            print('\nWARNING: Too many datapaths defined.')
            print(f'Need 2 datapaths to compute statistics for {stat}')
            print(f'Resulting table will use the following datapaths: {datakeys[0]}, {datakeys[1]}')
        else:
            for year in object_settings['years']:
                row = f'{year}'
                for month in self.Report.Constants.mo_str_3:
                    if numflagsneeded == 1:
                        row += f'|%%{stat}.{data[datakeys[0]]["flag"]}.MONTH={month.upper()}%%'
                    else:
                        row += f'|%%{stat}.{data[datakeys[0]]["flag"]}.MONTH={month.upper()}.{data[datakeys[1]]["flag"]}.MONTH={month.upper()}%%'
                rows.append(row)
            if 'includeallyears' in object_settings.keys():
                if object_settings['includeallyears'].lower() == 'true':
                    row = 'All'
                    for month in self.Report.Constants.mo_str_3:
                        if numflagsneeded == 1:
                            row += f'|%%{stat}.{data[datakeys[0]]["flag"]}.MONTH={month.upper()}%%'
                        else:
                            row += f'|%%{stat}.{data[datakeys[0]]["flag"]}.MONTH={month.upper()}.{data[datakeys[1]]["flag"]}.MONTH={month.upper()}%%'
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