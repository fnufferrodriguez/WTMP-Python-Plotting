'''
Created on 7/15/2021
@author: scott
@organization: Resource Management Associates
@contact: scott@rmanet.com
@note:
'''


class XMLReport(object):

    def __init__(self, XML_fn):
        self.XML_fn = XML_fn
        # self.ReadXML()
        self.MakeXML()
        self.PrimeCounters()

#########################################################################################
                            #Main functions#
#########################################################################################

    # def ReadXML(self):
    #     '''
    #     Probably dont need if we dont open every time...
    #     read XML file and assign XML lines to object
    #     :return: class object list of lines
    #     '''
    #     XML_read = open(self.XML_fn, 'r')
    #     self.XMLFile = XML_read.readlines()
    #     XML_read.close()

    def MakeXML(self):
        with open(self.XML_fn, 'w') as XML:
            XML.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            XML.write('<USBR_Automated_Report>\n')


    def PrimeCounters(self):
        self.current_fig_num = 1
        self.current_table_num = 1
        self.current_reportelem_num = 0
        self.current_model_num = 0
        self.current_reportgroup_num = 0

    def writeCover(self, title):
        '''
        writes the cover for the report.
        '''
        with open(self.XML_fn, 'a') as XML:
            XML.write('<Cover_Page><Title>{0}</Title></Cover_Page>\n'.format(title))

    def writeIntroStart(self):
        with open(self.XML_fn, 'a') as XML:
            XML.write('<Report_Group ReportGroupOrder="{0}" ReportGroupName="Introduction">\n'.format(self.current_reportgroup_num))
            XML.write('<Report_Subgroup ReportSubgroupOrder="0">\n')
            XML.write('<Report_Element ReportElementOrder="{0}" Element="Introduction">\n'.format(self.current_reportelem_num))

        self.current_reportgroup_num += 1
        self.current_reportelem_num += 1

    def writeIntroLine(self, Plugin):
        with open(self.XML_fn, 'a') as XML:
            XML.write('<Model ModelOrder="0" >{1}</Model>\n'.format(self.current_model_num, Plugin))
        self.current_model_num += 1

    def writeIntroEnd(self):
        with open(self.XML_fn, 'a') as XML:
            XML.write('</Report_Element>\n')
            XML.write('</Report_Subgroup>\n')
            XML.write('</Report_Group>\n')

    def writeChapterStart(self, ChapterName):
        self.current_reportsubgroup_num = 0
        with open(self.XML_fn, 'a') as XML:
            XML.write('<Report_Group ReportGroupOrder="{0}" ReportGroupName="{1}">\n'.format(self.current_reportgroup_num, ChapterName))
        self.current_reportgroup_num += 1

    def writeChapterEnd(self):
        with open(self.XML_fn, 'a') as XML:
            XML.write('</Report_Group>\n')

    def writeReportEnd(self):
        with open(self.XML_fn, 'a') as XML:
            XML.write('</USBR_Automated_Report>\n')

    def writeSectionHeader(self, section_header):
        with open(self.XML_fn, 'a') as XML:
            XML.write('<Report_Subgroup ReportSubgroupOrder="{0}" ReportSubgroupDescription="{1}">\n'.format(self.current_reportsubgroup_num, section_header))
        self.current_reportsubgroup_num += 1

    def writeSectionHeaderEnd(self):
        with open(self.XML_fn, 'a') as XML:
            XML.write('</Report_Subgroup>\n')

    def writeTimeSeriesPlot(self, figname, figdesc):
        with open(self.XML_fn, 'a') as XML:
            XML.write('<Report_Element ReportElementOrder="{0}" Element="Control_Point_Plots">\n'.format(self.current_reportelem_num))
            XML.write('<Output_Temp_Flow Location="{0}">\n'.format(figdesc))
            XML.write('<Output_Image FigureNumber="{0}" FigureDescription="{1}">{2}</Output_Image>\n'.format(self.current_fig_num ,figdesc, figname))
            XML.write('</Output_Temp_Flow>\n')
            XML.write('</Report_Element>\n')

        self.current_reportelem_num += 1
        self.current_fig_num += 1

    def writeProfilePlotStart(self, reservoir):
        with open(self.XML_fn, 'a') as XML:
            XML.write('<Report_Element ReportElementOrder="{0}" Element="Reservoir_Profile">\n'.format(self.current_reportelem_num))
            XML.write('<Reservoir_Profiles Reservoir="{0}">\n'.format(reservoir))

        self.current_reportelem_num += 1

    def writeProfilePlotFigure(self, figname, figdesc):
        with open(self.XML_fn, 'a') as XML:
            XML.write('<Profile_Image FigureNumber="{0}" FigureDescription="{1}">{2}</Profile_Image>\n'.format(self.current_fig_num, figdesc, figname))
        self.current_fig_num += 1

    def writeProfilePlotEnd(self):
        with open(self.XML_fn, 'a') as XML:
            XML.write('</Reservoir_Profiles>\n')
            XML.write('</Report_Element>\n')


    def writeTableStart(self,desc,type):
        with open(self.XML_fn, 'a') as XML:
            XML.write('<Report_Element ReportElementOrder="{0}" Element="Control_Point_Tables">\n'.format(self.current_reportelem_num))
            XML.write('<Output_Temp_Flow Location="{0}">\n'.format(desc))
            XML.write('<Output_Table TableNumber="{0}" TableDescription="{1}" TableType="{2}">\n'.format(self.current_table_num, desc, type))

        self.current_reportelem_num += 1
        self.current_table_num += 1
        self.column_order = 0

    def writeTableColumn(self, header, rows):
        with open(self.XML_fn, 'a') as XML:
            XML.write('<Column Column_Order="{0}" Column_Name="{1}">\n'.format(self.column_order, header))
            for i, row in enumerate(rows):
                s_row = row.split('|')
                rowname = s_row[0]
                rowval = s_row[1]
                XML.write('<Row Row_Order="{0}" Row_name="{1}">{2}</Row>\n'.format(i, rowname, rowval))
            XML.write('</Column>\n')
        self.column_order += 1

    def writeTableEnd(self):
        with open(self.XML_fn, 'a') as XML:
            XML.write('</Output_Table>\n')
            XML.write('</Output_Temp_Flow>\n')
            XML.write('</Report_Element>\n')





    def write_Reservoir(self, model_name, GroupHeader_Text, Res_Text, TS_Text):
        '''
        Reads through template file and finds the flags to be replaces, and adds in the right value. Then sets up
        entire section for reservoir, adding in subsections for temp profile plots, ts plots and stat tables
        :param model_name: name of model
        :param GroupHeader_Text: XML header for reservoir
        :param Res_Text: Water Temperature profile XML lines
        :param TS_Text: XML text for time series plots and stats tables
        '''
        with open(self.XML_fn, 'w') as XML:
            for line in self.XMLFile:
                sline = line.strip()
                if sline.startswith('<!--$$ModelInfo$$-->'):
                    self.write_Model_Name(XML, model_name) #writes model info
                elif sline.startswith("<!--$$REGION$$-->"):
                    '''if order ever changes or is customizable, do it here!'''
                    XML.write(GroupHeader_Text)
                    XML.write(Res_Text)
                    XML.write(TS_Text)
                    XML.write('     </Report_Group>\n')
                    XML.write('     <!--$$REGION$$-->\n')
                else:
                    XML.write(line)

        self.ReadXML() #re-read the file to get the current text

    def write_Model_Name(self, XML, model_name):
        '''
        Writes model info
        :param XML: XML object to write to
        :param model_name: name of model
        '''
        XML.write('	        <Model ModelOrder="{0}" >{1}</Model>\n'.format(self.current_model_num, model_name))
        XML.write('         <!--$$ModelInfo$$-->\n')
        self.current_model_num += 1

#########################################################################################
                         #Reservoir functions#
#########################################################################################

    def XML_reservior(self, profile_stats, region_name):
        # only writing one reservoir XLM section here
        XML = ''

        #res name, year, list of pngs, stats dictionary
        for subdomain_name, figure_sets in profile_stats.items():
            if len(figure_sets) > 0:
                for ps in figure_sets:
                    reservoir, yr, fig_names, stats = ps
                    # fig_names = [os.path.join('..','Images', n) for n in fig_names]
                    subgroup_desc = self.get_reservoir_description(region_name)
                    XML += self.make_ReservoirSubgroup_lines(reservoir,fig_names, subgroup_desc, yr)

        return XML

    def get_reservoir_description(self, region):
        return "{0} Reservoir Temperature Profiles Near Dam".format(region.capitalize())

    def make_ReservoirSubgroup_lines(self, res_name, res_figs, subgroupdesc, yr):
        '''
        Writes the XML lines for a reservoir subgroup chunk.
        :param res_name: Reservoir name
        :param res_figs: list of Water temperature profile figures
        :param subgroupdesc: description of subgroup region
        :param yr: year
        :return: XML lines to be added to XML report
        '''

        res_keys = {"$$ORDER$$": self.current_reportelem_num,
                    "$$RESERVOIR_NAME$$": res_name,
                    "$$RESERVOIR_FIGS$$": res_figs,
                    "$$SUBGROUPORDER$$": self.current_subgroup,
                    "$$SUBGROUPDESC$$": subgroupdesc}

        res_lines = ['        <Report_Subgroup ReportSubgroupOrder="$$SUBGROUPORDER$$" ReportSubgroupDescription="$$SUBGROUPDESC$$">',
                     '           <Report_Element ReportElementOrder="$$ORDER$$" Element="Reservoir_Profile">',
                     '              <Reservoir_Profiles Reservoir="$$RESERVOIR_NAME$$">',
                     '                  $$RESERVOIR_FIGS$$',
                     '              </Reservoir_Profiles>',
                     '           </Report_Element>',
                     '        </Report_Subgroup>']

        out = ''
        for i, line in enumerate(res_lines):
            for key in res_keys.keys():
                if key == "$$RESERVOIR_FIGS$$" and key in line:
                    outfigs = ''
                    for fi, fig in enumerate(res_figs):
                        figdesc = res_name + ' %i: %i of %i' % (yr, fi+1, len(res_figs))
                        outfigs += self.make_ReservoirFig_lines(fig, figdesc)
                        if fi+1 != len(res_figs):
                            print(fi, len(res_figs))
                            outfigs += '\n'
                    line = outfigs

                elif key == '$$SUBGROUPORDER$$' and key in line:
                    line = line.replace(key, str(res_keys[key]))
                    self.current_subgroup += 1

                elif key == '$$ORDER$$' and key in line:
                    line = line.replace(key, str(res_keys[key]))
                    self.current_reportelem_num += 1

                else:
                    line = line.replace(key, str(res_keys[key]))

            out += line+'\n'

        return out

    def make_Reservoir_Group_header(self, group_name):
        '''
        Builds the text for the reservoir group header. Iterates through predetermined line(s) and replaces specific
        flags with inputs.
        :param group_name: region name
        :return: XML lines to be added to XML report
        '''
        self.current_subgroup = 1 #reset as it will always start at 0 if this is being called..

        res_group_keys = {"$$GROUPORDER$$": self.current_reportgroup_num,
                          "$$GROUPNAME$$": group_name}

        res_lines = ['    <Report_Group ReportGroupOrder="$$GROUPORDER$$" ReportGroupName="$$GROUPNAME$$">']

        out = ''
        for i, line in enumerate(res_lines):
            for key in res_group_keys.keys():
                if key == '$$GROUPORDER$$' and key in line:
                    line = line.replace(key, str(res_group_keys[key])) #when the default was 0..
                    self.current_reportgroup_num += 1
                else:
                    line = line.replace(key, str(res_group_keys[key]))
            out += line+'\n'
        return out

    def make_ReservoirFig_lines(self, fig_filename, fig_description):
        '''
        Writes XML lines to include a figure in the reservoir subgroup for Profile plots
        :param fig_filename: figure file name
        :param fig_description: figure description
        :return: XML lines to be added to XML report
        '''

        fig_keys = {"$$FIG_NUM$$": self.current_fig_num,
                    "$$FIG_DESCRIPTION$$": fig_description,
                    "$$FIG_FILENAME$$": fig_filename}

        fig_line = ['                    <Profile_Image FigureNumber="$$FIG_NUM$$" FigureDescription="$$FIG_DESCRIPTION$$">$$FIG_FILENAME$$</Profile_Image>']

        out = ''
        for i, line in enumerate(fig_line):
            for key in fig_keys.keys():
                if key == '$$FIG_NUM$$' and key in line:
                    line = line.replace(key, str(fig_keys[key])) #plus one when the starting value was 0..
                    self.current_fig_num += 1
                else:
                    line = line.replace(key, str(fig_keys[key]))
            out += line
        return out

#########################################################################################
                            #Timeseries functions#
#########################################################################################

    def XML_time_series(self, ts_results, mo_str_3):
        stats_labels = {
            'Mean Bias': r'Mean Bias (&lt;sup&gt;O&lt;/sup&gt;C)',
            'MAE': r'MAE (&lt;sup&gt;O&lt;/sup&gt;C)',
            'RMSE': r'RMSE (&lt;sup&gt;O&lt;/sup&gt;C)',
            'NSE': r'Nash-Sutcliffe (NSE)',
            'COUNT': r'COUNT',
        }
        stats_ordered = ['Mean Bias', 'MAE', 'RMSE', 'NSE', 'COUNT']
        self.mo_str_3 = mo_str_3
        XML = ""
        for ts in ts_results:
            station, metric, desc, fig_name, stats, stats_mo = ts
            # fig_name = os.path.join('..','Images', fig_name)
            # subgroup_desc = get_ts_description(station)
            XML += self.make_TS_Subgroup_lines(station, fig_name, desc)
            XML += self.make_TS_Tables_lines(station, stats, stats_mo, stats_ordered, stats_labels)
            XML += '        </Report_Subgroup>\n'

        return XML

    def make_TS_Subgroup_lines(self, ts_name, ts_fig, subgroupdesc):
        '''
        Writes XML lines to include a subgroup in the reservoir subgroup for Time series plots
        :param ts_name: name of time series figure
        :param ts_fig: name of the time series plot
        :param subgroupdesc: description of the subgroup location
        :return: XML lines to be added to XML report
        '''

        TS_keys = {"$$SUBGROUPORDER$$": self.current_subgroup,
                   "$$SUBGROUPDESC$$": subgroupdesc,
                   "$$ORDER$$": self.current_reportelem_num,
                   "$$LOCATION$$": ts_name,
                   "$$TS_FIG$$": ts_fig}

        ts_lines = ['        <Report_Subgroup ReportSubgroupOrder="$$SUBGROUPORDER$$" ReportSubgroupDescription="$$SUBGROUPDESC$$">',
                    '           <Report_Element ReportElementOrder="$$ORDER$$" Element="Control_Point_Plots">',
                    '              <Output_Temp_Flow Location="$$LOCATION$$">',
                    '                 $$TS_FIG$$',
                    '              </Output_Temp_Flow>',
                    '           </Report_Element>']

        out = ''
        for i, line in enumerate(ts_lines):
            for key in TS_keys.keys():
                if key == "$$TS_FIG$$" and key in line:
                    outfig = self.make_TSFig_lines(ts_fig, ts_name)
                    line = outfig
                elif key == '$$SUBGROUPORDER$$' and key in line:
                    line = line.replace(key, str(TS_keys[key]))
                    self.current_subgroup += 1
                elif key == '$$ORDER$$' and key in line:
                    line = line.replace(key, str(TS_keys[key]))
                    self.current_reportelem_num += 1
                else:
                    line = line.replace(key, str(TS_keys[key]))
            out += line+'\n'
        return out

    def make_TSFig_lines(self, figfilename, fig_description):
        '''
        Creates XML lines to add a time series figure to a reservoir
        :param figfilename: file name of plot image to be added to report
        :param fig_description: description of file location
        :return: XML lines to be added to XML report
        '''

        fig_keys = {"$$FIG_NUM$$": self.current_fig_num,
                    "$$FIG_DESCRIPTION$$": fig_description,
                    "$$FIG_FILENAME$$": figfilename}

        fig_line = ['                    <Output_Image FigureNumber="$$FIG_NUM$$" FigureDescription="$$FIG_DESCRIPTION$$">$$FIG_FILENAME$$</Output_Image>']

        out = ''
        for i, line in enumerate(fig_line):
            for key in fig_keys.keys():
                if key == '$$FIG_NUM$$' and key in line:
                    line = line.replace(key, str(fig_keys[key])) #plus one when the starting value was 0..
                    self.current_fig_num += 1
                else:
                    line = line.replace(key, str(fig_keys[key]))
            out += line
        return out

    def make_TS_Tables_lines(self, station, stats, stats_mo, stats_ordered, stats_labels):
        '''
        create lines for statistics table for time series tables
        :param station: station location name
        :param stats: dictionary of potential stats to be included in the table and values
        :param stats_mo: monthly statistics for monthly stats table
        :param stats_ordered: ordered list of stats to be included
        :param stats_labels: formatted labels for each statistic
        :return: XML lines to be added to XML report
        '''

        ts_table_keys = {"$$ORDER$$": self.current_reportelem_num,
                         "$$LOCATION$$": station}

        table_lines = ['            <Report_Element ReportElementOrder="$$ORDER$$" Element="Control_Point_Tables">',
                       '               <Output_Temp_Flow Location="$$LOCATION$$">']

        out = ''
        for i, line in enumerate(table_lines):
            for key in ts_table_keys.keys():
                if key == '$$ORDER$$' and key in line:
                    line = line.replace(key, str(ts_table_keys[key]))#plus one when the starting value was 0..
                    self.current_reportelem_num += 1
                else:
                    line = line.replace(key, str(ts_table_keys[key]))
            out += line+'\n'


        out += self.make_TS_error_stats_table(station, stats, stats_ordered, stats_labels)
        out += self.make_TS_mean_monthly_stats_table(station, stats_mo)

        out += '                 </Output_Temp_Flow>\n'
        out += '            </Report_Element>\n'


        return out

    def make_TS_error_stats_table(self, station, stats, stats_ordered, stats_labels):
        '''
        creates table lines for error stats table
        :param station: station name for description
        :param stats: dictionary with stat values
        :param stats_ordered: ordered list of stats to be included
        :param stats_labels: formatted labels for each statistic
        :return: XML lines to be added to XML report
        '''

        error_table_keys = {"$$TABLENUM$$": self.current_table_num,
                            "$$TABLE_DESC$$": "{0} Error Statistics".format(station)}

        error_table_lines = ['                  <Output_Table TableNumber="$$TABLENUM$$" TableDescription="$$TABLE_DESC$$" TableType="Statistics">']

        out = ''
        self.column_order = 0
        for line in error_table_lines:
            for key in error_table_keys:
                if key == '$$TABLENUM$$' and key in line:
                    line = line.replace(key, str(error_table_keys[key])) #plus one when the starting value was 0..
                    self.current_table_num += 1
                else:
                    line = line.replace(key, str(error_table_keys[key]))
            out += line+'\n'

        for j, colname in enumerate(stats.keys()):
            column_keys = {'$$COL_ORDER$$': j,
                           '$$COL_NAME$$': colname}
            column_line = '                        <Column Column_Order="$$COL_ORDER$$" Column_Name="$$COL_NAME$$">'
            for key in column_keys:
                if key == '$$COL_ORDER$$':
                    column_line = column_line.replace(key, str(column_keys[key]))
                    self.column_order += 1
                else:
                    column_line = column_line.replace(key, str(column_keys[key]))
            out += column_line + '\n'
            for i, st in enumerate(stats_ordered):
                row = '                            <Row Row_Order="$$ORDER$$" Row_name="$$STATS_LABEL$$">$$STAT_NUM$$</Row>'


                row = row.replace("$$ORDER$$", str(i))
                if st == 'COUNT':
                    row = row.replace("$$STAT_NUM$$", '%i' % stats[colname][st])
                else:
                    row = row.replace("$$STAT_NUM$$", '%2f' % stats[colname][st])
                row = row.replace("$$STATS_LABEL$$", stats_labels[st])
                out += row + '\n'
            out += '                        </Column>\n'
        out += '                  </Output_Table>\n'

        return out

    def make_TS_mean_monthly_stats_table(self, station, stats_mo):
        '''
        creates table lines for monthly stats table
        :param station: station name for description
        :param stats_mo: dictionary containing monthly stat values
        :return: XML lines to be added to XML report
        '''
        month_table_keys = {"$$TABLENUM$$": self.current_table_num,
                            "$$TABLE_DESC$$": "{0} Mean Monthly Statistics".format(station)}
        month_table_lines = ['                  <Output_Table TableNumber="$$TABLENUM$$" TableDescription="$$TABLE_DESC$$" TableType="Month">']

        out = ''
        self.column_order = 0
        for line in month_table_lines:
            for key in month_table_keys:
                if key == '$$TABLENUM$$' and key in line:
                    line = line.replace(key, str(month_table_keys[key]))
                    self.current_table_num += 1
                else:
                    line = line.replace(key, str(month_table_keys[key]))
            out += line+'\n'

        col_names = list(stats_mo.keys())

        for j, colname in enumerate(sorted(col_names)):
            column_keys = {'$$COL_ORDER$$': j,
                           '$$COL_NAME$$': colname}
            stats_col = stats_mo[colname]
            column_line = '                        <Column Column_Order="$$COL_ORDER$$" Column_Name="$$COL_NAME$$">'
            for key in column_keys:
                if key == '$$COL_ORDER$$':
                    column_line = column_line.replace(key, str(column_keys[key]))
                    # self.column_order += 1
                elif key == '$$COL_NAME$$':
                    column_line = column_line.replace(key, str(column_keys[key]))
            out += column_line + '\n'
            for mo in range(1, 13):
                row = '                            <Row Row_Order="$$ORDER$$" Row_name="$$STATS_LABEL$$">$$STAT_NUM$$</Row>'
                row = row.replace("$$ORDER$$", str(mo))
                row = row.replace("$$STATS_LABEL$$", self.mo_str_3[mo - 1])
                if stats_col[self.mo_str_3[mo - 1]] is None:
                    row = row.replace("$$STAT_NUM$$", 'nan')
                else:
                    row = row.replace("$$STAT_NUM$$", '%2f' % stats_col[self.mo_str_3[mo - 1]])
                out += row + '\n'
            out += '                        </Column>\n'
        out += '                    </Output_Table>\n'

        return out