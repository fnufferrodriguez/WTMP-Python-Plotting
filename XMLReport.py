'''
Created on 6/8/2021
@author: scott
@organization: Resource Management Associates
@contact: scott@rmanet.com
@note:
Writes the XML file, and reads existing one.
Reads existing XML file to get figure table and group numbers
Work flow includes calling WQ_Plotter multiple times, and then reading an XML file that is opened/closed,
and adding to it
'''

class makeXMLReport(object):

    def __init__(self, XML_fn):
        self.XML_fn = XML_fn
        self.read_XML() #open XML
        self.get_XML_status()
        self.current_subgroup = 0 #everytime we open this, reset this, as we will be writing a new group when its called
        self.column_order = 0 #see above..

        self.mo_str_3 = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    def read_XML(self):
        '''
        read XML file and assign XML lines to object
        :return: class object list of lines
        '''
        XML_read = open(self.XML_fn, 'r')
        self.XMLFile = XML_read.readlines()
        XML_read.close()

    def get_XML_status(self):
        '''
        get current status of XML file. Will read any existing features so we get a current
        number of figures/tables/etc
        :return: sets class counters for figures, model numbers, report groups, report elements, and table numbers.
        '''
        self.get_figure_nums() #read figure numbers
        self.get_model_nums() #read model numbers
        self.get_ReportGroup_nums() #read report groups
        self.get_ReportElement_nums() #read report Elements
        self.get_Table_nums() #read table numbers


    def get_figure_nums(self):
        '''
        read through XML file, find 'FigureNumber=*' flag, and use that to find the last figure number
        :return: set class object self.current_fig_num
        '''
        figures = []
        for line in self.XMLFile:
            if "FigureNumber" in line:
                sline = line.split(' ')
                fig_num = [int(s.split('=')[1].replace('"','')) for s in sline if 'FigureNumber' in s][0]
                figures.append(fig_num)
        if len(figures) > 0:
            self.current_fig_num = max(figures) + 1
        else:
            self.current_fig_num = 1

    def get_model_nums(self):
        '''
        read through XML file, find 'ModelOrder=*' flag, and use that to find the last model number
        :return: set class object self.current_model_num
        '''
        models = []
        for line in self.XMLFile:
            if "ModelOrder" in line:
                sline = line.split(' ')
                model_num = [int(s.split('=')[1].replace('"','')) for s in sline if 'ModelOrder' in s][0]
                models.append(model_num)
        if len(models) > 0:
            self.current_model_num = max(models) + 1
        else:
            self.current_model_num = 0


    def get_ReportGroup_nums(self):
        '''
        read through XML file, find 'ReportGroupOrder=*' flag, and use that to find the last group order number
        :return: set class object self.current_reportgroup_num
        '''
        groups = []
        for line in self.XMLFile:
            if "ReportGroupOrder" in line:
                print(line)
                sline = line.split(' ')
                ReportGroup_num = [int(s.split('=')[1].replace('"','')) for s in sline if 'ReportGroupOrder' in s][0]
                groups.append(ReportGroup_num)
        if len(groups) > 0:
            self.current_reportgroup_num = max(groups) + 1
        else:
            self.current_reportgroup_num = 0

    def get_ReportElement_nums(self):
        '''
        read through XML file, find 'ReportElementOrder=*' flag, and use that to find the last element order number
        :return: set class object self.current_reportelem_num
        '''

        elements = []
        for line in self.XMLFile:
            if "ReportElementOrder" in line:
                sline = line.split(' ')
                ReportElem_num = [int(s.split('=')[1].replace('"','')) for s in sline if 'ReportElementOrder' in s][0]
                elements.append(ReportElem_num)
        if len(elements) > 0:
            self.current_reportelem_num = max(elements) + 1
        else:
            self.current_reportelem_num = 1

    def get_Table_nums(self):
        '''
        read through XML file, find 'TableNumber=*' flag, and use that to find the last table number
        :return: set class object self.current_table_num
        '''
        tables = []
        for line in self.XMLFile:
            if "TableNumber" in line:
                sline = line.split(' ')
                Table_num = [int(s.split('=')[1].replace('"','')) for s in sline if 'TableNumber' in s][0]
                tables.append(Table_num)
        if len(tables) > 0:
            self.current_table_num = max(tables) + 1
        else:
            self.current_table_num = 1

    def writeCover(self, report_date):
        '''
        writes the cover for the report. Replaces the date string with current date
        '''
        cover_keys = {"$$REPORT_DATE$$": report_date}
        with open(self.XML_fn, 'w') as xmlf:
            for line in self.XMLFile:
                for key in cover_keys.keys():
                    line = line.replace(key, cover_keys[key])
                xmlf.write(line)

    def write_Reservoir(self, region, model_name, GroupHeader_Text, Res_Text, TS_Text):
        '''
        Reads through template file and finds the flags to be replaces, and adds in the right value. Then sets up
        entire section for reservoir, adding in subsections for temp profile plots, ts plots and stat tables
        :param region: name of the region
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
                elif sline.startswith("<!--$${0}$$-->".format(region)):
                    '''if order ever changes or is customizable, do it here!'''
                    XML.write(GroupHeader_Text)
                    XML.write(Res_Text)
                    XML.write(TS_Text)
                    XML.write('     </Report_Group>\n')
                else:
                    XML.write(line)


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

    def make_TS_Subgroup_lines(self,  ts_name, ts_fig, subgroupdesc):
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

    def write_Model_Name(self, XML, model_name):
        '''
        Writes model info
        :param XML: XML object to write to
        :param model_name: name of model
        '''
        XML.write('	        <Model ModelOrder="{0}" >{1}</Model>\n'.format(self.current_model_num, model_name))
        XML.write('         <!--$$ModelInfo$$-->\n')

