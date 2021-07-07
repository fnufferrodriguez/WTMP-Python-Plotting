'''
Created on 6/8/2021
@author: scott
@organization: Resource Management Associates
@contact: scott@rmanet.com
@note:
'''

class makeXMLReport(object):

    def __init__(self, XML_fn):
        self.XML_fn = XML_fn
        self.read_XML()
        self.get_figure_nums()
        self.get_model_Num()
        self.get_ReportGroup_nums()
        self.get_ReportElement_nums()
        self.get_Table_nums()
        self.current_subgroup = 0
        self.column_order = 0

        self.mo_str_3 = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    def read_XML(self):
        XML_read = open(self.XML_fn, 'r')
        self.XMLFile = XML_read.readlines()
        XML_read.close()

    def get_figure_nums(self):
        self.current_fig_num = 0
        for line in self.XMLFile:
            if "FigureNumber" in line:
                sline = line.split(' ')
                fig_num = [int(s.split('=')[1].replace('"','')) for s in sline if 'FigureNumber' in s][0]
                if fig_num > self.current_fig_num:
                    self.current_fig_num = fig_num

    def get_model_Num(self):
        self.current_model_num = 0
        for line in self.XMLFile:
            if "ModelOrder" in line:
                sline = line.split(' ')
                model_num = [int(s.split('=')[1].replace('"','')) for s in sline if 'ModelOrder' in s][0]
                if model_num > self.current_model_num:
                    self.current_model_num = model_num

    def get_ReportGroup_nums(self):
        self.current_reportgroup_num = 0
        for line in self.XMLFile:
            if "ReportGroupOrder" in line:
                sline = line.split(' ')
                ReportGroup_num = [int(s.split('=')[1].replace('"','')) for s in sline if 'ReportGroupOrder' in s][0]
                if ReportGroup_num > self.current_reportgroup_num:
                    self.current_reportgroup_num = ReportGroup_num

    def get_ReportElement_nums(self):
        self.current_reportelem_num = 1
        for line in self.XMLFile:
            if "ReportElementOrder" in line:
                sline = line.split(' ')
                ReportElem_num = [int(s.split('=')[1].replace('"','')) for s in sline if 'ReportElementOrder' in s][0]
                if ReportElem_num > self.current_reportelem_num:
                    self.current_reportelem_num = ReportElem_num

    def get_Table_nums(self):
        self.current_table_num = 0
        for line in self.XMLFile:
            if "TableNumber" in line:
                sline = line.split(' ')
                Table_num = [int(s.split('=')[1].replace('"','')) for s in sline if 'TableNumber' in s][0]
                if Table_num >= self.current_table_num:
                    self.current_table_num = Table_num


    def writeCover(self, report_date):
        cover_keys = {"$$REPORT_DATE$$": report_date}
        with open(self.XML_fn, 'w') as xmlf:
            for line in self.XMLFile:
                for key in cover_keys.keys():
                    line = line.replace(key, cover_keys[key])
                xmlf.write(line)

    def write_Reservoir(self, region,model_name, Group_Text, Res_Text, TS_Text):
        with open(self.XML_fn, 'w') as XML:
            for line in self.XMLFile:
                sline = line.strip()
                if sline.startswith('<!--$$ModelInfo$$-->'):
                    self.write_Model_Name(XML, model_name)
                elif sline.startswith("<!--$${0}$$-->".format(region)):
                    XML.write(Group_Text)
                    XML.write(Res_Text)
                    XML.write(TS_Text)
                    XML.write('     </Report_Group>\n')
                else:
                    XML.write(line)


    def make_Reservoir_Group_header(self, group_name):
        self.current_subgroup = 0 #reset
        res_group_keys = {"$$GROUPORDER$$": self.current_reportgroup_num,
                          "$$GROUPNAME$$": group_name}
        res_lines = ['    <Report_Group ReportGroupOrder="$$GROUPORDER$$" ReportGroupName="$$GROUPNAME$$">']
        out = ''
        for i, line in enumerate(res_lines):
            for key in res_group_keys.keys():
                if key == '$$GROUPORDER$$' and key in line:
                    line = line.replace(key, str(res_group_keys[key]+1))
                    self.current_reportgroup_num += 1
                line = line.replace(key, str(res_group_keys[key]))
            out += line+'\n'
        return out

    def make_ReservoirSubgroup_lines(self, res_name, res_figs, subgroupdesc, yr):
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
                        outfigs += self.make_ReservoirFig_lines(figdesc, fig)
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

    def make_ReservoirFig_lines(self, fig_description, figfilename):
        fig_keys = {"$$FIG_NUM$$": self.current_fig_num,
                    "$$FIG_DESCRIPTION$$": fig_description,
                    "$$FIG_FILENAME$$": figfilename}
        fig_line = ['                    <Profile_Image FigureNumber="$$FIG_NUM$$" FigureDescription="$$FIG_DESCRIPTION$$">$$FIG_FILENAME$$</Profile_Image>']
        out = ''
        for i, line in enumerate(fig_line):
            for key in fig_keys.keys():
                if key == '$$FIG_NUM$$' and key in line:
                    line = line.replace(key, str(fig_keys[key]+1))
                    self.current_fig_num += 1
                line = line.replace(key, str(fig_keys[key]))
            out += line
        return out

    def make_TS_Subgroup_lines(self,  ts_name, ts_fig, subgroupdesc):
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
                    outfig = self.make_TSFig_lines(ts_name, ts_fig)
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

    def make_TSFig_lines(self, fig_description, figfilename):
        fig_keys = {"$$FIG_NUM$$": self.current_fig_num,
                    "$$FIG_DESCRIPTION$$": fig_description,
                    "$$FIG_FILENAME$$": figfilename}
        fig_line = ['                    <Output_Image FigureNumber="$$FIG_NUM$$" FigureDescription="$$FIG_DESCRIPTION$$">$$FIG_FILENAME$$</Output_Image>']
        out = ''
        for i, line in enumerate(fig_line):
            for key in fig_keys.keys():
                if key == '$$FIG_NUM$$' and key in line:
                    line = line.replace(key, str(fig_keys[key]+1))
                    self.current_fig_num += 1
                line = line.replace(key, str(fig_keys[key]))
            out += line
        return out

    def make_TS_Tables_lines(self, station, stats, stats_mo, stats_ordered, stats_labels):
        ts_table_keys = {"$$ORDER$$": self.current_reportelem_num,
                         "$$LOCATION$$": station}
        table_lines = ['            <Report_Element ReportElementOrder="$$ORDER$$" Element="Control_Point_Tables">',
                       '               <Output_Temp_Flow Location="$$LOCATION$$">',]
        out = ''
        for i, line in enumerate(table_lines):
            for key in ts_table_keys.keys():
                if key == '$$ORDER$$' and key in line:
                    line = line.replace(key, str(ts_table_keys[key]))
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
        error_table_keys = {"$$TABLENUM$$": self.current_table_num,
                            "$$TABLE_DESC$$": "{0} Error Statistics".format(station)}
        error_table_lines = ['                  <Output_Table TableNumber="$$TABLENUM$$" TableDescription="$$TABLE_DESC$$" TableType="Statistics">']

        out = ''
        self.column_order = 0
        for line in error_table_lines:
            for key in error_table_keys:
                if key == '$$TABLENUM$$' and key in line:
                    line = line.replace(key, str(error_table_keys[key]+1))
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
        month_table_keys = {"$$TABLENUM$$": self.current_table_num,
                            "$$TABLE_DESC$$": "{0} Mean Monthly Statistics".format(station)}
        month_table_lines = ['                  <Output_Table TableNumber="$$TABLENUM$$" TableDescription="$$TABLE_DESC$$" TableType="Month">']

        out = ''
        self.column_order = 0
        for line in month_table_lines:
            for key in month_table_keys:
                if key == '$$TABLENUM$$' and key in line:
                    line = line.replace(key, str(month_table_keys[key]+1))
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

    def write_Model_Name(self,XML, model_name):
        XML.write('	        <Model ModelOrder="{0}" >{1}</Model>\n'.format(self.current_model_num, model_name))
        XML.write('         <!--$$ModelInfo$$-->\n')

