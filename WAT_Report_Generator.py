'''
Created on 7/15/2021
@author: scott
@organization: Resource Management Associates
@contact: scott@rmanet.com
@note:
'''

import datetime as dt
import math
import os
import sys
import shutil
import matplotlib.pyplot as plt
import numpy as np
import xml.etree.ElementTree as ET

import WAT_DataReader as WDR
import WAT_Functions as WF
import WAT_XML_Utils as XML_Utils

use_depth = False



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

        self.readSimulationInfo(simulationInfoFile)
        self.DefineUnits()
        self.DefinePaths()
        self.DefineMonths()
        if self.reportType == 'single': #Eventually be able to do comparison reports, put that here
            for simulation in self.simulationInfo.keys():
                self.cleanOutputDirs()
                self.initializeXML()
                self.DefineSimulationVars(simulation)
                self.FindRegions()
                self.ReadProfileStationsMetaFile()
                self.ReadTimeSeriesStations()
                self.MakePlots()

    def readSimulationInfo(self, simulationInfoFile):
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

        self.modelAlternatives = {}
        self.simulationInfo = {}
        tree = ET.parse(simulationInfoFile)
        root = tree.getroot()

        self.reportType = root.find('ReportType').text
        self.studyDir = root.find('Study/Directory').text
        self.observedDir = root.find('Study/ObservedData').text

        Simulations = root.find('Simulations')
        for simulation in Simulations:
            simname = simulation.find('Name').text
            self.simulationInfo[simname] = {'BaseName': simulation.find('BaseName').text,
                                            'Directory': simulation.find('Directory').text,
                                            'DSSFile': simulation.find('DSSFile').text,
                                            'StartTime': simulation.find('StartTime').text,
                                            'EndTime': simulation.find('EndTime').text,
                                            'LastComputed': simulation.find('LastComputed').text
                                            }
            self.modelAlternatives[simname] = {}
            for modelAlt in simulation.find('ModelAlternatives'):
                self.modelAlternatives[simname][modelAlt.find('Name').text] = {'Program': modelAlt.find('Program').text,
                                                                               'Fpart': modelAlt.find('FPart').text
                                                                               }

    def cleanOutputDirs(self):
        '''
        cleans the images output directory, so pngs from old reports aren't mistakenly
        added to new reports. Creates directory if it doesn't exist.
        '''

        if not os.path.exists(self.images_path):
            os.makedirs(self.images_path)

        WF.clean_output_dir(self.images_path, '.png')

    def initializeXML(self):
        '''
        copys a new version of the template XML file, initiates the XML class and writes the cover page
        :return: sets class variables
                    self.XML
        '''
        print('MAKING XML')
        xml_template = 'report_template.xml'
        new_xml = 'USBRAutomatedReportOutput.xml' #required name for file
        shutil.copyfile(xml_template, new_xml)

        report_date = dt.datetime.now().strftime('%Y-%m-%d %H:%M')

        self.XML = XML_Utils.XMLReport(new_xml)
        self.XML.writeCover(report_date)

    def MakePlots(self):
        '''
        Loops over numbered regions and reads data to generate plots, then writes them to the XML file
        '''

        for order in self.regInfo.keys(): #make sure we read these in the right order.
            _foundModelAlt = self.getModelAltForRegion(order)
            if _foundModelAlt:
                self.profile_stats = {}
                self.station_results = []
                print('CURRENTLY WORKING ON', self.region)
                print('')

                if self.plugin.lower() == "ressim":
                    self.mr = WDR.ResSim_Results(self.simulationDir, self.alternativeFpart, self.StartTime, self.EndTime)
                elif self.plugin.lower() == 'cequalw2':
                    self.mr = WDR.W2_Results(self.simulationDir, self.region, self.modelAltName, self.StartTime,
                                             self.EndTime)
                else:
                    print('ERROR: Model {0} not accepted'.format(self.plugin))
                    print('Skipping Region..')
                    continue

                self.MakeTemperatureProfilePlot()

                #Time series
                for station, data in self.TimeSeriesStations.items():
                    region = data['region'].lower()
                    metric = data['metric'].lower()
                    longname = data['longname']
                    if self.region.lower() == region.lower():
                        ot, ov = WDR.ReadObservedTimeSeriesData(data, self.observedDir, metric, self.StartTime,
                                                                self.EndTime)
                        mt, mv = self.mr.readModelTimeseriesData(data, metric)
                        self.MakeTimeSeriesPlots(ot, ov, mt, mv, station, metric)
                        self.ComputeTimeSeriesStats(ot, ov, mt, mv, station, metric, longname)

                self.WriteXML()

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

        self.modelAltName = self.regInfo[order]['modelAltName'].strip()
        self.region = self.regInfo[order]['region'].strip()
        self.plugin = self.regInfo[order]['plugin'].strip()
        _model_alt_set = False
        for model_alt in self.modelAlternatives[self.simulationName].keys():
            cur_MA_name = model_alt
            cur_MA_plugin = self.modelAlternatives[self.simulationName][model_alt]['Program']
            print(cur_MA_name, cur_MA_plugin)
            if cur_MA_name == self.modelAltName and cur_MA_plugin == self.plugin:
                self.alternativeFpart = self.modelAlternatives[self.simulationName][model_alt]['Fpart']
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

        self.mo_str_3 = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    def DefineUnits(self):
        '''
        creates dictionary with units for vars for labels
        :return: set class variable
                    self.units
        '''

        WQ_metrics = ['temperature', 'DO', 'DO_sat']
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

    def DefineSimulationVars(self, simulation):
        '''
        defines simulation variables for the currently selected variable based off of values in the simulation
        info file
        :param simulation: name of simulation
        :return: set class variables
                    self.simulationName
                    self.BaseName
                    self.simulationDir
                    self.DSSFile
                    self.StartTime
                    self.EndTime
                    self.LastComputed
        '''

        self.simulationName = simulation
        self.baseSimulationName = self.simulationInfo[simulation]['BaseName']
        self.simulationDir = self.simulationInfo[simulation]['Directory']
        self.DSSFile = self.simulationInfo[simulation]['DSSFile']
        self.LastComputed = self.simulationInfo[simulation]['LastComputed']

        StartTimeStr = self.simulationInfo[simulation]['StartTime']
        if '24:00' in StartTimeStr:
            tstrtmp = (StartTimeStr).replace('24:00', '23:00')
            self.StartTime = dt.datetime.strptime(tstrtmp, '%d %B %Y, %H:%M')
            self.StartTime += dt.timedelta(hours=1)
        else:
            self.StartTime = dt.datetime.strptime(StartTimeStr, '%d %B %Y, %H:%M')

        EndTimeStr = self.simulationInfo[simulation]['EndTime']
        if '24:00' in EndTimeStr:
            tstrtmp = (EndTimeStr).replace('24:00', '23:00')
            self.EndTime = dt.datetime.strptime(tstrtmp, '%d %B %Y, %H:%M')
            self.EndTime += dt.timedelta(hours=1)
        else:
            self.EndTime = dt.datetime.strptime(EndTimeStr, '%d %B %Y, %H:%M')


    def FindRegions(self):
        '''
        finds the correct RPTRGN file and gets the region info
        :return: sets class variable
                    self.regInfo
        '''
        self.regInfo = WDR.find_rptrgn(self.baseSimulationName, self.studyDir)
        print('REGINFO', self.regInfo)

    def MakeTemperatureProfilePlot(self):
        '''
        Finds the corresponding obs data file and creates plots for temperature profiles
        :return:
        '''

        for reservoir, meta in self.ProfileStations.items():
            print('Currently making Temperature plots for Reservoir:', reservoir)

            if self.region.lower() == meta['region'].lower():
                print(self.region, meta['region'])
                n_profiles_per_page = 9
                syear = self.StartTime.year
                eyear = self.EndTime.year
                if self.EndTime.month == 1:
                    eyear -= 1
                print(syear, eyear)

                profile_results = []
                for yr in range(syear, eyear + 1):
                    print('WORKING ON {0}'.format(yr))
                    observed_data_file_name = " ".join(['Profile', reservoir, meta['metric'].lower()]) + '_{0}.txt'.format(yr)
                    observed_data_file_name = os.path.join(self.observedDir, observed_data_file_name)
                    if os.path.exists(observed_data_file_name):
                        obs_times, obs_values, obs_depths = WDR.readObservedProfiles(observed_data_file_name)
                        nof_profiles = len(obs_times)
                        n_pages = math.ceil(nof_profiles / n_profiles_per_page)
                        model_values, model_elev,  model_depths = self.mr.getWaterTemperatureProfiles(obs_times,
                                                                                                     resname=reservoir)
                        if not use_depth:
                            obs_elev = WF.convert_obs_depths(obs_depths, model_elev)

                        # break profile indices into page groups
                        prof_indices = list(range(nof_profiles))
                        n = n_profiles_per_page
                        page_indices = [prof_indices[i * n:(i + 1) * n] for i in range((len(prof_indices) + n - 1) // n)]

                        fig_names = []
                        stats = []
                        for page_i, pgi in enumerate(page_indices):

                            subplot_rows, subplot_cols = WF.get_subplot_config(len(pgi))
                            n_nrow_active = np.ceil(len(pgi) / subplot_cols)
                            fig = plt.figure(figsize=(7, 1 + 3 * n_nrow_active))
                            # fig = plt.figure(figsize=(7, 10))

                            for i, j in enumerate(pgi):
                                pax = fig.add_subplot(subplot_rows, subplot_cols, i + 1)

                                # observation
                                dt_profile = obs_times[j]
                                obs_val = obs_values[j]
                                model_val = model_values[j]

                                model_msk = np.where(~np.isnan(model_val))
                                model_val = model_val[model_msk]


                                if use_depth:
                                    obs_levels = obs_depths[j]
                                    model_levels = model_depths[j][model_msk]
                                else:
                                    obs_levels = obs_elev[j]
                                    model_levels = model_elev[j][model_msk]


                                lflag, xflag, yflag = self.get_plot_label_masks(i, len(pgi), subplot_cols)
                                self.obs_model_profile_plot(pax, obs_levels, obs_val, model_levels, model_val, 'temperature', dt_profile[0],
                                                       show_legend=lflag, show_xlabel=xflag, show_ylabel=yflag, use_depth=use_depth)

                                # why do stats here? because it may be time consuming to pull observations and model data a second time
                                stats.append(WF.series_stats(model_levels, model_val, obs_levels, obs_val,
                                                          start_limit=None, end_limit=None, time_series=False))

                            plt.tight_layout()
                            # plt.show()
                            fig_names.append(reservoir + '_' + "temperature" + '_' + str(yr) + '_%02i.png' % page_i)
                            plt.savefig(os.path.join(self.images_path, fig_names[-1]), dpi=600)
                            plt.close('all')
                        profile_results.append([reservoir, yr, fig_names, stats])
                    else:
                        print('No %s profile observations for year %i' % (reservoir, yr))
                self.profile_stats[reservoir] = profile_results

    def MakeTimeSeriesPlots(self, obs_dates, obs_vals, comp_dates, comp_vals, station, metric):
        '''
        takes in observed and modeled data and makes time series plots for a station. saves png to the images directory
        :param obs_dates: array of observed data dates
        :param obs_vals: array of corresponding observed data values
        :param comp_dates: array of computed dates
        :param comp_vals: array of corresponding computed values
        :param station: station name
        :param metric: metric of data
        :return:
        '''

        if len(comp_dates) > 0:
            print('Making Timeseries Plot for {0}'.format(station))
            print('')


            fig = plt.figure(figsize=(12, 6))
            ax = fig.add_subplot(111)
            ax.plot(obs_dates, obs_vals, label='Observed')
            ax.plot(comp_dates, comp_vals, '-.', label='Computed')

            stime = comp_dates[0]
            etime = comp_dates[-1]

            ax.set_xlim([stime, etime])
            omsk = (obs_dates >= stime) & (obs_dates <= etime)
            cmsk = (comp_dates >= stime) & (comp_dates <= etime)
            ax.legend(loc='upper right')
            if metric.lower() == 'flow':
                units_str = self.units[metric]
                ax.set_ylim([0., max([np.nanmax(obs_vals[omsk]), np.nanmax(comp_vals[cmsk])])])
            elif metric.lower() == 'elevation':
                ylabel_str = 'Surface Elevation'
                units_str = 'ft'
                ax.set_ylim([min([np.nanmin(obs_vals[omsk]), np.nanmin(comp_vals[cmsk])]),
                             max([np.nanmax(obs_vals[omsk]), np.nanmax(comp_vals[cmsk])])])
            elif metric.lower() == 'temperature':
                ylabel_str = 'Water Temperature'
                units_str = self.units[metric.lower()]
                ax.set_ylim([0., max([np.nanmax(obs_vals[omsk]), np.nanmax(comp_vals[cmsk])])])
            elif metric.lower() == 'do':
                ylabel_str = 'Dissolved Oxygen'
                units_str = self.units[metric]
                ax.set_ylim([0., max([np.nanmax(obs_vals[omsk]), np.nanmax(comp_vals[cmsk])])])
            elif metric.lower() == 'do_sat':
                ylabel_str = 'Dissolved Oxygen Saturation'
                units_str = self.units[metric]
                ax.set_ylim([min([np.nanmin(obs_vals[omsk]), np.nanmin(comp_vals[cmsk])]),
                             max([np.nanmax(obs_vals[omsk]), np.nanmax(comp_vals[cmsk])])])
            ax.set_ylabel('{0} ({1})'.format(ylabel_str, units_str))
            ax.set_title('{0}, Simulation: {1}, {2}'.format(station, os.path.basename(self.simulationDir), self.simulationName))
            plt.grid()
            fig_name = metric.capitalize() + '_' + station.replace(' ', '_') + '.png'
            plt.savefig(os.path.join(self.images_path, fig_name), bbox_inches='tight', dpi=600)
            plt.close()

    def ComputeTimeSeriesStats(self, obs_dates, obs_vals, comp_dates, comp_vals, station, metric, longname):
        '''
        takes in observed and modeled data and calculates statistics, then adds them to self.station_results
        :param obs_dates: array of observed data dates
        :param obs_vals: array of corresponding observed data values
        :param comp_dates: array of computed dates
        :param comp_vals: array of corresponding computed values
        :param station: station name
        :param metric: metric of data
        :param longname: full name of data
        :return:
        '''

        fig_name = metric.capitalize() + '_' + station.replace(' ', '_') + '.png'

        if len(comp_dates) > 0:
            print('Making Timeseries Stats for {0}'.format(station))
            print('')
            stats = {}
            stats_months = {}
            n_years = obs_dates[-1].year - obs_dates[0].year + 1
            for yr in range(obs_dates[0].year, obs_dates[-1].year + 1):
                stdate = dt.datetime(yr, 1, 1)
                enddate = dt.datetime(yr + 1, 1, 1)
                # at some point should add metric to key, which is used as label
                # print(station, metric, yr)
                stats_yr = WF.series_stats(comp_dates, comp_vals, obs_dates, obs_vals,
                                        start_limit=stdate, end_limit=enddate, time_series=True)
                if stats_yr is not None:
                    stats[str(yr)] = stats_yr
                    stats_months[str(yr) + ' Obs. Mean'] = {}
                    stats_months[str(yr) + ' Comp. Mean'] = {}
                    for mo in range(1, 13):
                        enddate = dt.datetime(yr + 1, 1, 1) if mo == 12 else dt.datetime(yr, mo + 1, 1)
                        mo_stat = WF.series_stats(comp_dates, comp_vals, obs_dates, obs_vals,
                                               start_limit=dt.datetime(yr, mo, 1), end_limit=enddate,
                                               means_only=True, time_series=True)
                        if mo_stat is None:
                            stats_months[str(yr) + ' Obs. Mean'][self.mo_str_3[mo - 1]] = None
                            stats_months[str(yr) + ' Comp. Mean'][self.mo_str_3[mo - 1]] = None
                        else:
                            stats_months[str(yr) + ' Obs. Mean'][self.mo_str_3[mo - 1]] = mo_stat['Obs. Mean']
                            stats_months[str(yr) + ' Comp. Mean'][self.mo_str_3[mo - 1]] = mo_stat['Comp. Mean']

        self.station_results.append([station, metric, longname, fig_name, stats, stats_months])

    def ReadProfileStationsMetaFile(self):
        '''
        reads the profile_stations.txt file and gets relavent info
        :return: set class value
                    self.ProfileStations
        '''

        stations = {}
        with open(self.ProfileStations_meta_file) as osf:
            for line in osf:
                line = line.strip()
                if line.startswith('start station'):
                    name = ''
                    metric = ''
                    region = ''
                elif line.startswith('name'):
                    name = line.split('=')[1]
                elif line.startswith('metric'):
                    metric = line.split('=')[1]
                elif line.startswith('region'):
                    region = line.split('=')[1]
                elif line.startswith('end station'):
                    stations[name] = {'metric': metric, 'region':region}

        self.ProfileStations = stations

    def ReadTimeSeriesStations(self):
        '''
        Read in stations file and return dictionary containing station information
        station files follow format of "Start Station", then information seperated by '=', and end with "End station".
        ex:
        start station
        name=Shasta Outflow
        longname=Shasta Reservoir Outflow Temperature
        metric=Temperature
        easting=660908
        northing=14788945
        w2_path=/W2:TWO_77.OPT/SEG 77 WITHDRAWAL/TEMP-TWO//1HOUR/$$FPART$$/
        dss_path=/SHASTA DAM-CORR/FLOW_WT/TEMP_G2//1DAY/TEMPERATURE (F)/
        dss_fn=Historic-UpperSac.dss
        region=Shasta
        end station
        :param obs_ts_meta_file: full path to stations file
        :return: set class object
                    self.TimeSeriesStations
        '''
        self.TimeSeriesStations = {}
        with open(self.TimeSeries_Stations_meta_file) as osf:
            for line in osf:
                line = line.strip()
                if line.startswith('start station'):
                    name = ''
                    longname=''
                    metric = ''
                    easting = 0
                    northing = 0
                    dss_path = None
                    dss_fn = None
                    region = ''
                    w2_path = ''
                elif line.startswith('name'):
                    name = line.split('=')[1]
                elif line.startswith('longname'):
                    longname = line.split('=')[1]
                elif line.startswith('metric'):
                    metric = line.split('=')[1]
                elif line.startswith('easting'):
                    easting = float(line.split('=')[1])
                elif line.startswith('northing'):
                    northing = float(line.split('=')[1])
                elif line.startswith('dss_path'):
                    dss_path = line.split('=')[1]
                elif line.startswith('dss_fn'):
                    dss_fn = line.split('=')[1]
                elif line.startswith('region'):
                    region = line.split('=')[1]
                elif line.startswith('w2_path'):
                    w2_path = line.split('=')[1]
                elif line.startswith('end station'):
                    self.TimeSeriesStations[name] = {'easting': easting, 'northing': northing, 'metric': metric,
                                                     'region':region, 'dss_path': dss_path, 'longname': longname,
                                                     'dss_fn': dss_fn, 'w2_path': w2_path}

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

    def obs_model_profile_plot(self, ax, obs_elev, obs_value, model_elev, model_value, metric, dt_profile,
                               show_legend=False, show_xlabel=False, show_ylabel=False, use_depth=False):
        '''
        Creates subplots for temperature profile plots
        :param ax: current axis in plot
        :param obs_elev: observed elev/depth values
        :param obs_value: observed data values
        :param model_elev: modeled elev/depth values
        :param model_value: modeled data values
        :param metric: current metric
        :param dt_profile: datetime profiles
        :param show_legend: boolean to show legend
        :param show_xlabel: boolean to show xlabel
        :param show_ylabel: boolean to show ylabel
        :param use_depth: determines if plots use depth or elevation
        '''

        ax.plot(obs_value, obs_elev, zorder=4, label='Observed')
        ax.grid(zorder=0)
        if use_depth:
            ax.invert_yaxis()

        # modeled
        # ax.plot(model_value, model_elev, '-.', marker='o', zorder=4, label='Modeled')
        ax.plot(model_value, model_elev, zorder=4, label='Modeled')
        if metric.lower() == 'temperature':
            ax.set_xlim([0, 30])
            xlab = r'Temperature ($^\circ$C)'
        elif metric.lower() == 'do':
            ax.set_xlim([0, 14])
            xlab = 'Dissolved O$^2$ (mg/L)'
        elif metric.lower() == 'do_sat':
            ax.set_xlim([0, 130])
            xlab = 'O$^2$ saturation (%)'

        if show_legend:
            plt.legend(loc='lower right')
        if show_xlabel:
            ax.set_xlabel(xlab)
        if show_ylabel:
            if use_depth:
                ax.set_ylabel('Depth (ft)')
            else:
                ax.set_ylabel('Elevation (ft)')
        ttl_str = dt.datetime.strftime(dt_profile, '%d %b %Y')
        xbufr = 0.05
        ybufr = 0.05
        xl = ax.get_xlim()
        yl = ax.get_ylim()
        xtext = xl[0] + xbufr * (xl[1] - xl[0])
        ytext = yl[1] - ybufr * (yl[1] - yl[0])
        ax.text(xtext, ytext, ttl_str, ha='left', va='top', size=10, bbox=dict(boxstyle='round', facecolor='w',
                                                                               alpha=0.35), zorder=10)

    def WriteXML(self):
        '''
        writes the XML section for a single region
        '''

        XML_res = ""
        XML_ts = ""
        XML_Groupheader = self.XML.make_Reservoir_Group_header(self.region)

        if len(self.profile_stats) > 0:
            XML_res = self.XML.XML_reservior(self.profile_stats, self.region)
        if len(self.station_results) > 0:
            XML_ts = self.XML.XML_time_series(self.station_results, self.mo_str_3)
        if len(self.station_results) + len(self.profile_stats) > 0:
            self.XML.write_Reservoir(self.plugin, XML_Groupheader, XML_res, XML_ts)
        else:
            print('No Results found for Region', self.region)




if __name__ == '__main__':
    simInfoFile = sys.argv[1]

    ar = MakeAutomatedReport(simInfoFile)

