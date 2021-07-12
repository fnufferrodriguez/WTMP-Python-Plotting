'''
Created on 6/8/2021
@author: scott
@organization: Resource Management Associates
@contact: scott@rmanet.com
@note:
Jython script to be called from WAT. Finds all needed DSS data, and uses hec.hecmath.DSS to read in data and convert
to CSV files so the following python scripts are able to read in data.
'''


import os
import datetime as dt
import traceback
from hec.hecmath import DSS

class reportPreprocess(object):

    def __init__(self, studyFolder, simulationFolder,modelName,
                       alternativeName, obsDataFolder, alternativeFpart,
                       simulationName, startTime, endTime, baseSimulationName,
                       dssFile):

        self.studyFolder = studyFolder
        self.simulationFolder = simulationFolder
        self.modelName = modelName
        self.alternativeName = alternativeName
        self.obsDataFolder = obsDataFolder
        self.alternativeFpart = alternativeFpart
        self.simulationName = simulationName
        self.startTime = startTime
        self.endTime = endTime
        self.baseSimulationName = baseSimulationName
        self.dssFile = dssFile
        print('Processing for modeltype:', self.modelName)

    def PreProcess(self):
        '''Main controlling function in the class
        Find the regions, and convert dss'''
        try:
            self.get_regions() #get a list of 1 to several regions
            # self.make_output_Folder()
            for region in self.region_names:
                print('WORKING ON REGION', region)
                self.region = region #set current region
                self.convert_DSS_Records()
            return 0

        except:
            print("Critical Error in Script.")
            print(traceback.format_exc())
            return 1

    def get_regions(self):
        '''
        Read the rptrgn file to figure out what region is simulated by which run
        :return: list of regions
        '''
        try:
            reg_info = self.find_rptrgn(self.baseSimulationName)
            print('reg_info', reg_info)
            self.region_names = reg_info[self.alternativeName.replace(' ', '_')]['regions']
            '''some entries contain multiple regions, will loop through if the case.'''
        except:
            '''Will return empty list if something goes wrong. Should be a much bigger deal.'''
            self.region_names = [] #TODO: this can be better.
        print('Found Regions', self.region_names)


    def Get_DSS_Commands(self):
        '''
        Reads the time series station text file and organizes relevant information in a dictionary
        Builds a series of time series from water temperature profile depths and WSE for elevations
        Also builds a W2 modeled data version of observed time series records to compare against
        :return: dictionary of DSS information to be read and converted
        '''

        stations_file = os.path.join(self.obsDataFolder, 'TS_stations.txt')
        if not os.path.exists(stations_file):
            print('Stations DSS file {0} does not exist.'.format(stations_file))
            return []
        station_information = self.read_stations_file(stations_file) #read stations file
        dss_records = {}

        for station in station_information.keys():
            region = station_information[station]['region']
            if self.region.lower() == region.lower(): #check if data is in current region, skip if not
                dss_records[station] = {}
                if station_information[station]['dss_path'] != '': #check if dss defined
                    print('self.studyFolder', self.studyFolder)
                    print('station_information', station_information[station]['dss_path'])
                    dss_records[station]['dss_path'] = os.path.join(self.studyFolder, 'shared', station_information[station]['dss_path'])
                    dss_records[station]['dss_fn'] = station_information[station]['dss_fn']
                    dss_records[station]['metric'] = station_information[station]['metric']

            print(station_information[station]['w2_path'])

            #if there is a w2 path defined, and our model is W2, we need to grab the data for obs stations from
            # the results DSS file
            if station_information[station]['w2_path'] != "''" and self.modelName == 'CeQualW2':
                print('MAKING RECORD FOR w2 STATION')
                dss_records[station+'_Fromw2'] = {}
                dss_path = station_information[station]['w2_path']
                dss_path = dss_path.replace('$$FPART$$', self.alternativeFpart)
                dss_records[station+'_Fromw2']['dss_path'] = dss_path

                # build dss path
                simfolderrev = self.simulationFolder.split('\\')
                simfolderrev.reverse()
                print(self.dssFile)
                dss_records[station+'_Fromw2']['dss_fn'] = self.dssFile
                dss_records[station+'_Fromw2']['metric'] = station_information[station]['metric']


        #If the model is W2, we also need to get the temperature profiles from the results DSS. the depth profiles are
        #built in a predictable way, so we can use a template to build the paths. Also grab WSE for elevations later
        if self.modelName == 'CeQualW2': #get temp profiles
            if self.region.lower() == 'shasta':
                dss_path = r'/W2:TSR_$$INDEX$$_SEG77.OPT/TSR SEG 77 DEPTH $$DEPTH$$.00/TEMP-WATER//1HOUR/$$FPART$$/'
                elev_dss_path = r'/W2:TSR_1_SEG77.OPT/TSR SEG 77 DEPTH 0.00/ELEV//1HOUR/{0}/'.format(self.alternativeFpart)
            elif self.region.lower() == 'keswick':
                dss_path = r'/W2:TSR_$$INDEX$$_SEG32.OPT/TSR SEG 32 DEPTH $$DEPTH$$.00/TEMP-WATER//1HOUR/$$FPART$$/'
                elev_dss_path = r'/W2:TSR_1_SEG32.OPT/TSR SEG 32 DEPTH 0.00/ELEV//1HOUR/{0}/'.format(self.alternativeFpart)
            else:
                '''If not in region with a temperature profile, no need to continue. Return back with what we have.'''
                return dss_records

            dss_path = dss_path.replace('$$FPART$$', self.alternativeFpart)
            print('USING DSS PATH {0} for W2 TEMPERATURE PROFILE'.format(dss_path))

            #standard depth levels for W2. If that ever changes, this will need to be changed.
            #maybe read through all records, and find everything by using partial B parts?
            start = 0
            increment = 2
            end = 160
            depths = range(start, end, increment)

            #loop through all depths, fill in DSS paths, read data, convert to CSV
            for i, depth in enumerate(depths):
                d_pth = dss_path.replace('$$DEPTH$$', str(depth))
                d_pth = d_pth.replace('$$INDEX$$', str(i+1))
                TempProfile = self.region.lower() + '_tempprofile_Depth{0}_Idx{1}'.format(depth, i+1)
                dss_records[TempProfile] = {}
                dss_records[TempProfile]['dss_path'] = d_pth
                dss_records[TempProfile]['dss_fn'] = self.dssFile
                dss_records[TempProfile]['metric'] = 'Temperature'

            #Read water surface elevation and convert for elevation vs depth later
            ElevProfile = self.region.lower() + '_WaterSurfaceElev'
            dss_records[ElevProfile] = {}
            dss_records[ElevProfile]['dss_path'] = elev_dss_path
            dss_records[ElevProfile]['dss_fn'] = self.dssFile
            dss_records[ElevProfile]['metric'] = 'Elev'

        return dss_records

    def read_stations_file(self, stations_file):
        '''
        Read Stations file and pull out relevant details. Stations file must be set up correctly.
        :param stations_file: full path to stations file
        :return: dictionary containing Time series details
        '''
        stations = {}
        with open(stations_file) as sf:
            for line in sf:
                line = line.strip()
                if line.startswith('start station'):
                    name = ''
                    metric = ''
                    easting = 0
                    northing = 0
                    dss_path = ''
                    dss_fn = ''
                    region = ''
                    w2_path = ''
                elif line.startswith('name'):
                    name = line.split('=')[1]
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
                    stations[name] = {'easting': easting, 'northing': northing, 'metric': metric,
                                      'dss_path': dss_path, 'region':region, 'w2_path':w2_path, 'dss_fn': dss_fn}
        return stations

    def read_DSS_Record(self, dssFile, dssrecord):
        '''
        Read DSS from java call
        :param dssFile: dss file object
        :param dssrecord: name of dss record
        :return: dates list, values list
        '''

        try:
            print("Opening record {0} from {1} to {2}".format(dssrecord, self.startTime, self.endTime))
            vals = dssFile.read(dssrecord, self.startTime, self.endTime)
            print(vals)
        except:
            #if theres any issues, just write out empty and move on
            print("Error with DSS data, record empty or missing.")
            return [], []
        else:
            print("DSS reading Successful.")

        dates = []
        #loop through returned values and extract data but also convert dates to datetime objects
        for i in range(len(vals.getData().times)):
            #HECTIME uses 2400, datetime uses 0000 the next day... convert as needed
            if vals.getData().getHecTime(i).hour() == 24:
                modtime = vals.getData().getHecTime(i)
                modtime.subtract(60)
                modtime_str = (modtime.dateAndTime(4))
                modtime_dt = dt.datetime.strptime(modtime_str, "%d%b%Y, %H:%M")
                modtime_dt += dt.timedelta(hours=1)
                dates.append(modtime_dt)
            else:
                dates.append(dt.datetime.strptime(vals.getData().getHecTime(i).dateAndTime(4), "%d%b%Y, %H:%M"))

        return dates, vals.getData().values

    def write_DSS_CSV_file(self, station, metric, dates, vals):
        '''
        writes CSV file based off of read DSS info
        :param station: name of the observed station
        :param metric: metric for record (ie temperature, flow, etc)
        :param dates: date values in array or list
        :param vals: data values in array or list
        :return:
        '''

        file_name = '{0}_{1}.csv'.format(station.replace(' ', '_'), metric)
        csv_directory = os.path.join(self.output_folder)
        with open(os.path.join(csv_directory, file_name), 'w') as out:
            if len(dates) == 0: #in the case of no data
                out.write('No Data Found.')
            else:
                for i, date in enumerate(dates):
                    date_frmt = date.strftime('%d%b%Y, %H%M')
                    out.write('{0};{1}\n'.format(date_frmt, vals[i]))
                out.write('END')


    def convert_DSS_Records(self):
        '''
        iterates through DSS data and converts it to CSV files for python to read.
        first checks the observed data for a full region and grabs data from there
        then, if its W2, find the results dss file and export temperature profile
        depth values, and Water surface elevation for the top layer.
        :return:
        '''
        dss_records = self.Get_DSS_Commands() #get dss dictionary for what we need to convert
        # print('dss_records', dss_records) #this ends up being huge. debug only.

        for dssrec in dss_records:
            if len(dss_records[dssrec]) > 0:
                dssFile = DSS.open(os.path.join(self.obsDataFolder, dss_records[dssrec]['dss_fn']))
                dss_path = dss_records[dssrec]['dss_path']
                dates, vals = self.read_DSS_Record(dssFile, dss_path) #read
                self.write_DSS_CSV_file(dssrec, dss_records[dssrec]['metric'], dates, vals) #write
                dssFile.close() #close

    def make_output_Folder(self):
        self.output_folder = os.path.join(self.studyFolder, 'reports', 'CSV')
        print('OUTPUT FOLDER:', self.output_folder)
        if not os.path.isdir(self.output_folder):
            os.makedirs(self.output_folder)

    def find_rptrgn(self, simulation_name):
        '''
           Read the right rptrgn file, and determine what region you are working with.
           RPTRGN files are named after the simulation, and consist of plugin, model alter name, and then region(s)
           :returns: dictionary containing information from file
           '''
        #find the rpt file go up a dir, reports, .rptrgn
        rptrgn_file = os.path.join(self.studyFolder, 'reports', '{0}.rptrgn'.format(simulation_name.replace(' ', '_')))
        print('Looking for rptrgn file at:', rptrgn_file)
        if not os.path.exists(rptrgn_file):
            print('ERROR: no RPTRGN file for simulation:', simulation_name)
            exit()
        reg_info = {}
        with open(rptrgn_file, 'r') as rf:
            for line in rf:
                sline = line.strip().split(',')
                plugin = sline[0].strip()
                model_alt_name = sline[1].strip()
                regions = sline[2:]
                regions = [n.strip() for n in regions]
                reg_info[model_alt_name] = {'plugin': plugin,
                                            'regions': regions}
        return reg_info



'''
Main function called to run the code. Acts as the __main__.

These vars are set by WAT and do not have to be defined
studyFolder
simulationFolder
modelName  (ResSim)
alternativeName
alternativeFpart
simulationName
obsDataFolder
startTime
endTime

EXAMPLE INPUTS
('simulationFolder', u'C:\\WAT\\USBR_FrameworkTest_r3\\runs\\Shasta-Keswick_W2\\2014\\')
('modelName', u'CeQualW2')
('simName', u'Shasta-Keswick W2 14-val2014')
('alternativeFpart', u'Shasta-Kes:W2 Import :CeQualW2-Keswick from Shasta W2')
('alternativeName', u'Keswick from Shasta W2')
('baseSimulationName', u'Shasta-Keswick W2 14')
('dssFile', u'C:\\WAT\\USBR_FrameworkTest_r3\\runs\\Shasta-Keswick_W2\\2014\\Shasta-Keswick_W2_14-val2014.dss')
('obs data', u'C:\\WAT\\USBR_FrameworkTest_r3\\shared')
'''
print('STARTING JYTHON CODE')
print('simulationFolder', simulationFolder)
print('modelName', modelName)
print('simName', simulationName)
print('alternativeFpart', alternativeFpart)
print('alternativeName', alternativeName)
print('baseSimulationName', baseSimulationName)
print('dssFile', dssFile)
print('obs data', obsDataFolder)

rgp = reportPreprocess(studyFolder, simulationFolder,modelName,
                       alternativeName, obsDataFolder, alternativeFpart,
                       simulationName, startTime, endTime, baseSimulationName, dssFile)
rv = rgp.PreProcess()


