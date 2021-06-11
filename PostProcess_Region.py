'''
Created on 6/8/2021
@author: scott
@organization: Resource Management Associates
@contact: scott@rmanet.com
@note:
'''


import os, sys
import datetime as dt
import traceback
import subprocess
from hec.hecmath import DSS

class reportPreprocess(object):

    def __init__(self, studyFolder, simulationFolder,modelName,
                       alternativeName, obsDataFolder, alternativeFpart,
                       simulationName, startTime, endTime):

        self.studyFolder = studyFolder
        self.simulationFolder = simulationFolder
        self.modelName = modelName
        self.alternativeName = alternativeName
        self.obsDataFolder = obsDataFolder
        self.alternativeFpart = alternativeFpart
        self.simulationName = simulationName
        self.startTime = startTime
        self.endTime = endTime

    def PreProcess(self):
        try:
            self.get_regions()
            self.make_output_Folder()
            for region in self.region_names:
                print('WORKING ON REGION', region)
                self.region = region
                self.convert_DSS_Records()
            return 0

        except:
            print("Critical Error in Script.")
            print(traceback.format_exc())
            return 1

    def get_regions(self):
        try:
            reg_info = self.find_rptrgn(self.simulationName)
            print('reg_info', reg_info)
            self.region_names = reg_info[self.alternativeName.replace(' ', '_')]['regions']
        except:
            self.region_names = [] #TODO: this can be better


    def Get_DSS_Commands(self):
        stations_file = os.path.join(self.obsDataFolder, 'TS_stations.txt')
        if not os.path.exists(stations_file):
            print('Stations DSS file {0} does not exist.'.format(stations_file))
            return []
        station_information = self.read_stations_file(stations_file)
        dss_records = {}
        for station in station_information.keys():
            region = station_information[station]['region']
            if self.region.lower() == region.lower():
                dss_records[station] = {}
                #only grab observed, but we wont do anything with this for a little..
                if station_information[station]['dss_path'] != '': 
                    print('self.studyFolder', self.studyFolder)
                    print('station_information', station_information[station]['dss_path'])
                    dss_records[station]['dss_path'] = os.path.join(self.studyFolder, 'shared', station_information[station]['dss_path'])
                    dss_records[station]['dss_fn'] = [station_information[station]['dss_fn']]
                    dss_records[station]['metric'] = station_information[station]['metric']
            print(station_information[station]['w2_path'])
            if station_information[station]['w2_path'] != "''":
                dss_records[station+'_Fromw2'] = {}
                correct_fpart = self.alternativeFpart
                correct_fpart = correct_fpart.replace(self.simulationName[:10]+'_', self.simulationName[:10]+':')
                correct_fpart = correct_fpart.replace('_'+self.modelName, ':'+self.modelName)
                correct_fpart = correct_fpart.split(self.modelName)
                if self.region.lower() == 'shasta':
                    dss_orig = '-SHASTA FROM DSS'
                elif self.region.lower() == 'keswick':
                    dss_orig = '-KESWICK FROM SHASTA W2'
                correct_fpart = correct_fpart[0] + self.modelName + dss_orig
                dss_path = station_information[station]['w2_path']
                dss_path = dss_path.replace('$$FPART$$', correct_fpart)
                dss_records[station+'_Fromw2']['dss_path'] = dss_path
                # build dss path
                simfolderrev = self.simulationFolder.split('\\')
                simfolderrev.reverse()
                for item in simfolderrev: #TODO: fix this, this is sloppy
                    try:
                        int(item)
                        analysis_per = item
                    except ValueError:
                        continue
                dss_fn = os.path.join(self.simulationFolder, self.simulationName.replace(' ', '_') + '-val{0}.dss'.format(analysis_per))
                print(dss_fn)
                dss_records[station+'_Fromw2']['dss_fn'] = dss_fn
                dss_records[station+'_Fromw2']['metric'] = station_information[station]['metric']



        if self.modelName == 'CeQualW2': #get temp profiles
            if self.region.lower() == 'shasta':
                dss_path = r'/W2:TSR_$$INDEX$$_SEG77.OPT/TSR SEG 77 DEPTH $$DEPTH$$.00/TEMP-WATER//1HOUR/$$FPART$$/'
            elif self.region.lower() == 'keswick':
                dss_path = r'/W2:TSR_14_$$INDEX$$_SEG32.OPT/TSR SEG 32 DEPTH $$DEPTH$$.00/TEMP-WATER//1HOUR/$$FPART$$/' #TODO: read these from a text file
            else:
                return dss_records


            correct_fpart = self.alternativeFpart
            correct_fpart = correct_fpart.replace(self.simulationName[:10]+'_', self.simulationName[:10]+':')
            correct_fpart = correct_fpart.replace('_'+self.modelName, ':'+self.modelName)
            correct_fpart = correct_fpart.replace('_', ' ')
            dss_path = dss_path.replace('$$FPART$$', correct_fpart)
            print('USING DSS PATH {0} for W2 TEMPERATURE PROFILE'.format(dss_path))
            start = 0
            increment = 2
            end = 160
            depths = range(start, end, increment)
            for i, depth in enumerate(depths):
                d_pth = dss_path.replace('$$DEPTH$$', str(depth))
                d_pth = d_pth.replace('$$INDEX$$', str(i+1))

                TempProfile = self.region.lower() + '_tempprofile_Depth{0}_Idx{1}'.format(depth, i+1)
                dss_records[TempProfile] = {}

                dss_records[TempProfile]['dss_path'] = d_pth

                # build dss path
                simfolderrev = self.simulationFolder.split('\\')
                simfolderrev.reverse()
                for item in simfolderrev: #TODO: fix this, this is sloppy
                    try:
                        int(item)
                        analysis_per = item
                    except ValueError:
                        continue
                dss_fn = os.path.join(self.simulationFolder, self.simulationName.replace(' ', '_') + '-val{0}.dss'.format(analysis_per))
                print(dss_fn)
                dss_records[TempProfile]['dss_fn'] = dss_fn
                dss_records[TempProfile]['metric'] = 'Temperature'


        return dss_records

    def read_stations_file(self, stations_file):
        stations = {}
        obs_dss_file = None
        with open(stations_file) as sf:
            for line in sf:
                line = line.strip()
                if line.startswith('OBS_FILE'):
                    obs_dss_file = os.path.join(os.getcwd(), line.split('=')[1])
                elif line.startswith('start station'):
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
        try:
            print("Opening record {0} from {1} to {2}".format(dssrecord, self.startTime, self.endTime))
            vals = dssFile.read(dssrecord, self.startTime, self.endTime)
            print(vals)
        except:
            print("Error with DSS data, record empty or missing.")
            return [], []
        else:
            print("DSS reading Successful.")

        dates = []
        for i in range(len(vals.getData().times)):
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


        file_name = '{0}_{1}.csv'.format(station.replace(' ', '_'), metric)
        csv_directory = os.path.join(self.output_folder)
        with open(os.path.join(csv_directory, file_name), 'w') as out:
            if len(dates) == 0:
                out.write('No Data Found.')
            else:
                for i, date in enumerate(dates):
                    date_frmt = date.strftime('%d%b%Y, %H%M')
                    # date_frmt = date.strftime('%d %m %Y %H %M')

                    out.write('{0};{1}\n'.format(date_frmt, vals[i]))
                out.write('END')


    def convert_DSS_Records(self):
        dss_records = self.Get_DSS_Commands()
        print('dss_records', dss_records)
        
        for dssrec in dss_records:
            if len(dss_records[dssrec]) > 0:
                print(os.path.join(self.obsDataFolder, dss_records[dssrec]['dss_fn']))
                dssFile = DSS.open(os.path.join(self.obsDataFolder, dss_records[dssrec]['dss_fn']))
                dss_path = dss_records[dssrec]['dss_path']
                dates, vals = self.read_DSS_Record(dssFile, dss_path)
                self.write_DSS_CSV_file(dssrec, dss_records[dssrec]['metric'], dates, vals)
                dssFile.close()

    def make_output_Folder(self):
        self.output_folder = os.path.join(self.studyFolder, 'reports', 'CSV')
        print('OUTPUT FOLDER:', self.output_folder)
        if not os.path.isdir(self.output_folder):
            os.makedirs(self.output_folder)

    def write_Date_File(self):
        with open(os.path.join(self.output_folder, 'dates.txt'), 'w') as dates:
            dates.write('starttime={0}\n'.format(self.starttime_str))
            dates.write('endtime={0}\n'.format(self.endtime_str))

    def find_rptrgn(self, simulation_name):
        #find the rpt file go up a dir, reports, .rptrgn
        rptrgn_file = os.path.join(self.studyFolder, 'reports', '{0}.rptrgn'.format(simulation_name.replace(' ', '_')))
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
'''
print('STARTING JYTHON CODE')
print('simulationFolder', simulationFolder)
print('modelName', modelName)
print('simName', simulationName)
print('alternativeFpart', alternativeFpart)
print('alternativeName', alternativeName)

rgp = reportPreprocess(studyFolder, simulationFolder,modelName,
                       alternativeName, obsDataFolder, alternativeFpart,
                       simulationName, startTime, endTime) #TODO: pass in other args
rv = rgp.PreProcess()


