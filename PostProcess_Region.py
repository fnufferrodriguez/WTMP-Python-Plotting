'''
Created on 6/8/2021
@author: scott
@organization: Resource Management Associates
@contact: scott@rmanet.com
@note:
'''


import os
import datetime as dt
import traceback
import subprocess
from hec.hecmath import DSS

class reportPreprocess(object):

    def __init__(self, runpath, model, region, rtw): #TODO: edit to take in more args

        self.runpath = runpath #TODO: get this
        self.model = model
        self.region = region
        self.rtw = rtw
        self.starttime_str = self.rtw.getStartTimeString() #Forecast Time
        self.endtime_str = self.rtw.getEndTimeString() # End of Forecast time
        print(self.starttime_str, self.endtime_str)

    def PreProcess(self):
        try:
            self.make_output_Folder()
            self.write_Date_File()
            self.convert_DSS_Records()
            self.run_Bat_File()
            return 0

        except:
            print("Critical Error in Script.")
            print(traceback.format_exc())
            return 1

    def run_Bat_File(self):
        bat_file_cmd = ['runPython38.bat', self.watershed_folder, self.sim_folder, self.model, self.alt_name,
                        self.csv_folder, self.region]
        subprocess.call(bat_file_cmd)

    def Get_DSS_Commands(self):
        stations_file = os.path.join('stations', '{0}_stations.txt'.format(self.region)) #TODO: does this need model?
        dss_records = self.read_stations_file(stations_file)
        return dss_records

    def read_stations_file(self, stations_file):
        dss_records = {}
        station_name, dss_path, dss_fn, metric = ''
        with open(stations_file, 'r') as sf:
            for line in sf:
                sline = line.strip().lower()
                if sline.startswith('end station'):
                    dss_records[station_name] = {'dss_path': dss_path,
                                                 'dss_fn': dss_fn,
                                                 'metric': metric}
                elif sline.startswith('name'):
                    station_name = sline.split('=')[1]
                elif sline.startswith('metric'):
                    metric = sline.split('=')[1]
                elif sline.startswith('dss_fn'):
                    dss_fn = sline.split('=')[1]
                elif sline.startswith('dss_path'):
                    dss_path = sline.split('=')[1]
        return dss_records



    def read_DSS_Record(self, dssFile, dssrecord):
        try:
            print("Opening record {0} from {1} to {2}".format(dssrecord, self.lookback_str, self.endtime_str))
            vals = dssFile.read(dssrecord, self.lookback_str, self.endtime_str)
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

    def write_DSS_CSV_file(self, DSSRecordinfo, dates, vals):
        dssname = DSSRecordinfo['name']
        metric = DSSRecordinfo['metric']

        file_name = '{0}_{1}.csv'.format(dssname.replace(' ', '_'), metric)
        csv_directory = os.path.join(self.csv_folder)
        with open(os.path.join(csv_directory, file_name), 'w') as out:
            if len(dates) == 0:
                out.write('No Data Found.')
            else:
                for i, date in enumerate(dates):
                    date_frmt = date.strftime('%d %m %Y %H %M')
                    out.write('{0},{1}\n'.format(date_frmt, vals[i]))


    def convert_DSS_Records(self):
        dss_records = self.Get_DSS_Commands()
        for dssrec in dss_records:
            dssFile = DSS.open(dss_records[dssrec]['dss_fn'])
            dss_path = dss_records[dssrec]['dss_path']
            dates, vals = self.read_DSS_Record(dssFile, dss_path)
            self.write_DSS_CSV_file(dss_records[dssrec], dates, vals)
            dssFile.close()

    def make_output_Folder(self):
        self.output_folder = os.path.join('PostProcessing', '{0}-{1}'.format(self.region, self.model)) #TODO: get runpath
        if not os.path.isdir(self.report_folder):
            os.makedirs(self.report_folder)
        self.csv_folder = os.path.join(self.output_folder, 'CSV')

    def write_Date_File(self):
        with open(os.path.join(self.output_folder, 'dates.txt'), 'w') as dates:
            dates.write('starttime={0}\n'.format(self.starttime_str))
            dates.write('endtime={0}\n'.format(self.endtime_str))



def computeAlternative(currentAlternative, computeOptions):
    '''
    Main function called to run the code. Acts as the __main__.
    :param currentAlternative: forecast info from RTS
    :param computeOptions: Compute options from RTS
    :return: 0 and 1 depending on completion of script. 0 for success, 1 for fail

    VARS I NEED TODO
    rem %1 the watershed folder
    rem %2 the simulation folder      <sim dir>
    rem %3 model name ...ie. ResSim   <modelname>
    rem %4 alternative name           <alt name>
    rem %5 obs data folder            <obs dir>
    rem %6 region name ...ie.ShastaRes<reg name>

    '''


    rtw = computeOptions.getRunTimeWindow()
    #TODO: Get passed in region
    global _currentAlt
    _currentAlt = currentAlternative
    rgp = reportPreprocess(rtw) #TODO: pass in other args
    rv = rgp.PreProcess()
    print(rv)
    if rv == 0:
        return 1
    return 0
