"""
Created on 7/1/2020

@author: Stephen Andrews, Ben Saenz, Scott Burdick
@organization: Resource Management Associates
@contact: steve@rmanet.com
@note:
Script to organize and plot data, then procedurally build XML file for Jasper
"""
__updated__ = '11-21-2019 13:14'

import datetime as dt
import math
import os
import sys
import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from scipy.constants import convert_temperature
from sklearn.metrics import mean_absolute_error
import XMLReport
from collections import Counter


sat_data_do = [14.60, 14.19, 13.81, 13.44, 13.09, 12.75, 12.43, 12.12, 11.83, 11.55, 11.27, 11.01, 10.76, 10.52, 10.29,
               10.07, 9.85, 9.65, 9.45, 9.26, 9.07, 8.90, 8.72, 8.56, 8.40, 8.24, 8.09, 7.95, 7.81, 7.67, 7.54, 7.41,
               7.28, 7.16, 7.05, 6.93, 6.82, 6.71, 6.61, 6.51, 6.41, 6.31, 6.22, 6.13, 6.04, 5.95]
sat_data_temp = [0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20., 21.,
                 22., 23., 24., 25., 26., 27., 28., 29., 30., 31., 32., 33., 34., 35., 36., 37., 38., 39., 40., 41.,
                 42., 43., 44., 45.]
f_interp = interpolate.interp1d(sat_data_temp, sat_data_do,
                                fill_value=(sat_data_do[0], sat_data_do[-1]), bounds_error=False)



def read_obs_ts_meta_file(obs_ts_meta_file):
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
    :return: dictionary object with station information
    '''
    stations = {}
    with open(obs_ts_meta_file) as osf:
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
                stations[name] = {'easting': easting, 'northing': northing, 'metric': metric,
                                  'region':region, 'dss_path': dss_path, 'longname': longname,
                                  'dss_fn': dss_fn, 'w2_path': w2_path}
    return stations


def read_obs_profile_meta_file(obs_profile_meta_file):
    '''
    reads the profile_stations.txt file and gets relavent info
    :param obs_profile_meta_file:
    :return: dictionary of stations
    '''
    stations = {}
    with open(obs_profile_meta_file) as osf:
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
    return stations

def observed_ts_txt_file(ts_data_path, station_name, metric):
    '''
    Build file name and read data
    :param ts_data_path: path to where observed data is
    :param station_name: name of station to be read
    :param metric: metric of data
    :return: arrays of times and values of observed data
    '''
    dss_util_fname = os.path.join(ts_data_path, '{0} {1}.txt'.format(metric, station_name))
    return observed_ts_txt_file_read(dss_util_fname)

def observed_ts_txt_file_read(file_name):
    '''
    reads observed text files from PostProcess_Region.py
    :param file_name: full path to the text file
    :return: array of times and values
    '''
    dss_util_fname = file_name
    t = []
    v = []
    print(file_name)
    with open(dss_util_fname) as f:
        for line in f:
            try:
                if line.startswith('END'):
                    break
                elif line.startswith('No Data Found.'):
                    print('No Data found for {0}'.format(file_name))
                    return [], []
                else:
                    sline = line.split(';')
                    try:
                        if '2400' in sline[0]:
                            tstrtmp = (sline[0]).replace('2400', '2300')
                            dt_tmp = dt.datetime.strptime(tstrtmp, '%d%b%Y, %H%M')
                            dt_tmp += dt.timedelta(hours=1)
                        else:
                            dt_tmp = dt.datetime.strptime(sline[0], '%d%b%Y, %H%M')
                    except ValueError:
                        dt_tmp = dt.datetime.strptime(sline[0], '%Y%m%d, %H%M')
                    t.append(dt_tmp)
                    v.append(float(sline[1]))
            except:
                print('Error in File. Skipping line:', line)
    time_difference = [ti - t[i] for i, ti in enumerate(t[1:])]
    td_counter = Counter(time_difference) #get the time deltas
    #if the amount of time intervals is a single interval, then return. Else, fix data to be the same interval
    if len(td_counter) in [0, 1]:
        return np.array(t), np.array(v)
    else:
        most_common = max(td_counter.values())
        td_keys = list(td_counter.keys())
        td_values = list(td_counter.values())
        pos = td_values.index(most_common)
        mc_interval = td_keys[pos] #get most common time interval, this is probably what we want
        new_t = []
        new_v = []
        current_t = t[0] #no option but to assume this is the start..
        end_t = t[-1] #no option again...
        while current_t < end_t:
            new_t.append(current_t)
            if current_t in t:
                ti = t.index(current_t)
                new_v.append(v[ti])
            else:
                new_v.append(np.nan)
            current_t += mc_interval
        return np.array(new_t), np.array(new_v)

def clean_missing(indata):
    '''
    removes data with -901. flags
    :param indata: array of data to be cleaned
    :return: cleaned data array
    '''
    indata[indata == -901.] = np.nan
    return indata


def clean_computed(indata):
    '''
    removes data with -9999 flags
    :param indata: array of data to be cleaned
    :return: cleaned data array
    '''
    indata[indata == -9999.] = np.nan
    return indata

def do_saturation(temp, diss_ox):
    '''
    calulates dissolved oxygen saturation. uses a series of pre computed DO values interpolated
    :param temp: temperature value
    :param diss_ox: dissolved oxygen
    :return: dissolved oxygen value
    '''
    do_sat = f_interp(temp)
    return diss_ox / do_sat * 100.

def calc_computed_dosat(vtemp, vdo):
    '''
    calculates the computed dissolved saturated oxygen
    :param vtemp: temperature values
    :param vdo: values for dissovled oxy
    :return: DOSat values
    '''
    v = np.zeros_like(vtemp)
    for j in range(len(v)):
        if np.isnan(vtemp[j]) or np.isnan(vdo[j]):
            v[j] = np.nan
        else:
            v[j] = do_saturation(vtemp[j], vdo[j])
    return v

def calc_observed_dosat(ttemp, vtemp, vdo):
    '''
     calc dissolved saturated oxygen for observed data
    :param ttemp: times for data
    :param vtemp: temperature values
    :param vdo: values for dissovled oxy
    :return: time and DOSat values
    '''
    v = np.zeros_like(vtemp)
    for j in range(len(v)):
        if np.isnan(vtemp[j]) or np.isnan(vdo[j]):
            v[j] = np.nan
        else:
            v[j] = do_saturation(vtemp[j], vdo[j])
    return ttemp, v


def read_observed_ts_data(obs_ts_data_path, station_name, metric):
    '''
    reads in observed data time series data
    :param obs_ts_data_path: full path to observed data text file
    :param station_name: name of station for data
    :param metric: data metric type
    :return: arrays of time and values
    '''
    if metric.lower() == 'do_sat':
        tt, vt = observed_ts_txt_file(obs_ts_data_path, station_name, 'Temperature')
        vt = clean_missing(vt)
        tdo, vdo = observed_ts_txt_file(obs_ts_data_path, station_name, 'DO')
        vdo = clean_missing(vdo)
        t, v = calc_observed_dosat(tt, vt, vdo)
    else:
        primary_csv_name = os.path.join(study_dir, 'reports', 'CSV', '{0}_{1}.csv'.format(station_name.replace(' ', '_'),metric))
        if os.path.exists(primary_csv_name):
            t, v = observed_ts_txt_file_read(primary_csv_name)
            print('Primary CSV grabbed..')
        else:
            print('Primary CSV from DSS failed. Grabbing backup..')
            backup_txt_name = os.path.join(obs_ts_data_path, '{0} {1}.txt'.format(metric, station_name))
            t, v = observed_ts_txt_file_read(backup_txt_name)
        v = clean_missing(v)
        if metric.lower() == 'temperature':
            if np.any(v > 45):
                v = convert_temperature(v, 'F', 'C')
    return t, v

def find_rptrgn(simulation_name, studyfolder):
    '''
       Read the right rptrgn file, and determine what region you are working with.
       RPTRGN files are named after the simulation, and consist of plugin, model alter name, and then region(s)
       :param simulation_name: name of simulation to find file
       :param studyfolder: full path to study folder
       :returns: dictionary containing information from file
    '''
    #find the rpt file go up a dir, reports, .rptrgn
    rptrgn_file = os.path.join(studyfolder, 'reports', '{0}.rptrgn'.format(simulation_name.replace(' ', '_')))
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

def read_observed(observed_data_filename):
    '''
    reads in observed data files and returns values for Temperature Profiles
    TODO: change to not just read 10k lines.. make smarter
    :param observed_data_filename: file name
    :return: returns values, depths and times
    '''
    f = open(observed_data_filename)
    f.readline()  # header
    max_lines = 10000
    t = []
    wt = []
    d = []
    t_profile = []
    wt_profile = []
    d_profile = []
    hold_dt = dt.datetime(1933, 10, 15)
    for j in range(max_lines):
        line = f.readline()
        if not line:
            break
        sline = line.split(',')
        dt_str = sline[0]
        dt_tmp = dt.datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')

        if (dt_tmp.year != hold_dt.year or dt_tmp.month != hold_dt.month or dt_tmp.day != hold_dt.day) and j != 0:
            # new profile
            if len(t_profile) != 0 and len(wt_profile) != 0 and len(d_profile) != 0:
                t.append(np.array(t_profile))
                wt.append(np.array(wt_profile))
                d.append(np.array(d_profile))
            t_profile = []
            wt_profile = []
            d_profile = []
        else:
            t_profile.append(dt_tmp)
            wt_profile.append(float(sline[1]))
            d_profile.append(float(sline[2]))
        hold_dt = dt_tmp

    return t, wt, d

def nse(simulations, evaluation):
    """Nash-Sutcliffe Efficiency (NSE) as per `Nash and Sutcliffe, 1970
    <https://doi.org/10.1016/0022-1694(70)90255-6>`_.

    :Calculation Details:
        .. math::
           E_{\\text{NSE}} = 1 - \\frac{\\sum_{i=1}^{N}[e_{i}-s_{i}]^2}
           {\\sum_{i=1}^{N}[e_{i}-\\mu(e)]^2}

        where *N* is the length of the *simulations* and *evaluation*
        periods, *e* is the *evaluation* series, *s* is (one of) the
        *simulations* series, and *μ* is the arithmetic mean.

        source: https://pypi.org/project/hydroeval

    """
    nse_ = 1 - (
            np.sum((evaluation - simulations) ** 2, axis=0, dtype=np.float64)
            / np.sum((evaluation - np.mean(evaluation)) ** 2, dtype=np.float64)
    )

    return nse_


def series_stats(t_comp, v_comp, t_obs, v_obs, start_limit=None, end_limit=None, time_series=False,
                 means_only=False, v_obs_min=5):
    '''
    Takes in obs and modeled data and calculates statistics based on match
    :param t_comp: times for computed
    :param v_comp: values for computed
    :param t_obs: times for observed data
    :param v_obs: values for observed data
    :param start_limit: start time to do stats. Allows for subset of data to be calc
    :param end_limit: end time to do stats
    :param time_series: flag to get times from time arrays
    :param means_only: flag to only grab means
    :param v_obs_min: min number of observed values
    :return: stats Mean Bias, MAE, RMSE, NSE, COUNT
    '''
    # will this fail if t_comp is not ascending?
    stime = t_comp[0] if start_limit is None else start_limit
    etime = t_comp[-1] if end_limit is None else end_limit

    if not time_series and stime > etime:
        tmp = etime
        etime = stime
        stime = tmp

    msk = (t_comp >= stime) & (t_comp <= etime)
    t_comp = t_comp[msk]
    v_comp = v_comp[msk]
    msk = (t_obs >= stime) & (t_obs <= etime) & np.isfinite(v_obs)
    t_obs = t_obs[msk]
    v_obs = v_obs[msk]

    mean_diff = 0.
    sq_error = 0.
    count = 0
    if len(v_obs) > v_obs_min and len(v_comp) > v_obs_min:

        if time_series:
            tstamp_comp = np.array([t_comp[j].timestamp() for j in range(len(t_comp))])
            tstamp_obs = np.array([t_obs[j].timestamp() for j in range(len(t_obs))])
        else:
            tstamp_comp = t_comp
            tstamp_obs = t_obs
        f_computed = interpolate.interp1d(tstamp_comp, v_comp, bounds_error=False, fill_value=np.nan)

        comp_matched_to_obs = f_computed(tstamp_obs)
        obs_comp_match_mask = np.isfinite(comp_matched_to_obs)

        v_obs_stats = v_obs[obs_comp_match_mask]
        v_comp_stats = comp_matched_to_obs[obs_comp_match_mask]

        if means_only:
            return {'Obs. Mean': np.mean(v_obs_stats),
                    'Comp. Mean': np.mean(v_comp_stats)}
        else:

            diff = v_comp_stats - v_obs_stats
            count = len(v_comp_stats)
            mean_diff = np.sum(diff) / count
            rmse = np.sqrt(np.sum(diff ** 2) / count)

            nash = nse(v_comp_stats, v_obs_stats)
            try:
                mae = mean_absolute_error(v_obs_stats, v_comp_stats)
            except:
                print('Failed to calculate MAE!')
                mae = np.nan
            return {'Mean Bias': mean_diff, 'MAE': mae, 'RMSE': rmse, 'NSE': nash, 'COUNT': count}
            # return mean_diff,mae,rmse,nash,len(v_obs)
    else:
        return None



class ModelResults(object):
    """
    Subclasses please conform to units
    """
    WQ_metrics = ['temperature', 'DO', 'DO_sat']
    WQ_units = ['C', 'mg/L', '%']
    metrics = ['flow', ] + WQ_metrics
    metric_units = ['m3/s', ] + WQ_units
    units = dict(zip(metrics, metric_units))

    # datetime objects that must be created by subclasses
    stime = None
    etime = None

    def get_profile(self, time_in, WQ_metric, xy=None, name=None):
        """
        Subclass must pass back elevation (m) and corresponding metric values
        :param time_in: python datetime of desired profile
        :param WQ_metric: metric
        :param xy: [x,y] coordinates, optional if name is used
        :param name: identifier for reservoir or profile location, optional if xy used
        :return: (elevation, values) paired numpy arrays
        """
        raise NotImplementedError('subclass must define get_profile()')
        elevation = np.array([0, ])
        values = np.array([0, ])
        return elevation, values

    def get_time_series(self, time_start, time_end, metric, xy=None, dss_path=None):
        """
        Subclass must pass back a time array, and corresponding metric values
        :param time_start:
        :param time_end:
        :param WQ_metric: metric
        :param xy: [x,y] coordinates, optional if name is used
        :param name: identifier for reservoir or profile location, optional if xy used
        :return: (time, values) paired numpy arrays
        """
        raise NotImplementedError('subclass must define get_time_series()')
        elevation = np.array([0, ])
        values = np.array([0, ])
        return elevation, values


class W2ModelResults(ModelResults):
    '''
    Class to organize ResSim results for plotting
    '''

    def __init__(self, study_dir, region_name):
        '''
        initialized function for W2 class
        :param study_dir: study directory full path
        :param region_name: name of region
        '''
        self.region_name = region_name
        self.csv_results_dir = os.path.join(study_dir, 'reports', 'CSV')
        self.build_depths() #make output depths
        self.load_time() #load time values

    def build_depths(self):
        '''
        Depth values for pulling out W2. W2 is output at conistant intervals and converted to text files. There
        may not be values at every single depth, but those are represented with a no data flag.
        TODO: Currently hardcoded. Should be potentially changed to read in all output files in dir and build depths
        from that
        :return: list of depths for profiles
        '''
        start = 0
        increment = 2
        end = 160
        depths = range(start, end, increment)
        self.segment_depths = np.array(list(depths), dtype=np.int64)

    def load_time(self):
        '''
        Load times by reading temp profile files and finding time values. Keep trying files until one works.
        TODO: too hardcoded, search for files in results dir and use those.
        :return: start time and end time class objects
        '''
        print('Reading times..')
        for i ,depth in enumerate(self.segment_depths):
            ts_data_file = '{0}_tempprofile_Depth{1}_Idx{2}_Temperature.csv'.format(self.region_name.lower(), depth, i+1)
            t, v = observed_ts_txt_file_read(os.path.join(self.csv_results_dir, ts_data_file))
            try:
                self.stime = t[0]
                self.etime = t[-1]
                print('found time!')
                return
            except:
                print('Unable to set start and end time.')

    def load_time_array(self, t_in):
        '''
        turns a list of datetime objects to ordinal values
        TODO: this should return the list instead of setting class object.
        TODO: this should also be a generalized function
        :param t_in: list of datetime objects
        :return: class list object of ordinal times
        '''
        t = []
        for tt in t_in:
            ttmp = tt.toordinal() + float(tt.hour) / 24. + float(tt.minute) / (24. * 60.)
            t.append(ttmp)
        self.t = np.array(t)

    def get_profile(self, time_in, WQ_metric, name=None, return_depth=False):
        '''
        depreciated function, now we cal get_profiles() to grab all at once..
        :param time_in:
        :param WQ_metric:
        :param name:
        :param return_depth:
        :return:
        '''
        vals = np.zeros(len(self.segment_depths), dtype=np.float64)
        for i, depth in enumerate(self.segment_depths):
            ts_data_file = '{0}_tempprofile_Depth{1}_Idx{2}_Temperature.csv'.format(self.region_name, depth, i+1)
            t, v = observed_ts_txt_file_read(os.path.join(self.csv_results_dir, ts_data_file))
            if len(t) == 0:
                vals[i:] = np.NaN
                break
            #if i == 0:  # find index on first entry
            self.load_time_array(t)
            timestep = self.get_idx_for_time(time_in)
            if timestep == -1:
                vals[i:] = np.NaN
                break
            vals[i] = v[timestep]
        return self.segment_depths*3.28, vals #TODO: convert better.

    def get_profiles(self,times, metric, resname=None):
        '''
        load through values to extract values, depths and values
        In order to get elevations, a WaterSurfaceElev csv file needs to be read in
        :param times: array of time values to be retrieved. does not have to be in order or in a row
        :param metric: (NOT NEEDED) metric for data to be grabbed.
                        Should be pulled out of the function in the future for new metrics
                        Only here for consistancy with Ressim function
        :param resname: name of reservoir for H5 pathing, not needed for this but makes it easier to call
        :return: list of arrays of values, elevations and depths
        '''
        unique_times = [n[0] for n in times]
        values = np.full((len(unique_times), len(self.segment_depths)), np.NaN, dtype=np.float64)
        # elevations = np.full((len(unique_times), len(self.segment_depths)), np.NaN, dtype=np.float64)
        elevations = []
        # values = []
        depths = []
        for i, depth in enumerate(self.segment_depths):
            ts_data_file = '{0}_tempprofile_Depth{1}_Idx{2}_Temperature.csv'.format(self.region_name, depth, i+1)
            t, v = observed_ts_txt_file_read(os.path.join(self.csv_results_dir, ts_data_file))
            # vals = []
            if len(t) > 0:
                for j, time in enumerate(unique_times):
                    self.load_time_array(t)
                    timestep = self.get_idx_for_time(time)
                    if timestep > -1:
                        try:
                            values[j][i] = v[timestep]
                            # vals.append(v[timestep])
                        except:
                            print('No Data at idx {0} Depth {1} at time {2}'.format(i, depth, timestep))
            # values.append(np.asarray(vals))

        ts_elev_data =self.region_name + '_WaterSurfaceElev_Elev.csv'
        t, elev = observed_ts_txt_file_read(os.path.join(self.csv_results_dir, ts_elev_data))
        for j, time in enumerate(unique_times):
            e = []
            self.load_time_array(t)
            timestep = self.get_idx_for_time(time)
            if timestep > -1:
                WSE = elev[timestep] #Meters
                for depth in self.segment_depths:
                    e.append((WSE - depth) * 3.28) #conv to feet
            # e.reverse()
            elevations.append(np.asarray(e))
            depths.append(self.segment_depths * 3.28)

        return values, elevations, depths

    def get_idx_for_time(self, t_in):
        '''
        finds timestep for date
        TODO: generalize
        :param t_in: time step
        :return: timestep index
        '''
        ttmp = t_in.toordinal() + float(t_in.hour) / 24. + float(t_in.minute) / (24. * 60.)
        min_diff = np.min(np.abs(self.t - ttmp))
        tol = 1. / (24. * 60.)  # 1 minute tolerance
        timestep = np.where((np.abs(self.t - ttmp) - min_diff) < tol)[0][0]
        if min_diff > 1.:
            print('Error: nearest time step > 1 day away')
            return -1
        return timestep

    def get_time_series(self, metric, station):
        '''
        reads csv files of converted time series (see PostProcess_Region.py)
        sets stime and etime
        TODO: is stime and etime needed?
        :param metric: metric to grab for file
        :param station: location of data
        :return: arrays of time and values
        '''
        csv_name = '{0}_Fromw2_{1}.csv'.format(station.replace(' ', '_'), metric) #ex: Shasta_Outflow_Fromw2_Temperature
        csv_path = os.path.join(self.csv_results_dir, csv_name)
        t, v = observed_ts_txt_file_read(csv_path)
        if len(t) > 0:
            self.stime = t[0]
            self.etime = t[-1]

        return t, v

# Hi Scott.  You are awesome.
# thanks Ben, you too.

class ResSimModelResults(ModelResults):
    '''
    Class to organize ResSim results for plotting
    '''

    def __init__(self, sim_drct, trl_name, study_dir, subdomain=None, h5_filepath=None):
        if h5_filepath is not None:
            # use a different hdf file, for instance, is alternative parameters were changed in simulation mode, and
            # different results were generated manually and saved somewhere besides the rss directory.
            self.h = h5py.File(h5_filepath, 'r')
        else:
            h5fname = os.path.join(sim_drct, 'rss', trl_name + '.h5')
            self.h = h5py.File(h5fname, 'r')
        self.sim_dss_fn = os.path.join(sim_drct, 'simulation.dss')
        self.simulation_drct = sim_drct
        self.simulation_name = os.path.basename(sim_drct)
        self.trial_name = trl_name
        self.subdomain_name = subdomain
        self.load_time() #load time vars from h5
        self.csv_results_dir = os.path.join(study_dir, 'reports', 'CSV')
        self.output_dir = os.path.join(study_dir, 'reports', 'Images')

        self.hold_year = -1 #placeholder
        self.load_subdomains()

    def get_profile(self, time_in, WQ_metric, xy=None, name=None, return_depth=False):
        '''depreciated class. Now we grab all profiles at once, see get_profiles()'''
        self.load_elevation(alt_subdomain_name=name)
        self.load_results(time_in, WQ_metric, alt_subdomain_name=name)
        timestep = self.get_idx_for_time(time_in)
        ktop = self.get_top_layer(timestep)
        v = self.vals[:ktop + 1]
        el = self.elev[:ktop + 1]
        if return_depth:
            return np.max(el) - el[:], v
        else:
            print(el)
            return el, v

    def get_profiles(self, times, metric, resname=None):
        '''
        load through values to extract values, depths and values
        :param times: array of time values to be retrieved. does not have to be in order or in a row
        :param metric: metric for data to be grabbed. Should be pulled out of the function in the future for new metrics
        :param resname: name of reservoir for H5 pathing
        :return: list of arrays of values, elevations and depths
        '''
        self.load_elevation(alt_subdomain_name=resname)
        unique_times = [n[0] for n in times]

        vals = []
        elevations = []
        depths = []
        for j, time_in in enumerate(unique_times):
            timestep = self.get_idx_for_time(time_in)
            self.load_results(time_in, metric, alt_subdomain_name=resname)
            ktop = self.get_top_layer(timestep) #get waterlevel top layer to know where to grab data from
            v_el = self.vals[:ktop + 1]
            el = self.elev[:ktop + 1]
            d_step = []
            e_step = []
            v_step = []
            for ei, e in enumerate(el):
                d_step.append(np.max(el) - e)
                e_step.append(e)
                v_step.append(v_el[ei])

            depths.append(np.asarray(d_step))
            elevations.append(np.asarray(e_step))
            vals.append(np.asarray(v_step))

        return vals, elevations, depths

    def load_subdomains(self):
        '''
        creates a dictionary of all subdomains in the H5 file and grabes their XY coordinates for later reference
        :return: dictionary class object of subdomain XY coords
        '''
        self.subdomains = {}
        group = self.h['Geometry/Subdomains']
        for subdomain in group:
            dataset = self.h['Geometry/Subdomains/{0}/Cell Center Coordinate'.format(subdomain)]
            ncells = (np.shape(dataset))[0]
            x = np.array([dataset[i][0] for i in range(ncells)])
            y = np.array([dataset[i][1] for i in range(ncells)])
            self.subdomains[subdomain] = {'x': x, 'y': y}

    def find_computed_station_cell(self, xy):
        '''
        finds subdomains that are closest to observed station coordinates
        TODO: add some kind of tolerance or max distance?
        :param xy: XY coordinates for observed station
        :return: cell index and subdomain information closest to observed data
        '''
        easting = xy[0]
        northing = xy[1]
        nearest_dist = 1e6
        for subdomain, sd_data in self.subdomains.items():
            x = sd_data['x']
            y = sd_data['y']
            dist = np.sqrt((x - easting) * (x - easting) + (y - northing) * (y - northing))
            min_dist = np.min(dist)
            if min_dist < nearest_dist:
                min_cell = np.argmin(dist)
                data_index = min_cell
                data_subdomain = subdomain
                nearest_dist = min_dist
        return data_index, data_subdomain

    def load_computed_time(self):
        '''
        loads computed time values, replacing 24 hr date values with 0000 the next day
        grabs all values instead of user defined, if none are defined
        TODO: is this still needed? require user input.
        :return: sets list class object with times
                self.t_computed - list of times used in computation
                self.stime - start time of run
                self.etime - end time of run
        '''
        tstr = self.h['Results/Subdomains/Time Date Stamp']
        tstr0 = (tstr[0]).decode("utf-8")
        tstr1 = (tstr[1]).decode("utf-8")
        ttmp = self.h['Results/Subdomains/Time']
        nt = len(ttmp)
        try:
            ttmp0 = dt.datetime.strptime(tstr0, '%Y-%m-%d, %H:%M')
        except ValueError:
            tstrtmp = tstr0.replace('24:00', '23:00')
            ttmp0 = dt.datetime.strptime(tstrtmp, '%Y-%m-%d, %H:%M')
            ttmp0 += dt.timedelta(hours=1)
        try:
            ttmp1 = dt.datetime.strptime(tstr1, '%Y-%m-%d, %H:%M')
        except ValueError:
            tstrtmp = tstr1.replace('24:00', '23:00')
            ttmp1 = dt.datetime.strptime(tstrtmp, '%Y-%m-%d, %H:%M')
            ttmp1 += dt.timedelta(hours=1)
        delta_t = ttmp1 - ttmp0
        self.t_computed = []
        for j in range(nt):
            self.t_computed.append(ttmp0 + j * delta_t)
        self.t_computed = np.array(self.t_computed)

        if not hasattr(self, 'stime'):
            self.stime = self.t_computed[0]
            self.etime = self.t_computed[-1]

    def get_time_series(self, time_start, time_end, metric, xy=None):
        '''
        Gets Time series values from Ressim H5 files.
        :param time_start: start time to grab time series data
        :param time_end: end time to grab time series data
        :param metric: metric of data
        :param xy: XY coordinates of the observed data to be passed into self.find_computed_station_cell(xy) to find
                    location of modeled data

        :return: times and values arrays for selected metric and time window
        '''

        if xy is None:
            raise ValueError('xy must be set for ResSimModelResults')
        else:
            i, subdomain_name = self.find_computed_station_cell(xy)

            if metric.lower() == 'flow':
                dataset_name = 'Cell flow'
                dataset = self.h['Results/Subdomains/{0}/{1}'.format(subdomain_name, dataset_name)]
                v = np.array(dataset[:, i])
                v = clean_computed(v)
            elif metric.lower() == 'elevation':
                dataset_name = 'Water Surface Elevation'
                dataset = self.h['Results/Subdomains/{0}/{1}'.format(subdomain_name, dataset_name)]
                v = np.array(dataset[:])
                v = clean_computed(v)
            elif metric.lower() == 'temperature':
                dataset_name = 'Water Temperature'
                dataset = self.h['Results/Subdomains/{0}/{1}'.format(subdomain_name, dataset_name)]
                print('Results/Subdomains/{0}/{1}'.format(subdomain_name, dataset_name))
                v = np.array(dataset[:, i])
                v = clean_computed(v)
            elif metric.lower() == 'do':
                dataset_name = 'Dissolved Oxygen'
                dataset = self.h['Results/Subdomains/{0}/{1}'.format(subdomain_name, dataset_name)]
                v = np.array(dataset[:, i])
                v = clean_computed(v)
            elif metric.lower() == 'do_sat':
                dataset_name = 'Water Temperature'
                dataset = self.h['Results/Subdomains/{0}/{1}'.format(subdomain_name, dataset_name)]
                vt = np.array(dataset[:, i])
                dataset_name = 'Dissolved Oxygen'
                dataset = self.h['Results/Subdomains/{0}/{1}'.format(subdomain_name, dataset_name)]
                vdo = np.array(dataset[:, i])
                vt = clean_computed(vt)
                vdo = clean_computed(vdo)
                v = calc_computed_dosat(vt, vdo)

        if not hasattr(self, 't_computed'):
            self.load_computed_time()
        istart = 0
        iend = -1
        if time_start is not None:
            istart = np.argmin(self.t_computed, time_start)[0]
            if self.t_computed[istart] < time_start:
                istart += 1
        if time_end is not None:
            iend = np.argmin(self.t_computed, time_end)[0]
            if self.t_computed[iend] > time_end:
                iend -= 1
        self.stime = self.t_computed[0]
        self.etime = self.t_computed[-1]
        return self.t_computed[istart:iend], v[istart:iend]



    def load_time(self):
        '''
        Loads times from H5 file
        TODO: do we need this, or the load computed times?
        :return: list class object of time values
                self.tstr - time string
                self.t - times list
                self.nt - num times
                self.t_offset - time offset
                self.stime - start time
                self.etime - end time
        '''
        self.tstr = self.h['Results/Subdomains/Time Date Stamp']
        tstr0 = (self.tstr[0]).decode("utf-8")
        ttmp = self.h['Results/Subdomains/Time']
        self.t = ttmp[:]
        self.nt = len(self.t)
        try:
            ttmp = dt.datetime.strptime(tstr0, '%Y-%m-%d, %H:%M')
        except ValueError:
            tstrtmp = tstr0.replace('24:00', '23:00')
            ttmp = dt.datetime.strptime(tstrtmp, '%Y-%m-%d, %H:%M')
            ttmp += dt.timedelta(hours=1)
        t_offset = ttmp.toordinal() + float(ttmp.hour) / 24. + float(ttmp.minute) / (24. * 60.)
        self.t_offset = t_offset
        self.stime = ttmp
        tstrm1 = (self.tstr[-1]).decode("utf-8")
        try:
            ttmp = dt.datetime.strptime(tstrm1, '%Y-%m-%d, %H:%M')
        except ValueError:
            tstrtmp = tstrm1.replace('24:00', '23:00')
            ttmp = dt.datetime.strptime(tstrtmp, '%Y-%m-%d, %H:%M')
            ttmp += dt.timedelta(hours=1)
        self.etime = ttmp

    def load_elevation(self, alt_subdomain_name=None):
        '''
        loads elevations from the H5 file
        :param alt_subdomain_name: alternate field if the domain is not class defined subdomain name
        :return: assign elevations to class
                self.ncells - number of cells
                self.elev - elevation time series for profile
                self.elev_ts - elevation time series

        '''
        this_subdomain = self.subdomain_name if alt_subdomain_name is None else alt_subdomain_name
        cell_center_xy = self.h['Geometry/Subdomains/' + this_subdomain + '/Cell Center Coordinate']
        self.ncells = (np.shape(cell_center_xy))[0]
        self.elev = np.array([cell_center_xy[i][2] for i in range(self.ncells)])
        elev_ts = self.h['Results/Subdomains/' + this_subdomain + '/Water Surface Elevation']
        self.elev_ts = np.array([elev_ts[i] for i in range(self.nt)])

    def get_top_layer(self, timestep_index):
        '''
        grabs the top active layer of water for a given timestep
        :param timestep: timestep index to grab data at
        :return: returns index for top layer of water column
        '''
        elev = self.elev_ts[timestep_index] #elevations at a timestep
        for k in range(len(self.elev) - 1): #for each cell..
            cell_z = self.elev[k]  # layer midpoint
            cell_z1 = self.elev[k + 1]  # layer above midpoint
            top_of_cell_z = 0.5 * (cell_z + cell_z1)
            if elev < top_of_cell_z:
                break
        return k

    def load_results(self, t_in, metrc, alt_subdomain_name=None):
        '''
        loads results for a specific time step from h5 file
        :param t_in: time in datetime object
        :param metrc: metric to get data from
        :param alt_subdomain_name: alt field if data is grabbed from a different location than the default
        :return: assign values to class object
                self.t_data - timestep
                self.vals - array of values
        '''
        this_subdomain = self.subdomain_name if alt_subdomain_name is None else alt_subdomain_name

        timestep = self.get_idx_for_time(t_in) #get timestep index for current date
        print(t_in, (self.tstr[timestep]).decode("utf-8"))
        self.t_data = t_in
        if metrc.lower() == 'temperature':
            metric_name = 'Water Temperature'
            try:
                v = self.h['Results/Subdomains/' + this_subdomain + '/' + metric_name]
            except KeyError:
                raise KeyError('WQ Simulation does not have results for metric: {0}'.format(metric_name))
            vals = np.array([v[timestep][i] for i in range(self.ncells)])
        elif metrc == 'diss_oxy' or metrc.lower() == 'do':
            metric_name = 'Dissolved Oxygen'
            try:
                v = self.h['Results/Subdomains/' + this_subdomain + '/' + metric_name]
            except KeyError:
                raise KeyError('WQ Simulation does not have results for metric: {0}'.format(metric_name))
            vals = np.array([v[timestep][i] for i in range(self.ncells)])
        elif metrc.lower() == 'do_sat':
            metric_name = 'Water Temperature'
            vtmp = self.h['Results/Subdomains/' + this_subdomain + '/' + metric_name]
            vt = np.array([vtmp[timestep][i] for i in range(self.ncells)])
            metric_name = 'Dissolved Oxygen'
            vtmp = self.h['Results/Subdomains/' + this_subdomain + '/' + metric_name]
            vdo = np.array([vtmp[timestep][i] for i in range(self.ncells)])
            vals = calc_computed_dosat(vt, vdo)
        self.vals = vals

    def get_idx_for_time(self, t_in):
        '''
        Returns closest index to timestep datetime object from H5 file times
        :param t_in: datetime object
        :return: index closest to time step
        '''
        ttmp = t_in.toordinal() + float(t_in.hour) / 24. + float(t_in.minute) / (24. * 60.) - self.t_offset
        min_diff = np.min(np.abs(self.t - ttmp))
        tol = 1. / (24. * 60.)  # 1 minute tolerance
        timestep = np.where((np.abs(self.t - ttmp) - min_diff) < tol)[0][0]
        if min_diff > 1.:
            print('Error: nearest time step > 1 day away')
            print(t_in, timestep)
        return timestep

def get_plot_label_masks(idx, nprofiles, rows, cols):
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


def get_subplot_config(n_profiles):
    # factor = n_profiles / 4.
    factor = n_profiles / 3.
    if factor < 1:
        return 1, n_profiles
    else:
        # return math.ceil(factor), 4
        return math.ceil(factor), 3


def obs_model_profile_plot(ax, obs_elev, obs_value, model_elev, model_value, metric, dt_profile,
                           show_legend=False, show_xlabel=False, show_ylabel=False, use_depth=False):

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


def plot_profiles(mr, metric, observed_data_drct, reservoir, out_path, use_depth=False):
    """
    generate profile plots and report back error stats

    """
    # hard_coded for now
    # n_profiles_per_page = 12
    n_profiles_per_page = 9

    syear = mr.stime.year
    eyear = mr.etime.year
    if mr.etime.month == 1:
        eyear -= 1

    profile_results = []
    for yr in range(syear, eyear + 1):
        observed_data_file_name = " ".join(['Profile', reservoir, metric.lower()]) + '_{0}.txt'.format(yr)
        observed_data_file_name = os.path.join(observed_data_drct, observed_data_file_name)
        if os.path.exists(observed_data_file_name):
            obs_times, obs_values, obs_depths = read_observed(observed_data_file_name)
            nof_profiles = len(obs_times)
            n_pages = math.ceil(nof_profiles / n_profiles_per_page)
            model_values, model_elev,  model_depths = mr.get_profiles(obs_times, metric, resname=reservoir)

            if not use_depth:
                obs_elev = convert_obs_depths(obs_depths, model_elev)


            # break profile indices into page groups
            prof_indices = list(range(nof_profiles))
            n = n_profiles_per_page
            page_indices = [prof_indices[i * n:(i + 1) * n] for i in range((len(prof_indices) + n - 1) // n)]

            fig_names = []
            stats = []
            for page_i, pgi in enumerate(page_indices):

                subplot_rows, subplot_cols = get_subplot_config(len(pgi))
                n_nrow_active = np.ceil(len(pgi) / subplot_cols)
                # print('n_nrow_active',n_nrow_active)
                fig = plt.figure(figsize=(7, 1 + 3 * n_nrow_active))

                for i, j in enumerate(pgi):
                    pax = fig.add_subplot(subplot_rows, subplot_cols, i + 1)

                    # observation
                    dt_profile = obs_times[j]
                    obs_val = obs_values[j]
                    model_val = model_values[j]

                    if use_depth:
                        obs_levels = obs_depths[j]
                        model_levels = model_depths[j]
                    else:
                        obs_levels = obs_elev[j]
                        model_levels = model_elev[j]

                    lflag, xflag, yflag = get_plot_label_masks(i, len(pgi), subplot_rows, subplot_cols)
                    obs_model_profile_plot(pax, obs_levels, obs_val, model_levels, model_val, metric, dt_profile[0],
                                           show_legend=lflag, show_xlabel=xflag, show_ylabel=yflag, use_depth=use_depth)

                    # why do stats here? because it may be time consuming to pull observations and model data a second time
                    stats.append(series_stats(model_levels, model_val, obs_levels, obs_val,
                                              start_limit=None, end_limit=None, time_series=False))

                plt.tight_layout()
                # plt.show()
                fig_names.append(reservoir + '_' + metric + '_' + str(yr) + '_%02i.png' % page_i)
                plt.savefig(os.path.join(out_path, fig_names[-1]), dpi=600)
                plt.close('all')
            profile_results.append([reservoir, yr, fig_names, stats])
        else:
            print('No %s profile observations for year %i' % (reservoir, yr))
    return profile_results

def convert_obs_depths(obs_depths, model_elevs):
    obs_elev = []
    for i, d in enumerate(obs_depths):
        e = []
        topwater_elev = max(model_elevs[i])
        for depth in d:
            e.append(topwater_elev - depth)
        obs_elev.append(np.asarray(e))
    return obs_elev





mo_str_3 = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


def plot_time_series(mr, observed_data_meta_file, fig_path_stub, rss_sim_name, rss_alt_name, region_name,
                     model, temperature_only=False):
    # parse observed data file
    stations = read_obs_ts_meta_file(observed_data_meta_file)
    ts_data_path, _ = os.path.split(observed_data_meta_file)  # put that meta file with data!

    station_results = []

    for station, data in stations.items():
        # print('Creating Plot for', station)
        # print('Getting observed data')
        # obs_dates, obs_vals = self.get_observed_data(station)
        x = data['easting']
        y = data['northing']
        metric = data['metric']
        region = data['region'].lower()
        longname = data['longname']
        if region != region_name.lower():
            continue
        if not temperature_only or metric.lower() == 'temperature':
            obs_dates, obs_vals = read_observed_ts_data(ts_data_path, station, metric)
            # print('Getting computed data')
            if model.lower() == 'ressim':
                comp_dates, comp_vals = mr.get_time_series(None, None, metric, xy=[x, y])
            else:
                comp_dates, comp_vals = mr.get_time_series(metric, station)
            # print('Making plot')

            if len(comp_dates) > 0:

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
                    ylabel_str = 'Flow'
                    units_str = mr.units[metric]
                    ax.set_ylim([0., max([np.nanmax(obs_vals[omsk]), np.nanmax(comp_vals[cmsk])])])
                elif metric.lower() == 'elevation':
                    ylabel_str = 'Surface Elevation'
                    units_str = 'ft'
                    ax.set_ylim([min([np.nanmin(obs_vals[omsk]), np.nanmin(comp_vals[cmsk])]),
                                 max([np.nanmax(obs_vals[omsk]), np.nanmax(comp_vals[cmsk])])])
                elif metric.lower() == 'temperature':
                    ylabel_str = 'Water Temperature'
                    units_str = mr.units[metric.lower()]
                    ax.set_ylim([0., max([np.nanmax(obs_vals[omsk]), np.nanmax(comp_vals[cmsk])])])
                elif metric.lower() == 'do':
                    ylabel_str = 'Dissolved Oxygen'
                    units_str = mr.units[metric]
                    ax.set_ylim([0., max([np.nanmax(obs_vals[omsk]), np.nanmax(comp_vals[cmsk])])])
                elif metric.lower() == 'do_sat':
                    ylabel_str = 'Dissolved Oxygen Saturation'
                    units_str = mr.units[metric]
                    ax.set_ylim([min([np.nanmin(obs_vals[omsk]), np.nanmin(comp_vals[cmsk])]),
                                 max([np.nanmax(obs_vals[omsk]), np.nanmax(comp_vals[cmsk])])])
                ax.set_ylabel('{0} ({1})'.format(metric.capitalize(), units_str))
                ax.set_title('{0}, Simulation: {1}, {2}'.format(station, os.path.basename(rss_sim_name), rss_alt_name))
                plt.grid()
                fig_name = metric.capitalize() + '_' + station.replace(' ', '_') + '.png'
                plt.savefig(os.path.join(fig_path_stub, fig_name), bbox_inches='tight', dpi=600)
                plt.close()
        if len(comp_dates) > 0:
            # break stats into years, months
            # find n years
            n_years = obs_dates[-1].year - obs_dates[0].year + 1
            stats = {}
            stats_months = {}
            for yr in range(obs_dates[0].year, obs_dates[-1].year + 1):
                stdate = dt.datetime(yr, 1, 1)
                enddate = dt.datetime(yr + 1, 1, 1)
                # at some point should add metric to key, which is used as label
                # print(station, metric, yr)
                stats_yr = series_stats(comp_dates, comp_vals, obs_dates, obs_vals,
                                        start_limit=stdate, end_limit=enddate, time_series=True)
                if stats_yr is not None:
                    stats[str(yr)] = stats_yr
                    stats_months[str(yr) + ' Obs. Mean'] = {}
                    stats_months[str(yr) + ' Comp. Mean'] = {}
                    for mo in range(1, 13):
                        enddate = dt.datetime(yr + 1, 1, 1) if mo == 12 else dt.datetime(yr, mo + 1, 1)
                        mo_stat = series_stats(comp_dates, comp_vals, obs_dates, obs_vals,
                                               start_limit=dt.datetime(yr, mo, 1), end_limit=enddate,
                                               means_only=True, time_series=True)
                        if mo_stat is None:
                            stats_months[str(yr) + ' Obs. Mean'][mo_str_3[mo - 1]] = None
                            stats_months[str(yr) + ' Comp. Mean'][mo_str_3[mo - 1]] = None
                        else:
                            stats_months[str(yr) + ' Obs. Mean'][mo_str_3[mo - 1]] = mo_stat['Obs. Mean']
                            stats_months[str(yr) + ' Comp. Mean'][mo_str_3[mo - 1]] = mo_stat['Comp. Mean']

            station_results.append([station, x, y, metric, longname, fig_name, stats, stats_months])

    return station_results

def get_reservoir_description(region):
    return "{0} Reservoir Temperature Profiles Near Dam".format(region.capitalize())


def XML_reservior(profile_stats, XML_class, region_name):
    # only writing one reservoir XLM section here
    XML = ''

    #res name, year, list of pngs, stats dictionary
    for subdomain_name, figure_sets in profile_stats.items():
        if len(figure_sets) > 0:
            for ps in figure_sets:
                reservoir, yr, fig_names, stats = ps
                # fig_names = [os.path.join('..','Images', n) for n in fig_names]
                subgroup_desc = get_reservoir_description(region_name)
                XML += XML_class.make_ReservoirSubgroup_lines(reservoir,fig_names, subgroup_desc, yr)

    return XML



def XML_time_series(ts_results, XML_class):
    stats_labels = {
        'Mean Bias': r'Mean Bias (&lt;sup&gt;O&lt;/sup&gt;C)',
        'MAE': r'MAE (&lt;sup&gt;O&lt;/sup&gt;C)',
        'RMSE': r'RMSE (&lt;sup&gt;O&lt;/sup&gt;C)',
        'NSE': r'Nash-Sutcliffe (NSE)',
        'COUNT': r'COUNT',
    }
    stats_ordered = ['Mean Bias', 'MAE', 'RMSE', 'NSE', 'COUNT']

    XML = ""
    for ts in ts_results:
        station, x, y, metric, desc, fig_name, stats, stats_mo = ts
        # fig_name = os.path.join('..','Images', fig_name)
        # subgroup_desc = get_ts_description(station)
        XML += XML_class.make_TS_Subgroup_lines(station, fig_name, desc)
        XML += XML_class.make_TS_Tables_lines(station, stats, stats_mo, stats_ordered, stats_labels)
        XML += '        </Report_Subgroup>\n'

    return XML

def XML_write(profile_stats, ts_results, region_name):

    XML = XMLReport.makeXMLReport("USBRAutomatedReportOutput.xml")

    XML_res = ""
    XML_ts = ""
    XML_Groupheader = XML.make_Reservoir_Group_header(region_name) #TODO CHANGE GROUP ORDER

    if len(profile_stats) > 0:
        XML_res = XML_reservior(profile_stats, XML, region_name)
    if len(ts_results) > 0:
        XML_ts = XML_time_series(ts_results,XML)
    if len(ts_results) + len(profile_stats) > 0:
        XML.write_Reservoir(region_name, plugin_name, XML_Groupheader, XML_res, XML_ts)
    else:
        print('No Results found for Region', region_name)


XML_fname = 'USBRAutomatedReportOutput.xml'


def clean_output_dir(dir_name):
    files_in_directory = os.listdir(dir_name)
    filtered_files = [file for file in files_in_directory if file.endswith(".png")]
    for file in filtered_files:
        path_to_file = os.path.join(dir_name, file)
        os.remove(path_to_file)

def generate_region_plots_ResSim(simulation_path, alternative, observed_data_directory, region_name, study_dir,
                                 temperature_only=False, use_depth=False):
    '''
    Function making regional XML section for ResSim models.
    :param simulation_path: full path to the simulation dir
    :param alternative: alternative name
    :param observed_data_directory: full path to observed data dir
    :param region_name: name of current region (Shasta, Keswick, etc)
    :param study_dir: full path to the study directory
    :param temperature_only: flag for using only temp. This is a bit older carry over code and should be reworked.
    :param use_depth: Make plots use depth if True, elevation if False
    :return:
    '''

    #find path to output images
    images_path = os.path.join(study_dir, 'reports', 'Images')

    #build path to station files
    profile_meta_file = os.path.join(observed_data_directory, "Profile_stations.txt")
    ts_meta_file = os.path.join(observed_data_directory, "TS_stations.txt")

    # we should only need to generate a single instance of ResSimModelResults; while we need to pass in a
    # subdomain/reservoir for legacy purposes at this point, and subdomain/reservoir can be passed to through
    # to the get_profile command
    profile_subdomains = read_obs_profile_meta_file(profile_meta_file) #read profiles
    mr = ResSimModelResults(simulation_path, alternative, study_dir) #create class instance for ressim class

    profile_stats = {}
    for subdomain, meta in profile_subdomains.items():
        if region_name.lower() == meta['region'].lower():
            if meta['metric'].lower() == 'temperature' or not temperature_only:
                profile_stats[subdomain] = plot_profiles(mr, meta['metric'], observed_data_directory, subdomain,
                                                         images_path,
                                                         use_depth=use_depth)

    ts_results = plot_time_series(mr, ts_meta_file, images_path, simulation_path, alternative, region_name,
                                  'ressim', temperature_only=temperature_only)

    XML_write(profile_stats, ts_results, region_name)

def generate_region_plots_W2(simulation_path, alternative, observed_data_directory, region_name, study_dir,
                                temperature_only=False, use_depth=False):
    # output_path = r"Z:\USBR\test"  # WAT will start path where we need to write
    # output_path = "."  # WAT will start path where we need to write
    # images_path = os.path.join(output_path, '..', "Images")
    # output_path = os.path.join(study_dir, 'reports')
    images_path = os.path.join(study_dir, 'reports', 'Images')
    output_path = os.path.join(study_dir, 'reports')

    profile_meta_file = os.path.join(observed_data_directory, "Profile_stations.txt")
    ts_meta_file = os.path.join(observed_data_directory, "TS_stations.txt")


    # we should only need to generate a single instance of ResSimModelResults; while we need to pass in a
    # subdomain/reservoir for legacy purposes at this point, and subdomain/reservoir can be passed to through
    # to the get_profile command
    profile_subdomains = read_obs_profile_meta_file(profile_meta_file)
    mr = W2ModelResults(study_dir, region_name)

    profile_stats = {}
    for subdomain, meta in profile_subdomains.items():
        if region_name.lower() == meta['region'].lower():
            if meta['metric'].lower() == 'temperature' or not temperature_only:
                profile_stats[subdomain] = plot_profiles(mr, meta['metric'], observed_data_directory, subdomain,
                                                         images_path,
                                                         use_depth=use_depth)

    ts_results = plot_time_series(mr, ts_meta_file, images_path, simulation_path, alternative, region_name, 'CeQualW2',
                                  temperature_only=temperature_only)

    XML_write(profile_stats, ts_results, region_name)



if __name__ == '__main__':
    # rem %1 studyDir,
    # rem %2 simDir,
    # rem %3 program name ( ResSim, RAS etc)
    # rem %4 fpart (ressim .h5 file)
    # rem %5 obs data folder
    # rem %6 model alternative name
    # rem %7 simulation name (for the .rptgen file


    study_dir = sys.argv[1]
    simulation_directory = sys.argv[2]
    plugin_name = sys.argv[3]
    alternative_name = sys.argv[4]
    obs_data_path = sys.argv[5]
    model_alt_name = sys.argv[6]
    simulation_name = sys.argv[7]

    reg_info = find_rptrgn(simulation_name, study_dir)
    print('SYSARG', sys.argv)
    print('REGINFO', reg_info)

    #Flag for temperature profile plots. Determines if plots are shown as depth values (0 -> down) or as
    #elevations (Real WSE -> down).
    use_depth = False

    try:
        #find region info. Potentially returns multiple regions
        region_names = reg_info[model_alt_name.replace(' ', '_')]['regions']
    except Exception:
        print('Error finding region')
        exit()

    #for each region found, generate a region section in the XML file
    for region_name in region_names:

        if plugin_name == 'ResSim':

            generate_region_plots_ResSim(simulation_directory, alternative_name, obs_data_path, region_name, study_dir,
                                         temperature_only=True, use_depth=use_depth)
        elif plugin_name == 'CeQualW2':

            generate_region_plots_W2(simulation_directory, alternative_name, obs_data_path, region_name, study_dir,
                                     temperature_only=True, use_depth=use_depth)
        else:
            #somehow got a region not taken account for. Only Ressim and W2 right now.
            print('model "%s" not understood!' % sys.argv[0])
    exit()
