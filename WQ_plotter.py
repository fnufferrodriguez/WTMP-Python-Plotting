"""
Created on 7/1/2020

@author: Stephen Andrews, Ben Saenz
@organization: Resource Management Associates
@contact: steve@rmanet.com
@note: 
"""
__updated__ = '11-21-2019 13:14'

import copy
import datetime as dt
import math
import os
import pickle
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.dates import date2num
from scipy import interpolate
from scipy.constants import convert_temperature
from sklearn.metrics import mean_absolute_error
import XMLReport
import report_utils as RU


sat_data_do = [14.60, 14.19, 13.81, 13.44, 13.09, 12.75, 12.43, 12.12, 11.83, 11.55, 11.27, 11.01, 10.76, 10.52, 10.29,
               10.07, 9.85, 9.65, 9.45, 9.26, 9.07, 8.90, 8.72, 8.56, 8.40, 8.24, 8.09, 7.95, 7.81, 7.67, 7.54, 7.41,
               7.28, 7.16, 7.05, 6.93, 6.82, 6.71, 6.61, 6.51, 6.41, 6.31, 6.22, 6.13, 6.04, 5.95]
sat_data_temp = [0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20., 21.,
                 22., 23., 24., 25., 26., 27., 28., 29., 30., 31., 32., 33., 34., 35., 36., 37., 38., 39., 40., 41.,
                 42., 43., 44., 45.]
f_interp = interpolate.interp1d(sat_data_temp, sat_data_do,
                                fill_value=(sat_data_do[0], sat_data_do[-1]), bounds_error=False)


def do_saturation(temp, diss_ox):
    do_sat = f_interp(temp)
    return diss_ox / do_sat * 100.


def calc_computed_dosat(vtemp, vdo):
    v = np.zeros_like(vtemp)
    for j in range(len(v)):
        if np.isnan(vtemp[j]) or np.isnan(vdo[j]):
            v[j] = np.nan
        else:
            v[j] = do_saturation(vtemp[j], vdo[j])
    return v


def read_DSS_txt_header(fp):
    """Example headr:
    /TEMPPROFILE/SHASTA/PROFILE/TEMPF/01DEC2009/TEMPERATURE (F)/
    PD  Ver:  1   Prog:DssVue  LW:17MAY21  22:33:10   Tag:Tag        Prec:-1
    1 Curve(s), 132 Ordinates.  IHORIZ: 1
    First Var. Units: DegF   Type: unt Second Var. Units: feet   Type: unt
    Label 1: TempF
    """
    hdr = {}
    hdr['DSS_record'] = fp.readline()
    if not hdr['DSS_record'] or 'END FILE' in hdr['DSS_record']:
        return None
    tokens = hdr['DSS_record'].split('/')
    hdr['Location'] = tokens[2]
    hdr['Datetime'] = dt.datetime.strptime(tokens[5], '%d%b%Y')
    hdr['Timestamp'] = pd.to_datetime(hdr['Datetime'])
    l1 = fp.readline()
    l2 = fp.readline()
    l2_tokens = l2.split(' ')
    hdr['n'] = int(l2_tokens[2])
    l3 = fp.readline()
    l3_tokens = l3.split(' ')
    hdr['units'] = l3_tokens[3]
    fp.readline()  # burn last of 5 total lines
    return hdr


def read_DSS_txt_profile(fp, n):
    values = np.array([float(fp.readline()) for i in range(n)])
    values = convert_temperature(values, 'F', 'C')
    elevation = np.array([float(fp.readline()) for i in range(n)])
    elevation = -1 * (elevation - elevation[0])
    order = np.flip(np.argsort(elevation))
    elevation = elevation[order]
    values = values[order]
    fp.readline()  # burn 'End Data' line
    return values, elevation


def write_profiles_to_AnnualFile(profiles, yr, file_name_stub):
    with open(file_name_stub + '%i.txt' % yr, 'w') as fp:
        fp.write('date,temp,elev\n')
        for p in profiles:
            if p['Datetime'].year == yr:
                dstr = p['Datetime'].strftime('%Y-%m-%d %H:%M:%S')
                for i in range(p['n']):
                    fp.write(dstr + ',%f,%f\n' % (p['values'][i], p['elevation'][i]))


def DSS_profile_txt_to_other_stuff(DSS_profile_txt, years, file_name_stub):
    """Utility to write out annual profile text files from a DSS text export of lots of paired profile data.
    """
    with open(DSS_profile_txt) as fp:
        p = []
        maxprofiles = 10000
        for i in range(maxprofiles):
            print('Reading profile:', i)
            profile = read_DSS_txt_header(fp)
            if profile is None:
                break
            profile['values'], profile['elevation'] = read_DSS_txt_profile(fp, profile['n'])
            p.append(profile)
    # sort by date
    p.sort(key=profile_dn)

    for y in years:
        write_profiles_to_AnnualFile(p, y, file_name_stub)
    with open(file_name_stub + '.pickle', 'wb') as fp:
        pickle.dump(p, fp)


def profile_dn(profile):
    return date2num(profile['Datetime'])


def read_obs_ts_meta_file(obs_ts_meta_file):
    stations = {}
    obs_dss_file = None
    with open(obs_ts_meta_file) as osf:
        for line in osf:
            line = line.strip()
            if line.startswith('OBS_FILE'):
                obs_dss_file = os.path.join(os.getcwd(), line.split('=')[1])
            elif line.startswith('start station'):
                name = ''
                metric = ''
                easting = 0
                northing = 0
                dss_computed = None
                dss_path = None
                dss_fn = None
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
                                  'dss_computed': dss_computed, 'region':region, 'dss_path': dss_path,
                                  'dss_fn': dss_fn, 'w2_path': w2_path}
    return stations, obs_dss_file


def read_obs_profile_meta_file(obs_profile_meta_file):
    stations = {}
    obs_dss_file = None
    with open(obs_profile_meta_file) as osf:
        for line in osf:
            line = line.strip()
            if line.startswith('OBS_FILE'):
                obs_dss_file = os.path.join(os.getcwd(), line.split('=')[1])
            elif line.startswith('start station'):
                name = ''
                metric = ''
                region = ''
                easting = 0
                northing = 0
            elif line.startswith('name'):
                name = line.split('=')[1]
            elif line.startswith('metric'):
                metric = line.split('=')[1]
            elif line.startswith('region'):
                region = line.split('=')[1]
            elif line.startswith('end station'):
                stations[name] = {'metric': metric, 'region':region}
    return stations, obs_dss_file


def observed_ts_txt_file(ts_data_path, station_name, metric):
    #TODO: change reading to assume regular, do normal way, and check end date, if not what we expect, go back find missing dates
    dss_util_fname = os.path.join(ts_data_path, '{0} {1}.txt'.format(metric, station_name))
    return observed_ts_txt_file_read(dss_util_fname)

def observed_ts_txt_file_read(file_name, skipheader=False):
     #TODO: change reading to assume regular, do normal way, and check end date, if not what we expect, go back find missing dates
    dss_util_fname = file_name
    t = []
    v = []
    print(file_name)
    with open(dss_util_fname) as f:
        if not skipheader:
            for _ in range(4):
                next(f)
        for _ in range(2):
            line = f.readline()
            if line.startswith('No Data Found.'):
                print('No Data found.')
                return [], []
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
        delta_t = t[1] - t[0]
        for line in f:
            if line.startswith('END'):
                break
            sline = line.split(';')
            t.append(t[-1] + delta_t)
            # empty DSS data is missing from txt file, deal with it
            try:
                v.append(float(sline[1]))
            except:
                v.append(np.nan)
    return np.array(t), np.array(v)


def clean_missing(indata):
    indata[indata == -901.] = np.nan
    return indata


def clean_computed(indata):
    indata[indata == -9999.] = np.nan
    return indata


def calc_computed_dosat(vtemp, vdo):
    v = np.zeros_like(vtemp)
    for j in range(len(v)):
        if np.isnan(vtemp[j]) or np.isnan(vdo[j]):
            v[j] = np.nan
        else:
            v[j] = do_saturation(vtemp[j], vdo[j])
    return v


def calc_observed_dosat(ttemp, vtemp, tdo, vdo):
    v = np.zeros_like(vtemp)
    for j in range(len(v)):
        if np.isnan(vtemp[j]) or np.isnan(vdo[j]):
            v[j] = np.nan
        else:
            v[j] = do_saturation(vtemp[j], vdo[j])
    return ttemp, v


def read_observed_ts_data(ts_data_path, station_name, metric):
    if metric.lower() == 'do_sat':
        tt, vt = observed_ts_txt_file(ts_data_path, station_name, 'Temperature')
        vt = clean_missing(vt)
        tdo, vdo = observed_ts_txt_file(ts_data_path, station_name, 'DO')
        vdo = clean_missing(vdo)
        t, v = calc_observed_dosat(tt, vt, tdo, vdo)
    else:
        t, v = observed_ts_txt_file(ts_data_path, station_name, metric)
        v = clean_missing(v)
        if metric.lower() == 'temperature':
            if np.any(v > 45):
                v = convert_temperature(v, 'F', 'C')
    return t, v


def read_observed(observed_data_filename):
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
        # if not stime <= dt_tmp <= etime:
        #     continue
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

            # for j in range(len(tstamp_obs)):
            #     t = tstamp_obs[j]
            #     vo = v_obs[j]
            #     if not np.isnan(vo):
            #         vc = f_computed(t)
            #         if not np.isnan(vc):
            #             mean_diff += vc - vo
            #             sq_error += (vc - vo)**2
            #             count += 1
            # mean_diff = mean_diff / count
            # rmse = np.sqrt(sq_error / count)

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


def dss_get(dss, path, startDateStr=None, endDateStr=None):
    df, _, _ = dss.read_rts(path, startDateStr, endDateStr)
    dt = pd.to_datetime(df.index).to_pydatetime()
    val = df[df.columns[0]].values
    return dt, val


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
    # 'ELEV' seems to be the same for all layers/depths
    def __init__(self, simulation_path, alternative, region_name):
        self.region_name = region_name
        output_path = "."
        self.csv_results_dir = os.path.join(output_path, '..', 'CSV')
        self.build_depths()
        self.load_time()

    def build_depths(self):
        start = 0
        increment = 2
        end = 160
        depths = range(start, end, increment)
        self.segment_depths = np.array(list(depths), dtype=np.int)

    def load_time(self):
        depth = 0
        i = 0
        ts_data_file = '{0}_tempprofile_Depth{1}_Idx{2}_Temperature.csv'.format(self.region_name.lower(), depth, i+1)
        t, v = observed_ts_txt_file_read(os.path.join(self.csv_results_dir, ts_data_file), skipheader=True) #TODO: rename skip header
        self.stime = t[0]
        self.etime = t[-1]

    def load_time_array(self, t_in):
        t = []
        for tt in t_in:
            ttmp = tt.toordinal() + float(tt.hour) / 24. + float(tt.minute) / (24. * 60.)
            t.append(ttmp)
        self.t = np.array(t)

    def get_profile(self, time_in, WQ_metric, name=None, return_depth=False):
        vals = np.zeros(len(self.segment_depths), dtype=np.float)
        for i, depth in enumerate(self.segment_depths):
            ts_data_file = '{0}_tempprofile_Depth{1}_Idx{2}_Temperature.csv'.format(self.region_name, depth, i+1)
            t, v = observed_ts_txt_file_read(os.path.join(self.csv_results_dir, ts_data_file), skipheader=True)
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

    def get_idx_for_time(self, t_in):
        #TODO: merge this with other get_idx_for_time method?
        ttmp = t_in.toordinal() + float(t_in.hour) / 24. + float(t_in.minute) / (24. * 60.)
        min_diff = np.min(np.abs(self.t - ttmp))
        tol = 1. / (24. * 60.)  # 1 minute tolerance
        timestep = np.where((np.abs(self.t - ttmp) - min_diff) < tol)[0][0]
        if min_diff > 1.:
            print('Error: nearest time step > 1 day away')
            return -1
        return timestep

    def get_time_series(self, time_start, time_end, metric, xy=None, dss_path=None):
        w2_name = dss_path['w2_path']



# Hi Scott.  You are awesome.


class ResSimModelResults(ModelResults):

    def __init__(self, sim_drct, trl_name, subdomain=None, h5_filepath=None):
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
        self.load_time()
        # self.load_elevation()
        # self.make_output_dir()
        self.hold_year = -1
        self.load_subdomains()

    def get_profile(self, time_in, WQ_metric, xy=None, name=None, return_depth=False):
        self.load_elevation(alt_subdomain_name=name)
        self.load_results(time_in, WQ_metric, alt_subdomain_name=name)
        timestep = self.get_idx_for_time(time_in)
        ktop = self.get_top_layer(timestep)
        v = self.vals[:ktop + 1]
        el = self.elev[:ktop + 1]
        if return_depth:
            return np.max(el) - el[:], v
        else:
            return el, v

    def load_subdomains(self):
        self.subdomains = {}
        group = self.h['Geometry/Subdomains']
        for subdomain in group:
            dataset = self.h['Geometry/Subdomains/{0}/Cell Center Coordinate'.format(subdomain)]
            ncells = (np.shape(dataset))[0]
            x = np.array([dataset[i][0] for i in range(ncells)])
            y = np.array([dataset[i][1] for i in range(ncells)])
            self.subdomains[subdomain] = {'x': x, 'y': y}

    def find_computed_station_cell(self, xy):
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

    def get_time_series(self, time_start, time_end, metric, xy=None, dss_path=None):

        if dss_path is not None:
            dss = pyhecdss.DSSFile(self.sim_dss_fn)
            dt, v = dss_get(dss, dss_path)
        else:
            if xy is None:
                raise ValueError('xy or dss_path must be set for ResSimModelResults')
            else:
                i, subdomain_name = self.find_computed_station_cell(xy)

                if metric.lower() == 'flow':
                    dataset_name = 'Cell flow'
                    dataset = self.h['Results/Subdomains/{0}/{1}'.format(subdomain_name, dataset_name)]
                    v = np.array(dataset[:, i])
                    v = self.clean_computed(v)
                elif metric.lower() == 'elevation':
                    dataset_name = 'Water Surface Elevation'
                    dataset = self.h['Results/Subdomains/{0}/{1}'.format(subdomain_name, dataset_name)]
                    v = np.array(dataset[:])
                    v = self.clean_computed(v)
                elif metric.lower() == 'temperature':
                    dataset_name = 'Water Temperature'
                    dataset = self.h['Results/Subdomains/{0}/{1}'.format(subdomain_name, dataset_name)]
                    print('Results/Subdomains/{0}/{1}'.format(subdomain_name, dataset_name))
                    v = np.array(dataset[:, i])
                    v = self.clean_computed(v)
                elif metric.lower() == 'do':
                    dataset_name = 'Dissolved Oxygen'
                    dataset = self.h['Results/Subdomains/{0}/{1}'.format(subdomain_name, dataset_name)]
                    v = np.array(dataset[:, i])
                    v = self.clean_computed(v)
                elif metric.lower() == 'do_sat':
                    dataset_name = 'Water Temperature'
                    dataset = self.h['Results/Subdomains/{0}/{1}'.format(subdomain_name, dataset_name)]
                    vt = np.array(dataset[:, i])
                    dataset_name = 'Dissolved Oxygen'
                    dataset = self.h['Results/Subdomains/{0}/{1}'.format(subdomain_name, dataset_name)]
                    vdo = np.array(dataset[:, i])
                    vt = self.clean_computed(vt)
                    vdo = self.clean_computed(vdo)
                    v = self.calc_computed_dosat(vt, vdo)

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
        return self.t_computed[istart:iend], v[istart:iend]

    @staticmethod
    def clean_missing(indata):
        indata[indata == -901.] = np.nan
        return indata

    @staticmethod
    def clean_computed(indata):
        indata[indata == -9999.] = np.nan
        return indata

    @staticmethod
    def calc_computed_dosat(vtemp, vdo):
        v = np.zeros_like(vtemp)
        for j in range(len(v)):
            if np.isnan(vtemp[j]) or np.isnan(vdo[j]):
                v[j] = np.nan
            else:
                v[j] = do_saturation(vtemp[j], vdo[j])
        return v

    @staticmethod
    def calc_observed_dosat(ttemp, vtemp, tdo, vdo):
        v = np.zeros_like(vtemp)
        for j in range(len(v)):
            if np.isnan(vtemp[j]) or np.isnan(vdo[j]):
                v[j] = np.nan
            else:
                v[j] = do_saturation(vtemp[j], vdo[j])
        return ttemp, v

    def load_time(self):
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
        this_subdomain = self.subdomain_name if alt_subdomain_name is None else alt_subdomain_name
        cell_center_xy = self.h['Geometry/Subdomains/' + this_subdomain + '/Cell Center Coordinate']
        self.ncells = (np.shape(cell_center_xy))[0]
        self.elev = np.array([cell_center_xy[i][2] for i in range(self.ncells)])
        elev_ts = self.h['Results/Subdomains/' + this_subdomain + '/Water Surface Elevation']
        self.elev_ts = np.array([elev_ts[i] for i in range(self.nt)])

    def get_top_layer(self, timestep):
        elev = self.elev_ts[timestep]
        for k in range(len(self.elev) - 1):
            cell_z = self.elev[k]  # layer midpoint
            cell_z1 = self.elev[k + 1]  # layer above midpoint
            top_of_cell_z = 0.5 * (cell_z + cell_z1)
            if elev < top_of_cell_z:
                break
        return k

    def load_results(self, t_in, metrc, alt_subdomain_name=None):
        this_subdomain = self.subdomain_name if alt_subdomain_name is None else alt_subdomain_name

        timestep = self.get_idx_for_time(t_in)
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
            vals = self.calc_computed_dosat(vt, vdo)
        self.vals = vals

    def get_idx_for_time(self, t_in):
        ttmp = t_in.toordinal() + float(t_in.hour) / 24. + float(t_in.minute) / (24. * 60.) - self.t_offset
        min_diff = np.min(np.abs(self.t - ttmp))
        tol = 1. / (24. * 60.)  # 1 minute tolerance
        timestep = np.where((np.abs(self.t - ttmp) - min_diff) < tol)[0][0]
        if min_diff > 1.:
            print('Error: nearest time step > 1 day away')
        return timestep

    def make_output_dir(self):
        base_name = self.simulation_name.lower()
        if not os.path.exists(base_name):
            os.mkdir(base_name)
        self.output_dir = os.path.join(base_name, self.trial_name)
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

    def plot_obs_data(self, ax, vobs, dep):
        self.obs_data = vobs
        self.obs_depth = dep
        ax.plot(vobs, dep, '-.', zorder=4, label='Observed')
        ax.grid(zorder=0)
        ax.invert_yaxis()

    def load_obs_data(self, vobs, dep):
        self.obs_data = vobs
        self.obs_depth = dep

    def plot_modeled_data(self, ax, t_in, metrc, show_legend=True, show_xlabel=True, show_ylabel=True):
        timestep = self.get_idx_for_time(t_in)
        ktop = self.get_top_layer(timestep)
        v = self.vals[:ktop + 1]
        el = self.elev[:ktop + 1]
        dep = np.max(el) - el[:]
        self.comp_data = v
        self.comp_depth = dep
        ax.plot(v, dep, '-.', zorder=4, label='Modeled')
        if metrc == 'temperature':
            ax.set_xlim([0, 30])
            xlab = r'Temperature ($^\circ$C)'
        elif metrc == 'diss_oxy':
            ax.set_xlim([0, 14])
            xlab = 'Dissolved oxygen (mg/L)'
        elif metrc == 'do_sat':
            ax.set_xlim([0, 130])
            xlab = 'Dissolved oxygen saturation (%)'
        if show_legend:
            plt.legend(loc='lower right')
        if show_xlabel:
            ax.set_xlabel(xlab)
        if show_ylabel:
            ax.set_ylabel('Depth (ft)')
        ttl_str = dt.datetime.strftime(self.t_data, '%d %b %Y')
        xbufr = 0.05
        ybufr = 0.05
        xl = ax.get_xlim()
        yl = ax.get_ylim()
        xtext = xl[0] + xbufr * (xl[1] - xl[0])
        ytext = yl[1] - ybufr * (yl[1] - yl[0])
        ax.text(xtext, ytext, ttl_str, ha='left', va='top', size=10, bbox=dict(boxstyle='round', facecolor='w',
                                                                               alpha=0.35), zorder=10)

    def save_fig(self, year, metrc):
        fig_name = 'profiles_{0}_{1}_{2}.png'.format(metrc, year, self.subdomain_name.replace(' ', '_'))
        plt.savefig(os.path.join(self.output_dir, fig_name), bbox_inches='tight')
        print('saved:', os.path.join(self.output_dir, fig_name))

    def open_error_stats(self, metrc):
        output_fname = 'profile_error_stats_{0}_{1}.txt'.format(metrc, self.subdomain_name.replace(' ', '_'))
        error_stats_fname = os.path.join(self.output_dir, output_fname)
        self.error_file = open(error_stats_fname, 'w')
        self.error_file.write('Year,RMSE\n')

    def calc_error_stats(self, current_year):
        rmse = self.rmse()
        print('RMSE', rmse)
        if self.hold_year != current_year:
            self.profile_rmse = rmse
        else:
            self.profile_rmse += rmse
        self.hold_year = current_year

    def rmse(self):
        """To be called after plot_obs_data and plot_ so that """
        interp_fnct = interpolate.interp1d(self.comp_depth, self.comp_data,
                                           fill_value=(self.comp_data[0], self.comp_data[-1]), bounds_error=False)
        sq_error = 0.
        n = len(self.obs_depth)
        for j in range(n):
            d = self.obs_depth[j]
            vo = self.obs_data[j]
            vc = interp_fnct(d)
            sq_error += (vc - vo) ** 2
        return np.sqrt(sq_error / n)

    def output_error_stats(self, year, n_profiles):
        avg_rmse = self.profile_rmse / n_profiles
        print(year, 'Average RMSE', avg_rmse)
        self.error_file.write('{0},{1}\n'.format(year, avg_rmse))

    def close_error_stats(self):
        self.error_file.close()


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
    factor = n_profiles / 4.
    if factor < 1:
        return 1, n_profiles
    else:
        return math.ceil(factor), 4


def plot_temp_profile_comparison(simulation_drct, trial_name, metric, obs_file_stub=None, reservoir_name=None,
                                 output_path=''):
    """

    :param simulation_drct:
    :param trial_name:
    :param metric: one of ['temperature', 'diss_oxy', 'do_sat']
    :param obs_file_stub:
    :param reservoir_name:
    :param output_path:
    :return:
    """

    observed_data_drct = 'profile_data'
    res_name = 'Lake Mendocino' if reservoir_name is None else reservoir_name  # reservoir name

    mresults = ResSimModelResults(simulation_drct, trial_name, res_name)

    syear = mresults.stime.year
    eyear = mresults.etime.year
    if mresults.etime.month == 1:
        eyear -= 1

    mresults.open_error_stats(metric)
    for yr in range(syear, eyear + 1):
        if obs_file_stub is None:
            if metric == 'temperature':
                observed_data_file_name = 'wt_data_{0}.txt'.format(yr)
            elif metric == 'diss_oxy':
                observed_data_file_name = 'do_data_{0}.txt'.format(yr)
            elif metric == 'do_sat':
                observed_data_file_name = 'dosat_data_{0}.txt'.format(yr)
        else:
            observed_data_file_name = obs_file_stub + '{0}.txt'.format(yr)

        obs_times, obs_temps, obs_depths = read_observed(os.path.join(observed_data_drct, observed_data_file_name))
        nof_profiles = len(obs_times)

        fig = plt.figure(figsize=(12, 8))
        subplot_rows, subplot_cols = get_subplot_config(nof_profiles)

        for j in range(nof_profiles):
            pax = fig.add_subplot(subplot_rows, subplot_cols, j + 1)
            to = obs_times[j]
            wto = obs_temps[j]
            depo = obs_depths[j]
            mresults.plot_obs_data(pax, wto, depo)
            mresults.load_results(to[0], metric)
            lflag, xflag, yflag = get_plot_label_masks(j, nof_profiles, subplot_rows, subplot_cols)
            mresults.plot_modeled_data(pax, to[0], metric, show_legend=lflag,
                                       show_xlabel=xflag, show_ylabel=yflag)
            mresults.calc_error_stats(yr)
        mresults.save_fig(yr, metric)
        plt.close('all')
        mresults.output_error_stats(yr, nof_profiles)
    mresults.close_error_stats()


def read_observed_data(obs_data_filepath, metric='temperature', dss_path_list=None):
    if obs_data_filepath.lower().endswith('dss'):
        # dss paths should match the compare_var!
        return read_observed_DSS(obs_data_filepath, dss_path_list)
    else:
        return read_observed(obs_data_filepath)


# def plot_time_series(location_name,):


def plot_multiple_alternatives(h5_alts, obs_data_filepath, metric='temperature', dss_path_list=None,
                               title=None, rmse=True, save_filepath=None):
    """
    :param h5_alternatives:
    :param obs_data_filepath:
    :param metric:
    :param dss_path_list:
    :param title:
    :return:
    """
    obs_times, obs_temps, obs_depths = read_observed_data(obs_data_filepath, metric, dss_path_list)
    nof_profiles = len(obs_times)

    fig = plt.figure(figsize=(12, 8))
    subplot_rows, subplot_cols = get_subplot_config(nof_profiles)

    alts = {h5a['alt_label']: ResSimModelResults(h5a['sim_path'], h5a['alt_name'], h5a['reservoir_name'],
                                                 h5a['h5_filepath']) for h5a in h5_alts}

    names_labels = ['Observed', ]

    for j in range(nof_profiles):
        pax = fig.add_subplot(subplot_rows, subplot_cols, j + 1)
        to = obs_times[j]
        wto = obs_temps[j]
        depo = obs_depths[j]

        rmse_labels = ['Obs']
        for k, (alabel, mr) in enumerate(alts.items()):
            if j == 0:
                names_labels.append(alabel)
            if k == 0:
                mr.plot_obs_data(pax, wto, depo)
            else:
                mr.load_obs_data(wto, depo)
            mr.load_results(to[0], metric)
            lflag, xflag, yflag = get_plot_label_masks(j, nof_profiles, subplot_rows, subplot_cols)
            mr.plot_modeled_data(pax, to[0], metric, show_legend=False,
                                 show_xlabel=xflag, show_ylabel=yflag)
            rmse_labels.append('{:.2f}'.format(mr.rmse()))

        if j == 0:
            leg = pax.legend(names_labels, bbox_to_anchor=(0.0, 1.02), loc='lower left', fontsize='small')

        if rmse:
            pax.legend(rmse_labels, loc='lower right')
            if j == 0:
                pax.add_artist(leg)  # re-add name legend after matplotlib deletes

    if title is not None:
        plt.suptitle(title)
    if save_filepath is not None:
        plt.savefig(save_filepath, dpi=600, bbox_inches='tight')
    else:
        plt.show()


def batch_plot_temp_profile_comparison():
    simulation_directory = r'J:\ResSim_watersheds\ResSim-dev\base\RussianRiver\rss\2020.09.22-1300'
    # start_run = 414  # for redoing some of them
    # end_run = 415
    run_count = 1024
    for irun in range(run_count):
        # for irun in range(start_run, end_run):
        print('Processing run {0} of {1}'.format(irun + 1, run_count))
        trial_name = 'run{0}'.format(irun)
        plot_temp_profile_comparison(simulation_directory, trial_name, temperature_only=True)


def obs_model_profile_plot(ax, obs_elev, obs_value, model_elev, model_value, metric, dt_profile,
                           show_legend=False, show_xlabel=False, show_ylabel=False):
    # deal with depths vs. elevations?

    # observed
    # ax.plot(obs_value, obs_elev, '-.',zorder=4, label='Observed')
    # if len(obs_value) <= 30:
    #     ax.plot(obs_value, obs_elev, marker='o', zorder=4, label='Observed')
    # else:
    ax.plot(obs_value, obs_elev, zorder=4, label='Observed')
    ax.grid(zorder=0)
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
        ax.set_ylabel('Depth (ft)')
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
    n_profiles_per_page = 12

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

            # break profile indices into page groups
            prof_indices = list(range(nof_profiles))
            n = n_profiles_per_page
            page_indices = [prof_indices[i * n:(i + 1) * n] for i in range((len(prof_indices) + n - 1) // n)]

            fig_names = []
            stats = []
            prof_start = 0
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
                    obs_elev = obs_depths[j]

                    # modeled/computed
                    # model_elev, model_val = mr.get_profile(dt_profile[j], metric, name=reservoir, return_depth=use_depth)
                    # lflag, xflag, yflag = get_plot_label_masks(i, len(pgi), subplot_rows, subplot_cols)
                    # obs_model_profile_plot(pax, obs_elev, obs_val, model_elev, model_val, metric, dt_profile[j],
                    #                        show_legend=lflag, show_xlabel=xflag, show_ylabel=yflag)
                    model_elev, model_val = mr.get_profile(dt_profile[0], metric, name=reservoir,
                                                           return_depth=use_depth)
                    lflag, xflag, yflag = get_plot_label_masks(i, len(pgi), subplot_rows, subplot_cols)
                    obs_model_profile_plot(pax, obs_elev, obs_val, model_elev, model_val, metric, dt_profile[0],
                                           show_legend=lflag, show_xlabel=xflag, show_ylabel=yflag)

                    # why do stats here? because it may be time consuming to pull observations and model data a second time
                    stats.append(series_stats(model_elev, model_val, obs_elev, obs_val,
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


mo_str_3 = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


def plot_time_series(mr, observed_data_meta_file, fig_path_stub, rss_sim_name, rss_alt_name, region_name,
                     temperature_only=False):
    # parse observed data file
    stations, obs_ts_dss_file = read_obs_ts_meta_file(observed_data_meta_file)
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
        if region != region_name.lower():
            continue
        if not temperature_only or metric.lower() == 'temperature':
            obs_dates, obs_vals = read_observed_ts_data(ts_data_path, station, metric)
            # print('Getting computed data')
            if data['dss_computed'] is None:
                comp_dates, comp_vals = mr.get_time_series(None, None, metric, xy=[x, y])
            else:
                comp_dates, comp_vals = mr.get_time_series(None, None, metric, dss_path=data)
            # print('Making plot')

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

        # break stats into years, months
        # find n years
        n_years = obs_dates[-1].year - obs_dates[0].year + 1
        stats = {}
        stats_months = {}
        for yr in range(obs_dates[0].year, obs_dates[-1].year + 1):
            stdate = dt.datetime(yr, 1, 1)
            enddate = dt.datetime(yr + 1, 1, 1)
            # at some point should add metric to key, which is used as label
            print(station, metric, yr)
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

        station_results.append([station, x, y, metric, fig_name, stats, stats_months])

    return station_results


def get_XML_template():
    XML_template = {}
    XML_template['start'] = """<?xml version="1.0" encoding="UTF-8"?>
    <USBR_Automated_Report Date="$$REPORT_DATE$$" SimulationName="$$SIMULATION_NAME$$">
       <!--This section contains the required text and image paths for the cover page.-->
       <Cover_Page>
          <Title>DRAFT Temperature Validation Summary Report</Title>
          <Title>Shasta / Keswick</Title>
          <Contact_Info>TODO: Contact Info - May need more layers</Contact_Info>
          <Pictures>TODO: Path to any pictures</Pictures>
       </Cover_Page>
    """
    XML_template['reservoir'] = """   <Report_Element Order="$$ORDER$$" Element="Reservoir_Profile">
          <Reservoir_Profiles Reservoir="$$RESERVOIR_NAME$$">
    $$RESERVOIR_FIGS$$
          </Reservoir_Profiles>
       </Report_Element>
    """
    XML_template['reservoir_fig'] = """         <Profile_Image FigureNumber="$$FIG_NUM$$" FigureDescription="$$FIG_DESCRIPTION$$">$$FIG_FILENAME$$</Profile_Image>
    """

    XML_template['time_series'] = """   <Report_Element Order="$$ORDER$$" Element="Output">
          <Output_Temp_Flow Location="$$TS_NAME$$">
    $$TS_FIGURE$$
    $$TS_TABLE$$
          </Output_Temp_Flow>
       </Report_Element>
    """

    XML_template['time_series_fig'] = """         <Output_Image FigureNumber="$$FIG_NUM$$" FigureDescription="$$FIG_DESCRIPTION$$">$$FIG_FILENAME$$</Output_Image>
    """

    XML_template['time_series_table'] = """         <Output_Table TableNumber="$$TABLE_NUM$$" TableDescription="$$TABLE_DESCRIPTION$$" TableType="statistics">
    $$TABLE_ELEMENTS$$
             </Output_Table>
    """
    XML_template['time_series_table_col'] = """            <Column Column_Name="$$TABLE_COL_NAME$$">
    $$TABLE_ROWS$$
                </Column>
    """
    XML_template['end'] = """</USBR_Automated_Report>
    """

    return XML_template


def get_reservoir_description(region):
    if region.lower() == 'shasta':
        return "Shasta Reservoir Temperature Profiles Near Dam"
    elif region.lower() == 'keswick':
        return "Keswick Reservoir Temperature Profiles Near Dam"
    elif region.lower() == 'uppersac':
        return "Sacramento River Above Clear Creek"

def get_ts_description(region):
    if region.lower() == 'shasta':
        return "Shasta Reservoir Outflow and Outflow Temperature"
    elif region.lower() == 'keswick':
        return "Keswick Reservoir Outflow and Outflow Temperature"
    elif region.lower() == 'uppersac':
        return "Sacramento River Above Clear Creek"


def XML_reservior(profile_stats, XML_class, region_name, n_element=0, n_fig=0, n_table=0):
    # only writing one reservoir XLM section here

    XML = ''

    #res name, year, list of pngs, stats dictionary
    for subdomain_name, figure_sets in profile_stats.items():
        # XML += XML_class.make_Reservoir_Group_header(subdomain_name) #TODO CHANGE GROUP ORDER
        if len(figure_sets) > 0:
            for ps in figure_sets:
                reservoir, yr, fig_names, stats = ps
                # fig_names = [os.path.join('..','Images', n) for n in fig_names]
                subgroup_desc = get_reservoir_description(region_name)
                XML += XML_class.make_ReservoirSubgroup_lines(reservoir,fig_names, subgroup_desc, yr)

    return XML, n_element, n_fig, n_table #TODO these are all extra, do we need them?

    # XML = ""
    # for subdomain_name, figure_sets in profile_stats.items():
    #     if len(figure_sets) > 0:
    #         XML_figs = ""
    #         for ps in figure_sets:
    #             reservoir, yr, fig_names, stats = ps
    #             for i, fn in enumerate(fig_names):
    #                 XML_f = copy.copy(get_XML_template()['reservoir_fig'])
    #                 XML_f = XML_f.replace("$$FIG_NUM$$", str(n_fig))
    #                 n_fig += 1
    #                 XML_f = XML_f.replace("$$FIG_DESCRIPTION$$", reservoir + ' %i: %i of %i' % (yr, i, len(fig_names)))
    #                 XML_f = XML_f.replace("$$FIG_FILENAME$$", os.path.join('..', 'Images', fn))
    #                 XML_figs = XML_figs + XML_f
    #
    #         XML_r = copy.copy(get_XML_template()['reservoir'])
    #         XML_r = XML_r.replace("$$ORDER$$", str(n_element))
    #         n_element += 1
    #         XML_r = XML_r.replace("$$RESERVOIR_NAME$$", reservoir)
    #         XML_r = XML_r.replace("$$RESERVOIR_FIGS$$", XML_figs)
    #     else:
    #         XML_r = ""  # reservoir plots requested, but no observation data found
    #     XML = XML + XML_r
    #
    # return XML, n_element, n_fig, n_table


def XML_time_series(ts_results, XML_class, region_name, n_element=0, n_fig=0, n_table=0):
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
        station, x, y, metric, fig_name, stats, stats_mo = ts
        # fig_name = os.path.join('..','Images', fig_name)
        subgroup_desc = get_ts_description(region_name)
        XML += XML_class.make_TS_Subgroup_lines(station, fig_name, subgroup_desc)
        XML += XML_class.make_TS_Tables_lines(station, stats, stats_mo, stats_ordered, stats_labels)
        XML += '        </Report_Subgroup>\n'
        # XML += '     </Report_Group>\n'

    # for ts in ts_results:
    #     station, x, y, metric, fig_name, stats, stats_mo = ts
    #
    #     XML_f = copy.copy(get_XML_template()['time_series_fig'])
    #     XML_f = XML_f.replace("$$FIG_NUM$$", str(n_fig))
    #     n_fig += 1
    #     XML_f = XML_f.replace("$$FIG_DESCRIPTION$$", station)
    #     XML_f = XML_f.replace("$$FIG_FILENAME$$", os.path.join('..', 'Images', fig_name))
    #
    #     # error stats table by year-column
    #     n_table, XML_t_error = XML_error_stats_table(n_table, station, stats, stats_ordered, stats_labels)
    #     n_table, XML_t_month_means = XML_mean_monthly_stats_table(n_table, station, stats_mo)
    #
    #     XML_ts = copy.copy(get_XML_template()['time_series'])
    #     XML_ts = XML_ts.replace("$$ORDER$$", str(n_element))
    #     n_element += 1
    #     XML_ts = XML_ts.replace("$$TS_NAME$$", station)
    #     XML_ts = XML_ts.replace("$$TS_FIGURE$$", XML_f)
    #     XML_ts = XML_ts.replace("$$TS_TABLE$$", XML_t_error + XML_t_month_means)
    #
    #     XML += XML_ts

    return XML, n_element, n_fig, n_table


def XML_error_stats_table(n_table, station, stats, stats_ordered, stats_labels):
    XML_t = copy.copy(get_XML_template()['time_series_table'])
    XML_t = XML_t.replace("$$TABLE_NUM$$", str(n_table))
    n_table += 1
    XML_t = XML_t.replace("$$TABLE_DESCRIPTION$$", station)

    XML_cols = ""
    for j, colname in enumerate(stats.keys()):

        XML_rows = ""
        for i, st in enumerate(stats_ordered):
            row = r"""               <Row Row_Order="$$ORDER$$" Row_name="$$STATS_LABEL$$">$$STAT_NUM$$</Row>
"""
            row = row.replace("$$ORDER$$", str(i))
            if st == 'COUNT':
                row = row.replace("$$STAT_NUM$$", '%i' % stats[colname][st])
            else:
                row = row.replace("$$STAT_NUM$$", '%2f' % stats[colname][st])
            row = row.replace("$$STATS_LABEL$$", stats_labels[st])
            XML_rows += row

        col = copy.copy(get_XML_template()['time_series_table_col'])
        col = col.replace("$$TABLE_COL_NAME$$", colname)
        col = col.replace("$$TABLE_ROWS$$", XML_rows)

        XML_cols += col

    XML_t = XML_t.replace("$$TABLE_ELEMENTS$$", XML_cols)
    return n_table, XML_t


def XML_mean_monthly_stats_table(n_table, station, stats_mo):
    XML_t = copy.copy(get_XML_template()['time_series_table'])
    XML_t = XML_t.replace("$$TABLE_NUM$$", str(n_table))
    n_table += 1
    XML_t = XML_t.replace("$$TABLE_DESCRIPTION$$", station)

    XML_cols = ""

    col_names = list(stats_mo.keys())

    for j, colname in enumerate(sorted(col_names)):
        stats_col = stats_mo[colname]
        XML_rows = ""
        for mo in range(1, 13):
            row = r"""               <Row Row_Order="$$ORDER$$" Row_name="$$STATS_LABEL$$">$$STAT_NUM$$</Row>
"""
            row = row.replace("$$ORDER$$", str(mo))
            row = row.replace("$$STATS_LABEL$$", mo_str_3[mo - 1])
            if stats_col[mo_str_3[mo - 1]] is None:
                row = row.replace("$$STAT_NUM$$", 'nan')
            else:
                row = row.replace("$$STAT_NUM$$", '%2f' % stats_col[mo_str_3[mo - 1]])
            XML_rows += row

        col = copy.copy(get_XML_template()['time_series_table_col'])
        col = col.replace("$$TABLE_COL_NAME$$", colname)
        col = col.replace("$$TABLE_ROWS$$", XML_rows)
        XML_cols += col

    XML_t = XML_t.replace("$$TABLE_ELEMENTS$$", XML_cols)
    return n_table, XML_t


def XML_write(output_path, profile_stats, ts_results, report_name, region_name):

    XML = XMLReport.makeXMLReport("USBRAutomatedReportOutput.xml")

    n_element = 1
    n_fig = 1
    n_table = 1
    XML_res = ""
    XML_ts = ""
    XML_Groupheader = XML.make_Reservoir_Group_header(region_name) #TODO CHANGE GROUP ORDER

    if len(profile_stats) > 0:
        XML_res, n_element, n_fig, n_table = XML_reservior(profile_stats, XML, region_name, n_element, n_fig, n_table)
    if len(ts_results) > 0:
        XML_ts, n_element, n_fig, n_table = XML_time_series(ts_results,XML, region_name,  n_element, n_fig, n_table)


    XML.write_Reservoir(region_name, XML_Groupheader, XML_res, XML_ts)


    # XML_start = copy.copy(get_XML_template()['start'])
    # XML_start = XML_start.replace("$$SIMULATION_NAME$$", report_name)
    # XML_start = XML_start.replace("$$REPORT_DATE$$", dt.datetime.now().strftime('%Y-%m-%d %H:%M'))
    # XML_end = copy.copy(get_XML_template()['end'])

    # XML = XML_start + XML_res + XML_ts + XML_end
    # with open(os.path.join(output_path, XML_fname), 'w') as fp:
    #     fp.write(XML)


XML_fname = 'USBRAutomatedReportOutput.xml'


def clean_output_dir(dir_name):
    files_in_directory = os.listdir(dir_name)
    filtered_files = [file for file in files_in_directory if file.endswith(".png")]
    for file in filtered_files:
        path_to_file = os.path.join(dir_name, file)
        os.remove(path_to_file)
    # xml_file = os.path.join(dir_name, XML_fname)
    # if os.path.exists(xml_file):
    #     os.remove(xml_file)


# def generate_report_plots_ResSim(simulation_path, alternative, observed_data_directory, temperature_only=False,
#                                  use_depth=False, clean_out_dir=True):
#     output_path = r"Z:\USBR\test"  # WAT will start path where we need to write
#     # output_path = "."  # WAT will start path where we need to write
#
#     if clean_out_dir:
#         clean_output_dir(output_path)
#
#     profile_meta_file = os.path.join(observed_data_directory, "Profile_stations.txt")
#     ts_meta_file = os.path.join(observed_data_directory, "TS_stations.txt")
#
#     # we should only need to generate a single instance of ResSimModelResults; while we need to pass in a
#     # subdomain/reservoir for legacy purposes at this point, and subdomain/reservoir can be passed to through
#     # to the get_profile command
#     profile_subdomains, _ = read_obs_profile_meta_file(profile_meta_file)
#     mr = ResSimModelResults(simulation_path, alternative)
#
#     profile_stats = {}
#     for subdomain, meta in profile_subdomains.items():
#         if meta['metric'].lower() == 'temperature' or not temperature_only:
#             profile_stats[subdomain] = plot_profiles(mr, meta['metric'], observed_data_directory, subdomain,
#                                                      output_path,
#                                                      use_depth=use_depth)
#
#     ts_results = plot_time_series(mr, ts_meta_file, output_path, simulation_path, alternative,
#                                   temperature_only=temperature_only)
#
#     _, sim_name = os.path.split(simulation_path)
#     report_name = " : ".join([sim_name, alternative])
#     XML_write(output_path, profile_stats, ts_results, report_name)

def generate_region_plots_ResSim(simulation_path, alternative, observed_data_directory, region_name, temperature_only=False,
                                     use_depth=False, clean_out_dir=False):
    # output_path = r"Z:\USBR\test"  # WAT will start path where we need to write
    output_path = "."  # WAT will start path where we need to write
    images_path = os.path.join(output_path, '..', "Images")
    if not os.path.exists(images_path):
        os.makedirs(images_path)

    if clean_out_dir:
        clean_output_dir(images_path)


    profile_meta_file = os.path.join(observed_data_directory, "Profile_stations.txt")
    ts_meta_file = os.path.join(observed_data_directory, "TS_stations.txt")


    # we should only need to generate a single instance of ResSimModelResults; while we need to pass in a
    # subdomain/reservoir for legacy purposes at this point, and subdomain/reservoir can be passed to through
    # to the get_profile command
    profile_subdomains, _ = read_obs_profile_meta_file(profile_meta_file)
    mr = ResSimModelResults(simulation_path, alternative)

    profile_stats = {}
    for subdomain, meta in profile_subdomains.items():
        if region_name.lower() == meta['region'].lower():
            if meta['metric'].lower() == 'temperature' or not temperature_only:
                profile_stats[subdomain] = plot_profiles(mr, meta['metric'], observed_data_directory, subdomain,
                                                         images_path,
                                                         use_depth=use_depth)

    ts_results = plot_time_series(mr, ts_meta_file, images_path, simulation_path, alternative, region_name,
                                  temperature_only=temperature_only)

    _, sim_name = os.path.split(simulation_path)
    report_name = " : ".join([sim_name, alternative])
    XML_write(output_path, profile_stats, ts_results, report_name, region_name)

def generate_region_plots_W2(simulation_path, alternative, observed_data_directory, region_name, temperature_only=False,
                                     use_depth=False, clean_out_dir=False):
    # output_path = r"Z:\USBR\test"  # WAT will start path where we need to write
    output_path = "."  # WAT will start path where we need to write
    images_path = os.path.join(output_path, '..', "Images")
    if not os.path.exists(images_path):
        os.makedirs(images_path)

    if clean_out_dir:
        clean_output_dir(images_path)


    profile_meta_file = os.path.join(observed_data_directory, "Profile_stations.txt")
    ts_meta_file = os.path.join(observed_data_directory, "TS_stations.txt")


    # we should only need to generate a single instance of ResSimModelResults; while we need to pass in a
    # subdomain/reservoir for legacy purposes at this point, and subdomain/reservoir can be passed to through
    # to the get_profile command
    profile_subdomains, _ = read_obs_profile_meta_file(profile_meta_file)
    mr = W2ModelResults(simulation_path, alternative, region_name)

    profile_stats = {}
    for subdomain, meta in profile_subdomains.items():
        if region_name.lower() == meta['region'].lower():
            if meta['metric'].lower() == 'temperature' or not temperature_only:
                profile_stats[subdomain] = plot_profiles(mr, meta['metric'], observed_data_directory, subdomain,
                                                         images_path,
                                                         use_depth=use_depth)

    # ts_results = plot_time_series(mr, ts_meta_file, images_path, simulation_path, alternative, region_name,
    #                               temperature_only=temperature_only)
    ts_results = []

    _, sim_name = os.path.split(simulation_path)
    report_name = " : ".join([sim_name, alternative])
    XML_write(output_path, profile_stats, ts_results, report_name, region_name)



if __name__ == '__main__':

    # mr = W2ModelResults(r'C:\Users\benjamin\Downloads\Short_2014_Shasta.dss',None)

    # exit()


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

    reg_info = RU.find_rptrgn(simulation_name)
    print('SYSARG', sys.argv)
    print('REGINFO', reg_info)
    try:
        print(reg_info[model_alt_name.replace(' ', '_')]['plugin'], plugin_name)
        region_names = reg_info[model_alt_name.replace(' ', '_')]['regions']
    except:
        exit()
    for region_name in region_names:

        if plugin_name == 'ResSim':

            # ben/scott test values
            # generate_region_plots_ResSim(r'D:\Work2021\USBR\UpperSacTemperature-demo.2021.06.07.jfd\UpperSacTemperature-demo\rss\test02_longModel-2015', 'test02_LM-0',
            #                              r'D:\Work2021\USBR\observed_data_Shasta', 'Keswick',
            #                              temperature_only=True, use_depth=True)
            # exit()

            generate_region_plots_ResSim(simulation_directory, alternative_name, obs_data_path, region_name,
                                         temperature_only=True, use_depth=True)
        elif plugin_name == 'CeQualW2':
            generate_region_plots_W2(simulation_directory, alternative_name, obs_data_path, region_name,
                                     temperature_only=True, use_depth=True)
        else:
            print('WAT model "%s" not understood!' % sys.argv[0])
    exit()
