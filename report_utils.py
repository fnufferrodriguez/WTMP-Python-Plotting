'''
Created on 6/10/2021
@author: scott
@organization: Resource Management Associates
@contact: scott@rmanet.com
@note:
'''

import os

def find_rptrgn(simulation_name, studyfolder):
    #find the rpt file go up a dir, reports, .rptrgn
    rptrgn_file = os.path.join(studyfolder, 'reports', '{0}.rptrgn'.format(simulation_name.replace(' ', '_')))
    # rptrgn_file = os.path.join('..', '{0}.rptrgn'.format(simulation_name.replace(' ', '_')))
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

