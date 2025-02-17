#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 08:52:23 2025

@author: philippe.gris@clermont.in2p3.fr
"""

from optparse import OptionParser
import numpy as np
import pandas as pd
import yaml


def process_ddf(grp):

    print(len(grp))

    rr = grp.groupby(['night']).apply(
        lambda x: process_night(x))

    print(rr)
    return rr


def process_night(grp):

    bands = 'ugrizy'

    resdict = {}

    resdict['nvisits'] = [len(grp)]
    for b in bands:
        idx = grp['band'] == b
        sel = grp[idx]
        resdict[b] = [len(sel)]

    seq = []
    count = []
    grp = grp.sort_values(by='mjd')

    cb = 0
    for i, row in grp.iterrows():
        band = row['band']
        if len(seq) == 0:
            seq = [band]
        if band != seq[-1]:
            seq.append(band)
            count.append(cb)
            cb = 0
        cb += 1
    # add the last one
    count.append(cb)
    resdict['seq'] = [''.join(seq)]

    count = list(map(str, count))
    resdict['seq_visits'] = ['/'.join(count)]

    res = pd.DataFrame.from_dict(resdict)

    return res


parser = OptionParser(
    description='Script to study DDF visits on a nightly basis from pointings')

parser.add_option("--dbDir", type="str",
                  default='../DB_Files',
                  help="file directory [%default]")
parser.add_option("--dbName", type="str",
                  default='desc_ddf_v4.2.1_10yrs',
                  help="OS name [%default]")
parser.add_option("--ddf_list", type="str",
                  default="DD:COSMOS,DD:ECDFS,DD:EDFS_a,DD:EDFS_b,DD:ELAISS1,DD:XMM_LSS",
                  help="list of ddf [%default]")
parser.add_option("--config", type="str",
                  default="config.yaml",
                  help="config file [%default]")

opts, args = parser.parse_args()
# Load parameters
dbDir = opts.dbDir
dbName = opts.dbName
ddf_list = opts.ddf_list.split(',')
config = opts.config


fName = '{}/{}.npy'.format(dbDir, dbName)

obs = pd.DataFrame(np.load(fName))

print(len(obs))

idx = obs['target_name'].isin(ddf_list)

idx = obs['target_name'].isin(['DD:COSMOS'])
obs = obs[idx]
print(obs['target_name'].unique())

dd = obs.groupby(['target_name']).apply(
    lambda x: process_ddf(x)).reset_index()

print(dd)
ddf_exp = yaml.load(config)

print(ddf_exp)
