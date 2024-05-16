#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 08:24:50 2024

@author: gris@clermont.in2p3.fr
"""

from optparse import OptionParser
import pandas as pd
import glob
from sn_tools.sn_utils import n_z
import matplotlib.pyplot as plt


def load_data(dataDir, dbName, zType, fieldType):

    fullPath = '{}/{}/{}_{}'.format(dataDir, dbName, fieldType, zType)

    fis = glob.glob('{}/*.hdf5'.format(fullPath))

    df = pd.DataFrame()

    for fi in fis:
        dd = pd.read_hdf(fi)
        df = pd.concat((df, dd))

    return df


parser = OptionParser()

parser.add_option("--dataDir", type=str, default='../Output_SN_WFD_sigmaInt_0.0_Hounsell_z_smflux_G10_JLA',
                  help="data directory [%default]")

parser.add_option("--dbList", type=str, default='list_OS_new.csv',
                  help="db to process [%default]")
parser.add_option("--fieldType", type=str, default='WFD',
                  help="field type to process [%default]")

parser.add_option("--norm_factor", type=int, default=30,
                  help="simulation normalisation factor [%default]")
opts, args = parser.parse_args()

dataDir = opts.dataDir
dbList = opts.dbList
fieldType = opts.fieldType
norm_factor = opts.norm_factor

thelist = pd.read_csv(dbList, comment='#')

data_dict = {}
for i, ll in thelist.iterrows():
    dbName = ll['dbName']
    zType = ll['zType']
    tt = zType.split('sat_')[1]
    nn = '{}_{}'.format(dbName, tt)
    data_dict[nn] = load_data(dataDir, dbName, zType, fieldType)

fig, ax = plt.subplots()

for key, vals in data_dict.items():
    print('processing', key)
    nsn_tot = len(vals)
    idx = vals['remove_sat'] == 0
    sel = vals[idx]
    print(key, len(sel), nsn_tot-len(sel))
    nsn_z = n_z(sel, norm_factor=norm_factor)
    ax.plot(nsn_z['z'], nsn_z['nsn'])
    nsn_zb = n_z(vals[~idx], norm_factor=norm_factor)
    ax.plot(nsn_zb['z'], nsn_zb['nsn'], linestyle='dotted')
plt.show()
