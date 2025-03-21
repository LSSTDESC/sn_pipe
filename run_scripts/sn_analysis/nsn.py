#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 13:48:17 2025

@author: philippe.gris@clermont.in2p3.fr
"""
from optparse import OptionParser
from sn_tools.sn_utils import get_val
import pandas as pd
from sn_tools.sn_io import load_DataFrame
from sn_tools.sn_utils import n_z
from sn_analysis.sn_selection import selection_criteria
from sn_analysis.sn_calc_plot import select
import numpy as np


def effi(grp, sellist, bins=np.arange(0.005, 1.11, 0.01)):

    nsn_ref = n_z(grp, 'zmeas', bins=bins)

    # select data
    sel = select(grp, sellist)

    nsn_sel = n_z(sel, 'zmeas', bins=bins)

    """
    print(nsn_ref)
    print(nsn_sel)
    """

    df_t = nsn_ref.merge(nsn_sel, left_on=['zmeas'],
                         right_on=['zmeas'], suffixes=['_ref', '_sel'])

    effi = df_t['nsn_sel']/df_t['nsn_ref']
    err_effi = np.sqrt(df_t['nsn_sel']*(1.-effi))/df_t['nsn_ref']

    df_effi = pd.DataFrame(df_t['zmeas'].to_list(), columns=['zmeas'])
    df_effi['effi'] = effi
    df_effi['err_effi'] = err_effi

    df_effi = df_effi.fillna(0)
    # print(df_effi)

    return df_effi


parser = OptionParser(
    'Script to estimate the number of supernovae estimated fromm observing efficiency')

parser.add_option("--dataDir", type=str,
                  default='../Output_SN_sigmaInt_0.0_Hounsell',
                  help="data dir[%default]")
parser.add_option("--zType", type=str,
                  default='spectroz', help="z type (spectroz/photz) [%default]")
parser.add_option("--listFields", type=str,
                  default='COSMOS,CDFS,XMM-LSS,ELAISS1,EDFSa,EDFSb',
                  help=" [%default]")
parser.add_option("--fieldType", type=str,
                  default='DDF',
                  help="field type [%default]")
parser.add_option('--runType', type=str,
                  default='spectroz',
                  help='run type  [%default]')
parser.add_option('--timescale', type=str,
                  default='year',
                  help='timescale of the files to process [%default]')
parser.add_option('--timeslots', type=str,
                  default='1-10',
                  help='time slot (season or year) to process [%default]')
parser.add_option('--dataType', type=str,
                  default='DataFrame',
                  help='data type [%default]')
parser.add_option('--dbList', type=str,
                  default='list_OS.csv',
                  help='list of OS to process [%default]')
parser.add_option('--selconfig', type=str,
                  default='G10_JLA',
                  help='selection [%default]')

opts, args = parser.parse_args()

dataDir = opts.dataDir
fieldType = opts.fieldType
runType = opts.runType
timeslots = opts.timeslots
timescale = opts.timescale
timeslots = get_val(timeslots)
dataType = opts.dataType
dbList = opts.dbList
selconfig = opts.selconfig

# OS to process
dbNames = pd.read_csv(dbList, comment='#')

# load selection
sellist = selection_criteria()[selconfig]

# load data
for i, row in dbNames.iterrows():
    tt = 'load_{}(\'{}\',\'{}\',\'{}\',\'{}\',{},\'{}\')'.format(
        dataType, dataDir, row['dbName'], runType,
        timescale, timeslots, fieldType)
    print('rrr', tt)
    df = eval(tt)
    print(len(df))


idx = df['field'] == 'COSMOS'
# idx &= df['healpixID'] == 109032.
df = df[idx]
hpix = df['season'].unique()

effis = df.groupby(['healpixID', 'season']).apply(lambda x: effi(x, sellist))

print(effis)
