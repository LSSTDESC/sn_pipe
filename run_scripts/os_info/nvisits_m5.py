#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 10:10:05 2025

@author: philippe.gris@clermont.in2p3.fr
"""
from optparse import OptionParser
import pandas as pd
import numpy as np
from sn_tools.sn_io import checkDir


def load_data(dirFile, dbName, fieldlist):
    """
    Function to load the data

    Parameters
    ----------
    dirFile : str
        File directory.
    dbName : str
        OS to load.
    fieldlist : list(str)
        list of fields to process.

    Returns
    -------
    res : pandas df
        Loaded data.

    """

    fName = '{}/{}.npy'.format(dirFile, dbName)

    tt = np.load(fName)

    res = pd.DataFrame.from_records(tt)

    idx = res['target_name'].isin(fieldlist)

    res = pd.DataFrame(res[idx])

    res['dbName'] = dbName

    if 'year' not in res.columns:
        res = get_year(res)

    return res


def get_year(df):
    """
    Function to estimate the year of observations

    Parameters
    ----------
    df : pandas df
        Data to process.

    Returns
    -------
    df_tot : pandas df
        Original df+year.

    """

    df_tot = pd.DataFrame()
    for i in range(10):
        night_min = i*365
        night_max = (i+1)*365.
        idx = df['night'] >= night_min
        idx &= df['night'] < night_max
        rr = pd.DataFrame(df[idx])
        rr['year'] = i+1
        df_tot = pd.concat((rr, df_tot))

    return df_tot


def get_infos(grp, bands='ugrizy'):
    """
    Function to get infos

    Parameters
    ----------
    grp : pandas df
        Data to process.
    bands : str, optional
        List of filter to consider. The default is 'ugrizy'.

    Returns
    -------
    res : pandas df
        Output data.

    """

    # grab the total number of visits per band

    dd = {}
    for b in bands:
        idx = grp['filter'] == b
        sel = grp[idx]
        dd['Nvisits_{}'.format(b)] = [len(sel)]
        m5_coadd = 1.25*np.log10(np.sum(10**(0.8*sel['fiveSigmaDepth'])))
        dd['fiveSigmaDepth_{}'.format(b)] = [m5_coadd]

    res = pd.DataFrame.from_dict(dd)

    return res


parser = OptionParser(
    description='Script to check calib requirements from PZ,WL from pointings')

parser.add_option("--dirFile", type="str",
                  default='../DB_Files',
                  help="file directory [%default]")
parser.add_option('--dd_list', type=str,
                  default='ddf_list.csv',
                  help='OS DD list[%default]')
parser.add_option('--fields', type=str,
                  default='DD:COSMOS,DD:XMM_LSS,DD:ECDFS,DD:ELAISS1,DD:EDFS_a,DD:EDFS_b',
                  help='DD fields to consider [%default]')
parser.add_option('--outDir', type=str,
                  default='../nvisits_m5',
                  help='output dir [%default]')

opts, args = parser.parse_args()

dirFile = opts.dirFile
dd_list = opts.dd_list
fields = opts.fields.split(',')
outDir = opts.outDir

# create outputdir if necessary
checkDir(outDir)

# load the config file
df_list = pd.read_csv(dd_list, comment='#')

for i, row in df_list.iterrows():
    data = load_data(dirFile, row['dbName'], fields)
    print(len(data))
    print(data.columns)
    print(data['target_name'].unique())
    df_info = data.groupby(['dbName', 'target_name', 'year']).apply(
        lambda x: get_infos(x)).reset_index()
    print(df_info)
    outName = '{}/{}.hdf5'.format(outDir, row['dbName'])
    df_info.to_hdf(outName, key='nvisits_m5')
