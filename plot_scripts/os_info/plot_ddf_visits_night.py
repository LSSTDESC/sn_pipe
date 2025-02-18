#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 09:46:22 2025

@author: philippe.gris@clermont.in2p3.fr
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from optparse import OptionParser


def analyze_simu_exp(data):
    """
    Function to analyze sim+expected data

    Parameters
    ----------
    data : pandas df
        Data to process.

    Returns
    -------
    None.

    """

    # add new columns as diff

    for vv in ['nvisits', 'u', 'g', 'r', 'i', 'z', 'y']:
        data['diff_{}'.format(vv)] = data['{}_exp'.format(vv)] - data[vv]

    print(data)

    res_stat = data.groupby(['target_name', 'season']).apply(
        lambda x: stat_simu_exp(x)).reset_index()

    print(res_stat)

    plot_stat(res_stat)


def stat_simu_exp(grp):
    """
    Function to get some stat on simu+expected os data

    Parameters
    ----------
    grp : pandas df
        Data to process.

    Returns
    -------
    rr : pandas df
        output data.

    """

    dict_frac = {}
    nnights = len(grp)
    for vv in ['nvisits', 'u', 'g', 'r', 'i', 'z', 'y']:
        myvar = 'diff_{}'.format(vv)
        idxa = grp[myvar] >= 1
        idxb = grp[myvar] <= -1
        idxc = np.abs(grp[myvar]) < 0.5
        dict_frac['{}_missing'.format(vv)] = [len(grp[idxa])/nnights]
        dict_frac['{}_excess'.format(vv)] = [len(grp[idxb])/nnights]
        dict_frac['{}_perfect'.format(vv)] = [len(grp[idxc])/nnights]

    rr = pd.DataFrame.from_dict(dict_frac)

    return rr


def plot_stat(data):

    fields = data['target_name'].unique()
    for vval in ['perfect', 'missing', 'excess']:
        fig, ax = plt.subplots(figsize=(14, 8))
        for field in fields:
            idx = data['target_name'] == field
            sel = data[idx]
            ax.plot(sel['season'], sel['nvisits_{}'.format(vval)])

        ax.grid(visible=True)

    plt.show()


parser = OptionParser(
    description='Script to analyse DDF visits on a nightly basis from pointings')

parser.add_option("--dbDir", type="str",
                  default='../ddf_visits_night',
                  help="file directory [%default]")
parser.add_option("--dbName", type="str",
                  default='desc_ddf_v4.2.1_10yrs',
                  help="OS name [%default]")

opts, args = parser.parse_args()

# Load parameters
dbDir = opts.dbDir
dbName = opts.dbName

fName = '{}/{}.hdf5'.format(dbDir, dbName)

data = pd.read_hdf(fName)

analyze_simu_exp(data)
