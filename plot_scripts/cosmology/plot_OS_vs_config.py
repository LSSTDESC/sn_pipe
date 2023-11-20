#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 09:29:51 2023

@author: philippe.gris@clermont.in2p3.fr
"""
from optparse import OptionParser
import pandas as pd
from sn_plotter_cosmology.cosmoplot import Process_OS, plot_allOS
from sn_plotter_cosmology import plt
import numpy as np


def load_data(theDir, dbName, frac_sigmaC):
    """
    Function to load data

    Parameters
    ----------
    theDir : str
        Data location dir.
    dbName : str
        OS to process.
    frac_sigmaC : float
        fraction of SN with sigma<= 0.04.

    Returns
    -------
    df : pandas df
        Data.

    """

    fullPath = '{}/cosmo_{}.hdf5'.format(theDir, dbName)
    df = pd.read_hdf(fullPath)
    df['frac_sigmaC'] = frac_sigmaC
    df = df.round({'frac_sigmaC': 2})

    return df


parser = OptionParser(
    description='Script to plot cosmos results vs survey config')

parser.add_option('--dbList', type=str,
                  default='input/DESC_cohesive_strategy/survey_list_dir.csv',
                  help='List of dirs for cosmo results [%default]')
parser.add_option('--dbName', type=str, default='DDF_DESC_0.80_SN',
                  help='OS to process [%default]')

opts, args = parser.parse_args()

dbList = opts.dbList
dbName = opts.dbName

# read config file
toproc = pd.read_csv(dbList, comment='#')

print(toproc)

# load all the data

df = pd.DataFrame()
for i, row in toproc.iterrows():
    dfa = load_data(row['dbDir'], dbName, row['frac_sigmaC'])
    df = pd.concat((df, dfa))

print(df.columns)

df['MoM_DETF'] = 1./(np.sqrt(df['Cov_w0_w0_fit'])*np.sqrt(df['Cov_wa_wa_fit']))

# print(test)

grpCol = ['season', 'dbName_DD', 'prior', 'frac_sigmaC']
# resdf = process_OS(data, grpCol)
resdf = Process_OS(df, grpCol).res
print(resdf.columns)

vvars = ['MoM', 'sigma_w', 'sigma_w0', 'sigma_wa']
leg = dict(zip(vvars, [r'Relative $SMoM$', r'$\sigma_w$[%]',
           r'$\sigma_{w_0}$ [%]', r'$\sigma_{w_a}$ [%]']))
dbNameb = '_'.join(dbName.split('_')[:-1])
figtitle = '{}'.format(dbNameb)
leg_prefix = 'frac$^{WFD}_{\sigma_{C} \leq 0.04} = $'
for vary in vvars:
    plot_allOS(resdf, toproc, dataCol='frac_sigmaC', configCol='frac_sigmaC',
               vary=vary, legy=leg[vary], prior='prior', figtitle=figtitle, leg_prefix=leg_prefix)

plt.show()
