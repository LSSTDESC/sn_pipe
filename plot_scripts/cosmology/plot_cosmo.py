#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 14:08:41 2023

@author: philippe.gris@clermont.in2p3.fr
"""

import pandas as pd
from sn_plotter_cosmology import plt
import numpy as np
from sn_plotter_cosmology.cosmoplot import Process_OS, plot_allOS
from optparse import OptionParser


def load_data(dbDir, config):
    """
    Method to load data

    Parameters
    ----------
    dbDir : str
        Data dir.
    config : pandas df
        config (dbName+OS params).

    Returns
    -------
    df : pandas df
        output data.

    """

    df = pd.DataFrame()

    for i, row in config.iterrows():
        fi = '{}/cosmo_{}.hdf5'.format(dbDir, row['dbName'])
        dfa = pd.read_hdf(fi)
        df = pd.concat((df, dfa))

    return df


parser = OptionParser(description='Script to analyze SN prod')

parser.add_option('--dbDir', type=str, default='../cosmo_fit',
                  help='OS location dir[%default]')
parser.add_option('--dbList', type=str,
                  default='input/DESC_cohesive_strategy/config_ana.csv',
                  help='OS name[%default]')
parser.add_option('--budget_DD', type=float,
                  default=0.07,
                  help='DD budget[%default]')
opts, args = parser.parse_args()

dbDir = opts.dbDir
dbList = opts.dbList
budget = np.round(opts.budget_DD, 2)

config = pd.read_csv(dbList, comment='#')
config['dbName'] += '_{}'.format(budget)
data = load_data(dbDir, config)
print(data.columns)
# data['dbName_DD'] += '_{}'.format(budget)


grpCol = ['season', 'dbName_DD', 'prior']
# resdf = process_OS(data, grpCol)
resdf = Process_OS(data, grpCol).res

resdf['UD_mean'] = 0
resdf['DD_mean'] = 0

for UD in ['COSMOS', 'XMM-LSS']:
    resdf['UD_mean'] += resdf['{}_mean'.format(UD)]

for DD in ['CDFS', 'COSMOS', 'EDFSa', 'EDFSb', 'ELAISS1']:
    resdf['DD_mean'] += resdf['{}_mean'.format(DD)]

resdf['DDF_mean'] = resdf['UD_mean']+resdf['DD_mean']

for tt in ['DDF', 'UD', 'DD', 'WFD']:
    resdf['{}_std'.format(tt)] = 0

print(resdf, config)
print(resdf.columns)

vvars = ['MoM', 'sigma_w', 'sigma_w0', 'sigma_wa']
leg = dict(zip(vvars, [r'$SMoM$', r'$\sigma_w$[%]',
           r'$\sigma_{w_0}$ [%]', r'$\sigma_{w_a}$ [%]']))
priors = ['prior', 'noprior']
dd = dict(zip(priors, ['with prior', 'no prior']))
for vary in vvars:
    for prior in priors[:1]:
        plot_allOS(resdf, config, vary=vary,
                   legy=leg[vary], prior=prior, figtitle=dd[prior])

vvarsb = ['DDF', 'UD', 'DD', 'WFD']
legb = [r'$N^{DDF}$', r'$N^{UD}$', r'$N^{DD}$', r'$N^{WFD}$']
lleg = dict(zip(vvarsb, legb))
for vary in vvarsb:
    plot_allOS(resdf, config, vary=vary,
               legy=lleg[vary], prior='prior', figtitle='')


# ax.legend()
# idx = resdf['prior'] == 'noprior'
# cosmo_plot(resdf[idx])

# cosmo_four(resdf)
plt.show()
