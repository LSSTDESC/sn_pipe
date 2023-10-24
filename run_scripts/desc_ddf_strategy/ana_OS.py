#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 14:34:46 2023

@author: philippe.gris@clermont.in2p3.fr
"""

import numpy as np
import pandas as pd
from optparse import OptionParser
from sn_plotter_analysis import plt, filtercolors
from sn_tools.sn_io import checkDir
from sn_desc_ddf_strategy.ana_os_reqs import Anaplot_OS
from sn_desc_ddf_strategy.ana_OS_class import Plot_cadence


def plot_NSN_season(dbDir, df_config_scen):

    dbNames = list(df_config_scen['scen'].unique())

    vala = 'observationStartMJD'
    valb = 'numExposures'
    valc = 'MJD_season'

    # for dbName in dbNames:
    rmoon = []
    for dbName in ['DDF_Univ_SN', 'DDF_Univ_WZ']:
        fig, ax = plt.subplots()
        data = np.load('{}/{}.npy'.format(dbDir, dbName))
        data = pd.DataFrame.from_records(data)

        fields = data['note'].unique()

        for field in fields:
            idx = data['note'] == field
            sela = data[idx]

            seasons = sela['season'].unique()

            for seas in seasons:
                idxb = sela['season'] == seas
                selb = sela[idxb]
                selb = selb.sort_values(by=[valb])
                tp = np.cumsum(selb[valb])
                ax.plot(selb[vala], tp)
                idxm = selb['moonPhase'] <= 20.
                selm = selb[idxm]
                print(dbName, field, seas, tp.to_list()
                      [-1], len(selm)/len(selb))
                rmoon.append((dbName, field, seas, tp.to_list()
                              [-1], len(selm)/len(selb)))

    resmoon = np.rec.fromrecords(
        rmoon, names=['name', 'field', 'seas', 'Nvisits_season', 'moon_frac'])

    figb, axb = plt.subplots()
    axb.plot(resmoon['moon_frac'], resmoon['Nvisits_season'], 'ko')
    print('moon frac', np.mean(resmoon['moon_frac']))


parser = OptionParser(description='Script to analyze Observing Strategy')

parser.add_option('--dbDir', type=str, default='../DB_Files',
                  help='OS location dir [%default]')
parser.add_option('--configScenario', type='str',
                  default='input/DESC_cohesive_strategy/config_scenarios.csv',
                  help='config file to use[%default]')
parser.add_option('--configdB', type='str',
                  default='input/DESC_cohesive_strategy/list_scen.csv',
                  help='config file for plots [%default]')
parser.add_option('--outDir', type=str, default='Plot_OS',
                  help='Where to store the plots [%default]')
parser.add_option('--Nvisits_LSST', type=float, default=2.1e6,
                  help='Total number of LSST visits[%default]')
parser.add_option('--budget_DD', type=float, default=0.07,
                  help='DD budget [%default]')
parser.add_option('--pz_requirement', type=str,
                  default='input/DESC_cohesive_strategy/pz_requirements.csv',
                  help='PZ requirement file [%default]')
parser.add_option('--filter_alloc_req', type=str,
                  default='input/DESC_cohesive_strategy/filter_allocation.csv',
                  help='Filter alloc for WL reqs [%default]')
parser.add_option('--Nvisits_WL', type=int,
                  default=8000,
                  help='Number of visits after ten years [%default]')

opts, args = parser.parse_args()

dbDir = opts.dbDir
configScenario = opts.configScenario
configdB = opts.configdB
outDir = opts.outDir
Nvisits_LSST = opts.Nvisits_LSST
pz_requirement = opts.pz_requirement
filter_alloc_req = opts.filter_alloc_req
Nvisits_WL = opts.Nvisits_WL

budget = np.round(opts.budget_DD, 2)

if outDir != '':
    checkDir(outDir)


# this is to plot the budget, PZ req and WL req

config_db = pd.read_csv(configdB, comment='#')
if budget > 0:
    config_db['dbName'] += '_{}'.format(budget)

"""
pp = Anaplot_OS(dbDir, config_db, Nvisits_LSST, budget, outDir='',
                pz_requirement=pz_requirement,
                filter_alloc_req=filter_alloc_req, Nvisits_WL=Nvisits_WL)

# pp.plot_budget()
# pp.plot_m5_PZ()
# pp.plot_Nvisits_WL()
# pp.plot_cadence_mean()
#plt.show()
"""

# plot cadence

df_config_scen = pd.read_csv(configScenario, comment='#')
df_config_scen['scen'] = df_config_scen['scen']+'_{}'.format(budget)
#dbNames = df_config_scen['scen'].unique()
dbNames = np.asarray(config_db['dbName'].unique())
# dbNames = ['DDF_DESC_0.70_co_0.07']
# dbNames = ['DDF_SCOC_pII_0.07']
# dbNames = ['DDF_Univ_SN_0.07']
for dbName in dbNames:
    print('processing', dbName)
    Plot_cadence(dbDir, dbName, df_config_scen, outDir=outDir)
plt.show()
