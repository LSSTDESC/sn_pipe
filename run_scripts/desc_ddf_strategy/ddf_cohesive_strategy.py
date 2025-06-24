from optparse import OptionParser
import numpy.lib.recfunctions as rf
from sn_desc_ddf_strategy.dd_scenario import DD_Scenario
from sn_desc_ddf_strategy.dd_scenario import nvisits_from_m5, reshuffle
from sn_desc_ddf_strategy.dd_scenario import get_final_scenario, moon_recovery
from sn_desc_ddf_strategy import plt
from sn_desc_ddf_strategy.dd_scenario import Delta_m5, Delta_nvisits
from sn_desc_ddf_strategy.dd_scenario import Budget_time, Scenario_time
from sn_desc_ddf_strategy.dd_scenario import reverse_df, uniformize
from sn_desc_ddf_strategy.dd_scenario import FiveSigmaDepth_Nvisits
from sn_desc_ddf_strategy.dd_scenario import Calc_UD_visits

import pandas as pd
import numpy as np
import itertools


def get_nfconfig(config):

    nud = config['Nud'].to_list()
    nsud = config['Nsud'].to_list()
    z = list(itertools.zip_longest(nud, nsud))

    return z


def get_nvisits(df):

    print(df.columns)

    nvisits = np.sum(df['Nfields']*df['nvisits_band_season'])

    print('hhhh', nvisits)


parser = OptionParser(
    description='Design a cohesive DESC DDF Strategy')

parser.add_option("--Nvisits_WL_season", type=int,
                  default=800,
                  help="Nvisits WL requirement [%default]")
parser.add_option("--budget_DD", type=float,
                  default=0.07,
                  help="DD budget [%default]")
parser.add_option("--Nf_DD_y1", type=int,
                  default=5,
                  help="N DD fields Y1[%default]")
parser.add_option("--sl_UD", type=int,
                  default=180,
                  help="season length UD fields [%default]")
parser.add_option("--cad_UD", type=float,
                  default=2.,
                  help="cadence UD fields [%default]")
parser.add_option("--NDDF", type=int,
                  default=5,
                  help="total number of DDFs[%default]")
parser.add_option("--Ns_DD", type=int,
                  default=9,
                  help="Number of season of of the DD fields [%default]")
parser.add_option("--obs_UD_DD", type=int,
                  default=1,
                  help="Observe UD fields as DD fields [%default]")
parser.add_option("--Nv_LSST", type=float,
                  default=2.1e6,
                  help="Total number of LSST visits(10 years) [%default]")
parser.add_option("--frac_moon", type=float,
                  default=0.30,
                  help="Fraction of visits (in a season) \
                  with the low-phase Moon [%default]")
parser.add_option("--sl_DD", type=int,
                  default=180,
                  help="season length DD fields [%default]")
parser.add_option("--cad_DD", type=float,
                  default=3.,
                  help="cadence DD fields [%default]")
parser.add_option("--swap_filter_moon", type=str,
                  default='y',
                  help="Filter to remove when Moon at low phases [%default]")
parser.add_option("--recover_from_moon", type=int,
                  default=1,
                  help="Modify Nvisits for the swap filter - high \
                  moon phases. [%default]")
parser.add_option("--m5_single_file", type=str,
                  default='input/DESC_cohesive_strategy/m5_single_med.csv',
                  help="m5 single visit file (all bands) [%default]")
parser.add_option("--filter_alloc_file", type=str,
                  default='input/DESC_cohesive_strategy/filter_allocation.csv',
                  help="filter allocation file (all bands) [%default]")
parser.add_option("--pz_requirements", type=str,
                  default='input/DESC_cohesive_strategy/pz_requirements.csv',
                  help="m5 pz requirements [%default]")
parser.add_option("--m5_from_db", type=int,
                  default=0,
                  help="1 to grab m5 from db [%default]")
parser.add_option("--dbDir", type=str,
                  default='../DB_Files',
                  help="OS location dir [%default]")
parser.add_option("--dbName", type=str,
                  default='draft_connected_v2.99_10yrs.npy',
                  help="dbName to get DB infos [%default]")
parser.add_option("--Nv_DD_max", type=int,
                  default=3500,
                  help="max number of Nvisits per DD/season [%default]")
parser.add_option("--showPlot", type=int,
                  default=0,
                  help="to show plot or not [%default]")
parser.add_option("--config_scenario", type=str,
                  default='input/DESC_cohesive_strategy/scenario_to_generate/ddf_desc_scenario.csv',
                  help="scenarios to consider [%default]")

opts, args = parser.parse_args()

pparams = vars(opts)

myclass = Calc_UD_visits(pparams)

res = myclass()

print(res.dtype)

### m5_resu ###

m5_resu = nvisits_from_m5(res, myclass.m5class)
print('m5_resu')
print(m5_resu)
print('res', res.dtype.names)
print(res)

# plt.show()
# finish the data
"""
print('finishing')
idx = res['name'] == 'DDF_Univ_SN'
res = res[idx]
"""
df_res = myclass.finish(res)

toprint = ['name', 'Nf_UD', 'Ns_UD', 'nvisits_UD_night',
           'g', 'r', 'i', 'z', 'y', 'delta_z',
           'nvisits_DD_season', 'budget']

df_res = df_res.round({'budget': 2})
print(df_res[toprint])
df_res[toprint].to_csv('ddf_res1.csv', index=False)


# transform df_res
"""
db_ref = 'DDF_Univ_SN'
idx = df_res['name'] == db_ref

print(m5_resu[m5_resu['name'] == db_ref])
"""

df_resb = reshuffle(df_res, m5_resu,
                    pparams['sl_UD'], pparams['cad_UD'],
                    pparams['frac_moon'], pparams['swap_filter_moon'])
print(df_resb)

# get the final scenario
m5single = myclass.m5class.msingle_calc

vv = ['band', 'm5_med_single', 'Nvisits_WL_PZ_y1', 'Nvisits_WL_PZ_y2_y10',
      'm5_WL_PZ_y1', 'm5_WL_PZ_y2_y10']

print(m5single[vv])
print(m5single['Nvisits_WL_PZ_y2_y10'].sum())

# print(test)

resa, resb = myclass.m5class.m5_band_from_Nvisits(m5_resu, m5single,
                                                  sl_DD=pparams['sl_DD'],
                                                  cad_DD=pparams['cad_DD'],
                                                  frac_moon=pparams['frac_moon'],
                                                  swap_filter_moon=pparams['swap_filter_moon'])
print('allllllllll')
print(df_resb.columns)

dfres = df_resb.groupby('name').apply(
    lambda x: get_final_scenario(x, pparams['NDDF'], resa, resb)).reset_index()

dfres['nvisits_night'] = dfres['nvisits_night'].astype(int)

pd.set_option('display.max_columns', None)
print('before recovery', dfres)

ll_norecover = ['DDF_SCOC_pII', 'DDF_Univ_SN', 'DDF_Univ_WZ']
if pparams['recover_from_moon']:
    idx = dfres['year'] > 1
    for db in ll_norecover:
        idx &= dfres['name'] == db
    dfresm = moon_recovery(dfres[idx], pparams['swap_filter_moon'])
    dfres = pd.concat((dfres[~idx], dfresm))

print('after recovery', dfres[['name', 'year',
      'band', 'nvisits_night']])

dfres = uniformize(dfres, 'DDF_Univ_SN',
                   Nv_LSST=pparams['Nv_LSST'], budget=pparams['budget_DD'])


print('uniformize', dfres[['name', 'year',
      'band', 'nvisits_night']])

##### Final plots ####


# estimate and plot delta_m5 for each scenario
# Delta_m5(dfres, m5_nvisits)

# estimate and plot visit ratio for each scenario
# Delta_nvisits(dfres, m5_nvisits)


# plot budget vs time for each scenario
# Budget_time(dfres, pparams['Nv_LSST'], pparams['budget_DD'])


# plot scenario vs time - does not work any more
# Scenario_time(dfres, swap_filter_moon=pparams['swap_filter_moon'])
# see run_scripts/desc_ddf_strategy/ana_OS.py for this


if pparams['showPlot']:
    plt.show()

# check total number of visits
print(dfres.columns)

# Nvisits = get_nvisits(dfres)

dfres['cad'] = dfres['cad'].astype(int)
pp = ['name', 'year', 'fieldType', 'cad', 'sl']
tt = dfres.groupby(pp).apply(lambda x: reverse_df(x)).reset_index()
tt['budget_DD'] = pparams['budget_DD']
tt.to_csv('scenarios_{}.csv'.format(pparams['budget_DD']), index=False)

sumCols = ['nvisits_band_season', 'nvisits_band_season_fields']
dfres['nvisits_band_season_fields'] = dfres['nvisits_band_season']*dfres['Nfields']
res = dfres.groupby(['name', 'fieldType', 'year', 'band'])[
    sumCols].sum().reset_index()

print(res)

resb = res.groupby(['name', 'fieldType', 'year'])[sumCols].sum().reset_index()

print(resb)

resc = dfres.groupby(['name'])[sumCols].sum().reset_index()

print(resc)

print(myclass.m5_nvisits)

print(myclass.m5_nvisits['Nvisits_y2_y10'] /
      myclass.m5_nvisits['nseason_y2_y10'])
myclass.m5_nvisits.to_csv('resc.csv', index=False)

if pparams['showPlot']:
    plt.show()
