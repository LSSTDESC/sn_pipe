import numpy.lib.recfunctions as rf
from sn_desc_ddf_strategy.dd_scenario import DD_Scenario
from sn_desc_ddf_strategy.dd_scenario import nvisits_from_m5, reshuffle
from sn_desc_ddf_strategy.dd_scenario import get_final_scenario, moon_recovery
from sn_desc_ddf_strategy import plt
from sn_desc_ddf_strategy.dd_scenario import Delta_m5, Delta_nvisits
from sn_desc_ddf_strategy.dd_scenario import Budget_time, Scenario_time
from sn_desc_ddf_strategy.dd_scenario import reverse_df, uniformize
from sn_desc_ddf_strategy.dd_scenario import FiveSigmaDepth_Nvisits

import pandas as pd

from optparse import OptionParser


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
parser.add_option("--Nf_combi", type=str,
                  default='(2,2),(2,3),(2,4)',
                  help="to show plot or not [%default]")
parser.add_option("--zcomp", type=str,
                  default='0.80,0.75,0.70',
                  help="redshift completeness configuration [%default]")
parser.add_option("--scen_names", type=str,
                  default='DDF_DESC_0.80,DDF_DESC_0.75,DDF_DESC_0.70',
                  help="scenario names corresponding to zcomp [%default]")

opts, args = parser.parse_args()

pparams = vars(opts)
pparams['Nf_combi'] = eval('[{}]'.format(pparams['Nf_combi']))

bands = 'ugrizy'

###############################################
##### get Nvisits from m5 req.          #######
## m5_dict: total number of visits (PZ req.) ##
###############################################

m5_single_band = {}
frac_band = {}

if pparams['m5_from_db']:
    # getting m5 single exposure from a simulated OS
    from sn_desc_ddf_strategy.dd_scenario import DB_Infos
    db_info = DB_Infos(pparams['dbDir'], pparams['dbName'])
    m5_single_band = db_info.m5_single
    frac_band = db_info.filter_alloc
else:
    m5_fi = pd.read_csv(pparams['m5_single_file'])
    for i, row in m5_fi.iterrows():
        m5_single_band[row['band']] = row['m5_single']
    frac_fi = pd.read_csv(pparams['filter_alloc_file'])
    for i, row in frac_fi.iterrows():
        frac_band[row['band']] = row['frac_band']


m5class = FiveSigmaDepth_Nvisits(
    Nvisits_WL_season=pparams['Nvisits_WL_season'],
    frac_band=frac_band,
    m5_single=m5_single_band, Ns_y2_y10=opts.Ns_DD)


msingle = m5class.msingle
print('hello', msingle)


m5_summary = m5class.summary
m5_nvisits = m5class.msingle_calc
m5_dict = m5_summary.to_dict()

print(m5_summary)
print(m5_dict)
print(m5_nvisits)

## get (Nvisits_UD vs N_visits_DD for (Kf_UD, Ns_UD) combinations ####

corresp = dict(zip(['Nvisits_y1', 'Nvisits_y2_y10'], ['PZ_y1', 'PZ_y2_y10']))
nseasons = dict(zip(['Nvisits_y1', 'Nvisits_y2_y10'], [1, opts.Ns_DD]))
corresp = dict(zip(['Nvisits_WL_PZ_y1', 'Nvisits_WL_PZ_y2_y10'], [
    'WL_PZ_y1', 'WL_PZ_y2_y10']))
nseasons = dict(
    zip(['Nvisits_WL_PZ_y1', 'Nvisits_WL_PZ_y2_y10'], [1, opts.Ns_DD]))

pz_wl_req = {}
for key, vals in corresp.items():
    pz_wl_req[vals] = [95, int(m5_dict[key]/nseasons[key])]

# pz_wl_req['WL_10xWFD'] = [85, 800]
pz_wl_req_err = {}
# pz_wl_req_err['PZ_y2_y10'] = (m5_dict['Nvisits_y2_y10_m']/9.,
#                              m5_dict['Nvisits_y2_y10_p']/9.)
pz_wl_req_err['WL_PZ_y2_y10'] = (m5_dict['Nvisits_WL_PZ_y2_y10_m']/opts.Ns_DD,
                                 m5_dict['Nvisits_WL_PZ_y2_y10_p']/opts.Ns_DD)
# Parameters
Nv_DD_y1 = int(m5_dict['Nvisits_WL_PZ_y1'])

myclass = DD_Scenario(budget_DD=pparams['budget_DD'],
                      Nf_combi=pparams['Nf_combi'],
                      zcomp=list(map(float, pparams['zcomp'].split(','))),
                      scen_names=pparams['scen_names'].split(','),
                      m5_single_OS=msingle,
                      Nf_DD_y1=pparams['Nf_DD_y1'],
                      Nv_DD_y1=Nv_DD_y1,
                      sl_UD=pparams['sl_UD'], cad_UD=pparams['cad_UD'],
                      sl_DD=pparams['sl_DD'], cad_DD=pparams['cad_DD'],
                      Ns_DD=pparams['Ns_DD'],
                      NDDF=pparams['NDDF'], Nv_LSST=pparams['Nv_LSST'],
                      frac_moon=pparams['frac_moon'],
                      obs_UD_DD=pparams['obs_UD_DD'],
                      Nv_DD_max=pparams['Nv_DD_max'])

restot = myclass.get_combis()
zcomp_req = myclass.get_zcomp_req()
zcomp_req_err = myclass.get_zcomp_req_err()
scenario = myclass.get_scenario()

### plot the result and get scenarios ######

nvisits = '$N_{visits}^{LSST}$'
cadud = '$cad^{UD}$'
ftit = 'DD budget={}% - {}={} million'.format(int(100*myclass.budget_DD),
                                              nvisits, myclass.Nv_LSST/1.e6)
ffig = '{} \n'.format(ftit)
ffiga = '{} \n'.format(ftit)
ffig += '{}={} days, season length={} days'.format(cadud, myclass.cad_UD,
                                                   int(myclass.sl_UD))
#ffiga += 'season length={} days'.format(int(myclass.sl_UD))

"""
restot = rf.append_fields(restot, 'Nv_UD_season',
                          restot['Nv_UD']/restot['Ns_UD'])
"""
myclass.plot(restot, varx='Nv_DD',
             legx='N$_{v}^{DD}/season}$',
             vary='Nv_UD',
             legy='N$_{v}^{UD}/season}$', figtitle=ffiga)


# zcomp_req = {}
# zcomp_req_err = {}
# pz_wl_req = {}
# pz_wl_req_err = {}
# scenario = {}


Nvisits_avail = myclass.budget_DD*myclass.Nv_LSST-myclass.Nf_DD_y1*myclass.Nv_DD_y1
Nv_DD_SCOC_pII = Nvisits_avail/52.
Nv_UD_SCOC_pII = (10*Nv_DD_SCOC_pII/3)*opts.cad_UD/opts.sl_UD
deep_universal = {}
scoc_pII = {}
du_pos = 140
if pparams['budget_DD'] < 0.06:
    du_pos = 90
deep_universal['Deep Universal'] = [
    Nvisits_avail/(opts.Ns_DD*opts.NDDF), du_pos]
scoc_pII['SCOC_pII'] = [Nv_DD_SCOC_pII, Nv_UD_SCOC_pII]


res = myclass.plot(restot, varx='Nv_DD',
                   legx='N$_{visits}^{DD}/season}$', scenario=scenario,
                   zcomp_req=zcomp_req, zcomp_req_err=zcomp_req_err,
                   pz_wl_req=pz_wl_req, pz_wl_req_err=pz_wl_req_err,
                   deep_universal=deep_universal, scoc_pII=scoc_pII,
                   figtitle=ffig)

print(res)
print(res.dtype)
print(deep_universal)
print(scoc_pII)
print(Nvisits_avail)
### m5_resu ###

m5_resu = nvisits_from_m5(res, m5class)
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

# get the final scenario
m5single = m5class.msingle_calc

vv = ['band', 'm5_med_single', 'Nvisits_WL_PZ_y1', 'Nvisits_WL_PZ_y2_y10',
      'm5_WL_PZ_y1', 'm5_WL_PZ_y2_y10']

print(m5single[vv])


resa, resb = m5class.m5_band_from_Nvisits(m5_resu, m5single,
                                          sl_DD=pparams['sl_DD'],
                                          cad_DD=pparams['cad_DD'],
                                          frac_moon=pparams['frac_moon'],
                                          swap_filter_moon=pparams['swap_filter_moon'])
print(resa[resa['name'] == 'DDF_Univ_SN'])
print(resb)

dfres = df_resb.groupby('name').apply(
    lambda x: get_final_scenario(x, pparams['NDDF'], resa, resb)).reset_index()

dfres['nvisits_night'] = dfres['nvisits_night'].astype(int)
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
Delta_m5(dfres, m5_nvisits)

# estimate and plot visit ratio for each scenario
Delta_nvisits(dfres, m5_nvisits)


# plot budget vs time for each scenario
Budget_time(dfres, pparams['Nv_LSST'], pparams['budget_DD'])

if pparams['showPlot']:
    plt.show()

# plot scenario vs time
# Scenario_time(dfres, swap_filter_moon=pparams['swap_filter_moon'])


# check total number of visits
print(dfres.columns)
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

print(m5_nvisits)

print(m5_nvisits['Nvisits_y2_y10']/m5_nvisits['nseason_y2_y10'])
m5_nvisits.to_csv('resc.csv', index=False)

if pparams['showPlot']:
    plt.show()
