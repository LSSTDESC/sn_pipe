from sn_desc_ddf_strategy.dd_scenario import FiveSigmaDepth_Nvisits, DD_Scenario
from sn_desc_ddf_strategy.dd_scenario import nvisits_from_m5, reshuffle
from sn_desc_ddf_strategy.dd_scenario import get_final_scenario
from sn_desc_ddf_strategy import plt
from sn_desc_ddf_strategy.dd_scenario import Delta_m5, Delta_nvisits
from sn_desc_ddf_strategy.dd_scenario import Budget_time, Scenario_time

################################################
##### get Nvisits from m5 req.          ########
## m5_dict: total number of visits (PZ req.) ##
###############################################
bands = 'ugrizy'

m5class = FiveSigmaDepth_Nvisits()

m5_summary = m5class.summary
m5_nvisits = m5class.msingle_calc
m5_dict = m5_summary.to_dict()

print(m5_dict)
print(m5_nvisits)

## get (Nvisits_UD vs N_visits_DD for (Kf_UD, Ns_UD) combinations ####

corresp = dict(zip(['Nvisits_y1', 'Nvisits_y2_y10'], ['PZ_y1', 'PZ_y2_y10']))
nseasons = dict(zip(['Nvisits_y1', 'Nvisits_y2_y10'], [1, 9]))

pz_wl_req = {}
for key, vals in corresp.items():
    pz_wl_req[vals] = [85, int(m5_dict[key]/nseasons[key])]

pz_wl_req['WL_10xWFD'] = [85, 800]
pz_wl_req_err = {}
pz_wl_req_err['PZ_y2_y10'] = (m5_dict['Nvisits_y2_y10_m']/9.,
                              m5_dict['Nvisits_y2_y10_p']/9.)

# Parameters
Nf_DD_y1 = 5  # UD the second year
Nv_DD_y1 = int(m5_dict['Nvisits_y1'])
sl_UD = 180.
cad_UD = 2.
NDD = 5
Nv_LSST = 2.1e6
frac_moon = 0.30
sl_DD = 180.
cad_DD = 3.

myclass = DD_Scenario(Nf_combi=[(2, 2), (2, 3), (2, 4)],
                      zcomp=[0.80, 0.75, 0.70],
                      scen_names=['DDF_DESC_0.80',
                                  'DDF_DESC_0.75',
                                  'DDF_DESC_0.70'],
                      Nf_DD_y1=Nf_DD_y1,
                      Nv_DD_y1=Nv_DD_y1,
                      sl_UD=sl_UD, cad_UD=cad_UD, NDD=NDD, Nv_LSST=Nv_LSST,
                      frac_moon=frac_moon)

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
ffig += '{}={} days, season length={} days'.format(cadud, myclass.cad_UD,
                                                   int(myclass.sl_UD))

myclass.plot(restot, varx='Nv_DD',
             legx='N$_{visits}^{DD}/season}$', figtitle=ffig)

# zcomp_req = {}
# zcomp_req_err = {}
# pz_wl_req_err = {}
# scenario = {}

res = myclass.plot(restot, varx='Nv_DD',
                   legx='N$_{visits}^{DD}/season}$', scenario=scenario,
                   zcomp_req=zcomp_req, zcomp_req_err=zcomp_req_err,
                   pz_wl_req=pz_wl_req, pz_wl_req_err=pz_wl_req_err,
                   figtitle=ffig)

# plt.show()
### m5_resu ###

m5_resu = nvisits_from_m5(res, m5class)

# finish the data

df_res = myclass.finish(res)

toprint = ['name', 'Nf_UD', 'Ns_UD', 'nvisits_UD_night',
           'g', 'r', 'i', 'z', 'y', 'delta_z',
           'nvisits_DD_season', 'budget']

df_res = df_res.round({'budget': 2})
print(df_res[toprint])
df_res[toprint].to_csv('ddf_res1.csv', index=False)


# transform df_res
df_resb = reshuffle(df_res, m5_resu, sl_UD, cad_UD, frac_moon)
print(df_resb.columns)

# get the final scenarioreshu
m5single = m5class.msingle_calc
resa, resb = m5class.m5_band_from_Nvisits(m5_resu, m5single, sl_DD=sl_DD,
                                          cad_DD=cad_DD, frac_moon=frac_moon)
dfres = df_resb.groupby('name').apply(
    lambda x: get_final_scenario(x, NDD, resa, resb)).reset_index()
print(dfres)

##### Final plots ####


# estimate and plot delta_m5 for each scenario
Delta_m5(dfres, m5_nvisits)

# estimate and plot visit ratio for each scenario
Delta_nvisits(dfres, m5_nvisits)


# plot budget vs time for each scenario
Budget_time(dfres, Nv_LSST)


# plot scenario vs time
Scenario_time(dfres)


plt.show()
