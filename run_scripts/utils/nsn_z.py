from sn_tools.sn_rate import SN_Rate
import matplotlib.pyplot as plt
import numpy as np
from optparse import OptionParser
import pandas as pd


def getNSN(rate='Perrett', H0=70, Om0=0.3,
           zmin=0.01, zmax=1.1, dz=0.01,
           season_length=180,
           survey_area=9.6, account_for_edges=False,
           min_rf_phase=-15, max_rf_phase=30):

    sn_rate = SN_Rate(rate=rate, H0=H0, Om0=Om0,
                      min_rf_phase=min_rf_phase,
                      max_rf_phase=max_rf_phase)
    zmax += dz/2

    zz, rateb, err_rate, nsn, err_nsn = sn_rate(
        zmin=zmin, zmax=zmax, dz=dz,
        account_for_edges=account_for_edges,
        duration=season_length, survey_area=survey_area)

    nsn_sum = np.cumsum(nsn)
    err_nsn_sum = np.sqrt(np.cumsum(err_nsn**2))

    res = pd.DataFrame(nsn_sum, columns=['nsn'])
    res['err_nsn'] = err_nsn_sum
    res['z'] = zz
    res['rate'] = rate
    res['edges'] = account_for_edges
    res['min_rf_phase'] = min_rf_phase
    res['max_rf_phase'] = max_rf_phase

    return res


def effi(grp, grp_ref):

    print(grp.name)
    rate = grp.name[0]
    # select proper rate for grp_ref
    idx = grp_ref['rate'] == rate
    ref = grp_ref[idx]

    res = grp.merge(ref, left_on=['rate', 'z'], right_on=[
                    'rate', 'z'], suffixes=['', '_ref'])
    res['nsn_ratio'] = res['nsn']/res['nsn_ref']
    return res[['nsn_ratio', 'z', 'rate', 'min_rf_phase', 'max_rf_phase']]


parser = OptionParser()

parser.add_option("--H0", type=float, default=70.,
                  help="H0 parameter[%default]")
parser.add_option("--Om0", type=float, default=0.3,
                  help="Omega0 parameter[%default]")
parser.add_option("--min_rf_phase", type=float, default=-15.,
                  help="min rf phase[%default]")
parser.add_option("--max_rf_phase", type=float, default=30.,
                  help="max rf phase[%default]")
parser.add_option("--season_length", type=float, default=180.,
                  help="season length[%default]")
parser.add_option("--zmin", type=float, default=0.01,
                  help="zmin [%default]")
parser.add_option("--zmax", type=float, default=1.1,
                  help="zmax [%default]")
parser.add_option("--dz", type=float, default=0.01,
                  help="dz for z binning [%default]")
parser.add_option("--survey_area", type=float, default=9.6,
                  help="survey area in deg2 [%default]")
parser.add_option("--account_for_edges", type=int, default=0,
                  help="to account for edges in nsn estimation [%default]")

opts, args = parser.parse_args()

H0 = opts.H0
Om0 = opts.Om0
min_rf_phase = opts.min_rf_phase
max_rf_phase = opts.max_rf_phase
season_length = opts.season_length
zmin = opts.zmin
zmax = opts.zmax
dz = opts.dz
survey_area = opts.survey_area
account_for_edges = opts.account_for_edges


# loop on rate types


rates = ['Perrett', 'Ripoche', 'Dilday', 'combined', 'Hounsell']
edges = [(-15, 25), (-15, 25), (-15, 30), (-15, 40)]
account_edges = [0, 1, 1, 1]
H0 = 70
Om0 = 0.3

params = vars(opts)

df = pd.DataFrame()
for rr in rates:
    params['rate'] = rr
    for i, vv in enumerate(account_edges):
        params['account_for_edges'] = vv
        params['min_rf_phase'] = edges[i][0]
        params['max_rf_phase'] = edges[i][1]
        tt = getNSN(**params)
        df = pd.concat((df, tt))


# selection eff
ida = df['edges'] == 0
sela = pd.DataFrame(df[ida])
idb = df['edges'] == 1
selb = pd.DataFrame(df[idb])

print(selb)
effi_df = selb.groupby(['rate', 'min_rf_phase', 'max_rf_phase']).apply(
    lambda x: effi(x, sela))

print(effi_df)

fig, ax = plt.subplots(figsize=(10, 8))
for rate in sela['rate'].unique():
    idx = sela['rate'] == rate
    sel = sela[idx]
    ax.errorbar(sel['z'], sel['nsn'], yerr=sel['err_nsn'], label=rate)
    idxb = sel['z'] <= 1.1
    print('total number of SN', rate, sel[idxb]['nsn'].max())

ax.grid()
ax.set_xlim(0.01, 1.1)
ax.legend()

"""
figb, axb = plt.subplots(figsize=(10, 8))
for rate in effi_df['rate'].unique():
    idx = effi_df['rate'] == rate
    sel = effi_df[idx]
    sel = sel.sort_values(by=['z'])
    for vv in edges[1:]:
        idc = sel['min_rf_phase'] == vv[0]
        idc &= sel['max_rf_phase'] == vv[1]
        selc = sel[idc]
        ll = '{} - {} - {}'.format(rate, vv[0], vv[1])
        axb.plot(selc['z'], selc['nsn_ratio'], label=ll)
axb.grid()
axb.set_xlim(0.01, 1.1)
axb.legend()
"""
plt.show()
