import pandas as pd
import matplotlib.pyplot as plt
import glob
from sn_analysis.sn_tools import load_cosmo_data, get_spline


dirs = ['../cosmo_fit_WFD_TiDES', '../cosmo_fit_WFD_TiDES_DESI',
        '../cosmo_fit_WFD_TiDES_DESI_DESI2']

timescale = 'year'
dbName = 'baseline_v3.0_10yrs'

df = pd.DataFrame()
for dd in dirs:
    spectro_config = dd.split('cosmo_fit_WFD_')[-1]
    print('akkk', spectro_config)
    res = load_cosmo_data(dd, dbName, timescale, spectro_config)
    df = pd.concat((df, res))

configs = df['spectro_config'].unique()
ls = dict(zip(configs, ['solid', 'dashed', 'dotted']))
colors = dict(zip(configs, ['lightgrey', 'silver', 'darkgrey']))
fig, ax = plt.subplots(figsize=(12, 8))


df['MoM_plus_sigma'] = df['MoM_mean']+df['MoM_std']
df['MoM_minus_sigma'] = df['MoM_mean']-df['MoM_std']

for conf in configs:
    idx = df['spectro_config'] == conf
    tt = df[idx]
    print(conf, tt[['MoM_mean', timescale]])
    tt = tt.sort_values(by=timescale)
    # ax.errorbar(tt[timescale], tt['MoM_mean'],
    #            yerr=tt['MoM_std'], linestyle=ls[conf], color='k')
    color = colors[conf]
    ax.fill_between(tt[timescale], tt['MoM_plus_sigma'],
                    tt['MoM_minus_sigma'], color=color)

    #xnew, ynew = get_spline(tt, timescale, 'MoM_mean')

    ax.plot(tt[timescale], tt['MoM_mean'], linestyle=ls[conf], color='k')

ax.grid(visible=True)
ax.set_xlabel(r'{}'.format(timescale))
ax.set_ylabel(r'SMoM')
plt.show()
