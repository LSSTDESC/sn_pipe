import glob
from sn_tools.sn_tools.sn_io import loopStack
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot(df_totb, yaxis='timeproc', ylabel=r'PT/pixel [sec]'):

    fig, ax = plt.subplots(figsize=(15, 8))

    itag = 0
    ns = []
    ymin = df_totb[yaxis].min()
    ymax = df_totb[yaxis].max()
    for vv in df_totb['zStep'].unique():
        idx = np.abs(df_totb['zStep']-vv) < 1.e-5
        sel = df_totb[idx]
        sel = sel.sort_values(by=['daymaxStep'])
        itag += 1
        plotFig(sel.to_records(index=False), ymin, ymax, yaxis=yaxis, ylabel=ylabel, ls='solid',
                ax=ax, itag=itag, yaxis_twin='')
        ns += sel['daymaxStep'].to_list()
    ax.set_xticklabels(ns)
    ax.set_xlabel(r'$\Delta T_0$ [day]', fontsize=20)
    ax.set_ylim([0.9*ymin, 1.08*ymax])
    ax.set_xlim([-0.7, None])
    ax.grid()


def plot2(df_totb, yaxis='timeproc', ylabel=r'PT/pixel [sec]', yaxisb='timeproc', ylabelb='tt'):

    fig, ax = plt.subplots(figsize=(15, 8))

    itag = 0
    ns = []
    ymin = df_totb[yaxis].min()
    ymax = df_totb[yaxis].max()
    for vv in df_totb['zStep'].unique():
        idx = np.abs(df_totb['zStep']-vv) < 1.e-5
        sel = df_totb[idx]
        sel = sel.sort_values(by=['daymaxStep'])
        itag += 1
        plotFig(sel.to_records(index=False), ymin, ymax, yaxis=yaxis, ylabel=ylabel, ls='solid',
                ax=ax, itag=itag, yaxis_twin='')
        plotFig(sel.to_records(index=False), ymin, ymax, yaxis=yaxisb, ylabel=ylabel, ls='dotted',
                ax=ax, itag=itag, yaxis_twin='')
        ns += sel['daymaxStep'].to_list()
    ax.set_xticklabels(ns)
    ax.set_xlabel(r'$\Delta T_0$ [day]', fontsize=20)
    ax.set_ylim([0.9*ymin, 1.08*ymax])
    ax.set_xlim([-0.7, None])
    ax.grid()


def plotFig(df, ymin, ymax, xaxis='config', yaxis='timeproc', ylabel=r'PT/pixel [sec]', yaxis_twin='timeproc_ratio', ytwinlabel=r'$\frac{PT^{ref}}{PT}$', figtitle='Processing Time (PT)', ls='None', marker='o', ax=None, itag=0, yerr=False):

    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 9))
        fig.subplots_adjust(bottom=0.25)
        fig.suptitle(figtitle)

    #ax.plot(df[xaxis], df[yaxis], color='k', ls=ls, marker=marker)
    yerr = None
    if yerr:
        yerr = df['{}_std'.format(yaxis)]

    ax.errorbar(df[xaxis], df[yaxis],
                yerr=yerr, color='k', ls=ls, marker=marker)

    # ax.grid()
    # ax.tick_params(axis='x', labelrotation=30.)
    ax.tick_params(axis='x')
    ax.set_ylabel(ylabel, fontsize=20)
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)
    # for tick in ax.xaxis.get_majorticklabels():
    #    tick.set_horizontalalignment("right")

    dd = 5.5*(itag-1)+0.5*(itag-2)
    ax.plot([dd, dd],
            [0.9*ymin, 1.1*ymax], ls='dotted', color='k', lw=2)
    ax.plot([dd+6., dd+6.],
            [0.9*ymin, 1.1*ymax], ls='dotted', color='k', lw=2)
    dz = np.round(df['zStep'].mean(), 3)
    ax.text(dd+1., 1.03*ymax, r'$\Delta z=$' +
            '{}'.format(dz), color='b', fontsize=15)

    # ax.plot([5.5*itag, 5.5*itag], [0., 70.], ls='dotted', color='k', lw=2)
    if yaxis_twin != '':
        axb = ax.twinx()
        axb.plot(df[xaxis], df[yaxis_twin],
                 color='b', ls=ls, marker=marker)
        axb.set_ylabel(ytwinlabel, fontsize=20, color='b')
        axb.tick_params(axis='y', labelsize=20, color='b')


zSteps = [0.005, 0.01, 0.02, 0.03, 0.04, 0.06, 0.08]
daymaxSteps = [1, 2, 3, 5, 10, 15]
zlim_coeff = 0.95
metricName = 'NSNY'
dbName = 'baseline_v2.0_10yrs'

df_tot = pd.DataFrame()
mainDir = 'Metric_check_100pixels'
# mainDir = '.'
for zStep in zSteps:
    for daymaxStep in daymaxSteps:
        metricDir = '{}/MetricOutput_{}_{}_{}'.format(mainDir,
                                                      zStep, daymaxStep, zlim_coeff)
        theDir = '{}/{}/{}/*.hdf5'.format(metricDir, dbName, metricName)
        print('looking for', theDir)
        fis = glob.glob(theDir)
        print(fis)
        df = loopStack(fis, objtype='astropyTable').to_pandas()
        df['zStep'] = zStep
        df['daymaxStep'] = daymaxStep
        df['zlim_coeff'] = zlim_coeff
        df['config'] = 'conf_{}_{}_{}'.format(zStep, daymaxStep, zlim_coeff)
        df_tot = pd.concat((df_tot, df))

print(df_tot)

# get the reference
idx = np.abs(df_tot['zStep']-0.005) <= 1.e-5
idx &= np.abs(df_tot['daymaxStep']-1.) <= 1.e-5

print(df_tot[idx])
df_ref = pd.DataFrame(
    df_tot[idx][['healpixID', 'season', 'zcomp', 'nsn', 'timeproc']])
df_ref = df_ref.rename(
    columns={'zcomp': 'zcomp_ref', 'nsn': 'nsn_ref', 'timeproc': 'timeproc_ref'})

print(df_ref)

df_tot = df_tot.merge(df_ref, left_on=['healpixID', 'season'], right_on=[
                      'healpixID', 'season'])

df_tot['timeproc_ratio'] = df_tot['timeproc_ref']/df_tot['timeproc']
df_tot['nsn_ratio'] = df_tot['nsn_ref']/df_tot['nsn']
df_tot['nsn_diff'] = (df_tot['nsn']-df_tot['nsn_ref'])/df_tot['nsn']
df_tot['dnsn'] = df_tot['dnsn']/df_tot['nsn']
df_tot['zcomp_diff'] = df_tot['zcomp_ref']-df_tot['zcomp']
df_tot['nsimu_sec'] = df_tot['nsimu']/df_tot['timeproc']

print(df_tot.columns)
vvmed = ['timeproc', 'timeproc_ratio', 'nsn_ratio', 'dnsn', 'dzcomp',
         'zcomp_diff', 'zStep', 'daymaxStep', 'nsimu', 'nsimu_sec', 'nsn_diff']
df_mean = df_tot.groupby(
    'config')[vvmed].mean().reset_index()

df_std = df_tot.groupby(
    'config')[vvmed].std().reset_index()

df_summary = df_mean.merge(df_std, left_on=['config'], right_on=[
    'config'], suffixes=['', '_std'])
df_summary = df_summary.round({'zStep': 3})
df_summary['daymaxStep'] = df_summary['daymaxStep'].astype(int)
df_summary = df_summary.sort_values(by=['zStep', 'daymaxStep'])
df_summary['sigma_zcomp'] = np.sqrt(
    df_summary['dzcomp']**2+df_summary['zcomp_diff_std']**2)
df_summary['sigma_nsn'] = np.sqrt(
    df_summary['dnsn']**2+df_summary['nsn_diff_std']**2)

print(df_summary[['config', 'timeproc_ratio',
                  'nsn_ratio', 'zcomp_diff', 'zcomp_diff_std', 'dzcomp', 'nsn_diff_std']])

plot(df_summary, yaxis='timeproc', ylabel=r'PT/pixel [sec]')
plot(df_summary, yaxis='timeproc_ratio', ylabel=r'$\frac{PT^{ref}}{PT}$')
plot(df_summary, yaxis='zcomp_diff',
     ylabel=r'$\Delta z_{comp}$')
plot(df_summary, yaxis='nsn_ratio',
     ylabel=r'$\frac{N_{SN}^{ref}}{N_{SN}}$')
plot(df_summary, yaxis='nsimu',
     ylabel=r'$N_{SN}^{simu}$')
plot(df_summary, yaxis='sigma_zcomp',
     ylabel=r'$\sigma_{z_{complete}}$ [stat+syst]')
plot2(df_summary, yaxis='dzcomp',
      ylabel=r'$\Delta z_{complete}$', yaxisb='zcomp_diff_std')
plot(df_summary, yaxis='sigma_nsn',
     ylabel=r'$\frac{\sigma_{N_{SN}}}{N_{SN}}$ [stat+syst]')
plot2(df_summary, yaxis='dnsn',
      ylabel=r'$\frac{\Delta N_{SN}}{N_{SN}}$', yaxisb='nsn_diff_std')

"""
plotFig(df_totb, xaxis='config', yaxis='zcomp_diff',
        ylabel=r'$\Delta z_{comp}$', yaxis_twin='', ytwinlabel='', figtitle='', ls='solid')
plotFig(df_totb, xaxis='config', yaxis='nsn_ratio',
        ylabel=r'$\frac{N_{SN}^{ref}}{N_{SN}}$', yaxis_twin='', ytwinlabel='', figtitle='', ls='solid')
"""
plt.show()
