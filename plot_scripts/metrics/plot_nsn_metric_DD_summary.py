import numpy as np
import sn_plotter_metrics.cadencePlot as sn_plot
import sn_plotter_metrics.nsnPlot as nsn_plot
from optparse import OptionParser
import numpy.lib.recfunctions as rf
import os
import csv
import pandas as pd
import scipy.stats
from sn_plotter_metrics import plt
from sn_plotter_metrics.utils import MetricValues
from sn_plotter_metrics.utils import dumpcsv_medcad, get_dist
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


def dumpcsv_pixels(metricTot, x1=-2.0, color=0.2, ebvofMW=0.0, snrmin=1.0, error_model=1, errmodrel=0.1, bluecutoff=380., redcutoff=800., simulator='sn_fast', fitter='sn_fast', Nvisits=dict(zip('grizy', [1, 1, 2, 2, 2]))):

    metricTot = pd.DataFrame(metricTot)
    idx = metricTot['nsn_med_faint'] >= 0
    ro = metricTot[idx]
    ro['healpixID'] = ro['healpixID'].astype(int)
    ro['season'] = ro['season'].astype(int)
    ro['tagprod'] = ro['healpixID'].astype(str) + '_' + \
        ro['season'].astype(str)
    ro['x1'] = x1
    ro['color'] = color
    ro['ebvofMW'] = ebvofMW
    ro['snrmin'] = snrmin
    ro['error_model'] = error_model
    ro['errmodrel'] = errmodrel
    ro['bluecutoff'] = bluecutoff
    ro['redcutoff'] = redcutoff
    ro['simulator'] = simulator
    ro['fitter'] = fitter
    for b in Nvisits.keys():
        ro['N{}'.format(b)] = Nvisits[b]
        cadname = 'cadence_{}'.format(b)
        ro[cadname] = ro['cadence']
        # ro[cadname] = ro[cadname].astype(int)
    for b in Nvisits.keys():
        nn = 'm5_{}'.format(b)
        ro = ro.rename(columns={'m5_med_{}'.format(b): nn})
        ro = ro.round({nn: 2})

    for b in 'grizy':
        ro = ro.round({'cadence_{}'.format(b): 2})

    print(ro.columns)
    tocsv_vars = ['tagprod', 'x1', 'color', 'ebvofMW', 'snrmin', 'error_model',
                  'errmodrel', 'bluecutoff', 'redcutoff', 'simulator', 'fitter',
                  'Ng', 'Nr', 'Ni', 'Nz', 'Ny', 'm5_g', 'm5_r', 'm5_i', 'm5_z', 'm5_y',
                  'cadence_g', 'cadence_r', 'cadence_i', 'cadence_z', 'cadence_y',
                  'season', 'healpixID', 'season_length']

    max_rows = 500
    dataframes = []
    df = ro[tocsv_vars]
    while len(df) > max_rows:
        top = df[:max_rows]
        dataframes.append(top)
        df = df[max_rows:]
    else:
        dataframes.append(df)

    for i, dd in enumerate(dataframes):
        dd.to_csv('config_DD_obs_pixels_{}.csv'.format(i), index=False)


def fill(selb, tagprod='', x1=-2.0, color=0.2, ebvofMW=0.0, snrmin=1.0, error_model=1, errmodrel=0.1, bluecutoff=380., redcutoff=800., simulator='sn_fast', fitter='sn_fast', Nvisits=dict(zip('grizy', [1, 1, 2, 2, 2])), cadence=dict(zip('grizy', [1, 1, 1, 1, 1]))):

    fmed = []
    r = pd.DataFrame()
    for b in 'grizy':
        fmed.append('m5_med_{}'.format(b))

    ro = selb.groupby(['season'])[fmed].median().reset_index()

    ro['tagprod'] = tagprod
    ro['x1'] = x1
    ro['color'] = color
    ro['ebvofMW'] = ebvofMW
    ro['snrmin'] = snrmin
    ro['error_model'] = error_model
    ro['errmodrel'] = errmodrel
    ro['bluecutoff'] = bluecutoff
    ro['redcutoff'] = redcutoff
    ro['simulator'] = simulator
    ro['fitter'] = fitter
    for b in Nvisits.keys():
        ro['N{}'.format(b)] = Nvisits[b]
        ro['cadence_{}'.format(b)] = cadence[b]

    for b in Nvisits.keys():
        nn = 'm5_{}'.format(b)
        ro = ro.rename(columns={'m5_med_{}'.format(b): nn})
        ro = ro.round({nn: 2})

    ro['season'] = ro['season'].astype(int)
    ro['tagprod'] = ro['tagprod'] + '_' + \
        ro['season'].astype(str)+'_'+ro['cadence_z'].astype(str)

    return ro


def plotAllBinned(ax, metricTot, forPlot=pd.DataFrame(), xp='cadence', yp='nsn_med_faint', legx='cadence [day]', legy='N$_{\mathrm{SN}} ^ {z \leq z_{\mathrm{complete}}}$', bins=10, therange=(0.5, 10.5), yerrplot=False, legend=True, drawrec={}):

    print(forPlot)
    metricTot = pd.DataFrame(metricTot)
    idx = metricTot['nsn_med_faint'] >= 0
    sel = metricTot[idx]
    # plt.plot(sel['cadence'], sel['nsn_med_faint'], 'ko')
    print(sel[['cadence', 'nsn_med_faint']])
    # fig, ax = plt.subplots(figsize=(10, 10))
    # fig.subplots_adjust(top=0.85)
    dbNames = np.unique(sel['dbName'])
    lsb = dict(zip(dbNames, ['solid', 'dashed', 'dotted',
               'dashdot', (0, (5, 1)), (0, (3, 10, 1, 10)), (0, (5, 10))]))
    for dbName in dbNames:
        ij = sel['dbName'] == dbName
        selb = sel[ij]
        ic = forPlot['dbName'] == dbName
        color = forPlot[ic]['color'].tolist()[0]
        name = '_'.join(dbName.split('_')[:2])

        if name.split('_')[0] in ['descddf', 'baseline', 'daily', 'agnddf']:
            name = name.split('_')[0]

        if name == 'ddf_dither0.00':
            name = name.split('_')[1]
        plotBinned(ax, selb, xp=xp, yp=yp,
                   label=name, color=color, yerrplot=yerrplot, bins=bins, therange=therange, ls=lsb[dbName])

    ax.grid()
    weight = 'normal'
    ax.set_xlabel(legx, weight=weight)
    ax.set_ylabel(legy, weight=weight)
    if legend:
        ax.legend(frameon=False)
        ax.legend(loc='upper left', bbox_to_anchor=(
            -0.05, 1.42), ncol=3, frameon=False)
    ylims = np.copy(ax.get_ylim())
    xlims = np.copy(ax.get_xlim())
    if xp == 'cadence':
        ylims[0] = 0
    if yp == 'cadence':
        ylims[0] = 1
    print('alors', xlims, ylims)
    if drawrec:
        ax.fill_between(drawrec['x'], drawrec['y'], color='yellow', alpha=0.1)

    print('alors', xlims, ylims)
    ax.set_xlim([xlims[0], xlims[1]])
    ax.set_ylim([ylims[0], ylims[1]])


def plotAllBinned_old(metricTot, xp='cadence', yp='nsn_med_faint', legx='cadence [day]', legy='$N_{SN} ^ {z < z_{complete}}$'):

    metricTot = pd.DataFrame(metricTot)
    idx = metricTot['nsn_med_faint'] >= 0
    sel = metricTot[idx]
    # plt.plot(sel['cadence'], sel['nsn_med_faint'], 'ko')
    print(sel[['cadence', 'nsn_med_faint']])
    fig, ax = plt.subplots(ncols=2, nrows=6, sharex=True)
    fig.subplots_adjust(hspace=0)
    # zlim_theo = pd.read_csv('config_z_0.csv', comment="#")
    # zlim_theo['fieldName'] = zlim_theo['tagprod'].str.split('_').str.get(0)
    ff = ['COSMOS', 'XMM-LSS', 'ELAIS', 'CDFS', 'ADFS1', 'ADFS2']
    ipos = [0, 1, 2, 3, 4, 5, 6]
    iposi = dict(zip(ff, ipos))
    for fieldName in np.unique(sel['fieldname']):
        # fig, ax = plt.subplots()
        ipp = iposi[fieldName]
        ij = sel['fieldname'] == fieldName
        selb = sel[ij]
        plotBinned(ax[ipp, 0], selb, xp=xp, yp=yp, label=fieldName)
        plotBinned(ax[ipp, 1], selb, xp=xp,
                   yp='nsn_med_faint', label=fieldName)
        # ax[ipp,0].plot(selb['cadence'],selb['zlim_faint'],'k.')
        """
        idx = zlim_theo['fieldName'] == fieldName
        selz = zlim_theo[idx]
        mink = selz.groupby(['cadence_z']).min().reset_index()
        maxk = selz.groupby(['cadence_z']).max().reset_index()
        # mink = mink.sort_values(by=['cadence_z'],ascending=False)
        print(mink)
        print(maxk)
        # maxk = pd.concat((maxk,mink))
        print(maxk)
        # ax.plot(selz['cadence_z'], selz['zlim'], 'ko')
        ax[ipp, 0].fill_between(
            mink['cadence_z'], mink['zlim'], maxk['zlim'], color='yellow')
        ax[ipp, 0].grid()
        # break
        """
        # ax.set_xlabel(legx, fontweight='bold')
        # ax.set_ylabel(legy, fontweight='bold')
    # ax.legend()

    plt.show()


def plotBinned(ax, metricTot, xp='cadence', yp='nsn_med_faint', label='', color='k', yerrplot=True, bins=8, therange=(0.5, 8.5), ls='solid'):

    x = metricTot[xp]
    y = metricTot[yp]

    means_result = scipy.stats.binned_statistic(
        x, [y, y**2], bins=bins, range=therange, statistic='mean')
    means, means2 = means_result.statistic
    standard_deviations = np.sqrt(means2 - means**2)
    bin_edges = means_result.bin_edges
    bin_centers = (bin_edges[:-1] + bin_edges[1:])/2.

    yerr = standard_deviations
    if not yerrplot:
        yerr = None
    ax.errorbar(x=bin_centers, y=means, yerr=yerr,
                marker='.', label=label, color=color, ls=ls, linewidth=3)


def cadenceTable(metricTot):

    cadences = [1., 2., 3., 4., 5.]
    ncads = len(cadences)
    metricTot = pd.DataFrame(metricTot)
    idx = metricTot['nsn_med_faint'] >= 0
    sel = metricTot[idx]
    r = []
    for dbName in np.unique(sel['dbName']):
        ij = sel['dbName'] == dbName
        selb = sel[ij]
        name = '_'.join(dbName.split('_')[:2])
        if name.split('_')[0] in ['descddf', 'baseline', 'daily', 'agnddf']:
            name = name.split('_')[0]
        if name == 'ddf_dither0.00':
            name = name.split('_')[1]

        nevts = len(selb)

        for il in range(ncads):
            ido = selb['cadence'] >= cadences[il]
            if il < ncads-1:
                ido &= selb['cadence'] < cadences[il+1]

            nn = len(selb[ido])
            print(name, cadences[il], nn/nevts)
            r.append((name, cadences[il], 100.*nn/nevts))

    print(r)
    df = pd.DataFrame(r, columns=['dbName', 'cadence', 'frac'])
    df['cadence'] = df['cadence'].astype(int)
    df = df.round({'frac': 1})

    lig = []
    lig.append('\\begin{table}[!htbp]')
    lig.append(
        '\\caption{Cadence distribution for a set of strategies studied in this paper.}\\label{tab:cadencesum}')
    lig.append('\\begin{center}')
    lig.append('\\begin{tabular}{c|c|c|c|c|c}')
    lig.append('\\hline')
    lig.append('\\hline')
    lig.append(
        '\\diagbox[innerwidth=3.cm,innerleftsep=-1.cm,height=3\line]{Strategy}{cadence \\n [day]} & 1 & 2 & 3 & 4 & $\geq$ 5\\\\')
    lig.append('\\hline')

    for dbName in df['dbName'].unique():
        idx = df['dbName'] == dbName
        sel = df[idx]
        lbg = dbName.replace('_', '\\_') + ' & '
        for cad in range(1, 6):
            ib = sel['cadence'] == cad
            selb = sel[ib]
            lbg += '{}\\%'.format(str(selb['frac'].tolist()[0]))
            if cad < 5:
                lbg += ' & '
            else:
                lbg += ' \\\\ '
        print(lbg)
        lig.append(lbg)

    # lig.append('\\hline')
    lig.append('\\end{tabular}')
    lig.append('\\end{center}')
    lig.append('\\end{table}')

    fia = open('cadenceTable.tex', 'w')
    for vv in lig:
        print(vv)
        fia.write(vv+' \n')

    fia.close()


def zcomp_frac(grp, frac=0.95):

    nsn = np.sum(grp['nsn'])
    selfi = get_dist(grp)
    nmax = np.max(selfi['nsn'])

    idx = selfi['nsn'] <= frac*nmax

    res = selfi[idx]
    if len(res) > 0:
        dist_cut = np.min(res['dist'])
        idd = selfi['dist'] <= dist_cut
        zcomp = np.median(selfi[idd]['zcomp'])
    else:
        zcomp = np.median(selfi['zcomp'])

    """
    print('zcomp_frac', grp.name, dist_cut,
              nmax, selfi[['zcomp', 'nsn', 'dist']])
    fig, ax = plt.subplots()
    ax.plot(selfi['dist'], selfi['nsn'], 'ko')
    plt.show()
    """

    return pd.DataFrame({'nsn': [nsn], 'zcomp': [zcomp]})


def zcomp_weighted(grp):

    nsn = np.sum(grp['nsn'])
    zcomp = np.sum(grp['nsn']*grp['zcomp'])/nsn

    return pd.DataFrame({'nsn': [nsn], 'zcomp': [zcomp]})


def zcomp_cumsum(grp, frac=0.95):

    xvar, yvar = 'zcomp', 'nsn'
    nsn = np.sum(grp[yvar])
    from scipy.interpolate import interp1d

    selp = grp.sort_values(by=[xvar], ascending=False)
    print('ee', grp.name, selp[[xvar, yvar]])
    if len(selp) >= 2:
        cumulnorm = np.cumsum(selp[yvar])/np.sum(selp[yvar])
        interp = interp1d(
            cumulnorm, selp[xvar], bounds_error=False, fill_value=0.)
        zcompl = interp(frac)
        io = selp[xvar] >= zcompl
        zcomp = np.median(selp[io][xvar])
    else:
        zcomp = np.median(selp[xvar])

    return pd.DataFrame({'nsn': [nsn], 'zcomp': [zcomp]})


parser = OptionParser(
    description='Display (NSN,zlim) metric results for DD fields')
parser.add_option("--dirFile", type="str",
                  default='/sps/lsst/users/gris/MetricOutput',
                  help="file directory [%default]")
parser.add_option("--nside", type="int", default=64,
                  help="nside for healpixels [%default]")
parser.add_option("--fieldType", type="str", default='DD',
                  help="field type - DD, WFD, Fake [%default]")
parser.add_option("--dbList", type="str", default='plot_scripts/cadenceCustomize_fbs14.csv',
                  help="list of cadences to display[%default]")
parser.add_option("--snType", type="str", default='faint',
                  help="SN type: faint or medium[%default]")
parser.add_option("--outName", type="str", default='Summary_DD_fbs14.npy',
                  help="output name for the summary[%default]")
parser.add_option("--fieldNames", type="str", default='COSMOS,CDFS,XMM-LSS,ELAISS1,EDFSa,EDFSb',
                  help="fields to process [%default]")
parser.add_option("--metric", type="str", default='NSNY',
                  help="metric name [%default]")
parser.add_option("--prefix_csv", type="str", default='metric_summary_DD',
                  help="prefix for csv output files [%default]")

opts, args = parser.parse_args()

# Load parameters
dirFile = opts.dirFile
nside = opts.nside
fieldType = opts.fieldType
metricName = opts.metric
snType = opts.snType
outName = opts.outName
fieldNames = opts.fieldNames.split(',')
prefix_csv = opts.prefix_csv

# Loading input file with the list of cadences to take into account and siaplay features
filename = opts.dbList

# forPlot = pd.read_csv(filename).to_records()
forPlot = pd.read_csv(filename, comment='#')

print(forPlot)


# get DD fields
# fields_DD = getFields(5.)
"""
fields_DD = DDFields()

lengths = [len(val) for val in forPlot['dbName']]
adjl = np.max(lengths)


metricTot = None
metricTot_med = None
"""


# Summary: to reproduce the plots faster


metricTot = MetricValues(dirFile, forPlot['dbName'].to_list(), metricName,
                         fieldType, fieldNames, nside).data

# figs 6 and 7
"""
fig, ax = plt.subplots(figsize=(9, 16), nrows=2)
fig.subplots_adjust(top=0.85)
drawrec = {}
drawrec['x'] = [0, 3, 3, 0]
drawrec['y'] = [0., 0., 3., 3.]
plotAllBinned(ax[0], metricTot, forPlot, drawrec=drawrec)
drawrecb = {}
drawrecb['x'] = [1., 15., 15., 1.]
drawrecb['y'] = [1., 1., 3., 3.]
plotAllBinned(ax[1], metricTot, forPlot, xp='gap_max', yp='cadence',
              legx='max inter-night gap [day]', legy='cadence [day]', bins=10, therange=(0.5, 60.5), legend=False, drawrec=drawrecb)
plt.show()
"""
# dumpcsv_pixels(metricTot)
# plotAllBinned(metricTot, yp='zlim_faint', legy='$z_{complete}^{0.95}$')
# plt.tight_layout()
# plt.show()

"""
cadenceTable(metricTot)
"""

"""
print('oo', np.unique(
    metricTot[['dbName', 'fieldname', 'zlim_faint']]), type(metricTot))
"""
# fieldNames = ['COSMOS', 'CDFS', 'XMM-LSS', 'ELAIS', 'ADFS1', 'ADFS2']
# fieldNames = ['COSMOS']
# nsn_plot.plot_DDArea(metricTot, forPlot, sntype='faint')

# df = pd.DataFrame(np.copy(metricTot))
# print('jj', type(metricTot), metricTot.columns)
df = metricTot
var = 'nsn'
varz = 'zcomp'
idx = df[var] > 0.
idx &= df[varz] > 0.

dfb = df[idx].groupby(['dbName', 'fieldname', 'season'])[
    var].sum().reset_index()
idx = dfb['dbName'] != ''
ssel = dfb[idx]
for i, row in ssel.iterrows():
    print(row[['dbName', 'fieldname', 'season', var]].values)

# print(test)


idx = metricTot[var] > 0.
idx &= metricTot[varz] > 0.

metricPlot = metricTot[idx]

print(metricPlot.columns)
# estimate new zcomp (for the plot)

# metricPlot = metricPlot.groupby(['dbName', 'season', 'fieldname']).apply(
#    lambda x: zcomp_weighted(x)).reset_index()

# metricPlot = tt.to_records(index=False)
metricPlotb = metricPlot.merge(forPlot, left_on=['dbName'],right_on=['dbName'])
dumpcsv_medcad(metricPlotb,prefix=prefix_csv)
# print(test)
# nsn_plot.plot_DDSummary(metricPlot, forPlot, sntype=snType,
#                        fieldNames=fieldNames, nside=nside, figtit=opts.fieldNames)

summary = metricPlot.groupby(['dbName']).agg({'nsn': 'sum',
                                              'zcomp': 'median',
                                              }).reset_index()

if fieldType == 'DD':
    summary_season = metricPlot.groupby(['dbName', 'fieldname', 'season']).apply(
        lambda x: zcomp_frac(x)).reset_index()

    # print(test)
    summary_field = summary_season.groupby(['dbName', 'fieldname']).apply(
        lambda x: zcomp_cumsum(x)).reset_index()

    summary = summary_field.groupby(['dbName']).agg({'nsn': 'sum',
                                                     'zcomp': 'median',
                                                     }).reset_index()
nsn_plot.plotNSN(summary, forPlot, varx='zcomp', vary='nsn',
                 legx='${z_{\mathrm{complete}}}$', legy='N$_{\mathrm{SN}} (z<z_{\mathrm{complete}})}$', figtit=opts.fieldNames)


# nsn_plot.plot_DD_Moll(metricTot, 'ddf_dither0.00_v1.7_10yrs', 1, 128)
# nsn_plot.plot_DD_Moll(metricTot, 'descddf_v1.5_10yrs', 1, 128)
plt.show()

# print(test)
fontsize = 15
fields_DD = DDFields()
# print(metricTot[['cadence','filter']])

# grab median values
"""
df = pd.DataFrame(np.copy(metricTot)).groupby(
    ['healpixID','fieldnum','filter','cadence']).median().reset_index()

# print(df)
metricTot = df.to_records(index=False)
idx = metricTot['filter']=='all'
sel = metricTot[idx]
"""

figleg = 'nside = {}'.format(nside)
# sn_plot.plotDDLoop(nside, dbNames, metricTot, 'zlim_faint',
#                   '$z_{lim}^{faint}$', markers, colors, mfc, adjl, fields_DD, figleg)
# sn_plot.plotDDLoop(nside,dbNames,sel,'cadence_mean','cadence [days]',markers,colors,mfc,adjl,fields_DD,figleg)

# fig,ax = plt.subplots()


# print(metricTot.dtype,type(metricTot))


# sn_plot.plotDDLoopCorrel(nside,dbNames,metricTot,'zlim_faint','zlim_medium','$z_{lim}^{faint}$','$z_{lim}^{med}$',markers,colors,mfc,adjl,fields_DD,figleg)

# sn_plot.plotDDLoopCorrel(nside,dbNames,metricTot,'nsn_med_zfaint','nsn_med_zmedium','$z_{lim}^{faint}$','$z_{lim}^{med}$',markers,colors,mfc,adjl,fields_DD,figleg)

# sn_plot.plotDDFit(metricTot,'nsn_med_zmedium','nsn_zmedium')
# sn_plot.plotDDFit(metricTot,'zlim_faint','zlim_medium')

# sn_plot.plotDDLoopCorrel(nside,dbNames,metricTot,'nsn_med_zfaint','nsn_med_zmedium','$z_{lim}^{faint}$','$z_{lim}^{med}$',markers,colors,mfc,adjl,fields_DD,figleg)

# sn_plot.plotDDFit(metricTot,'nsn_med_zmedium','nsn_med_zfaint','zlim_medium','zlim_faint')


df = pd.DataFrame(metricTot)
dbNames = np.unique(df['cadence'])
"""
print(df.columns)
# sums = df.groupby(['cadence','season'])['pixArea'].sum().reset_index()
sums = df.groupby(['fieldname', 'cadence', 'season'])[
                  'pixArea'].sum().reset_index()

idx = sums['pixArea'] > 1.
sn_plot.plotDDLoop(nside, dbNames, sums[idx], 'pixArea', 'area [deg2]',
                   mmarkers, colors_cadb, mfc_cad, adjl, fields_DD, figleg)

"""
# plt.show()


"""
for band in 'grizy':
    idx = metricTot['filter']==band
    sel = metricTot[idx]
    figlegb = '{} - {} band'.format(figleg,band)
    sn_plot.plotDDLoop(nside,dbNames,sel,'visitExposureTime',
                       'Exposure Time [s]',markers,colors,mfc,adjl,fields_DD,figlegb)
"""
filtercolors = 'cgyrm'
filtermarkers = ['o', '*', 's', 'v', '^']
mfiltc = ['None']*len(filtercolors)

"""
vars = ['visitExposureTime', 'cadence_mean', 'gap_max', 'gap_5']
legends = ['Exposure Time [sec]/night', 'cadence [days]',
           'max gap [days]', 'frac gap > 5 days']
"""
vars = ['N_total', 'cadence', 'gap_max', 'gap_med']
legends = ['Number of visits', 'cadence [days]',
           'max gap [days]', 'med gap [day]']

print(df.columns)


todraw = dict(zip(vars, legends))
for dbName in dbNames:
    for key, vals in todraw.items():
        sn_plot.plotDDCadence_new(df, dbName, key, vals)


"""

for season in np.unique(sel['season']):
    idf = (sel['season'] == season)&(sel['season_length']>10.)
    selb = sel[idf]
    plt.plot(selb['fieldnum'],selb['season_length'],marker='.',
             lineStyle='None',label='season {}'.format(season))

plt.legend()
plt.xticks(fields_DD['fieldnum'], fields_DD['fieldname'], fontsize=fontsize)
"""
plt.show()


print(metricTot.dtype)
