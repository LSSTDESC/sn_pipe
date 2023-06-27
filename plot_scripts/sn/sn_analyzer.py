import matplotlib.pyplot as plt
from sn_analysis.sn_tools import *
from sn_analysis.sn_calc_plot import Calc_zlim, histSN_params, select
from sn_analysis.sn_analysis import sn_load_select, get_nsn, nsn_vs_sel
from sn_plotter_analysis.plotNSN import plotNSN, plot_NSN
from sn_tools.sn_io import checkDir
from sn_analysis.sn_analysis import processNSN, processNSN_z
from sn_analysis.sn_selection import selection_criteria

from optparse import OptionParser
import numpy as np
import pandas as pd
import operator


def plotNSN_please(process_data, dd, dbDir, prodType,
                   listDDF, dict_sel, outDir, norm_factor):
    """
    Function to plot the number of SN

    Parameters
    ----------
    process_data : int
        To process data.
    dd : pandas df
        list of DBname to process (plus plot params).
    dbDir : str
        location dir of the dbs to process.
    prodType : str
        priduction type (DDF_spectroz/DDF_photoz).
    listDDF : list(str)
        list of DDF to process.
    dict_sel : dict
        Selection dict.
    outDir : str
        output dir for files+plot.
    norm_factor : float
        normalization factor.

    Returns
    -------
    None.

    """

    if process_data:
        processNSN(dd, dbDir, prodType, listDDF, dict_sel, outDir, norm_factor)

    plt_NSN = plotNSN(listDDF, dd, selconfig, selconfig_ref,
                      plotDir=plotDir, fDir=outDir)

    plt_NSN.plot_NSN_season()
    plt_NSN.plot_NSN_season_cumul()
    plt_NSN.plot_observing_efficiency()


class Plot_simu_params:
    def __init__(self, dd, dbDir, prodType,
                 listDDF, dict_sel, fDir, norm_factor):
        """
        class to plot simulation params

        Parameters
        ----------
        dd : pandas df
            data to process.
        dbDir : str
            location dir of the files to process.
        prodType : str
            production type.
        listDDF : list(str)
            list of DDFs to consider.
        dict_sel : dict
            selection dict.
        fDir : str
            location dir of the files.
        norm_factor : float
            Normalization factor.

        Returns
        -------
        None.

        """

        self.dd = dd
        self.dbDir = dbDir
        self.prodType = prodType
        self.listDDF = listDDF
        self.dict_sel = dict_sel
        self.fDir = fDir
        self.norm_factor = norm_factor

        data = self.loadData()

        self.stat(data)

    def loadData(self):

        restot = pd.DataFrame()
        for io, row in self.dd.iterrows():
            dbName = row['dbName']

            dd = sn_load_select(self.dbDir, dbName, self.prodType,
                                listDDF='COSMOS,CDFS,XMM-LSS,ELAISS1,EDFSa,EDFSb',
                                fDir=self.fDir, norm_factor=self.norm_factor)

            res = dd.data
            restot = pd.concat((restot, res))

        print(restot, len(restot))

        return restot

    def stat(self, res):
        stat = res.groupby(['dbName', 'field', 'season']).apply(
            lambda x: pd.DataFrame({'nsn': [len(x)]})).reset_index()

        print(stat)

        res['chisq_ndof'] = res['chisq']/res['ndof']

        selconfig = 'G10_JLA'
        selconfig = 'G10_sigmaC'
        data = select(res, self.dict_sel[selconfig])
        dataref = select(res, self.dict_sel['nosel'])

        idx = data['season'] == 2
        sel = data[idx]
        idxb = dataref['season'] == 2
        selref = dataref[idxb]

        idc = selref['SNID'].isin(sel['SNID'].to_list())
        selnot = selref[~idc]
        sel = sel.sort_values(by=['daymax'], ascending=False)
        selnot = selnot.sort_values(by=['daymax'], ascending=False)

        print(sel.columns)
        vv = ['SNID', 'daymax', 'z', 'sigmat0', 'sigmax1', 'chisq_ndof']
        print(sel[vv])
        print('not selected', selnot[vv])

        rr = data.groupby(['dbName', 'field', 'season']).apply(
            lambda x: self.effi(x, dataref))

        print(rr)

    def effi(self, grp, data_ref):

        nn = grp.name
        idx = data_ref['dbName'] == nn[0]
        idx &= data_ref['field'] == nn[1]
        idx &= data_ref['season'] == nn[2]

        sel_ref = data_ref[idx]

        eff = len(grp)/len(sel_ref)

        dd = pd.DataFrame({'nsn': [len(grp)], 'effi': [eff]})

        return dd


def plot_nsn_selcriteria(data, dd, listDDF, selcriteria='G10_JLA'):
    """
    Function to plot the number of SNe Ia as a function of the selection criteria

    Parameters
    ----------
    data : pandas df
        data to plot.
    dd : pandas df
        OS to process.
    selcriteria : str, optional
        Selection criteria chosen. The default is 'G10_JLA'.

    Returns
    -------
    None.

    """

    idx = data['seldict'] == selcriteria

    df = pd.DataFrame(data[idx])

    # df = df.merge(dd, left_on=['name'], right_on=['dbName'])

    dbNames = df['name'].unique()

    fig, ax = plt.subplots(figsize=(12, 9))
    fig.subplots_adjust(top=0.9, right=0.8)
    fig.suptitle(listDDF)
    ccols = ['seldict', 'sel', 'cutnum', 'NSN', 'NSN_ref', 'err_NSN', 'name']

    cols_orig = ['fitstatus', 'n_bands_m8_p10',
                 'n_epochs_m10_p35',
                 'n_epochs_m10_p5',
                 'n_epochs_p5_p20',
                 'n_epochs_phase_minus_10',
                 'n_epochs_phase_plus_20',
                 'sigmat0', 'sigmax1', 'nosel', 'sigmaC']
    cols_new = ['fit ok', 'n$_{bands}^{-8\leq p \leq +10}\geq 2$',
                'n$_{epochs}^{-10\leq p \leq +35}\geq 4$',
                'n$_{epochs}^{-10\leq p \leq +5}\geq 1$',
                'n$_{epochs}^{+5\leq p \leq +20}\geq 1$',
                'n$_{epochs}^{p \leq -10}\geq 1$',
                'n$_{epochs}^{p \geq +20}\geq 1$',
                '$\sigma_{T_0} \leq$2',
                '$\sigma_{x_1}\leq$1',
                '', '$\sigma_C \leq 0.04$']
    newcols = pd.DataFrame(cols_orig, columns=['sel'])
    newcols['selnew'] = cols_new

    print('oo', newcols)

    for dbName in dbNames:
        idx = df['name'] == dbName
        sel = df[idx]
        idxb = dd['dbName'] == dbName
        selconf = dd[idxb]

        r = [(selcriteria, 'nosel', 0, sel['NSN_ref'].mean(),
              sel['NSN_ref'].mean(), 0, dbName)]
        df_ref = pd.DataFrame(r, columns=ccols)

        sel = pd.concat((sel, df_ref))
        sel = sel.merge(newcols, left_on=['sel'], right_on=['sel'])

        sel = sel.sort_values(by=['cutnum'])
        ls = selconf['ls'].unique()[0]
        marker = selconf['marker'].unique()[0]
        color = selconf['color'].unique()[0]

        ax.errorbar(sel['selnew'], sel['NSN'], yerr=sel['err_NSN'],
                    linestyle=ls, marker=marker,
                    label=dbName, mfc='None', color=color)

    ax.grid()
    ax.tick_params(axis='x', labelrotation=20., labelsize=14)
    ax.set_xlim([0, None])
    ax.legend(loc='upper center',
              bbox_to_anchor=(1.15, 0.7),
              ncol=1, fontsize=12, frameon=False)
    ax.set_ylabel('NSN')
    # plt.tight_layout()


parser = OptionParser(description='Script to analyze SN prod')

parser.add_option('--dbDir', type=str, default='../Output_SN',
                  help='OS location dir[%default]')
parser.add_option('--dbList', type=str,
                  default='input/DESC_cohesive_strategy/config_ana.csv',
                  help='OS name[%default]')
parser.add_option('--prodType', type=str, default='DDF_spectroz',
                  help='type prod (DDF_spectroz/DDF_photoz) [%default]')
parser.add_option('--norm_factor', type=float, default=30.,
                  help='normalization factor on prod [%default]')
parser.add_option('--process_data', type=int, default=1,
                  help='to process data[%default]')
parser.add_option('--listDDF', type=str,
                  default='COSMOS,CDFS,XMM-LSS,ELAISS1,EDFSa,EDFSb',
                  help='DDF list to process [%default]')
parser.add_option('--selconfig', type=str,
                  default='G10_sigmaC',
                  help='selection for plot [%default]')
parser.add_option('--selconfig_ref', type=str,
                  default='nosel',
                  help='ref data for efficiency plot [%default]')
parser.add_option('--plotDir', type=str,
                  default='Plot_OS',
                  help='output directory for the plots [%default]')
parser.add_option('--outDir', type=str,
                  default='SN_analysis',
                  help='output directory for the produced files [%default]')

opts, args = parser.parse_args()


dbDir = opts.dbDir
dbList = opts.dbList
prodType = opts.prodType
norm_factor = opts.norm_factor
process_data = opts.process_data
listDDF = opts.listDDF
selconfig = opts.selconfig
selconfig_ref = opts.selconfig_ref
plotDir = opts.plotDir
outDir = opts.outDir

checkDir(outDir)
# res_fast = load_complete('Output_SN_fast', dbName)


dict_sel = selection_criteria()

# load the dbName, etc, to process
dd = pd.read_csv(dbList, comment='#')

gime_simuParam = False
gime_plotNSN = True
gime_plotNSN_selcriteria = False
gime_plotNSN_vs_z = False


if gime_simuParam:
    Plot_simu_params(dd, dbDir, prodType,
                     listDDF, dict_sel, outDir, norm_factor)

if gime_plotNSN:
    plotNSN_please(process_data, dd, dbDir, prodType,
                   listDDF, dict_sel, outDir, norm_factor)

if gime_plotNSN_selcriteria:
    res = nsn_vs_sel(dd, dbDir, prodType, listDDF,
                     dict_sel, outDir, norm_factor)

    plot_nsn_selcriteria(res, dd, listDDF)

if gime_plotNSN_vs_z:
    ro = processNSN_z(dd, dbDir, prodType,
                      listDDF, dict_sel, outDir, norm_factor, seldict=selconfig)

    ro = ro.merge(dd, left_on=['dbName'], right_on=['dbName'])

    title = '{} \n {}'.format(listDDF, selconfig)
    plot_NSN(ro,
             xvar='z', xlabel='z',
             yvar='NSN', ylabel='$N_{SN}$',
             bins=None, norm_factor=1, loopvar='dbName',
             dict_sel={},
             title=title, plotDir='', plotName='')

plt.show()
print(test)


"""
print('aooo', len(res), res.columns, np.unique(res['healpixID']))
histSN_params(res)
plotSN_2D(res)
print(res.columns)
"""


# selectc only fitted LC
idx = res['fitstatus'] == 'fitok'
res = res[idx]


sel = {}
# select data here
sel['noCut'] = res
for selvar in ['G10_sigmaC', 'G10_JLA']:
    sel[selvar] = select(res, dict_sel[selvar])

nsn_fields = pd.DataFrame()
for key, vals in sel.items():
    # dd['NSN'] = vals.groupby(['field']).transform('size').reset_index()
    dd = vals.groupby(['field']).size().to_frame('NSN').reset_index()
    dd['NSN'] /= norm_factor
    dd['selconfig'] = key
    nsn_fields = pd.concat((nsn_fields, dd))

print(nsn_fields)
print(test)

# print the results
pp = {}
for field in fields:
    idx = res['field'] == field
    pp['nSN_tot'] = len(res[idx])/norm_factor
    idxb = sel['field'] == field
    idxc = selb['field'] == field
    print(field, len(res[idx])/norm_factor, len(sel[idxb]) /
          norm_factor, len(selb[idxc])/norm_factor)


for field in fields:
    idx = sel['field'] == field
    sela = sel[idx]
    for season in np.unique(sela['season']):
        idxb = sela['season'] == season
        selc = sela[idxb]
        print(field, season, len(selc)/norm_factor)

print(len(sel)/norm_factor)

histSN_params(sel)
plotSN_2D(sel)
plot_NSN(sel, norm_factor=norm_factor)
idx = sel['z'] >= 0.7
idxb = selb['z'] >= 0.7

print('high-z', len(sel[idx])/norm_factor, len(selb[idxb])/norm_factor)
plt.show()

"""
sel = select(res, dict_sel['G10'])
fig, ax = plt.subplots()
xmin = sel['n_epochs_aft'].min()
xmax = sel['n_epochs_aft'].max()
idx = sel['z'] > 0.5
ax.hist(sel[idx]['n_epochs_aft'], histtype='step', bins=range(xmin, xmax))

fig, ax = plt.subplots()
xmin = sel['n_epochs_bef'].min()
xmax = sel['n_epochs_bef'].max()
ax.hist(sel[idx]['n_epochs_bef'], histtype='step', bins=range(xmin, xmax))

plt.show()
"""
combi = []
for nbef in range(3, 5):
    for naft in range(3, 11):
        combi.append((nbef, naft))

combi = [(3, 8), (4, 10), (3, 6)]
for (nbef, naft) in combi:
    print(nbef, naft)
    seltype = 'metric_{}_{}'.format(nbef, naft)
    dict_sel[seltype] = [('n_epochs_phase_minus_10', operator.ge, 1),
                         ('n_epochs_phase_plus_20', operator.ge, 1),
                         ('n_epochs_bef', operator.ge, nbef),
                         ('n_epochs_aft', operator.ge, naft),
                         # ('sigmaC', operator.le, 0.04),
                         ]

fig, ax = plt.subplots(figsize=(10, 8))
for key, vals in dict_sel.items():
    sel = select(res, vals)
 #   selfast = select(res_fast, vals)
 #   print(key, len(sel), len(selfast))
    print(key, len(sel))
    plot_effi(res, sel, leg=key, fig=fig, ax=ax)
#    plot_effi(res, selfast, leg=key, fig=fig, ax=ax)

ax.legend()

# plt.show()

"""
plotSN_2D(sel, vary='sigmaC', legy='$\sigma_C$')
plotSN_2D(sel, vary='sigmat0', legy='$\sigma_{T_0}$')
plotSN_2D_binned(sel, bins=np.arange(0.1, 0.7, 0.01),
                 vary='sigmaC', legy='$\sigma_C$')
plotSN_2D(sel, varx='n_epochs_aft',
          legx='N$_{epochs}^{aft}$', vary='sigmaC', legy='$\sigma_C$')

"""


"""
plotSN_2D(sel, varx='n_epochs_bef',
          legx='N$_{epochs}^{bef}$', vary='sigmaC', legy='$\sigma_C$')
"""
plt.show()
"""
plotSN_effi(sel)
plt.show()
"""
