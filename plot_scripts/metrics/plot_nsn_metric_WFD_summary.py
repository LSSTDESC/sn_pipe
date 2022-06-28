import numpy as np
from optparse import OptionParser
import glob
import healpy as hp
import pandas as pd
import os

from sn_plotter_metrics.utils import Infos, Simu, ProcessData, ProcessFile
from sn_tools.sn_io import loopStack
import sn_plotter_metrics.nsnPlot as nsn_plot


class ProcessFileNSN(ProcessFile):

    def __init__(self, info, metricName, fieldType, nside, npixels):
        """
        class to analyze results from NSN metric

        Parameters
        ---------------
        dirFile: str
          file directory
        dbName: str
          db name to process
        metricName: str
          metric name
        fieldType: str
          type of field to process
        nside: int
           healpix nside parameter


        """
        super().__init__(info, metricName, fieldType, nside, npixels)

    def process(self, fileNames):
        """
        Method to process metric values from files

        Parameters
        ---------------
        fileNames: list(str)
          list of files to process

        Returns
        ----------
        resdf: pandas df with a summary of metric infos

        """
        metricValues = np.array(loopStack(fileNames, 'astropyTable'))
        pixel_area = hp.nside2pixarea(self.nside, degrees=True)

        print('hello', metricValues.dtype)

        #idx = metricValues['status'] == 1
        #idx &= metricValues['zcomp'] > 0.

        zzval = 'zlim_faint'
        zzval = 'zcomp'
        idx = metricValues[zzval] > 0.

        data = pd.DataFrame(metricValues[idx])
        data = data.applymap(
            lambda x: x.decode() if isinstance(x, bytes) else x)

        print(len(np.unique(data[['healpixID', 'season']])))
        self.ratiopixels = 1
        self.npixels_eff = len(data['healpixID'].unique())
        if self.npixels > 0:
            self.ratiopixels = float(
                npixels)/float(self.npixels_eff)

        nsn_dict = self.nSN_tot(data)
        nsn_extrapol = {}
        for key, nsn in nsn_dict.items():
            nsn_extrapol[key] = int(np.round(nsn*self.ratiopixels))

        meds = data.groupby(['healpixID']).median().reset_index()
        meds = meds.round({zzval: 5})
        med_meds = meds.median()
        resdf = pd.DataFrame(
            [self.info['dbName']], columns=['dbName'])

        resdf[zzval] = med_meds[zzval]

        for key, vals in nsn_dict.items():
            resdf[key] = [vals]
            #resdf['sig_nsn'] = [sig_nsn]
            resdf['{}_extrapol'.format(key)] = [nsn_extrapol[key]]
        #resdf['dbName'] = self.dbInfo['dbName']
        resdf['simuType'] = self.info['simuType']
        resdf['simuNum'] = self.info['simuNum']
        resdf['family'] = self.info['family']
        resdf['color'] = self.info['color']
        resdf['marker'] = self.info['marker']
        resdf['cadence'] = [med_meds['cadence']]
        #resdf['season_length'] = [med_meds['season_length']]
        resdf['gap_max'] = [med_meds['gap_max']]
        resdf['survey_area'] = self.npixels_eff*pixel_area
        for key, vals in nsn_dict.items():
            resdf['{}_per_sqdeg'.format(key)] = resdf[key]/resdf['survey_area']

        means = data.groupby(['healpixID']).mean().reset_index()
        #stds  = data.groupby(['healpixID']).std().reset_index()

        print(means.columns)
        # for vv in ['cad_sn_mean', 'gap_sn_mean']:
        for vv in ['cadence_sn', 'gap_max_sn']:
            resdf[vv] = means[vv]
        # for vv in ['cad_sn_std','gap_sn_std']:
         #   resdf[vv] = stds[vv]

        print(resdf)
        return resdf

    def nSN_tot(self, data):
        """
        Method to estimate the total number of supernovae(and error)

        Returns
        -----------
        nsn, sig_nsn: int, int
          number of sn and sigma
        """
        sums = data.groupby(['healpixID']).sum().reset_index()

        dictout = {}
        dictout['nsn'] = sums['nsn'].sum()
        #dictout['nsn'] = sums['nsn_zlim_faint'].sum()
        """
        for vv in self.ztypes:
            dictout['nsn_{}'.format(vv)] = sums['nsn_{}_{}'.format(vv,self.sntype)].sum()
        """
        return dictout


def mscatter(x, y, ax=None, m=None, **kw):
    import matplotlib.markers as mmarkers
    ax = ax or plt.gca()
    sc = ax.scatter(x, y, **kw)
    if (m is not None) and (len(m) == len(x)):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(
                marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    return sc


def print_best(resdf, ref_var='nsn', num=10, name='a'):
    """
    Method to print the "best" OS maximizing ref_var

    Parameters
    --------------
    resdf: pandas df
      data to process
    ref_var: str, opt
      variable chosen to rank the strategies (default: nsn)
    num: int, opt
      number of OS to display)

    """

    ressort = pd.DataFrame(resdf)
    ressort = ressort.sort_values(by=[ref_var], ascending=False)
    ressort['rank'] = ressort[ref_var].rank(
        ascending=False, method='first').astype('int')
    print(ressort[['dbName', ref_var, 'rank']][:num])
    ressort['dbName'] = ressort['dbName'].str.split('v1.4_10yrs').str[0]
    ressort['dbName'] = ressort['dbName'].str.rstrip('_')
    ressort[['dbName', ref_var, 'rank']][:].to_csv(
        'OS_best_{}.csv'.format(name), index=False)


def rankCadences(resdf, ref_var='nsn'):
    """
  Method to print the "best" OS maximizing ref_var

  Parameters
  --------------
  resdf: pandas df
    data to process
  ref_var: str, opt
    variable chosen to rank the strategies (default: nsn)

    Returns
    -----------
    original pandas df plus rank
    """

    ressort = pd.DataFrame(resdf)
    ressort = ressort.sort_values(by=[ref_var], ascending=False)
    ressort['rank'] = ressort[ref_var].rank(
        ascending=False, method='first').astype('int')

    return ressort


def plotSummary(resdf, text=True, ref=False, ref_var='nsn'):
    """
    Method to draw the summary plot nSN vs zlim

    Parameters
    ---------------
    resdf: pandas df
      dat to plot
    text: bool, opt
      to write the dbNames or not
    ref: bool, opt
      if true, results are displayed from a reference cadence (default: False)
    ref_var: str, opt
      column from which the reference OS is chosen (default: nsn_ref

    """

    fig, ax = plt.subplots(figsize=(14, 8))

    zlim_ref = -1
    nsn_ref = -1

    if ref:
        ido = np.argmax(resdf[ref_var])
        zlim_ref = resdf.loc[ido, 'zlim']
        nsn_ref = resdf.loc[ido, 'nsn']

    print(zlim_ref, nsn_ref)
    """
    if zlim_ref > 0:
        mscatter(zlim_ref-resdf['zlim'], resdf['nsn']/nsn_ref, ax=ax,
                 m=resdf['marker'].to_list(), c=resdf['color'].to_list())
    else:
        mscatter(resdf['zlim'], resdf['nsn'], ax=ax,
                 m=resdf['marker'].to_list(), c=resdf['color'].to_list())
    """
    for ii, row in resdf.iterrows():
        if zlim_ref > 0:
            ax.text(zlim_ref-row['zlim'], row['nsn']/nsn_ref, row['dbName'])
        else:
            ax.plot(row['zlim'], row['nsn'], marker=row['marker'],
                    color=row['color'], ms=10)
            if text:
                ax.text(row['zlim']+0.001, row['nsn'], row['dbName'], size=12)

    ax.grid()
    ax.set_xlabel('$z_{faint}$')
    ax.set_ylabel('$N_{SN}(z\leq z_{faint})$')
    patches = []
    for col in np.unique(resdf['color']):
        idx = resdf['color'] == col
        tab = resdf[idx]
        lab = '{}_{}'.format(
            np.unique(tab['simuType']).item(), np.unique(tab['simuNum']).item())
        patches.append(mpatches.Patch(color=col, label=lab))
    plt.legend(handles=patches)


def plotCorrel(resdf, x=('', ''), y=('', '')):
    """
    Method for 2D plots

    Parameters
    ---------------
    resdf: pandas df
      data to plot
    x: tuple
      x-axis variable (first value: colname in resdf; second value: x-axis label)
    y: tuple
      y-axis variable (first value: colname in resdf; second value: y-axis label)
    """

    fig, ax = plt.subplots()

    resdf = filter(resdf, ['alt_sched'])
    for ik, row in resdf.iterrows():
        varx = row[x[0]]
        vary = row[y[0]]

        ax.plot(varx, vary, marker=row['marker'], color=row['color'])
        # ax.text(varx+0.1, vary, row['dbName'], size=10)

    ax.set_xlabel(x[1])
    ax.set_ylabel(y[1])
    ax.grid()


def plotBarh(resdf, varname, leg):
    """
    Method to plot varname - barh

    Parameters
    ---------------
    resdf: pandas df
      data to plot
    varname: str
       column to plot

    """

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.subplots_adjust(left=0.3)

    resdf = resdf.sort_values(by=[varname])
    resdf['dbName'] = resdf['dbName'].str.split('_10yrs', expand=True)[0]
    ax.barh(resdf['dbName'], resdf[varname], color=resdf['color'])
    ax.set_xlabel(r'{}'.format(leg))
    ax.tick_params(axis='y', labelsize=15.)
    plt.grid(axis='x')
    # plt.tight_layout
    plt.savefig('Plots_pixels/Summary_{}.png'.format(varname))


def filter(resdf, strfilt=['_noddf']):
    """
    Function to remove OS according to their names

    Parameters
    ---------------
    resdf: pandas df
      data to process
    strfilt: list(str),opt
     list of strings used to remove OS (default: ['_noddf']

    """

    for vv in strfilt:
        idx = resdf['dbName'].str.contains(vv)
        resdf = pd.DataFrame(resdf[~idx])

    return resdf


parser = OptionParser(
    description='Display NSN metric results for WFD fields')

parser.add_option("--configFile", type=str, default='plot_scripts/input/config_NSN_WFD.csv',
                  help="config file [%default]")
# parser.add_option("--nside", type=int, default=64,
#                  help="nside for healpixels [%default]")
parser.add_option("--tagbest", type=str, default='snpipe_a',
                  help="tag for the best OS [%default]")
parser.add_option("--nproc", type=int, default=3,
                  help="number of proc when multiprocessing used [%default]")
parser.add_option("--colors", type=str, default='k,r,b,m,g,grey,darkgreen',
                  help="colors for the plot [%default]")
parser.add_option("--metric", type=str, default='NSNY',
                  help="metric name [%default]")

opts, args = parser.parse_args()

# Load parameters
#nside = opts.nside
nproc = opts.nproc

metricName = opts.metric

list_to_process = pd.read_csv(opts.configFile, comment='#')

simu_list = []

for i, row in list_to_process.iterrows():
    simu_list.append(Simu(row['simuType'], row['simuNum'],
                          row['dirFile'], row['dbList'], row['nside']))

# get the data to be plotted
resdf = pd.DataFrame()
colors = opts.colors.split(',')
for ip, vv in enumerate(simu_list):
    outFile = 'Summary_{}_WFD_{}_{}_{}.npy'.format(
        metricName, vv.type, vv.num, vv.nside)

    if not os.path.isfile(outFile):
        toprocess = Infos(vv, ip).resdf
        #processMulti(toprocess, outFile, nside, metricName, 'WFD', nproc=nproc)
        proc = ProcessData(vv.nside, metricName, 'WFD')
        proc.processMulti(toprocess, outFile,
                          process_class=ProcessFileNSN, nproc=nproc)

    tabdf = pd.DataFrame(np.load(outFile, allow_pickle=True))
    print('hello color', ip, colors)
    tabdf['color'] = colors[ip]
    tabdf['nside'] = vv.nside
    resdf = pd.concat((resdf, tabdf))

    """
    rfam = []
    for io, row in resdf.iterrows():
        rfam.append(family(row['dbName'], resdf['dbName'].to_list()))
    resdf['family'] = rfam
    """
# print(test)

metricTot = None
metricTot_med = None

# filter cadences here
"""
resdf = filter(
    resdf, ['_noddf', 'footprint_stuck_rolling', 'weather', 'wfd_depth'])
"""

# summary plot
#nsn_plot.NSN_zlim_GUI(resdf,xvar='zpeak',yvar='nsn_zpeak',xlabel='$z_{peak}$',ylabel='$N_{SN}(z\leq z_{peak})$',title='(nSN,zpeak) supernovae metric')
#nsn_plot.NSN_zlim_GUI(resdf,xvar='zlim',yvar='nsn_zlim',xlabel='$z_{lim}$',ylabel='$N_{SN}(z\leq z_{lim})$',title='(nSN,zlim) supernovae metric')
nsn_plot.NSN_zlim_GUI(resdf, xvar='zcomp', yvar='nsn',
                      xlabel='$z_{complete}$', ylabel='$N_{SN}(z\leq z_{complete})$', title='(nSN,$z_{complete}$) supernovae metric')
#nsn_plot.NSN_zlim_GUI(resdf,xvar='cad_sn_mean',yvar='gap_sn_mean',xlabel='SN cadence [day]',ylabel='SN gap [day]',title='(cadence , gap) SN')
# nsn_plot.PlotSummary_Annot(resdf)
# plt.show()
# plotSummary(resdf)
# plt.show()
# print(test)

# 2D plots

"""
plotCorrel(resdf, x=('cadence', 'cadence'), y=('nsn', '#number of supernovae'))
plotBarh(resdf, 'cadence')
"""
plotBarh(resdf, 'cadence', 'cadence')
plotBarh(resdf, 'survey_area', 'survey area')
plotBarh(resdf, 'nsn_per_sqdeg', 'NSN per sqdeg')
plotBarh(resdf, 'season_length', 'season length')
"""
bandstat = ['u','g','r','i','z','y','gr','gi',
    'gz','iz','uu','gg','rr','ii','zz','yy']
for b in bandstat:
    plotBarh(resdf, 'cadence_{}'.format(b),
             'Effective cadence - {} band'.format(b))
    print('hello',resdf)
"""
# plotBarh(resdf, 'N_{}_tot'.format(b))
"""
for bb in 'grizy':
plotBarh(resdf, 'N_{}{}'.format(b,bb))
"""
# plotCorrel(resdf, x=('cadence_{}'.format(b), 'cadence_{}'.format(b)), y=(
#    'nsn', '#number of supernovae'))

# plotBarh(resdf, 'N_total')
# print_best(resdf, num=20, name=tagbest)
plt.show()
