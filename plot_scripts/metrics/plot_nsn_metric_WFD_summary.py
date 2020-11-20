import numpy as np
import sn_plotter_metrics.nsnPlot as nsn_plot
import matplotlib.patches as mpatches
import matplotlib.pylab as plt
import argparse
from optparse import OptionParser
import glob
# from sn_tools.sn_obs import dataInside
import healpy as hp
import numpy.lib.recfunctions as rf
import pandas as pd
import os
import multiprocessing
from dataclasses import dataclass


@dataclass
class Simu:
    type: str
    num: str
    dir: str
    list: str


class Infos:
    """
    class to build a dataframe
    with requested infos to make plots

    Parameters
    ---------------
    simu: dataclass of type Simu

    """

    def __init__(self, simu):

        self.simu = simu
        self.families = []
        self.colors = ['b', 'k', 'r', 'g', 'm', 'c']
        self.markers = [".", "o", "v", "^", "<", ">", "1", "2", "3", "4", "8",
                        "s", "p", "P", "*", "h", "H", "+", "x", "X", "D", "d", "|", "_"]

        dbList = pd.read_csv(simu.list, comment='#')
        print(dbList)

        # self.rlist = self.cleandbName(dbList)
        self.rlist = dbList['dbName'].to_list()
        self.resdf = self.dfInfos()

    def cleandbName(self, dbList):
        """
        Method to clean dbNames by removing all char after version number (included): v_..

        Parameters
        ---------------
        dbList: pandas df
          containing the list of dbNames

        Returns
        ----------
        list of 'cleaned' dbNames

        """
        r = []
        # get 'cleaned' dbName
        for i, val in dbList.iterrows():
            spl = val['dbName'].split('v{}'.format(vv.num))[0]
            if spl[-1] == '_':
                spl = spl[:-1]
            r.append(spl)

        return r

    def clean(self, fam):
        """
        Method to clean a string

        Parameters
        ----------------
        fam: str
          the string to clean

        Returns
        -----------
        the final string

        """
        if fam[-1] == '_':
            return fam[:-1]

        if fam[-1] == '.':
            return fam[:-2]

        return fam

    def family(self, dbName):
        """
        Method to get a family from a dbName

        Parameters
        --------------
        dbName: str
           the dbName to process

        Returns
        ----------
        str: the name of the 'family'


        """

        ro = []
        fam = dbName
        for i in range(len(dbName)):
            stre = dbName[:i+1]
            num = 0
            for kk in self.rlist:
                if stre == kk[:i+1]:
                    num += 1
            # print(stre, num)
            ro.append(num)
            if i > 5 and ro[-1]-ro[-2] < 0:
                fam = dbName[:i]
                break

        return self.clean(fam)

    def dfInfos(self):
        """
        Method to build a pandas df
        with requested infos for the plotter
        """

        resdf = pd.DataFrame()
        # get families and fill infos
        for va in self.rlist:
            resdf = pd.concat((resdf, self.getInfos(va)))

        return resdf

    def getInfos(self, dbName):
        """
        Method to build a df with infos for plotter
        for a single dbName

        Parameters
        ---------------
        dbName: str
          dbName to process

        Returns
        -----------
        pandas df with infos as cols.


        """
        fam = self.family(dbName)
        # print(vv, fam)
        if fam not in self.families:
            self.families.append(fam)
        imark = self.families.index(fam)
        print(vv.type, vv.dir, dbName, fam,
              self.colors[ip], self.markers[self.families.index(fam)])

        return pd.DataFrame({'simuType': [self.simu.type],
                             'simuNum': [self.simu.num],
                             'dirFile': [self.simu.dir],
                             'dbName': [dbName],
                             'family': [fam],
                             'color': [self.colors[ip]],
                             'marker': [self.markers[self.families.index(fam)]]})


def processMulti(toproc, outFile, nproc=1):
    """
    Function to analyze metric output using multiprocesses
    The results are stored in outFile (npy file)

    Parameters
    --------------
    toproc: pandas df
      data to process
    outFile: str
       output file name
    nproc: int, opt
      number of cores to use for the processing

    """

    nfi = len(toproc)
    tabfi = np.linspace(0, nfi, nproc+1, dtype='int')

    print(tabfi)
    result_queue = multiprocessing.Queue()

    # launching the processes
    for j in range(len(tabfi)-1):
        ida = tabfi[j]
        idb = tabfi[j+1]

        p = multiprocessing.Process(name='Subprocess-'+str(j), target=processLoop, args=(
            toproc[ida:idb], 'WFD', j, result_queue))
        p.start()

    # grabing the results
    resultdict = {}

    for j in range(len(tabfi)-1):
        resultdict.update(result_queue.get())

    for p in multiprocessing.active_children():
        p.join()

    resdf = pd.DataFrame()
    for j in range(len(tabfi)-1):
        resdf = pd.concat((resdf, resultdict[j]))

    print('finally', resdf.columns)
    # saving the results in a npy file
    np.save(outFile, resdf.to_records(index=False))


def processLoop(toproc, fieldType='WFD', j=0, output_q=None):
    """
    Function to analyze a set of metric result files

    Parameters
    --------------
    toproc: pandas df
      data to process
    j: int, opt
       internal int for the multiprocessing
    output_q: multiprocessing.queue
      queue for multiprocessing

    Returns
    -----------
    pandas df with the following cols:
    zlim, nsn, sig_nsn, nsn_extra, dbName, plotName, color,marker
    """
    # this is to get summary values here
    resdf = pd.DataFrame()
    for index, val in toproc.iterrows():
        metricdata = nsn_plot.NSNAnalysis(val, metricName, fieldType,
                                          nside, npixels=0)

        # metricdata.plot()
        # plt.show()
        if metricdata.data_summary is not None:
            resdf = pd.concat((resdf, metricdata.data_summary))

    if output_q is not None:
        output_q.put({j: resdf})
    else:
        return resdf


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

parser.add_option("--configFile", type="str", default='plot_scripts/input/config_NSN_WFD.csv',
                  help="config file [%default]")
parser.add_option("--nside", type="int", default=64,
                  help="nside for healpixels [%default]")
parser.add_option("--tagbest", type="str", default='snpipe_a',
                  help="tag for the best OS [%default]")

opts, args = parser.parse_args()

# Load parameters
nside = opts.nside
metricName = 'NSN'

list_to_process = pd.read_csv(opts.configFile, comment='#')

simu_list = []

for i, row in list_to_process.iterrows():
    simu_list.append(Simu(row['simuType'], row['simuNum'],
                          row['dirFile'], row['dbList']))

# get the data to be plotted
resdf = pd.DataFrame()
for ip, vv in enumerate(simu_list):
    outFile = 'Summary_WFD_{}_{}.npy'.format(vv.type, vv.num)

    if not os.path.isfile(outFile):
        toprocess = Infos(vv).resdf
        processMulti(toprocess, outFile, nproc=3)

    resdf = pd.concat((resdf, pd.DataFrame(
        np.load(outFile, allow_pickle=True))))

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
nsn_plot.NSN_zlim_GUI(resdf)
# nsn_plot.PlotSummary_Annot(resdf)
# plt.show()
# plotSummary(resdf)
# plt.show()
print(test)

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
#print_best(resdf, num=20, name=tagbest)
plt.show()
