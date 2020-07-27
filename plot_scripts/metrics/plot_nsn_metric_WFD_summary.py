import numpy as np
import sn_plotter_metrics.nsnPlot as nsn_plot
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


def processMulti(toproc, Npixels, outFile, nproc=1):
    """
    Function to analyze metric output using multiprocesses
    The results are stored in outFile (npy file)

    Parameters
    --------------
    toproc: pandas df
      data to process
    Npixels: numpy array
      array of the total number of pixels per OS
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
            toproc[ida:idb], Npixels, j, result_queue))
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


def processLoop(toproc, Npixels, j=0, output_q=None):
    """
    Function to analyze a set of metric result files

    Parameters
    --------------
    toproc: pandas df
      data to process
    Npixels: numpy array
      array of the total number of pixels per OS
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
        dbName = val['dbName']
        idx = Npixels['dbName'] == dbName
        npixels = Npixels[idx]['npixels'].item()
        metricdata = nsn_plot.NSNAnalysis(dirFile, val, metricName, fieldType,
                                          nside, npixels=npixels)

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


def plotSummary(resdf, ref=False, ref_var='nsn'):
    """
    Method to draw the summary plot nSN vs zlim

    Parameters
    ---------------
    resdf: pandas df
      dat to plot
    ref: bool, opt
      if true, results are displayed from a reference cadence (default: False)
    ref_var: str, opt
      column from which the reference OS is chosen (default: nsn_ref

    """

    fig, ax = plt.subplots()

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
            ax.text(row['zlim']+0.001, row['nsn'], row['dbName'], size=12)

    ax.grid()
    ax.set_xlabel('$z_{faint}$')
    ax.set_ylabel('$N_{SN}(z\leq z_{faint})$')


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
        #ax.text(varx+0.1, vary, row['dbName'], size=10)

    ax.set_xlabel(x[1])
    ax.set_ylabel(y[1])
    ax.grid()


def plotBarh(resdf, varname):
    """
    Method to plot varname - barh

    Parameters
    ---------------
    resdf: pandas df
      data to plot
    varname: str
       column to plot

    """

    fig, ax = plt.subplots()

    resdf = resdf.sort_values(by=[varname])
    resdf['dbName'] = resdf['dbName'].str.split('_10yrs', expand=True)[0]
    ax.barh(resdf['dbName'], resdf[varname], color=resdf['color'])
    ax.set_xlabel(r'{}'.format(varname))
    ax.tick_params(axis='y', labelsize=10.)
    plt.grid(axis='x')
    plt.tight_layout


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

parser.add_option("--dirFile", type="str", default='/sps/lsst/users/gris/MetricOutput',
                  help="file directory [%default]")
parser.add_option("--nside", type="int", default=64,
                  help="nside for healpixels [%default]")
parser.add_option("--fieldType", type="str", default='WFD',
                  help="field type - DD, WFD, Fake [%default]")
parser.add_option("--nPixelsFile", type="str", default='ObsPixels_fbs14_nside_64.npy',
                  help="file with the total number of pixels per obs. strat.[%default]")
parser.add_option("--listdb", type="str", default='plot_scripts/input/WFD_test.csv',
                  help="list of dbnames to process [%default]")
parser.add_option("--tagbest", type="str", default='snpipe_a',
                  help="tag for the best OS [%default]")

opts, args = parser.parse_args()

# Load parameters
dirFile = opts.dirFile
nside = opts.nside
fieldType = opts.fieldType
metricName = 'NSN'
nPixelsFile = opts.nPixelsFile
listdb = opts.listdb
tagbest = opts.tagbest

metricTot = None
metricTot_med = None

toproc = pd.read_csv(listdb)

pixArea = hp.nside2pixarea(nside, degrees=True)
x1 = -2.0
color = 0.2

if os.path.isfile(nPixelsFile):
    Npixels = np.load(nPixelsFile)
else:
    print('File with the total number of pixels not found')
    r = toproc.copy()
    r['npixels'] = 0.
    Npixels = r.to_records(index=False)

print(Npixels.dtype)

outFile = 'Summary_WFD_{}.npy'.format(tagbest)

if not os.path.isfile(outFile):
    processMulti(toproc, Npixels, outFile, nproc=4)

resdf = pd.DataFrame(np.load(outFile, allow_pickle=True))
print(resdf.columns)
resdf['dbName'] = resdf['dbName'].str.split('_10yrs').str[0]
# filter cadences here
resdf = filter(resdf, ['_noddf', 'footprint_stuck_rolling', 'weather'])

# rank cadences
resdf = rankCadences(resdf)

# select only the first 20


idx = resdf['rank'] <= 22

resdf = resdf[idx]

# Summary plot

#plotSummary(resdf, ref=False)


# 2D plots

"""
plotCorrel(resdf, x=('cadence', 'cadence'), y=('nsn', '#number of supernovae'))
plotBarh(resdf, 'cadence')
"""

for b in 'ugrizy':
    plotBarh(resdf, 'cadence_{}'.format(b))
    # plotCorrel(resdf, x=('cadence_{}'.format(b), 'cadence_{}'.format(b)), y=(
    #    'nsn', '#number of supernovae'))

print_best(resdf, num=20, name=tagbest)
plt.show()
