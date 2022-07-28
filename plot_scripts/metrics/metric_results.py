import pandas as pd
import numpy as np
from sn_plotter_metrics.utils import Infos, ProcessFile
from sn_tools.sn_io import loopStack
import matplotlib.pyplot as plt
from optparse import OptionParser
import glob


class Data(ProcessFile):

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

        return metricValues


def getResults(dirFile, dbName, metricName, fieldType, nside, npixels=-1):

    info = pd.DataFrame()

    info['simuType'] = ['fbs']
    info['simuNum'] = ['2.0']
    info['dirFile'] = [dirFile]
    info['dbName'] = [dbName]
    info['family'] = ['baseline']
    info['color'] = ['b']
    info['marker'] = ['o']

    df = pd.DataFrame()
    for io, row in info.iterrows():
        df = Data(row, metricName, fieldType, nside, npixels).data_summary

    idx = df['zlim_faint'] > 0

    df = df[idx]
    zlim_mean = np.round(np.mean(df['zlim_faint']), 4)
    nsn_tot = int(np.sum(df['nsn_zlim_faint']))

    return (dbName, nside, zlim_mean, nsn_tot)


def processMultiple(dirFile):
    dbList = glob.glob('{}/*'.format(dirFile))

    dbList = list(map(lambda elem: elem.split('/')[-1], dbList))

    res = []
    io = -1
    for nside in [64, 16]:
        for dbName in dbList:
            rr = getResults(dirFile, dbName, metricName, fieldType, nside)
            res.append(rr)
            io += 1
        # if io > 0:
        #    break

    rt = pd.DataFrame(res, columns=['dbName', 'nside', 'zcomp', 'nsn'])
    print(rt)
    # np.savetxt('sample.csv', rt, delimiter=",")
    rt.to_csv('sample.csv', index=False)
    # print(getResults(dirFile, dbName, metricName, fieldType, nside))


def plotBinned(df, xcol='cadence', ycol='nsn', bins_x=np.arange(0.5, 12, 1.)):

    fig, ax = plt.subplots()
    # bins = np.arange(5., 120, 1.)
    ddf = pd.DataFrame(df.copy())
    group = ddf.groupby(pd.cut(ddf[xcol], bins_x))
    # group = ddf.groupby(pd.cut(ddf.gap_max_sn, bins))
    plot_centers = (bins_x[:-1] + bins_x[1:])/2
    plot_values = group[ycol].mean()
    ax.plot(df[xcol], df[ycol], 'ko', mfc='None')
    ax.plot(plot_centers, plot_values, color='r')


def plotIt(dirFile, dbName, metricName, fieldType, nside, zlimstr, nsnstr, npixels=-1):

    info = pd.DataFrame()

    info['simuType'] = ['fbs']
    info['simuNum'] = ['2.0']
    info['dirFile'] = [dirFile]
    info['dbName'] = [dbName]
    info['family'] = ['baseline']
    info['color'] = ['b']
    info['marker'] = ['o']

    df = pd.DataFrame()
    for io, row in info.iterrows():
        df = Data(row, metricName, fieldType, nside, npixels).data_summary

    print(df.dtype.names)
    idx = df[zlimstr] > 0

    df = df[idx]
    zlim_mean = np.mean(df[zlimstr])
    nsn_tot = np.sum(df[nsnstr])
    nseasons = len(np.unique(df['season']))
    print('resultat', zlim_mean, nsn_tot, nseasons,
          np.median(df[nsnstr]), len(df))

    idx = df[zlimstr] >= zlim_mean
    idx &= df[zlimstr] <= 1.05*zlim_mean
    idx = df['healpixID'] == 2218

    pix = np.unique(df[idx]['healpixID'])
    print(pix, len(pix))
    sel = pd.DataFrame(df[idx])
    print(sel.columns)
    # cols = ['healpixID', 'season', 'gap_max', 'cadence',
    #        'season_length', zlimstr, nsnstr]
    cols = ['healpixID', 'season', 'cadence', zlimstr, nsnstr]
    print(sel[cols])

    idx = df[zlimstr] >= 0.40
    idx &= df[zlimstr] <= 0.42

    #idx = np.abs(df['healpixID']-1449.) < 1.
    print(np.unique(df[idx]['healpixID']), len(df[idx]))
    print(df[idx][['healpixID', 'pixRA', 'pixDec', 'season', zlimstr, nsnstr]])

    np.save('dfb.npy', np.copy(df))
    fig, ax = plt.subplots()

    # ax.hist(df[zlimstr], histtype='step')
    ax.plot(df[zlimstr], df['healpixID'], 'ko')

    fig, ax = plt.subplots()

    # ax.hist(df[zlimstr], histtype='step')
    ax.plot(df[nsnstr], df['healpixID'], 'ko')

    fig, ax = plt.subplots(ncols=3)
    ax[0].hist(df[zlimstr], histtype='step', bins=20)
    ax[1].hist(df[nsnstr], histtype='step', bins=20)
    ax[2].hist(df['timeproc'], histtype='step', bins=20)

    fig, ax = plt.subplots()

    ax.plot(df['pixRA'], df['pixDec'], 'ko')

    """
    for b in 'grizy':
        print('filter alloc ', b, np.median(df['frac_{}'.format(b)]))
    """

    plotBinned(df)
    plotBinned(df, xcol='cadence', ycol='zcomp')
    plotBinned(df, xcol='cadence', ycol='timeproc')
    print(df.dtype.names)
    plt.show()


parser = OptionParser()

parser.add_option("--dbName", type="str", default='baseline_v2.0_10yrs',
                  help="db name [%default]")
parser.add_option("--dbExtens", type="str", default='npy',
                  help="db extension [%default]")
parser.add_option("--dbDir", type="str",
                  default='../MetricOutput_pixels', help="db dir [%default]")
parser.add_option("--metric", type="str", default='NSN',
                  help="metric to process [%default]")
parser.add_option("--fieldType", type="str", default='WFD',
                  help="field type DD or WFD[%default]")
parser.add_option("--nside", type="int", default=64,
                  help="healpix nside [%default]")
parser.add_option("--zlimstr", type="str", default='zcomp',
                  help="zlim column in df [%default]")
parser.add_option("--nsnstr", type="str", default='nsn',
                  help="nsn column in df [%default]")

opts, args = parser.parse_args()


dirFile = opts.dbDir
dbName = opts.dbName
metricName = opts.metric
fieldType = opts.fieldType
nside = opts.nside
zlimstr = opts.zlimstr
nsnstr = opts.nsnstr

# processMultiple(dirFile)
plotIt(dirFile, dbName, metricName, fieldType,
       nside, zlimstr, nsnstr, npixels=-1)
