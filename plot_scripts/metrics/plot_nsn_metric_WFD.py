import numpy as np
import sn_plotters.sn_cadencePlotters as sn_plot
import sn_plotters.sn_NSNPlotters as nsn_plot
from sn_tools.sn_io import loopStack
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
        metricdata = MetricAna(dirFile, val, metricName, fieldType,
                               nside, npixels=npixels)
        # metricdata.plot()
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


class MetricAna:
    def __init__(self, dbDir, dbInfo,
                 metricName='NSN', fieldType='WFD',
                 nside=64,
                 x1=-2.0, color=0.2, npixels=-1):
        """
        class to analyze results from NSN metric

        Parameters
        ---------------
        dbDir: str
          location directory where the files to process are
        dbInfo: pandas df
          info from observing strategy (dbName, plotName, color, marker)
        metricName: str
          name of the metric used to generate the files
        fieldType: str, opt
          field type (DD, WFD) (default: WFD)
        nside: int,opt
          healpix nside parameter (default: 64)
        x1: float, opt
          x1 SN value (default: -2.0)
        color: float, opt
          color SN value (default: 0.2)
        npixels: int, opt
          total number of pixels for this strategy

        """

        self.nside = nside
        self.npixels = npixels
        self.x1 = x1
        self.color = color
        self.dbInfo = dbInfo

        # loading data (metric values)
        search_path = '{}/{}/{}/*NSNMetric_{}*_nside_{}_*.hdf5'.format(
            dirFile, dbInfo['dbName'], metricName, fieldType, nside)
        print('looking for', search_path)
        fileNames = glob.glob(search_path)
        # fileName='{}/{}_CadenceMetric_{}.npy'.format(dirFile,dbName,band)
        # print(fileNames)
        if len(fileNames) > 0:
            self.data_summary = self.process(fileNames)
        else:
            print('Missing files for', dbInfo['dbName'])
            self.data_summary = None

    def process(self, fileNames):

        metricValues = np.array(loopStack(fileNames, 'astropyTable'))
        idx = np.abs(metricValues['x1']-self.x1) < 1.e-6
        idx &= np.abs(metricValues['color']-self.color) < 1.e-6
        idx &= metricValues['status'] == 1
        idx &= metricValues['zlim'] > 0.
        idx &= metricValues['nsn_med'] > 0.

        self.data = pd.DataFrame(metricValues[idx])

        print('data', self.data[['healpixID', 'pixRA', 'pixDec', 'x1', 'color', 'zlim',
                                 'nsn_med', 'nsn', 'season']], self.data.columns)
        print(len(np.unique(self.data[['healpixID', 'season']])))
        self.ratiopixels = 1
        self.npixels_eff = len(self.data['healpixID'].unique())
        if self.npixels > 0:
            self.ratiopixels = float(
                npixels)/float(self.npixels_eff)

        zlim = self.zlim_med()
        nsn, sig_nsn = self.nSN_tot()
        nsn_extrapol = int(np.round(nsn*self.ratiopixels))

        resdf = pd.DataFrame([zlim], columns=['zlim'])
        resdf['nsn'] = [nsn]
        resdf['sig_nsn'] = [sig_nsn]
        resdf['nsn_extra'] = [nsn_extrapol]
        resdf['dbName'] = self.dbInfo['dbName']
        resdf['plotName'] = self.dbInfo['plotName']
        resdf['color'] = self.dbInfo['color']
        resdf['marker'] = self.dbInfo['marker']
        return resdf

    def zlim_med(self):
        """
        Method to estimate the median redshift limit over the pixels

        Returns
        ----------
        median zlim (float)
        """
        meds = self.data.groupby(['healpixID']).median().reset_index()
        meds = meds.round({'zlim': 2})

        return meds['zlim'].median()

    def nSN_tot(self):
        """
        Method to estimate the total number of supernovae (and error)

        Returns
        -----------
        nsn, sig_nsn: int, int
          number of sn and sigma
        """
        sums = self.data.groupby(['healpixID']).sum().reset_index()
        sums['nsn_med'] = sums['nsn_med'].astype(int)

        return sums['nsn_med'].sum(), int(np.sqrt(sums['var_nsn_med'].sum()))

    def plot(self):
        """
        Method to plot two Mollview of the metric results:
        - redshift limit 
        - number of well-sampled supernovae

        """

        # this is to estimate the median zlim over the sky
        meds = self.data.groupby(['healpixID']).median().reset_index()
        meds = meds.round({'zlim': 2})
        self.plotMollview(meds, 'zlim', 'zlimit', np.median,
                          xmin=0.01, xmax=np.max(meds['zlim'])+0.1)

        # this is to plot the total number of SN (per pixels) over the sky
        sums = self.data.groupby(['healpixID']).sum().reset_index()
        sums['nsn_med'] = sums['nsn_med'].astype(int)
        self.plotMollview(sums, 'nsn_med', 'NSN', np.sum,
                          xmin=0., xmax=np.max(sums['nsn_med'])+1)

    def plotMollview(self, data, varName, leg, op, xmin, xmax):
        """
        Method to display results as a Mollweid map

        Parameters
        ---------------
        data: pandas df
          data to consider
        varName: str
          name of the variable to display
        leg: str
          legend of the plot
        op: operator
          operator to apply to the pixelize data (median, sum, ...)
        xmin: float
          min value for the display
        xmax: float
         max value for the display

        """
        npix = hp.nside2npix(self.nside)

        hpxmap = np.zeros(npix, dtype=np.float)
        hpxmap = np.full(hpxmap.shape, -0.1)
        hpxmap[data['healpixID'].astype(
            int)] += data[varName]
        norm = plt.cm.colors.Normalize(xmin, xmax)
        cmap = plt.cm.jet
        cmap.set_under('w')
        resleg = op(data[varName])
        title = '{}: {}'.format(leg, resleg)

        hp.mollview(hpxmap, min=xmin, max=xmax, cmap=cmap,
                    title=title, nest=True, norm=norm)
        hp.graticule()


parser = OptionParser(
    description='Display NSN metric results for WFD fields')

parser.add_option("--dirFile", type="str", default='',
                  help="file directory [%default]")
parser.add_option("--nside", type="int", default=64,
                  help="nside for healpixels [%default]")
parser.add_option("--fieldType", type="str", default='WFD',
                  help="field type - DD, WFD, Fake [%default]")
parser.add_option("--nPixelsFile", type="str", default='ObsPixels_fbs14_nside_64.npy',
                  help="file with the total number of pixels per obs. strat.[%default]")
parser.add_option("--listdb", type="str", default='plot_scripts/input/WFD_test.csv',
                  help="list of dbnames to process [%default]")

opts, args = parser.parse_args()

# Load parameters
dirFile = opts.dirFile
if dirFile == '':
    dirFile = '/sps/lsst/users/gris/MetricOutput'

nside = opts.nside
fieldType = opts.fieldType
metricName = 'NSN'
nPixelsFile = opts.nPixelsFile
listdb = opts.listdb

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

outFile = 'Summary_WFD.npy'

if not os.path.isfile(outFile):
    processMulti(toproc, Npixels, outFile, nproc=4)

resdf = pd.DataFrame(np.load(outFile, allow_pickle=True))
print(resdf.columns)

# Summary plot

# color=resdf['color'].tolist())

fig, ax = plt.subplots()

mscatter(resdf['zlim'], resdf['nsn'], ax=ax,
         m=resdf['marker'].to_list(), c=resdf['color'].to_list())

ax.grid()
ax.set_xlabel('$z_{faint}$')
ax.set_ylabel('$N_{SN}(z\leq z_{faint})$')

"""
plt.scatter(resdf['zlim'], resdf['nsn'], lineStyle='None',
            markers=resdf['marker'].to_list())
"""
plt.show()
