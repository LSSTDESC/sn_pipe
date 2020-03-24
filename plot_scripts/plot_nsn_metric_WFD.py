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


class MetricAna:
    def __init__(self, dbDir, dbName,
                 metricName='NSN', fieldType='WFD',
                 nside=64,
                 x1=-2.0, color=0.2, npixels=-1):
        """
        class to analyze results from NSN metric

        Parameters
        ---------------
        dbDir: str
          location directory where the files to process are
        dbName: str
          observing strategy name
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

        # loading data (metric values)
        search_path = '{}/{}/{}/*NSNMetric_{}*_nside_{}_*.hdf5'.format(
            dirFile, dbName, metricName, fieldType, nside)
        print('looking for', search_path)
        fileNames = glob.glob(search_path)
        # fileName='{}/{}_CadenceMetric_{}.npy'.format(dirFile,dbName,band)
        # print(fileNames)
        metricValues = np.array(loopStack(fileNames, 'astropyTable'))
        idx = np.abs(metricValues['x1']-x1) < 1.e-6
        idx &= np.abs(metricValues['color']-color) < 1.e-6
        idx &= metricValues['status'] == 1
        # idx &= metricValues['season'] == 5
        self.data = pd.DataFrame(metricValues[idx])

        self.ratiopixels = 1
        self.npixels_eff = len(self.data['healpixID'].unique())
        if npixels > 0:
            self.ratiopixels = float(
                npixels)/float(self.npixels_eff)

        self.zlim = self.zlim_med()
        self.nsn, self.sig_nsn = self.nSN_tot()
        self.nsn_extrapol = int(np.round(self.nsn*self.ratiopixels))

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
                          xmin=0.1, xmax=0.4)

        # this is to plot the total number of SN (per pixels) over the sky
        sums = self.data.groupby(['healpixID']).sum().reset_index()
        sums['nsn_med'] = sums['nsn_med'].astype(int)
        self.plotMollview(sums, 'nsn_med', 'NSN', np.sum,
                          xmin=10., xmax=25.)

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
    description='Display Cadence metric results for DD fields')
parser.add_option("--dirFile", type="str", default='',
                  help="file directory [%default]")
parser.add_option("--nside", type="int", default=64,
                  help="nside for healpixels [%default]")
parser.add_option("--fieldType", type="str", default='WFD',
                  help="field type - DD, WFD, Fake [%default]")
parser.add_option("--nPixelsFile", type="str", default='ObsPixels_fbs14_nside_64.npy',
                  help="file with the total number of pixels per obs. strat.[%default]")


opts, args = parser.parse_args()

# Load parameters
dirFile = opts.dirFile
if dirFile == '':
    dirFile = '/sps/lsst/users/gris/MetricOutput'

nside = opts.nside
fieldType = opts.fieldType
metricName = 'NSN'
nPixelsFile = opts.nPixelsFile

dbNames = ['descddf_v1.4_10yrs']

metricTot = None
metricTot_med = None
pixArea = hp.nside2pixarea(nside, degrees=True)
x1 = -2.0
color = 0.2
Npixels = np.load(nPixelsFile)
print(Npixels.dtype)

toproc = pd.read_csv('toread.csv')
for index, val in toproc.iterrows():
    dbName = val['dbName']
    dirFile = val['dirFile']
    idx = Npixels['dbName'] == dbName
    myana = MetricAna(dirFile, dbName, metricName, fieldType,
                      nside, npixels=Npixels[idx]['npixels'].item())
    print(myana.zlim, myana.nsn, myana.sig_nsn,
          myana.npixels_eff, myana.nsn_extrapol)
    # myana.plot()

plt.show()
