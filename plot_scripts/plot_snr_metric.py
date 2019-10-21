import numpy as np
import matplotlib.pylab as plt
import sn_plotters.sn_snrPlotters as sn_plot
from optparse import OptionParser
from sn_tools.sn_io import loopStack
import glob

parser = OptionParser(description='Display SNR metric results')
parser.add_option("--dbName", type="str",
                  default='kraken_2026', help="db name [%default]")
parser.add_option("--dirFile", type="str", default='',
                  help="file directory [%default]")
parser.add_option("--band", type="str", default='r', help="band [%default]")
parser.add_option("--fieldtype", type="str", default='WFD', help="band [%default]")
parser.add_option("--metricName", type="str",
                  default='SNRMetric', help="metric name[%default]")
parser.add_option("--nside", type="int", default=64, help="nside from healpix [%default]")

opts, args = parser.parse_args()

band = opts.band
dirFile = opts.dirFile
if dirFile == '':
    dirFile = '/sps/lsst/users/gris/MetricOutput'
dbName = opts.dbName
metricName = opts.metricName
fieldtype = opts.fieldtype
nside = opts.nside

namesRef = ['SNCosmo']

#fileName = '{}/{}_{}_{}.npy'.format(dirFile, dbName, metricName, band)
#metricValues = np.load(fileName)

fileNames = glob.glob('{}/{}/*{}_{}_nside_{}*'.format(dirFile,dbName,metricName,fieldtype,nside))
#fileName='{}/{}_CadenceMetric_{}.npy'.format(dirFile,dbName,band)
print(fileNames)

metricValues = loopStack(fileNames,'astropyTable')

print(metricValues.dtype)
sn_plot.detecFracPlot(metricValues, nside, namesRef)
#sn_plot.detecFracHist(metricValues, namesRef)

plt.show()
