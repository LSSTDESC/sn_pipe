import numpy as np
import matplotlib.pylab as plt
import sn_plotters.sn_snrPlotters as sn_plot
from optparse import OptionParser

parser = OptionParser(description='Display SNR metric results')
parser.add_option("--dbName", type="str", default='kraken_2026', help="db name [%default]")
parser.add_option("--dirFile", type="str", default='', help="file directory [%default]")
parser.add_option("--band", type="str", default='r', help="band [%default]")

opts, args = parser.parse_args()

band = opts.band
dirFile = opts.dirFile
if dirFile == '':
    dirFile = '/sps/lsst/users/gris/MetricOutput'
dbName = opts.dbName

namesRef = ['SNCosmo']

fileName='{}/{}_SNRMetric_{}.npy'.format(dirFile,dbName,band) 
metricValues = np.load(fileName)
sn_plot.detecFracPlot(metricValues,64, namesRef)
sn_plot.detecFracHist(metricValues,namesRef)

plt.show()
