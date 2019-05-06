import numpy as np
import matplotlib.pylab as plt
import sn_plotters.sn_snrPlotters as sn_plot

bands = 'r'
dirFile = '/sps/lsst/users/gris/MetricOutput'
dbNames = ['kraken_2026']

namesRef = ['SNCosmo']

for dbName in dbNames:
    for band in bands:
        fileName='{}/{}_SNRMetric_{}.npy'.format(dirFile,dbName,band) 
        metricValues = np.load(fileName)
        sn_plot.detecFracPlot(metricValues,64, namesRef)
        sn_plot.detecFracHist(metricValues,namesRef)

plt.show()
