import numpy as np
import sn_plotters.sn_cadencePlotters as sn_plot
import matplotlib.pylab as plt

bands = 'r'
dirFile = '/sps/lsst/users/gris/MetricOutput'
dbNames = ['kraken_2026']

refDir = 'reference_files'
x1 = -2.0
color = 0.2
namesRef = ['SNCosmo']
Li_files = []
mag_to_flux_files = []

#SNR = dict(zip('griz', [25., 25., 30., 35.]))  # SNR for DD
#dt_range = [0.5, 25.]  # DD dt range
#mag_range = [23., 27.5]  # DD mag range

SNR = dict(zip('griz', [30., 40., 30., 20.]))  # SNR for WFD
dt_range = [0.5, 30.]  # WFD dt range
mag_range = [21., 25.5]  # WFD mag range

for name in namesRef:
    Li_files = ['{}/Li_{}_{}_{}.npy'.format(refDir,name,x1,color)]
    mag_to_flux_files = ['{}/Mag_to_Flux_{}.npy'.format(refDir,name)]


for dbName in dbNames:
    for band in bands:
        fileName='{}/{}_CadenceMetric_{}.npy'.format(dirFile,dbName,band)
        print(fileName)
        metricValues = np.load(fileName)
        sn_plot.plotMollview(64,metricValues,'cadence_mean','cadence','days',1.,band)
        sn_plot.plotMollview(64,metricValues,'m5_mean','m5','mag',24.,band)
        sn_plot.plotCadence(band,Li_files,mag_to_flux_files,
                            SNR[band],
                            metricValues,
                            namesRef,
                            mag_range=mag_range, dt_range=dt_range)
plt.show()
