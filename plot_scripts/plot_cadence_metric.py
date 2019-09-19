import numpy as np
import sn_plotters.sn_cadencePlotters as sn_plot
from sn_tools.sn_io import loopStack
import matplotlib.pylab as plt
import argparse
from optparse import OptionParser
import glob

parser = OptionParser(description='Display Cadence metric results')
parser.add_option("--dbName", type="str", default='kraken_2026', help="db name [%default]")
parser.add_option("--dirFile", type="str", default='', help="file directory [%default]")
parser.add_option("--band", type="str", default='r', help="band [%default]")
parser.add_option("--x1", type="float", default=-2.0, help="SN x1 [%default]")
parser.add_option("--color", type="float", default=0.2, help="SN color [%default]")
parser.add_option("--fieldtype", type="str", default='WFD', help="file directory [%default]")

opts, args = parser.parse_args()

# Load parameters
dirFile = opts.dirFile
if dirFile == '':
    dirFile = '/sps/lsst/users/gris/MetricOutput'
dbName = opts.dbName
band = opts.band
x1 = opts.x1
color = opts.color
fieldtype = opts.fieldtype

#
refDir = 'reference_files'
namesRef = ['SNCosmo']
Li_files = []
mag_to_flux_files = []

if fieldtype == 'WFD':
    SNR = dict(zip('griz', [30., 40., 30., 20.]))  # SNR for WFD
    dt_range = [0.5, 30.]  # WFD dt range
    mag_range = [23., 26.5]  # WFD mag range

if fieldtype == 'DD':
    SNR = dict(zip('griz', [25., 25., 30., 35.]))  # SNR for DD
    dt_range = [0.5, 25.]  # DD dt range
    mag_range = [23., 27.5]  # DD mag range


for name in namesRef:
    Li_files = ['{}/Li_{}_{}_{}.npy'.format(refDir,name,x1,color)]
    mag_to_flux_files = ['{}/Mag_to_Flux_{}.npy'.format(refDir,name)]


fileNames = glob.glob('{}/{}/*CadenceMetric*{}*'.format(dirFile,dbName,fieldtype))
#fileName='{}/{}_CadenceMetric_{}.npy'.format(dirFile,dbName,band)
print(fileNames)
#metricValues = np.load(fileName)
metricValues = loopStack(fileNames).to_records(index=False)

idx = metricValues['filter'] == band

metricValues = np.copy(metricValues[idx])

sn_plot.plotMollview(64,metricValues,'cadence_mean','cadence','days',1.,band,dbName,saveFig=True)
sn_plot.plotMollview(64,metricValues,'m5_mean','m5','mag',24.,band,dbName,saveFig=True)
sn_plot.plotCadence(band,Li_files,mag_to_flux_files,
                    SNR[band],
                    metricValues,
                    namesRef,
                    mag_range=mag_range, dt_range=dt_range,
                    dbName=dbName,
                    saveFig=True)
plt.show()
