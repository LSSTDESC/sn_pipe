from optparse import OptionParser
import glob
from astropy.table import Table
import sn_plotter_metrics.cadencePlot as sn_plot
import pandas as pd
from sn_tools.sn_io import loopStack
import numpy as np

def process(grp,dirFile, fieldtype, nside,band,Li_files, mag_to_flux_files,SNR,names_Ref,mag_range,dt_range,var):

    dbName = grp.name
    metricValues = load(dirFile, dbName, fieldtype, nside,band)
    res = sn_plot.plotCadence(band, Li_files, mag_to_flux_files,
                    SNR,
                    metricValues,
                    namesRef,
                    mag_range=mag_range, dt_range=dt_range,
                    dbName=dbName,
                    saveFig=False, m5_str='m5_median', web_path=opts.web_path)

    return pd.DataFrame({var: [np.median(res[var])]})

    
def load(dirFile, dbName, fieldtype, nside,band):
    search_file = '{}/{}/Cadence/*CadenceMetric_{}_nside_{}*.hdf5'.format(
        dirFile, dbName, fieldtype, nside)
    print('searching for', search_file)
    fileNames = glob.glob(search_file)

    metricValues = loopStack(fileNames, 'astropyTable')
    idx = metricValues['filter'] == band

    return metricValues[idx]
    
parser = OptionParser(description='Estimate Cadence metric summary results ')
parser.add_option("--dbList", type="str",
                  default='cadenceCustomize_fbs14.csv', help="db list [%default]")
parser.add_option("--dirFile", type="str", default='',
                  help="file directory [%default]")
parser.add_option("--band", type="str", default='r', help="band [%default]")
parser.add_option("--x1", type="float", default=-2.0, help="SN x1 [%default]")
parser.add_option("--color", type="float", default=0.2,
                  help="SN color [%default]")
parser.add_option("--fieldtype", type="str", default='WFD',
                  help="file directory [%default]")
parser.add_option("--nside", type="int", default=64,
                  help="file directory [%default]")
parser.add_option("--web_path", type=str, default='https://me.lsst.eu/gris/DESC_SN_pipeline',
                  help='url where to find some files necessary to run[%default]')

opts, args = parser.parse_args()

dirFile = opts.dirFile
dbList= opts.dbList
#metricName = opts.metricName
fieldtype = opts.fieldtype
nside = opts.nside
band = opts.band
x1 = opts.x1
color = opts.color
#var = opts.var
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
    Li_files = ['{}/Li_{}_{}_{}.npy'.format(refDir, name, x1, color)]
    mag_to_flux_files = ['{}/Mag_to_Flux_{}.npy'.format(refDir, name)]


toprocess = pd.read_csv(dbList, comment='#')
print('processing',toprocess[:2])
df = toprocess.groupby(['dbName']).apply(lambda x: process(x,dirFile, fieldtype, nside,band,Li_files, mag_to_flux_files,SNR[band],namesRef,mag_range,dt_range,'zlim_SNCosmo')).reset_index()

    
print(df)
metricName='Cadence'
df.to_csv('Metric_{}.cvs'.format(metricName),index=False)
