from optparse import OptionParser
import glob
from astropy.table import Table
import sn_plotter_metrics.cadencePlot as sn_plot
import pandas as pd
from sn_tools.sn_io import loopStack
import numpy as np
import multiprocessing


class ProcessCadence:
    def __init__(self, dirFile, fieldtype, nside, band, Li_files, mag_to_flux_files, SNR, namesRef, mag_range, dt_range, var, web_path):

        self.dirFile = dirFile
        self.fieldtype = fieldtype
        self.nside = nside
        self.band = band
        self.Li_files = Li_files
        self.mag_to_flux_files = mag_to_flux_files
        self.SNR = SNR
        self.namesRef = namesRef
        self.mag_range = mag_range
        self.dt_range = dt_range
        self.var = var
        self.web_path = web_path

    def __call__(self, toprocess):

        nproc = 8
        nz = len(toprocess)
        t = np.linspace(0, nz, nproc+1, dtype='int')
        result_queue = multiprocessing.Queue()

        procs = [multiprocessing.Process(name='Subprocess-'+str(j), target=self.processLoop,
                                         args=(toprocess[t[j]:t[j+1]]['dbName'], dirFile, fieldtype, nside, band, Li_files, mag_to_flux_files, SNR[band], namesRef, mag_range, dt_range, 'zlim_SNCosmo', j, result_queue))
                 for j in range(nproc)]

        for p in procs:
            p.start()

        resultdict = {}
        # get the results in a dict

        for i in range(nproc):
            resultdict.update(result_queue.get())

        for p in multiprocessing.active_children():
            p.join()

        df = pd.DataFrame()

        # gather the results
        for key, vals in resultdict.items():
            df = pd.concat((df, vals), ignore_index=True)

        print('done here')

        print(df)
        metricName = 'Cadence'
        df.to_csv('Metric_{}.cvs'.format(metricName), index=False)

    def processLoop(self, dbNames, j=0, output_q=None):

        res = pd.DataFrame()

        for dbName in dbNames:
            resu = self.process(dbName)
            res = pd.concat((res, resu))

        if output_q is not None:
            return output_q.put({j: res})
        else:
            return res

    def load(self):

        search_file = '{}/{}/Cadence/*CadenceMetric_{}_nside_{}*.hdf5'.format(
            self.dirFile, self.dbName, self.fieldtype, self.nside)
        print('searching for', search_file)
        fileNames = glob.glob(search_file)

        metricValues = loopStack(fileNames, 'astropyTable')
        idx = metricValues['filter'] == band

        return metricValues[idx]

    def process(self, dbName):

        metricValues = load(self.dirFile, self.dbName,
                            self.fieldtype, self.nside, self.band)
        res = sn_plot.plotCadence(self.band, self.Li_files, self.mag_to_flux_files,
                                  self.SNR[self.band],
                                  metricValues,
                                  self.namesRef,
                                  mag_range=self.mag_range, dt_range=self.dt_range,
                                  dbName=dbName,
                                  saveFig=False, m5_str='m5_median', web_path=self.web_path)

        return pd.DataFrame({var: [np.median(res[self.var])]})


def processLoop(dbNames, dirFile, fieldtype, nside, band, Li_files, mag_to_flux_files, SNR, names_Ref, mag_range, dt_range, var, j=0, output_q=None):

    res = pd.DataFrame()

    for dbName in dbNames:
        resu = process(dbName, dirFile, fieldtype, nside, band, Li_files,
                       mag_to_flux_files, SNR, names_Ref, mag_range, dt_range, var)
        res = pd.concat((res, resu))

    if output_q is not None:
        return output_q.put({j: res})
    else:
        return res


def process(dbName, dirFile, fieldtype, nside, band, Li_files, mag_to_flux_files, SNR, names_Ref, mag_range, dt_range, var):

    metricValues = load(dirFile, dbName, fieldtype, nside, band)
    res = sn_plot.plotCadence(band, Li_files, mag_to_flux_files,
                              SNR,
                              metricValues,
                              namesRef,
                              mag_range=mag_range, dt_range=dt_range,
                              dbName=dbName,
                              saveFig=False, m5_str='m5_median', web_path=opts.web_path)

    return pd.DataFrame({var: [np.median(res[var])],
                         'dbName': [dbName]})


def processdf(grp, dirFile, fieldtype, nside, band, Li_files, mag_to_flux_files, SNR, names_Ref, mag_range, dt_range, var):

    dbName = grp.name
    metricValues = load(dirFile, dbName, fieldtype, nside, band)
    res = sn_plot.plotCadence(band, Li_files, mag_to_flux_files,
                              SNR,
                              metricValues,
                              namesRef,
                              mag_range=mag_range, dt_range=dt_range,
                              dbName=dbName,
                              saveFig=False, m5_str='m5_median', web_path=opts.web_path)

    return pd.DataFrame({var: [np.median(res[var])]})


def load(dirFile, dbName, fieldtype, nside, band):
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
dbList = opts.dbList
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
print('processing', len(toprocess))
#df = toprocess.groupby(['dbName']).apply(lambda x: process(x,dirFile, fieldtype, nside,band,Li_files, mag_to_flux_files,SNR[band],namesRef,mag_range,dt_range,'zlim_SNCosmo')).reset_index()

var = 'zlim_SNCosmo'
processCadence = ProcessCadence(dirFile, fieldtype, nside, band, Li_files,
                                mag_to_flux_files, SNR, namesRef, mag_range, dt_range, var, opts.web_path)

processCadence(toprocess)
