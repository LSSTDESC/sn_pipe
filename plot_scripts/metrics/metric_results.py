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

opts, args = parser.parse_args()


dirFile = opts.dbDir
dbName = opts.dbName
metricName = opts.metric
fieldType = opts.fieldType
nside = opts.nside

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
#np.savetxt('sample.csv', rt, delimiter=",")
rt.to_csv('sample.csv', index=False)
#print(getResults(dirFile, dbName, metricName, fieldType, nside))


"""
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
zlim_mean = np.mean(df['zlim_faint'])
nsn_tot = np.sum(df['nsn_zlim_faint'])
print('resultat', zlim_mean, nsn_tot)
"""
"""
pix = np.unique(df['healpixID'])
print(pix, len(pix))

idx = df['zlim_faint'] >= 0.25
idx &= df['zlim_faint'] <= 0.26

idx = np.abs(df['healpixID']-1449.) < 1.
print(np.unique(df[idx]['healpixID']), len(df[idx]))
print(df[idx][['pixRA', 'pixDec', 'season', 'zlim_faint', 'nsn_zlim_faint']])
fig, ax = plt.subplots()

#ax.hist(df['zlim_faint'], histtype='step')
ax.plot(df['zlim_faint'], df['healpixID'], 'ko')

fig, ax = plt.subplots()

#ax.hist(df['zlim_faint'], histtype='step')
ax.plot(df['nsn_zlim_faint'], df['healpixID'], 'ko')
plt.show()
"""
