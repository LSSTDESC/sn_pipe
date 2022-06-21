import pandas as pd
import numpy as np
from sn_plotter_metrics.utils import Infos, ProcessFile
from sn_tools.sn_io import loopStack
import matplotlib.pyplot as plt


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


dirFile = '../MetricOutput_pixels'
dbName = 'baseline_v2.0_10yrs'
metricName = 'NSN'
npixels = -1
fieldType = 'WFD'

info = pd.DataFrame()

info['simuType'] = ['fbs']
info['simuNum'] = ['2.0']
info['dirFile'] = [dirFile]
info['dbName'] = [dbName]
info['family'] = ['baseline']
info['color'] = ['b']
info['marker'] = ['o']

nside = 16

df = pd.DataFrame()
for io, row in info.iterrows():
    df = Data(row, metricName, fieldType, nside, npixels).data_summary

print(df)
idx = df['zlim_faint'] > 0

#df = df[idx]
print(df.dtype)
print('resultat', np.mean(df['zlim_faint']), np.sum(df['nsn_zlim_faint']))
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
