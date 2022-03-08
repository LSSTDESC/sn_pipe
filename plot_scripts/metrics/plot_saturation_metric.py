import matplotlib.pyplot as plt
from optparse import OptionParser
import pandas as pd
import os
import numpy as np

from sn_plotter_metrics.utils import Infos, Simu, ProcessData, ProcessFile
from sn_tools.sn_io import loopStack
import sn_plotter_metrics.nsnPlot as nsn_plot


class ProcessFileSat(ProcessFile):

    def __init__(self, info, metricName, fieldType, nside, npixels=-1):
        """
        class to analyze results from Saturation metric

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
        npixels: int
          total number of processed pixels


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
        resdf = loopStack(fileNames, 'astropyTable').to_pandas()

        resdf = resdf.groupby(
            ['healpixID', 'pixRA', 'pixDec']).median().reset_index()

        for vv in ['dbName', 'marker', 'simuType', 'simuNum', 'family']:
            resdf[vv] = self.info[vv]

        return resdf


parser = OptionParser(
    description='Display saturation metrics results for WFD fields')

parser.add_option("--configFile", type=str, default='plot_scripts/input/config_NSN_WFD.csv',
                  help="config file [%default]")
parser.add_option("--nside", type=int, default=64,
                  help="nside for healpixels [%default]")
parser.add_option("--nproc", type=int, default=8,
                  help="number of proc when multiprocessing used [%default]")
parser.add_option("--colors", type=str, default='k,r,b,m,g',
                  help="colors for the plot [%default]")
parser.add_option("--metric", type=str, default='Saturation',
                  help="metric name [%default]")

opts, args = parser.parse_args()

# Load parameters
nside = opts.nside
nproc = opts.nproc

metricName = opts.metric

list_to_process = pd.read_csv(opts.configFile, comment='#')

simu_list = []

for i, row in list_to_process.iterrows():
    simu_list.append(Simu(row['simuType'], row['simuNum'],
                          row['dirFile'], row['dbList']))

print(simu_list)

# get the data to be plotted
resdf = pd.DataFrame()
colors = opts.colors.split(',')

# getting the data here
for ip, vv in enumerate(simu_list):
    outFile = 'Summary_{}_WFD_{}_{}.npy'.format(metricName, vv.type, vv.num)

    if not os.path.isfile(outFile):
        toprocess = Infos(vv, ip).resdf
        proc = ProcessData(nside, metricName, 'WFD')
        proc.processMulti(toprocess, outFile,
                          process_class=ProcessFileSat, nproc=nproc)

    tabdf = pd.DataFrame(np.load(outFile, allow_pickle=True))
    tabdf['color'] = colors[ip]
    resdf = pd.concat((resdf, tabdf))

# make the plots here

idx = resdf['probasat'] > 0.
#idx = resdf['dbName'] == 'baseline_nexp1_v1.7_10yrs'
resdf = resdf[idx]
"""
for i, row in resdf.iterrows():
    print(row['deltaT_befsat'])
plt.hist(resdf['deltaT_befsat'])
plt.show()
"""
# print(test)
resplot = resdf.groupby(['dbName', 'family', 'color',
                        'marker', 'simuType', 'simuNum']).median().reset_index()

print(resplot.columns,
      resplot[['dbName', 'family', 'marker', 'fractwi', 'deltaT_befsat']])

# nsn_plot.NSN_zlim_GUI(resplot, xvar='deltaT_befsat', yvar='fractwi',
#                      xlabel='$\Delta T_{before sat}$ [day]', ylabel='Twi frac', title='Saturation metric')

nsn_plot.NSN_zlim_GUI(resplot, xvar='probasat', yvar='effipeak',
                      xlabel='Proba sat', ylabel='Effi peak', title='Saturation metric')
