import glob
from sn_tools.sn_io import loopStack
import numpy as np
from optparse import OptionParser
import pandas as pd
import os
import multiprocessing


class TransformSL:
    """
    class to transform SL metric files (hdf5) to light numpy array

    Parameters
    ---------------
    metricName: str
      metric name
    dirFile: str 
      location dir of the files to process
    fieldtype: str
      type of field (WFD or DD)
    nside: int 
        healpix nside parameter
    outDir: str, opt
      outputdirectory (default: MetricOutput) 
    nproc: int,opt
     number of procs to use (default: 8)

    """

    def __init__(self, metricName, dirFile, fieldtype, nside, outDir='MetricOutput', nproc=8):

        self.metricName = metricName
        self.dirFile = dirFile
        self.fieldtype = fieldtype
        self.nside = nside
        self.outDir = outDir
        self.nproc = nproc

        print('parameters', self.nproc)

    def __call__(self, toprocess):
        """
        Method to process data using multiprocessing

        Parameters
        --------------
        toprocess: list(str)
           list of db to process

        """
        nz = len(toprocess)
        t = np.linspace(0, nz, self.nproc+1, dtype='int')
        result_queue = multiprocessing.Queue()

        procs = [multiprocessing.Process(name='Subprocess-'+str(j), target=self.processLoop,
                                         args=(toprocess[t[j]:t[j+1]]['dbName'].tolist(),))
                 for j in range(self.nproc)]

        for p in procs:
            p.start()

    def processLoop(self, dbNames):
        """
        Method to process data

        Parameters
        --------------
        dbNames: list(str)
           list of db to process

        """
        for dbName in dbNames:
            search_path = '{}/{}/{}/{}*{}Metric*.hdf5'.format(self.dirFile,dbName,
                                                           self.metricName, dbName, self.metricName)
            print(search_path)
            fis = glob.glob(search_path)

            metricValues = loopStack(fis, objtype='astropyTable')

            tosave = metricValues['healpixId', 'pixRA',
                                  'pixDec', 'season_length', 'gap_median', 'area']

            outName = '{}/{}_{}.npy'.format(self.outDir,
                                            self.metricName, dbName)

            np.save(outName, np.copy(tosave))


parser = OptionParser(description='Transform SL hdf5 to light numpy array')
parser.add_option("--dbList", type="str",
                  default='WFD_fbs16_test.csv', help="db list [%default]")
parser.add_option("--dirFile", type="str", default='',
                  help="file directory [%default]")
parser.add_option("--fieldtype", type="str", default='WFD',
                  help="file directory [%default]")
parser.add_option("--nside", type="int", default=64,
                  help="file directory [%default]")
parser.add_option("--outDir", type="str", default='MetricOutput',
                  help="output directory [%default]")
parser.add_option("--metric", type="str", default='SL',
                  help="metric name [%default]")
parser.add_option("--nproc", type="int", default=8,
                  help="number of procs [%default]")

opts, args = parser.parse_args()

# create Output dir if does not exist

if not os.path.isdir(opts.outDir):
    os.makedirs(opts.outDir)



toprocess = pd.read_csv(opts.dbList)
trans = TransformSL(opts.metric, opts.dirFile,
                    opts.fieldtype, 64, opts.outDir,opts.nproc)

trans(toprocess)

"""

search_path = '{}/{}/{}*{}Metric*.hdf5'.format(thedir,
                                               metricName, dbName, metricName)
print(search_path)
fis = glob.glob(search_path)

print(fis, len(fis))

metricValues = loopStack(fis, objtype='astropyTable')

tosave = metricValues['healpixId', 'pixRA',
                      'pixDec', 'season_length', 'gap_median', 'area']

np.save('toto.npy', np.copy(tosave))
"""
