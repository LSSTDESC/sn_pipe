import os
import glob
import numpy as np
import pandas as pd
from optparse import OptionParser


def count(grp, dirFiles, metric):
    """
    Function to count the number of *hdf5 files in a directory

    Parameters
    ---------------
    grp: pandas df row
    dirFiles: str
      main location dir of the files
    metric: str
      name of the corresponding metric

    Returns
    -----------
    number of corresponding files

    """
    fi = glob.glob('{}/{}/{}/*.hdf5'.format(dirFiles, grp['dbName'], metric))
    return len(fi)


parser = OptionParser()

parser.add_option("--dirFiles", type="str", default='../Files/MetricOutput_fbs14',
                  help="main dir where the files are located [%default]")
parser.add_option("--dbList", type="str", default='for_batch/input/WFD_fbs14.csv',
                  help="list of the files to check[%default]")
parser.add_option("--metric", type="str", default='NSN',
                  help="name of the metric[%default]")
parser.add_option("--outFile", type="str", default='missing_fbs14.csv',
                  help="name of output file[%default]")

opts, args = parser.parse_args()

dirFiles = opts.dirFiles
dbList = opts.dbList
metric = opts.metric
outFile = opts.outFile

# load the list
toprocess = pd.read_csv('{}'.format(dbList))
# get the number of files for each OS
toprocess['nfiles'] = toprocess.apply(
    lambda x: count(x, dirFiles, metric), axis=1)

# check for missing files
idx = toprocess['nfiles'] < 80
dfmiss = toprocess[idx]
dfmiss = dfmiss.loc[:, dfmiss.columns != 'nfiles']
dfmiss.to_csv(outFile, index=False)
