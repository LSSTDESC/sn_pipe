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
    mm = metric
    fieldName = grp['fieldName']
    if fieldName != 'WFD':
        mm = '{}_{}'.format(metric,fieldName)
    search_path = '{}/{}/{}/*.hdf5'.format(dirFiles, grp['dbName'], mm)
    print('searching',search_path)
    fi = glob.glob(search_path)
    print(grp['dbName'],len(fi))
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
parser.add_option("--fieldType", type="str", default='WFD',
                  help="field type - WFD or DD [%default]")
parser.add_option('--fieldNames', type=str, default='COSMOS,CDFS,ELAISS1,XMM-LSS,EDFSa,EDFSb',help='DD fields to process [%default]')

opts, args = parser.parse_args()

dirFiles = opts.dirFiles
dbList = opts.dbList
metric = opts.metric
outFile = opts.outFile
fieldNames = opts.fieldNames.split(',')
fieldType = opts.fieldType

# load the list
toprocess = pd.read_csv('{}'.format(dbList),comment='#')

fields = pd.DataFrame(fieldNames,columns=['fieldName'])

dbNames = toprocess['dbName'].unique()
toprocess_all = pd.DataFrame()


for dbName in dbNames:
    fields['dbName'] = dbName
    toprocess_all  = pd.concat((toprocess_all,fields))

# get the number of files for each OS
toprocess_all['nfiles'] = toprocess_all.apply(
    lambda x: count(x, dirFiles, metric), axis=1)
    
print(toprocess_all)

# check for missing files
min_val = 80
if fieldType == 'DD':
    min_val = 8

idx = toprocess_all['nfiles'] < min_val
dfmiss = toprocess_all[idx]
#dfmiss = dfmiss.loc[:, dfmiss.columns != 'nfiles']

#dfmiss.drop(['nfiles','fieldName'])
idb = toprocess['dbName'].isin(dfmiss['dbName'].unique())

toprocess[idb].to_csv(outFile, index=False)
