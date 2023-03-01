import numpy as np
import pandas as pd
from optparse import OptionParser
import glob

def check_dir(dirFile,dbName,subDir):
    """
    Function to get a summary of the production

    Parameters
    ----------------
    dirFile: str
      location dir of the files to analyze.
    dbName: str
      db name to process
    subDir: str
     subdirectory where files are located

    Returns
    -----------
    pandas df with 3 cols: ProductionID, list of seasons, max(seasons)

    """

    fullPath = '{}/{}/{}'.format(dirFile,dbName,subDir)
    fi_split = subDir.split('_')[-1]
    fi_split = '_{}'.format(fi_split)

    # get hdf5 files

    fis = glob.glob('{}/*.hdf5'.format(fullPath))

    print(fis)

    data = pd.DataFrame(fis, columns=['fullPath'])
    print('dd',data)
    bb = data['fullPath'].str.split('/').str.get(-1)
    data['ProductionID'] = bb.str.split(fi_split).str.get(0)+fi_split
    data['season'] = bb.str.split('.hdf5').str.get(0).str.split('_').str.get(-1)

    data['season'] = data['season'].astype(int)
    print(data)
    data = data.sort_values(by=['season'])
    tt = data.groupby('ProductionID')['season'].apply(list).reset_index()
    #tt['season'] = data.groupby('ProductionID')['season'].transform(lambda x: list(x))
    tt['season_max'] = tt['season'].apply(max)

    return tt

parser = OptionParser()

parser.add_option("--dirFile", type="str", default='/sps/lsst/users/gris/Output_SN', help="location dir of the files [%default]")
parser.add_option("--csvList", type="str", default='WFD_fbs_2.99.csv', help="list of DBs [%default]")
parser.add_option("--subDir", type="str", default='WFD_photz', help="sub dir for files [%default]")

opts, args = parser.parse_args()

# read csv list

dbList = pd.read_csv(opts.csvList,comment='#')

print(dbList)
torep = pd.DataFrame()
for i, row in dbList.iterrows():
    dbName = row['dbName']
    res = check_dir(opts.dirFile,dbName,opts.subDir)
    torep = pd.concat((torep,res))

outName = '{}_{}.csv'.format(opts.csvList.split('.csv')[0],opts.subDir.split('_')[-1])

torep.to_csv(outName,index=False)
