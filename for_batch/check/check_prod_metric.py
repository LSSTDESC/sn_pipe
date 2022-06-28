import glob
import pandas as pd
from optparse import OptionParser
import numpy as np

parser = OptionParser()

parser.add_option("--llist", type=str, default='WFD_fbs_17.csv',
                  help="list of csv files containing list of db names [%default]")
parser.add_option("--theDir", type=str, default='/sps/lsst/users/gris/MetricOutput_fbs17_circular_dust',
                  help="location dir of dbs [%default]")

opts, args = parser.parse_args()

llist = opts.llist.split(',')

for ll in llist:
    df = pd.read_csv(ll)
    list_processed = glob.glob('{}/*'.format(opts.theDir))
    df_processed = pd.DataFrame(list_processed, columns=['dbName'])
    df_processed = df_processed.sort_values(by=['dbName'])
    df_processed['dbName'] = df_processed.dbName.str.split('/').str[-1]
    #df_processed['dbName'] = df['dbName'].apply(lambda x : x[:])
    #df['processed?'] = np.where(df['dbName'] == df_processed['dbName'], 'True', 'False')
    
    df = df.merge(df_processed, left_on='dbName',right_on='dbName',how='outer')
    df = df.dropna()
    idx = np.isin(df['dbName'],df_processed['dbName'].to_list())
    print('rrr',len(df),len(df[idx]))
    to_reprocess = df[~idx]
    if len(to_reprocess) > 0:
        vv = ['nside','coadd','nproc','simuType']
        to_reprocess[vv] = to_reprocess[vv].astype(int)
        to_reprocess.to_csv('torepro_{}'.format(ll),index=False)
    
