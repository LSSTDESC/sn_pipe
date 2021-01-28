import glob
import pandas as pd
from optparse import OptionParser


parser = OptionParser()

parser.add_option("--llist", type=str, default='WFD_fbs_161.csv',
                  help="list of csv files containing list of db names [%default]")
parser.add_option("--theDir", type=str, default='/sps/lsst/users/gris/ObsPixelized_circular_new',
                  help="location dir of dbs [%default]")

opts, args = parser.parse_args()

llist = opts.llist.split(',')

for ll in llist:
    df = pd.read_csv(ll)
    #print(df)
    for i,row in df.iterrows():
        fi = glob.glob('{}/{}/*.npy'.format(opts.theDir,row['dbName']))
        #print(row['dbName'],len(fi))
        if len(fi) != 10:
            print('problem with',row['dbName'],len(fi))
    
