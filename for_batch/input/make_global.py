import pandas as pd
from optparse import OptionParser


parser = OptionParser()

parser.add_option("--fileList", type="str", default='WFD_fbs00.csv',
                  help="list of files to transform [%default]")

parser.add_option("--dbDir", type="str", default='/sps/lsst/users/gris/Global',
                  help="directory where the db are [%default]")

parser.add_option("--dbExtens", type="str", default='db',
                  help="db extension [%default]")



opts, args = parser.parse_args()

df = pd.read_csv(opts.fileList, comment='#')

df['dbExtens'] = opts.dbExtens
df['dbDir'] = opts.dbDir

print(df.columns)

firstline='dbDir,dbName,dbExtens'

df[['dbDir','dbName','dbExtens']].to_csv('test.csv',header=firstline,index=False)
