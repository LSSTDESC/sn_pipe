import pandas as pd
import os

from optparse import OptionParser

parser = OptionParser()

parser.add_option("--dataDir", type=str,
                  default='../Output_SN_sigmaInt_0.0_Hounsell',
                  help="data dir[%default]")
parser.add_option("--dbList", type=str,
                  default='list_OS.csv', help="DB list to process [%default]")

opts, args = parser.parse_args()

dbList = opts.dbList
dataDir = opts.dataDir


db = pd.read_csv(dbList, comment='#')

script = 'python run_scripts/cosmology/sn_selection.py'

for i, row in db.iterrows():
    cmd = '{} --dbName {} --dataDir {}'.format(script, row['dbName'], dataDir)
    os.system(cmd)
