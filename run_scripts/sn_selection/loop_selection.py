import pandas as pd
import os

from optparse import OptionParser

parser = OptionParser()

parser.add_option("--dataDir", type=str,
                  default='../Output_SN_sigmaInt_0.0_Hounsell',
                  help="data dir[%default]")
parser.add_option("--dbList", type=str,
                  default='list_OS.csv', help="DB list to process [%default]")
parser.add_option("--timescale", type=str,
                  default='year',
                  help="Time scale for NSN estimation. [%default]")
parser.add_option("--runType", type=str,
                  default='DDF_spectroz', help=" [%default]")
parser.add_option("--listFields", type=str,
                  default='COSMOS,CDFS,XMM-LSS,ELAISS1,EDFSa,EDFSb',
                  help=" [%default]")
parser.add_option("--fieldType", type=str,
                  default='DDF',
                  help=" [%default]")
parser.add_option("--nsn_factor", type=int,
                  default=30, help="MC normalisation factor [%default]")
parser.add_option("--nproc", type=int,
                  default=8,
                  help="Number of procs for multiprocessing [%default]")

opts, args = parser.parse_args()

dbList = opts.dbList
dataDir = opts.dataDir
timescale = opts.timescale
runType = opts.runType
listFields = opts.listFields
fieldType = opts.fieldType
nsn_factor = opts.nsn_factor
nproc = opts.nproc


db = pd.read_csv(dbList, comment='#')

script = 'python run_scripts/sn_selection/sn_selection.py'

for i, row in db.iterrows():
    cmd = '{} --dbName {} --dataDir {}'.format(script, row['dbName'], dataDir)
    cmd += ' --timescale={}'.format(timescale)
    cmd += ' --runType={}'.format(runType)
    cmd += ' --fieldType={}'.format(fieldType)
    cmd += ' --listFields={}'.format(listFields)
    cmd += ' --nsn_factor={}'.format(nsn_factor)
    cmd += ' --nproc={}'.format(nproc)
    print(cmd)
    os.system(cmd)
