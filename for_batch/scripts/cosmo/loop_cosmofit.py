import pandas as pd
from optparse import OptionParser
from sn_tools.sn_batchutils import BatchIt
parser = OptionParser()

parser.add_option("--dbList", type="str", default='DD_fakes.csv',
                  help="db list to process  [%default]")
parser.add_option("--outDir", type="str",
                  default='/sps/lsst/users/gris/cosmo_fit', help="output directory [%default]")
parser.add_option("--inputDir_DD", type="str",
                  default='/sps/lsst/users/gris/Output_SN_sigmaInt_0.0_Hounsell_G10_JLA', help="input directory for DD files[%default]")
parser.add_option("--inputDir_WFD", type="str",
                  default='/sps/lsst/users/gris/Output_SN_WFD_sigmaInt_0.0_Hounsell_G10_JLA', help="input directory for WFD files[%default]")
parser.add_option("--dbName_WFD", type="str",
                  default='draft_connected_v2.99_10yrs', help="dbName for WFD data[%default]")
opts, args = parser.parse_args()

dbList = opts.dbList
outDir = opts.outDir
inputDir_DD = opts.inputDir_DD
inputDir_WFD = opts.inputDir_WFD
dbName_WFD = opts.dbName_WFD

# load OS files to process
fis = pd.read_csv(dbList, comment='#')

# loop on files and create batches

script = 'run_scripts/cosmology/cosmology.py'
for i, row in fis.iterrows():
    dbName = row['dbName']
    processName = 'cosmo_{}'.format(dbName)
    mybatch = BatchIt(processName=processName)
    params = {}
    params['dataDir_DD'] = inputDir_DD
    params['dbName_DD'] = dbName
    params['dataDir_WFD'] = inputDir_WFD
    params['dbName_WFD'] = dbName_WFD
    params['outDir'] = outDir
    mybatch.add_batch(script, params)
    mybatch.go_batch()
