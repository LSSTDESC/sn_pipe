import pandas as pd
from optparse import OptionParser
from sn_tools.sn_batchutils import BatchIt
parser = OptionParser()

parser.add_option("--dbList", type="str",
                  default='for_batch/input/cosmofit/cosmoList.csv',
                  help="db list to process  [%default]")
parser.add_option("--outDir", type="str",
                  default='/sps/lsst/users/gris/cosmo_fit',
                  help="output directory [%default]")
parser.add_option("--inputDir_DD", type="str",
                  default='/sps/lsst/users/gris/Output_SN_sigmaInt_0.0_Hounsell_G10_JLA',
                  help="input directory for DD files[%default]")
parser.add_option("--inputDir_WFD", type="str",
                  default='/sps/lsst/users/gris/Output_SN_WFD_sigmaInt_0.0_Hounsell_G10_JLA',
                  help="input directory for WFD files[%default]")
parser.add_option("--survey", type=str,
                  default='input/DESC_cohesive_strategy/survey_scenario.csv',
                  help=" survey to use[%default]")
parser.add_option("--tag", type=str,
                  default='-1.0',
                  help="tag for job name [%default]")
parser.add_option("--frac_WFD_low_sigmaC", type=float,
                  default=0.3,
                  help="fraction of SNe Ia WFD low sigmaC  [%default]")
parser.add_option('--timescale', type=str, default='year',
                  help='timescale for the cosmology (year or season)[%default]')
parser.add_option('--seasons_cosmo', type=str,
                  default='1-10',
                  help='Seasons to estimate cosmology params [%default]')
parser.add_option('--nrandom', type=int,
                  default=50,
                  help='number of random sample (per season/year) to generate [%default]')

opts, args = parser.parse_args()

dbList = opts.dbList
outDir = opts.outDir
inputDir_DD = opts.inputDir_DD
inputDir_WFD = opts.inputDir_WFD
# dbName_WFD = opts.dbName_WFD
survey = opts.survey
tag = opts.tag
frac_WFD_low_sigmaC = opts.frac_WFD_low_sigmaC
timescale = opts.timescale
seasons_cosmo = opts.seasons_cosmo
nrandom = opts.nrandom
# load OS files to process
fis = pd.read_csv(dbList, comment='#')

# loop on files and create batches

script = 'run_scripts/cosmology/cosmology.py'
for i, row in fis.iterrows():
    dbName_DD = row['dbName_DD']
    dbName_WFD = row['dbName_WFD']
    processName = 'cosmo_{}_{}'.format(dbName_DD, tag)
    mybatch = BatchIt(processName=processName)
    params = {}
    params['dataDir_DD'] = inputDir_DD
    params['dbName_DD'] = dbName_DD
    params['dataDir_WFD'] = inputDir_WFD
    params['dbName_WFD'] = dbName_WFD
    params['outDir'] = outDir
    params['survey'] = survey
    params['frac_WFD_low_sigmaC'] = frac_WFD_low_sigmaC
    params['timescale'] = timescale
    params['seasons_cosmo'] = seasons_cosmo
    params['nrandom'] = nrandom
    mybatch.add_batch(script, params)
    mybatch.go_batch()
