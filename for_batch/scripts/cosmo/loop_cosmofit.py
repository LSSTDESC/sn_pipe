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
                  default='input/cosmology/scenarios/survey_scenario.csv',
                  help=" survey to use[%default]")
parser.add_option("--tag", type=str,
                  default='-1.0',
                  help="tag for job name [%default]")
parser.add_option("--low_z_optimize", type=int,
                  default=0,
                  help="to optimize low-z sample  [%default]")
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
low_z_optimize = opts.low_z_optimize
timescale = opts.timescale
seasons_cosmo = opts.seasons_cosmo
nrandom = opts.nrandom
# load OS files to process
fis = pd.read_csv(dbList, comment='#')

# loop on files and create batches

seasons_cosmo = []
for i in range(2, 12):
    dd = list(range(1, i))
    seasons_cosmo.append(dd)

seasons_cosmo = list(range(1, 11))


script = 'run_scripts/cosmology/cosmology.py'
for i, row in fis.iterrows():
    dbName_DD = row['dbName_DD']
    dbName_WFD = row['dbName_WFD']
    wfd_tagsurvey = 'notag'
    dd_tagsurvey = 'notag'
    if 'wfd_tagsurvey' in fis.columns:
        wfd_tagsurvey = row['wfd_tagsurvey']
    if 'dd_tagsurvey' in fis.columns:
        dd_tagsurvey = row['dd_tagsurvey']

    processName = 'cosmo_{}_{}_{}'.format(
        dbName_DD, wfd_tagsurvey, dd_tagsurvey)
    mybatch = BatchIt(processName=processName)
    params = {}
    params['dataDir_DD'] = inputDir_DD
    params['dbName_DD'] = dbName_DD
    params['dataDir_WFD'] = inputDir_WFD
    params['dbName_WFD'] = dbName_WFD
    params['outDir'] = outDir
    params['survey'] = survey
    params['low_z_optimize'] = low_z_optimize
    params['timescale'] = timescale
    params['nrandom'] = nrandom
    params['seasons_cosmo'] = ','.join(list(map(str, seasons_cosmo)))
    params['wfd_tagsurvey'] = wfd_tagsurvey
    params['dd_tagsurvey'] = dd_tagsurvey

    mybatch.add_batch(script, params)
    # mybatch.go_batch()
