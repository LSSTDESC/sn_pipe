import pandas as pd
from optparse import OptionParser
from sn_tools.sn_batchutils import BatchIt
parser = OptionParser()

parser.add_option("--dbName_DD", type="str",
                  default='baseline_v3.6_10yrs',
                  help="db Name DD to process [%default]")
parser.add_option("--dbName_WFD", type="str",
                  default='baseline_v3.6_10yrs',
                  help="db Name WFD to process [%default]")
parser.add_option("--outDir", type="str",
                  default='/sps/lsst/users/gris/cosmo_fit',
                  help="output directory [%default]")
parser.add_option("--inputDir_DD", type="str",
                  default='/sps/lsst/users/gris/Output_SN_sigmaInt_0.0_Hounsell_G10_JLA',
                  help="input directory for DD files[%default]")
parser.add_option("--inputDir_WFD", type="str",
                  default='/sps/lsst/users/gris/Output_SN_WFD_sigmaInt_0.0_Hounsell_G10_JLA',
                  help="input directory for WFD files[%default]")
parser.add_option("--surveyList", type=str,
                  default='surveylist.csv',
                  help=" survey list (+outName) to process [%default]")
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
parser.add_option('--fitparam_names', type=str,
                  default='w0,wa,Om0',
                  help='fit parameter names [%default]')
parser.add_option('--fitparam_values', type=str,
                  default='-1.0,0.0,0.3',
                  help='fit parameter values [%default]')
parser.add_option('--prior', type=int,
                  default=1,
                  help='prior for the fit [%default]')

opts, args = parser.parse_args()

dbName_DD = opts.dbName_DD
dbName_WFD = opts.dbName_WFD
outDir = opts.outDir
inputDir_DD = opts.inputDir_DD
inputDir_WFD = opts.inputDir_WFD
# dbName_WFD = opts.dbName_WFD
surveyList = opts.surveyList
# = opts.tag
low_z_optimize = opts.low_z_optimize
timescale = opts.timescale
seasons_cosmo = opts.seasons_cosmo
nrandom = opts.nrandom
fitparam_names = opts.fitparam_names
fitparam_values = opts.fitparam_values
prior = opts.prior


# load OS files to process
fis = pd.read_csv(surveyList, comment='#')

# loop on files and create batches

seasons_cosmo = []
for i in range(2, 12):
    dd = list(range(1, i))
    seasons_cosmo.append(dd)

seasons_cosmo = list(range(1, 11))


script = 'run_scripts/cosmology/cosmology.py'
for i, row in fis.iterrows():
    wfd_tagsurvey = 'notag'
    dd_tagsurvey = 'notag'
    """
    if 'wfd_tagsurvey' in fis.columns:
        wfd_tagsurvey = row['wfd_tagsurvey']
    if 'dd_tagsurvey' in fis.columns:
        dd_tagsurvey = row['dd_tagsurvey']

    processName = 'cosmo_{}_{}_{}'.format(
        dbName_DD, wfd_tagsurvey, dd_tagsurvey)
    """
    ttag = row['survey'].split('survey_scenario_')[-1]
    processName = 'cosmo_fit_{}'.format(ttag)
    mybatch = BatchIt(processName=processName)
    params = {}
    params['dataDir_DD'] = inputDir_DD
    params['dbName_DD'] = dbName_DD
    params['dataDir_WFD'] = inputDir_WFD
    params['dbName_WFD'] = dbName_WFD
    params['outDir'] = outDir
    params['survey'] = '{}.csv'.format(row['survey'])
    params['outName'] = row['outName']
    params['low_z_optimize'] = low_z_optimize
    params['timescale'] = timescale
    params['nrandom'] = nrandom
    params['seasons_cosmo'] = ','.join(list(map(str, seasons_cosmo)))
    params['wfd_tagsurvey'] = wfd_tagsurvey
    params['dd_tagsurvey'] = dd_tagsurvey
    params['fitparam_names'] = fitparam_names
    params['fitparam_values'] = fitparam_values
    params['prior'] = prior
    mybatch.add_batch(script, params)
    mybatch.go_batch()
