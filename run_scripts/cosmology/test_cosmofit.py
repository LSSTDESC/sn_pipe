import os
from optparse import OptionParser


parser = OptionParser(description='Script to test cosmofit')

parser.add_option('--dataDir_DD', type=str,
                  default='../Output_SN_sigmaInt_0.0_Hounsell_z_smflux_test_G10_JLA',
                  help='DD data dir [%default]')
parser.add_option('--dbName_DD', type=str,
                  default='DDF_DESC_0.80_WZ_0.07',
                  help='DD dbName [%default]')
parser.add_option('--dataDir_WFD', type=str,
                  default='../Output_SN_WFD_sigmaInt_0.0_Hounsell_z_smflux_test_G10_JLA',
                  help='WFD data dir [%default]')
parser.add_option('--dbName_WFD', type=str,
                  default='baseline_v3.0_10yrs',
                  help='WFD dbName [%default]')
parser.add_option('--timescale', type=str,
                  default='season',
                  help='timescale for fit estimation [%default]')
parser.add_option('--outDir', type=str,
                  default='../cosmo_fit_WFD_sigmaC_test',
                  help='output dir for results [%default]')
parser.add_option('--surveyDir', type=str,
                  default='../test_survey',
                  help='output survey dir [%default]')
parser.add_option('--survey', type=str,
                  default='survey_scenario_paper.csv',
                  help='survey configuration [%default]')
parser.add_option('--seasons', type=str,
                  default='1,2,3,4,5,6,7,8,9,10',
                  help='seasons to process [%default]')
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


cmd = 'python run_scripts/cosmology/cosmology.py'
cmd += ' --dataDir_DD={}'.format(opts.dataDir_DD)
cmd += ' --dbName_DD={}'.format(opts.dbName_DD)
cmd += ' --dataDir_WFD={}'.format(opts.dataDir_WFD)
cmd += ' --dbName_WFD={}'.format(opts.dbName_WFD)
cmd += ' --timescale={}'.format(opts.timescale)
cmd += ' --survey={}'.format(opts.survey)
cmd += ' --outDir={}'.format(opts.outDir)
cmd += ' --seasons={}'.format(opts.seasons)
cmd += ' --surveyDir={}'.format(opts.surveyDir)
cmd += ' --nrandom=1'
cmd += ' --plot_test=0 --test_mode=0 --nproc=1 --low_z_opti=0'
cmd += ' --fitparam_names={}'.format(opts.fitparam_names)
cmd += ' --fitparam_values={}'.format(opts.fitparam_values)
cmd += ' --prior={}'.format(opts.prior)

print(cmd)
os.system(cmd)
