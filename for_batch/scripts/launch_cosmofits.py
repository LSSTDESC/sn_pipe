import os
from optparse import OptionParser



parser = OptionParser()
parser.add_option("--Ny", type=int, default=40,help="y-band max visits at z=0.9 [%default]")
parser.add_option("--fit_parameters", type=str, default='Om,w0',
                  help="parameters to fit [%default]")
parser.add_option("--sigma_mu_photoz", type=str, default='None',
                  help="mu error from photoz [%default]")
parser.add_option("--sigma_mu_bias_x1_color", type=str, default='sigma_mu_bias_x1_color_1_sigma',
                  help="mu error bias from x1 and color n-sigma variation [%default]")
parser.add_option("--outDir", type=str, default='/sps/lsst/users/gris/cosmofit/Fit_bias',
                  help="output directory [%default]")
parser.add_option("--nsn_bias_simu", type=str, default='nsn_bias_Ny_40',
                  help="nsn_bias file for distance moduli simulation [%default]")
parser.add_option("--tagscript", type=str, default='',
                  help="tag for the scripts[%default]")
parser.add_option("--surveytype", type=str, default='10_years',
                  help="survey type (10_years/yearly/all) [%default]")
parser.add_option("--nsn_WFD_yearly", type=int, default=-1,
                  help="nsn WFD per year [%default]")

opts, args = parser.parse_args()

Ny = opts.Ny
fit_parameters = opts.fit_parameters
sigma_mu_photoz = opts.sigma_mu_photoz
sigma_mu_bias_x1_color = opts.sigma_mu_bias_x1_color
outDir = opts.outDir
nsn_bias_simu = opts.nsn_bias_simu
tagscript = opts.tagscript
surveytype = opts.surveytype
nsn_WFD_yearly = opts.nsn_WFD_yearly

configs = []
nsn_WFD_yearly_list = []

if surveytype == '10_years':
    configs += ['config_cosmoSN_universal_10.csv',
                #'config_cosmoSN_deep_rolling_0.90_0.90_2_2.csv',
                'config_cosmoSN_deep_rolling_0.80_0.80_2_2.csv',
                'config_cosmoSN_deep_rolling_2_2_mini.csv']

if surveytype == 'yearly':

    configs += ['config_cosmoSN_deep_rolling_0.80_0.80_yearly.csv',
                'config_cosmoSN_deep_rolling_2_2_mini_yearly.csv',
                'config_cosmoSN_universal_yearly.csv']
           #'config_cosmoSN_deep_rolling_2_2_mini_0.65_yearly.csv',
           #'config_cosmoSN_deep_rolling_2_2_mini_0.60_yearly.csv']
nsn_WFD_yearly_list += [nsn_WFD_yearly]*len(configs)


#configs = ['config_cosmoSN_deep_rolling_2_2_mini_0.65_yearly.csv']

scr = 'python for_batch/scripts/loop_cosmo_scen.py --fileName'

for io,conf in enumerate(configs):
    cmd = '{} {}'.format(scr,conf)
    cmd += ' --outDir {}'.format(outDir)
    cmd += ' --Ny {}'.format(Ny)
    cmd += ' --fit_parameters {}'.format(fit_parameters)
    cmd += ' --sigma_mu_photoz {}'.format(sigma_mu_photoz)
    cmd += ' --sigma_mu_bias_x1_color {}'.format(sigma_mu_bias_x1_color)
    cmd += ' --nsn_bias_simu {}'.format(nsn_bias_simu)
    cmd += ' --tagscript {}'.format(tagscript)
    cmd += ' --nsn_WFD_yearly {}'.format(nsn_WFD_yearly_list[io])
    print(cmd)
    os.system(cmd)
