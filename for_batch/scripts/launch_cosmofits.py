import os
from optparse import OptionParser



parser = OptionParser()
parser.add_option("--Ny", type=int, default=20,help="y-band max visits at z=0.9 [%default]")
parser.add_option("--fit_parameters", type=str, default='Om,w0',
                  help="parameters to fit [%default]")
parser.add_option("--sigma_mu_photoz", type=str, default='None',
                  help="mu error from photoz [%default]")
parser.add_option("--sigma_mu_bias_x1_color", type=str, default='sigma_mu_bias_x1_color_1_sigma',
                  help="mu error bias from x1 and color n-sigma variation [%default]")
parser.add_option("--outDir", type=str, default='/sps/lsst/users/gris/fake/Fit_bias',
                  help="output directory [%default]")

opts, args = parser.parse_args()

Ny = opts.Ny
fit_parameters = opts.fit_parameters
sigma_mu_photoz = opts.sigma_mu_photoz
sigma_mu_bias_x1_color = opts.sigma_mu_bias_x1_color
outDir = opts.outDir

configs = ['config_cosmoSN_universal_10.csv',
           'config_cosmoSN_deep_rolling_0.90_0.90_2_2.csv',
           'config_cosmoSN_deep_rolling_0.80_0.80_2_2.csv',
           'config_cosmoSN_deep_rolling_2_2_mini.csv']
           #'config_cosmoSN_deep_rolling_0.80_2.csv']
           #'config_cosmoSN_deep_rolling_0.90_0.90_3_3.csv',
           #'config_cosmoSN_deep_rolling_0.80_0.80_3_3.csv']
#configs = ['config_cosmoSN_deep_rolling_2_2_mini.csv']

configs = ['config_cosmoSN_deep_rolling_0.80_0.80_yearly.csv',
           'config_cosmoSN_deep_rolling_2_2_mini_yearly.csv',
           'config_cosmoSN_universal_yearly.csv']

configs = ['config_cosmoSN_deep_rolling_2_2_mini_yearly.csv']

configs = ['config_cosmoSN_deep_rolling_2_2_mini_0.65.csv',
           'config_cosmoSN_deep_rolling_2_2_mini_0.60.csv',
           'config_cosmoSN_deep_rolling_2_2_mini_0.65_yearly.csv',
           'config_cosmoSN_deep_rolling_2_2_mini_0.60_yearly.csv']

configs = ['config_cosmoSN_deep_rolling_2_2_mini_0.65_yearly.csv']

scr = 'python for_batch/scripts/loop_cosmo_scen.py --fileName'

for conf in configs:
    cmd = '{} {}'.format(scr,conf)
    cmd += ' --outDir {}'.format(outDir)
    cmd += ' --Ny {}'.format(Ny)
    cmd += ' --fit_parameters {}'.format(fit_parameters)
    cmd += ' --sigma_mu_photoz {}'.format(sigma_mu_photoz)
    cmd += ' --sigma_mu_bias_x1_color {}'.format(sigma_mu_bias_x1_color)
    print(cmd)
    os.system(cmd)