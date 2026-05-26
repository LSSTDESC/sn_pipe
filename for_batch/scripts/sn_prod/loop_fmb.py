import os
from optparse import OptionParser
import pandas as pd
desc = 'script to launch a set of sim_to_fit for dedicted (x1,color) SNe Ia'
parser = OptionParser(description=desc)

parser.add_option("--x1", type=float, default=0.0,
                  help="stretch [%default]")    
parser.add_option("--color", type=float, default=0.0,
                  help="color [%default]")
parser.add_option("--config_atmos", type=str, default='confa',
                  help="color [%default]") 

opts, args = parser.parse_args()

df_atmos = pd.read_csv('for_batch/input/sn_prod/config_atmos_orig.csv',comment='#')

idx = df_atmos['config'] == opts.config_atmos
sel_atmos = df_atmos[idx]

atm_params = ['sigma_airmass','sigma_ozone','sigma_aerosol','sigma_pwv']

z = [(0.01,0.2),(0.2,0.4),(0.4,0.6),(0.6,0.8),(0.8,0.85),(0.9,0.95),(1.0,1.1)]
nbins = [2,3,3,4,2,2,2]
nsn = [100]*4+[300]*3
x1 = opts.x1
color = opts.color

scr = 'python for_batch/scripts/sn_prod/prod_simu_ddf_fmb.py'

for i in range(len(z)):
    zmin = z[i][0]
    zmax = z[i][1]
    
    cmd = scr
    cmd += ' --zmin={} --zmax={}'.format(zmin,zmax)
    cmd += ' --nbins={} --x1={} --color={}'.format(nbins[i],x1,color)
    cmd += ' --nsn={}'.format(nsn[i])
    for vv in atm_params:
        cmd += ' --{}={}'.format(vv,sel_atmos[vv].values[0])
    cmd += " --config_atmos={}".format(opts.config_atmos)
    
    print(cmd)
    os.system(cmd)
