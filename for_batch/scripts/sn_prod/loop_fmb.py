import os
from optparse import OptionParser
import pandas as pd
import numpy as np

desc = 'script to launch a set of sim_to_fit for dedicted (x1,color) SNe Ia'
parser = OptionParser(description=desc)

parser.add_option("--x1", type=float, default=-2.0,
                  help="stretch [%default]")    
parser.add_option("--color", type=float, default=0.2,
                  help="color [%default]")
parser.add_option("--config_atmos", type=str, default='confe',
                  help="color [%default]") 
parser.add_option("--DD_list", type=str,
                  default='COSMOS,CDFS,EDFS_a,EDFS_b,ELAISS1,XMM-LSS',
                  help="List of DDFs to process [%default]")
parser.add_option('--dbList_DD', type=str, default='DD_fbs_5.3_extract.csv',
                  help='list of OS to process [%default]')

opts, args = parser.parse_args()

df_atmos = pd.read_csv('for_batch/input/sn_prod/config_atmos_orig.csv',comment='#')

idx = df_atmos['config'] == opts.config_atmos
sel_atmos = df_atmos[idx]

assert len(sel_atmos) > 0,"pb: the configuration was not found"

atm_params = ['sigma_airmass','sigma_ozone','sigma_aerosol','sigma_pwv']

z = [(0.01,0.2),(0.2,0.4),(0.4,0.6),(0.6,0.8),(0.8,0.85),(0.9,0.95),(1.0,1.1)]
nbins = [2,3,3,4,2,2,2]
nsn = [100]*4+[300]*3
x1 = opts.x1
color = opts.color
dd_list = opts.DD_list
dbList_DD = opts.dbList_DD

scr = 'python for_batch/scripts/sn_prod/prod_simu_ddf_fmb.py'
ntrial = 1000
if opts.config_atmos == 'confe':
    ntrial=1

dz = 0.05
z = np.arange(0.,1.15,dz)

for i in range(len(z)-1):
    nbins=1
    nsn = 100
    zmin = z[i]
    zmax = zmin+dz
    
    if zmin >=0.8:
        nsn = 300
    cmd = scr
    cmd += ' --zmin={} --zmax={}'.format(np.round(zmin,2),np.round(zmax,2))
    cmd += ' --nbins={} --x1={} --color={}'.format(nbins,x1,color)
    cmd += ' --nsn={}'.format(nsn)
    for vv in atm_params:
        cmd += ' --{}={}'.format(vv,sel_atmos[vv].values[0])
    cmd += " --config_atmos={}".format(opts.config_atmos)
    cmd += " --ntrial={}".format(ntrial)
    cmd += " --DD_list={}".format(dd_list)
    cmd += " --dbList_DD={}".format(dbList_DD)
    print(cmd)
    os.system(cmd)
