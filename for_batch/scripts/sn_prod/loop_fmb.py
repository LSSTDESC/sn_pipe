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
parser.add_option("--x1_type", type=str, default='unique',
                  help="x1 type of run [%default]")    
parser.add_option("--color_type", type=str, default='unique',
                  help="color type of run [%default]")
parser.add_option("--z_type", type=str, default='uniform',
                  help="z type of run [%default]")
parser.add_option("--daymax_type", type=str, default='random',
                  help="daymax type of run [%default]")
parser.add_option("--config_atmos", type=str, default='confe',
                  help="color [%default]") 
parser.add_option("--DD_list", type=str,
                  default='COSMOS,CDFS,EDFS_a,EDFS_b,ELAISS1,XMM-LSS',
                  help="List of DDFs to process [%default]")
parser.add_option('--dbList', type=str, default='DD_fbs_5.3_extract.csv',
                  help='list of OS to process [%default]')
parser.add_option('--runType', type=str, default='DDF',
                  help='type of run to process [%default]')

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
dbList = opts.dbList
runType = opts.runType

pp = vars(opts)

ccols = ['x1_type','color_type','z_type','daymax_type']

scr = 'python for_batch/scripts/sn_prod/prod_simu_ddf_fmb.py'
ntrial = 1000
if opts.config_atmos == 'confe':
    ntrial=1

dz = 0.05
zmin = 0.
zmax = 1.15

if runType == 'WFD':
    zmax = 0.5
z = np.arange(zmin,zmax,dz)

for i in range(len(z)-1):
    nbins=1
    nsn = 100
    zmin_l = z[i]+i*dz
    if zmin_l < 0.01:
        zmin_v = 0.01
    else:
        zmin_v = zmin_l
    zmax_l = zmin_v+dz
    
    if zmin_l >=0.8:
        nsn = 300
        
    if zmin_l > zmax:
        break
    cmd = scr
    cmd += ' --z_min={} --z_max={}'.format(np.round(zmin_v,2),np.round(zmax_l,2))
    cmd += ' --nbins={} --x1={} --color={}'.format(nbins,x1,color)
    cmd += ' --nsn={}'.format(nsn)
    for vv in atm_params:
        cmd += ' --{}={}'.format(vv,sel_atmos[vv].values[0])
    cmd += " --config_atmos={}".format(opts.config_atmos)
    cmd += " --ntrial={}".format(ntrial)
    cmd += " --DD_list={}".format(dd_list)
    cmd += " --dbList={}".format(dbList)
    cmd += " --runType={}".format(opts.runType)
    for vv in ccols:
        cmd += " --{}={}".format(vv,pp[vv])
    print(cmd)
    os.system(cmd)
