import os
import glob


dirScen = 'input/DESC_cohesive_strategy'

list_scen = glob.glob('{}/DDF_DESC*.csv'.format(dirScen))

list_scen += ['{}/DDF_SCOC_pII.csv'.format(dirScen),
              '{}/DDF_Univ_SN.csv'.format(dirScen),
              '{}/DDF_Univ_WZ.csv'.format(dirScen)]

cmd = 'python run_scripts/fakes/make_scenario.py --configFile {}'.format(
    dirScen)

for scen in list_scen:
    cmd_ = '{}/{}'.format(cmd, scen.split('/')[-1])
    print(cmd_)
    os.system(cmd_)
