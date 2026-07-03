import os

from optparse import OptionParser
from sn_tools.sn_io import make_dict_from_config
from sn_tools.sn_io import add_parser

confDict= make_dict_from_config('for_batch/input/sn_prod', 'config_fmb_prod.txt')

parser = OptionParser()
# parser for simulation parameters : 'dynamical' generation
add_parser(parser, confDict)

opts, args = parser.parse_args()

pp = vars(opts)

ttag = 'fmb'
if pp['x1_type'] == 'random':
    ttag = 'all'

cmd="python for_batch/scripts/sim_to_fit/prodIt.py" 
cmd += " --dbList_DD={}".format(pp['dbList'])
cmd += " --dbList_WFD={}".format(pp['dbList'])
cmd += " --Observations_coadd=0 --mem=10Gb"
cmd += " --InstrumentSimu_ntrial_zp={}".format(pp['ntrial'])
cmd += " --outDir_DD=dd_sn_{}_{}".format(ttag,pp['config_atmos'])
cmd += " --outDir_WFD=wfd_sn_{}_{}".format(ttag,pp['config_atmos'])
cmd += " --sigma_airmass={}".format(pp['sigma_airmass'])
cmd += " --sigma_ozone={}".format(pp['sigma_ozone'])
cmd += " --sigma_aerosol={}".format(pp['sigma_aerosol'])
cmd += " --sigma_pwv={}".format(pp['sigma_pwv'])
tag_script = 'sn_{}'.format(ttag)
if pp['x1_type'] == 'unique':
    tag_script += '_{}_{}'.format(pp['x1'],pp['color'])
if pp['z_type'] == 'unique':
    tag_script += '_{}'.format(pp['z_min'])
else:
    tag_script += '_{}_{}'.format(pp['z_min'],pp['z_max'])
    
tag_script += '_{}'.format(pp['config_atmos'])

cmd += " --tag_script={}".format(tag_script) 
cmd += " --runType={}".format(pp['runType'])
cmd += " --SN_z_type={}".format(pp['z_type'])
cmd += " --SN_z_min={} --SN_z_max={}".format(pp['z_min'],pp['z_max'])
cmd += " --SN_daymax_type={}".format(pp['daymax_type'])
cmd += " --SN_z_nbins={}".format(pp['nbins'])
cmd += " --SN_x1_type={}".format(pp['x1_type']) 
cmd += " --SN_x1_min={}".format(pp['x1'])
cmd += " --SN_color_type={}".format(pp['color_type'])
cmd += " --SN_color_min={}".format(pp['color'])
cmd += " --SN_NSNabsolute_DDF={}".format(pp['nsn']) 
cmd += " --SN_NSNabsolute_WFD={}".format(pp['nsn'])
cmd += " --DD_list={}".format(pp['DD_list'])

print(cmd)
os.system(cmd)
