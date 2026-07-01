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

cmd="python for_batch/scripts/sim_to_fit/prodIt.py" 
cmd += " --dbList_DD={}".format(pp['dbList'])
cmd += " --dbList_WFD={}".format(pp['dbList'])
cmd += " --Observations_coadd=0 --mem=10Gb"
cmd += " --InstrumentSimu_ntrial_zp={}".format(pp['ntrial'])
cmd += " --outDir_DD=sn_fmb_{}".format(pp['config_atmos'])
cmd += " --outDir_WFD=wfd_sn_fmb_{}".format(pp['config_atmos'])
cmd += " --sigma_airmass={}".format(pp['sigma_airmass'])
cmd += " --sigma_ozone={}".format(pp['sigma_ozone'])
cmd += " --sigma_aerosol={}".format(pp['sigma_aerosol'])
cmd += " --sigma_pwv={}".format(pp['sigma_pwv'])
cmd += " --tag_script=sn_fmb_{}_{}_{}_{}_{}".format(pp['x1'],pp['color'],
                                                    pp['zmin'],pp['zmax'],
                                                    pp['config_atmos']) 
cmd += " --runType={}".format(pp['runType'])
cmd += " --SN_z_min={} --SN_z_max={}".format(pp['zmin'],pp['zmax'])
cmd += " --SN_z_nbins={}".format(pp['nbins'])
cmd += " --SN_x1_type=unique --SN_x1_min={}".format(pp['x1'])
cmd += " --SN_color_type=unique --SN_color_min={}".format(pp['color'])
cmd += " --SN_NSNabsolute_DDF={}".format(pp['nsn']) 
cmd += " --SN_z_type=unique --SN_daymax_type=random"
cmd += " --DD_list={}".format(pp['DD_list'])

print(cmd)
os.system(cmd)
