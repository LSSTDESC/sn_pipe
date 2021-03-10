import numpy as np
from optparse import OptionParser
import os


def cmd(x1=-2.0, color=0.2, ebv=0.0, bluecutoff=380., redcutoff=800., error_model=1, errmodrel=0.1, simu='sn_cosmo', fitter='sn_cosmo', nproc=4, outputDir='.', configFile='test.csv'):

    scriptName = 'run_scripts/fakes/simu_fit.py'
    script_cmd = 'python {}'.format(scriptName)
    script_cmd += ' --SN_x1_min {}'.format(x1)
    script_cmd += ' --SN_x1_type unique'
    script_cmd += ' --SN_color_min {}'.format(color)
    script_cmd += ' --SN_color_type unique'
    script_cmd += ' --SN_ebvofMW {}'.format(ebv)
    script_cmd += ' --SN_blueCutoff {}'.format(bluecutoff)
    script_cmd += ' --SN_redCutoff {}'.format(redcutoff)
    script_cmd += ' --Simulator_errorModel {}'.format(error_model)
    script_cmd += ' --LCSelection_errmodrel {}'.format(errmodrel)
    script_cmd += ' --Simulator_name sn_simulator.{}'.format(simu)
    script_cmd += ' --Fitter_name sn_fitter.fit_{}'.format(fitter)
    script_cmd += ' --OutputSimu_save 0'
    script_cmd += ' --MultiprocessingFit_nproc {}'.format(nproc)
    script_cmd += ' --outputDir {}'.format(outputDir)
    script_cmd += ' --config {}'.format(configFile)
    script_cmd += ' --SN_z_min 0.01'
    script_cmd += ' --SN_z_max 1.'
    script_cmd += ' --SN_z_type uniform'
    script_cmd += ' --SN_z_step 0.05'
    script_cmd += ' --Observations_coadd 0'
    script_cmd += ' --Observations_fieldtype Fake'
    script_cmd += ' --LCSelection_nbands 0'
    script_cmd += ' --zlim_calc 1'

    return script_cmd


parser = OptionParser()

parser.add_option(
    '--outputDir', help='main output directory [%default]', default='/sps/lsst/users/gris/config_zlim', type=str)
parser.add_option(
    '--config', help='config file of parameters [%default]', default='config_z_test.csv', type=str)

opts, args = parser.parse_args()

cmd_ = cmd(outputDir=opts.outputDir, configFile=opts.config)

os.system(cmd_)
