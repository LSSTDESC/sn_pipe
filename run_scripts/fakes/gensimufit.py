import numpy as np
from optparse import OptionParser
import os
import pandas as pd
import csv


def cmd(x1=-2.0, color=0.2, ebv=0.0, bluecutoff=380., redcutoff=800., error_model=1, errmodrel=0.1, simu='sn_cosmo', fitter='sn_cosmo', zlim_calc=0, nsn_calc=0, survey_area=0.21, mbcov_estimate=0, nproc=4, outputDir='.', config=pd.DataFrame(), confName='', tagprod=-1, zmin=0.1, zmax=1.0, zstep=0.05, plot=0):

    #configName = 'config_z_{}.csv'.format(tagprod)
    configName = confName.replace('.csv', '_zlim.csv')
    my_dict = dict(config.to_dict())
    config.to_csv(configName, index=False)

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
    script_cmd += ' --LCSelection_errmodinlcerr 0'
    script_cmd += ' --Simulator_name sn_simulator.{}'.format(
        np.unique(config['simulator']).item())
    script_cmd += ' --Fitter_name sn_fitter.fit_{}'.format(
        np.unique(config['fitter']).item())
    script_cmd += ' --OutputSimu_save 0'
    script_cmd += ' --MultiprocessingFit_nproc {}'.format(nproc)
    script_cmd += ' --outputDir {}'.format(outputDir)
    script_cmd += ' --config {}'.format(configName)
    script_cmd += ' --SN_z_min {}'.format(zmin)
    script_cmd += ' --SN_z_max {}'.format(zmax)
    script_cmd += ' --SN_z_type uniform'
    script_cmd += ' --SN_z_step {}'.format(zstep)
    script_cmd += ' --Observations_coadd 0'
    script_cmd += ' --Observations_fieldtype Fake'
    script_cmd += ' --LCSelection_nbands 0'
    script_cmd += ' --zlim_calc {}'.format(zlim_calc)
    script_cmd += ' --nsn_calc {}'.format(nsn_calc)
    script_cmd += ' --survey_area {}'.format(survey_area)
    script_cmd += ' --mbcov_estimate {}'.format(mbcov_estimate)
    script_cmd += ' --tagprod {}'.format(tagprod)
    script_cmd += ' --plot {}'.format(plot)

    return script_cmd


parser = OptionParser()

parser.add_option(
    '--outputDir', help='main output directory [%default]', default='/sps/lsst/users/gris/config_zlim', type=str)
parser.add_option(
    '--config', help='config file of parameters [%default]', default='config_z_test.csv', type=str)
parser.add_option(
    '--tagprod', help='tag for output file [%default]', default=-1, type=int)
parser.add_option(
    '--zlim_calc', help='to estimate zlim or not [%default]', default=0, type=int)
parser.add_option(
    '--nsn_calc', help='to estimate nsn or not [%default]', default=0, type=int)
parser.add_option(
    '--survey_area', help='area for nsn estimation in deg2 [%default]', default=0.21, type=float)
parser.add_option(
    '--zmin', help='min redshift value  [%default]', default=0.5, type=float)
parser.add_option(
    '--zmax', help='max redshift value [%default]', default=1.0, type=float)
parser.add_option(
    '--zstep', help='redshift step value [%default]', default=0.05, type=float)
parser.add_option(
    '--mbcov_estimate', help='to estimate mb after fit [%default]', default=0, type=int)
parser.add_option(
    '--nproc', help='nproc for multiproc [%default]', default=8, type=int)
parser.add_option('--plot', type=int, default=0,
                  help='to display some results [%default]')

opts, args = parser.parse_args()

# clean outputdir if already exist
if os.path.exists(opts.outputDir):
    cmd_d = 'rm -rf {}/*'.format(opts.outputDir)
    os.system(cmd_d)

confp = pd.read_csv(opts.config, comment='#')
confp['season'] = confp['season'].astype(str)
io = -1
for simu in confp['simulator'].unique():
    idx = confp['simulator'] == simu
    sel = confp[idx]
    for fitter in sel['fitter'].unique():
        ida = sel['fitter'] == fitter
        io += 1
        selb = sel[ida]
        x1_color = selb[['x1', 'color', 'error_model']].to_records(index=False)
        for (x1, color, error_model) in np.unique(x1_color[['x1', 'color', 'error_model']]):
            cmd_ = cmd(x1=x1, color=color,
                       error_model=error_model,
                       zlim_calc=opts.zlim_calc,
                       nsn_calc=opts.nsn_calc,
                       survey_area=opts.survey_area,
                       mbcov_estimate=opts.mbcov_estimate,
                       nproc=opts.nproc,
                       outputDir=opts.outputDir,
                       config=sel[ida],
                       confName=opts.config,
                       tagprod=io,
                       zmin=opts.zmin,
                       zmax=opts.zmax,
                       zstep=opts.zstep,
                       plot=opts.plot)

            os.system(cmd_)
