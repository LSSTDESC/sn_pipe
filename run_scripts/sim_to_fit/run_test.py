import subprocess
import os
from dataclasses import dataclass


@dataclass
class CombiRun:
    simulator_name: str
    simulator_model: str
    simulator_version: str
    fitter_name: str
    fitter_model: str
    fitter_version: str


def go(script, pars):
    cmd = script
    for key, vals in pars.items():
        cmd += ' --{} {}'.format(key, vals)
    return cmd


def make_csv(fName, vprod):

    tt = open(fName, "w")
    tt.write('nickname,dirfile,filename\n')
    for vv in vprod:
        tt.write('{},{},{}\n'.format(vv[0], vv[1], vv[2]))


def add_threads(tt):

    ll = ['export MKL_NUM_THREADS=1 \n',
          'export NUMEXPR_NUM_THREADS=1 \n',
          'export OMP_NUM_THREADS=1 \n',
          'export OPENBLAS_NUM_THREADS=1']

    for vv in ll:
        tt.write(vv)


def main_params():

    pars = {}
    pars['dbName'] = 'Fakes'
    pars['dbDir '] = '../Fake_Observations'
    pars['nside'] = 128
    pars['SN_NSNabsolute '] = 1
    pars['OutputSimu_throwafterdump'] = 0
    pars['SN_x1_type'] = 'unique'
    pars['SN_x1_min '] = -2.0
    pars['SN_color_type'] = 'unique'
    pars['SN_color_min'] = 0.2
    pars['SN_z_type'] = 'uniform'
    pars['SN_z_min'] = 0.01
    pars['SN_z_max'] = 1.1
    pars['SN_z_step'] = 0.02
    pars['SN_daymax_type'] = 'unique'
    pars['SN_daymax_step '] = 1
    pars['MultiprocessingSimu_nproc '] = 1
    pars['fieldType'] = 'Fake'
    pars['OutputSimu_save'] = 0
    pars['OutputSimu_directory'] = 'Output_SN/Fakes'
    pars['OutputFit_directory'] = 'Output_SN/Fakes'
    # pars['ProductionIDSimu'] = 'Fake_simufast_salt3'
    pars['SN_ebvofMW'] = 0.0
    pars['Observations_coadd'] = 0
    # pars['Simulator_model'] = 'salt3'
    pars['MultiprocessingFit_nproc'] = 8

    return pars


def add_combis(script, pars_combi, vprod):

    cmd = 'python run_scripts/sim_to_fit/run_sim_to_fit.py'
    # get main
    pars = main_params()
    cmd = go(cmd, pars)
    for key, vals in pars_combi.items():
        script.write('\n \n')
        print('jj', vals)
        cmd_ = cmd
        cmd_ += ' --ProductionIDSimu {}'.format(key)
        cmd_ += ' --Simulator_name sn_simulator.{}'.format(vals.simulator_name)
        cmd_ += ' --Simulator_model {}'.format(vals.simulator_model)
        cmd_ += ' --Simulator_version {}'.format(vals.simulator_version)
        cmd_ += ' --Fitter_name sn_fitter.fit_{}'.format(vals.fitter_name)
        cmd_ += ' --Fitter_model {}'.format(vals.fitter_model)
        cmd_ += ' --Fitter_version {}'.format(vals.fitter_version)
        script.write(cmd_)
        va = [key, pars['OutputFit_directory'], 'SN_{}.hdf5'.format(key)]
        vprod.append(va)


def add_plot(script, fName):

    sc = 'python plot_scripts/fitlc/plot_lcfit.py'
    pars = {}
    pars['prodids'] = fName

    script.write('\n \n')
    script.write(go(sc, pars))


def make_script(script):

    script.write('#!/bin/bash \n')
    add_threads(script)


scriptName = "mytest.sh"
script = open(scriptName, "w")
make_script(script)

combis = {}
combis['full_salt3_full_salt3'] = CombiRun(
    'sn_cosmo', 'salt3', '1.0', 'sn_cosmo', 'salt3', '1.0')
combis['fast_salt3_fast_salt3'] = CombiRun(
    'sn_fast', 'salt3', '1.0', 'sn_fast', 'salt3', '1.0')

vprod = []
add_combis(script, combis, vprod)

# prepare outputFile for plot
fName = 'plot_test.csv'
make_csv(fName, vprod)

add_plot(script, fName)

script.close()
os.system("sh "+scriptName)
#os.system("sh run_scripts/sim_to_fit/run_test.sh")
