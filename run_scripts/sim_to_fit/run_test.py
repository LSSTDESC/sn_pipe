import pandas as pd
import sn_script_input
import os
from dataclasses import dataclass
from sn_tools.sn_io import make_dict_from_config
from sn_tools.sn_fake_utils import add_option
from optparse import OptionParser


@dataclass
class CombiRun:
    simulator_name: str
    simulator_model: str
    simulator_version: str
    fitter_name: str
    fitter_model: str
    fitter_version: str


def go(script, pars):
    """
    Function to buil the cmd line to run a script

    Parameters
    ----------
    script : str
        script name.
    pars : dict
        script parameters.

    Returns
    -------
    cmd : str
        cmd line.

    """
    cmd = script
    for key, vals in pars.items():
        cmd += ' --{} {}'.format(key, vals)
    return cmd


def make_csv(fName, vprod):
    """
    Function to build a csv file used to plot the results

    Parameters
    ----------
    fName : str
        output name file.
    vprod : list(list)
        parameters to write in the csv file.

    Returns
    -------
    None.

    """

    tt = open(fName, "w")
    tt.write('nickname,dirfile,filename,plotname\n')
    for vv in vprod:
        tt.write('{},{},{},{}\n'.format(vv[0], vv[1], vv[2], vv[3]))


def add_threads(tt):
    """
    Function to add info in the script file

    Parameters
    ----------
    tt : file
        where to write infos.

    Returns
    -------
    None.

    """

    ll = ['export MKL_NUM_THREADS=1 \n',
          'export NUMEXPR_NUM_THREADS=1 \n',
          'export OMP_NUM_THREADS=1 \n',
          'export OPENBLAS_NUM_THREADS=1']

    for vv in ll:
        tt.write(vv)


def main_params():
    """
        Function to get the main parameters of the script run_to_sim.py

    Returns
    -------
    pars : dict
        parameters.

    """

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
    pars['SN_z_min'] = 0.2
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


def add_combis(script, pars_combi, vprod, plotNames):
    """
    Function to add the combination (simulator,fitter)

    Parameters
    ----------
    script : file
        where to write infos.
    pars_combi : dict(dataclass)
        combi dict
    vprod : list(list)
        parameters to produce the csv file
    plotNames: dict
      dict of plotNames

    Returns
    -------
    None.

    """

    cmd = 'python run_scripts/sim_to_fit/run_sim_to_fit.py'
    # get main
    pars = main_params()
    cmd = go(cmd, pars)
    for key, vals in pars_combi.items():
        script.write('\n \n')
        cmd_ = cmd
        cmd_ += ' --ProductionIDSimu {}'.format(key)
        cmd_ += ' --Simulator_name sn_simulator.{}'.format(vals.simulator_name)
        cmd_ += ' --Simulator_model {}'.format(vals.simulator_model)
        cmd_ += ' --Simulator_version {}'.format(vals.simulator_version)
        cmd_ += ' --Fitter_name sn_fitter.fit_{}'.format(vals.fitter_name)
        cmd_ += ' --Fitter_model {}'.format(vals.fitter_model)
        cmd_ += ' --Fitter_version {}'.format(vals.fitter_version)
        script.write(cmd_)
        va = [key, pars['OutputFit_directory'], 'SN_{}.hdf5'.format(key),
              plotNames[key]]
        vprod.append(va)


def add_plot(script, fName):
    """
    Function to add a line in the script for plot

    Parameters
    ----------
    script : file
        where to write infos.
    fName : str
        fileName for the plot (csv file).

    Returns
    -------
    None.

    """

    sc = 'python plot_scripts/fitlc/plot_lcfit.py'
    pars = {}
    pars['prodids'] = fName

    script.write('\n \n')
    script.write(go(sc, pars))


def start_script(script):
    """
    Function to write the start of the script file

    Parameters
    ----------
    script : script file
        where to write infos.

    Returns
    -------
    None.

    """

    script.write('#!/bin/bash \n')
    add_threads(script)


def add_genfakes(script, dict_opt):
    """
    Function to add a line in the script to add obs gen

    Parameters
    ----------
    script : script file
        where to write infos.
    dict_opt : dict
        parameters of the script.

    Returns
    -------
    None.

    """

    scr = 'python run_scripts/fakes/make_fake.py'
    script.write('\n \n')
    script.write(go(scr, dict_opt))


def add_sequence(script, dict_opt, vprod, combis_simu, seqName, plotName):
    """
    Function to add a sequence gen obs+sim_to_fit

    Parameters
    ----------
    script : script file
        xhere to write infos.
    dict_opt : dict
        fake obs params.
    vprod : list(list)
        params for the plot (csv file).
    combis_simu: pandas df
      simu+fit configs.
    seqName: str
        name of the seauence
    plotName: str
        plot name of the sequence

    Returns
    -------
    None.

    """

    add_genfakes(script, dict_opt)

    nvisits = 0
    for b in 'grizy':
        nvisits += dict_opt['Nvisits_{}'.format(b)]

    print('nvisits', nvisits)

    combis = {}
    """
    combis['full_salt3_full_salt3_{}'.format(nvisits)] = CombiRun(
        'sn_cosmo', 'salt3', '1.0', 'sn_cosmo', 'salt3', '1.0')
    combis['fast_salt3_fast_salt3_{}'.format(nvisits)] = CombiRun(
        'sn_fast', 'salt3', '1.0', 'sn_fast', 'salt3', '1.0')
    """
    plotNames = {}
    for i, row in combis_simu.iterrows():
        combis[seqName] = CombiRun(
            row['simulator_name'],
            row['simulator_model'],
            row['simulator_version'],
            row['fitter_name'],
            row['fitter_model'],
            row['fitter_version'])
        plotNames[seqName] = plotName

    add_combis(script, combis, vprod, plotNames)


def param_ana(dict_opt, parName='Nvisits'):

    nl = []
    vv = {}
    for b in 'grizy':
        ll = dict_opt['{}_{}'.format(parName, b)].split(',')
        vv[b] = list(map(int, ll))
        nl.append(len(vv[b]))

    """
    nmax = np.max(nl)

    for b in 'grizy':
        nref = len(vv[b])
        if nref < nmax:
            vv[b] += [vv[b]]*(nmax-nref)
    """
    return vv


# this is to load option for fake cadence
path = sn_script_input.__path__
confDict_fake = make_dict_from_config(path[0], 'config_test_simtofit.txt')


parser = OptionParser(description='Script to generate fake obs, simu and fit')

add_option(parser, confDict_fake)


opts, args = parser.parse_args()

scriptName = "mytest.sh"
script = open(scriptName, "w")

# first lines of the script
start_script(script)

# Generation of Fake Observations
dict_opt = vars(opts)

dict_opt['saveData'] = 1

vprod = []

# visits = param_ana(dict_opt, 'Nvisits')

# read config files
combis_obs = pd.read_csv(opts.config_obs, comment='#')
combis_simu = pd.read_csv(opts.config_simu, comment='#')

ccols = list(combis_obs.columns)
print(ccols)
ccols.remove('tagName')
ccols.remove('plotName')

del dict_opt['config_obs']
del dict_opt['config_simu']

for j, row in combis_obs.iterrows():
    for col in ccols:
        dict_opt[col] = row[col]

    seqName = row['tagName']
    plotName = row['plotName']
    add_sequence(script, dict_opt, vprod, combis_simu, seqName, plotName)

# prepare outputFile for plot
fName = 'plot_test.csv'
make_csv(fName, vprod)

add_plot(script, fName)

script.close()

# now run the script
os.system("sh "+scriptName)
