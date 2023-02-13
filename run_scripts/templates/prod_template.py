#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 17:12:38 2023

@author: philippe.gris@clermont.in2p3.fr
"""
import os
from optparse import OptionParser
import numpy as np
from sn_tools.sn_utils import multiproc


def stack_lc(x1, color, ebvofMW, lcDir, outDir, sn_model,
             sn_version, **kwargs):
    """
    Method to stack light curves

    Parameters
    ----------
    x1 : float
        SN x1.
    color : float
        SN color.
    ebvofMW : float
        E(B-V) of MW.
    lcDir : str
        lc dir.
    outDir : str
        output dir.
    sn_model : str
        SN model (in sn_cosmo).
    sn_version : str
        SN model version (in sn_cosmo)
    **kwargs : dict
        potential additionnal params.

    Returns
    -------
    None.

    """

    pars = {}
    pars['x1'] = x1
    pars['color'] = color
    pars['ebvofMW'] = ebvofMW
    pars['lcDir'] = lcDir
    pars['outDir'] = outDir
    pars['sn_model'] = sn_model
    pars['sn_version'] = sn_version

    script = 'python run_scripts/templates/run_template_vstack.py'
    go(script, pars)

    """
    pars = []
    pars['x1'] = x1
    # stack produced LCs
   cmd = 'python run_scripts/templates/run_template_vstack.py'
   cmd = addoption(cmd, 'x1', x1)
   cmd = addoption(cmd, 'color', color)
   cmd = addoption(cmd, 'bluecutoff', bluecutoff)
   cmd = addoption(cmd, 'redcutoff', redcutoff)
   cmd = addoption(cmd, 'ebvofMW', ebvofMW)
   cmd = addoption(cmd, 'lcDir', '{}/fake_simu_data'.format(outDirLC))
   cmd = addoption(cmd, 'outDir', outDirTemplates)
   cmd = addoption(cmd, 'sn_model', sn_model)
   cmd = addoption(cmd, 'sn_version', sn_version)
   print(cmd)
   os.system(cmd)
   """


def go(script, params):
    """
    Method to execute a strip

    Parameters
    ----------
    script : str
        script to launch.
    params : dict
        script parameters.

    Returns
    -------
    None.

    """

    cmd = script
    for key, vals in params.items():
        cmd += ' --{} {}'.format(key, vals)

    print('executing', cmd)
    os.system(cmd)


def multi_simu(zvals, params, j, output_q=None):
    """
    Method to perform simulations using multiprocessing

    Parameters
    ----------
    zvals : list(float)
        redshifts.
    params : dict
        parameters.
    j : int
        tag for multiproc.
    output_q : multiprocessing queue, optional
        output queue for multiprocessing output. The default is None.

    Returns
    -------
    int
        output of the script (required to finish multiprocessing properly).

    """

    for z in zvals:
        params['z'] = z
        Simulation(**params)

    if output_q is not None:
        output_q.put({j: 0})
    else:
        return 0


class Simulation:
    def __init__(self, x1, color, z, sn_type, sn_model, sn_version, ebvofMW,
                 error_model,
                 maindir='.', **kwargs):
        """
        class to perform LC simulation

        Parameters
        ----------
        x1 : float
            SN stretch.
        color : float
            SN color.
        z : float
            SN redshift.
        sn_model : str
            sn model.
        sn_version : str
            sn model version.
        ebvofMW : float
            E(B-V).

        Returns
        -------
        None.

        """

        self.x1 = x1
        self.color = color
        self.z = np.round(z, 2)
        self.sn_type = sn_type
        self.sn_model = sn_model
        self.sn_version = sn_version
        self.ebvofMW = ebvofMW
        self.error_model = error_model

        # output infos
        self.fake_dir = '{}/fake_obs'.format(maindir)
        self.fake_name = 'Fakes_z_{}'.format(np.round(z, 2))
        self.lc_dir = '{}/fake_simu'.format(maindir)

        # generate fake obs
        self.gen_fakes()

        # simulate LCs from these fakes
        self.simu_lc()

    def gen_fakes(self):
        """
        Method to generate fake obs

        Returns
        -------
        None.

        """

        mjd_min = -21.*(1.+self.z)
        mjd_max = 63.*(1.+self.z)
        cad = 0.1*(1.+self.z)
        pars = {}
        pars['seasonLength'] = int(mjd_max-mjd_min)
        for b in 'grizy':
            pars['cadence_{}'.format(b)] = cad

        pars['MJDmin'] = mjd_min
        pars['saveData'] = 1
        pars['outDir'] = self.fake_dir
        pars['outName'] = self.fake_name
        pars['shiftDays'] = 0.0

        script = 'python run_scripts/fakes/make_fake.py'
        go(script, pars)

    def simu_lc(self):
        """
        Methode to simulate LCs

        Returns
        -------
        None.

        """
        pars = {}
        pars['SN_ebvofMW'] = 0.0
        pars['SN_z_type'] = 'unique'
        pars['SN_z_min'] = self.z
        pars['SN_daymax_type'] = 'unique'
        pars['SN_x1_type'] = 'unique'
        pars['SN_color_type'] = 'unique'
        pars['SN_x1_min'] = self.x1
        pars['SN_color_min'] = self.color
        pars['SN_NSNabsolute'] = 1
        pars['SN_differentialFlux'] = 1
        pars['fieldType'] = 'Fake'
        pars['dbName'] = self.fake_name
        pars['dbExtens'] = 'npy'
        pars['dbDir'] = self.fake_dir
        pars['MultiprocessingSimu_nproc'] = 1
        pars['OutputSimu_save'] = 1
        pars['OutputSimu_directory'] = self.lc_dir
        pars['nside'] = 128
        pars['Simulator_model'] = self.sn_model
        pars['Simulator_version'] = self.sn_version
        pars['Simulator_errorModel'] = self.error_model
        pars['Observations_coadd'] = 0
        cutoff = 'cutoff'

        if self.error_model:
            cutoff = self.error_model

        fname = '{}_{}'.format(self.sn_type, self.sn_model)
        if 'salt' in self.sn_model:
            fname = '{}_{}'.format(self.x1, self.color)

        fname = '{}_{}_{}_{}_ebvofMW_{}'.format(
            fname, cutoff, self.sn_model, self.sn_version, self.ebvofMW)
        prodid = 'Fake_{}_{}'.format(fname, self.z)

        pars['ProductionIDSimu'] = prodid

        script = 'python run_scripts/simulation/run_simulation.py'

        print('simulation here')
        go(script, pars)


parser = OptionParser()

parser.add_option(
    '--x1', help='SN x1 [%default]', default=-2.0, type=float)
parser.add_option(
    '--color', help='SN color [%default]', default=0.2, type=float)
parser.add_option(
    '--zmin', help='min redshift value  [%default]', default=0.01, type=float)
parser.add_option(
    '--zmax', help='max redshift value [%default]', default=1.1, type=float)
parser.add_option(
    '--zstep', help='redshift step value [%default]', default=0.01, type=float)
parser.add_option(
    '--sn_model', help='SN model [%default]', default='salt3', type=str)
parser.add_option(
    '--sn_version', help='SN model version [%default]', default='2.0',
    type=str)
parser.add_option(
    '--sn_type', help='SN type [%default]', default='SNIa',
    type=str)
parser.add_option(
    '--ebvofMW', help='ebvofMW[%default]', default=0.0, type=float)
parser.add_option(
    '--nproc', help='nproc for multiproc [%default]', default=8, type=int)
parser.add_option(
    '--error_model', help='error model flag [%default]', default=0, type=int)

opts, args = parser.parse_args()

"""
x1 = opts.x1
color = opts.color
zmin = opts.zmin
zmax = opts.zmax
zstep = opts.zstep
sn_model = opts.sn_model
sn_version = opts.sn_version
ebvofMW = opts.ebvofMW
nproc = opts.nproc
"""
opts_dict = vars(opts)

outDirLC = 'fakes_for_templates_{}_{}_ebvofMW_{}'.format(
    opts.x1, opts.color, opts.ebvofMW)

opts_dict['maindir'] = outDirLC
zvals = np.arange(opts.zmin, opts.zmax+opts.zstep, opts.zstep)

"""
opts_dict['z'] = 0.3
Simulation(**opts_dict)
"""

nproc = opts.nproc
multiproc(zvals, opts_dict, multi_simu, nproc)


outDirTemplates = 'Template_LC_{}_{}_ebvofMW_{}'.format(
    opts.x1, opts.color, opts.ebvofMW)

opts_dict['lcDir'] = '{}/fake_simu'.format(outDirLC)
opts_dict['outDir'] = outDirTemplates

stack_lc(**opts_dict)
