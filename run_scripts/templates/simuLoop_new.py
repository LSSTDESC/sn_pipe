#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 17:12:38 2023

@author: philippe.gris@clermont.in2p3.fr
"""
import os
from optparse import OptionParser


class Simulation:
    def __init__(self, x1, color, z, sn_model, sn_version, ebvofMW, **kwargs):
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
        self.z = z
        self.sn_model = sn_model
        self.sn_version = sn_version
        self.ebvofMW = ebvofMW

        self.gen_fakes()

    def gen_fakes(self):

        mjd_min = -30.*(1.+self.z)
        mjd_max = 90.*(1.+self.z)
        cad = 0.1*(1.+self.z)
        par = {}
        par['seasonLength'] = int(mjd_max-mjd_min)
        for b in 'grizy':
            par['cadence_{}'.format(b)] = cad

        par['saveData'] = 1
        cmd = 'python run_scripts/fakes/make_fake.py'
        for key, vals in par.items():
            cmd += ' --{} {}'.format(key, vals)

        print(cmd)
        os.system(cmd)


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
    '--sn_version', help='SN model version [%default]', default='1.0',
    type=str)
parser.add_option(
    '--ebvofMW', help='ebvofMW[%default]', default=0.0, type=float)
parser.add_option(
    '--nproc', help='nproc for multiproc [%default]', default=8, type=int)

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

opts_dict['z'] = 0.3

Simulation(**opts_dict)
