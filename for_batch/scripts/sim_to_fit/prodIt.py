#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 15:56:31 2025

@author: philippe.gris@clermont.in2p3.fr
"""
from optparse import OptionParser
import os

parser = OptionParser(
    description='Script to produce SN for WFD and DDF surveys')

parser.add_option("--runType", type="str", default='DDF',
                  help="type of run: DDF, WFD, DDF+WFD [%default]")
parser.add_option("--dbList_DD", type="str", default='DD_fbs_5.0.0.csv',
                  help="dbList DD to process  [%default]")
parser.add_option("--dbList_WFD", type="str", default='WFD_fbs_5.0.0.csv',
                  help="dbList WFD to process  [%default]")
parser.add_option("--outDir_main", type="str",
                  default='/sps/lsst/groups/cadence/LSST_SN_PhG/prod_simu',
                  help="Main output dir [%default]")
parser.add_option("--outDir_DD", type="str",
                  default='Output_SN_DD_sigmaInt_0.0_Hounsell_z_smflux_notelrot_airmass',
                  help="output dir for DDF [%default]")
parser.add_option("--outDir_WFD", type="str",
                  default='Output_SN_WFD_sigmaInt_0.0_Hounsell_z_smflux_notelrot_airmass',
                  help="output dir for WFD [%default]")
parser.add_option("--SN_smearFlux", type=int,
                  default=1,
                  help="SN flux smearing [%default]")
parser.add_option("--Fitter_sigmaz", type=float,
                  default=1.e-05,
                  help="sigma_z for the fitter [%default]")
parser.add_option("--Observations_coadd", type=int,
                  default=0,
                  help="coadd observations [%default]")
parser.add_option("--LC_coadd", type=int,
                  default=1,
                  help="coadd lc points [%default]")
parser.add_option("--saturation_effect", type=int,
                  default=0,
                  help="to include saturation effects [%default]")
parser.add_option("--sigma_airmass", type=float,
                  default=0.0,
                  help="sigma airmass [%default]")
parser.add_option("--sigma_pwv", type=float,
                  default=0.0,
                  help="sigma pwv [%default]")
parser.add_option("--sigma_ozone", type=float,
                  default=0.0,
                  help="sigma ozone [%default]")
parser.add_option("--sigma_aerosol", type=float,
                  default=0.0,
                  help="sigma aerosol [%default]")
parser.add_option("--InstrumentSimu_ntrial_zp", type=int,
                  default=1,
                  help="ntrials to estimate zp values [%default]")
parser.add_option("--tag_script", type=str,
                  default='no_tag',
                  help="script name tag [%default]")
parser.add_option("--mem", type=str, default='8Gb',
                  help="memory for batch jobs [%default]")
parser.add_option("--FoV", type=float, default=9.6,
                  help="telescope field of view [%default]")
parser.add_option('--Cosmology_deparams', type=str,
                  default='w0,wa',
                  help='DE eos parameters [%default]')
parser.add_option('--Cosmology_devalues', type=str,
                  default='-1.,0.',
                  help='DE eos parameter values [%default]')
parser.add_option('--Cosmology_declass', type=str,
                  default='w0waCDM',
                  help='DE class to use (w0waCDM/DDE_FLRW) [%default]')
parser.add_option('--Cosmology_classloc', type=str,
                  default='astropy.cosmology',
                  help='DE class location \
                        (astropy.cosmology/sn_tools.sn_cosmo_model) [%default]')
parser.add_option('--Cosmology_demodel', type=str,
                  default='CPL',
                  help='DE eos model name [%default]')
parser.add_option('--Cosmology_deeos', type=str,
                  default='w0+wa*z/(1+z)',
                  help='DE eos model [%default]')
parser.add_option('--Cosmology_H0', type=float,
                  default=70.,
                  help='DE eos model [%default]')
parser.add_option('--Cosmology_Om0', type=float,
                  default=0.30,
                  help='Omega_matter [%default]')
parser.add_option('--Cosmology_Ode0', type=float,
                  default=0.70,
                  help='Omega_DE [%default]')
parser.add_option("--SN_z_max", type=float,
                  default=1.1,
                  help="zmax for sn prod [%default]")
parser.add_option("--SN_z_min", type=float,
                  default=0.0,
                  help="zmin for sn prod [%default]")
parser.add_option("--SN_z_nbins", type=int,
                  default=10,
                  help="number of z bins for simulation [%default]")

opts, args = parser.parse_args()

params = vars(opts)

dbList_DD = ''
dbList_WFD = ''
outDir_DD = ''
outDir_WFD = ''

if 'DDF' in params['runType']:
    outDir_DD = '{}/{}'.format(params['outDir_main'], params['outDir_DD'])
    dbList_DD = params['dbList_DD']

if 'WFD' in params['runType']:
    outDir_WFD = '{}/{}'.format(params['outDir_main'],
                                str(params['outDir_WFD']))
    dbList_WFD = params['dbList_WFD']


script = 'python for_batch/scripts/sn_prod/prod_sn_dd_wfd.py'

cct = ['SN_smearFlux', 'Fitter_sigmaz',
       'Observations_coadd', 'LC_coadd', 'saturation_effect',
       'InstrumentSimu_ntrial_zp', 'tag_script', 'mem','FoV',
       'SN_z_max','SN_z_min','SN_z_nbins']
cosmo_params=['Cosmology_deparams', 'Cosmology_devalues', 'Cosmology_declass', 
       'Cosmology_classloc','Cosmology_demodel', 'Cosmology_deeos', 
       'Cosmology_H0', 'Cosmology_Om0', 'Cosmology_Ode0']

cct += cosmo_params

for vv in ['airmass', 'pwv', 'ozone', 'aerosol']:
    cct.append('sigma_{}'.format(vv))

scr_ = script
scr_ += ' --outDir_DD={}'.format(outDir_DD)
scr_ += ' --dbList_DD={}'.format(dbList_DD)
scr_ += ' --outDir_WFD={}'.format(outDir_WFD)
scr_ += ' --dbList_WFD={}'.format(dbList_WFD)

for vv in cct:
    if isinstance(params[vv],str) and "(" in params[vv]:
        scr_ += ' --{}=\'{}\''.format(vv, params[vv])
    else:
        scr_ += ' --{}={}'.format(vv, params[vv])
print(scr_)
os.system(scr_)
