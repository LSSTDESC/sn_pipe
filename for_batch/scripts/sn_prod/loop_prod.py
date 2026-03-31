#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 16:36:13 2023

@author: philippe.gris@clermont.in2p3.fr
"""
import pandas as pd
from optparse import OptionParser
import os


def go_batch(script, params):

    cmd = 'python {}'.format(script)
    for key, vals in params.items():
        cmd += ' --{} {}'.format(key, vals)

    print(cmd)
    os.system(cmd)


parser = OptionParser()

parser.add_option("--dbList", type="str", default='DD_fbs_2.99.csv',
                  help="db list to process[%default]")
parser.add_option("--outputDir", type="str",
                  default='/sps/lsst/users/gris/Output_SN',
                  help="main output dir [%default]")
parser.add_option("--Fitter_parnames", type="str", default='t0,x1,c,x0',
                  help="parameters to fit [%default]")
parser.add_option("--reprocList", type="str", default='None.csv',
                  help="to reproc some of the db files only [%default]")
parser.add_option("--SN_sigmaInt", type=float, default=0.0,
                  help="SN intrinsic dispersion [%default]")
parser.add_option("--SN_z_rate", type=str, default='Hounsell',
                  help="SN production rate [%default]")
parser.add_option("--SN_NSNfactor", type=int, default=30,
                  help="SN production scale factor [%default]")
parser.add_option("--SN_NSNabsolute", type=int,
                  default=-1,
                  help="absolute nsn for production[%default]")
parser.add_option("--Observations_coadd", type=int, default=1,
                  help="Coadd obs per night [%default]")
parser.add_option("--LC_coadd", type=int,
                  default=1,
                  help="coadd lc points [%default]")
parser.add_option("--SN_smearFlux", type=int, default=1,
                  help="LC flux smearing [%default]")
parser.add_option("--Fitter_sigmaz", type=float, default=1.e-5,
                  help="sigmaz for LC fits [%default]")
parser.add_option("--simuParams_fromFile", type=int, default=0,
                  help="to use simulation params from file [%default]")
parser.add_option("--simuParams_dir", type=str,
                  default='/sps/lsst/users/gris/Output_SN_WFD_sigmaInt_0.0_Hounsell_nnew',
                  help="Dir for the simu param files [%default]")
parser.add_option("--DD_list", type=str,
                  default='COSMOS,CDFS,EDFS,ELAISS1,XMM-LSS',
                  help="List of DDFs to process [%default]")
parser.add_option("--lookup_ddf", type=str,
                  default='input/simulation/lookup_ddf.csv',
                  help="Look up table for DDFs [%default]")
parser.add_option("--saturation_effect", type=int,
                  default=0,
                  help="to include saturation effects [%default]")
parser.add_option("--saturation_psf", type=str,
                  default='single_gauss',
                  help="psf to use for saturation effects [%default]")
parser.add_option("--saturation_ccdfullwell", type=float,
                  default=120000,
                  help="ccd full well to use for saturation effects [%default]")
parser.add_option("--SN_z_max", type=float,
                  default=1.1,
                  help="zmax for sn prod [%default]")
parser.add_option("--SN_z_min", type=float,
                  default=0.01,
                  help="zmin for sn prod [%default]")
parser.add_option("--SN_z_step", type=float,
                  default=0.1,
                  help="zstep for sn prod [%default]")
parser.add_option("--SN_z_type", type=str,
                  default='random',
                  help="z type for sn prod [%default]")
parser.add_option("--SN_sigmaz", type=float, default=1.e-5,
                  help="sigmaz for LC zsim [%default]")
parser.add_option("--fit_remove_sat", type=str,
                  default='0',
                  help="to remove LC saturated points [%default]")
parser.add_option("--code", type=str,
                  default='new',
                  help="code to use (new/old) [%default]")
parser.add_option("--SN_minRFphaseQual", type=float,
                  default=-10.,
                  help="min RF phase quality [%default]")
parser.add_option("--SN_maxRFphaseQual", type=float, default=35.,
                  help="max RF phase quality [%default]")
parser.add_option("--SN_x1_type", type=str, default='random',
                  help="x1 simulation type [%default]")
parser.add_option("--SN_x1_min", type=float, default=-2.0,
                  help="x1 simulation min value [%default]")
parser.add_option("--SN_color_type", type=str, default='random',
                  help="color simulation type [%default]")
parser.add_option("--SN_color_min", type=float, default=0.2,
                  help="color min value [%default]")
parser.add_option("--mem", type=str, default='8Gb',
                  help="memore for batch jobs [%default]")
parser.add_option("--atmosType", type=str, default='const',
                  help="atmos type (const/dep) [%default]")
parser.add_option("--fit_coadded", type=int, default=0,
                  help="to fitted coadded LC points [%default]")
parser.add_option("--w0", type=float, default=-1.0,
                  help="w0 dark energy parameter [%default]")
parser.add_option("--wa", type=float, default=-1.0,
                  help="wa dark energy parameter [%default]")
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
                  default=0.30,
                  help='Omega_DE [%default]')

opts, args = parser.parse_args()

params = vars(opts)

cmd_scr = 'python for_batch/scripts/sn_prod/loop_prod.py'
cmd_scr += ' --SN_sigmaInt=0.0'
search_key = 'Cosmology'
cosmo_dict = dict(filter(lambda item: search_key in item[0], params.items()))

cosmo_list = list(cosmo_dict.keys())
# load csv file

df = pd.read_csv(opts.dbList, comment='#')

print(df)

procDict = {}
script = 'for_batch/scripts/sn_prod/sn_prod.py'

for i, row in df.iterrows():
    procDict['dbDir'] = row['dbDir']
    procDict['dbExtens'] = row['dbExtens']
    procDict['dbName'] = row['dbName']
    procDict['fieldType'] = row['fieldType']
    procDict['nside'] = row['nside']
    procDict['OutputSimu_directory'] = opts.outputDir
    procDict['OutputFit_directory'] = opts.outputDir
    procDict['Fitter_parnames'] = opts.Fitter_parnames
    procDict['reprocList'] = opts.reprocList
    procDict['SN_sigmaInt'] = opts.SN_sigmaInt
    procDict['SN_z_rate'] = opts.SN_z_rate
    procDict['SN_NSNfactor'] = opts.SN_NSNfactor
    procDict['SN_NSNabsolute'] = opts.SN_NSNabsolute
    procDict['Observations_coadd'] = opts.Observations_coadd
    procDict['LC_coadd'] = opts.LC_coadd
    procDict['InstrumentSimu_telescope_tag'] = row['teltag']
    procDict['InstrumentFit_telescope_tag'] = row['teltag']
    procDict['SN_smearFlux'] = opts.SN_smearFlux
    procDict['Fitter_sigmaz'] = opts.Fitter_sigmaz
    procDict['simuParams_fromFile'] = opts.simuParams_fromFile
    procDict['simuParams_dir'] = opts.simuParams_dir
    procDict['DD_list'] = opts.DD_list
    procDict['lookup_ddf'] = opts.lookup_ddf
    procDict['saturation_effect'] = opts.saturation_effect
    procDict['saturation_psf'] = opts.saturation_psf
    procDict['saturation_ccdfullwell'] = opts.saturation_ccdfullwell
    procDict['SN_z_max'] = opts.SN_z_max
    procDict['SN_z_min'] = opts.SN_z_min
    procDict['SN_z_step'] = opts.SN_z_step
    procDict['SN_z_type'] = opts.SN_z_type
    procDict['SN_z_sigmaz'] = opts.SN_sigmaz
    procDict['fit_remove_sat'] = opts.fit_remove_sat
    # procDict['InstrumentSimu_airmassType'] = 'const'
    # procDict['InstrumentFit_airmassType'] = 'const'
    procDict['LCSelection_snrmin'] = 1.
    procDict['code'] = opts.code
    procDict['SN_minRFphaseQual'] = opts.SN_minRFphaseQual
    procDict['SN_maxRFphaseQual'] = opts.SN_maxRFphaseQual
    procDict['SN_x1_type'] = opts.SN_x1_type
    procDict['SN_x1_min'] = opts.SN_x1_min
    procDict['SN_color_type'] = opts.SN_color_type
    procDict['SN_color_min'] = opts.SN_color_min
    procDict['mem'] = opts.mem
    procDict['InstrumentSimu_atmosType'] = opts.atmosType
    procDict['fit_coadded'] = opts.fit_coadded
    procDict['Cosmology_w0'] = opts.w0
    procDict['Cosmology_wa'] = opts.wa
    procDict['InstrumentSimu_sigma_airmass'] = opts.sigma_airmass
    procDict['InstrumentSimu_sigma_pwv'] = opts.sigma_pwv
    procDict['InstrumentSimu_sigma_aerosol'] = opts.sigma_aerosol
    procDict['InstrumentSimu_sigma_ozone'] = opts.sigma_ozone
    procDict['InstrumentSimu_ntrial_zp'] = opts.InstrumentSimu_ntrial_zp
    procDict['tag_script'] = opts.tag_script
    procDict['FoV'] = opts.FoV
    for vv in cosmo_list: 
        if isinstance(params[vv],str):
            procDict[vv]= '\'{}\''.format(params[vv])
        else:
            procDict[vv]= params[vv]


    go_batch(script, procDict)
