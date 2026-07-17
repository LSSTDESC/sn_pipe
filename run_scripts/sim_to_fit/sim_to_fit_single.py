#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 14:32:41 2026

@author: philippe.gris@clermont.in2p3.fr
"""

from optparse import OptionParser
import os
from sn_tools.sn_io import checkDir

parser = OptionParser(description='Script to simulate light curves+fit')

parser.add_option('--dbName', type=str, default='baseline_v5.3.0_10yrs',
                  help='dbName to process [%default]')
parser.add_option('--dbExtens', type=str, default='npy',
                  help='db extens [%default]')
parser.add_option('--dbDir', type=str, default='../DB_Files',
                  help='dbDir of the OS to process [%default]')
parser.add_option('--x1_type', type=str, default='unique',
                  help='x1 type for simulation [%default]')
parser.add_option('--x1', type=float, default=-2.0,
                  help='x1 value [%default]')
parser.add_option('--color_type', type=str, default='unique',
                  help='color type for simulation [%default]')
parser.add_option('--color', type=float, default=0.2,
                  help='color value [%default]')
parser.add_option('--z_type', type=str, default='uniform',
                  help='z type for simulation [%default]')
parser.add_option('--zmin', type=float, default=0.01,
                  help='z min value [%default]')
parser.add_option('--zmax', type=float, default=0.1,
                  help='z max value [%default]')
parser.add_option('--zstep', type=float, default=0.05,
                  help='z step value [%default]')
parser.add_option('--daymax_type', type=str, default='random',
                  help='daymax type for simulation [%default]')
parser.add_option('--daymax_step', type=float, default=5,
                  help='daymax step for simulation [%default]')
parser.add_option('--nsn_abs', type=int, default=1,
                  help='absolute number for nsn simu [%default]')
parser.add_option('--nsn_factor', type=int, default=30,
                  help='factor for nsn simu from rate [%default]')
parser.add_option('--nside', type=int, default=128,
                  help='healpix nside parameter [%default]')
parser.add_option('--fieldType', type=str, default='DD',
                  help='type of field to simulate (DD or WFD) [%default]')
parser.add_option('--runType', type=str, default='DDF_spectroz',
                  help='type of run [%default]')
parser.add_option('--outDir', type=str, default='../test_fmb',
                  help='main out dir [%default]')
parser.add_option('--fieldName', type=str, default='COSMOS',
                  help='field to process [%default]')
parser.add_option('--seasons', type=str, default='1-11',
                  help='seasons to process [%default]')
parser.add_option('--pixelList', type=str, default='pixelList.csv',
                  help='pixelList (None if NA) [%default]')
parser.add_option('--save_LC', type=int, default=0,
                  help='to save LC on disk [%default]')
parser.add_option('--sigma_airmass', type=float, default=0.0,
                  help='sigma airmass [%default]')
parser.add_option('--sigma_ozone', type=float, default=0.0,
                  help='sigma ozone [%default]')
parser.add_option('--sigma_aerosol', type=float, default=0.0,
                  help='sigma aerosol [%default]')
parser.add_option('--sigma_pwv', type=float, default=0.0,
                  help='sigma pwv [%default]')
parser.add_option('--smear_flux', type=int, default=1,
                  help='flux smearing [%default]')
parser.add_option('--SN_simuFile', type=str, default='None',
                  help='to insert a simu file [%default]')
parser.add_option('--obs_coadd',type=int,default=0,
                  help = 'to coadd obs [%default]')
parser.add_option('--lc_coadd',type=int,default=1,
                  help = 'to coadd LCs [%default]')
parser.add_option('--fit_lc',type=int,default=1,
                  help = 'to fit LCs or not [%default]')

opts, args = parser.parse_args()

pp = vars(opts)

ntrial_zp = 1
atmos_cols = ['airmass','ozone','aerosol','pwv']

for vv in atmos_cols:
    if pp['sigma_{}'.format(vv)] >= 1.e-5:
        ntrial_zp = 1000
        
cmd = "python run_scripts/sim_to_fit/run_sim_to_fit.py" 
cmd += " --dbName={}".format(pp['dbName'])
cmd += " --dbDir={}".format(pp['dbDir'])
cmd += " --dbExtens={}".format(pp['dbExtens'])
cmd += " --OutputSimu_save={}".format(pp['save_LC'])
cmd += " --OutputSimu_throwafterdump=0"
#SN parameters
cmd += " --SN_x1_type={}".format(pp['x1_type'])
cmd += " --SN_x1_min={}".format(pp['x1'])
cmd += " --SN_color_type={}".format(pp['color_type'])
cmd += " --SN_color_min={}".format(pp['color'])
cmd += " --SN_z_type={}".format(pp['z_type'])
cmd += " --SN_z_min={}".format(pp['zmin'])
cmd += " --SN_z_max={}".format(pp['zmax'])
cmd += " --SN_z_step={}".format(pp['zstep'])
cmd += " --SN_daymax_type={}".format(pp['daymax_type'])
cmd += " --SN_daymax_step={}".format(pp['daymax_step'])
cmd += " --SN_daymax_restricted=1"
cmd += " --SN_z_nbins=10"
cmd += " --SN_sigmaInt=0.0"
cmd += " --SN_z_rate=Hounsell"
cmd += " --SN_NSNabsolute={}".format(pp['nsn_abs'])
cmd += " --SN_NSNfactor={}".format(pp['nsn_factor'])
#simulator
cmd += " --Simulator_model=salt3"
cmd += " --Simulator_version=2.0"
cmd += " --MultiprocessingSimu_nproc=8"
#fitter
cmd += " --Fitter_model=salt3"
cmd += " --Fitter_version=2.0"
cmd += " --MultiprocessingFit_nproc=8"
cmd += " --Fitter_parnames=t0,x1,c,x0"
cmd += " --fit_selected=0"
#pixel processing
cmd += " --nproc_pixels=8"
cmd += " --nproc=1"
cmd += " --ebvofMW_pixel=0.0"
outDir = "{}/{}/{}".format(pp['outDir'],pp['dbName'],pp['runType'])
#create output dir if necessary'
checkDir(outDir)
cmd += " --OutputSimu_directory={}".format(outDir)
cmd += " --OutputFit_directory={}".format(outDir)
cmd += " --fieldType={}".format(pp['fieldType'])
cmd += " --nside={}".format(pp['nside'])
cmd += " --Pixelisation_nside={}".format(pp['nside'])
cmd += " --fieldName={}".format(pp['fieldName'])
cmd += " --Observations_season={}".format(pp['seasons'])
cmd += " --Observations_coadd={}".format(pp['obs_coadd'])

#miscellaneous parameters
#cmd += " --Observations_coadd=0"
cmd += " --LC_coadd={}".format(pp['lc_coadd'])
cmd += " --InstrumentSimu_telescope_tag=1.9"
cmd += " --InstrumentFit_telescope_tag=1.9"
cmd += " --Fitter_sigmaz=1e-05"
cmd += " --SN_smearFlux={}".format(pp['smear_flux'])
cmd += " --lookup_ddf=input/simulation/lookup_ddf.csv"
cmd += " --saturation_effect=0"
cmd += " --saturation_psf=single_gauss"
cmd += " --saturation_ccdfullwell=120000.0"
cmd += " --SN_z_sigmaz=1e-05"
cmd += " --fit_remove_sat=0"
cmd += " --LCSelection_snrmin=1.0"
cmd += " --code=new"
cmd += " --SN_minRFphaseQual=-10.0"
cmd += " --SN_maxRFphaseQual=35.0"
cmd += " --InstrumentSimu_atmosType=const"
cmd += " --fit_coadded=0"
for vv in atmos_cols:
    cmd += " --InstrumentSimu_sigma_{}={}".format(vv,pp['sigma_{}'.format(vv)])
    
cmd += " --InstrumentSimu_ntrial_zp={}".format(ntrial_zp)
cmd += " --FoV=9.6"
cmd += " --Cosmology_Om0=0.3"
cmd += " --Cosmology_Ode0=0.7"
cmd += " --Cosmology_H0=70.0"
cmd += " --Cosmology_deparams=w0,wa"
cmd += " --Cosmology_devalues=-1.,0."
cmd += " --Cosmology_declass=w0waCDM"
cmd += " --Cosmology_classloc=astropy.cosmology"
cmd += " --Cosmology_demodel=CPL"
cmd += " --Cosmology_deeos='w0+wa*z/(1+z)'"

cmd += " --fit_lc={}".format(pp['fit_lc'])

seasons = pp['seasons']
if '-' in seasons:
    seasons=seasons.replace('-','_')
if ',' in seasons:
    seasons=seasons.replace(',','and')
    
prodID = 'SN_{}_{}_{}_{}_{}'.format(pp['fieldType'],pp['dbName'],
                                    pp['x1'],pp['color'],seasons)

cmd += " --ProductionIDSimu={}".format(prodID)

if pp['pixelList'] != "None":
    cmd += " --pixelList={}".format(pp['pixelList'])

cmd += ' --SN_simuFile={}'.format(pp['SN_simuFile'])

print(cmd)
os.system(cmd)