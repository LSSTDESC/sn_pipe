#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 14:26:22 2026

@author: philippe.gris@clermont.in2p3.fr
"""
from optparse import OptionParser
import matplotlib.pyplot as plt
from sn_analysis.sn_fit_tools import load_fit_atmos_data
  
parser = OptionParser(description='Fit sigma_zp and sigma_mean_wave \
                      vs sigma of atmos params')
                      
parser.add_option('--dataDir', type=str, default='../zp_atmos',
                  help='data dir [%default]')
parser.add_option('--atmos_param', type=str, default='airmass,ozone,aerosol,pwv',
                  help='atmospheric parameters [%default]')
parser.add_option('--plots', type=str, default='vs_airmass,summary,from_sigmas',
                  help='plots [%default]')
parser.add_option('--sigmas', type=str, default='3e-3,20,5e-3,0.2',
                  help='sigmas of atmos params [%default]')
parser.add_option('--unit', type=str, default=',DU,,mm',
                  help='unit of sigmas of atmos params [%default]')

opts, args = parser.parse_args()

theDir = opts.dataDir
atmos_params= opts.atmos_param.split(',')
plots = opts.plots.split(',')
sigmas = opts.sigmas.split(',')
unit = opts.unit.split(',')
sigmas = list(map(float, sigmas))
sigmas = dict(zip(atmos_params,sigmas))
unit = dict(zip(atmos_params,unit))

df_zp, df_wave = load_fit_atmos_data(theDir, atmos_params)

if 'summary' in plots:
    from sn_plotter_tools.plot_atmos_tools import plot_all_summary
    plot_all_summary(df_zp, df_wave)
  
if 'vs_airmass' in plots:
    from sn_plotter_tools.plot_atmos_tools import plot_atmos_data_airmass
    plot_atmos_data_airmass(theDir,atmos_params)

if 'from_sigmas' in plots:
    from sn_plotter_tools.plot_atmos_tools import plot_perf_obs_param
    plot_perf_obs_param(df_zp,sigmas,unit,unit='mmag')
    plot_perf_obs_param(df_wave,sigmas,unit,obs_param='mean_wave',unit = 'nm',
                    ylabel='mean\ wave',ylines=[0.1],yannot=['0.1 nm'])

plt.show()
