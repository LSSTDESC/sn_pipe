#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 14:12:04 2026

@author: philippe.gris@clermont.in2p3.fr
"""

from optparse import OptionParser
from sn_analysis.sn_fit_tools import load_fit_atmos_data
from sn_analysis.sn_atmos_tools import process_obs_data 
from sn_analysis.sn_atmos_tools import merge_zp_wave
import pandas as pd
    
    
parser = OptionParser(description='Scan the atmos parameter sigma space')
                      
parser.add_option('--dataDir', type=str, default='../zp_atmos',
                  help='data dir [%default]')
parser.add_option('--atmos_param', type=str, default='airmass,ozone,aerosol,pwv',
                  help='atmospheric parameters [%default]')
parser.add_option('--config', type=str, default='config_atmos.csv',
                  help='sigma atmos parameters [%default]')

opts, args = parser.parse_args()

theDir = opts.dataDir
atmos_params= opts.atmos_param.split(',')
config = opts.config

#get interpolated values
df_zp, df_wave = load_fit_atmos_data(theDir, atmos_params)

#load atmos sigmas
df_atm = pd.read_csv(config,comment='#')

print(df_atm)
sigma = df_atm.to_dict(orient='list')

print(sigma)

combi_zp = process_obs_data(df_zp,sigma,atmos_params,do_combi=False)
combi_wave = process_obs_data(df_wave,sigma,atmos_params,do_combi=False)

combi_tot = merge_zp_wave(combi_zp,combi_wave,atmos_params)

print(combi_tot)