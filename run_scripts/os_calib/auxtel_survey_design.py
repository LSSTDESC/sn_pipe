#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 09:59:43 2026

@author: philippe.gris@clermont.in2p3.fr
"""
from optparse import OptionParser
from sn_tools.sn_io import make_dict_from_config
from sn_tools.sn_io import add_parser
from sn_analysis.sn_os_atmos_calib_tools import load_OS,ddf_wfd
import numpy as np
import matplotlib.pyplot as plt
from sn_tools.os_auxtel_visu import AuxTelSurvey

parser = OptionParser(description='auxtel survey design')

confDict = make_dict_from_config('input_script', 'auxtel_survey_design.txt')

# parser for simulation parameters : 'dynamical' generation
add_parser(parser, confDict)

opts, args = parser.parse_args()

params=vars(opts)

#load obs

obs = load_OS(params['dbDir'], params['dbName'], params['dbExtens'])

nside=32
ddf,wfd = ddf_wfd(obs.to_records(index=False),nside=nside)

#concatenate

obs = np.concat((ddf,wfd))

obs.sort(order='mjd')

print(obs.dtype)

survey = AuxTelSurvey(obs,nside=nside)

varx = 'RA'
vary='Dec'
xmin, xmax = np.min(obs[varx]),np.max(obs[varx])
ymin, ymax = np.min(obs[vary]),np.max(obs[vary])

for i in range(len(obs)):
    fig, ax = plt.subplots()
    
    ax.plot(obs[:i][varx],obs[:i][vary],'ko')
    
    ax.plot(obs[i][varx],obs[i][vary],'ro')
    
    ax.set_xlim([xmin,xmax])
    ax.set_ylim([ymin,ymax])
    
    if i > 0:
        ra_a = obs[i]['RA']
        time_a = obs[i]['mjd']
        ra_b = obs[i-1]['RA']
        dec_a = obs[i]['Dec']
        dec_b = obs[i-1]['Dec']
        time_b = obs[i-1]['mjd']
        visit_b = obs[i-1]['visitTime']
        slew_b = obs[i-1]['slewTime']
        slew_a = obs[i]['slewTime']
        dist = ((ra_a-ra_b)*np.cos(np.deg2rad(dec_a)))**2
        dist =+ (dec_a-dec_b)**2
        dist = np.sqrt(dist)
        deltaT = 24*3600*(time_a-time_b)-visit_b
        
        print(deltaT,dist,dist/deltaT,slew_a,slew_b)
    
    
    plt.show()
    
