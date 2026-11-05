#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 09:27:07 2026

@author: philippe.gris@clermont.in2p3.fr
"""

import pandas as pd
from sn_plotter_analysis import plt


def plot(df,xvar=)

dataDir = '../auxtel_surveys'

files = ['auxtel_survey_baseline_v5.3.0_10yrs_star_calib_asynchronous_1.hdf5',
         'auxtel_survey_baseline_v5.3.0_10yrs_star_calib_synchronous_1.hdf5']
config=['async','sync']

dd = dict(zip(config,files))

df_dict = {}

for key,vals in dd.items():
    fName = '{}/{}'.format(dataDir,vals)
    df_dict[key] = pd.read_hdf(fName)
    
    
print(df_dict)