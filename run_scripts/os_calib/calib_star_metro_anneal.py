#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 09:35:24 2026

@author: philippe.gris@clermont.in2p3.fr
"""

from sn_analysis.sn_os_atmos_calib_tools import load_star_calib
from sn_analysis.sn_algo_metrop import Metrop_Anneal
import pandas as pd

fName = 'input/scheduler/calspec_gaia_matching.csv'

df = load_star_calib(fName)[:5]

idx = df['RA_deg'] >= 0. 
idx &= df['RA_deg'] <= 360.

df=df[idx]

df_fi = pd.DataFrame()
for i in range(50):
    mm = Metrop_Anneal(df)

    dd = mm()
    
    dd['config'] = 'config_{}'.format(i)
    
    df_fi = pd.concat((df_fi,dd))
    
print(df_fi)

print(dd)