#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 15:48:05 2024

@author: philippe.gris@clermont.in2p3.fr
"""

import pandas as pd
import glob


def load(fName):

    df = pd.read_hdf(fName)
    survey = fName.split('/')[-1].split('.hdf5')[0].split('cosmo_fit_')[-1]
    df['survey'] = survey
    return df


theDir = '../cosmo_fit_desc_desi'

fis = glob.glob('{}/*.hdf5'.format(theDir))

df = pd.DataFrame()
for fi in fis:
    dfa = load(fi)
    df = pd.concat((df, dfa))

print(df['survey'].unique())
