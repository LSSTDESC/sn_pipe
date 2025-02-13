#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 16:42:18 2025

@author: philippe.gris@clermont.in2p3.fr
"""
from optparse import OptionParser
import pandas as pd
import matplotlib.pyplot as plt

parser = OptionParser(description='Script to plot pixel level OS infos')

parser.add_option('--dbName', type=str, default='test_newb',
                  help='dbName to process [%default]')
parser.add_option('--dbDir', type=str, default='../test_metric',
                  help='dbDir of the OS to process [%default]')

opts, args = parser.parse_args()

dbDir = opts.dbDir
dbName = opts.dbName

fName = '{}/{}.hdf5'.format(dbDir, dbName)

df = pd.read_hdf(fName)

print(df.columns)

fig, ax = plt.subplots(figsize=(12, 8))

idx = df['season'] > 0
idx = df['season'] < 11
sel = df[idx]

ax.plot(df['season'], df['cadence'], 'ko')

plt.show()
