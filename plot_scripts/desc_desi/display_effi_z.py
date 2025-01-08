#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 10:47:09 2024

@author: philippe.gris@clermont.in2p3.fr
"""
import pandas as pd
import matplotlib.pyplot as plt
import glob

plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['legend.facecolor'] = 'w'
plt.rcParams['figure.figsize'] = (11, 6)
plt.rcParams['figure.titlesize'] = 12
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['figure.titleweight'] = 'bold'
plt.rcParams['font.size'] = 20
plt.rcParams['font.weight'] = 'bold'

inputDir = 'input/cosmology/host_effi'

search_path = '{}/*v3*.csv'.format(inputDir)

fis = glob.glob(search_path)
fis += ['{}/host_effi_TiDES.csv'.format(inputDir)]

fig, ax = plt.subplots()
for fi in fis:
    df = pd.read_csv(fi, comment='#')
    fName = fi.split('/')[-1].split('.csv')[0].split('host_effi_')[-1]
    ls = 'solid'
    if 'desi2' in fName:
        ls = 'dashed'
    ax.plot(df['z'], df['effi'], label=fName, linestyle=ls)

ax.grid(visible='True')
ax.legend()
ax.set_xlabel(r'Redshift')
ax.set_ylabel(r'Efficiency')
plt.show()
