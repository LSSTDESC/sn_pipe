#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 10:12:26 2026

@author: philippe.gris@clermont.in2p3.fr
"""
import pandas as pd
from astropy.table import Table
from optparse import OptionParser
from sn_plotter_tools.plot_tools import plot_grid,plot_airmass
import matplotlib.pyplot as plt

    
parser = OptionParser(description='analyze and plot zp and mean wave')

parser.add_option('--dataDir', type=str, default='../zp_atmos',
                  help='data dir [%default]')
parser.add_option('--atmos_param', type=str, default='ozone',
                  help='atmospheric parameter[%default]')
parser.add_option('--band', type=str, default='y',
                  help='band to plot [%default]')
parser.add_option('--plots', type=str, default='map,sigma',
                  help='what to plot [%default]')

opts, args = parser.parse_args()

theDir = opts.dataDir
atmos_param= opts.atmos_param
theFile = 'zp_atmos_{}.hdf5'.format(atmos_param)
theband = opts.band
plots = opts.plots.split(',')

fName = '{}/{}'.format(theDir,theFile)

df = pd.read_hdf(fName)

idx = df['sigma_aerosol'] <= 0.0125
df = df[idx]
print(df.columns)

print(df[['mean_airmass','sigma_pwv','mean_mean_wave_z', 'std_mean_wave_z']])

for b in 'grizy':
    df['std_zp_{}'.format(b)] *= 1000 # in mmag
    
#grid plots

varx = ['sigma_pwv','sigma_aerosol','sigma_airmass','sigma_ozone']
legx = ['$\sigma_{PWV}$ [mm]',
        '$\sigma_{aerosol}$',
        '$\sigma_{airmass}$',
        '$\sigma_{ozone}$ [DU]']

xxtext = [0.15,0.011,0.01,25]

xtext = dict(zip(varx,xxtext))

labdict = dict(zip(varx,legx))

thevar = 'sigma_{}'.format(atmos_param)

tab = Table.from_pandas(df,index=False)

if 'map' in plots:
    plot_grid(tab,varx='mean_airmass',
              vary=thevar,ylabel=labdict[thevar],
              varz='std_zp_{}'.format(theband),
              figtitle='$\sigma_{ZP}^{'+theband+'}$ [mag]',smoothIt=False)
    plot_grid(tab,varx='mean_airmass',
              vary=thevar,ylabel=labdict[thevar],
              varz='std_mean_wave_{}'.format(theband),
              figtitle='$\sigma_{mean wave}^{'+theband+'}$ [nm]',
              iso=[0.05,0.1,0.15],
              txt_iso=['0.05 nm','0.1 nm','0.15 nm'],
              x_iso=[1.5]*3,smoothIt=True)
if 'sigma' in plots:
    airmass=[1.2,2.0]
    plot_airmass(df,varx=thevar,xlabel=labdict[thevar],
                 vary_prefix='std_zp',airmass=airmass, 
                 y_iso=[1,2,3,5],
                 txt_iso=['1 mmag','2 mmag','3 mmag','5 mmag'],
                 xtext=xtext[thevar],smoothIt=False,fitIt=True)
    plot_airmass(df,varx=thevar,xlabel=labdict[thevar],
                     vary_prefix='std_mean_wave',
                     ylabel='$\sigma_{meanwave}$ [mm]',
                     airmass=airmass,
                     y_iso=[0.05,0.1,0.15],
                     txt_iso=['0.05 nm','0.1 nm','0.15 nm'],
                     ymax=0.2,deltay_txt=0.005,
                     xtext=xtext[thevar],smoothIt=True)
    
plt.show()





