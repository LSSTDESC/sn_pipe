#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 14:26:22 2026

@author: philippe.gris@clermont.in2p3.fr
"""
import pandas as pd
from optparse import OptionParser
import matplotlib.pyplot as plt
from sn_analysis.sn_tools import fit_lin

def linfit_atmos(df,varxp='pwv',vary_prefix='zp',airmass=[1.2,2.0],bands='grizy'):
    
    ro = []
    print(df.columns)
    varx = 'sigma_{}'.format(varxp)
    for airm in airmass:
        idx = df['mean_airmass'] == airm
        sel = df[idx]
        for b in bands:
            yvar = 'std_{}_{}'.format(vary_prefix,b)
            print('fitting',df[[varx,yvar]])
            res = list(fit_lin(sel,varx,yvar))
            res += [b,airm]
            ro.append(res)
    dfn = pd.DataFrame(ro,columns=['slope','intercept','band','airmass'])
    dfn['atmos_param'] = varxp
    dfn['obs_param'] = vary_prefix
    dfn['obs_param_value'] = df['mean_{}'.format(varxp)].mean()
    
    return dfn
    
def plot(df,obs_param='zp',airmass=1.2):
    
    idx = df['obs_param'] == obs_param
    idx &= df['airmass'] == airmass
    
    sel = df[idx]
    
    atmos_params = sel['atmos_param'].unique()
    
    fig, ax = plt.subplots(figsize=(12,8))
    fig.suptitle('airmass={}'.format(airmass))
    for atm in atmos_params:
        idx = sel['atmos_param'] == atm
        selb = sel[idx]
        ax.plot(selb['band'],1./selb['slope'],label=atm)
        
       
    #add auxtel performance
    ax.grid(visible=True)
    ax.legend()
    ax.set_yscale("log")
    
    #add auxtel performance
    xmin, xmax = ax.get_xlim()
    yv_auxtel=[0.2,20,3e-3,5e-3]
    
    for yy in yv_auxtel:
        r = []
        for b in 'grizy':
            r.append((b,yy))
            
        dfaux = pd.DataFrame(r,columns=['band','auxres'])
        
        ax.plot(dfaux['band'],dfaux['auxres'],color='k',linestyle='dashed')
    
    
    
    
        
    
    
    

parser = OptionParser(description='Fit sigma_zp and sigma_mean_wave \
                      vs sigma of atmos params')
                      
parser.add_option('--dataDir', type=str, default='../zp_atmos',
                  help='data dir [%default]')
parser.add_option('--atmos_param', type=str, default='airmass,ozone,aerosol,pwv',
                  help='atmospheric parameters [%default]')

opts, args = parser.parse_args()

theDir = opts.dataDir
atmos_params= opts.atmos_param.split(',')

df_zp = pd.DataFrame()
df_wave = pd.DataFrame()
for atm in atmos_params:
    fName = '{}/zp_atmos_{}.hdf5'.format(theDir,atm)
    df = pd.read_hdf(fName)
    for b in 'grizy':
        df['std_zp_{}'.format(b)] *= 1000
    df= df.round({'mean_airmass':2})
    dfa = linfit_atmos(df,varxp=atm,
                       vary_prefix='zp',
                       airmass=[1.2,2.0],bands='grizy')
    dfb = linfit_atmos(df,varxp=atm,
                       vary_prefix='mean_wave',
                       airmass=[1.2,2.0],bands='grizy')
    df_zp = pd.concat((df_zp,dfa))
    df_wave = pd.concat((df_wave,dfb))
   
print(df_zp)

print(df_wave)
    
plot(df_zp)
plot(df_zp,airmass=2.0)

plt.show()
