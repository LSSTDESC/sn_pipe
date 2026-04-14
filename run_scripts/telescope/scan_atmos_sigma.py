#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 08:52:23 2026

@author: philippe.gris@clarmont.in2p3.fr
"""
from optparse import OptionParser
from sn_analysis.sn_fit_tools import load_fit_atmos_data
from sn_analysis.sn_atmos_tools import get_atmos_data  
import numpy as np
import pandas as pd

def get_values(grp,sigma):
    
    print(grp.name)
    atmos_param = grp.name[2]
    sigmas = sigma[atmos_param]
    
    interp = grp['interp'].values[0]
    
    res = interp(sigmas)
    
    df = pd.DataFrame(res,columns=['sigma_obs_param'])
    
    df['sigma_atmos_param'] = sigmas
    
    return df

def make_combi(grp,atmos_params=['airmass','ozone','aerosol','pwv']):
    
    print('ooooo',grp)
    cols = ['sigma_atmos_param','sigma_obs_param']
    df_combi = pd.DataFrame()
    for atm in atmos_params:
        idx = grp['atmos_param'] == atm
        sel = pd.DataFrame(grp[idx][cols])
        print('booo',sel)
        sigma_atm = 'sigma_{}'.format(atm)
        sigma_obs = 'sigma_obs_param_{}'.format(atm)
        sel = sel.rename(columns={'sigma_atmos_param':sigma_atm,
                                  'sigma_obs_param':sigma_obs})
        if df_combi.empty:
            df_combi = pd.DataFrame(sel)
        else:
            df_combi = df_combi.merge(sel, how='cross')
            
    print(df_combi)
        
        
    return df_combi

def process_data(df_zp):
    
    interp_zp = get_atmos_data(df_zp,atmos_params=atmos_params)
    cols = ['band', 'airmass', 'atmos_param', 'obs_param']
    df_values = interp_zp.groupby(cols).apply(lambda x: get_values(x,sigma),include_groups=False).reset_index()

    print(df_values)

    combis = df_values.groupby(['band','airmass','obs_param']).apply(lambda x:make_combi(x),include_groups=False).reset_index()

    print(combis)

    #estimate sigma_tot

    combis['sigma_tot'] = 0

    for atm in atmos_params:
        combis['sigma_tot'] += combis['sigma_obs_param_{}'.format(atm)]**2
    
    combis['sigma_tot'] = np.sqrt(combis['sigma_tot'])

    print(combis)

    return combis

def rename(dfa):
    
    df = pd.DataFrame(dfa)
    obs_param = df['obs_param'].unique()[0]
    for atm in atmos_params:
        vvara = 'sigma_obs_param_{}'.format(atm)
        vvarb = 'sigma_{}_{}'.format(obs_param,atm)
        df = df.rename(columns={vvara:vvarb})
    
    df = df.rename(columns={'sigma_tot':'sigma_{}_tot'.format(obs_param)})
    
    return df

def plot_sigma_obs(df,obs_param='zp',sigma_obs_param=1,
                   band='y',xvar='sigma_pwv',yvar='sigma_aerosol',
                   airmass=1.2,fig=None,ax=None,
                   ellipse_color='yellow',ellipse_hatch='None',
                   frac_select=False):
    
    if fig is None:
        fig, ax = plt.subplots(figsize=(12,8))

    atmos_params = ['airmass','ozone','aerosol','pwv']
    
    for atm in atmos_params:
        thevar = 'frac_{}_{}'.format(obs_param,atm)
        denom = 'sigma_{}_{}'.format(obs_param,atm)
        num = 'sigma_{}_tot'.format(obs_param)
        combi_tot[thevar] = 100.*(combi_tot[denom]**2/combi_tot[num]**2)
        


    vv = 'sigma_{}_tot'.format(obs_param)
    idx = combi_tot['sigma_zp_tot'] >=0.95*sigma_obs_param
    idx &= combi_tot['sigma_zp_tot'] <=1.05*sigma_obs_param
    idx &= df['airmass'] == airmass
    idx &= df['band'] == band
    if frac_select:
        print('selecting',len(df[idx]))
        idx &= (df['frac_zp_pwv']+df['frac_zp_aerosol'])>= 80.
    
    sel = df[idx]
    print(len(sel))
    ax.plot(sel['sigma_pwv'],sel['sigma_aerosol'],'k.')
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    from matplotlib.patches import Ellipse
    #draw ellipse
    diff_pwv = sel['sigma_pwv'].diff().mean()
    ell_radius_x = sel['sigma_pwv'].max()+diff_pwv
    ell_radius_y = sel['sigma_aerosol'].max()
    ellipse = Ellipse((0, 0),
       width=ell_radius_x * 2,
       height=ell_radius_y * 2,
       facecolor=ellipse_color,hatch=ellipse_hatch)
    
    ax.add_patch(ellipse)
    
    ax.set_xlim([0.,xmax])
    ax.set_ylim([0.,ymax])
    
parser = OptionParser(description='Scan the atmos parameter sigma space')
                      
parser.add_option('--dataDir', type=str, default='../zp_atmos',
                  help='data dir [%default]')
parser.add_option('--atmos_param', type=str, default='airmass,ozone,aerosol,pwv',
                  help='atmospheric parameters [%default]')

opts, args = parser.parse_args()

theDir = opts.dataDir
atmos_params= opts.atmos_param.split(',')

df_zp, df_wave = load_fit_atmos_data(theDir, atmos_params)

sigma = {}
sigma['airmass'] = np.arange(0.0,0.012,0.001)
sigma['ozone'] = np.arange(0,50,5)
sigma['aerosol'] = np.arange(0,0.02,0.0001)
sigma['pwv'] = np.arange(0.,0.3,0.001)


combi_zp = process_data(df_zp)
combi_wave = process_data(df_wave)

#rename and merge

combi_zp = rename(combi_zp)
combi_wave = rename(combi_wave)

ccols = ['band','airmass']
for atm in atmos_params:
    ccols += ['sigma_{}'.format(atm)]
    
combi_tot = combi_zp.merge(combi_wave,left_on=ccols,right_on=ccols)
    
print(combi_tot.columns)

import matplotlib.pyplot as plt

sigma_obs_param=1
fig, ax = plt.subplots(figsize=(12,8))
"""
plot_sigma_obs(combi_tot,sigma_obs_param=sigma_obs_param,fig=fig,ax=ax,
               ellipse_color='yellow',ellipse_hatch=None)
"""
plot_sigma_obs(combi_tot,sigma_obs_param=sigma_obs_param,fig=fig,ax=ax,
               ellipse_color='yellow',ellipse_hatch='/',frac_select=True)
#plot_sigma_obs(combi_tot,airmass=2.0,sigma_obs_param=sigma_obs_param)

plt.show()