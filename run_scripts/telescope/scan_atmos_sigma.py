#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 08:52:23 2026

@author: philippe.gris@clarmont.in2p3.fr
"""
from optparse import OptionParser
from sn_analysis.sn_fit_tools import load_fit_atmos_data
#from sn_analysis.sn_atmos_tools import get_atmos_data 
from sn_analysis.sn_atmos_tools import rename,merge_zp_wave
from sn_analysis.sn_atmos_tools import process_obs_data
import numpy as np
import pandas as pd
import os

def process_all_data(theDir,atmos_params):
    """
    Data processing (general script)

    Parameters
    ----------
    theDir : str
        Data dir.
    atmos_params : list(str)
        List of atmos params.

    Returns
    -------
    combi_tot : pandas df
        Processed data.

    """
    
    df_zp, df_wave = load_fit_atmos_data(theDir, atmos_params)
    
    sigma = {}
    sigma['airmass'] = np.arange(0.0,0.012,0.001)
    sigma['ozone'] = np.arange(0,50,5)
    sigma['aerosol'] = np.arange(0,0.02,0.0001)
    sigma['pwv'] = np.arange(0.,0.3,0.001)
    
    combi_zp = process_obs_data(df_zp,sigma,atmos_params,do_combi=True)
    combi_wave = process_obs_data(df_wave,sigma,atmos_params,do_combi=True)
    
    #rename and merge
    
    combi_tot = merge_zp_wave(combi_zp,combi_wave,atmos_params)
    
    return combi_tot


def plot_sigma_obs(df,obs_param='zp',sigma_obs_param=1,
                   band='y',xvar='sigma_pwv',yvar='sigma_aerosol',
                   airmass=1.2,fig=None,ax=None,
                   ellipse_color='yellow',ellipse_hatch='None',
                   frac_select=False):
    """
    To plot some results

    Parameters
    ----------
    df : pandas df
        Data to plot.
    obs_param : str, optional
        obs parameter to consider. The default is 'zp'.
    sigma_obs_param : float, optional
        sigma obs param to select. The default is 1.
    band : str, optional
        band to consider. The default is 'y'.
    xvar : str, optional
        x-axis variable. The default is 'sigma_pwv'.
    yvar : str, optional
        y-axis variable. The default is 'sigma_aerosol'.
    airmass : float, optional
        airmass value. The default is 1.2.
    fig : matplotlib figure, optional
        Figure for the plot. The default is None.
    ax : matplotlib axis, optional
        axis for the plot. The default is None.
    ellipse_color : str, optional
        ellipse color. The default is 'yellow'.
    ellipse_hatch : str, optional
        ellipse hatch type. The default is 'None'.
    frac_select : bool, optional
        To select what to display. The default is False.

    Returns
    -------
    None.

    """
    
    if fig is None:
        fig, ax = plt.subplots(figsize=(12,8))

    atmos_params = ['airmass','ozone','aerosol','pwv']
    
    for atm in atmos_params:
        thevar = 'frac_{}_{}'.format(obs_param,atm)
        denom = 'sigma_{}_{}'.format(obs_param,atm)
        num = 'sigma_{}_tot'.format(obs_param)
        combi_tot[thevar] = 100.*(combi_tot[denom]**2/combi_tot[num]**2)
        


    vv = 'sigma_{}_tot'.format(obs_param)
    idx = df[vv] >=0.95*sigma_obs_param
    idx &= df[vv] <=1.05*sigma_obs_param
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

#data processing
fName = 'combi_atmos.hfd5'

if not os.path.isfile(fName):
    combi_tot = process_all_data(theDir, atmos_params)
    combi_tot.to_hdf(fName,key='atmos')
    
combi_tot=pd.read_hdf(fName)

#show plots
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