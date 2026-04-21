#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 14:12:04 2026

@author: philippe.gris@clermont.in2p3.fr
"""

from optparse import OptionParser
from sn_analysis.sn_fit_tools import load_fit_atmos_data
from sn_analysis.sn_atmos_tools import process_obs_data 
from sn_analysis.sn_atmos_tools import merge_zp_wave,get_sigmas_atmos
from sn_plotter_tools.plot_atmos_tools import plot_results_config
from sn_plotter_tools.plot_atmos_tools import plot_sigma_obs_param
import pandas as pd
import matplotlib.pyplot as plt    
from sn_tools.sn_io import checkDir


def plot_from_config(theDir,atmos_params,config,plotDir=''):
    """
    Plot of sigma_zp (tot) from configuration file of sigma_atmos.'

    Parameters
    ----------
    theDir : str
        Data input dir.
    atmos_params : list(str)
        List of atmos parameters.
    config : csv file
        config file.
    plotDir : str, optional
        Output dir for the plots. The default is ''.

    Returns
    -------
    None.

    """
    
    #get interpolated values
    df_zp, df_wave = load_fit_atmos_data(theDir, atmos_params)
    
    #load atmos sigmas
    df_atm = pd.read_csv(config,comment='#')
    
    sigma = df_atm.to_dict(orient='list')
    
    combi_zp = process_obs_data(df_zp,sigma,atmos_params,do_combi=False)
    combi_wave = process_obs_data(df_wave,sigma,atmos_params,do_combi=False)
    
    combi_tot = merge_zp_wave(combi_zp,combi_wave,atmos_params)
        
    plot_results_config(combi_tot,df_atm,plotDir=plotDir)   
    

    
def plot_sigma_atmos_band(theDir,atmos_params,plotDir=''):
    """
    Plot sigma_atmos vs band for a set of sigma_zp_atmos values

    Parameters
    ----------
    theDir : str
        Data dir.
    atmos_params : list(str)
        List of atmospheric parameters.
    plotDir : str, optional
        Output dir for the plots. The default is ''.

    Returns
    -------
    None.

    """
    
    #get interpolated values
    df_zp, df_wave = load_fit_atmos_data(theDir, atmos_params)

    ccols = ['band','airmass','atmos_param','obs_param','atmos_param_value']
    res = df_zp.groupby(ccols).apply(lambda x:get_sigmas_atmos(x),include_groups=False).reset_index()

    plot_sigma_obs_param(res,plotDir=plotDir)
    auxtel_data=[100.*3.e-3/1.2,100.*20/300,100.*5.e-3/0.05,100.*0.2/5.]
    limy =[[0.,5.],[0.,30.],[0.,20.],[0.,30.]]
    plot_sigma_obs_param(res,err_rel=True,
                         auxtel_data=auxtel_data,limy=limy,plotDir=plotDir)





parser = OptionParser(description='Scan the atmos parameter sigma space')
                      
parser.add_option('--dataDir', type=str, default='../zp_atmos',
                  help='data dir [%default]')
parser.add_option('--atmos_param', type=str, default='airmass,ozone,aerosol,pwv',
                  help='atmospheric parameters [%default]')
parser.add_option('--config', type=str, default='config_atmos.csv',
                  help='sigma atmos parameters [%default]')
parser.add_option('--plots', type=str, default='from_config,sigmas',
                  help='plots to perform [%default]')
parser.add_option('--plotDir', type=str, default='../iso_zp',
                  help='where the plots will be saved [%default]')
opts, args = parser.parse_args()

theDir = opts.dataDir
atmos_params= opts.atmos_param.split(',')
config = opts.config
plots = opts.plots.split(',')
plotDir = opts.plotDir

#create outDir 
checkDir(plotDir)

if 'from_config' in plots:
    plot_from_config(theDir, atmos_params, config,plotDir=plotDir)
    
if 'sigmas' in plots:
    plot_sigma_atmos_band(theDir, atmos_params,plotDir=plotDir)
    
plt.show()