#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 14:26:22 2026

@author: philippe.gris@clermont.in2p3.fr
"""

from optparse import OptionParser
import pandas as pd
import numpy as np
from astropy.table import Table
from sn_plotter_analysis import plt
from sn_plotter_tools.plot_tools import plot_grid
from sn_tools.sn_io import checkDir

def plot(df,b='g',xvar='airmass',yvar='zp',
         bias_var='airmass',unit_y='%',unit_z='mmag'):
    """
    Function to make plots

    Parameters
    ----------
    df : pandas df
        Data to plot.
    b : str, optional
        filter. The default is 'g'.
    xvar : str, optional
        x-axis variable. The default is 'airmass'.
    yvar : str, optional
        y-axis variable. The default is 'zp'.
    bias_var : str, optional
        atmos bias param. The default is 'airmass'.
    unit_y : str, optional
        yvar unit. The default is '%'.
    unit_z : str, optional
        bias_var unit. The default is 'mmag'.

    Returns
    -------
    df : pandas df
        output data.

    """
    
    xxvar = 'orig_{}'.format(xvar)
    yyvar = 'mean_{}_{}'.format(yvar,b)
    
    bias_str = 'bias_{}'.format(bias_var)
    
    bias_values = df[bias_str]
    
    #add ref 
    cols = 'orig_{}'.format(xvar)
    dfa = df.groupby([cols]).apply(lambda x: get_var_ref(x,xvar,yvar,b),include_groups=False).reset_index()
    
    df = df.merge(dfa, left_on=cols,right_on=cols)

    yyvar_delta = 'delta_{}_{}'.format(yvar,b)
    yyvar_ref = 'mean_{}_{}_ref'.format(yvar,b)
    df[yyvar_delta] = 1000.*(df[yyvar]-df[yyvar_ref])

    idx = df[cols] >= 1.1
    idx &= df[cols] <= 2.4
    #idx &= np.abs(df[yyvar_delta]) <=5
    df = pd.DataFrame(df[idx])
    
    #multiply biases by 100 to be in %
    
    df[bias_str] *= 100
    ylabel = '{} bias [%]'.format(bias_var)
    figtit = '$\Delta_{'+yvar+'}$ ['+unit_z+']'
    figtit += '\n'
    figtit += '{} band'.format(b)
    
    df = plot_grid(Table.from_pandas(df),varx='orig_airmass',
              vary=bias_str,ylabel=ylabel,unit_y=unit_y,
              varz=yyvar_delta,
              figtitle=figtit,
              iso=[1.,2.,5.,-1,-2,-5],
              txt_iso=['1 mmag','2 mmag','5 mmag','-1 mmag','-2 mmag','-5 mmag'],
              x_iso_tag=[1.6]*6,k_ytext=1.1,
              lstyles = ['solid','dashed','dotted']*2,
              smoothIt=True,
              add_plot_y=[-1.,1.],
              add_plot_str=['-1 %','+1 %'],
              add_plot_color='magenta')
    
    return df
    """
    print('allll',bias_str)
    fig, ax = plt.subplots(figsize=(12,10))  
    ax.plot(df[cols],df[bias_str],'ko')
    

    for bb in bias_values:
        idx = df[bias_str] == bb
      
        sel = pd.DataFrame(df[idx])
        
        
        
        #ax.plot(sel[xxvar],sel[yyvar_delta],'ko')
        
        
    ax.set_xlabel(r'{}'.format(xvar))
    yvar_str = '$\Delta$'+'{}'.format(yvar)+'[{}]'.format(unit)
    ax.set_ylabel(r'{}'.format(yvar_str))
        
    ax.grid(visible=True)
    """
        
def get_var_ref(grp,xvar='airmass',yvar='zp',band='g'):
    """
    Function to grab the ref values (bias=0)

    Parameters
    ----------
    grp : pandas df
        Data to process.
    xvar : str, optional
        x-axis variable. The default is 'airmass'.
    yvar : str, optional
        y-axis variable. The default is 'zp'.
    band : str, optional
        filter to consider. The default is 'g'.

    Returns
    -------
    res : pandas df
        with ref values.

    """
    
    
    bias_str = 'bias_{}'.format(xvar)
    var_ref = 'mean_{}_{}'.format(yvar,band)    
    
    idx = np.abs(grp[bias_str]) < 1e-5
    
    rr = [grp[idx][var_ref].mean()]
    
    res = pd.DataFrame(rr,columns=['{}_ref'.format(var_ref)])
    
    return res
    
    
    

parser = OptionParser(description='analyze and plot zp and mean wave from bias run')

parser.add_option('--dataDir', type=str, default='../zp_atmos_bias',
                  help='data dir [%default]')
parser.add_option('--atmos_param', type=str, default='airmass',
                  help='bias atmospheric parameter [%default]')
parser.add_option('--obs', type=str, default='zp',
                  help='variable to plot [%default]')
parser.add_option('--bands', type=str, default='grizy',
                  help='filters to plot [%default]')
parser.add_option('--outDir', type=str, default='../zp_atmos_bias_summary',
                  help='filters to plot [%default]')

opts, args = parser.parse_args()

theDir = opts.dataDir
atmos_param= opts.atmos_param
obs=opts.obs
bands = opts.bands
outDir = opts.outDir

checkDir(outDir)
theFile = '{}/zp_atmos_{}.hdf5'.format(theDir,atmos_param)

df = pd.read_hdf(theFile)

dfr = pd.DataFrame()
for b in bands:
    dfa = plot(df,b,yvar=obs,bias_var=atmos_param)
    dfr = pd.concat((dfr,dfa))
 
fName = '{}/{}_atmos_bias_{}.hdf5'.format(outDir,obs,atmos_param)
print(dfr)
dfr.to_hdf(fName,key='bias')

plt.show()