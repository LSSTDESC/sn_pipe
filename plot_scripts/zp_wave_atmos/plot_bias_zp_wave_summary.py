#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 15:12:30 2026

@author: philippe.gris@clermont.in2p3.fr
"""

from optparse import OptionParser
from sn_plotter_analysis import plt,filtercolors,filtermarkers
import pandas as pd
import numpy as np
from sn_tools.sn_io import checkDir

def plot_atmos_bias(df,col='band',colx='bias_value [%]',
                    xlabel='airmass bias [%]',
                    coly='delta_zp [mmag]',
                    ylabel = '$\Delta zp$ [mmag]',
                    bands='grizy',
                    ymax=5,
                    ll=[1],ll_str='1 mmag',
                    fig=None,ax=None,outDir=None):
    
    if fig is None:
        fig, ax = plt.subplots(figsize=(12,8))
    
    #idx = df['airmass'] == airmass
    idx = df[coly] > 0.
    idx &= df[col].isin(list(bands))
    
    sel = df[idx]
    
    params = sel[col].unique()
    print(sel)
    
    airmass = sel['airmass'].unique()
    lstyle = ['solid','dotted']
    
    
    for i,airm in enumerate(airmass):
        idxb = sel['airmass'] == airm
        selb = sel[idxb]
       
        for param in params:
            
            idxc = selb[col]==param
        
            selc =selb[idxc]
        
            selc = selc.sort_values(by=colx)
            b = selc[col].unique()[0]
            label=None
            if i == 0:
                label='{} band'.format(param)
            ax.plot(np.abs(selc[colx]),selc[coly],
                    color=filtercolors[b],marker=filtermarkers[b],
                    markevery=40,
                    mfc='None',markersize=10,
                    linestyle=lstyle[i],
                    label=label)
        

    ax.grid(visible=True)
    ax.set_xlabel(r'{}'.format(xlabel))
    ax.set_ylabel(r'{}'.format(ylabel))
    ax.set_xlim([0.1,4.0])
    ax.set_ylim([0.,ymax])
    
    nbands = len(bands)
    nb = [1,2,3,4,5]
    xb = [0.30,0.25,0.20,0.10,0.05]
    xxb = dict(zip(nb,xb))
    
    ax.legend(loc='upper left',
                  bbox_to_anchor=(xxb[nbands], 1.15), 
                  ncol=5, frameon=False, fontsize=15)
    
    xmin, xmax = ax.get_xlim()
    ax.plot([xmin,xmax],ll*2,linestyle='dashed',lw=2,color='k')
    
    ax.text(4.05,1.,ll_str,fontsize=15)

    x_trans=0.25
    ax.annotate('', xy=(x_trans+0.,1.05), 
                xycoords='axes fraction', xytext=(x_trans+0.05, 1.05),
                arrowprops=dict(arrowstyle="-", color='k'))
    ax.text(x_trans+0.055,1.04,'airmass={}'.format(airmass[0]),
            fontsize=12,transform=ax.transAxes)
    ax.annotate('', xy=(x_trans+0.2,1.05), xycoords='axes fraction',
                xytext=(x_trans+0.25, 1.05),
               arrowprops=dict(arrowstyle="-", color='k',linestyle='dotted'))
    ax.text(x_trans+0.255,1.04,'airmass={}'.format(airmass[1]),
            fontsize=12,transform=ax.transAxes)
    
    if outDir is not None:
        fName='{}/bias_{}.png'.format(outDir,xlabel.split(' ')[0])
        plt.savefig(fName)

parser = OptionParser(description='summry plots from bias results')

parser.add_option('--dataDir', type=str, default='../zp_atmos_bias_summary',
                  help='data dir [%default]')
parser.add_option('--atmos_params', type=str, default='airmass,ozone,aerosol,pwv',
                  help='bias atmospheric parameters [%default]')
parser.add_option('--obs', type=str, default='zp',
                  help='variable to use as ref [%default]')
parser.add_option('--obs_unit', type=str, default='mmag',
                  help='unit variable to use as ref [%default]')
parser.add_option('--bands', type=str, default='grizy',
                  help='filters to plot [%default]')
parser.add_option('--outDir', type=str, default='../iso_zp',
                  help='output directory[%default]')

opts, args = parser.parse_args()

theDir = opts.dataDir
atmos_params= opts.atmos_params.split(',')
obs=opts.obs
obs_unit=opts.obs_unit
bands = opts.bands
outDir = opts.outDir

checkDir(outDir)

#load the data

df = pd.DataFrame()
for vv in atmos_params:
    fName = '{}/{}_atmos_bias_{}.hdf5'.format(theDir,obs,vv)
    da = pd.read_hdf(fName)
    df = pd.concat((df,da))
    
atmos_params=['airmass','ozone','aerosol','pwv']
bands=['grizy','gr','grizy','izy'] 
ymax = [5,2.5,5,5]
yymax=dict(zip(atmos_params,ymax))
bb = dict(zip(atmos_params,bands))

for atmos_param in atmos_params:
    #var_obs = 'delta_{} [{}]'.format(obs,obs_unit)
    idx = df['atmos_param'] == atmos_param

    plot_atmos_bias(df[idx],
                    xlabel='{} bias [%]'.format(atmos_param),
                    bands=bb[atmos_param],
                    ymax=yymax[atmos_param],outDir=outDir)
#print(df)

plt.show()