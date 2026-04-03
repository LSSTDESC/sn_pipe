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
from sn_plotter_tools.plot_tools import plot_airmass
import numpy as np
from scipy.interpolate import interp1d

def load_data(theDir,atmos_params):
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

    return df_zp,df_wave

def linfit_atmos(df,varxp='pwv',vary_prefix='zp',airmass=[1.2,2.0],bands='grizy'):
    
    ro = []
    print(df.columns)
    varx = 'sigma_{}'.format(varxp)
    for airm in airmass:
        idx = df['mean_airmass'] == airm
        sel = df[idx]
        for b in bands:
            yvar = 'std_{}_{}'.format(vary_prefix,b)
            yvar_rel = 'std_{}_{}_rel'.format(vary_prefix,b)
            sel[yvar_rel] = sel[yvar]/sel['mean_{}'.format(varxp)]
            print('fitting',sel[[varx,yvar,yvar_rel]])
            res = list(fit_lin(sel,varx,yvar))
            res += [b,airm]
            ro.append(res)
    dfn = pd.DataFrame(ro,columns=['slope','intercept','band','airmass'])
    dfn['atmos_param'] = varxp
    dfn['obs_param'] = vary_prefix
    dfn['obs_param_value'] = df['mean_{}'.format(varxp)].mean()
    dfn['sigma_max'] = df['{}'.format(varx)].max()
    return dfn
    
def plot(df,obs_param='zp',airmass=1.2,
         fig=None,ax=None,labelIt=True,
         linestyle='solid',color='k',valref=1):
    
    
    if fig is None:
        fig,ax = plt.subplots(figsize=(12,8))
    
    idx = df['obs_param'] == obs_param
    idx &= df['airmass'] == airmass
    
    sel = df[idx]
    
    atmos_params = sel['atmos_param'].unique()
    
    #fig, ax = plt.subplots(figsize=(12,8))
    #fig.suptitle('airmass={}'.format(airmass))
    
    atm_ref = ['airmass','pwv','ozone','aerosol']
    markers = ['o','s','P','h']
    marks = dict(zip(atm_ref,markers))
    
    for atm in atmos_params:
        #fig, ax = plt.subplots(figsize=(12,8))
        label = None
        if labelIt:
            label = atm
        idx = sel['atmos_param'] == atm
        selb = sel[idx]
        ax.plot(selb['band'],valref/selb['slope'],
                label=label,linestyle=linestyle,
                marker=marks[atm],mfc='None',markersize=12,color=color)
        
def plot_summary(df_zp,obs_param='zp',valref=1,unit='mmag'):
    
    fig, ax = plt.subplots(figsize=(12,8))  
    plot(df_zp,obs_param=obs_param,fig=fig,ax=ax,valref=valref)
    plot(df_zp,obs_param=obs_param,airmass=2.0,
         fig=fig,ax=ax,labelIt=False,linestyle='dotted',color='b',
         valref=valref)
    
    ax.legend(loc='upper left',
                  bbox_to_anchor=(0.1, 1.15), ncol=5, frameon=False, fontsize=15)
    
    ax.grid(visible=True)
    
    ax.set_yscale("log")
    x_trans=0.25
    ax.annotate('', xy=(x_trans+0.,1.05), 
                xycoords='axes fraction', xytext=(x_trans+0.05, 1.05),
                arrowprops=dict(arrowstyle="-", color='k'))
    ax.text(x_trans+0.055,1.04,'airmass=1.2',
            fontsize=12,transform=ax.transAxes)
    ax.annotate('', xy=(x_trans+0.2,1.05), xycoords='axes fraction',
                xytext=(x_trans+0.25, 1.05),
               arrowprops=dict(arrowstyle="-", color='k',linestyle='dotted'))
    ax.text(x_trans+0.255,1.04,'airmass=2.0',
            fontsize=12,transform=ax.transAxes)
    
    ax.set_xlabel(r'band')
    ylabel = '$\sigma_{atmos\ param}$'
    po = obs_param.replace('_','\ ')
    
    ylabel += '($\sigma_{'+po+'}$='+'{}'.format(valref)+' {}'.format(unit)+')'
    ax.set_ylabel(r'{}'.format(ylabel))
    
    #add auxtel performance
    xmin, xmax = ax.get_xlim()
    yv_auxtel=[0.2,20,3e-3,5e-3]
    coeff= [1.10]*2+[0.7]+[1.15]
    ttxt = ['$\sigma_{PWV}$','$\sigma_{ozone}$',
            '$\sigma_{airmass}$','$\sigma_{aerosol}$']
    units = ['mm','DU','','']
    for io,yy in enumerate(yv_auxtel):
        r = []
        for b in 'grizy':
            r.append((b,yy))
            
        dfaux = pd.DataFrame(r,columns=['band','auxres'])
        
        ax.plot(dfaux['band'],dfaux['auxres'],color='r',linestyle='dashed')
        """
        ax.text(x_trans+0.055,ypos[io],ttxt[io]+'='+'{}'.format(yy),
            fontsize=12,transform=ax.transAxes)
        """
        ax.text(3.2,coeff[io]*yy,ttxt[io]+'='+'{}'.format(yy)+' {}'.format(units[io]),
            fontsize=12,color='r')    

def plot_all_summary(df_zp,df_wave):
    
    plot_summary(df_zp)
    plot_summary(df_wave,obs_param='mean_wave',valref=0.1,unit='nm')
    
def plot_atmos_data_airmass(theDir,
                            atmos_params=['pwv','aerosol','airmass','ozone']):
    
    all_atm = ['pwv','aerosol','airmass','ozone']
    legxx = ['$\sigma_{PWV}$ [mm]',
            '$\sigma_{aerosol}$',
            '$\sigma_{airmass}$',
            '$\sigma_{ozone}$ [DU]']
    
    airmass=[1.2,2.0]
    xt = [0.15,0.011,0.01,25]
    xxtext=dict(zip(all_atm,xt))
    legx = dict(zip(all_atm,legxx))
    for vv in atmos_params:
        theFile = 'zp_atmos_{}.hdf5'.format(vv)
        fName = '{}/{}'.format(theDir,theFile)

        df = pd.read_hdf(fName)
        
        for b in 'grizy':
            df['std_zp_{}'.format(b)] *= 1000 # in mmag
           
        
        plot_airmass(df,varx='sigma_{}'.format(vv),xlabel=legx[vv],
                         vary_prefix='std_zp',airmass=airmass, 
                         y_iso=[1,2,3,5],
                         txt_iso=['1 mmag','2 mmag','3 mmag','5 mmag'],
                         xtext=xxtext[vv],smoothIt=False,fitIt=True) 
        
        plot_airmass(df,varx='sigma_{}'.format(vv),xlabel=legx[vv],
                         vary_prefix='std_mean_wave',
                         ylabel='$\sigma_{meanwave}$ [mm]',
                         airmass=airmass,
                         y_iso=[0.05,0.1,0.15],
                         txt_iso=['0.05 nm','0.1 nm','0.15 nm'],
                         ymax=0.2,deltay_txt=0.005,
                         xtext=xxtext[vv],smoothIt=False,fitIt=True)
    
def get_atmos_data(df,atmos_params=['airmass','ozone','aerosol','pwv']):
    
    cols = ['band','airmass',
                     'atmos_param','obs_param']
    dd = df.groupby(cols).apply(lambda x:get_interp(x),include_groups=False).reset_index()
    return dd

def get_interp(grp):
    
    
    a = grp['slope'].values[0]
    b = grp['intercept'].values[0]
    xmin = 0.
    xmax = grp['sigma_max'].values[0]
    xvals = np.linspace(xmin,xmax,100)
    yvals = a*xvals+b
    interp = interp1d(xvals,yvals)

    dd = {}
    dd['interp'] = [interp]
    return pd.DataFrame.from_dict(dd)
    
    
    
def get_values(df,sigmas=dict(zip(['airmass','ozone','aerosol','pwv'],
                                  [3e-3,20,5e-3,0.2]))):
    
    cols = ['band','airmass',
                    'atmos_param','obs_param']
    da = df.groupby(cols).apply(lambda x: get_val(x,sigmas),include_groups=False).reset_index()
     
    db = da.groupby(['band','airmass']).apply(lambda x: calc_combi(x),include_groups=False).reset_index()
   
    return db
def get_val(grp, thedict):
    
    atmos_param = grp.name[2]
    sigma = thedict[atmos_param]
    
    dd = {}
    myinterp = grp['interp'].values[0]
    dd['sigma_value'] = [myinterp(sigma)]
    
    return pd.DataFrame.from_dict(dd)
    
def calc_combi(grp):
    
    atmos_params = grp['atmos_param'].to_list()
    sigma_values = grp['sigma_value'].to_list()
    what = grp['obs_param'].unique()[0]
    
    rr = np.sqrt(np.sum(np.array(sigma_values)**2))
    
    atmos_params.append('total')
    po = 'sigma_{}_'.format(what)
    atmos_params = list(map(lambda x: po+x,atmos_params))
    sigma_values.append(rr)
    
    res = pd.DataFrame([sigma_values],columns=atmos_params)
    res = res.astype(float)
    res =res.round(decimals=4)
   
    return res

    
    
    
    
parser = OptionParser(description='Fit sigma_zp and sigma_mean_wave \
                      vs sigma of atmos params')
                      
parser.add_option('--dataDir', type=str, default='../zp_atmos',
                  help='data dir [%default]')
parser.add_option('--atmos_param', type=str, default='airmass,ozone,aerosol,pwv',
                  help='atmospheric parameters [%default]')

opts, args = parser.parse_args()

theDir = opts.dataDir
atmos_params= opts.atmos_param.split(',')

df_zp, df_wave = load_data(theDir, atmos_params)
rr = get_atmos_data(df_zp)

print(rr)

fres = get_values(rr)

print(fres)
#plot_all_summary(df_zp, df_wave)
  
#print('go',atmos_params)
#plot_atmos_data_airmass(theDir,atmos_params)

#print(df_zp)




plt.show()
