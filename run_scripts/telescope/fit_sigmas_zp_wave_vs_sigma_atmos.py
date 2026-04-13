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

atm_ref = ['airmass','pwv','ozone','aerosol','total']
markers = ['o','s','P','h','v']
marks = dict(zip(atm_ref,markers))

def load_atmos_data(theDir,atmos_params):
    """
    Function to load data (zp, mean_wave) vs sigma_atmos_params

    Parameters
    ----------
    theDir : str
        Data dir.
    atmos_params : list(str)
        List of atmospheric parameters.

    Returns
    -------
    df_zp : pandas df
        zp data.
    df_wave : pandas df
        mean wave data.

    """
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

def linfit_atmos(df,varxp='pwv',
                 vary_prefix='zp',airmass=[1.2,2.0],bands='grizy'):
    """
    Function to perform a linear fit of zp/mean_wave vs atmos params

    Parameters
    ----------
    df : pandas df
        Data to fit.
    varxp : str, optional
        atmos parameter. The default is 'pwv'.
    vary_prefix : str, optional
        prefix obs (zp/mean_wave). The default is 'zp'.
    airmass : list(float), optional
        List of airmass values o consider. The default is [1.2,2.0].
    bands : str, optional
        Filters to consider. The default is 'grizy'.

    Returns
    -------
    dfn : pandas df
        output data.

    """
    
    ro = []
    #print(df.columns)
    varx = 'sigma_{}'.format(varxp)
    for airm in airmass:
        idx = df['mean_airmass'] == airm
        sel = pd.DataFrame(df[idx])
        for b in bands:
            yvar = 'std_{}_{}'.format(vary_prefix,b)
            yvar_rel = 'std_{}_{}_rel'.format(vary_prefix,b)
            sel[yvar_rel] = sel[yvar]/sel['mean_{}'.format(varxp)]
            #print('fitting',sel[[varx,yvar,yvar_rel]])
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
    """
    Function to plot zp,mean_wave vs inverse of the fit slope

    Parameters
    ----------
    df : pandas df
        Data to plot.
    obs_param : str, optional
        obs parameter. The default is 'zp'.
    airmass : float, optional
        airmass value. The default is 1.2.
    fig : matplotlib figure, optional
        Figure for the plot. The default is None.
    ax : matplotlib axis, optional
        axis for the plot. The default is None.
    labelIt : bool, optional
        To add a label to the plot. The default is True.
    linestyle : str, optional
        line style for the plot. The default is 'solid'.
    color : str, optional
        color for the plot. The default is 'k'.
    valref: float, optional.
        reference value. The default is 1.

    Returns
    -------
    None.

    """
    
    
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
    """
    summary plot

    Parameters
    ----------
    df_zp : pandas df
        Data to plot.
    obs_param : str, optional
        obs parameter. The default is 'zp'.
    valref : float, optional
        reference value. The default is 1.
    unit : str, optional
        plot unit. The default is 'mmag'.

    Returns
    -------
    None.

    """
    
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
    """
    plot all summary

    Parameters
    ----------
    df_zp : pandas df
        zp data.
    df_wave : pandas df
        mean wave data.

    Returns
    -------
    None.

    """
    
    plot_summary(df_zp)
    plot_summary(df_wave,obs_param='mean_wave',valref=0.1,unit='nm')
    
def plot_atmos_data_airmass(theDir,
                            atmos_params=['pwv','aerosol','airmass','ozone']):
    """
    plot atmos data

    Parameters
    ----------
    theDir : str
        Data dir.
    atmos_params : list(str), optional
        List of atmos parameters. 
        The default is ['pwv','aerosol','airmass','ozone'].

    Returns
    -------
    None.

    """
    
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
    """
    Function to grab atmos data (interpolator)

    Parameters
    ----------
    df : pandas df
        Data to process.
    atmos_params : list(str), optional
        List of atmos parameters. 
        The default is ['airmass','ozone','aerosol','pwv'].

    Returns
    -------
    dd : pandas df
        Output data.

    """
    
    cols = ['band','airmass',
                     'atmos_param','obs_param']
    dd = df.groupby(cols).apply(lambda x:get_interp(x),include_groups=False).reset_index()
    return dd

def get_interp(grp):
    """
    Function to grab interpolator of atmos params

    Parameters
    ----------
    grp : pandas df
        Data to process.

    Returns
    -------
    pandas df
        table of interpolators.

    """
    
    
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
    """
    Function to estimate sigma_zp and sigma_mean_wave 
    for a set of sigmas of atmos params

    Parameters
    ----------
    df : pandas df
        Data to process.
    sigmas : dict, optional
        Atmos parameters. 
        The default is dict(zip(['airmass','ozone','aerosol','pwv'],                                  [3e-3,20,5e-3,0.2])).

    Returns
    -------
    db : pandas df
        Output data.

    """
    
    cols = ['band','airmass',
                    'atmos_param','obs_param']
    da = df.groupby(cols).apply(lambda x: get_val(x,sigmas),include_groups=False).reset_index()
     
    db = da.groupby(['band','airmass']).apply(lambda x: calc_combi(x),include_groups=False).reset_index()
   
    return db

def get_val(grp, thedict):
    """
    Function to grab interp values

    Parameters
    ----------
    grp : pandas df
        Data to process.
    thedict : dict
        sigma values.

    Returns
    -------
    pandas df
        output data.

    """
    
    atmos_param = grp.name[2]
    sigma = thedict[atmos_param]
    
    dd = {}
    myinterp = grp['interp'].values[0]
    dd['sigma_value'] = [myinterp(sigma)]
    
    return pd.DataFrame.from_dict(dd)
    
def calc_combi(grp):
    """
    Function to estimate combination of sigmas

    Parameters
    ----------
    grp : pandas df
        Data to process.

    Returns
    -------
    res : pandas df
        Output data.

    """
    
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

    
def plot_perf(data,x_main='sigma',
              obs_param='zp',unit='mmag',ylabel='zp',
              atmos_params=['airmass','ozone','aerosol','pwv','total'],
              airmass=[1.2,2.],ylines=[1,5,10],
              yannot=['1 mmag','5 mmag','10 mmag'],extra_leg=''):
    """
    Function to draw a perf plot

    Parameters
    ----------
    data : pandas df
        Data to plot.
    x_main : str, optional
        prefix var. The default is 'sigma'.
    obs_param : str, optional
        obs param. The default is 'zp'.
    unit : str, optional
        obs param unit. The default is 'mmag'.
    ylabel : str, optional
        y-axis label. The default is 'zp'.
    atmos_params : list(str), optional
        list of atmos params. 
        The default is ['airmass','ozone','aerosol','pwv','total'].
    airmass : list(float), optional
        List of airmass to consider. The default is [1.2,2.].
    ylines : list(float), optional
        y-values of lines to draw. The default is [1,5,10].
    yannot : list(str), optional
        ylines annot. The default is ['1 mmag','5 mmag','10 mmag'].
    extra_leg: str, optional.
        extra legend to add to the plot. The default is ''.

    Returns
    -------
    None.

    """
    
    fig, ax = plt.subplots(figsize=(12,8))
    
    ls = dict(zip(airmass,['solid','dotted']))
    color = dict(zip(airmass,['black','red']))
    
    for airm in airmass:
        idx = data['airmass'] == airm
        sel = data[idx]
        
        for atm_param in atmos_params:
            label =atm_param
            if airm > 1.2:
                label = None
            obs_str = '{}_{}_{}'.format(x_main,obs_param,atm_param)
            ax.plot(sel['band'],sel[obs_str],
                    marker=marks[atm_param],mfc='None',
                    linestyle=ls[airm],color=color[airm],label=label)
    
    xmin,xmax = ax.get_xlim()
    for io,yl in enumerate(ylines):
        ax.plot([xmin,xmax],[yl]*2,linestyle='dashed',color='b')
        ax.text(1.02*xmax,yl,'{}'.format(yannot[io]),
            fontsize=12,color='b')       
    ax.grid(visible=True)
    
    if x_main == 'sigma':
        ylabel = '$\\'+x_main+'_{'+ylabel+'}$ ['+unit+']'
    else:
        #ylabel = x_main+'$_{\sigma_{'+ylabel+'}}$ ['+unit+']'
        ylabel = '$\sigma_{'+ylabel+'}$'+' budget ['+unit+']'
    ax.set_ylabel(r'{}'.format(ylabel))
    ax.set_xlabel(r'band')
    
    ax.legend(loc='upper left',
                 bbox_to_anchor=(0.1, 1.15), 
                 ncol=5, frameon=False, fontsize=15)
    
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
    
    ax.set_xlim([xmin,xmax])
    
    if extra_leg != '':
       ax.text(-0.15,0.97,extra_leg,
            fontsize=12,transform=ax.transAxes,color='b') 
    
def plot_perf_obs_param(df_zp,sigmas,unit_atmos,
                    obs_param='zp',unit='nm',ylabel='zp',
                    ylines=[1,5,10],
                    yannot=['1 mmag','5 mmag','10 mmag']):
    """
    Function to draw perf plots for an obs_param

    Parameters
    ----------
    df_zp : pandas df
        Data to plot.
    sigmas : dict
        sigmas of atmos params.
    unit_atmos : dict
        unit for sigmas of atmos params.    
    obs_param : str, optional
        obs param. The default is 'zp'.
    unit : str, optional
        obs param unit. The default is 'nm'.
    ylabel : str, optional
        y-axis label. The default is 'zp'.
    ylines : list(float), optional
        y-values of lines to draw. The default is [1,5,10].
    yannot : list(str), optional
        y-lines annot. The default is ['1 mmag','5 mmag','10 mmag'].

    Returns
    -------
    None.

    """
    
    bands = 'grizy'
    b_index = [0,1,2,3,4]
    dfb = pd.DataFrame(list(bands),columns=['band'])
    dfb['band_index'] = b_index
  
    
    extra_leg = ''
    
    for key,vals in sigmas.items():
        vvar = '$\sigma_{'+key+'}$='+'{}'.format(vals)
        extra_leg += vvar + ' ' +unit_atmos[key]+'\n'
        
    rr_zp = get_atmos_data(df_zp)
    res_zp = get_values(rr_zp,sigmas=sigmas)
  
    res_zp = res_zp.merge(dfb,left_on=['band'],right_on=['band'])
  
    res_zp = res_zp.sort_values(by=['band_index'])
  
  
    res_zp['frac_check'] = 0
    vart = 'sigma_{}_total'.format(obs_param)
    for atm_param in atmos_params:
        fracx = 'frac_{}_{}'.format(obs_param,atm_param)
        varx = 'sigma_{}_{}'.format(obs_param,atm_param)
        res_zp[fracx] = 100.*res_zp[varx]**2/res_zp[vart]**2
        res_zp['frac_check'] += res_zp[fracx]
  
    plot_perf(res_zp,obs_param=obs_param,unit =unit,
            ylabel=ylabel,ylines=ylines,yannot=yannot,extra_leg=extra_leg)
    
    plot_perf(res_zp,x_main='frac',obs_param=obs_param,unit='%',ylabel=ylabel,
              atmos_params=['airmass','ozone','aerosol','pwv'],ylines=[],
              extra_leg=extra_leg) 
  
  
parser = OptionParser(description='Fit sigma_zp and sigma_mean_wave \
                      vs sigma of atmos params')
                      
parser.add_option('--dataDir', type=str, default='../zp_atmos',
                  help='data dir [%default]')
parser.add_option('--atmos_param', type=str, default='airmass,ozone,aerosol,pwv',
                  help='atmospheric parameters [%default]')
parser.add_option('--plots', type=str, default='vs_airmass,summary,from_sigmas',
                  help='plots [%default]')
parser.add_option('--sigmas', type=str, default='3e-3,20,5e-3,0.2',
                  help='sigmas of atmos params [%default]')
parser.add_option('--unit', type=str, default=',DU,,mm',
                  help='unit of sigmas of atmos params [%default]')

opts, args = parser.parse_args()

theDir = opts.dataDir
atmos_params= opts.atmos_param.split(',')
plots = opts.plots.split(',')
sigmas = opts.sigmas.split(',')
unit = opts.unit.split(',')
sigmas = list(map(float, sigmas))
sigmas = dict(zip(atmos_params,sigmas))
unit = dict(zip(atmos_params,unit))

df_zp, df_wave = load_atmos_data(theDir, atmos_params)

if 'summary' in plots:
    plot_all_summary(df_zp, df_wave)
  
if 'vs_airmass' in plots:
    plot_atmos_data_airmass(theDir,atmos_params)

if 'from_sigmas':
    plot_perf_obs_param(df_zp,sigmas,unit)
    plot_perf_obs_param(df_wave,sigmas,unit,obs_param='mean_wave',unit = 'nm',
                    ylabel='mean\ wave',ylines=[0.1],yannot=['0.1 nm'])

plt.show()
