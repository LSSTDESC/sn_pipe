#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 14:12:04 2026

@author: philippe.gris@clermont.in2p3.fr
"""

from optparse import OptionParser
from sn_analysis.sn_fit_tools import load_fit_atmos_data
from sn_analysis.sn_atmos_tools import process_obs_data 
from sn_analysis.sn_atmos_tools import merge_zp_wave
import pandas as pd
import matplotlib.pyplot as plt    
from sn_tools.sn_io import checkDir

def add_index_band(df,bands='grizy'):
    """
    Function to add an index corresponding to filters

    Parameters
    ----------
    df : pandas df
        input data.
    bands: str, optional.
        list of bands to consider. The default is 'grizy'
        

    Returns
    -------
    df : pandas df
        orig data plus index col.

    """
    
    indx = range(len(bands))
    df_index = pd.DataFrame(list(bands),columns=['band'])
    df_index['index'] = indx
    
    df = df.merge(df_index,left_on=['band'],right_on=['band'])
    
    df = df.sort_values(by=['index'])
    
    return df
def get_str(df,atmos_params=['airmass','ozone','aerosol','pwv']):
    
    str_ = '('
    for i,vv in enumerate(atmos_params):
        val = df[vv].values[0]
        if val < 1:
            str_ += '{:.0e}'.format(val)
        else:
            str_ += '{}'.format(val)
        if i < len(atmos_params)-1:
            str_+= ','
        else:
            str_ += ')'
    
    return str_
def plot_results(df,config_df,obs_param='zp',unit='mmag',plotDir='',
                 tagline=[1,2,5,10]):
    """
    Function to plot the results

    Parameters
    ----------
    df : pandas df
        Data to plot.
    obs_param : str, optional
        obs parameter to plot (zp/mean_wave). The default is 'zp'.
    unit : str, optional
        unit corresponding to obs_param (mmag/nm). The default is 'mmag'.

    Returns
    -------
    None.

    """

    
    #add index
    df = add_index_band(df)
    
    fig, ax = plt.subplots(figsize=(12,8))
    fig.subplots_adjust(top=0.85,right=0.9)
    configs = df['config'].unique()
    airmass = df['airmass'].unique().tolist()
    airmass = list(map(float, airmass))
    
    yvar= 'sigma_{}_tot'.format(obs_param)
    
    lstyles = dict(zip(airmass,['solid','dotted']))
    markers = ['o','s','P','h','v']
    
    mmarks = dict(zip(configs,markers[:len(configs)]))
    
    for airm in airmass:
        idx = df['airmass'] == airm
        sel = df[idx]
        for config in configs:
            idxb = sel['config'] == config
            selb = sel[idxb]
            idxb = config_df['config']==config
            sel_config = config_df[idxb]
            label = get_str(sel_config,atmos_params=['airmass','ozone',
                                                     'aerosol','pwv'])
            label = '$\sigma_{atmos}$='+label
            if airm > 1.5:
                label = None
            ax.plot(selb['band'],selb[yvar],
                    marker=mmarks[config],mfc='None',
                    linestyle=lstyles[airm],label=label)
            
    ax.grid(visible=True)
    
    ax.legend(loc='upper left',
              bbox_to_anchor=(0.05, 1.2), ncol=2, frameon=False, fontsize=15)
    ax.set_xlabel(r'band')
    ylabel = '$\sigma_{'+obs_param+'}$'+ '[{}]'.format(unit)
    ax.set_ylabel(r'{}'.format(ylabel))
    xmin,xmax = ax.get_xlim()
    for tt in tagline:
        ax.plot([xmin,xmax],[tt]*2,linestyle='dashed',color='k')
        ax.text(1.02*xmax,tt,'{}'.format(tt)+' mmag',fontsize=12)
    ax.set_xlim([xmin,xmax])
    if plotDir != '':
        outName = '{}/summary_zp.png'.format(plotDir)
        plt.savefig(outName)

def plot_from_config(theDir,atmos_params,config,plotDir=''):
    
    #get interpolated values
    df_zp, df_wave = load_fit_atmos_data(theDir, atmos_params)
    
    #load atmos sigmas
    df_atm = pd.read_csv(config,comment='#')
    
    print(df_atm)
    sigma = df_atm.to_dict(orient='list')
    
    print(sigma)
    
    combi_zp = process_obs_data(df_zp,sigma,atmos_params,do_combi=False)
    combi_wave = process_obs_data(df_wave,sigma,atmos_params,do_combi=False)
    
    combi_tot = merge_zp_wave(combi_zp,combi_wave,atmos_params)
    
    print(combi_tot)
    
    plot_results(combi_tot,df_atm,plotDir=plotDir)   
    
def get_sigmas(grp,sigmas=[0.5,1.,2.,5.]):

    po = grp.name
    a = grp['slope'].values[0]
    b = grp['intercept'].values[0]
    
    from scipy.interpolate import interp1d
    import numpy as np
    
    sigma_ = np.arange(0.,10,0.001)
    rr = sigma_/a-b/a
    
    interp = interp1d(sigma_,rr,bounds_error=False, fill_value=0.)
    
    vv = interp(sigmas)
    
    sigma_col = 'sigma_obs_param'
    res = pd.DataFrame(sigmas,columns=[sigma_col])
    
    atm_col = 'sigma_atmos_param'
    
    res[atm_col] = vv
    
    return res
    
def plot_sigmas(theDir,atmos_params,plotDir=''):
    
    #get interpolated values
    df_zp, df_wave = load_fit_atmos_data(theDir, atmos_params)

    ccols = ['band','airmass','atmos_param','obs_param','atmos_param_value']
    res = df_zp.groupby(ccols).apply(lambda x:get_sigmas(x),include_groups=False).reset_index()
    
    res = add_index_band(res)

    plot_sigma_obs_param(res,plotDir=plotDir)
    auxtel_data=[100.*3.e-3/1.2,100.*20/300,100.*5.e-3/0.05,100.*0.2/5.]
    limy =[[0.,5.],[0.,30.],[0.,20.],[0.,30.]]
    plot_sigma_obs_param(res,err_rel=True,
                         auxtel_data=auxtel_data,limy=limy,plotDir=plotDir)

def plot_sigma_obs_param(res, obs_param='zp',unit_obs_param='mmag',
                         vary='sigma_atmos_param',err_rel=False,
                         atmos_params = ['airmass','ozone','aerosol','pwv'],
                         limy =[[0.,0.05],[0.,100.],[0.,0.0055],[0.,0.5]],
                         auxtel_data=[3.e-3,20,5.e-3,0.2],plotDir=''):
    
    
    
    if err_rel:
        res['sigma_atmos_param']/=res['atmos_param_value']/100.
        #limy = [0.,30.]*4
    
    auxtel_mes = dict(zip(atmos_params,auxtel_data))
    bands_atm = dict(zip(atmos_params,
                 ['grizy','gri','grizy','izy']))
    lstyle = dict(zip([1.2,2.0],['solid','dotted']))
    
    limy = dict(zip(atmos_params,limy))
    unit = dict(zip(atmos_params,['','[DU]','','[mm]']))

    sigmas = res['sigma_obs_param'].unique()
    
    markers = ['o','s','P','h']
    colors = ['m','r','b','g']
    
    mm = dict(zip(sigmas,markers))
    ccolors = dict(zip(sigmas,colors))
    
    for atm in atmos_params:
        idx = res['atmos_param'] == atm
        sela = res[idx]
        fig, ax = plt.subplots(figsize=(12,8))
        fig.subplots_adjust(right=0.85)
        airmass = sela['airmass'].unique()
        
        for airm in airmass:
            idx = sela['airmass'] == airm
            idx &= sela['band'].isin(list(bands_atm[atm]))
            selb = sela[idx]
            selb = selb.sort_values(by=['index','sigma_obs_param'])
            sigmas = selb['sigma_obs_param'].unique()
            
            for sig in sigmas:
                thelab = None
                if  airm == 1.2:
                    thelab = '$\sigma_{'+obs_param+'}$='+'{}'.format(sig)+' '+unit_obs_param
                
                idx = selb['sigma_obs_param'] == sig
                selc = selb[idx]
                ax.plot(selc['band'],selc[vary],
                        color=ccolors[sig],linestyle=lstyle[airm],
                        marker=mm[sig],mfc='None',label=thelab)
            
        ax.grid(visible=True)
        ax.set_ylim(limy[atm])
    
        if not err_rel:
            sig_atm = '$\sigma_{'+atm+'}$ '+format(unit[atm])
        else:
            sig_atm = '$\\frac{\sigma_{'+atm+'}}{<'+atm+'>}$ [%]'
            
        ax.set_ylabel(sig_atm)
        ax.set_xlabel('band')
        ax.legend(loc='upper left',
              bbox_to_anchor=(-0.1, 1.15), ncol=4, frameon=False, fontsize=15)
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
        # add auxtel typical measurements
        
        xmin,xmax = ax.get_xlim()
        vmes = auxtel_mes[atm]
        ax.plot([xmin,xmax],[vmes]*2,linestyle='dashed',color='k')
        ax.set_xlim([xmin,xmax])
        vunit = unit[atm].split('[')[-1].split(']')[0]
        if not err_rel:
            ttext = '$\sigma_{'+atm+'}$='+'{}'.format(vmes)+ ' {}'.format(vunit)
        else:
            import numpy as np
            ttext = '$\\frac{\sigma_{'+atm+'}}{<'+atm+'>}$='+'{}'.format(np.round(vmes,1))+ ' %'
        ax.text(1.02*xmax,vmes,ttext,fontsize=12)
    
        if plotDir != '':
            outName = '{}/sigma_{}_{}.png'.format(plotDir,atm,int(err_rel))
            plt.savefig(outName)

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
    plot_sigmas(theDir, atmos_params,plotDir=plotDir)
    
plt.show()