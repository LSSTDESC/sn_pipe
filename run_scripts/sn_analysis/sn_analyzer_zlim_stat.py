#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 10:58:22 2025

@author: philippe.gris@clermont.in2p3.fr
"""
from optparse import OptionParser
import glob
import pandas as pd
from sn_analysis.sn_selection import selection_criteria,select
from sn_analysis.sn_tools import complete_df
#from sn_analysis.sn_fit_tools import fit_linear,lin
from sn_analysis.sn_calc_plot import effi
import numpy as np
import re
#import operator
from sn_tools.sn_io import checkDir
from sn_tools.sn_utils import multiproc
import time
from sn_analysis.sn_nsn_effi import getRates
from scipy.interpolate import interp1d

def load_data(dbDir, dbName, runType, field,x1,color,nproc=8):
    """
    Function to load the data

    Parameters
    ----------
    dbDir : str
        data dir.
    dbName : str
        OS to process.
    runType : str
        runtype.
    field : str
        field.
    x1: float
        SN stretch
    color: float
        SN color
    nproc: int,optional.
        number of proc to use. The default is 8.

    Returns
    -------
    df : pandas df
        loaded data.

    """

    theDir = '{}/{}/{}'.format(dbDir, dbName, runType)

    print('scanning', theDir)
    search_path = '{}/*{}*_{}_{}*.hdf5'.format(theDir, field,x1,color)
    print(search_path)
    fis = glob.glob(search_path)

    print('files to load',len(fis))
    params = {}
    df = multiproc(fis,params,load_data_set,nproc)
    
    return df


def load_data_set(toproc, params, j=0, output_q=None):
    """
    Function to load a data set

    Parameters
    ----------
    toproc : list(str)
        List of files to load.
    params : dict
        parameters.
    j : int, optional
        internal tag for multiprocessing. The default is 0.
    output_q : multiprocessing queue, optional
        Where to put the results. The default is None.

    Returns
    -------
    pandas df
        Loaded data.

    """
    
    
    df = pd.DataFrame()

    for fi in toproc:
        df_ = pd.read_hdf(fi)
        df = pd.concat((df, df_))
        
    if output_q is not None:
            return output_q.put({j: df})
    else:
        return df

def select_str_deprecated(res, list_sel):
    """
    Function to select a pandas df

    Parameters
    ----------
    res : pandas df
        data to select.

    Returns
    -------
    pandas df
        selected df.

    """
    idx = True
    for vals in list_sel:
        idx &= vals[1](res[vals[0]], vals[2])
        mystr = '{} {} {}'.format(
            vals[0], get_symbol(vals[1].__doc__), vals[2])

    return mystr, res[idx]


def get_symbol(opdoc):
    """
    function to estimate symbol from operator.__doc__

    Parameters
    ----------
    opdoc : str
        operator.__doc__.

    Returns
    -------
    sym : str
        corresponding sym.

    """
    # sym = re.sub(r'.*\w\s?(\S+)\s?\w.*', '\\1', getattr(operator, op).__doc__)
    sym = re.sub(r'.*\w\s?(\S+)\s?\w.*', '\\1', opdoc)
    if re.match('^\\W+$', sym):
        return sym

def zlim_field(field,dbDir,dbName,runType,x1,color,nproc):
    """
    Function to estimate redshift limits

    Parameters
    ----------
    field : str
        Field name.
    dbDir : str
        Data location dir.
    dbName : str
        OS to consider.
    runType : str
        Type of run (DDF/WFD).
    x1: float
        SN Ia stretch
    color: float
        SN Ia color
    nproc: int
        n proc for multiprocessing

    Returns
    -------
    pandas df
        output data.

    """
    
    time_ref = time.time()
    
    dbDirs = dbDir.split(',')
    data = pd.DataFrame()
    for dbDir in dbDirs:
        data_ = load_data(dbDir, dbName, runType, field,x1,color)
        print('loaded',time.time()-time_ref)
        data_['field'] = field
        data_ = complete_df(data_)
        data_['config'] = dbDir.split('../')[1]
        data = pd.concat((data,data_))

    params = {}
    
    params['data'] = data
    from sn_tools.sn_utils import multiproc
    
    obs = data['healpixID'].unique().tolist()
    
    #obs = [143706.0]
    res = multiproc(obs,params,zlim_field_multiproc,nproc=nproc)
    
    return res
   
    
def zlim_field_multiproc(toproc, params, j=0, output_q=None):
    """
    Function to estimate redshift limits using multiprocessing

    Parameters
    ----------
    toproc : list(int)
        List of healpixIDs to process.
    params : dict
        Dict of parameters.
    j : int, optional
        internal int for multiprocessing. The default is 0.
    output_q : multiprocessing queue, optional
        Where to put the results. The default is None.

    Returns
    -------
    pandas df
        Output data.

    """
    
    data = params['data']
    
    idx = data['healpixID'].isin(toproc)
    sel_data = pd.DataFrame(data[idx])
    
    sel_data['healpixID'] = sel_data['healpixID'].astype(int)
    sel_data['season'] = sel_data['season'].astype(int)
    
    df_zlim = sel_data.groupby(['field','healpixID','pixRA','pixDec','season']).apply(
     lambda x: process_season_pixel(x, sellist,plot=True), 
       include_groups=False).reset_index()
    
    if output_q is not None:
        return output_q.put({j: df_zlim})
    else:
        return df_zlim
    
def process_db(dbDir, dbName, runType, fields,x1,color,nproc,sellist=None):
    """
    Function to process OS data

    Parameters
    ----------
    dbDir : str
        Data dir.
    dbName : str
        OS to process.
    runType : str
        run type.
    fields : list(str)
        List of fields to process.  
    x1: float
        SN stratch value
    color: float
        SN color value
    nproc: int
     n proc for multiprocessing
    sellist: dict, optional.
        slection criteria. The default is None.

    Returns
    -------
    df_effi : pandas df
        processed data.

    """

    df_zlim = pd.DataFrame()

    for field in fields:
        
        df_zlim_ = zlim_field(field, dbDir, dbName, runType,x1,color,nproc)
        df_zlim = pd.concat((df_zlim,df_zlim_))

    return df_zlim

def process_season_pixel(grp, sellist,plot=False):
    """
    Function to process a pixel/season

    Parameters
    ----------
    grp : pandas df
        Data to process.
    sellist : dict
        selection dict.
    plot : bool, optional
        To plot the results. The default is False.

    Returns
    -------
    None.

    """
    field = grp.name[0]
    hpix = '{}'.format(grp.name[1])
    season = '{}'.format(grp.name[-1])
    figtit = '{} {} {}'.format(field,hpix,season)
    print('test',grp.name)
    
    idx = np.abs(grp['sigmaC']) < 0.04
    grp = grp[idx]
    
    print('alllors',len(grp))
    
    #get selected group
    grp_sel = select(grp,sellist)
    
    print(len(grp),len(grp_sel))
  
    plot_hist(grp_sel)
    plot_indiv(grp_sel,figtit,varx='z_fit',vary='diff_mu',
                 var_std='sigma_mu',var_norm='')
    
    #get binned values
    
    ddf = grp_sel.groupby(['z','config']).apply(lambda x : get_vals(x),
                                   include_groups=False).reset_index()
    
    print('nsn',ddf['nsn'].sum())
    figtit = '{} {} {}'.format(field,hpix,season)
    
    plot_indiv(ddf,figtit)
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12,8))
    tit = '{} {} {}'.format(field,hpix,season)
    fig.suptitle(tit)
    configs = ddf['config'].unique()
    colors = ['k','r']
    lstyle = ['solid','dashed']
    
    for i,conf in enumerate(configs):
        idx = ddf['config'] == conf
        sel = ddf[idx]
        print('nsn',conf,sel['nsn'].sum())
        ax.errorbar(sel['z_fit'],sel['diff_mu']/sel['mu'],
                    yerr=sel['diff_mu_std']/sel['mu'],
                    color=colors[i],marker='o',linestyle=lstyle[i],
                    label=conf)
    ax.legend()
    """
    
    
    return

def plot_hist(df,fig=None,ax=None):
    
    
    if fig is None:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12,8))
       
    configs = df['config'].unique()
    colors = ['k','r']
    lstyle = ['solid','dashed']
    
    for i,conf in enumerate(configs):
        idx = df['config'] == conf
        sel = df[idx]  
        ax.hist(sel['sigma_mu']/sel['mu'],histtype='step',bins=20,
                linestyle=lstyle[i],color=colors[i],
                label=conf)
    
    ax.legend()
    

def plot_indiv(ddf,figtit,varx='z_fit',vary='diff_mu',
               var_std='diff_mu_std',var_norm='mu'):
    
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12,8))
    fig.suptitle(figtit)
    configs = ddf['config'].unique()
    colors = ['k','r']
    lstyle = ['solid','dashed']
    
    for i,conf in enumerate(configs):
        idx = ddf['config'] == conf
        sel = ddf[idx]
        if 'nsn' in sel.columns:
            print('nsn',conf,sel['nsn'].sum())
        vary_pl = sel[vary]
        vary_std_pl = sel[var_std]
        if var_norm != '':
            vary_pl /= sel[var_norm]
            vary_std_pl/=sel[var_norm]
        ax.errorbar(sel[varx],vary_pl,
                    yerr=vary_std_pl,
                    color=colors[i],marker='o',linestyle=lstyle[i],
                    label=conf)
    ax.legend()
    
    plt.show()
    
    """
    print(test)
    deltab=0.1
    zmin = 0.01
    zmax = grp['z_fit'].max()+deltab
    bins = np.arange(zmin,zmax,deltab)
   
    from sn_analysis.sn_calc_plot import bin_it_weighted
    
    res = bin_it_weighted(grp_sel,xvar='z_fit', 
                      yvar='diff_mu',yvar_err='diff_mu_std',bins=bins)
   
    print(res.columns)
   
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    
    ax.errorbar(res['z_fit'],res['diff_mu_weighted_mean'],
                yerr=res['diff_mu_std'],color='k',marker='o')
    
    plt.show()
    """
def get_vals(grp):
    
    
    print(grp.name,len(grp))
    
    dd = {}
    
    grp['coeff'] = 1./grp['sigma_mu']**2
    
    dd['z_fit'] = [grp['z_fit'].mean()]
    
    vals = np.sum(grp['coeff'])
    vv = np.sqrt(vals)
    
    dd['diff_mu_std'] = [1./vv]
    
    
    vvb = np.sum(grp['coeff']*grp['diff_mu'])/vals
    
    dd['diff_mu'] = [vvb]
    
    vvc = np.sum(grp['coeff']*grp['mu'])/vals
    
    dd['mu'] = [vvc]
    
    dd['nsn'] = [len(grp)]
    
    res = pd.DataFrame.from_dict(dd)
    
    return res
    
    
    
    
def process_season_pixel_old(grp, sellist,plot=False):
    """
    Function to process a season/pixel

    Parameters
    ----------
    grp : pandas df
        Data to process.
    sellist : list(str)
        Selection criteria.
    plot : bool, optional
        To display results. The default is False.

    Returns
    -------
    zlim_t : pandas df
        Output results (zlims).

    """
    
    #get selected group
    grp_sel = select(grp,sellist)
    
    #get selection efficiencies
    deltab=0.1
    zmin = 0.01
    zmax = grp['z_fit'].max()+deltab
    bins = np.arange(zmin,zmax,deltab)
    
    #grab efficiencies
    grp_effi=effi(grp,grp_sel,xvar='z_fit',bins=bins)
    
    zmin=0.01
    zmax = 1.1
    dz = 0.001
    #get the number of SN from the rate of explosion
    df_nsn_rate, sel_sn=get_all_nsn_from_rate(grp,grp_effi, zmin, zmax, dz)
   
   

    #axa.errorbar(sel_sn['z_fit'],sel_sn['ratio_nsn_norm'],yerr=sel_sn['ratio_nsn_err'])
    #axa.fill_between(sel_sn['z_fit'],sel_sn['ratio_nsn_norm_p'],sel_sn['ratio_nsn_norm_m'],color='yellow')
    nsn_accepted_loss=0.02
    nsn_inside = 1.-nsn_accepted_loss
    #axa.plot([0.1,1.1],[nsn_inside]*2)
    
    if plot:
        plot_proc(grp,grp_effi,df_nsn_rate)
        plot_test(sel_sn,nsn_inside=nsn_inside)
    """ 
    thedict = dict(zip(['zlim','zlim_p','zlim_m'],
                       ['ratio_nsn_norm',
                        'ratio_nsn_norm_p',
                        'ratio_nsn_norm_m']))
    """
    bins=np.arange(0.0,1.3,0.2)
    sel_sn['range'] = pd.cut(sel_sn['z_fit'],bins=bins,
                             right=False,include_lowest=True)
    sel_sn['range'] = sel_sn['range'].astype(str)
    
    df_d = {}
    for nsn_i in [0.95,0.98]:
        zlim_t = sel_sn.groupby('range').apply(lambda x: \
                                               get_zlim_from_df(x,nsn_inside=nsn_i),
                                               include_groups=False).reset_index()
        idx = zlim_t['zlim_{}'.format(nsn_i)].idxmax()
        vv = zlim_t.loc[idx]
        rr = vv.to_frame().T
        rr = rr.drop(columns=['range','level_1'])
        df_d[nsn_i] = rr
        
    zlim_f = df_d[0.98].merge(df_d[0.95],how='cross').reset_index()

    return zlim_f
    

def get_all_nsn_from_rate(grp,grp_effi,zmin,zmax,dz):
    """
    Function to estimate the number of supernovae from rate

    Parameters
    ----------
    grp : pandas df
        Data to process.
    grp_effi : pandas df
        efficiencies.
    zmin : float
        min redfshift.
    zmax :  float
        max redshift.
    dz : float
        delta redshift.

    Returns
    -------
    df_nsn_rate : pandas df
        nsn rate.
    sel_sn : pandas df
        selected sn (z>0.1).

    """
    
    zmin=0.01
    zmax = 1.1
    dz = 0.001
    df_nsn_rate = get_nsn_from_rate(grp,grp_effi,zmin,zmax,dz)
    
    idx = df_nsn_rate['z_fit']>0.1
    sel_sn = pd.DataFrame(df_nsn_rate[idx])
    sel_sn['nsn_effi_sum'] = np.cumsum(sel_sn['nsn_effi'])
    sel_sn['nsn_sum'] = np.cumsum(sel_sn['nsn'])
    sel_sn['nsn_effi_sum_err'] = np.sqrt(np.cumsum(sel_sn['nsn_effi_err']*sel_sn['nsn_effi_err']))
    sel_sn['ratio_nsn'] = sel_sn['nsn_effi_sum']/sel_sn['nsn_sum']
    sel_sn['ratio_nsn_err'] = sel_sn['nsn_effi_sum_err']/sel_sn['nsn_sum']
    
    ido = sel_sn['z_fit']<=0.5
    sel_norm = sel_sn[ido]
    norm = np.max(sel_sn['ratio_nsn'])
    
    sel_sn['ratio_nsn_norm']=sel_sn['ratio_nsn']/norm
    sel_sn['ratio_nsn_norm_p'] = sel_sn['ratio_nsn_norm']+sel_sn['ratio_nsn_err']
    sel_sn['ratio_nsn_norm_m'] = sel_sn['ratio_nsn_norm']-sel_sn['ratio_nsn_err']
    
    return df_nsn_rate,sel_sn
    
def plot_proc(grp,grp_effi,df_nsn_rate):
    """
    Function to perform some plot

    Parameters
    ----------
    grp : pandas df
        Data to plot.
    grp_effi : pandas df
        efficiencies to plot
    df_nsn_rate : pandas df
        nsn from rate.

    Returns
    -------
    None.

    """
    
    import matplotlib.pyplot as plt
    """
    #chisq LC fit
    fig, ax = plt.subplots()
    ax.hist(grp_sel['chisq_red'],histtype='step',bins=20)
    """
    fig, ax = plt.subplots()
    ll = grp.name
    vleg = "{} {} {}".format(ll[0],ll[1],ll[-1])
    
    fig.suptitle(vleg)
    ax.errorbar(grp_effi['z_fit'],grp_effi['effi'],yerr=grp_effi['effi_err'])
    axb = ax.twinx()
    nsn_tot_rate = np.max(np.cumsum(df_nsn_rate['nsn']))
    #axb.plot(grp_effi['z_fit'],np.cumsum(nsn)/nsn_tot,color='k')
    axb.plot(df_nsn_rate['z_fit'],np.cumsum(df_nsn_rate['nsn'])/nsn_tot_rate,color='r')
    axb.plot(df_nsn_rate['z_fit'],np.cumsum(df_nsn_rate['nsn_effi'])/nsn_tot_rate,color='b')
    ax.grid(visible='True')  
    
def plot_test(sel_sn,
              varx='z_fit',legx='z',
              vary='ratio_nsn_norm',legy='$N_{SN}$ ratio',
              vary_err='ratio_nsn_err',
              vary_p='ratio_nsn_norm_p',
              vary_m='ratio_nsn_norm_m',
              fig=None,ax=None,nsn_inside=0.98):
    """
    Function plot test

    Parameters
    ----------
    sel_sn : pandas df
        Data to plot.
    varx : str, optional
        x-axis variable. The default is 'z_fit'.
    legx : str, optional
        x-axis label. The default is 'z'.
    vary : str, optional
        y-axis variable. The default is 'ratio_nsn_norm'.
    legy : str, optional
        y-axis legend. The default is '$N_{SN}$ ratio'.
    vary_err : str, optional
        y-axis variable error. The default is 'ratio_nsn_err'.
    vary_p : str, optional
        y-axis variable+error. The default is 'ratio_nsn_norm_p'.
    vary_m : str, optional
        y-axis variable-error. The default is 'ratio_nsn_norm_m'.
    fig : matplotlib figure, optional
        plot figure. The default is None.
    ax : matplotlib axis, optional
        axis for the plot. The default is None.
    nsn_inside : float, optional
        fraction of sn to estimate zlim. The default is 0.98.

    Returns
    -------
    None.

    """
    
    if fig is None:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        
    ax.errorbar(sel_sn[varx],
                sel_sn[vary],
                yerr=sel_sn[vary_err])
    ax.fill_between(sel_sn[varx],sel_sn[vary_p],sel_sn[vary_m],color='yellow')   
    
    ax.plot([0.1,1.1],[nsn_inside]*2)
    
    ax.grid()
    
    plt.show()
    
def get_zlim_from_df(gra,thedict = dict(zip(['zlim','zlim_p','zlim_m'],
                       ['ratio_nsn_norm',
                        'ratio_nsn_norm_p',
                        'ratio_nsn_norm_m'])),nsn_inside=0.98):
    """
    Estimate zlim from df data

    Parameters
    ----------
    gra : pandas df
        Data to process.
    thedict : dict, optional
        Dict of parameters. 
        The default is dict(zip(['zlim','zlim_p','zlim_m'],
                                ['ratio_nsn_norm',
                                 'ratio_nsn_norm_p','ratio_nsn_norm_m'])).
    nsn_inside : float, optional
        frac of SNe Ia to estimate zlim. The default is 0.98.

    Returns
    -------
    res : pandas df
        Output data.

    """
    
    res = pd.DataFrame()
    if len(gra)< 3:
        return res
    zlim = {}
    #import matplotlib.pyplot as plt
    for key, vals in thedict.items():
        myinterp = interp1d(gra[vals],gra['z_fit'],
                            bounds_error=False, fill_value=0.)
        zlim['{}_{}'.format(key,nsn_inside)] = [myinterp(nsn_inside)]
      
    res = pd.DataFrame.from_dict(zlim)
       
    return res
    
    
def get_nsn_from_rate(grp,effis,zmin,zmax,dz):
    """
    Function to estimate the number of SNe Ia from rate

    Parameters
    ----------
    grp : pandas df
        Data to process.
    effis : pandas df
        Efficiency vs z.
    zmin : float
        min redshift.
    zmax : float
        max redshift.
    dz : float
        delta redshift.

    Returns
    -------
    df_nsn : pandas df
        nsn from rate+efficiency.

    """
    
    season_length = grp['season_length'].mean()
    survey_area = grp['survey_area'].mean()
    zz, rateInterp, rateInterp_err = getRates(zmin=zmin, zmax=zmax, dz=dz,
                                              survey_area=survey_area,
                                              season_length=season_length)
    # interpolate efficiency vs z
    effiInterp = interp1d(effis['z_fit'], effis['effi'], kind='linear',
                bounds_error=False, fill_value=0.)
    # interpolate variance efficiency vs z
    effiInterp_err = interp1d(effis['z_fit'], effis['effi_err'], kind='linear',
        bounds_error=False, fill_value=0.)
    
    nsn = effiInterp(zz)*rateInterp(zz)
    # get errors
    nsn_err = []
    for i in range(len(zz)):
        siga = effiInterp_err(zz[:i+1])*rateInterp(zz[:i+1])
        # sigb = effiInterp(zplot[:i+1])*rateInterp_err(zplot[:i+1])
        sigb = 0.
        nsn_err.append(np.sqrt(np.sum(siga**2 + sigb**2)))
    
    df_nsn = pd.DataFrame(zz, columns=['z_fit'])
    df_nsn['nsn'] = rateInterp(zz)
    df_nsn['nsn_effi'] = nsn
    df_nsn['nsn_effi_err'] = nsn_err
    
    return df_nsn
    

parser = OptionParser(description='Script to analyze SN selection criteria')

parser.add_option('--dbDir', type=str,
                  default='../sn_fmb_confe',
                  help='OS location dir[%default]')
parser.add_option('--dbName', type=str,
                  default='baseline_v5.3.0_10yrs',
                  help='OS name [%default]')
parser.add_option('--runType', type=str,
                  default='DDF_spectroz',
                  help='run type [%default]')
parser.add_option('--timescale', type=str,
                  default='season',
                  help='timescale [%default]')
parser.add_option('--fields', type=str,
                  default='COSMOS,CDFS,XMM-LSS,ELAISS1,EDFS_a,EDFS_b',
                  help='fields to process [%default]')
parser.add_option("--selconfig", type=str,
                  default='G10_JLA', help="sel config name [%default]")
parser.add_option('--outDir', type=str,
                  default='../zlim',
                  help='output Dir dir[%default]')
parser.add_option('--x1', type=str,
                  default='-2.0',
                  help='SN Ia stretch value [%default]')
parser.add_option('--color', type=str,
                  default='0.2',
                  help='SN Ia color value [%default]')
parser.add_option('--nproc', type=int,
                  default=8,
                  help='nproc for multiprocessing [%default]')
opts, args = parser.parse_args()

dbDir = opts.dbDir
dbName = opts.dbName
runType = opts.runType
runType = opts.runType
timescale = opts.timescale

fields = opts.fields.split(',')
selconfig = opts.selconfig
outDir = opts.outDir
x1 = opts.x1
color = opts.color
nproc = opts.nproc

# create output dir (if necessary)
checkDir(outDir)

# selection criteria
sellist = selection_criteria()[selconfig]

# add criteria
#sellist.append(('Nfilt_2', operator.ge, 3, 7))
# sellist.append(('Nfilt_5', operator.ge, 2, 7))
# sellist.append(('sigmaC', operator.le, 0.04, 7))

print(sellist)

df_zlim = process_db(dbDir, dbName, runType, fields,x1,color,
                     nproc,sellist=sellist)

# save the data
outName = '{}/zlim_{}.hdf5'.format(outDir, dbName)
df_zlim.to_hdf(outName, key='zlim')
