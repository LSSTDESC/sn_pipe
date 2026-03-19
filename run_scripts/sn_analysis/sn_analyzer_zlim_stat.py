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

def load_data(dbDir, dbName, runType, field,nproc=8):
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
    nproc: int,optional.
        number of proc to use. The default is 8.

    Returns
    -------
    df : pandas df
        loaded data.

    """

    theDir = '{}/{}/{}'.format(dbDir, dbName, runType)

    print('scanning', theDir)
    fis = glob.glob('{}/*{}*.hdf5'.format(theDir, field))

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

def zlim_field(field,dbDir,dbName,runType):
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

    Returns
    -------
    pandas df
        output data.

    """
    
    time_ref = time.time()
    data = load_data(dbDir, dbName, runType, field)
    print('loaded',time.time()-time_ref)
    data['field'] = field
    data = complete_df(data)

    params = {}
    
    params['data'] = data
    from sn_tools.sn_utils import multiproc
    
    obs = data['healpixID'].unique().tolist()
    
    #obs = [143706.0]
    res = multiproc(obs,params,zlim_field_multiproc,nproc=8)
    
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
    
    df_zlim = sel_data.groupby(['field','healpixID','pixRA','pixDec','season']).apply(
     lambda x: process_season_pixel(x, sellist,plot=False), 
       include_groups=False).reset_index()
    
    if output_q is not None:
        return output_q.put({j: df_zlim})
    else:
        return df_zlim
    
def process_db(dbDir, dbName, runType, fields,sellist=None):
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
    sellist: dict, optional.
        slection criteria. The default is None.

    Returns
    -------
    df_effi : pandas df
        processed data.

    """

    df_zlim = pd.DataFrame()

    for field in fields:
        
        df_zlim_ = zlim_field(field, dbDir, dbName, runType)
        df_zlim = pd.concat((df_zlim,df_zlim_))

    return df_zlim

def process_season_pixel(grp, sellist,plot=False):
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
    fig.suptitle(grp.name)
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
                  default='../Output_SN_DD_sigmaInt_0.0_Hounsell_z_smflux_notelrot_airmass_zfaint',
                  help='OS location dir[%default]')
parser.add_option('--dbName', type=str,
                  default='baseline_v5.0.0_10yrs',
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

opts, args = parser.parse_args()

dbDir = opts.dbDir
dbName = opts.dbName
runType = opts.runType
runType = opts.runType
timescale = opts.timescale

fields = opts.fields.split(',')
selconfig = opts.selconfig
outDir = opts.outDir

# create output dir (if necessary)
checkDir(outDir)

# selection criteria
sellist = selection_criteria()[selconfig]

# add criteria
#sellist.append(('Nfilt_2', operator.ge, 3, 7))
# sellist.append(('Nfilt_5', operator.ge, 2, 7))
# sellist.append(('sigmaC', operator.le, 0.04, 7))

#print(sellist)

df_zlim = process_db(dbDir, dbName, runType, fields,sellist=sellist)

# save the data
outName = '{}/zlim_{}.hdf5'.format(outDir, dbName)
df_zlim.to_hdf(outName, key='zlim')
