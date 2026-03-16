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
from sn_analysis.sn_tools import complete_df, get_pulls
from sn_analysis.sn_fit_tools import fit_linear,lin
from sn_analysis.sn_calc_plot import effi
import numpy as np
import re
import operator
from sn_tools.sn_io import checkDir
from sn_tools.sn_utils import multiproc
import time
from sn_analysis.sn_nsn_effi import getRates
from scipy.interpolate import interp1d

def load_data(dbDir, dbName, runType, field,nproc=16):
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
    
    
    df = pd.DataFrame()

    for fi in toproc:
        df_ = pd.read_hdf(fi)
        df = pd.concat((df, df_))
        
    if output_q is not None:
            return output_q.put({j: df})
    else:
        return df
    


def get_nsn(vala, valb, norm_factor):
    """
    Function to grab infos

    Parameters
    ----------
    vala : int
        number of sn after sel.
    valb : int
        number of sn before sel.
    norm_factor : float
        normalization factor.

    Returns
    -------
    list
        DESCRIPTION.

    """

    effi = vala/valb

    err_effi = np.sqrt(effi*(1.-effi)/valb)

    nsn = int(effi*valb/norm_factor)

    err_nsn = int(err_effi*valb/norm_factor)

    effi *= 100.
    err_effi *= 100.

    res = [(nsn, err_nsn, np.round(effi, 1), np.round(err_effi, 1))]
    cols = ['nsn', 'err_nsn', 'effi', 'err_effi']

    return pd.DataFrame(res, columns=cols)


def select_str(res, list_sel):
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


def process_season(data, seas, field, norm_factor):
    """
    Function to process a season

    Parameters
    ----------
    data : pandas df
        Data to process.
    seas : int
        season number.
    field : str
        Field of interest.
    norm_factor : float
        normalization factor.

    Returns
    -------
    df_effi : pandas df
        processed data.

    """

    idx = data['season'] == seas
    mysel = data[idx]

    n_nosel = int(len(mysel)/norm_factor)
    print('no sel', n_nosel)
    ra = get_pulls(mysel)
    ra['sel_str'] = 'nosel'
    ra['field'] = field
    ra['season'] = seas
    # dfa = pd.concat((dfa, ra))

    ro = get_nsn(len(mysel), len(mysel), norm_factor)
    ro['sel_str'] = 'nosel'
    ro['field'] = field
    ro['season'] = seas
    # dfb = pd.concat((dfb, ro))
    # get_pulls(mysel)
    for i in range(1, len(sellist)+1):
        # ro = [field, int(seas)]
        mystr, sel = select_str(mysel, sellist[:i])
        rasel = get_pulls(sel)
        rasel['sel_str'] = mystr
        rasel['field'] = field
        rasel['season'] = seas
        ra = pd.concat((ra, rasel))
        rosel = get_nsn(len(sel), len(mysel), norm_factor)
        rosel['sel_str'] = mystr
        rosel['field'] = field
        rosel['season'] = seas
        # dfb = pd.concat((dfb, ro))
        ro = pd.concat((ro, rosel))

    # merge the two Dataframes

    df_effi = ro.merge(ra,
                       left_on=['field', 'season', 'sel_str'],
                       right_on=['field', 'season', 'sel_str'],
                       suffixes=['', ''])

    return df_effi

def process_season_field_pixel(grp, norm_factor):
    """
    Function to process a season

    Parameters
    ----------
    data : pandas df
        Data to process.
    norm_factor : float
        normalization factor.

    Returns
    -------
    df_effi : pandas df
        processed data.

    """

    print(test)
    n_nosel = int(len(grp)/norm_factor)
    print('no sel', n_nosel)
    ra = get_pulls(grp)
    ra['sel_str'] = 'nosel'
    # dfa = pd.concat((dfa, ra))
    print('hh', ra)
    ro = get_nsn(len(grp), len(grp), norm_factor)
    ro['sel_str'] = 'nosel'
    # dfb = pd.concat((dfb, ro))
    # get_pulls(mysel)
    for i in range(1, len(sellist)+1):
        # ro = [field, int(seas)]
        mystr, sel = select_str(grp, sellist[:i])
        rasel = get_pulls(sel)
        rasel['sel_str'] = mystr
        ra = pd.concat((ra, rasel))
        rosel = get_nsn(len(sel), len(grp), norm_factor)
        rosel['sel_str'] = mystr
        # dfb = pd.concat((dfb, ro))
        ro = pd.concat((ro, rosel))

    # merge the two Dataframes

    df_effi = ro.merge(ra,
                       left_on=['sel_str'],
                       right_on=['sel_str'],
                       suffixes=['', ''])

    return df_effi
def process_season_field_pixel_deprecated(grp, norm_factor):
    """
    Function to process a season

    Parameters
    ----------
    data : pandas df
        Data to process.
    norm_factor : float
        normalization factor.

    Returns
    -------
    df_effi : pandas df
        processed data.

    """

    n_nosel = int(len(grp)/norm_factor)
    print('no sel', n_nosel)
    ra = get_pulls(grp)
    ra['sel_str'] = 'nosel'
    # dfa = pd.concat((dfa, ra))
    print('hh', ra)
    ro = get_nsn(len(grp), len(grp), norm_factor)
    ro['sel_str'] = 'nosel'
    # dfb = pd.concat((dfb, ro))
    # get_pulls(mysel)
    for i in range(1, len(sellist)+1):
        # ro = [field, int(seas)]
        mystr, sel = select_str(grp, sellist[:i])
        rasel = get_pulls(sel)
        rasel['sel_str'] = mystr
        ra = pd.concat((ra, rasel))
        rosel = get_nsn(len(sel), len(grp), norm_factor)
        rosel['sel_str'] = mystr
        # dfb = pd.concat((dfb, ro))
        ro = pd.concat((ro, rosel))

    # merge the two Dataframes

    df_effi = ro.merge(ra,
                       left_on=['sel_str'],
                       right_on=['sel_str'],
                       suffixes=['', ''])

    return df_effi

def zlim_field(field,dbDir,dbName,runType):
    
    print('processing',field)
    time_ref = time.time()
    data = load_data(dbDir, dbName, runType, field)
    print('loaded',time.time()-time_ref)
    data['field'] = field
    data = complete_df(data)

    print('there',len(data),data['healpixID'].unique())
    print(data.columns)
    
    params = {}
    
    params['data'] = data
    from sn_tools.sn_utils import multiproc
    
    obs = data['healpixID'].unique().tolist()
    
    res = multiproc(obs,params,zlim_field_multiproc,nproc=1)
    
    return res
    """
    df_effi = data.groupby(['field','healpixID', 'season']).apply(
       lambda x: process_season_pixel(x, norm_factor,sellist,plot=False), include_groups=False).reset_index()
    
    return df_effi
    """
    
def zlim_field_multiproc(toproc, params, j=0, output_q=None):
    
    data = params['data']
    
    idx = data['healpixID'].isin(toproc)
    sel_data = pd.DataFrame(data[idx])
    
    df_zlim = sel_data.groupby(['field','healpixID', 'season']).apply(
     lambda x: process_season_pixel(x, norm_factor,sellist,plot=True), 
       include_groups=False).reset_index()
    
    if output_q is not None:
        return output_q.put({j: df_zlim})
    else:
        return df_zlim
    
    
    

def process_db(dbDir, dbName, runType, fields,
               norm_factor, zmin=0.01, zmax=1.1,sellist=None):
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
    norm_factor : float
        normalization factor.
    zmin: float, optional.
        redshift min for data. The default is 0.01.
    zmax: float, optional.
        redshift max for data. The default is 1.11.   

    Returns
    -------
    df_effi : pandas df
        processed data.

    """

    df_zlim = pd.DataFrame()

    for field in fields:
        
        df_zlim_ = zlim_field(field, dbDir, dbName, runType)
        df_zlim = pd.concat((df_zlim,df_zlim_))
        print(df_zlim.columns)
        
        print(test)
        """
        idxz = data['z'] >= zmin
        idxz &= data['z'] <= zmax

        data = data[idxz]
        # print(field, len(data), len(data)/norm_factor)

        seasons = data['season'].unique()

        idx = data['healpixID'] == 108958

        print('fff', data.columns)
        df_effi = data[idx].groupby(['healpixID', 'season']).apply(
            lambda x: process_season_field_pixel(x, norm_factor), include_groups=False).reset_index()
        """
        """
        for seas in seasons:
            # print('processing', zmin, zmax, seas)
            dd = process_season(data, seas, field, norm_factor)
            df_effi = pd.concat((df_effi, dd))
        """
    """
    df_effi['dbName'] = dbName
    df_effi['zmin'] = np.round(zmin, 2)
    df_effi['zmax'] = np.round(zmax, 2)
    df_effi['field'] = field
    """
    

    return df_zlim

def process_season_pixel(grp, norm_factor,sellist,plot=False):
    
    #get selected group
    grp_sel = select(grp,sellist)
    
    #get selection efficiencies
    deltab=0.1
    bins = np.arange(0.01,1.1+deltab,deltab)
    print('there man',len(grp_sel),len(grp))
    grp_effi=effi(grp,grp_sel,xvar='z_fit',bins=bins)
    
    """
    print('alors',grp_effi)
    print(test)
    nsn = grp_effi['nsn']

    nsn_tot = np.max(np.cumsum(nsn))
    """
    zmin=0.01
    zmax = 1.1
    dz = 0.001
    #get the number of SN from the rate of explosion
    df_nsn_rate, sel_sn=get_all_nsn_from_rate(grp,grp_effi, zmin, zmax, dz)
   
    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.hist(grp_sel['chisq_red'],histtype='step',bins=20)
        fig, ax = plt.subplots()
        ax.errorbar(grp_effi['z_fit'],grp_effi['effi'],yerr=grp_effi['effi_err'])
        axb = ax.twinx()
        nsn_tot_rate = np.max(np.cumsum(df_nsn_rate['nsn']))
        #axb.plot(grp_effi['z_fit'],np.cumsum(nsn)/nsn_tot,color='k')
        axb.plot(df_nsn_rate['z_fit'],np.cumsum(df_nsn_rate['nsn'])/nsn_tot_rate,color='r')
        axb.plot(df_nsn_rate['z_fit'],np.cumsum(df_nsn_rate['nsn_effi'])/nsn_tot_rate,color='b')
        ax.grid(visible='True')
        plt.show()
 
    """
    idx = grp_effi['z_fit']<=0.45
    rrx = grp_effi[idx]['z_fit']
    nsn_sel = grp_effi[idx]['nsn']
    rry = np.cumsum(nsn_sel)/nsn_tot
    
    coeff,cov = fit_linear(rrx,rry)
    
    zvals = np.arange(0.01,1.11,0.01)
    
    yvals = lin(zvals,*coeff)
    
    axb.plot(zvals,yvals,color='b')
    
    print(coeff)
    """
    """
    zmin=0.01
    zmax = 1.1
    dz = 0.001
    df_nsn_rate = get_nsn_from_rate(grp,grp_effi,zmin,zmax,dz)
    nsn_tot_rate = np.max(np.cumsum(df_nsn_rate['nsn']))
    """
   

    """
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
    """
    #axa.errorbar(sel_sn['z_fit'],sel_sn['ratio_nsn_norm'],yerr=sel_sn['ratio_nsn_err'])
    #axa.fill_between(sel_sn['z_fit'],sel_sn['ratio_nsn_norm_p'],sel_sn['ratio_nsn_norm_m'],color='yellow')
    nsn_accepted_loss=0.02
    nsn_inside = 1.-nsn_accepted_loss
    #axa.plot([0.1,1.1],[nsn_inside]*2)
    
    plot_test(sel_sn,nsn_inside=nsn_inside)
    thedict = dict(zip(['zlim','zlim_p','zlim_m'],
                       ['ratio_nsn_norm',
                        'ratio_nsn_norm_p',
                        'ratio_nsn_norm_m']))
    
    bins=np.arange(0.0,1.3,0.2)
    sel_sn['range'] = pd.cut(sel_sn['z_fit'],bins=bins,right=False,include_lowest=True)
    sel_sn['range'] = sel_sn['range'].astype(str)
    zlim_t = sel_sn.groupby('range').apply(lambda x: \
                                          get_zlim_from_df(x),include_groups=False).reset_index()
    print(sel_sn)
    print(zlim_t.max())
    
    return zlim_t
    """
    zlim = {}
    for key, vals in thedict.items():
        zlim[key] = interp1d(sel_sn[vals],sel_sn['z_fit'])
    
    for key, vals in zlim.items():
        print(key,vals(nsn_inside))
    """
    

def get_all_nsn_from_rate(grp,grp_effi,zmin,zmax,dz):
    
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
    
def plot_test(sel_sn,
              varx='z_fit',legx='z',
              vary='ratio_nsn_norm',legy='$N_{SN}$ ratio',
              vary_err='ratio_nsn_err',
              vary_p='ratio_nsn_norm_p',
              vary_m='ratio_nsn_norm_m',
              fig=None,ax=None,nsn_inside=0.98):
    
    if fig is None:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        
    ax.errorbar(sel_sn[varx],
                sel_sn[vary],
                yerr=sel_sn[vary_err])
    ax.fill_between(sel_sn[varx],sel_sn[vary_p],sel_sn[vary_m],color='yellow')   
    
    ax.plot([0.1,1.1],[nsn_inside]*2)
    
    ax.grid()
    
def get_zlim_from_df(gra,thedict = dict(zip(['zlim','zlim_p','zlim_m'],
                       ['ratio_nsn_norm',
                        'ratio_nsn_norm_p',
                        'ratio_nsn_norm_m'])),nsn_inside=0.98):
    
    res = pd.DataFrame()
    if len(gra)< 3:
        return res
    zlim = {}
    import matplotlib.pyplot as plt
    for key, vals in thedict.items():
        myinterp = interp1d(gra[vals],gra['z_fit'],
                            bounds_error=False, fill_value=0.)
        zlim[key] = [myinterp(nsn_inside)]
      
    res = pd.DataFrame.from_dict(zlim)
       
    return res
    
    
def get_nsn_from_rate(grp,effis,zmin,zmax,dz):
    
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
parser.add_option('--seasons', type=str,
                  default='1',
                  help='seasons/years to process [%default]')
parser.add_option('--fields', type=str,
                  default='COSMOS,CDFS,XMM-LSS,ELAISS1,EDFS_a,EDFS_b',
                  help='fields to process [%default]')
parser.add_option('--norm_factor', type=float,
                  default=30.,
                  help='normalization factor [%default]')
parser.add_option("--selconfig", type=str,
                  default='G10_JLA', help="sel config name [%default]")
parser.add_option("--zrange", type=int,
                  default=0, help="to process data per zrange [%default]")
parser.add_option('--outDir', type=str,
                  default='../effi_pull_stat',
                  help='output Dir dir[%default]')

opts, args = parser.parse_args()

dbDir = opts.dbDir
dbName = opts.dbName
runType = opts.runType
runType = opts.runType
timescale = opts.timescale
seasons = opts.seasons.split(',')
seasons = list(map(int, seasons))
fields = opts.fields.split(',')
norm_factor = opts.norm_factor
selconfig = opts.selconfig
zrange = opts.zrange
outDir = opts.outDir

# create output dir (if necessary)
checkDir(outDir)

# selection criteria
sellist = selection_criteria()[selconfig]

# add criteria
#sellist.append(('Nfilt_2', operator.ge, 3, 7))
# sellist.append(('Nfilt_5', operator.ge, 2, 7))
# sellist.append(('sigmaC', operator.le, 0.04, 7))

print(sellist)
rb = []
# dfa = pd.DataFrame()
# dfb = pd.DataFrame()

zmin = 0.0
zmax = 1.1
deltaz = 1.1

if zrange:
    deltaz = 0.10

zvals = np.arange(zmin, zmax, deltaz)

# zvals[0] += 0.01
print(zvals)

for vv in zvals:
    zmi = vv
    if zmi < 0.001:
        zmi = 0.01
    zma = vv+deltaz
    df_zlim = process_db(dbDir, dbName, runType, fields,
                         norm_factor, zmin=zmi, zmax=zma,sellist=sellist)


print(df_zlim)
"""
# save the data
outName = '{}/{}.hdf5'.format(outDir, dbName)
df_effi.to_hdf(outName, key='effi_pull')
"""