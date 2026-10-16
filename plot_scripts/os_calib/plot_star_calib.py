#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 13:47:17 2026

@author: philippe.gris@clermont.in2p3.fr
"""

from optparse import OptionParser
import numpy as np
import pandas as pd
from sn_plotter_analysis import plt
from sn_tools.sn_obs import get_fields
from sn_tools.sn_utils import clean_level
import healpy as hp

def plot_hist(df,xvar='dist [deg]',legx='distance [deg]',figtit=''):
    """
    Function to plot histogram

    Parameters
    ----------
    df : array
        Data to plot.
    xvar : str, optional
        x-axis variable. The default is 'dist [deg]'.
    legx : str, optional
        x-axis legend. The default is 'distance [deg]'.
    figtit : str, optional
        figure title. The default is ''.

    Returns
    -------
    None.

    """
    
    fig, ax = plt.subplots(figsize=(15,9))
    if figtit != '':
        fig.suptitle(figtit)
    
    ax.hist(df[xvar],histtype='step',bins=20,lw=2)
    
    ax.grid(visible='True')
    
    ax.set_ylabel(r'Number of entries')
    ax.set_xlabel(r'{}'.format(legx))

def plot_hist_ddf(df,xvar='dist [deg]',
              legx='distance [deg]',figtit=''):
    """
    Function to plot histogram - loop on ddfs

    Parameters
    ----------
    df : array
        Data to plot.
    xvar : str, optional
        x-axis variable. The default is 'dist [deg]'.
    legx : str, optional
        x-axis legend. The default is 'distance [deg]'.
    figtit : str, optional
        figure title. The default is ''.

    Returns
    -------
    None.

    """    
    fig, ax = plt.subplots(figsize=(15,9))
    if figtit != '':
        fig.suptitle(figtit)
    
    fields = np.unique(df['field'])
    
    for field in fields:
        idx = df['field'] == field
        sel = df[idx]
        star_name = np.unique(sel['Star_name'])[0]
        
        label = '{}/{}'.format(field,star_name)
        n,bins,patches = ax.hist(sel[xvar],histtype='step',bins=20,lw=2)
        print(field,star_name,np.mean(sel[xvar]),np.max(n))
        x =np.mean(sel[xvar])
        y = np.max(n)
        if field == 'COSMOS':
            x-=3
        if field == 'EDFS_a':
            x-=4
        if field == 'EDFS_b':
            x+=1
            y-= 100
        ax.text(x,y,label,fontsize=12)
    
    ax.grid(visible='True')
    
    ax.set_ylabel(r'Number of entries')
    ax.set_xlabel(r'{}'.format(legx))
    
    
def ana_night(grp):
    """
    Function to analyze nights

    Parameters
    ----------
    grp : pandas df
        Data to process.

    Returns
    -------
    dd : pandas df
        processed data.

    """
    
    #dd = grp.groupby(['Star_name']).size().to_frame('n').reset_index()
    dd = grp.groupby(['Star_name']).apply(lambda x:get_info(x),include_groups=False).reset_index()
    
    dd['frac'] = 100.*dd['n']/dd['n'].sum()
   
    dd = clean_level(dd)
    
    
    return dd

def get_info(grp):
    """
    Function to grab infos

    Parameters
    ----------
    grp : pandas df
        Data to process.

    Returns
    -------
    res : pandas df
        output data.

    """

    rr = {}
    
    rr['n'] = [len(grp)]
    col = 'dist [deg]'
    rr[col] = [grp[col].median()]
    
    res = pd.DataFrame.from_dict(rr)
    
    return res
    

def plot_nstars(df,xvar='night',xleg='night',
                yvar='frac',yleg='frac',label='',
                fig=None,ax=None,labelrotation=0,labelsize=15):
    """
    Function to perform a bi-plot

    Parameters
    ----------
    df : pandas df
        Data to process.
    xvar : str, optional
        x-axis variable. The default is 'night'.
    xleg : str, optional
        x-axis legend. The default is 'night'.
    yvar : str, optional
        y-axis variable. The default is 'frac'.
    yleg : str, optional
        y-axis legend. The default is 'frac'.
    label : str, optional
        plot label. The default is ''.
    fig : matplotlib figure, optional
        plot figure. The default is None.
    ax : matplotlib axis, optional
        axis figure. The default is None.
    labelrotation : float, optional
        label rotation angle. The default is 0.
    labelsize : float, optional
        label size. The default is 15.

    Returns
    -------
    None.

    """
    
    if fig is None:
        fig, ax = plt.subplots(figsize=(12,9))

    df = df.sort_values(by=xvar)
    ax.plot(df[xvar],df[yvar],
            label=label)
        
    if xleg != '':
        ax.set_xlabel(r'{}'.format(xleg))
    if yleg != '':
        ax.set_ylabel(r'{}'.format(yleg))
    
    ax.tick_params(axis='x',
                   labelrotation=labelrotation,
                   labelsize=labelsize)

    ax.legend()
    ax.grid(visible=True)
        
def ddf_wfd(obs,nside=64):
    """
    Function to get ddf and wfd obs

    Parameters
    ----------
    obs : array
        observations.
    nside : int, optional
        healpix nside parameter. The default is 64.

    Returns
    -------
    ddf : array
        DDF obs.
    wfd : array
        WFD obs.

    """
    
    # get DDFs
    ddf = get_fields(obs,'input/simulation/lookup_ddf.csv')  
    
    cols = 'observationId'  
    obsids = ddf[cols].tolist()
    
    idx = np.isin(obs[cols],obsids)
    
    wfd = obs[~idx]
    
    #get pixels
    wfd=get_pixels(wfd,nside)
    
    #add ebvofMW
    wfd = get_ebv(wfd,
                  ebv_map='reference_files/dustmap_64_delta_mag_dust.hdf5')
    
    return ddf,wfd

def get_pixels(obs,nside):
    """
    Function to get pixels

    Parameters
    ----------
    obs : array
        Data to process.
    nside : int
        healpix nside parameter.

    Returns
    -------
    obs : pandas df
        original array+healpixID,pixRA, pixDec columns.

    """
    
    obs = pd.DataFrame.from_records(obs)
    #set pixelIds
    healpixIDs = hp.ang2pix(nside, obs['RA'],
                               obs['Dec'], nest=True, lonlat=True)
    
    # get pixel coordinates
    coords = hp.pix2ang(nside, healpixIDs,
                        nest=True, lonlat=True)
    pixRA, pixDec = coords[0], coords[1]
    
    obs['healpixID'] = healpixIDs
    obs['pixRA'] = pixRA
    obs['pixDec'] = pixDec
    """
    This is too slow
    import numpy.lib.recfunctions as rf
    obs = rf.append_fields(obs, 'healpixID',healpixIDs)
    obs = rf.append_fields(obs, 'pixRA',pixRA)
    obs = rf.append_fields(obs, 'pixDec',pixDec)   
    """
    return obs

def get_ebv(df,ebv_map):
    """
    Function to grab ebvofMW

    Parameters
    ----------
    df : pandas df
        input data.
    ebv_map : str
        file of ebv map.

    Returns
    -------
    res : array
        original array+ebvofMW value.

    """

    #load ebv map
    ebv = pd.read_hdf(ebv_map)
    
    #transform obs to df
    #df = pd.DataFrame.from_records(obs)
    
    #merge with ebv file
    
    cols = ['healpixID','ebvofMW']
    ebv = ebv[cols]
    df = df.merge(ebv, left_on=[cols[0]],right_on=[cols[0]])
    
    #return as record arrar
    
    res = df.to_records(index=False)
    del df
    
    return res
    
"""
def re_frac(grp):

    grp['frac'] = 100.*grp['n']/grp['n'].sum()

    return grp       
"""
def count_frac(grp,values=[500]):
    """
    Function to count frac of and nstars depending on min obs values

    Parameters
    ----------
    grp : pandas df
        Data to process.
    values : list(int), optional
        list of min obs. The default is [500].

    Returns
    -------
    res : TYPE
        DESCRIPTION.

    """
    
    nobs_tot = grp['n'].sum()
    
    r = []
    for vv in values:
        idxx = grp['n'] > vv
        sel = grp[idxx]
        nobs_sel = sel['n'].sum()
        nstars = len(sel)
        r.append((vv,nstars,100.*nobs_sel/nobs_tot))
    
    res = pd.DataFrame(r,columns=['nobs','nstars','frac_obs'])

    return res

def plot_loop(dfc,loop_var='year',xvar='Star_name',xleg='',
              yvar='n',yleg='Nobs',fig=None,ax=None,
              labelrotation=0,labelsize=20):
    """
    Function to perform a loop of plots

    Parameters
    ----------
    dfc : pandas df
        Data to process.
    loop_var : str, optional
        loop variable. The default is 'year'.
    xvar : str, optional
        x-axis var. The default is 'Star_name'.
    xleg : str, optional
        x-axis label. The default is ''.
    yvar : str, optional
        y-axis var. The default is 'n'.
    yleg : str, optional
        y-axis label. The default is 'Nobs'.
    fig : matplotlib figure, optional
        Figure for the plot. The default is None.
    ax : matplotlib axis, optional
        axis for the plot. The default is None.
    labelrotation : int, optional
        label rotation angle. The default is 0.
    labelsize : int, optional
        label size. The default is 20.

    Returns
    -------
    None.

    """
    
    years = dfc[loop_var].unique()

    for year in years:
        idx = dfc[loop_var] == year
        sel = dfc[idx]

        plot_nstars(sel,xvar=xvar,xleg=xleg,yvar=yvar,yleg=yleg,
                label='{}={}'.format(loop_var,year),
                fig=fig,ax=ax,
                labelrotation=labelrotation,
                labelsize=labelsize)
    
    

parser = OptionParser(description='figures related to obs and calib stars')

parser.add_option('--dataDir', type=str, default='../DB_Files_star_calib',
                  help='data dir [%default]')
parser.add_option('--dbName', type=str, default='baseline_v5.3.0_10yrs_star_calib',
                  help='OS name [%default]')
parser.add_option('--dbExtens', type=str, default='npy',
                  help='OS extens  [%default]')

opts, args = parser.parse_args()

dataDir = opts.dataDir
dbName = opts.dbName
dbExtens = opts.dbExtens

#load the data
fName = '{}/{}.{}'.format(dataDir,dbName,dbExtens)
obs = np.load(fName,allow_pickle=True)

ddf,wfd = ddf_wfd(obs)

#ebvofMW cut (wfd)
idx = wfd['ebvofMW'] <= 0.25
wfd = wfd[idx]

#plot dist hist

figtit = '{} \n WFD'.format(dbName)
plot_hist(wfd,figtit=figtit)

figtit = '{} \n DDF'.format(dbName)
plot_hist_ddf(ddf,figtit=figtit)


# wfd stats
df_obs = pd.DataFrame.from_records(wfd)

dfb = df_obs.groupby(['night']).apply(lambda x: ana_night(x),include_groups=False).reset_index()
dfb = clean_level(dfb)

#estimate years
dfb['year'] = dfb['night']/365+1.
dfb['year'] = dfb['year'].astype(int)


#nobs vs starname - per year
dfc = dfb.groupby(['Star_name','year'])['n'].sum().reset_index()

plot_loop(dfc,labelsize=12,labelrotation=270)


## plot frac_obs vs nstars
values = range(0,10000,10)
dfd = dfc.groupby(['year']).apply(lambda x: count_frac(x,values),include_groups=False).reset_index()

fig, ax = plt.subplots(figsize=(12,8))

idx = dfd['year'] <= 10
dfd = dfd[idx]

plot_loop(dfd,loop_var='year',
          xvar='nstars',xleg='$N_{calspec}$',
          yvar='frac_obs',yleg='frac_obs',fig=fig,ax=ax)
plt.show()