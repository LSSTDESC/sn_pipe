#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 16:42:18 2025

@author: philippe.gris@clermont.in2p3.fr
"""
from optparse import OptionParser
import pandas as pd
import matplotlib.pyplot as plt
from sn_plotter_metrics.plot4metric import multiplot_dist
from sn_plotter_metrics.plot4metric import plotMollview_seasons
import numpy as np
import glob


def plot_var_mean(datam, figtitle='', varx='season',
                  legx='season', vary='cadence',
                  legy='cadence [day]', plot_mean=True):
    """
    Function to make a plot and sumperimpose means

    Parameters
    ----------
    datam : pandas df
        Data to process.
    figtitle : str, optional
        figure title. The default is ''.
    varx : str, optional
        x-axis variable. The default is 'season'.
    legx : str, optional
        x-axis label. The default is 'season'.
    vary : str, optional
        y-axis variable. The default is 'cadence'.
    legy : str, optional
        y-axis label. The default is 'cadence [day]'.
    plot_mean : bool, optional
        To superimpose the <y>. The default is True.

    Returns
    -------
    None.

    """

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.suptitle(figtitle)

    idx = datam[vary] > 0
    data = datam[idx]

    ax.plot(data[varx], data[vary], 'k.', ms=8, mfc='None')

    if plot_mean:
        vv = data.groupby([varx])[vary].mean().reset_index()
        vvb = data.groupby([varx])[vary].std()
        vstd = f'{varx}_std'
        vv[vstd] = vvb.to_list()
        ax.errorbar(vv[varx], vv[vary],
                    yerr=vv[vstd], color='r')

    ax.set_xlabel(r'{}'.format(legx))
    ax.set_ylabel(r'{}'.format(legy))
    ax.set_ylim([0, None])
    ax.grid(visible=True)


def multiplot_season(sel, varx, legx, vary, legy):
    """
    plots of vary vs varx

    Parameters
    ----------
    sel : pandas df
        Data to plot.
    varx : str
        x-axis variable.
    legx : str
        x-axis legend.
    vary : str
        y-axis variable.
    legy : str
        y-axis legend.

    Returns
    -------
    None.

    """

    fields = sel['field'].unique()
    dbName = sel['dbName'].unique()[0]

    bands = 'ugrizy'
    for field in fields:
        idx = sel['field'] == field
        selb = sel[idx]
        figtitle = f'{dbName} - {field}'
        plot_var_mean(selb, figtitle=figtitle, varx=varx,
                      legx=legx, vary=vary, legy=legy)
        for b in bands:
            vvary = f'{vary}_{b}'
            figtitle = f'{dbName} - {field} - {b} band'
            plot_var_mean(selb, figtitle=figtitle, varx=varx,
                          legx=legx, vary=vvary, legy=legy)


def print_pixel_info(sel, healpixID):
    """
    Function to grab pixel info

    Parameters
    ----------
    sel : pandas df
        Data to process.
    healpixID : int
        healpix ID.

    Returns
    -------
    None.

    """

    idxb = sel['healpixID'] == healpixID
    selnc = sel[idxb]
    print(selnc[['healpixID', 'pixRA', 'pixDec', 'nvisits', 'cadence', 'season']])


def load_data(dbDir, dbName, fields, fieldType='DD'):
    """
    Function to load the data

    Parameters
    ----------
    dbDir : str
        data main directory.
    dbName : str
        OS to analyze.
    fields : list(str)
        list of fields to process.
    fieldType : str, optional
        type of field (DD/WFD). The default is 'DD'.

    Returns
    -------
    df : pandas df
        output data.

    """

    theDir = '{}/{}'.format(dbDir, dbName)
    prefix = '{}_pixels_{}'.format(fieldType, dbName)

    # grab the list of data
    list_data = []
    if fieldType == 'DD':
        # loop on ddfs
        for field in fields:
            fName = '{}/{}_{}*.hdf5'.format(theDir, prefix, field)
            fis = glob.glob(fName)
            if len(fis) == 0:
                print('data not found', fName)
            list_data += fis
    if fieldType == 'WFD':
        fName = '{}/{}_*.hdf5'.format(theDir, prefix)
        fis = glob.glob(fName)
        if len(fis) == 0:
            print('data not found', fName)
        list_data += fis

    # load the data
    df = pd.DataFrame()
    for fi in list_data:
        dd = pd.read_hdf(fi)
        df = pd.concat((df, dd))

    return df
    
def plot_mollviews(sel,seasons,var_to_plot):
    """
    Function to display Mollweid plots for a set of variavles

    Parameters
    ----------
    sel : TYPE
        DESCRIPTION.
    seasons : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    for vv in var_to_plot:
        plot_mollviews_db(sel,vv,seasons,yleg=dict_leg[vv])
       
def plot_mollviews_db(sel,vv,seasons,yleg='',timescale='year'):
    """
    Function to display mollview plots for OS

    Parameters
    ----------
    sel : pandas df
        Data to plot.
    vv : str
        variable to plot.
    seasons : list(int) or list(str)
        seasons to plot.
    yleg : str, optional
        y-axis legend. The default is ''.
    timescale : str, optional
        Time scale to use. The default is 'year'.

    Returns
    -------
    None.

    """
    
    
    if seasons == '10yrs':
        if vv.split('_')[0] =='nvisits':
            sel = sel.groupby(['healpixID','dbName'])[vv].sum().reset_index()
        if vv.split('_')[0] =='cadence':
            idx = sel[vv] < 30
            idx &= sel[vv] > 0
            sel = sel[idx].groupby(['healpixID','dbName'])[vv].median().reset_index()
            
        sel[timescale] = '10 yr survey'
        seasons = ['10 yr survey']
    
    if vv.split('_')[0] =='nvisits':
        op = np.mean
        
    if vv.split('_')[0] =='cadence':
        op = np.median
    
    dbNames = sel['dbName'].unique()
     
    for dbName in dbNames:
        idx = sel['dbName'] == dbName
        selb=sel[idx]
        plotMollview_seasons(nside, selb, dbName,
                             yvar=vv, yleg=dict_leg[vv],
                             op=op, seasons=seasons,outDir=mollview_outDir)
    ## add the diff
    
    if len(dbNames) ==2:
        dd = {}
        for i,dbName in enumerate(dbNames):
            idxa = sel['dbName'] == dbName
            dd[i] = pd.DataFrame(sel[idxa])
            
        ddb = dd[0].merge(dd[1],left_on=['healpixID',timescale],
                          right_on=['healpixID',timescale])
        newvar = 'delta_{}'.format(vv)
        ddb[newvar] = ddb['{}_x'.format(vv)]-ddb['{}_y'.format(vv)]
        dbNa = ddb['dbName_x'].unique()[0]
        dbNb = ddb['dbName_y'].unique()[0]
        dbName = '{}-{}'.format(dbNa,dbNb)
        #ddb = ddb.fillna(0)
        #ddb[timescale] = '10 yr survey'
        print('test',ddb[timescale].unique())
        plotMollview_seasons(nside, ddb, dbName,
                             yvar=newvar, yleg='$\Delta$ '+dict_leg[vv],
                             op=op, seasons=seasons,
                             outDir=mollview_outDir,themin=np.min(ddb[newvar]))
    
def high_nvisits_pixels(sel,varx='nvisits',thresh=2000):
    """
    Function to grab the list of hot pixels

    Parameters
    ----------
    sel : pandas df
        Data to process.
    varx : str, optional
        x-axis variable. The default is 'nvisits'.
    thresh : int, optional
        min number of visits. The default is 2000.

    Returns
    -------
    selpix : list(int)
        list of healpixIDs.

    """
    
    
    sel = sel.groupby(['healpixID','dbName'])[varx].sum().reset_index()
    
   
    idx = sel['nvisits'] >= thresh
    selpix = sel[idx]['healpixID'].unique().tolist()
 
    return selpix
    
    
    

def plot_histos(hist_var,df,seasons,timescale='year'):
    """
    Function to plot histograms of data

    Parameters
    ----------
    hist_var : list(str)
        List of variables to plot.
    df : pandas df
        Data to plot.
    seasons : list(int) or list(str)
        seasons to plot.
    timescale : str, optional
        Time scale to use. The default is 'year'.
    Returns
    -------
    None.

    """
    
    for vv in hist_var:
        if seasons == '10yrs':
            if vv.split('_')[0] == 'nvisits':
                df = df.groupby(['healpixID','dbName'])[hist_var].sum().reset_index()
            if vv == 'nvisits_night':
                df = df.groupby(['healpixID','dbName'])[hist_var].mean().reset_index()
            if vv.split('_')[0] == 'cadence':
                idx= df[vv]<=30
                idx &= df[vv]>0
                df = df[idx]
                df = df.groupby(['healpixID','dbName'])[hist_var].median().reset_index()
                
            plot_histos_db(df,vv,figtit='10 years')
        else:
            for seas in seasons:
                idx = df[timescale] == seas
                sel = df[idx]
                plot_histos_db(sel,vv,figtit='year {}'.format(seas))
                
        
        
        
def plot_histos_db(df,varx,fig=None,ax=None,figtit=''):
    """
    Function to plot histos for OS

    Parameters
    ----------
    df : pandas df
        Data to plot.
    varx : str
        x-axis variable.
    fig : matplotlib figure, optional
        Figure for the plot. The default is None.
    ax : matplotlib axis, optional
        axis for the plot. The default is None.
    figtit : str, optional
        figure title. The default is ''.

    Returns
    -------
    None.

    """
    
    ccols = ['r','k','b']
    lls = ['solid','dashed','dotted']
    
    dbNames = df['dbName'].unique().tolist()
    colors = dict(zip(dbNames,ccols[0:len(dbNames)]))
    ls = dict(zip(dbNames,lls[0:len(dbNames)]))
    
    if fig is None:
        fig, ax = plt.subplots(figsize=(12, 8))

    if figtit != '':
        fig.suptitle(figtit)
        
    d_df = {}
    for dbName in dbNames:
           idx = df['dbName'] == dbName
           sel = df[idx]
           if len(dbNames) == 2:
               d_df[dbName] = sel
           ax.hist(sel[varx], histtype='step', 
                   bins=100,label=dbName,
                   linestyle=ls[dbName],color=colors[dbName])
    
    ax.set_xlabel(r'{}'.format(dict_leg[varx]))
    ax.set_ylabel(r'Number of entries')
    ax.grid(visible=True)
    ax.set_xlim([0., None])
    ax.legend(fontsize=15,frameon=False)
    
    #plot the diff here
    
    if len(dbNames) == 2:
        dba = dbNames[0]
        dbb = dbNames[1]  
        df_m = d_df[dba].merge(d_df[dbb],
                                  left_on=['healpixID'],
                                  right_on=['healpixID'])
        newvar = 'diff_{}'.format(varx) 
        df_m[newvar] = df_m['{}_x'.format(varx)]-df_m['{}_y'.format(varx)]
        figb, axb = plt.subplots(figsize=(12, 8))
        axb.hist(df_m[newvar], histtype='step', 
                   bins=50,
                   linestyle='solid',color='k')
        axb.grid(visible=True)
        xlab = '$\Delta$'+'{}'.format(dict_leg[varx])
        axb.set_xlabel(r'{}'.format(xlab))
        axb.set_ylabel(r'Number of entries')
        
    
    
   
    
def plot_nvisits_cumsum(sel,varx='nvisits'):
    
    dbNames = sel['dbName'].unique()
    
    #nvisits full survey
    
    ddb = sel.groupby(['dbName','healpixID'])[varx].sum().reset_index()
    
    fig, ax = plt.subplots()
    for dbName in dbNames:
        idx = ddb['dbName'] == dbName
        selb = ddb[idx]
        selb = selb.sort_values(by=[varx])
        nv = selb[varx].sum()
        ax.plot(selb[varx],selb[varx].cumsum()/nv)
        #ax.hist(sel)
        
    plt.show()
    
    
def tag_hotspot(df,ra=224,dec=-29,width=20.,varx='nvisits'):
    
    sel = df.groupby(['dbName','healpixID','pixRA','pixDec'])[varx].sum().reset_index()
    
    print(sel[['pixRA','pixDec']])
    idx = sel['pixRA'] >= ra-width
    print(len(sel[idx]),ra-width,ra+width)
    idx &= sel['pixRA'] <= ra+width
    idx &= sel['pixDec'] >= dec-width
    idx &= sel['pixDec'] <= dec+width
    
    sel = sel[idx]
    
    
    dbNames = sel['dbName'].unique()
    
    fig, ax = plt.subplots()
    
    for dbName in dbNames:
        idx = sel['dbName'] == dbName
        selb = sel[idx]
        
        ax.hist(selb[varx],histtype='step')
        
    plt.show()
    
    
    
    
    
    
parser = OptionParser(description='Script to plot pixel level OS infos')

parser.add_option('--dbName', type=str, default='test_newb',
                  help='dbName to process [%default]')
parser.add_option('--dbDir', type=str, default='../test_metric',
                  help='dbDir of the OS to process [%default]')
parser.add_option('--nside', type=int, default=128,
                  help='healpix nside parameter [%default]')
parser.add_option('--plots', type=str,
                  default='gen_plots,mollview,hist',
                  help='plots to show [%default]')
parser.add_option('--seasons', type=str,
                  default='1-5',
                  help='seasons to show [%default]')
parser.add_option('--fields', type=str,
                  default='COSMOS,XMM-LSS,CDFS,ELAISS1,EDFS_a,EDFS_b',
                  help='fields to show [%default]')
parser.add_option('--fieldType', type=str,
                  default='DD',
                  help='type of field to process (DD/WFD) [%default]')
parser.add_option('--timescale', type=str,
                  default='year',
                  help='time scale for the plots (year/season) [%default]')
parser.add_option('--mollview_var', type=str,
                  default='cadence,nvisits',
                  help='var to plot in Mollview [%default]')
parser.add_option('--gen_var', type=str,
                  default='cadence_year,nvisits_year,cadence_dist,nvisits_dist',
                  help='gen var to plot [%default]')
parser.add_option('--hist_var', type=str,
                  default='nvisits_10yrs,cadence_year',
                  help='hist var to plot [%default]')
parser.add_option('--mollview_outDir', type=str,
                  default='None',
                  help='output dir for mollview figures [%default]')
parser.add_option('--nvisits_10yrs_min', type=int,
                  default=1200,
                  help='min nvisits after 10 yrs (to remove hot spots) [%default]')
parser.add_option('--dust', type=int,
                  default=0,
                  help='to apply E(B-V) cut [%default]')

opts, args = parser.parse_args()

dbDir = opts.dbDir
dbNames = opts.dbName.split(',')
nside = opts.nside
plots = opts.plots.split(',')
mollview_var = opts.mollview_var.split(',')
gen_var = opts.gen_var.split(',')
seasons = opts.seasons
hist_var = opts.hist_var.split(',')
fields = opts.fields.split(',')
fieldType = opts.fieldType
timescale = opts.timescale
mollview_outDir = opts.mollview_outDir
nvisits_10yrs_min=opts.nvisits_10yrs_min
dust = opts.dust

if '-' in seasons:
    cad_brk = seasons.split('-')
    seasons = list(range(int(cad_brk[0]), int(cad_brk[1])+1))
else:
    if ',' in seasons:
        seasons = list(map(int, seasons.split(',')))
    else:
        if seasons != '10yrs':
            seasons = [int(seasons)]

df = pd.DataFrame()
for dbName in dbNames:
    df_ = load_data(dbDir, dbName, fields, fieldType=fieldType)
    df_['dbName'] = dbName
    df = pd.concat((df,df_))

#load the dust_map
if dust:
    fDust = 'reference_files/dustmap_{}_delta_mag_dust.hdf5'.format(nside)
    df_dust = pd.read_hdf(fDust)

    df = df.merge(df_dust,left_on=['healpixID'],right_on=['healpixID'])

    #select on dust

    idx = df['ebvofMW'] < 0.25
    df = df[idx]

print('nentries',len(df))

if mollview_outDir != 'None':
    from sn_tools.sn_io import checkDir
    checkDir(mollview_outDir)

idx = df[timescale] > 0
idx &= df[timescale] < 11
idx &= df['cadence'] > 0.
sel = df[idx]

#tag_hotspot(sel)

#plot_nvisits_cumsum(sel)

#remove hot spots
hot_pixels = high_nvisits_pixels(sel,thresh=nvisits_10yrs_min)

idx = sel['healpixID'].isin(hot_pixels)

hot_spots = sel[idx]

sel = sel[~idx]

bands = 'ugrizy'
vvar = ['cadence', 'nvisits', 'm5_i','nvisits_night']
for b in bands:
    vvar.append('cadence_{}'.format(b))
    vvar.append('nvisits_{}'.format(b))
    
legvar = ['cadence [day]', '$\Sigma N_{visits}$/pixel', 
          '$m_{5}^{i}$','<N$_{visits}$>/night/pixel']

for b in bands:
    legvar.append('cadence {} [day]'.format(b))
    legvar.append('$\Sigma N_{visits}^{'+b+'}$/pixel')
    
dict_leg = dict(zip(vvar, legvar))

if 'gen_plots' in plots:
    for vv in gen_var:
        vary = vv.split('_')[0]
        varx = vv.split('_')[1]
        if varx == timescale:
            multiplot_season(sel, varx=timescale, legx=timescale,
                             vary=vary, legy=dict_leg[vary])
        if varx == 'dist':
            multiplot_dist(sel, yvar=vary,
                           yleg=r'{}'.format(dict_leg[vary]), timescale=timescale)

if 'mollview' in plots:
    plot_mollviews(sel,seasons,mollview_var)
            

if 'hist' in plots:
    plot_histos(hist_var,sel,seasons)
   
    

"""
print_pixel_info(sel, 109384)
print_pixel_info(sel, 109031)
"""

plt.show()
