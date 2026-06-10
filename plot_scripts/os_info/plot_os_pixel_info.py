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

def plot_histos_deprecated(hist_var,df,seasons):
    """
    Function to plot histograms of data

    Parameters
    ----------
    hist_var : list(str)
        List of variables to plot.
    df : pandas df
        Data to plot.

    Returns
    -------
    None.

    """
    
    dbNames = df['dbName'].unique().tolist()
    ccols = ['r','k','b']
    lls = ['solid','dashed','dotted']
    colors = dict(zip(dbNames,ccols[0:len(dbNames)]))
    ls = dict(zip(dbNames,lls[0:len(dbNames)]))
    
    for vv in hist_var:
        vary = ''
        if seasons=='10yrs':
            fig, ax = plt.subplots(figsize=(12, 8))
        
        for dbName in dbNames:
            idx = df['dbName'] == dbName
            sel = df[idx]
            
            if seasons == '10yrs':
                bb = sel.groupby(['healpixID'])[vv].sum().reset_index()
                idx = bb['nvisits'] < 1100.
                ax.hist(bb[idx][vv], histtype='step', 
                        bins=50,label=dbName,
                        linestyle=ls[dbName],color=colors[dbName])
            if vary == timescale:
                for year in sel[vary].unique():
                    idxb = sel[vary] == year
                    if varx == 'cadence':
                        idxb &= sel[varx] < 25.
                        selb = sel[idxb]
                        ax.hist(selb[vv], histtype='step', bins=50)

        ax.set_xlabel(r'{}'.format(dict_leg[vv]))
        ax.set_ylabel(r'Number of entries')
        ax.grid(visible=True)
        ax.set_xlim([0., None])
        ax.legend(fontsize=15,frameon=False)
        
def plot_histos(hist_var,df,seasons,timescale='year'):
    """
    Function to plot histograms of data

    Parameters
    ----------
    hist_var : list(str)
        List of variables to plot.
    df : pandas df
        Data to plot.

    Returns
    -------
    None.

    """
    
    if seasons == '10yrs':
        df = df.groupby(['healpixID','dbName'])[hist_var].sum().reset_index()
    
    for vv in hist_var:
        if seasons == '10yrs':
            plot_histos_db(df,vv,figtit='10 years')
        else:
            for seas in seasons:
                idx = df[timescale] == seas
                sel = df[idx]
                plot_histos_db(sel,vv,figtit='year {}'.format(seas))
                
        
        
        
def plot_histos_db(df,varx,fig=None,ax=None,figtit=''):
    
    ccols = ['r','k','b']
    lls = ['solid','dashed','dotted']
    
    dbNames = df['dbName'].unique().tolist()
    colors = dict(zip(dbNames,ccols[0:len(dbNames)]))
    ls = dict(zip(dbNames,lls[0:len(dbNames)]))
    
    if fig is None:
        fig, ax = plt.subplots(figsize=(12, 8))

    if figtit != '':
        fig.suptitle(figtit)
        
    for dbName in dbNames:
           idx = df['dbName'] == dbName
           sel = df[idx]
           ax.hist(sel[varx], histtype='step', 
                   bins=50,label=dbName,
                   linestyle=ls[dbName],color=colors[dbName])
    
    
    ax.set_xlabel(r'{}'.format(dict_leg[varx]))
    ax.set_ylabel(r'Number of entries')
    ax.grid(visible=True)
    ax.set_xlim([0., None])
    ax.legend(fontsize=15,frameon=False)
    
    
    
parser = OptionParser(description='Script to plot pixel level OS infos')

parser.add_option('--dbName', type=str, default='test_newb',
                  help='dbName to process [%default]')
parser.add_option('--dbDir', type=str, default='../test_metric',
                  help='dbDir of the OS to process [%default]')
parser.add_option('--nside', type=int, default=128,
                  help='healpix nside parameter [%default]')
parser.add_option('--plots', type=str,
                  default='gen_plots,mollview',
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

"""
if '-' in mollview_seasons:
    cad_brk = mollview_seasons.split('-')
    moll_seasons = list(range(int(cad_brk[0]), int(cad_brk[1])+1))
else:
    moll_seasons = list(map(int, mollview_seasons.split(',')))
"""
if '-' in seasons:
    cad_brk = seasons.split('-')
    seasons = list(range(int(cad_brk[0]), int(cad_brk[1])+1))
else:
    if ',' in seasons:
        seasons = list(map(int, seasons.split(',')))
    else:
        if seasons != '10yrs':
            seasons = [int(seasons)]
print('seasons moll', seasons)

# fName = '{}/{}.hdf5'.format(dbDir, dbName)

df = pd.DataFrame()
for dbName in dbNames:
    df_ = load_data(dbDir, dbName, fields, fieldType=fieldType)
    df_['dbName'] = dbName
    df = pd.concat((df,df_))

# df = pd.read_hdf(fName)
#df['dbName'] = dbName
print(df.columns)
# print(test)

if mollview_outDir != 'None':
    from sn_tools.sn_io import checkDir
    checkDir(mollview_outDir)

idx = df[timescale] > 0
idx &= df[timescale] < 11
idx &= df['cadence'] > 0.
sel = df[idx]

vvar = ['cadence', 'nvisits', 'm5_i']
legvar = ['cadence [day]', 'N$_{visits}$/pixel', '$m_{5}^{i}$']
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
    dbNames = sel['dbName'].unique()
    for vv in mollview_var:
        for dbName in dbNames:
            idx = sel['dbName'] == dbName
            selb=sel[idx]
            plotMollview_seasons(nside, selb, dbName,
                                 yvar=vv, yleg=dict_leg[vv],
                                 op=np.mean, seasons=seasons,outDir=mollview_outDir)
        ## add the diff
        if len(dbNames) ==2:
            dd = {}
            for i,dbName in enumerate(dbNames):
                idxa = df['dbName'] == dbName
                dd[i] = pd.DataFrame(df[idxa])
                
            ddb = dd[0].merge(dd[1],left_on=['healpixID',timescale],
                              right_on=['healpixID',timescale])
            newvar = 'delta_{}'.format(vv)
            ddb[newvar] = ddb['{}_x'.format(vv)]-ddb['{}_y'.format(vv)]
            dbNa = ddb['dbName_x'].unique()[0]
            dbNb = ddb['dbName_y'].unique()[0]
            dbName = '{}-{}'.format(dbNa,dbNb)
            plotMollview_seasons(nside, ddb, dbName,
                                 yvar=newvar, yleg='$\Delta$ '+dict_leg[vv],
                                 op=np.mean, seasons=seasons,outDir=mollview_outDir)
            
            

if 'hist' in plots:
    plot_histos(hist_var,sel,seasons)
    """
    print(sel.columns)
    for vv in hist_var:
        varx = vv.split('_')[0]
        vary = vv.split('_')[1]
        fig, ax = plt.subplots(figsize=(12, 8))
        if '10yrs' in vv:
            bb = sel.groupby(['healpixID'])[varx].sum().reset_index()
            idx = bb['nvisits'] < 1000.
            ax.hist(bb[idx][varx], histtype='step', bins=50)
        if vary == timescale:
            for year in sel[vary].unique():
                idxb = sel[vary] == year
                if varx == 'cadence':
                    idxb &= sel[varx] < 25.
                selb = sel[idxb]
                ax.hist(selb[varx], histtype='step', bins=50)

        ax.set_xlabel(r'{}'.format(dict_leg[varx]))
        ax.set_ylabel(r'Number of entries')
        ax.grid(visible=True)
        ax.set_xlim([0., None])
    """
"""
print_pixel_info(sel, 109384)
print_pixel_info(sel, 109031)
"""

plt.show()
