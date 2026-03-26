#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 10:12:26 2026

@author: philippe.gris@clermont.in2p3.fr
"""
import pandas as pd
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from astropy.table import Table
from sn_analysis.sn_tools import get_spline
from sn_plotter_tools import plt,filtercolors

def plot_grid(tab, varx='airmass',xlabel='airmass',
              vary='sigma_pwv',ylabel='$\sigma_{PWV}$ [mm]',
              varz='std_zp_y',figtitle='$\sigma_{ZP}^{y}$',
              iso=[1.,2.,5.],
              txt_iso=['1 mmag','2 mmag','5 mmag'],
              x_iso=[1.5]*3,smoothIt=True):
    """
    Function to make meshgrid plots

    Parameters
    ----------
    tab : astropy table
        Data to process
    varx : str, optional
        x-axis variable. The default is 'airmass'.
    xlabel : str, optional
        x-axis label. The default is 'airmass'.
    vary : str, optional
        y-axis variable. The default is 'sigma_pwv'.
    ylabel : str, optional
        y-axis label. The default is '$\sigma_{PWV}$ [mm]'.
    varz : str, optional
        z-axis variable. The default is 'std_zp_y'.
    figtitle : str, optional
        Figure title. The default is '$\sigma_{ZP}^{y}$'.
    iso : list(float), optional
        List of isocurve variables. The default is [1.,2.,5.].
    txt_iso : str, optional
        List of text for iso curves. The default is ['1 mmag','2 mmag','5 mmag'].
    x_iso : list(float), optional
        x-positions for txt_iso. The default is [1.5]*3.
    smoothIt : bool, optional
        To smooth the iso curves. The default is True.

    Returns
    -------
    None.

    """

    fig,ax = plt.subplots(figsize=(12,8))
    fig.suptitle(figtitle)
    xmin,xmax,xstep,nx = limVals(tab, varx)
    ymin,ymax,ystep,ny = limVals(tab, vary)
    
    xstep = np.round(xstep, 3)
    ystep = np.round(ystep, 3)
    
    xv = np.linspace(xmin, xmax, nx)
    yv = np.linspace(ymin, ymax, ny)
    
    index = np.lexsort((tab[vary], tab[varx]))
    print(tab[index][[varx,vary,varz]])
    flux = np.reshape(tab[index][varz], (nx, ny))
    
    for i in range(nx):
        for j in range(ny):
            print(xv[i],yv[j],flux[i,j])
    
    
    print(flux.shape,flux[0][0])
    grid=RegularGridInterpolator((xv,yv),flux, 
                                 method='linear', 
                                 bounds_error=False, fill_value=0.)
    #print(grid(()))
    xvp = np.linspace(xmin, xmax, 100*nx)
    yvp = np.linspace(ymin, ymax, 100*ny)
    
    X,Y = np.meshgrid(xvp,yvp)
    
    xmin = np.min(xvp)
    xmax = np.max(xvp)
    ymin = np.min(yvp)
    ymax = np.max(yvp)

    fluxpixels = grid((X,Y))
    #print(fluxpixels)
    #fluxpixels = np.round(fluxpixels,2)
    #print(fluxpixels)
    im = ax.imshow(fluxpixels,
                   extent=[xmin,xmax,ymin,ymax],
                   #vmin=np.min(fluxpixels),vmax=np.max(fluxpixels),
                   cmap=plt.cm.jet,aspect='auto',origin='lower')
    
    for io,vv in enumerate(iso):
        solutions = np.argwhere((fluxpixels>=vv)&(fluxpixels<=1.1*vv))
        ival = solutions[:,0].tolist()
        jval = solutions[:,1].tolist()
        x_iso = X[ival,jval]
        y_iso = Y[ival,jval]
        df_iso = pd.DataFrame(x_iso,columns=[varx])
        df_iso[vary] = y_iso
        df_iso = df_iso.sort_values(by=[varx])
        df_iso = df_iso.groupby(varx)[vary].mean().reset_index()
        
        if not smoothIt:
            ax.plot(df_iso[varx],df_iso[vary],
                    color='k',marker='.',markersize=0.05)
        else:
            """
            ax.plot(df_iso[varx],df_iso[vary],
                    color='r',marker='*',markersize=0.05)
            """
            xnew, spl_smooth = get_spline(df_iso,varx,vary,nx=10)
            ax.plot(xnew, spl_smooth,color='k',marker='.',markersize=0.05)
        
            ytext = df_iso[vary].max()+0.00005
            idd = np.argmin(np.abs(df_iso[varx]-x_iso[io]))
            ytext = df_iso.loc[idd,vary]*1.30
            ax.text(1.6,ytext,txt_iso[io])
        
    fig.colorbar(im)
    ax.grid(visible=True)
    ax.set_xlabel(r'{}'.format(xlabel))
    ax.set_ylabel(r'{}'.format(ylabel))

    """
    radius = int(xmax)

    print('radius',radius)
    r = []
    for x in np.arange(0,radius,0.01):
        r.append((x,np.sqrt(radius**2-x**2)))

    res = np.rec.fromrecords(r, names=['x','y'])
    ax.plot(res['x'],res['y'],'ko')
    """
    
def limVals(lc, field):
    """ Get unique values of a field in  a table
    Parameters
    ----------
    lc: Table
        astropy Table (here probably a LC)
    field: str
        name of the field of interest
    
    Returns
    -------
    vmin: float
        min value of the field
    vmax: float
        max value of the field
    vstep: float
        step value for this field (median)
    nvals: int
        number of unique values
    """

    lc.sort(field)
    #dfb = df.sort_values(by=[field])
    vals = np.unique(lc[field].data.round(decimals=4))
    
    vmin = np.min(vals)
    vmax = np.max(vals)
    vstep = np.median(vals[1:]-vals[:-1])

    return vmin, vmax, vstep, len(vals)

def plot_airmass(df,varx='sigma_pwv',xlabel='$\sigma_{PWV}$ [mm]',
                 vary_prefix='std_zp',ylabel='$\sigma_{ZP}$ [mmag]',
                 airmass=[1.2,2.5],
                 y_iso=[1,2,5],
                 txt_iso=['1 mmag','2 mmag','5 mmag'],
                 ymax=6,deltay_txt=0.03,xtext=0.015):
    
    df = df.round({'mean_airmass':2})
    fig, ax = plt.subplots(figsize=(12,8))
    
    bands = 'grizy'
    markers = ['o','P','s','*','h']
    mm = dict(zip(bands,markers))
    lstyle = ['solid','dotted']
    ls = dict(zip(airmass,lstyle))
    
    for airm in airmass:
        idx = df['mean_airmass'] == airm
        sel = df[idx]
        for b in bands:
            yvar = '{}_{}'.format(vary_prefix,b)
            lab = '{} band'.format(b)
            if airm > 1.5:
                lab=None
            plot_indiv(sel,xvar=varx,yvar=yvar,label=lab,
                       color=filtercolors[b],
                       marker=mm[b],lstyle=ls[airm],
                       fig=fig,ax=ax,smoothIt=True)
    
    
    
    idx = df['mean_airmass'].isin(airmass)
    sel = df[idx]
    xmin = sel[varx].min()
    xmax = sel[varx].max()
    ax.set_xlim([xmin,xmax])
    ax.set_ylim([0,ymax])
    for io,yvals in enumerate(y_iso):
        ax.plot([xmin,xmax],[yvals]*2,linestyle='dashed',color='k')
        #ax.text(xtext,yvals+deltay_txt,txt_iso[io],fontsize=12)
        #ax.lines(x=xtext, ymin=, ymax=250, color = 'black', linestyles="dashed")
        ax.text(x=xtext, y=yvals, s=txt_iso[io], 
                ha='center', va='center', backgroundcolor='white',fontsize=12)
        
    ax.set_xlabel(r'{}'.format(xlabel))
    ax.set_ylabel(r'{}'.format(ylabel))
    ax.legend(loc='upper left',
              bbox_to_anchor=(0., 1.15), ncol=5, frameon=False, fontsize=15)
    ax.grid(visible=True)
    #ax.text(0.2,1.05,'.... airmass=2.5 ',fontsize=12,transform=ax.transAxes)
    
    x_trans=0.25
    ax.annotate('', xy=(x_trans+0.,1.05), 
                xycoords='axes fraction', xytext=(x_trans+0.05, 1.05),
                arrowprops=dict(arrowstyle="-", color='k'))
    ax.text(x_trans+0.055,1.04,'airmass=1.2',fontsize=12,transform=ax.transAxes)
    ax.annotate('', xy=(x_trans+0.2,1.05), xycoords='axes fraction',
                xytext=(x_trans+0.25, 1.05),
               arrowprops=dict(arrowstyle="-", color='k',linestyle='dotted'))
    ax.text(x_trans+0.255,1.04,'airmass=2.5',fontsize=12,transform=ax.transAxes)
    
def plot_indiv(df,
               xvar='z', xlabel='z', 
               yvar='N', ylabel='NSN',label='',
               lstyle='solid',color='k',marker='o',
               figtitle='',fig=None, ax=None,smoothIt=False): 
    
    if fig is None:
        fig, ax = plt.subplots(figsize=(12,8))
    
    
    if not smoothIt:
        ax.plot(df[xvar],df[yvar],
                marker=marker,linestyle=lstyle,
                color=color,markersize=8,mfc='None',label=label)
    else:
        df = df.sort_values(by=[xvar])
        xnew, spl_smooth = get_spline(df,xvar,yvar,nx=10)
        ax.plot(xnew, spl_smooth,color=color,
                marker=marker,linestyle=lstyle,
                markersize=10,mfc='None',label=label) 
    

theDir = '../zp_atmos'

theFile = 'zp_atmos_ozone.hdf5'

fName = '{}/{}'.format(theDir,theFile)

df = pd.read_hdf(fName)

print(df.columns)

print(df[['mean_airmass','sigma_pwv','mean_mean_wave_z', 'std_mean_wave_z']])

for b in 'grizy':
    df['std_zp_{}'.format(b)] *= 1000 # in mmag
    
#grid plots

vary='sigma_pwv'
ylabel='$\sigma_{PWV}$ [mm]'
vary='sigma_aerosol'
ylabel='$\sigma_{aerosol}$'
b='z'
tab = Table.from_pandas(df,index=False)
"""
plot_grid(tab,varx='mean_airmass',
          vary=vary,ylabel=ylabel,
          varz='std_zp_{}'.format(b),smoothIt=True)
plot_grid(tab,varx='mean_airmass',
          vary=vary,ylabel=ylabel,
          varz='std_mean_wave_{}'.format(b),
          figtitle='$\sigma_{mean wave}^{z}$',iso=[0.05,0.1,0.15],
              txt_iso=['0.05 nm','0.1 nm','0.15 nm'],
              x_iso=[1.5]*3,smoothIt=False)
"""
varx = 'sigma_pwv'
xlabel = '$\sigma_{PWV}$ [mm]'
varx = 'sigma_aerosol'
xlabel = '$\sigma_{aerosol}$'
varx = 'sigma_airmass'
xlabel = '$\sigma_{airmass}$'
varx = 'sigma_ozone'
xlabel = '$\sigma_{ozone}$ [DU]'
xtext=25
plot_airmass(df,varx=varx,xlabel=xlabel,vary_prefix='std_zp',xtext=xtext)
plot_airmass(df,varx=varx,xlabel=xlabel,
                 vary_prefix='std_mean_wave',ylabel='$\sigma_{meanwave}$ [mm]',
                 airmass=[1.2,2.5],
                 y_iso=[0.05,0.1,0.15],
                 txt_iso=['0.05 nm','0.1 nm','0.15 nm'],
                 ymax=0.2,deltay_txt=0.005,xtext=xtext)

plt.show()





