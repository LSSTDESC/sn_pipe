#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 10:12:26 2026

@author: philippe.gris@clermont.in2p3.fr
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from astropy.table import Table
from sn_analysis.sn_tools import get_spline

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
    xvp = np.linspace(xmin, xmax, 50*nx)
    yvp = np.linspace(ymin, ymax, 50*ny)
    
    X,Y = np.meshgrid(xvp,yvp)
    
    xmin = np.min(xvp)
    xmax = np.max(xvp)
    ymin = np.min(yvp)
    ymax = np.max(yvp)

    fluxpixels = grid((X,Y))
    print(fluxpixels)
    #fluxpixels = np.round(fluxpixels,2)
    print(fluxpixels)
    im = ax.imshow(fluxpixels,
                   extent=[xmin,xmax,ymin,ymax],
                   #vmin=np.min(fluxpixels),vmax=np.max(fluxpixels),
                   cmap=plt.cm.jet,aspect='auto',origin='lower')
    
    for io,vv in enumerate(iso):
        solutions = np.argwhere((fluxpixels>=vv)&(fluxpixels<=vv+0.02))
        ival = solutions[:,0].tolist()
        jval = solutions[:,1].tolist()
        x_iso = X[ival,jval]
        y_iso = Y[ival,jval]
        df_iso = pd.DataFrame(x_iso,columns=[varx])
        df_iso[vary] = y_iso
        
        if not smoothIt:
            ax.plot(x_iso,y_iso,'k.')
        else:
            df_iso = df_iso.sort_values(by=[varx])
            df_iso = df_iso.groupby(varx)[vary].mean().reset_index()
        
            xnew, spl_smooth = get_spline(df_iso,varx,vary)
            ax.plot(xnew, spl_smooth,color='k',marker='.',markersize=0.05)
        
            ytext = df_iso[vary].max()+0.00005
            idx = df_iso[varx] >= x_iso[io]
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

def plot_airmass(df,varx='sigma_pwv',vary_prefix='std_zp',airmass=1.2):
    
    idx = df['mean_airmass'] == airmass
    
    sel = df[idx]
    
    fig, ax = plt.subplots(figsize=(12,8))
    
    
    for b in 'grizy':
        plot_indiv(sel,)
    
    
    

theDir = '../zp_atmos'

theFile = 'zp_atmos_pwv.hdf5'

fName = '{}/{}'.format(theDir,theFile)

df = pd.read_hdf(fName)

print(df.columns)

print(df[['mean_airmass','sigma_pwv','mean_mean_wave_z', 'std_mean_wave_z']])
print(test)
for b in 'grizy':
    df['std_zp_{}'.format(b)] *= 1000 # in mmag
    
#grid plots
"""
tab = Table.from_pandas(df,index=False)
plot_grid(tab,varx='mean_airmass',vary='sigma_pwv',varz='std_zp_y')
plot_grid(tab,varx='mean_airmass',vary='sigma_pwv',varz='std_mean_wave_y',
          figtitle='$\sigma_{mean wave}^{y}$',smoothIt=False)
"""

plot_airmass(df,varx='sigma_pwv',vary_prefix='std_zp')

plt.show()





