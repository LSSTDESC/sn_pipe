import h5py
import glob
from astropy.table import Table, vstack
import matplotlib.pyplot as plt
import numpy as np
import pprint

def plotLC(table,ax,band_id, inum = 0):
    fontsize = 10
    plt.yticks(size=fontsize)
    plt.xticks(size=fontsize)
    for band in 'ugrizy':
        i = band_id[band][0]
        j = band_id[band][1]
        #ax[i,j].set_yscale("log")
        idx = table['band'] == 'LSST::'+band
        sel = table[idx]
        #print('hello',band,inum,len(sel))
        #ax[band_id[band][0]][band_id[band][1]].errorbar(sel['time'],sel['mag'],yerr = sel['magerr'],color=colors[band])
        ax[i,j].errorbar(sel['time'],sel['flux_e_sec'],yerr = sel['flux_e_sec']/sel['snr_m5'],
                         markersize=200000.,color=colors[band],linewidth=1)
        if i > 1:
            ax[i,j].set_xlabel('MJD [day]',{'fontsize': fontsize})
        ax[i,j].set_ylabel('Flux [pe/sec]',{'fontsize': fontsize})
        ax[i,j].text(0.1, 0.9, band, horizontalalignment='center',
             verticalalignment='center', transform=ax[i,j].transAxes)

def plotParameters(fieldname, fieldid, tab, season):
    """ Plot simulation parameters
    parameters ('X1', 'Color', 'DayMax', 'z')
    Input
    ---------
    fieldname: (DD or WFD)
    fieldid: (as given by OpSim)
    tab: recarray of parameters
    season: season

    Returns
    ---------
    Plot (x1,color,dayMax,z)
    """

    idx = tab['season'] == season
    sel = tab[idx]
    thesize = 15
    toplot = ['x1', 'color', 'daymax', 'z']
    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(10, 9))
    title= '{} - fieldid {} - season {}'.format(fieldname,fieldid,season)
    fig.suptitle(title, fontsize=thesize)

    for i, var in enumerate(toplot):
        ix = int(i/2)
        iy = i % 2
        axis = ax[ix][iy]
        i#f var != 'z':
        axis.hist(sel[var],histtype='step') #bins=len(sel[var]))
        axis.set_xlabel(var, fontsize=20)
        axis.set_ylabel('Number of entries', fontsize=thesize)
        axis.tick_params(axis='x', labelsize=thesize)
        axis.tick_params(axis='y', labelsize=thesize)


        


LCDir = '/sps/lsst/users/gris/LC'
dbName = 'baseline_v1.3_10yrs'
dbName = 'descddf_illum5_v1.3_10yrs'
prefix = 'sncosmo_DD'

paramName = '{}/Simu_{}_{}_seas*.hdf5'.format(LCDir,prefix,dbName)

print('looking for',paramName)
paramFiles = glob.glob(paramName)


params = Table()
for paramFile in paramFiles:
    f = h5py.File(paramFile, 'r')
    print(f.keys(),len(f.keys()))
    for i, key in enumerate(f.keys()):
        params = vstack([params, Table.read(paramFile, path=key)])
    
    # params is an astropy table
    print(params[:10])
    print(type(params),params.dtype)

    print('NSN',len(params))

plt.plot(params['pixRa'],params['pixDec'],'ko')


for fieldid in np.unique(params['fieldid']):
    idx = params['fieldid']==fieldid
    field = params[idx]
    fieldname = np.unique(field['fieldname'])
    season = np.unique(field['season'])
    print(fieldname,fieldid,field,season)
    plotParameters(fieldname,fieldid,field,season=1)

for paramFile in paramFiles:
    lcFile = paramFile.replace('Simu','LC')
    f = h5py.File(lcFile, 'r')
    #print(f.keys(),len(f.keys()))

    
    bands='ugrizy'
    band_id = dict(zip(bands,[(0,0),(0,1),(1,0),(1,1),(2,0),(2,1)]))
    colors = dict(zip(bands,'bcgyrm'))
    #for i, key in enumerate(f.keys()):
    for val in params:
        fig, ax = plt.subplots(ncols=2, nrows=3,figsize=(15,10))
        fig.suptitle('z: {} - T0: {} - season {}'.format(val['z'],np.round(val['daymax'],1),val['season']))
        #lc = Table.read(lcFile, path=key)
        lc = Table.read(lcFile, path='lc_{}'.format(val['id_hdf5']))
        pprint.pprint(lc.meta) # metadata
        print(lc.dtype) # light curve points
        plotLC(lc,ax,band_id,i)
        plt.show()


plt.show()
