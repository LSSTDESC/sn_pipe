#from sn_plotters.sn_cadencePlotters import Lims
import numpy.lib.recfunctions as rf
from . import plt
import numpy as np
import pandas as pd
from sn_tools.sn_telescope import Telescope
from scipy import interpolate
from sn_tools.sn_cadence_tools import AnaOS
import os
from sn_tools.sn_io import loopStack
from astropy.table import Table, vstack
import time
from sn_tools.sn_calcFast import covColor
from scipy.spatial import distance
from . import filtercolors

class Lims_z_m5_band:

    def __init__(self,x1,color,band,fluxes, 
                 SNR,
                 mag_range = [23., 27.5], 
                 dt_range=[0.5, 25.],
                 m5_str='m5_mean',cadence_str='cadence_mean'):

        
        #namesRef = ['SNCosmo_new']
        Li_files = []
        mag_to_flux_files = []
        self.mag_range = mag_range
        self.dt_range = dt_range
        
        self.fluxes = fluxes
        self.band = band
        self.m5_to_f5 = m5_to_flux5([band])[band]
        self.lims = self.getLims_z(fluxes,SNR)
        self.interp()

    def getLims_z(self, tab, SNR):
        """
        Estimations of the limits

        Parameters
        ---------------

        band : str
          band to consider
        tab : numpy array
          table of data
        SNR : float
           Signal-to-Noise Ratio cut

        Returns
        -----------
        dict of limits with redshift and band as keys.
        """

        lims = {}

        tab['z'] = np.round(tab['z'],2)
        for z in np.unique(tab['z']):

            idx = np.abs(tab['z']-z)<1.e-5
            idx &= (tab['band'] == 'LSST::'+self.band)
            idx &= (tab['flux_e'] > 0.)
            sel = tab[idx]

            if len(sel) > 0:
                Li2 = np.sqrt(np.sum(sel['flux_e']**2))
                lim = 5. * Li2 / SNR
                if z not in lims.keys():
                    lims[z] = {}
                lims[z][self.band] = lim
                
        print(self.band,lims)
        return lims   

    def mesh(self):
        """
        Mesh grid to estimate five-sigma depth values (m5) from mags input.

        Parameters
        ---------------

        mag_to_flux : magnitude to flux values


        Returns
        -----------
        m5 values
        time difference dt (cadence)
        metric=sqrt(dt)*F5 where F5 is the 5-sigma flux
        """
        dt = np.linspace(self.dt_range[0], self.dt_range[1], 100)
        m5 = np.linspace(self.mag_range[0], self.mag_range[1], 100)
        """
        ida = mag_to_flux['band'] == self.band
        fa = interpolate.interp1d(
            mag_to_flux[ida]['m5'], mag_to_flux[ida]['flux_e'],bounds_error=False,fill_value=0.0)
        """
        f5 = self.m5_to_f5(m5)
        F5, DT = np.meshgrid(f5, dt)
        M5, DT = np.meshgrid(m5, dt)
        metric = np.sqrt(DT) * F5

        return M5, DT, metric

    def interp(self):
        """
        Estimate a grid of interpolated values
        in the plane (m5, cadence, metric)

        Parameters
        ---------------
        None

        """

        M5, DT, metric = self.mesh()

        sorted_keys = []
        #for i in range(len(self.lims)):
        print(self.lims)
        sorted_keys = np.sort([k for k in self.lims.keys()])[::-1]
        print(sorted_keys)
        figa, axa = plt.subplots()
        
        fmt = {}
        ll = [self.lims[zz][self.band] for zz in sorted_keys]
        cs = axa.contour(M5, DT, metric, ll)

        points_values = None
        
        for io, col in enumerate(cs.collections):
            if col.get_segments():
                
                myarray = col.get_segments()[0]
                
                res = np.array(myarray[:, 0], dtype=[('m5', 'f8')])
                res = rf.append_fields(res, 'cadence', myarray[:, 1])
                res = rf.append_fields(res, 'z', [sorted_keys[io]]*len(res))
                if points_values is None:
                    points_values = res
                else:
                    points_values = np.concatenate((points_values, res))
        self.Points_Ref = points_values
        #print(self.Points_Ref)

        #plt.show()
        plt.close(figa)  # do not display
    
    def plot(self, restot,dbName='',
             target={  # 'g': (26.91, 3.), # was 25.37
                              'r': (26.5, 3.),  # was 26.43
                              # was 25.37      # could be 25.3 (400-s)
                              'i': (26.16, 3.),
                              # was 24.68      # could be 25.1 (1000-s)
                              'z': (25.56, 3.),
                              'y': (24.68, 3.)},# was 24.72
             saveFig=False):
        """ Plot the cadence metric in the plane: median cadence vs m5

        Parameters
        --------------
        restot : array
          array of observations containing at least the following fields:
          m5_mean : mean five-sigma depth value (par season and per band)
          cadence_mean : mean cadence (per season and per band)
        target: dict
          Values corresponding to targets
        

        Returns
        ---------
        None

        """

        M5, DT, metric = self.mesh()
       
        sorted_keys = np.sort([k for k in self.lims.keys()])[::-1]

        plt.figure(figsize=(8, 6))
        plt.imshow(metric, extent=(
            self.mag_range[0], self.mag_range[1], self.dt_range[0], self.dt_range[1]), aspect='auto', alpha=0.25)

        plt.plot(restot['m5_mean'], restot['cadence_mean'], 'r+', alpha=0.9)

        color = ['k', 'b']
        
        fmt = {}
        ll = [self.lims[zz][self.band] for zz in sorted_keys]
        print('plotting',self.band,ll)
        cs = plt.contour(M5, DT,
                         metric, ll, colors='b')
        strs = ['$z=%3.2f$' % zz for zz in sorted_keys]
        for l, s in zip(cs.levels, strs):
            fmt[l] = s
        plt.clabel(cs, inline=True, fmt=fmt,
                       fontsize=16, use_clabeltext=True)

        t = target.get(self.band, None)
        if t is not None:
            plt.plot(t[0], t[1],
                     color='r', marker='*',
                     markersize=15)
        plt.xlabel('$m_{5\sigma}$', fontsize=18)
        plt.ylabel(r'Observer frame cadence $^{-1}$ [days]', fontsize=18)
        #plt.title('$%s$' % self.band.split(':')[-1], fontsize=18)
        plt.title('{} band'.format(self.band.split(':')[-1]), fontsize=18)
        plt.xlim(self.mag_range)
        plt.ylim(self.dt_range)
        plt.grid(1)
        if saveFig:
            plt.savefig('{}_{}_cad_vs_m5.png'.format(dbName,self.band))
    """
    def plot(self,res,dbName,saveFig=False):

        self.lim_z.plotCadenceMetric(res,dbName=dbName,saveFig=saveFig)
    """
    def getLims(self, data,m5_str,cadence_str,blue_zcut=-1):

        idx = (data[m5_str] >= self.mag_range[0]) & (
             data[m5_str] <= self.mag_range[1])
        idx &= (data[cadence_str] >= self.dt_range[0]) & (
            data[cadence_str] <= self.dt_range[1])
        data = data[idx]
        
        restot = None

        #self.plot(data,'test')

        if len(data) > 0:
            resu = np.copy(data)
            
            zlims = self.interpGriddata(data,m5_str=m5_str,cadence_str=cadence_str)
            zlims[np.isnan(zlims)] = -1
            resu = rf.append_fields(data, 'zlim', zlims)
            if blue_zcut >0:
                io = resu['zlim']<=blue_zcut
                resu = resu[io]

            if restot is None:
                restot = resu
            else:
                restot = np.concatenate((restot, resu))

        return restot    

    def interpGriddata(self,data,m5_str='m5_mean',cadence_str='cadence_mean'):
        """
        Estimate metric interpolation for data (m5,cadence)

        Parameters
        ---------------

        data : data where interpolation has to be done (m5,cadence)

        Returns
        -----------
        griddata interpolation (m5,cadence,metric)

        """

        ref_points = self.Points_Ref
        res = interpolate.griddata((ref_points['m5'], ref_points['cadence']), ref_points['z'], (
            data[m5_str], data[cadence_str]), method='cubic')
        return res


class Lims_z_m5:

    def __init__(self, x1,color,flux_file,
                 SNR= dict(zip('rizy', [25., 25., 30., 35.])),
                 blue_zcut=dict(zip('gri',[0.3,0.701,1.0]))):
        

        self.x1 = x1
        self.color = color
        self.SNR = SNR
        self.bands = SNR.keys()
        self.blue_zcut=blue_zcut
        self.fluxes = np.load(flux_file)
        # select data corresponding to (x1,color)

        idx = (self.fluxes['x1']-x1)<1.e-5
        idx &= (self.fluxes['color']-color)<1.e-5
        self.fluxes = self.fluxes[idx]
        
        print('there man',len(self.fluxes))

    def process(self,cadences=[3.,4.]):
                 
        zlims = None
        for band in self.bands:

            print('processing',band)
            idx = self.fluxes['band'] == 'LSST::'+band
            fluxes_b = self.fluxes[idx]
            print('fluxes band',band,len(fluxes_b))
            myclass = Lims_z_m5_band(self.x1,self.color,band,fluxes_b,SNR=self.SNR[band])
    
            mag_min = myclass.mag_range[0]
            mag_max = myclass.mag_range[1]

            r = []
            for mag in np.arange(mag_min,mag_max,0.001):
                for cad in cadences:
                    r.append((cad,mag,band))

            data = np.rec.fromrecords(r,names=['cadence_mean','m5_mean','band'])

            myclass.plot(data,'unknown')
            plt.show()

            blue_zcut = -1
            if band in self.blue_zcut.keys():
                blue_zcut = self.blue_zcut[band]
            zl = myclass.getLims(data,'m5_mean','cadence_mean',blue_zcut)
            
            print('zlims',zl.dtype)
            #apply the cut in z below - blue cutoff for SN

            if zlims is None:
                zlims = zl
            else:
                zlims = np.concatenate((zlims,zl))

        return zlims
    

    def plot(self,zlims,cadence,ystr='m5_mean',yleg='m$_5$',yscale='linear',locx=0.45,locy=500.):

        fig, ax = plt.subplots()

        #bands = np.unique(zlims['band'])
        bands = 'rizy'
        print('plotting lll',bands)
        filtercolors = dict(zip(bands,['g','y','r','m']))
        if 'band' in zlims.dtype.names:
            for band in bands:
                idx = zlims['band'] == band
                sel = zlims[idx]
                #print('allo',band,sel)
                self.plotIndiv(ax,sel,ystr,yleg,cadence,band,color=filtercolors[band],locx=locx,locy=locy)
        else:
            self.plotIndiv(ax,zlims,ystr,yleg,cadence,'all',color='k',locx=locx,locy=locy)

        fontsize = 12
        ax.legend(loc = 'upper left',fontsize=fontsize)

        ax.set_xlabel('z$_{faint}$',fontsize=fontsize)
        ax.set_ylabel(yleg,fontsize=fontsize)
        ax.set_yscale(yscale)
        ax.grid()
        print('plot done')

    def plotIndiv(self,ax,sel,ystr,yleg,cadence,band,color,locx,locy):
        
        
        #for cad in np.unique(sel['cadence_mean']):
        ls = ['-','--']
        for i,cad in enumerate(cadence):
            idb = np.abs(sel['cadence_mean']-cad)<1.e-5
            selb = sel[idb]
            zlim_name = 'zlim'
            idc = selb[zlim_name]>0.
            selc = selb[idc]
               
            if i == 0:
                ax.plot(selc[zlim_name],selc[ystr],marker=None,
                        color=color,
                        label='{}'.format(band),
                        ls=ls[i])
            else:
                ax.plot(selc[zlim_name],selc[ystr],marker=None,
                        color=color,
                        ls=ls[i]) 
            if band =='r' or band =='all':
                limsy = ax.get_ylim()
                yscale = limsy[1]-limsy[0]
                locyv = locy-0.06*yscale*i
                ax.plot([locx,locx+0.05],[locyv,locyv],ls=ls[i],color='k')
                ax.text(locx+0.06,locyv,'cadence: {} days'.format(int(cad)))
            
class Nvisits_m5:
    def __init__(self, tab, med_m5):

        cols = tab.columns[tab.columns.str.startswith('m5calc')].values

        self.bands = ''.join([col.split('_')[-1] for col in cols])
        self.f5_to_m5 = flux5_to_m5(self.bands)
        self.m5_to_f5 = m5_to_flux5(self.bands)
        self.snr = tab

        self.med_m5 = self.transform(med_m5)

        print('here medians',self.med_m5)

        idx = tab['z']>=0.2
        tab = tab[idx]

        self.nvisits = self.estimateVisits()


    def estimateVisits(self,):

        dict_cad_m5 = {}
        
        cads = pd.DataFrame(np.arange(0.,10.,1.),columns=['cadence'])

        idx = 0

        #add and index to both df for the merging

        m5 = self.med_m5.copy()
        snr = self.snr.copy()

        m5.loc[:,'idx'] = idx
        snr.loc[:,'idx'] = idx

        snr = snr.merge(m5,left_on=['idx'],right_on=['idx'])
        
        zvals = snr['z'].unique()
        
        df_combi = self.make_combi(zvals,cads) 

        df_merge = snr.merge(df_combi,left_on=['z'],right_on=['z'])

        print(df_merge)
        cols = []
        for b in self.bands:
            df_merge['flux_5_e_sec_{}'.format(b)]=df_merge['flux_5_e_sec_{}'.format(b)]/np.sqrt(df_merge['cadence'])
            df_merge['m5calc_{}'.format(b)] = self.f5_to_m5[b](df_merge['flux_5_e_sec_{}'.format(b)])
            df_merge.loc[:,'Nvisits_{}'.format(b)]=10**(0.8*(df_merge['m5calc_{}'.format(b)]-df_merge['m5single_{}'.format(b)]))
            cols.append('Nvisits_{}'.format(b))
            df_merge['Nvisits_{}'.format(b)]=df_merge['Nvisits_{}'.format(b)].fillna(0.0)
           
        
        #estimate the total number of visits
        df_merge.loc[:,'Nvisits'] = df_merge[cols].sum(axis=1)
        

        return df_merge


        """
        for val in med_m5:
            for season in sel['season'].values:
                seas= s[sel['season']==season]
                tab.loc[:,'season'] = season
                test=tab.merge(seas,left_on=['season'],right_on=['season'])
            tab.loc[:,'season'] = val['season']
        """

        for fieldname in med_m5['fieldname'].unique():
            idx = med_m5['fieldname']==fieldname
            sel = med_m5[idx]
            for season in sel['season'].values:
                seas= sel[sel['season']==season]
                tab.loc[:,'season'] = season
                test=tab.merge(seas,left_on=['season'],right_on=['season'])
                
                zvals = test['z'].unique()

                df_combi = self.make_combi(zvals,cads)

                #merge with test

                df_merge = test.merge(df_combi,left_on=['z'],right_on=['z'])

                print(df_merge)
                cols = []
                for b in self.bands:
                    df_merge['flux_5_e_sec_{}'.format(b)]=df_merge['flux_5_e_sec_{}'.format(b)]/np.sqrt(df_merge['cadence'])
                    df_merge['m5calc_{}'.format(b)] = self.f5_to_m5[b](df_merge['flux_5_e_sec_{}'.format(b)])
                    df_merge.loc[:,'Nvisits_{}'.format(b)]=10**(0.8*(df_merge['m5calc_{}'.format(b)]-df_merge['m5single_{}'.format(b)]))
                    
                    
                
                """
                fig, ax = plt.subplots()
                figb, axb = plt.subplots()
                for b in 'grizy':
                    test['flux_5_e_sec_{}'.format(b)]=test['flux_5_e_sec_{}'.format(b)]/np.sqrt(cadence)
                    test['m5calc_{}'.format(b)] = self.f5_to_m5[b](test['flux_5_e_sec_{}'.format(b)])
                    test.loc[:,'Nvisits_{}'.format(b)]=10**(0.8*(test['m5calc_{}'.format(b)]-test['m5single_{}'.format(b)]))
                    ax.plot(test['z'],test['Nvisits_{}'.format(b)],color=filtercolors[b])
                    axb.plot(test['z'],test['m5calc_{}'.format(b)],color=filtercolors[b])
                ax.grid()
                axb.grid()
                """
                #print(test)

                #plt.show()
    

        self.map_cad_m5 = df_merge
    
    def plot_map(self, dft):

        
        mag_range = [23., 27.5] 
        dt_range=[0.5, 20.]

        dt = np.linspace(dt_range[0], dt_range[1], 100)
        m5 = np.linspace(mag_range[0], mag_range[1], 100)
        
        
        zrange = np.arange(0.3,0.9,0.1)

        df = pd.DataFrame()
        for z in zrange:
            idb = np.abs(dft['z']-z)<1.e-5
            df = pd.concat([df,dft[idb]], sort=False)


        for b in self.bands: 

            f5 = self.m5_to_f5[b](m5)
            M5, DT = np.meshgrid(m5, dt)            
            F5, DT = np.meshgrid(f5, dt)
            metric = np.sqrt(DT) * F5

            fig = plt.figure(figsize=(8, 6))
            fig.suptitle('{} band'.format(b))
            plt.imshow(metric, extent=(mag_range[0],mag_range[1],dt_range[0],dt_range[1]), 
                       aspect='auto', alpha=0.25)

            idx = np.abs(df['cadence']-1.)<1.e-5
            dfsel = df[idx]

            dfsel = dfsel.sort_values(by=['flux_5_e_sec_{}'.format(b)])
            
            ll = dfsel['flux_5_e_sec_{}'.format(b)].values
           
            cs = plt.contour(M5, DT, metric, ll,colors='k')
            
            fmt = {}
            strs = ['$z=%3.2f$' % zz for zz in dfsel['z'].values]
            for l, s in zip(cs.levels, strs):
                fmt[l] = s
            plt.clabel(cs, inline=True, fmt=fmt,
                       fontsize=16, use_clabeltext=True)

           
            
            #for z in df['z'].unique():
            #    sel = df[np.abs(df['z']-z)<1.e-5]
            #    plt.plot(sel['m5calc_{}'.format(b)],sel['cadence'],'ro',mfc='None')
            
            plt.xlabel('$m_{5\sigma}$', fontsize=18)
            plt.ylabel(r'Observer frame cadence $^{-1}$ [days]', fontsize=18)
            plt.xlim(mag_range)
            plt.ylim(dt_range)
            plt.grid(1)
            
        #plt.show()
    

    def plot(self, data, cadences, what='m5_calc',legy='m5'):

        fig, ax = plt.subplots()
        axb = None
        ls = ['-','--','-.']

        if what == 'Nvisits':
            figb, axb = plt.subplots()
            #cols = []
            #for b in self.bands:
            #    cols.append('Nvisits_{}'.format(b))

        for io,cad in enumerate(cadences):
            idx = np.abs(data['cadence']-cad)<1.e-5
            sel = data[idx]

            for b in self.bands:
                if io == 0:
                    ax.plot(sel['z'].values,sel['{}_{}'.format(what,b)].values,color=filtercolors[b],label=b,ls=ls[io])
                else:
                   ax.plot(sel['z'].values,sel['{}_{}'.format(what,b)].values,color=filtercolors[b],ls=ls[io]) 

            if what == 'Nvisits':
                #estimate the total number of visits
                #sel.loc[:,'Nvisits'] = sel[cols].sum(axis=1)
                axb.plot(sel['z'].values,sel['Nvisits'].values,
                         color='k',ls=ls[io],
                         label='cadence: {} days'.format(int(cad)))

        ax.legend()
        ax.grid()
        ax.set_xlabel(r'z')
        ax.set_ylabel(r'{}'.format(legy))
        
        if axb:
            axb.legend()
            axb.grid()
            axb.set_xlabel(r'z')
            axb.set_ylabel(r'{}'.format(legy))



    def make_combi(self,zv, cad):

        df = pd.DataFrame()

        for z in zv:
            dfi = cad.copy()
            dfi.loc[:,'z'] = z
            df = pd.concat([df,dfi],ignore_index=True)

        return df

    def transform(self, med_m5):
 
        dictout = {}

        for band in 'grizy':
            idx = med_m5['filter'] == band
            sel = med_m5[idx]
            if len(sel)>0:
                dictout[band] = [np.median(sel['fiveSigmaDepth'].values)]
            else:
                dictout[band] = [0.0]

        
        
        return pd.DataFrame({'m5single_g': dictout['g'],
                             'm5single_r': dictout['r'],
                             'm5single_i': dictout['i'],
                             'm5single_z': dictout['z'],
                             'm5single_y': dictout['y']})
        """
        return pd.DataFrame({'m5single_g': [24.331887],
                             'm5single_r': [23.829639],
                             'm5single_i': [23.378532],
                             'm5single_z': [22.861743],
                             'm5single_y': [22.094841]})
        """
    """
    def transform_old(self, med_m5):

        df = med_m5.copy()
        
        #gr = df.groupby(['fieldname','season']).apply(lambda x: self.horizont(x))

        gr = df.groupby(['fieldname']).apply(lambda x: self.horizont(x)).reset_index()

        
        if 'season' not in gr.columns:
            gr.loc[:,'season'] = 0

        
        return pd.DataFrame({'fieldname': ['all'],
                             'season': [0],
                             'm5single_g': [0.],
                             'm5single_r': [23.9632],
                             'm5single_i': [23.5505],
                             'm5single_z': [23.0003],
                             'm5single_y': [22.2338]})
        
        return pd.DataFrame({'fieldname': ['all'],
                             'season': [0],
                             'm5single_g': [0.],
                             'm5single_r': [23.829639],
                             'm5single_i': [23.378532],
                             'm5single_z': [22.861743],
                             'm5single_y': [22.094841]})
        """
    def horizont(self,grp):
        
        dictout = {}

        for band in 'grizy':
            idx = grp['filter'] == band
            sel = grp[idx]
            if len(sel)>0:
                dictout[band] = [np.median(sel['fiveSigmaDepth'].values)]
            else:
                dictout[band] = [0.0]

        
        return pd.DataFrame({'m5single_g': dictout['g'],
                             'm5single_r': dictout['r'],
                             'm5single_i': dictout['i'],
                             'm5single_z': dictout['z'],
                             'm5single_y': dictout['y']})
        




"""
        def nvisits_deltam5(m5,m5_median):

            diff = m5-m5_median
        
            nv = 10**(0.8*diff)

            #return nv.astype(int)
            return nv
"""
        

def nvisits(zlims,med_m5):

    def nvisits_deltam5(m5,m5_median):

        diff = m5-m5_median
        
        nv = 10**(0.8*diff)

        #return nv.astype(int)
        return nv

    zlic = np.copy(zlims)
    res_nv = None
    for band in np.unique(zlic['band']):
        m5med = med_m5[band]
        print('here pal',band,m5med,zlic['m5_mean'],zlic['cadence_mean'])
        idx = (zlic['band']==band)&(zlic['m5_mean']>=m5med)
        sel = zlic[idx]
        
        nv = nvisits_deltam5(sel['m5_mean'],m5med)
        print('rr',len(sel),sel['cadence_mean'],nv)
        #print(nv)
        sel = rf.append_fields(sel,'nvisits',nv)
        if res_nv is None:
            res_nv = sel
        else:
            res_nv = np.concatenate((res_nv,sel))
        

    return res_nv


class AnaMedValues:

    def __init__(self,fname):
        
        medValues = np.load(fname)

        print(medValues.dtype)
        df = pd.DataFrame(np.copy(medValues))
        df['fieldname'] = df['fieldname'].map(lambda x: x.strip())
        df['season']=df['season'].astype(int)

        df = df[(df['filter']!='u')]

        dfgrp = df.groupby(['fieldname','season','filter']).median().reset_index()

        dfgrp_season = df.groupby(['fieldname','filter']).median().reset_index()
        dfgrp_season.loc[:,'season'] = 0

        dfgrp_filter =  df.groupby(['filter']).median().reset_index()
        dfgrp_filter.loc[:,'fieldname'] = 'all'
        dfgrp_filter.loc[:,'season'] = 0

        #print(dfgrp[['fieldname','season','filter','fiveSigmaDepth']])
   
        """
        if plot:
            self.plot(dfgrp,dfgrp_season)
        """
        #self.medValues =  df.groupby(['filter']).median().reset_index()
        self.median_m5_field_filter_season = dfgrp
        self.median_m5_field_filter = dfgrp_season
        self.median_m5_filter = dfgrp_filter

    def plot(self,dfgrp,dfgrp_season):

        fontsize = 12
        figres, axres = plt.subplots()
        for band in dfgrp['filter'].unique():
            dffilt = dfgrp[dfgrp['filter']==band]
            dffilt_season = dfgrp_season[dfgrp_season['filter']==band]
            """ this is to plot per band and display all seasons - check dispersion
            fig, ax = plt.subplots()
            fig.suptitle('{} band'.format(band))
            ax.plot(dffilt['fieldname'],dffilt['fiveSigmaDepth'],'ko',mfc='None')
            ax.plot(dffilt_season['fieldname'],dffilt_season['fiveSigmaDepth'],'rs')
            """
            axres.plot(dffilt_season['fieldname'],dffilt_season['fiveSigmaDepth'],marker='s',color=filtercolors[band],label='{} band'.format(band))

        axres.legend(loc='upper left', bbox_to_anchor=(0.01, 1.15),
                     ncol=4, fancybox=True, shadow=True, fontsize=fontsize)
        axres.set_ylabel('median m$_{5}^{single visit}$ [ten seasons]',fontsize=fontsize)
        axres.grid()

    




"""
for fieldname in np.unique(medValues['fieldname']):
    idf = medValues['fieldname']==fieldname
    print(fieldname)
    sel = metricValues[idf]
    for band in np.unique(sel['filter']):
        idfb = sel['filter'] == band
        selb = sel[idfb]
        for season in
"""

class RestFrameBands:

    def __init__(self,blue_cutoff=350.,red_cutoff=800.):

        self.blue_cutoff = blue_cutoff
        self.red_cutoff = red_cutoff
        self.bands = 'ugrizy'

        self.wave_frame = self.waves()

        self.zband = self.zcutoff()


    def waves(self):
        
        
        zvals = np.arange(0.01,1.2,0.01)

        telescope = Telescope(airmass=1.2)
        wave_frame = None
        for band in self.bands:
            mean_restframe_wavelength = telescope.mean_wavelength[band] /(1. + zvals)
            arr = np.array(mean_restframe_wavelength,dtype=[('lambda_rf','f8')])
            arr = rf.append_fields(arr,'band',[band]*len(arr)) 
            arr = rf.append_fields(arr,'z',zvals)
            #print(band,mean_restframe_wavelength)
            if wave_frame is None:
                wave_frame=arr
            else:
                wave_frame = np.concatenate((wave_frame,arr))

        return wave_frame

    def zcutoff(self):

        # now get the redshifts limits corresponding to the blu and red cutoffs

        r = []
        for band in self.bands:
            idx = self.wave_frame['band'] == band
            selw = self.wave_frame[idx]
            interp = interpolate.interp1d(selw['lambda_rf'],selw['z'],bounds_error=False,fill_value=(1.5,0.0))
            #print(band,interp(self.blue_cutoff),interp(self.red_cutoff))
            r.append((band,interp(self.blue_cutoff),interp(self.red_cutoff)))

        return np.rec.fromrecords(r, names=['band','z_blue','z_red'])

    def plot(self):


        fontsize = 12

        figw, axw = plt.subplots()
    
        for band in self.bands:
            ig = self.wave_frame['band']==band
            selig = self.wave_frame[ig]
            axw.plot(selig['lambda_rf'],selig['z'],color=filtercolors[band],label=band)

        axw.plot([self.blue_cutoff]*2,[0.,1.2],color='r',ls='--')
        axw.plot([self.red_cutoff]*2,[0.,1.2],color='r',ls='--')
        axw.set_xlabel(r'$\lambda^{LSST\ mean\ band}_{rf}$ [nm]', fontsize=fontsize)
        axw.set_ylabel('z', fontsize=fontsize)
        axw.set_ylim([0.,1.2])
        axw.legend(loc = 'upper right')
        axw.grid()


class Visits_z:

    def __init__(self,x1,color,fluxname,cadences=[1.,4.],plot=False):

        self.cadences = cadences

        # estimate m5 vs zlim
        myclass = Lims_z_m5(x1,color,fluxname)
        zlims = myclass.process(cadences)
       
        #plot the results
        if plot:
            myclass.plot(zlims,cadences,locx=0.7,locy=24.)

        print('after plot')
        #plt.show()
        #to convert m5 values to nvisits: need m5 one visit

        # load (and plot) med values

        finalMed = AnaMedValues('medValues.npy',plot=plot).medValues

        m5_median = dict(zip(finalMed['filter'],finalMed['fiveSigmaDepth']))

        print('median m5',m5_median)
        # now convert m5 to a number of visits
        nvisits_z = nvisits(zlims,m5_median)

        idd = nvisits_z['band']=='r'
        print(nvisits_z[idd][['band','cadence_mean','nvisits','zlim']])
        #remove outside range results
        idx = (nvisits_z['nvisits']>0)&(nvisits_z['zlim']>0.)
        nvisits_z = nvisits_z[idx]

        #print(nvisits_z[['band','cadence_mean','nvisits','zlim_SNCosmo']])

        # plot the results

        
        if plot:
            myclass.plot(nvisits_z,cadences,'nvisits','Nvisits',locy=500.)

        self.nvisits_tot_z = self.getNvisits_all_z(nvisits_z)
        if plot:
            myclass.plot(self.nvisits_tot_z,cadences,'nvisits','Nvisits',yscale='linear',locy=500)


    def getNvisits_all_z(self,nvisits_z):

        # now estimate the total number of visits vs z
        # two redshift range
        # [..,0.7]: riz bands
        # [0.7,...]: izy bands
        
        print('getting all visits')

        ida = nvisits_z['band'] != 'y'
        idb = (nvisits_z['band'] == 'y')&(nvisits_z['zlim'] >0.7)
        
        nvisits_all_z = np.concatenate((nvisits_z[ida],nvisits_z[idb]))
        
        z = np.arange(0.3,0.9,0.001)

        ssum = None

        arr_tot = None
        for cadence in self.cadences:
            ssum = None
    
            for band in 'rizy':
   
                ik = nvisits_all_z['band'] == band
                ik &= np.abs(nvisits_all_z['cadence_mean']-cadence)<1.e-5
                selb = nvisits_all_z[ik]
                print('hello',cadence,band,len(selb))
                f = interpolate.interp1d(selb['zlim'],
                                         selb['nvisits'],
                                         bounds_error=False,
                                         fill_value=0.)
                if ssum is None:
                    ssum = np.array(f(z))
                else:
                    ssum += np.array(f(z))
            arr_cad = np.array(z,dtype=[('zlim','f8')])
            arr_cad = rf.append_fields(arr_cad,'nvisits',ssum)
            arr_cad = rf.append_fields(arr_cad,'cadence_mean',[cadence]*len(arr_cad))
            if arr_tot is None:
                arr_tot = arr_cad
            else:
                arr_tot=np.concatenate((arr_tot,arr_cad))

        return arr_tot


class DDbudget_zlim:

    def __init__(self,x1=-2.0,color=0.2,fluxfile=''):
        
        # get the total number of visits per obs night
        
        print('visits per night')
        visits_per_night = Visits_z(x1,color,fluxfile,plot=True).nvisits_tot_z
        
        #plt.show()
        #first thing to be done: interplinear of nvisits vs z
        
        self.interp = {}
        self.reverse_interp = {}
        
        for cad in np.unique(visits_per_night['cadence_mean']):
            idx = visits_per_night['cadence_mean']==cad
            sel = visits_per_night[idx]
            self.interp['{}'.format(cad)] = interpolate.interp1d(sel['zlim'],
                                                            sel['nvisits'],
                                                            bounds_error=False,
                                                            fill_value=0.)
            self.reverse_interp['{}'.format(cad)] = interpolate.interp1d(sel['nvisits'],
                                                                sel['zlim'],
                                                                bounds_error=False,
                                                                fill_value=0.)

        """
        # define scenarios
        dict_scen = self.scenarios()

        #estimate budget vs zlim
        res = self.calc_budget_zlim(dict_scen)

        # plot the results
        Nvisits=2774123
        Nvisits = 2388477
        self.plot(res,Nvisits=Nvisits)
        """
        
    def scenarios(self):
        
        dict_scen = {}
        names = ['fieldname','Nfields','cadence','season_length','Nseasons','weight_visit']

        r = []
        r.append(('LSSTDDF',4,4.,6.0,10,1))
        r.append(('ADFS',2,4.,6.0,10,1))

        dict_scen['scen1'] = np.rec.fromrecords(r,names=names)

        r = []
        r.append(('LSSTDDF',4,3.,6.0,10,1))
        r.append(('ADFS',2,3.,6.0,10,1))

        dict_scen['scen2'] = np.rec.fromrecords(r,names=names)

        r = []
        r.append(('LSSTDDF',4,4.,6.0,2,1))
        r.append(('ADFS',2,4.,6.0,2,1))

        dict_scen['scen3'] = np.rec.fromrecords(r,names=names)

        r = []
        r.append(('LSSTDDF',4,3.,6.0,2,1))
        r.append(('ADFS',2,3.,6.0,2,1))

        dict_scen['scen4'] = np.rec.fromrecords(r,names=names)

        r = []
        r.append(('LSSTDDF',4,3.,6.0,2,1))
        r.append(('ADFS',2,4.,6.0,2,2))

        dict_scen['scen5'] = np.rec.fromrecords(r,names=names)

        return dict_scen


    def calc_budget_zlim(self,dict_scen):
    
        z = np.arange(0.3,0.89,0.01)
        res_tot = None
        for key, scen in dict_scen.items():
            arr_visits = None
            arr_frac = None
            res_scen = None
            for field in scen:
                cad = np.unique(field['cadence'])[0]
                fieldname = field['fieldname']
                print('hello',z,cad,self.interp.keys())

                zvals = np.copy(z)
                if field['weight_visit']>1:
                    #here we have to find the z range corresponding to this situation
                    # grab the number of visits from the other field
                    """
                    ik = scen['fieldname'] == 'LSSTDDF'
                    scensel = scen[ik]
                    cadb = scensel['cadence'][0]
                    """
                    nvisits = self.interp['{}'.format(cad)](z)
                    zvals = self.reverse_interp['{}'.format(cad)](2.*nvisits)

                nvisits = self.interp['{}'.format(cad)](zvals)
                nvisits*=field['season_length']*field['Nfields']*field['Nseasons']*30./field['cadence']
                #frac = nvisits/Nvisits
                print(nvisits)
                if arr_visits is None:
                    arr_visits = np.array(nvisits)
                    #arr_frac = np.array(frac)
                else:
                    arr_visits += nvisits
                    #arr_frac +=frac
                if res_scen is None:
                    res_scen = np.array(zvals,dtype=[('zlim_{}'.format(fieldname),'f8')])
                    res_scen = rf.append_fields(res_scen,'nvisits_{}'.format(fieldname),nvisits)
                else:
                    res_scen = rf.append_fields(res_scen,'zlim_{}'.format(fieldname),zvals)
                    res_scen = rf.append_fields(res_scen,'nvisits_{}'.format(fieldname),nvisits)
    
            res_scen = rf.append_fields(res_scen,'nvisits',arr_visits)
            #res_scen = rf.append_fields(res_scen,'frac_DDF',arr_frac)
            res_scen = rf.append_fields(res_scen,'scenario',[key]*len(res_scen))

            if res_tot is None:
                res_tot = res_scen
            else:
                res_tot = np.concatenate((res_tot,res_scen))
    
        return res_tot

    def plot(self,res_tot,Nvisits):


        fig, ax = plt.subplots()

        for scen in np.unique(res_tot['scenario']):
            iu = res_tot['scenario']==scen
            sel = res_tot[iu]
            sel.sort(order='nvisits')
            ia = (sel['zlim_ADFS']>0)&(sel['nvisits_ADFS']>0)
            sela = sel[ia]
            sela.sort(order='zlim_ADFS')
            label ='{}-{}'.format(scen,'any field')
            if scen == 'scen5':
                label = '{}-{}'.format(scen,'ADFS')
            ax.plot(sela['zlim_ADFS'],100.*sela['nvisits']/Nvisits,label=label)
            if scen == 'scen5':
                label = '{}-{}'.format(scen,'LSSTDDF')
                ib = (sel['zlim_LSSTDDF']>0)&(sel['nvisits_LSSTDDF']>0)
                selb = sel[ib]
                selb.sort(order='zlim_LSSTDDF')
                ax.plot(selb['zlim_LSSTDDF'],100.*selb['nvisits']/Nvisits,label=label)
                print(selb[['zlim_LSSTDDF','nvisits_LSSTDDF','nvisits_ADFS','nvisits']])

        ax.set_ylim([0.,6.0])
        ax.set_xlim([0.3,0.85])
        ax.legend()
        ax.grid()
        ax.plot([0.3,0.85],[4.5,4.5],color='b',ls='--')
        ax.text(0.4,4.6,'AGN White paper (Nvisits)',color='b')
        ax.set_xlabel(r'$z_{lim}$')
        ax.set_ylabel(r'DD budget [%]')
        

class anadf:

    def __init__(self,datadf):
    
        #datadf = pd.DataFrame(data.to_pandas())

        # calc SNR to check the background dominating regime
        self.grpfi = self.calcSNR(datadf)

        # estimate redshift limits
        
        dfzlim = self.zlim(self.grpfi)

        # estimate SNR per band

        dfsnr = self.SNR_band(self.grpfi)

        # merge the results in a single df

        self.dfana = dfsnr.merge(dfzlim,left_on=['x1','color'],right_on=['x1','color'])

        print(self.grpfi)
        print(self.dfana)


    def calcSNR(self,datadf):

        grpfi = pd.DataFrame()

        for key, grp in datadf.groupby(['x1', 'color']):
     
            if key[0] < 0.:
                idx = grp['z'] <= 0.9
    
            if key[0] >= 0.:
                idx = grp['z'] >= 0.05

            grp = grp[idx]

            grp.loc[:, 'err_back_reg'] = (5.*grp['flux_e_sec']/grp['flux_5'])**2.
        
            grpa = grp.groupby(['daymax','z']).sum().reset_index()
            grpa['sigmaC'] = np.sqrt(covColor(grpa))
            grpa = grpa[['daymax','z','sigmaC']]
        
            grpb = grp.groupby(['daymax','z','band']).sum().reset_index()
        
            grpb = grpb[['daymax','z','band','err_back_reg']]
            grpb['SNR'] = np.sqrt(grpb['err_back_reg'])
            grpbm = grpb.merge(grpa,left_on=['daymax','z'],right_on=['daymax','z'])
            grpbm.loc[:,'x1'] = key[0]
            grpbm.loc[:,'color'] = key[1]
            grpfi = pd.concat([grpfi, grpbm], sort=False)

        return grpfi

        #analyze this df to get values corresponding to sigma_c=0.04

        # get zlimits

    def zlim(self,grpfi):

        r = []
        for key,grp in grpfi.groupby(['x1','color']):
      
            interp = interpolate.interp1d(grp['sigmaC'],grp['z'],bounds_error=False,fill_value=0.0)
            resinterp = interp(0.04)
            r.append([key[0],key[1],np.round(resinterp,2)])
        dfana = pd.DataFrame(r,columns=('x1','color','zlim'))
        
        return dfana

    def SNR_band(self,grpfi):
        
        # get SNR
        
        dfb = grpfi.groupby(['x1','color','band'])
        
        
        r =[]
        for key, grpt in dfb:
            #band = '{}'.format(key[2].decode()[-1])
            band = key[2]
            interp = interpolate.interp1d(grpt['sigmaC'].values, grpt['SNR'].values,bounds_error=False,fill_value=0.0)
            SNR = interp(0.04)
            r.append([key[0],key[1],band,int(SNR)])

        dfsnr = pd.DataFrame(r,columns=('x1','color','band','SNR'))

        return dfsnr

        
            
    def plotzlim(self):

        fig, ax = plt.subplots(nrows=2)
        ls = dict(zip([(0.0,0.0),(-2.0,0.2)],['-','--']))
    
        for key,grp in self.grpfi.groupby(['x1','color']):
            ax[0].plot(grp['z'], grp['sigmaC'],ls=ls[(key[0],key[1])],color='k',label='(x1,c)=({},{})'.format(key[0],key[1]))
            
            idx = (self.dfana['x1']-key[0])<1.e-5
            idx &= (self.dfana['color']-key[1])<1.e-5

            zlim = self.dfana[idx]['zlim'].unique()
            ylims = ax[0].get_ylim()
            ax[0].plot([zlim,zlim],ylims,color='r',ls='dotted')
            ax[0].text(zlim-0.05,ylims[1],'{}={}'.format('$z_{lim}$',np.round(zlim,2)))

        ylims = ax[0].get_ylim()
        xlims = ax[0].get_xlim()
        ax[0].plot(xlims,[0.04,0.04],color='r',ls='dotted')
        ax[0].set_xlim(0.,1.)
        ax[0].set_ylim(0.,ylims[1])
        ax[0].set_xlabel('z')
        ax[0].set_ylabel(r'$\sigma_{C}$')
        ax[0].grid()
        ax[0].legend(loc='upper left')
    
        dfb = self.grpfi.groupby(['x1','color','band'])

        for key, grpt in dfb:
            #band = '{}'.format(key[2].decode()[-1])
            band = key[2]
            idx = (self.dfana['x1']-key[0])<1.e-5
            idx &= (self.dfana['color']-key[1])<1.e-5
            idx &= self.dfana['band'] == band

            
            SNR = self.dfana[idx]['SNR'].values

            if SNR >0.5:
                ax[1].plot(grpt['sigmaC'].values, grpt['SNR'].values, ls=ls[(key[0],key[1])],
                           color=filtercolors[band],label='$SNR_{}$={}'.format(band,int(SNR)))
            else:
                ax[1].plot(grpt['sigmaC'].values, grpt['SNR'].values, ls=ls[(key[0],key[1])],
                           color=filtercolors[band])

        ax[1].set_ylim(0.0,100.)
        ax[1].set_xlim(0.0,0.10)
        ax[1].plot([0.04,0.04],ax[1].get_ylim(),color='r',ls='dotted')
        ax[1].grid()
        ax[1].set_xlabel(r'$\sigma_{C}$')
        ax[1].set_ylabel(r'SNR')
        ax[1].legend(ncol=2)

class SNR_z:

    def __init__(self,dirFile,data,
                 SNR_par={}):
        
        self.x1 = data.x1
        self.color = data.color
        self.bands = data.bands
        self.SNR_par = SNR_par
        """
        self.zband = zband # z cutoff per band
        """
       
        self.lcdf = data.lc
        self.medm5 = data.m5_Band

        self.fracSignalBand = data.fracSignalBand.fracSignalBand
        #self.medm5 = medm5

        # Load LC template

        #lcData = self.load(dirFile,lcName)

        # select data for the type of SN (x1,color)
        """
        idx = (lcData['x1']-x1)<1.e-1
        idx &= (lcData['color']-color)<1.e-1
        lcData = lcData[idx]
        """
        #remove lc points with negative flux

        """
        idx = lcData['flux_e_sec']>=0
        lcData = lcData[idx]
        """

         
        # transform initial lcData to a pandas df
        """
        lcdf = pd.DataFrame(np.copy(lcData))
        lcdf['band'] = lcdf['band'].map(lambda x: x.decode()[-1])
        """
        #lcData['band'] = [b.decode()[-1] for b in lcData['band']]

        # apply wavelenght cutoffs (blue and red cutoff)
        # and remove bands not wanted

        #self.lcdf = self.wave_cutoff(lcdf)

        #print('there',self.lcdf)

        # get the signal distribution (fraction) per band vs redshift
        # this is useful if SNR_choice = fracflux is chosen

        #self.fracSignalBand = SignalBand(self.lcdf).fracSignalBand


        # this list is requested to estimate Fisher matrices

        self.listcol = ['F_x0x0', 'F_x0x1',
                        'F_x0daymax', 'F_x0color', 
                        'F_x1x1', 'F_x1daymax', 
                        'F_x1color','F_daymaxdaymax', 
                        'F_daymaxcolor', 'F_colorcolor']

        # map flux5-> m5

        self.f5_to_m5 = flux5_to_m5(self.bands)

    """
    def load(self,theDir,fileName):
         
        searchname = '{}/{}'.format(theDir, fileName)
        name, ext = os.path.splitext(searchname)

        print(searchname)
        res = loopStack([searchname], objtype='astropyTable')
    
        return res
        
    def wave_cutoff(self,df):

        # select obs with bands in self.bands

        df['selband'] = df['band'].isin(list(self.bands))

        idx = df['selband'] == True

        df = df[idx]

        # select zband vals wuith band in self.bands

        zbanddf = pd.DataFrame(np.copy(self.zband))

        zbanddf['selband'] = zbanddf['band'].isin(list(self.bands))

        idx = zbanddf['selband'] == True

        zbanddf = zbanddf[idx]

        # now merge the two dfs

        res = df.merge(zbanddf,left_on=['band','selband'],right_on=['band','selband'])

        # select obs with z > z_blue (blue cutoff)

        idx = res['z']<= res['z_blue']
        idx &= res['z'] >= res['z_red']

        res = res[idx]

        #remove selband and z_blue and z_red
        res = res.drop(columns=['selband','z_blue','z_red'])
        
        return res

       
    def signalBand(self):

        #df = pd.DataFrame(np.copy(tab))
        
        sumflux_band_z = self.lcdf.groupby(['x1','color','band','z'])['flux_e_sec'].sum().reset_index()
        
        print(type(sumflux_band_z))

        sumflux_z  = self.lcdf.groupby(['x1','color','z'])['flux_e_sec'].sum().reset_index()
    
        sumflux_z = sumflux_z.rename(columns={"flux_e_sec": "flux_e_sec_tot"})

        sum_merge= sumflux_band_z.merge(sumflux_z,left_on=['x1','color','z'],right_on=['x1','color','z'])
          
        print(sum_merge)
        #sum_merge['band'] = sum_merge['band'].map(lambda x: x.decode()[-1])
        sum_merge['fracfluxband'] = sum_merge['flux_e_sec']/sum_merge['flux_e_sec_tot']
        
        return sum_merge

    
    def plotSignalBand(self):

        sel = self.fracSignalBand
            
        fig, ax = plt.subplots()
        figtitle = '(x1,color)=({},{})'.format(self.x1,self.color)
        fig.suptitle(figtitle)
        
        for b in sel['band'].unique():
            ik = sel['band'] == b
            selb = sel[ik]
            ax.plot(selb['z'],selb['fracfluxband'],color=filtercolors[b],label=b)


        ax.legend(ncol=5)
        ax.set_xlabel(r'z')
        ax.set_xlim(0.01,0.9)
        ax.set_ylabel(r'Flux fraction per band')
        ax.grid()
    """
    def sumgrp(self,grp):

        sumli = grp[self.listcol].sum()
        sumflux = np.sqrt(np.sum(grp['flux_e_sec']**2.))
        SNR = 5.*sumflux/np.mean(grp['flux_5'])

        dicta = {}

        for ll in self.listcol:
            dicta[ll] = [sumli[ll]]

        dicta['sumflux'] = [np.sqrt(np.sum(grp['flux_e_sec']**2.))]
        dicta['SNR'] = [5.*sumflux/np.mean(grp['flux_5'])]
        dicta['flux_5'] = np.mean(grp['flux_5'])
   
        return pd.DataFrame.from_dict(dicta)


    def sigmaC_SNR(self):

        # make the sum of the self.listcol values
        boo = self.lcdf['band'] == 'r'
        boo &= np.abs(self.lcdf['z']-0.1)<1.e-5
        print('hhh',self.lcdf[boo][['x1','color','z','band','time','flux_e_sec']])


        df = self.lcdf.groupby(['x1','color','z','band']).apply(lambda x: self.sumgrp(x)).reset_index()
        print('rrr',df[['x1','color','z','band','sumflux','SNR','flux_5']])

        # make groups and estimate sigma_Color for combinations of SNR per band

        dfsigmaC = df.groupby(['x1','color','z']).apply(lambda x: self.sigmaC(x)).reset_index()
 
        cols = ['z']
        for val in ['SNRcalc','flux_5_e_sec']:
            for b in self.bands:
                cols.append('{}_{}'.format(val,b))

        print(dfsigmaC[cols])
        return dfsigmaC

    def sigmaC(self,grp):

        # init SNR values to zero
        SNR = {}
        for b in self.bands:
            SNR[b]=[0.0]

        # identify bands of interest and set SNRs
        # bands not present have SNR equal to 0.0
        dictband = {}

        SNR_min = 10.
        SNR_max = self.SNR_par['max']
        
        if grp.name[2]>=0.65:
            SNR_min = 20.
        

        for band in grp['band'].unique():
            idx = grp['band'] == band
            dictband[band] = grp[idx]
            SNR[band]=list(np.arange(SNR_min,SNR_max,self.SNR_par['step']))
            if band == 'y':
                snrlist = list(np.arange(0.,SNR_max,self.SNR_par['step']))
                snrlist[0] = 0.0001
                SNR[band] = snrlist

        #SNR = dict(zip('grizy',[[0.],[25.],[25.],[30.],[35.]]))        


        # existing bands
        bands = ''.join(list(dictband.keys()))

        # missing bands
        missing_bands = list(set(self.bands).difference(bands))
                         
    
        # We are going to build a df with all SNR combinations
        # let us start with the first band
        df_ref = dictband[bands[0]]

        # make the SNR combination for this first band
        df_ref = self.addSNR(df_ref,SNR[bands[0]],bands[0])

        # now make all the combinations
        df_tot = pd.DataFrame()
        df_merged = df_ref.copy()
        time_ref = time.time()
    
        for i in range(1,len(bands)):
            b = bands[i]
            df_to_merge = self.addSNR(dictband[b],SNR[b],b)
            if not df_tot.empty:
                df_merged = df_tot.copy()
            dfb = pd.DataFrame()
            for ikey in df_to_merge['key'].values:
                df_merged.loc[:,'key'] = ikey
                dfb = pd.concat([dfb,df_merged],sort=False)
        
            df_tot = dfb.merge(df_to_merge,left_on=['key'],right_on=['key'])

        print('after hh',time.time()-time_ref)

        listb = []
        for b in bands:
            listb.append('SNRcalc_{}'.format(b))
    

        df_tot['SNRcalc_tot'] = np.sqrt((df_tot[listb]*df_tot[listb]).sum(axis=1))
        df_tot['bands'] = ''.join(bands)

        time_ref = time.time()
    

        # Estimate sigma_Color for all the combinations build at the previous step
        for col in self.listcol:
            df_tot.loc[:,col]=df_tot.filter(regex='^{}'.format(col)).sum(axis=1)

        df_tot['sigmaC'] = np.sqrt(covColor(df_tot))

        
        #add the missing bands to have a uniform format z-independent
        for b in missing_bands:
            for col in self.listcol:
                df_tot.loc[:,'{}_{}'.format(col,b)] = 0.0 
            df_tot.loc[:,'SNRcalc_{}'.format(b)] = 0.0
            df_tot.loc[:,'flux_e_sec_{}'.format(b)] = 0.0
            df_tot.loc[:,'m5calc_{}'.format(b)] = 0.0

       

        # that's it - return the results
        return df_tot

    def addSNR(self,df,SNR,b):

        r = []
        df_tot = pd.DataFrame()

        for i,val in enumerate(SNR):
            r.append((i,val))
            df_cp = df.copy()
            df_cp.loc[:,'key'] = i
            df_tot = pd.concat([df_tot,df_cp])


        df_SNR = pd.DataFrame(r, columns=['key','SNRcalc'])
    
        df_tot = df_tot.merge(df_SNR,left_on='key',right_on='key')

        #df_tot = df_tot.drop(columns=['key'])

        # get the 5-sigma flux according to SNR vals
        df_tot['flux_5_e_sec']= 5.*df_tot['sumflux']/df_tot['SNRcalc']

        # get m5 from the 5-sigma flux
        df_tot['m5calc'] = self.f5_to_m5[b](df_tot['flux_5_e_sec'])

        # get the ratio of the 5-sigma fluxes: the one estimated two lines ago
        # and the original one, that is the one used for the original simulation
        # (and estimated from m5)

        df_tot['ratiof5'] = (df_tot['flux_5_e_sec']/df_tot['flux_5'])**2

        # correct Fisher matrix element from this ratio
        # so that these elements correspond to SNR values
        for col in self.listcol:
            df_tot[col] = df_tot[col]/df_tot['ratiof5']

        # drop the column used for the previous estimation
        df_tot = df_tot.drop(columns=['ratiof5'])

        # add suffix corresponding to the filter
        df_tot = df_tot.add_suffix('_{}'.format(b))
        # key here necessary for future merging
        df_tot = df_tot.rename(columns={'key_{}'.format(b):'key'})

        #that is it - return the result
        return df_tot

    def plotSigmaC_SNR(self, dfsigmaC):

        #figb, axb = plt.subplots()
        
        for z in dfsigmaC['z'].unique():
            ii = dfsigmaC['z'] == z
            sel = dfsigmaC[ii]
            fig, ax = plt.subplots()
            fig.suptitle('z={}'.format(np.round(z,2)))
            ax.plot(sel['SNRcalc_tot'],sel['sigmaC'],'ko',mfc='None')
            #ax.plot(sel['Nvisits'],sel['sigmaC'],'ko',mfc='None')
            ax.set_xlabel(r'SNR$_{tot}$')
            ax.set_ylabel(r'$\sigma_{C}$')
            xlims = ax.get_xlim()
            ax.plot(xlims,[0.04]*2,color='r')
            #ax.legend()

    def SNR(self,dfsigmaC):

        # select only combi with sigma_C ~ 0.04
        idx = np.abs(dfsigmaC['sigmaC']-0.04)<0.0004
        #idx = np.abs(dfsigmaC['sigmaC']-0.04)<0.1

        sel = dfsigmaC[idx].copy()

        # for these combi : estimate the SNR frac per band
        
        
        for b in self.bands:
            sel.loc[:,'fracSNR_{}'.format(b)] = sel['SNRcalc_{}'.format(b)]/sel['SNRcalc_tot']
            if self.medm5 is not None:
                sel.loc[:,'m5single_{}'.format(b)] = self.medm5[self.medm5['filter']==b]['fiveSigmaDepth'].values


        if self.SNR_par['choice'] == 'fracflux':
            res = sel.groupby(['x1','color','z']).apply(lambda x: self.SNR_redshift(x)).reset_index()
        else:
            res = sel.groupby(['x1','color','z']).apply(lambda x: self.SNR_visits(x)).reset_index()

        
        return res

    def SNR_visits(self,grp):

        cols = []
        for b in self.bands:
            cols.append('Nvisits_{}'.format(b))

        for b in self.bands:
            #print(grp[['m5calc_{}'.format(b),'m5single_{}'.format(b)]])
            grp.loc[:,'Nvisits_{}'.format(b)]=10**(0.8*(grp['m5calc_{}'.format(b)]-grp['m5single_{}'.format(b)]))
            grp['Nvisits_{}'.format(b)]=grp['Nvisits_{}'.format(b)].fillna(0.0)
            #print(b,grp['Nvisits_{}'.format(b)])
            #print(test)
        grp.loc[:,'Nvisits'] = grp[cols].sum(axis=1)

        idx = int(grp[[self.SNR_par['choice']]].idxmin())

        """
        if grp.name[2] > 0.7:
            print(grp['Nvisits'],idx)
            
            cols = []

            for b in self.bands:
                cols.append('Nvisits_{}'.format(b))
            print(grp.loc[idx,cols])
        
            print(test)
        """
        #print(grp.columns)
        cols =  []
        for colname in ['SNRcalc','m5calc','fracSNR','flux_5_e_sec']:
            for band in 'grizy':
                cols.append('{}_{}'.format(colname,band))
        
        #colindex = [grp.columns.get_loc(c) for c in cols if c in grp]
            
        return grp.loc[idx,cols]
        
 
    def SNR_redshift(self,grp):
        
        #print('ooo',grp.name,type(grp.name))
        ik = np.abs(self.fracSignalBand['z']-grp.name[2])<1.e-5
        selband = self.fracSignalBand[ik]
        
        r = []
        for b in self.bands:
            iop = selband['band'] == b
            if len(selband[iop])==0:
                r.append(0.0)
            else:
                r.append(selband[iop]['fracfluxband'].values[0])

        #print(r,grp[['fracSNR_i','fracSNR_z']].shape)
    
        cldist = []
        for b in self.bands:
            cldist.append('fracSNR_{}'.format(b))

        closest_index = distance.cdist([r], grp[cldist]) #.argmin()

        cols =  []
        for colname in ['SNRcalc','m5calc','fracSNR','flux_5_e_sec']:
            for band in self.bands:
                cols.append('{}_{}'.format(colname,band))
    
        colindex = [grp.columns.get_loc(c) for c in cols if c in grp]
        
            
        return grp.iloc[closest_index.argmin(),colindex]
       
class SignalBand:

    def __init__(self, lcdf):

        #df = pd.DataFrame(np.copy(tab))
        
        sumflux_band_z = lcdf.groupby(['x1','color','band','z'])['flux_e_sec'].sum().reset_index()
        
        print(type(sumflux_band_z))

        sumflux_z  = lcdf.groupby(['x1','color','z'])['flux_e_sec'].sum().reset_index()
    
        sumflux_z = sumflux_z.rename(columns={"flux_e_sec": "flux_e_sec_tot"})

        sum_merge= sumflux_band_z.merge(sumflux_z,left_on=['x1','color','z'],right_on=['x1','color','z'])
          
        print(sum_merge)
        #sum_merge['band'] = sum_merge['band'].map(lambda x: x.decode()[-1])
        sum_merge['fracfluxband'] = sum_merge['flux_e_sec']/sum_merge['flux_e_sec_tot']
        
        self.fracSignalBand = sum_merge

    
    def plotSignalBand(self,x1,color):

        sel = self.fracSignalBand
            
        fig, ax = plt.subplots()
        figtitle = '(x1,color)=({},{})'.format(x1,color)
        fig.suptitle(figtitle)
        
        for b in sel['band'].unique():
            ik = sel['band'] == b
            selb = sel[ik]
            ax.plot(selb['z'],selb['fracfluxband'],color=filtercolors[b],label=b)


        ax.legend(ncol=5)
        ax.set_xlabel(r'z')
        ax.set_xlim(0.01,0.9)
        ax.set_ylabel(r'Flux fraction per band')
        ax.grid()

def plotSNR_z(SNR,x1,color,bands):

    fig, ax = plt.subplots()
    fig.suptitle('(x1,color)=({},{})'.format(x1,color))
    for b in bands:
        ax.plot(SNR['z'],SNR['SNRcalc_{}'.format(b)],color=filtercolors[b],label=b)

    ax.set_xlabel(r'z')
    ax.set_ylabel(r'SNR')
    ax.legend()
    ax.grid()

def flux5_to_m5(bands):

    m5_range = np.arange(20.,28.0,0.01)
    
    #estimate the fluxes corresponding to this range
    
    telescope = Telescope(airmass=1.2)
    
    f5_dict={}
    for band in bands:
        flux_5 = telescope.mag_to_flux_e_sec(m5_range,[band]*len(m5_range),[30.]*len(m5_range))[:,1]
        f5_dict[band] = interpolate.interp1d(flux_5,m5_range,bounds_error=False,fill_value=0.0)


    return f5_dict

def m5_to_flux5(bands):

    m5_range = np.arange(20.,28.0,0.01)
    
    #estimate the fluxes corresponding to this range
    
    telescope = Telescope(airmass=1.2)
    
    m5_dict={}
    for band in bands:
        flux_5 = telescope.mag_to_flux_e_sec(m5_range,[band]*len(m5_range),[30.]*len(m5_range))[:,1]
        m5_dict[band] = interpolate.interp1d(m5_range,flux_5,bounds_error=False,fill_value=0.0)


    return m5_dict

def plotNvisits(dictplot,cad,bands='rizy',legy='Nvisits'):

    ls = ['-','--','-.',':']
    colors = ['k','b','r','g']
    fig, ax = plt.subplots()
    fig.suptitle('cadence = {} days'.format(cad))

    figb, axb = plt.subplots()
    figb.suptitle('cadence = {} days'.format(cad))

    keys = list(dictplot.keys())
    for key, data in dictplot.items():
        #for io,cad in enumerate(cadences):
        #    idx = np.abs(data['cadence']-cad)<1.e-5
        #    sel = data[idx] 
        cols = []
        for b in key.split('_')[-1]:
            cols.append('Nvisits_{}'.format(b))
        sel = data
            
        for b in key.split('_')[-1]:
            if key == keys[0]:
                ax.plot(sel['z'].values,sel['Nvisits_{}'.format(b)].values,
                        color=filtercolors[b],
                        label=b,ls=ls[keys.index(key)])
            else:
                ax.plot(sel['z'].values,sel['Nvisits_{}'.format(b)].values,
                        color=filtercolors[b],
                        ls=ls[keys.index(key)])

        sel.loc[:,'Nvisits'] = sel[cols].sum(axis=1)
        axb.plot(sel['z'].values,sel['Nvisits'].values,
                 color=colors[keys.index(key)],ls=ls[keys.index(key)],
                 label='{}'.format(key))
       
        
        figc, axc = plt.subplots()
        figc.suptitle(key)
        for b in bands:
            axc.plot(sel['z'],sel['Nvisits_{}'.format(b)]/sel['Nvisits'],color=filtercolors[b],label='{}'.format(b))

        axc.set_xlabel(r'z')
        axc.set_ylabel(r'Filter allocation')
        axc.legend()
        axc.grid()
            
        figd, axd = plt.subplots()
        figd.suptitle(key)
        for b in bands:
            axd.plot(sel['z'],sel['Nvisits_{}'.format(b)],color=filtercolors[b],label='{}'.format(b))

        axd.set_xlabel(r'z')
        axd.set_ylabel(r'N$_{visits}$/field/observing night')
        axd.legend()
        axd.grid()

    ax.legend()
    ax.grid()
    ax.set_xlabel(r'z')
    ax.set_ylabel(r'{}'.format(legy))
    
    axb.legend()
    axb.grid()
    axb.set_xlabel(r'z')
    axb.set_ylabel(r'{}'.format(legy))
    axb.set_xlim([0.3,0.85])
    rplot = []
    rplot.append((40,'6% - 10 seasons * 6 fields',-17))
    rplot.append((48,'6% - 10 seasons * 5 fields',+15))
    rplot.append((199,'6% - 2 seasons * 6 fields',-17))
    rplot.append((239,'6% - 2 seasons * 5 fields',+15))
    """
    for val in rplot:
        axb.plot(axb.get_xlim(),[val[0]]*2,color='m')
        axb.text(0.33,val[0]+val[2],val[1])
    """

    #theDir = 'MetricOutput/Fakes/NSN'
#fname = 'Fakes_NSNMetric_Fake_lc_nside_64_coadd_0_0.0_360.0_-1.0_-1.0_0.hdf5'

#myclass = SNR_z(theDir, fname)

#myclass.plotSignalBand()

#dfsigmaC = myclass.sigmaC_SNR()
 
#myclass.plotSigmaC_SNR(dfsigmaC)

#SNR = myclass.SNR(dfsigmaC)

#print(SNR)

#restframeBands()

#DDbudget_zlim()

#plt.show()

#dbDir = '/sps/lsst/cadence/LSST_SN_PhG/cadence_db/fbs_1.3'
#dbExtens = 'db'

#dbName = 'baseline_v1.3_10yrs'

#n_clusters = 5

#ana = AnaOS(dbDir, dbName,dbExtens,n_clusters)

#print('Total number of visits',ana.nvisits_DD+ana.nvisits_WFD,1./(1.+ana.nvisits_WFD/ana.nvisits_DD))

#plt.show()

