import numpy as np
import pandas as pd
from scipy import interpolate

from sn_tools.sn_calcFast import covColor
from . import plt, filtercolors

class AnaMedValues:

    def __init__(self,fname):
        """
        class to load and analyse a file
        with fiveSigmadepth values

        Parameters
        ----------
        fname: str
         file name


        """

        # load the data
        medValues = np.load(fname)

        # mv to pandas df format
        df = pd.DataFrame(np.copy(medValues))
        df['fieldname'] = df['fieldname'].map(lambda x: x.strip())
        df['season']=df['season'].astype(int)

        # remove u band
        df = df[(df['filter']!='u')]

        # estimate median per fieldname, season, filter
        dfgrp = df.groupby(['fieldname','season','filter']).median().reset_index()

        # estimate median per fieldname,filter
        dfgrp_season = df.groupby(['fieldname','filter']).median().reset_index()
        dfgrp_season.loc[:,'season'] = 0

        dfgrp_filter =  df.groupby(['filter']).median().reset_index()
        dfgrp_filter.loc[:,'fieldname'] = 'all'
        dfgrp_filter.loc[:,'season'] = 0

        # three possible values available
        self.median_m5_field_filter_season = dfgrp
        self.median_m5_field_filter = dfgrp_season
        self.median_m5_filter = dfgrp_filter

    def plot(self,dfgrp,dfgrp_season):
        """
        Method to plot median m5 vs fieldname

        Parameters
        ----------
        dfgrp, dfgrp_season: pandas df
         data to display
       

        Returns
        -------
        plot of m5 vs fieldname (per band)

        """

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

class Anadf:

    def __init__(self,datadf):
    
        """
        class to analyze LC :
        - redshift limit estimation

        Parameters
        ---------
        datadf: pandas df
         data to analyze

        """

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
        """
        Method to estimate SNR values and sigmaC

        Parameters
        ----------

        datadf: pandas df
         data to analyze
        
        Returns
        -------
        grpfi: pandas df
         a completed pandas df
        
        """


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
        """
        Method to estimate the redshift limit corresponding to sigmaC~0.04
        
        Parameters
        ----------
        grpfi: pandas df
         data to analyze


        Returns
        ------
        dfana: pandas df
         with the following columns: 'x1','color','zlim'

        """

        r = []
        for key,grp in grpfi.groupby(['x1','color']):
      
            interp = interpolate.interp1d(grp['sigmaC'],grp['z'],bounds_error=False,fill_value=0.0)
            resinterp = interp(0.04)
            r.append([key[0],key[1],np.round(resinterp,2)])
        dfana = pd.DataFrame(r,columns=('x1','color','zlim'))
        
        return dfana

    def SNR_band(self,grpfi):
        """
        Method to assess SNR per band
        corresponding to sigmaC ~ 0.04

        Parameters
        ----------
        grpfi: pandas df
         data to consider

        Returns
        -------
        dfsnr: pandas df
         with the following cols:'x1','color','band','SNR'

        """

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
        """
        Method to plot sigmaC vs z
        and superimpose redshift limits
        
        Parameters
        ----------

        Returns
        -------
        Two plots on one fig:
        - top: sigmaC vs z (z limits superimposed)
        - bottom: SNR-band vs sigmaC (SNR limits superimposed)

        """



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
