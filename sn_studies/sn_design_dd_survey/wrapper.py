import os
import pandas as pd
import numpy as np

from .signal_bands import RestFrameBands,SignalBand
from .ana_file import AnaMedValues,Anadf
from sn_tools.sn_io import loopStack


class Data:
    def __init__(self,theDir,fname,
                 x1=-2.0,
                 color=0.2,
                 blue_cutoff=380.,
                 red_cutoff=800.,
                 bands='grizy'):
        """
        class to handle data: 
        - LC points
        - m5 values
        - 

        Parameters
        --------------
        theDir: str
          directory where the input LC file is located
        fname: str 
          name of the input LC file
        x1: float, opt
         SN strech parameter (default: -2.0)
        color: float, opt
         SN color parameter (default: 0.2)
        blue_cutoff: float, opt
         wavelength cutoff for SN (blue part) (default: 380 nm)
        red_cutoff: float, opt
         wavelength cutoff for SN (red part) (default: 800 nm)
        bands: str, opt
         filters to consider (default: grizy)

        """


        self.x1 = x1
        self.color = color
        self.bands = bands
        self.blue_cutoff = blue_cutoff
        self.red_cutoff = red_cutoff

        # load lc
        lc = self.load_data(theDir, fname)

        # estimate zcutoff by bands
        self.zband = RestFrameBands(blue_cutoff=blue_cutoff,
                         red_cutoff=red_cutoff).zband

        #apply these cutoffs
        self.lc = self.wave_cutoff(lc)
        
        # get the flux fraction per band
        self.fracSignalBand = SignalBand(self.lc)

        # load median m5

        self.m5_FieldBandSeason, self.m5_FieldBand, self.m5_Band = self.load_m5('medValues.npy')

    def load_data(self, theDir, fname):
        """
        Method to load LC data

        Parameters
        ----------
        theDir: str
          directory where the input LC file is located
        fname: str 
          name of the input LC file

        Returns
        -----------
        pandas df with LC point infos (flux, fluxerr, ...)
        corresponding to (x1,color) parameters  
        """

        # get LC

        lcData = self.load(theDir,fname)
         
        # select data for the type of SN (x1,color)

        idx = (lcData['x1']-self.x1)<1.e-1
        idx &= (lcData['color']-self.color)<1.e-1
        lcData = lcData[idx]

        #remove lc points with negative flux

        idx = lcData['flux_e_sec']>=0
        lcData = lcData[idx]
         
        # transform initial lcData to a pandas df
        
        lcdf = pd.DataFrame(np.copy(lcData))
        lcdf['band'] = lcdf['band'].map(lambda x: x.decode()[-1])

        return lcdf
        
    def load(self,theDir,fname):
        """
        Method to load LC data

        Parameters
        ----------
        theDir: str
          directory where the input LC file is located
        fname: str 
          name of the input LC file

        Returns
        -----------
        astropy table with LC point infos (flux, fluxerr, ...)  
        """
         
        searchname = '{}/{}'.format(theDir, fname)
        name, ext = os.path.splitext(searchname)
        
        print(searchname)
        res = loopStack([searchname], objtype='astropyTable')
        
        return res

    def wave_cutoff(self,df):
        """
        Method to select lc data (here as df)
        Selection applied:
        - bands corresponding to self.bands
        - wavelenght cutoffs from self.zband

        Parameters
        ----------
        df: pandas df with LC infos

        Returns
        -----------
        pandas df (selected)   
        """

        # select obs with bands in self.bands

        df['selband'] = df['band'].isin(list(self.bands))

        idx = df['selband'] == True

        df = df[idx]

        # select zband vals with band in self.bands

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

    def load_m5(self,m5_file):

        """
        Method to load fiveSigmaDepth(m5) values

        Parameters
        ----------
        m5_file: str
         m5 file name
         

        Returns
        -----------
        median_m5_field_filter_season: pandas df
         median m5 per field per filter and per season
        median_m5_field_filter: pandas df
         median m5 per field per filter
        median_m5_filter: pandas df
         median m5 per filter

        """


        self.anamed = AnaMedValues(m5_file)

        median_m5_field_filter_season = self.anamed.median_m5_field_filter_season
        median_m5_field_filter  = self.anamed.median_m5_field_filter
    
        median_m5_filter= median_m5_field_filter.groupby(['filter'])['fiveSigmaDepth'].median().reset_index()
        #medm5_season = median_m5_season.groupby(['filter'])['fiveSigmaDepth'].median().reset_index()

        return median_m5_field_filter_season,median_m5_field_filter,median_m5_filter


    def plotzlim(self):
        """
        Method to plot zlim

        Parameters
        ----------

        Returns
        -----------
        Two plots in one figure:
        - sigma_C vs z
        - SNR_band vs sigma_C
        
        """
        Anadf(self.lc).plotzlim()

    def plotFracFlux(self):
        """
        Method to plot fraction of flux per band vs z

        Parameters
        ----------

        Returns
        -----------
        plot of the flux fraction per band vs z
        
        """

        self.fracSignalBand.plotSignalBand(self.x1,self.color)

    def plot_medm5(self):
        """
        Method to plot m5 values

        Parameters
        ----------

        Returns
        -----------
        plot of m5 values vs field
        
        """
        self.anamed.plot(self.m5_FieldBandSeason, self.m5_FieldBand)


class Nvisits_cadence:
    def __init__(self,snr_calc,cadence,m5_type,choice_type,bands):
        

        outName = 'Nvisits_cadence_{}_{}.npy'.format(choice_type,m5_type)

        if not os.path.isfile(outName):
            self.cadence = cadence
            self.snr_calc = snr_calc

            self.cols = ['z','Nvisits']
            for band in bands:
                self.cols.append('Nvisits_{}'.format(band))

            medclass = AnaMedValues('medValues.npy')
            m5 = eval('{}.{}'.format('medclass',m5_type))
            df_tot = pd.DataFrame()

            df_tot = m5.groupby(['fieldname','season']).apply(lambda x : self.getVisits(x)).reset_index()


            self.nvisits_cadence = df_tot

            np.save(outName,np.copy(df_tot.to_records(index=False)))

        else:
            self.nvisits_cadence = pd.DataFrame(np.load(outName))


    def getVisits(self,grp):

       
        
        df = Nvisits_m5(self.snr_calc,grp).nvisits
        io = np.abs(df['cadence']-self.cadence)<1.e-5
        print('iiii',df.columns)
        df = df.loc[io,self.cols]
        #df.loc[:,'fieldname'] = grp.name[0]
        #df.loc[:,'season'] = grp.name[1]
        return df

    def plot(self):

        # this for the plot
        print(self.nvisits_cadence.groupby(['fieldname','z']).apply(lambda x: np.min(x['Nvisits'])).reset_index())

        df = self.nvisits_cadence.groupby(['fieldname','z']).agg({'Nvisits': ['min','max','median']})
        # rename columns
        df.columns = ['Nvisits_min', 'Nvisits_max', 'Nvisits_median']

        # reset index to get grouped columns back
        df = df.reset_index()

        for fieldname in df['fieldname'].unique():
            io = df['fieldname']==fieldname
            sel = df[io]
            fig, ax = plt.subplots()
            fig.suptitle(fieldname)

            ax.fill_between(sel['z'],sel['Nvisits_min'],sel['Nvisits_max'],color='grey')
            ax.plot(sel['z'],sel['Nvisits_median'],color='k')

            ax.grid()
            ax.set_xlim([0.3,0.85])
            ax.set_xlabel(r'z')
            ax.set_ylabel(r'Number of visits')
 
            figb, axb = plt.subplots()
            figb.suptitle(fieldname)

            axb.fill_between(sel['z'],sel['Nvisits_min']-sel['Nvisits_median'],sel['Nvisits_max']-sel['Nvisits_median'],color='grey')
            #axb.plot(sel['z'],sel['Nvisits_median'],color='k')

            axb.grid()
            axb.set_xlim([0.3,0.85])
            axb.set_xlabel(r'z')
            axb.set_ylabel(r'$\Delta$Number of visits')
