import os
import pandas as pd
import numpy as np
import time
from scipy.spatial import distance

from .utils import flux5_to_m5
from .wrapper import Nvisits_cadence
from . import plt
from . import filtercolors
from sn_tools.sn_calcFast import covColor

class SNR:
    def __init__(self,SNRDir,data,
                 SNR_par):

        """
        Wrapper class to estimate SNR
        
        Parameters
        ---------
        SNRDir: str
        directory where SNR files are located
        data: pandas df
        data to process (LC)
        SNR_par: dict
        SNR parameters
        
        """
        # get the data parameters
        # useful for the SNR filename
        self.x1 = data.x1
        self.color = data.color
        self.bands = data.bands
        self.blue_cutoff = data.blue_cutoff
        self.red_cutoff = data.red_cutoff

        # define the SNR file name
        SNRName = self.name(SNRDir,
                            SNR_par)
        
        # process the data if necessary
        if not os.path.isfile(SNRName):

            myclass = SNR_z(SNRDir,data,
                            SNR_par=SNR_par)

            dfsigmaC = myclass.sigmaC_SNR()

            #myclass.plotSigmaC_SNR(dfsigmaC) # plot the results
        
            #plt.show()
            # now choose SNR corresponding to sigmaC~0.04 vs z

            SNR_dict = myclass.SNR(dfsigmaC)

            for key, vals in SNR_dict.items():
                SNR_par_dict = dict(zip(['max','step','choice'],[50.,2.,key]))
                thename = self.name(SNRDir,SNR_par_dict)
                # save the file
                np.save(thename,np.copy(vals.to_records(index=False)))
            
        self.SNR = pd.DataFrame(np.load(SNRName))

    
    def plot(self):
        """
        plt the results
        
        Parameters
        ---------
        
        Returns
        --------
        plt SNR-band vs z
        
        """
        
        plotSNR_z(self.SNR,x1=self.x1,color=self.color,bands=self.bands)

    def name(self,SNRDir,
             SNR_par):
        """
        method to define the name of the SNR file
        
        Parameters
        ---------
        SNRDir: str,
          location directory of the file
        SNR_par: dict
         SNR parameters

        Returns
        --------
        full path to SNR file (str)
        
        """

        name = '{}/SNR_{}_{}_{}_{}_{}_{}_{}.npy'.format(SNRDir,
                                                        self.x1,
                                                        self.color,
                                                        SNR_par['step'],
                                                        self.blue_cutoff,
                                                        self.red_cutoff,
                                                        self.bands,
                                                        SNR_par['choice'])
    

        return name

class SNR_z:

    def __init__(self,dirFile,data,
                 SNR_par={}):
        
        """
        class to estimate SNR per band vs redshift
        
        Parameters
        ---------
        dirFile: str
         directory where SNR files are located
        data: pandas df
         data to process (LC)
        SNR_par: dict
         SNR parameters
        
        """
        # get data parameters
        self.x1 = data.x1
        self.color = data.color
        self.bands = data.bands
       
        # get SNR parameters
        self.SNR_par = SNR_par
       
       
        # load LC
        self.lcdf = data.lc
        
        # load m5
        self.medm5 = data.m5_Band

        # load signal fraction per band
        self.fracSignalBand = data.fracSignalBand.fracSignalBand

        # this list is requested to estimate Fisher matrix elements

        self.listcol = ['F_x0x0', 'F_x0x1',
                        'F_x0daymax', 'F_x0color', 
                        'F_x1x1', 'F_x1daymax', 
                        'F_x1color','F_daymaxdaymax', 
                        'F_daymaxcolor', 'F_colorcolor']

        # map flux5-> m5

        self.f5_to_m5 = flux5_to_m5(self.bands)

    def sumgrp(self,grp):
        """
        Method to assess the sum of the grp column

        Parameters
        ----------
        grp: group (pandas df)

        Returns
        --------
        panda df with the following columns:
        - sumflux = sqrt(sum(flux**2))
        - flux5 = mean(flux_5)
        - SNR = 5*sumflux/flux_5
        - self.listcol = sum of Fisher elements


        """
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
        """
        Method to assess sigma_C as a function of SNRbands

        Parameters
        ----------

        Returns
        -------
        pandas df with the following columns:
        x1,color,z,band
        (sum_flux,SNR,self.listcol,flux_5_e_sec,m5_calc,flux_5)_band
        sigmaC,SNRcalc_tot


        """

        # group the LC df by ('x1','color','z','band') and estimate sums

        df = self.lcdf.groupby(['x1','color','z','band']).apply(lambda x: self.sumgrp(x)).reset_index()
       
        # make groups and estimate sigma_Color for combinations of SNR per band

        dfsigmaC = df.groupby(['x1','color','z']).apply(lambda x: self.sigmaC(x)).reset_index()
 
        """
        cols = ['z']
        for val in ['SNRcalc','flux_5_e_sec']:
            for b in self.bands:
                cols.append('{}_{}'.format(val,b))
        """
        return dfsigmaC

    def sigmaC(self,grp):

        """
        For a given group (grp), this method estimates sigmaC 
        for a set of SNR-band combinations

        Parameters
        ----------
        grp: pandas df group

        Returns
        -------
        pandas df with the following columns:
        x1,color,z,band
        (sum_flux,SNR,self.listcol,flux_5_e_sec,m5_calc,flux_5)_band
        sigmaC,SNRcalc_tot


        """

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
        

        # generate SNR values of interest (per band)
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
        """
        Method add SNR-band combinations
        the five-sigma flux and corresponding m5 are estimated (SNR dependent).
        Fisher elements are also re-evaluated (f5 dependence through LC flux errors)

        Parameters
        ----------
        df: pandas df
         input data 
        SNR: float
         SNR values
        b: str
         band

        Returns
        -------
        pandas df generated from the set of combinations
        
        """


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
        """
        Plot sigmaC vs SNRtot vs z 

        Parameters
        ----------
        dfsigmaC: pandas df
         data to plot

        Returns
        -------
        plot of sigmaC vs SNRtot (all SNR-band combis)
        one plot per z bin

        """
        
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
        """
        Method to select SNR-band combis according to three reqs:

        - fracfluxband: SNR distrub similar to the flux fraction (per band)
        - nvisits_min: SNR combis that minimizes the total number of visits
        - nvisits_y_min: SNR combis that minimizes the total number of visits in the y band

        Parameters
        ----------
        dfsigmaC: pandas df
         data to process

        Returns
        ------
        pandas df with the SNR-band combi corresponding to the 3 above mentioned conditions

        """
        # select only combi with sigma_C ~ 0.04
        idx = np.abs(dfsigmaC['sigmaC']-0.04)<0.0004
        #idx = np.abs(dfsigmaC['sigmaC']-0.04)<0.1

        sel = dfsigmaC[idx].copy()

        # for these combi : estimate the SNR frac per band
        
        
        for b in self.bands:
            sel.loc[:,'fracSNR_{}'.format(b)] = sel['SNRcalc_{}'.format(b)]/sel['SNRcalc_tot']
            if self.medm5 is not None:
                sel.loc[:,'m5single_{}'.format(b)] = self.medm5[self.medm5['filter']==b]['fiveSigmaDepth'].values

        """
        if self.SNR_par['choice'] == 'fracflux':
            res = sel.groupby(['x1','color','z']).apply(lambda x: self.SNR_redshift(x)).reset_index()
        else:
            res = sel.groupby(['x1','color','z']).apply(lambda x: self.SNR_visits(x)).reset_index()
        """

        res = {}
        res['fracflux'] = sel.groupby(['x1','color','z']).apply(lambda x: self.SNR_redshift(x)).reset_index()
        for vv in ['Nvisits','Nvisits_y']:
            res[vv] = sel.groupby(['x1','color','z']).apply(lambda x: self.SNR_visits(x,vv)).reset_index()
        
        return res

    def SNR_visits(self,grp,minPar='Nvisits'):
        """
        Method to estimate the SNR-band combination that minimizes minPar

        Parameters
        ----------
        grp: pandas df group
         data to process
        minPar: str, opt
         parameter used to minimize (default: Nvisits)

        Returns
        -------
        pandas df with the SNR combination corresponding to the above-mentioned criteria

        """

        # Estimate the requested number of visits (per band and in total)
        cols = []
        for b in self.bands:
            cols.append('Nvisits_{}'.format(b))

        for b in self.bands:
            grp.loc[:,'Nvisits_{}'.format(b)]=10**(0.8*(grp['m5calc_{}'.format(b)]-grp['m5single_{}'.format(b)]))
            grp['Nvisits_{}'.format(b)]=grp['Nvisits_{}'.format(b)].fillna(0.0)

        grp.loc[:,'Nvisits'] = grp[cols].sum(axis=1)

        idx = int(grp[[minPar]].idxmin())
        #idxa = int(grp[['Nvisits']].idxmin())
        #idxb = int(grp[['Nvisits_y']].idxmin())


        cols =  []
        for colname in ['SNRcalc','m5calc','fracSNR','flux_5_e_sec']:
            for band in 'grizy':
                cols.append('{}_{}'.format(colname,band))
        
        #colindex = [grp.columns.get_loc(c) for c in cols if c in grp]
            
        return grp.loc[idx,cols]
        #return grp.loc[idxa,cols], grp.loc[idxb,cols]

 
    def SNR_redshift(self,grp):
        """
        Method to estimate the SNR-band combination
        with a distribution closest to flux fraction per band

        Parameters
        ----------
        grp: pandas df group


        Returns
        -------
        pandas df with the SNR combination corresponding to the above-mentioned criteria

        """
        
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

class SNR_plot:

    def __init__(self,SNRDir,x1,color,
                 SNR_step,
                 blue_cutoff,
                 red_cutoff,
                 cadence):
        """
        class to plot SNR results

        Parameters
        ----------
        SNRDir: str
         location directory of the SNR files
        x1: float
         SN stretch parameter
        color: float
         SN color parameter
        SNR_step: float
         SNR step used when scanning the SNR-band parameter space
        blue_cutoff: float
         wavelength cutoff (blue) applied to the data used for SNR-b estimation
        red_cutoff: float
         wavelength cutoff (red) applied to the data used for SNR-b estimation
        cadence: float
         cadence choice for the display of the results
        

        """


        # load parameters
        self.SNRDir = SNRDir
        self.x1 = x1
        self.color = color
        self.SNR_step = SNR_step
        self.blue_cutoff = blue_cutoff
        self.red_cutoff = red_cutoff
        self.cadence=cadence

        # load SNR files
        self.dictplot=self.load()

    def nameFile(self,SNRDir,
                 x1,color,
                 SNR_step,
                 blue_cutoff,
                 red_cutoff,
                 bands,
                 SNR_choice):
        
        """
        Method defining the name of SNR file
        
        Parameters
        ----------  
        SNRDir: str
         location directory of the SNR files
        x1: float
         SN stretch parameter
        color: float
         SN color parameter
        SNR_step: float
         SNR step used when scanning the SNR-band parameter space
        blue_cutoff: float
         wavelength cutoff (blue) applied to the data used for SNR-b estimation
        red_cutoff: float
         wavelength cutoff (red) applied to the data used for SNR-b estimation
        bands: str
         filters considered for SNR-b estimation
        SNR_choice: str
         type of choice used to get the SNR-b combination

        Returns
        ------
        str: name of the SNR file

        """

        name = '{}/SNR_{}_{}_{}_{}_{}_{}_{}.npy'.format(SNRDir,
                                                        x1,color,
                                                        SNR_step,
                                                        blue_cutoff,
                                                        red_cutoff,
                                                        bands,
                                                        SNR_choice)

        return name

    def load(self):
        """
        Method to load SNR files

        Parameters
        ----------

        Returns
        -------
        dictplot: dict
         keys: SNR_choice_bands
         vals: number of visits (per band and in total) vs z


        """


        dictplot={}
        for SNR_choice in [('fracflux','rizy'),
                           ('Nvisits','rizy'),
                           ('Nvisits_y','rizy')]:
            #('Nvisits','riz')]:
            SNRNameb = self.nameFile(self.SNRDir,
                                self.x1,self.color,
                                self.SNR_step,
                                self.blue_cutoff,
                                self.red_cutoff,
                                SNR_choice[1],
                                SNR_choice[0])

            print('loading',SNRNameb)
            SNRb = pd.DataFrame(np.load(SNRNameb))

            #myvisits = Nvisits_m5(SNRb,medValues)
            m5_type='median_m5_filter'
            
            myvisits = Nvisits_cadence(SNRb,self.cadence,m5_type,SNR_choice[0],SNR_choice[1]).nvisits_cadence
            print(myvisits)
            dictplot['_'.join([SNR_choice[0],SNR_choice[1]])] = myvisits
        return dictplot
             
    def plotSummary(self):
        """
        Summary plot of the SNR results

        Parameters
        ----------

        Returns
        ------
        plot of the number of visits (total) vs z for the various configuration (SNR_choices)
        
        """


        ls = ['-','--','-.',':']
        colors = ['k','b','r','g']

        fig, ax = plt.subplots()
        fig.suptitle('cadence = {} days'.format(self.cadence))

        keys = list(self.dictplot.keys())
        for key, data in self.dictplot.items():
            sel = data
 
            ax.plot(sel['z'].values,sel['Nvisits'].values,
                     color=colors[keys.index(key)],ls=ls[keys.index(key)],
                     label='{}'.format(key))
        ax.legend()
        ax.grid()
        ax.set_xlabel(r'z')
        ax.set_ylabel(r'{}'.format('Nvisits'))
        ax.set_xlim([0.3,0.85])
        rplot = []
        rplot.append((40,'6% - 10 seasons * 6 fields',-17))
        rplot.append((48,'6% - 10 seasons * 5 fields',+15))
        rplot.append((199,'6% - 2 seasons * 6 fields',-17))
        rplot.append((239,'6% - 2 seasons * 5 fields',+15))     
        
        for val in rplot:
            ax.plot(ax.get_xlim(),[val[0]]*2,color='m')
            ax.text(0.33,val[0]+val[2],val[1])
    
    def plotSummary_band(self,bands='rizy',legy='Nvisits'):
        """
        Summary plot of the SNR results

        Parameters
        ----------

        Returns
        ------
        plot of the number of visits (per band) vs z for the various configuration (SNR_choices)
        
        """

        ls = ['-','--','-.',':']
        colors = ['k','b','r','g']

        fig, ax = plt.subplots()
        fig.suptitle('cadence = {} days'.format(self.cadence))
        keys = list(self.dictplot.keys())
        
        io = -1
        for key, data in self.dictplot.items():
            io +=1
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
            yval = 150.-10*io
            ax.plot([0.3,0.35],[yval]*2,ls[keys.index(key)],color='k')
            ax.text(0.37,yval,key)

        ax.legend()
        ax.grid()
        ax.set_xlabel(r'z')
        ax.set_ylabel(r'{}'.format(legy))     

    def plotIndiv(self, config,bands='rizy', legy='N$_{visits}$/field/observing night'):
        """
        plot individual (per SNR_choice) results

        Parameters
        ----------
        config: str
         config (=SNR_choice_band) chosen
        bands: str
         filters to plot
        legy: str, opt
         ylabel (default: 'N$_{visits}$/field/observing night')
         could also be: 'Filter allocation'


        Returns
        ------
        plot of the number of visits the number of visits or the filter allocation.
        
        """
        fig, ax = plt.subplots()
        fig.suptitle('cadence = {} days - {}'.format(self.cadence,config))

        keys = list(self.dictplot.keys())
        print(keys)
        data = self.dictplot[config]
        
        cols = []
        for b in config.split('_')[-1]:
            cols.append('Nvisits_{}'.format(b))
         
        for b in bands:
            if legy == 'Filter allocation':
                ax.plot(data['z'],data['Nvisits_{}'.format(b)]/data['Nvisits'],color=filtercolors[b],label='{}'.format(b))
            else:
               ax.plot(data['z'],data['Nvisits_{}'.format(b)],color=filtercolors[b],label='{}'.format(b)) 

        ax.set_xlabel(r'z')
        ax.set_ylabel(r'{}'.format(legy))
        ax.legend()
        ax.grid()
