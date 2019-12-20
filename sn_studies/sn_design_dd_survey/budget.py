import yaml
from scipy import interpolate
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
from . import filtercolors
from . import plt

class DD_Budget:
    def __init__(self,configName,df_visits_ref,df_visits,runtype='Nvisits_single'):
        
        config = yaml.load(open(configName), Loader=yaml.FullLoader)

        self.df_visits = df_visits
        self.interp_visits(df_visits)

        self.nvisits_ref = self.interp_ref(df_visits_ref)

        self.df_visits_ref = df_visits_ref
        
        self.budget = self.budget_calc(config,runtype)
        self.runtype = runtype
        if self.runtype == 'Nvisits_single':
            self.summary_Nvisits_single()
        

    def interp_visits(self, df_tot):

        fieldnames = ['COSMOS','CDFS','XMM-LSS','ELAIS','ADFS1','ADFS2']
        seasons = range(1,11)
        self.nvisits = {}
        self.nvisits_band = {}
        self.z = {}
        for fieldname in fieldnames:
            self.nvisits[fieldname] = {}
            self.nvisits_band[fieldname] = {}
            self.z[fieldname] = {}
            for season in seasons:
                idx = df_tot['fieldname'] == fieldname
                idx &= df_tot['season'] == season
                sel = df_tot[idx]
                     
                self.nvisits[fieldname][season]= interpolate.interp1d(sel['z'],sel['Nvisits'],bounds_error=False,fill_value=0.0)
                self.z[fieldname][season]= interpolate.interp1d(sel['Nvisits'],sel['z'],bounds_error=False,fill_value=0.0) 
                self.nvisits_band[fieldname][season] = {}
                
                for b in 'rizy':
                    self.nvisits_band[fieldname][season][b]= interpolate.interp1d(sel['z'],sel['Nvisits_{}'.format(b)],bounds_error=False,fill_value=0.0)
                

    def interp_ref(self, df_ref, what='Nvisits'):

        idx = df_ref['fieldname'] == 'all'
        idx &= df_ref['season'] == 0
        sel = df_ref[idx]

        print('blllll',df_ref.columns)
        nvisits_ref= interpolate.interp1d(sel['z'],sel[what],bounds_error=False,fill_value=0.0)

        return nvisits_ref

    def budget_calc(self,config,runtype):

        zr = np.arange(0.3,0.9,0.05)
        df_tot = pd.DataFrame()
        
        if runtype =='Nvisits_single':
            Nvisits_ref = self.nvisits_ref(zr)

        self.cols = []
        self.cols_night = []
        self.zcols=[]
        for fieldname in config['Fields']:
            print(config[fieldname])
            conf = config[fieldname]
            for season in conf['seasons']:
                print(self.nvisits[fieldname][season](zr))
                if runtype =='Nvisits_single':
                    nvisits_night = Nvisits_ref
                else:
                    nvisits_night = self.nvisits[fieldname][season](zr)
                nvisits_season = nvisits_night*30*config[fieldname]['season_length']/config[fieldname]['cadence']

                zvals = self.z[fieldname][season](nvisits_night)
                zname = '{}_{}_{}'.format('z',fieldname,season)
                vname = '{}_{}_{}'.format('Nvisits',fieldname,season)
                vname_night = '{}_{}_{}'.format('Nvisits_night',fieldname,season)
                self.cols.append(vname)
                self.cols_night.append(vname_night)
                self.zcols.append(zname)
                dfa = pd.DataFrame(nvisits_season, columns=[vname])
                dfa.loc[:,zname] = zvals
                dfa.loc[:,vname_night] = nvisits_night
                dfa.loc[:,'zref'] = zr
                if df_tot.empty:
                    df_tot = dfa.copy()
                else:
                    df_tot = df_tot.merge(dfa, left_on=['zref'],right_on=['zref'])

        print(self.cols)
        df_tot['Nvisits'] = df_tot[self.cols].sum(axis=1)
        df_tot['Nvisits_night'] = df_tot[self.cols_night].median(axis=1)
        df_tot['DD_budget'] = df_tot['Nvisits']/config['Nvisits']   
        print(df_tot)
        return df_tot
        
    def summary_Nvisits_single(self):

        z = np.arange(0.3,0.90,0.01)
        idx = (self.budget['DD_budget']<0.15)
        idx &= (self.budget['DD_budget']>=0.)
        toplot = self.budget[idx]

        print('hello',toplot)
        medz = toplot[self.zcols].median(axis=1) #median redshift limit
        medbud = toplot['DD_budget']
        medvisits = toplot['Nvisits_night']
    
        # get zlim_min and zlim_mac cols
        r = []
        
        #buds = np.arange(0.000001,0.10,0.0001)
        
        df_bud_z = pd.DataFrame()

        
        for io,col in enumerate(self.zcols):
            idx = toplot[col]>=0.3
            idx &= toplot[col]<=0.85
            #plt.plot(toplot[idx][col],toplot[idx]['DD_budget'],color='k')
            """
            interp_ddbudget = interpolate.interp1d(toplot[idx]['DD_budget'],
                                                        toplot[idx][col],
                                                        bounds_error=False,fill_value=0)
            interp_night = interpolate.interp1d(toplot[idx]['DD_budget'],
                                                        toplot[idx]['Nvisits_night'],
                                                        bounds_error=False,fill_value=0)
            """
            interp_ddbudget = interpolate.interp1d(toplot[idx][col],
                                                   toplot[idx]['DD_budget'],
                                                   bounds_error=False,fill_value=0)

            interp_night = interpolate.interp1d(toplot[idx]['DD_budget'],
                                                        toplot[idx]['Nvisits_night'],
                                                        bounds_error=False,fill_value=0)


            df = pd.DataFrame()
            df.loc[:,'budget'] = interp_ddbudget(z)
            df.loc[:,'z'] = z
            df.loc[:,'name'] = col
            df.loc[:,'Nvisits_night'] = interp_night(df['budget'])

            df_bud_z= pd.concat([df_bud_z,df],sort=False)
            
            #r.append((col,interp_ddbudget(0.04),interp_ddbudget(0.08)))

        df_buz_z = df_bud_z.sort_values(by=['budget'])
        #print(df_bud_z,len(buds),len(df_buz_z))

        """
        colz = np.rec.fromrecords(r, names=['col','zlim','zlimax'])
                 
        colz.sort(order='zlim')

        print('oooooooo',colz)
        colmin = colz[0]['col']
        colmax = colz[-1]['col']

        print('iii',colmin,colmax)
        self.zmax = colz[-1]['zlimax']
        """
        df_min = df_bud_z[df_bud_z['budget'].gt(0.)].groupby(['z']).min().reset_index()
        df_max = df_bud_z[df_bud_z['budget'].gt(0.)].groupby(['z']).max().reset_index()
        df_med =  df_bud_z[df_bud_z['budget'].gt(0.)].groupby(['z']).median().reset_index()

        """
        colors = dict(zip(['COSMOS','CDFS','XMM-LSS','ELAIS','ADFS1','ADFS2'],['k','b','r','g','m','orange']))
        for io,col in enumerate(self.zcols):
            idx = toplot[col]>=0.3
            idx &= toplot[col]<=0.85
            fieldname = col.split('_')[1]          
            plt.plot(toplot[idx][col],toplot[idx]['DD_budget'],color=colors[fieldname])
        
        plt.plot(df_min['z'],df_min['budget'])
        plt.plot(df_max['z'],df_max['budget'])
        plt.show()
        """
        self.interpmin = interpolate.interp1d(df_min['z'],df_min['budget'],bounds_error=False,fill_value=0.10)
        self.interpmax = interpolate.interp1d(df_max['z'],df_max['budget'],bounds_error=False,fill_value=0.0)

        self.interpmin_ddbudget = interpolate.interp1d(df_max['budget'],df_max['z'],bounds_error=False,fill_value=0.10)
        self.interpmax_ddbudget = interpolate.interp1d(df_min['budget'],df_min['z'],bounds_error=False,fill_value=0.0)

        self.interp_ddbudget = interpolate.interp1d(df_med['budget'],df_med['z'],bounds_error=False,fill_value=0.0)
        self.interp_z_ddbudget = interpolate.interp1d(df_med['z'],df_med['budget'],bounds_error=False,fill_value=0.0)
        self.interp_z =  interpolate.interp1d(df_med['z'],df_med['Nvisits_night'],bounds_error=False,fill_value=0.0)

        """
        self.interpmin = interpolate.interp1d(toplot[colmin],medbud,bounds_error=False,fill_value=0.10)
        self.interpmax = interpolate.interp1d(toplot[colmax],medbud,bounds_error=False,fill_value=0.0)

        self.interpmin_ddbudget = interpolate.interp1d(medbud,toplot[colmin],bounds_error=False,fill_value=0.10)
        self.interpmax_ddbudget = interpolate.interp1d(medbud,toplot[colmax],bounds_error=False,fill_value=0.0)
        
        self.interp_ddbudget = interpolate.interp1d(medbud,medz,bounds_error=False,fill_value=0.0)
        self.interp_z =  interpolate.interp1d(medz,medvisits,bounds_error=False,fill_value=0.0)
        """
        self.medbud = df_med['budget'].values 
        self.medz = df_med['z'].values 
        self.medvisits = df_med['Nvisits_night'].values 
        self.zmax = df_max['z'].max()


    def zlim_Nvisits_single(self, dd_value):

        zlim_median = self.interp_ddbudget(dd_value)
        zlim_min = self.interpmin_ddbudget(dd_value)
        zlim_max = self.interpmax_ddbudget(dd_value)
        nvisits_choice = self.interp_z(zlim_median)
        Nvisits_band = {}
        for b in 'rizy':
            myinterp = self.interp_ref(self.df_visits_ref,'Nvisits_{}'.format(b))
            Nvisits_band[b] = myinterp(zlim_median)

        return  zlim_median,zlim_min,zlim_max,nvisits_choice,Nvisits_band

    def plot_budget(self, dd_value):

        if self.runtype == 'Nvisits_adjusted':
            self.plot_budget_Nvisits_adjusted(dd_value)

        if self.runtype == 'Nvisits_single':
            self.plot_budget_Nvisits_single(dd_value)

    def plot_budget_Nvisits_adjusted(self, dd_value):

        #dd_value = 0.06
        fig, ax = plt.subplots()    

        idx = self.budget['zref']>=0.3
        idx &= self.budget['zref']<=0.85
        budget = self.budget[idx]

        ax.plot(budget['zref'],budget['DD_budget'],color='k')
        interp_budget_z = interpolate.interp1d(budget['DD_budget'],budget['zref'],bounds_error=False,fill_value=0.0)
        zlim = interp_budget_z(dd_value)
        ax.arrow(zlim,dd_value,0.,-dd_value,
                 length_includes_head=True,color='b',
                 head_length=0.005, head_width=0.01)
        ax.text(0.5,1.1*dd_value,'$z_{lim}$='+str(np.round(zlim,2)))
        ax.grid()
        ax.set_ylim([0.,0.11])
        ax.set_xlim([0.3,0.85])
        ax.set_xlabel(r'z')
        ax.set_ylabel(r'DD budget')
        ax.plot(ax.get_xlim(),[dd_value]*2,color='r')

        figb, axb = plt.subplots()

        for col in self.cols_night:
            #axb.plot(budget['zref'],budget[col],color='k')
            axb.plot(budget[col],budget['zref'],color='k')
            fieldname = col.split('_')[2]
            season = int(col.split('_')[3])
            print(fieldname,season)
            print('Nvisits_tot',np.round(self.nvisits[fieldname][season](zlim),1))
            for b in 'rizy':
                Nvisits = self.nvisits_band[fieldname][season][b](zlim)
                print(b,np.round(Nvisits,1),math.ceil(Nvisits))



        axb.grid()
        axb.set_xlabel(r'$z_{lim}$')
        axb.set_ylabel(r'Nvisits/night')

    def plot_budget_Nvisits_single(self, dd_value):

        zminval = 0.3
        z = np.arange(zminval,0.9,0.05)
        if dd_value is not None:
            zlim_median,zlim_min,zlim_max,nvisits_choice,Nvisits_band = self.zlim_Nvisits_single(dd_value)

        # Now plot


        print(plt.rcParams)
        fig, axs = plt.subplots(1,2,gridspec_kw={'hspace': 0, 'wspace': 0},figsize=(20,12))
        (ax1, ax2)= axs

        
        ax1.set_ylim(ymax= 0.10)
        print('ooo pal',self.zmax)
        ax1.set_xlim([zminval+0.01,self.zmax])
        ax1.set_ylim([self.interp_z_ddbudget(zminval),np.min([0.10,self.interp_z_ddbudget(self.zmax)])])
        zb = np.arange(zminval,self.zmax,0.01)
        ax1.fill_between(zb,self.interpmin(zb),self.interpmax(zb),color='yellow',alpha=0.5)
        ax1.plot(self.medz,self.medbud,color='k')
        print(self.medz,self.medbud)
        ax1.set_ylabel(r'DD budget')
        ax1.set_xlabel(r'z$_{lim}$')
        ax1.grid()
       
        
        axa = ax1.twinx()
        axa.plot(self.medz,self.medvisits,color='k')
        print('ooo',self.medz,self.medvisits,ax1.get_ylim())

        zlims = self.interp_ddbudget(ax1.get_ylim())
        Nvisitslim = self.interp_z(zlims)
        print('hhh',zlims,Nvisitslim)
        axa.set_ylim(Nvisitslim)
        #plt.show()

        if dd_value is not None:
            ax1.plot(ax1.get_xlim(),[dd_value]*2,color='r',ls='--')
            for ip, val in enumerate([('min',zlim_min),('max',zlim_max),('median',zlim_median)]):
                 ax1.arrow(val[1],dd_value,0.,-dd_value,
                      length_includes_head=True,color='b',
                      head_length=0.005, head_width=0.01)
                 ax1.text(0.35,0.05-0.005*ip,'$z_{lim}^{'+val[0]+'}$='+str(np.round(val[1],2)))

        zticks = self.interp_ddbudget(ax1.get_yticks())
        Nvisits_ticks = self.interp_z(zticks)
        axa.set_yticks(Nvisits_ticks)
       
        # second plot
        ## adjust axes
        ax2.set_ylim(axa.get_ylim())
        ax2.set_xlim(ax1.get_xlim())
        ax2.set_yticks(axa.get_yticks())
        # make the twin to have Nvisits on the right
        ax3 = ax2.twinx()
        ax3.set_ylim(axa.get_ylim())
        ax3.set_xlim(ax1.get_xlim())
        ax3.plot(z,self.nvisits_ref(z),color='k',label='all')
        #Nvisits_band = {}
        for b in 'rizy':
            myinterp = self.interp_ref(self.df_visits_ref,'Nvisits_{}'.format(b))
            ax3.plot(z,myinterp(z),color=filtercolors[b],label='{}'.format(b))
            
            #if dd_value is not None:
            #    Nvisits_band[b] = myinterp(zlim_median)
           
        ax2.grid()
        ax3.legend()
        ax2.yaxis.set_ticklabels([])
        axa.yaxis.set_ticklabels([])
        ax3.set_ylabel(r'Nvisits/night')
        ax2.set_xlabel(r'z')
        #print('oooo',int(ax2.get_yticks().tolist()))
        ax3.set_yticks(ax2.get_yticks())
        #ax3.yaxis.set_major_locator(MinNLocator(integer=True))
        ax3.set_yticklabels(np.ceil(ax3.get_yticks()).astype(int))
        """
        locs, labels = ax3.get_yticks()
        yint=[]
        for each in locs:
            yint.append(int(each))
        ax3.set_yticks(yint)
        """
        if dd_value is not None:
            ax2.plot(ax2.get_xlim(),[nvisits_choice]*2,color='r',ls='--')
            nvisits_choice_calc = 0
            ax2.arrow(zlim_median,nvisits_choice,0.0,ax3.get_ylim()[0]-nvisits_choice,
                      length_includes_head=True,color='b',
                      head_length=1., head_width=0.01)
            #ax2.text(0.35,1.1*nvisits_choice,'$N_{visits}$ - sum ='+str(int(nvisits_choice)))
            for io,band in enumerate('rizy'):
                #ax2.plot(ax2.get_xlim(),[Nvisits_band[band]]*2,color='r',ls='--')
                if Nvisits_band[band] >0:
                    nvisits_band = int(Nvisits_band[band])
                    ax2.arrow(zlim_median,Nvisits_band[band],
                              ax2.get_xlim()[1]-zlim_median,0.,
                              length_includes_head=True,color='b',
                              head_width=1., head_length=0.01)
                    ax2.text(0.35,0.9*nvisits_choice-0.1*io*nvisits_choice,'$N_{visits}$-'+band+'='+str(nvisits_band))
                    nvisits_choice_calc += nvisits_band
            ax2.text(0.35,1.1*nvisits_choice,'$N_{visits}$ - sum ='+str(int(nvisits_choice_calc))) 

        #fig.tight_layout()
