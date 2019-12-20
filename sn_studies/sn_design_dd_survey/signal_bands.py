import numpy as np
from sn_tools.sn_telescope import Telescope
import numpy.lib.recfunctions as rf
from . import plt,filtercolors
from scipy import interpolate


class RestFrameBands:

    def __init__(self,blue_cutoff=350.,red_cutoff=800.):
        """
        class to estimate redshift cutoffs (per band)
        due to blue and red cutoffs applied to 
        restframe wavelenghts of supernovae

        Parameters
        ----------
        blue_cutoff: float
         wavelength limit in the blue
        red_cutoff: float
         wavelength limit in the red 

        """
        self.blue_cutoff = blue_cutoff
        self.red_cutoff = red_cutoff
        self.bands = 'ugrizy'

        # estimate the restframe wavelengths (using the telescope wl)
        self.wave_frame = self.waves()

        # estimate the correponding cut on z (per band)
        self.zband = self.zcutoff()

    def waves(self):
        """
        Method to estimate the restframe wl as a function of z

        Parameters
        ---------

        Returns
        -------
        wave_frame: numpy array
         with the fields: lambda_rf, band, z


        """

        
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
        """
        Method to estimate the z cutoff per band

        Parameters
        ---------

        Returns
        -------
        wave_frame: numpy array
         with the fields: band,z_blue,z_red

        """
        # now get the redshifts limits corresponding to the blue and red cutoffs

        r = []
        for band in self.bands:
            idx = self.wave_frame['band'] == band
            selw = self.wave_frame[idx]
            interp = interpolate.interp1d(selw['lambda_rf'],selw['z'],bounds_error=False,fill_value=(1.5,0.0))
            #print(band,interp(self.blue_cutoff),interp(self.red_cutoff))
            r.append((band,interp(self.blue_cutoff),interp(self.red_cutoff)))

        return np.rec.fromrecords(r, names=['band','z_blue','z_red'])

    def plot(self):
        """
        Method to plot the results of the class:
        lambda_rf vs z with blu and red cutoffs superimposed

        Parameters
        ----------

        Returns
        -------
        plot lambda_rf vs z with blu and red cutoffs superimposed
        """
        

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

class SignalBand:

    def __init__(self, lcdf):
        """
        class to estimate the signal per band of a LC as a function of z

        Parameters
        ---------
        lcdf: pandas df
        LC to process

        """

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
        """
        Method to plot the flux fraction per band as a function of z

        Parameters
        ----------
        x1: float
         SN stretch
        color: float
         SN color
        
        Returns
        ------
        plot the signal fraction per band vs z

        """

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
