from sn_saturation.psf_pixels import PSF_pixels,PlotPixel,PlotMaxFrac
from sn_saturation.mag_saturation import MagToFlux,MagSaturation,PlotMagSat
from sn_saturation.observations import Observations,prepareYaml
from sn_saturation.sn_sat_effects import SaturationTime,plotTimeSaturation
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
import os

class PixelPSFSeeing:
    """
    class to estimate the flux fraction max as a function of the seeing depending on PSF

    Parameters
    --------------
    psf_type: str
      psf profile

    """

    def __init__(self,psf_type):

        self.seeings = np.arange(0.3, 2.6, 0.01)
        self.psf_type = psf_type
        prefix = 'PSF_pixel'
        
        resfi = self.loop_seeing()

        # save the results in a file
        np.save('{}_{}.npy'.format(prefix,psf_type),np.copy(resfi))

        summary = self.summary(resfi)
        np.save('{}_{}_summary.npy'.format(prefix,psf_type),summary)
        
    def summary(self, resfi):
        """
        Method to extract median, min and max values vs seeing

        Parameters
        --------------
        resfi: numpy array
          data to process

        Returns
        -----------
        record array with the following cols:
        seeing,pixel_frac_max,pixel_frac_min,pixel_frac_med

        """

        df = pd.DataFrame(np.copy(resfi))
        min_seeing = df['seeing'].min()
        max_seeing = df['seeing'].max()
    
        df = df.round({'xpixel': 2, 'ypixel': 2,'xc':2,'yc':2,'seeing':2})

        # take the pixel in (0,0)
        idx = np.abs(df['xpixel'])<1.e-5
        idx &= np.abs(df['ypixel'])<1.e-5
        df = df[idx]

        grp = df.groupby(['seeing']).apply(lambda x: pd.DataFrame({'pixel_frac_max':[x['pixel_frac'].max()],
                                                                   'pixel_frac_min':[x['pixel_frac'].min()],
                                                                   'pixel_frac_med':[x['pixel_frac'].median()]})).reset_index()

        finalresu = grp[['seeing','pixel_frac_max','pixel_frac_min','pixel_frac_med']].to_records(index=False)

        return finalresu
        

    def loop_seeing(self):
        """
        Method to loop on seeing and estimate the frac max pixel

        Returns
        ----------
        numpy array with the following cols:
        

        """
        resfi = None
        for seeing in self.seeings:
            toproc = PSF_pixels(seeing,self.psf_type)
            res= toproc.get_flux_map()
    
        if resfi is None:
            resfi = res
        else:
            resfi = np.concatenate((resfi,res))

        return resfi

def Simulations(nexp_expt=[(1,15),(1,30)],x1_color=[(0.0,0.0)],seasons = [2],cadence=3):
    """
    Function to perform a set of simulations

    Parameters
    ---------------
    nexp_expt: list(tuples)
      list of (nexp, exptime) combis
    x1_color: list(tuple)
      list of (x1,color) combi
    season: list(int)
      list of the seasons to gen
    cadence: int
      cadence of observations

    """
    
    nproc = 1
    input_yaml = 'input/saturation/param_simulation_gen.yaml'

    for (x1,color) in x1_color:
        for season in seasons:
            for (nexp,expt) in nexp_expt:
                output_yaml = 'param_simulation_{}_{}_{}_{}_{}.yaml'.format(x1,color,nexp,expt,season,cadence)
                prepareYaml(input_yaml,nexp,expt,x1,color,season,nproc,output_yaml,cadence)
                cmd = 'python run_scripts/simulation/run_simulation_from_yaml.py --config_yaml={} --npixels 1'.format(output_yaml)
                cmd += ' --radius 0.01'
                cmd += ' --RAmin 0.0'
                cmd += ' --RAmax 0.1'
                os.system(cmd)              


def estimateSaturationTime(dirFile,x1_color,seasons,nexp_expt,cadence_obs,nproc):
    """
    Function to estimate saturation time vs z

    Parameters
    --------------
    dirFile: str
      location dir of the simulation
    x1_color: list(tuple)
      list of (x1,color) combi   
    season: list(int)
      list of the seasons to gen
    nexp_expt: list(tuples)
      list of (nexp, exptime) combis
    cadence: int
      cadence of observations
    nproc: int
     number of proc for multiprocessing
    
    """
    full_wells = [90000,120000]

    timesat = pd.DataFrame()
    
    for (x1,color) in x1_color:
        for season in seasons:
            for (nexp,expt) in nexp_expt:
                sat = SaturationTime(dirFile,x1,color,nexp,expt,season,cadence_obs)
                for full_well in full_wells:
                    res = sat.multi_time(full_well,npp=nproc)
                    timesat = pd.concat((timesat,res))
                    print(res)


    np.save('TimeSat_{}.npy'.format(cadence_obs),timesat.to_records(index=False))
"""
seeing = 0.2355/3.
#for psf_type in ['single_gauss','double_gauss']:
for psf_type in ['single_gauss']:
    test = PSF_pixels(seeing,psf_type)
    resa = test.get_flux_map(integ_type='quad')
    resb = test.get_flux_map()
    idxa = np.argsort(resa['pixel_frac'])[::-1][0]
    idxb = np.argsort(resb['pixel_frac'])[::-1][0]
    print(resa.dtype)
    print(psf_type,resa[idxa])
    print(psf_type,resb[idxb])
"""



"""
time_ref = time.time()
psf_type='single_gauss'

PixelPSFSeeing(psf_type)
    
print('done',time.time()-time_ref)

PlotMaxFrac()
PlotMaxFrac(psf_type='double_gauss',title='Double gaussian profile')

PlotPixel(0.5,'single_gauss','xpixel',0.,'ypixel',0.,'xc','yc','Single Gaussian profile',type_plot='contour')
"""



######################Saturation mag
"""
mag_flux = MagToFlux()

mag_flux.plot()

psf_types=['single_gauss']
exptimes = [5.,15.,30.]
full_wells = [90000.,120000.]

res_sat = None
for psf_type in psf_types:
    mag_sat = MagSaturation(psf_type=psf_type)
    for exptime in exptimes:
        for full_well in full_wells:
            res = mag_sat(exptime,full_well)
            print(res)
            if res_sat is None:
                res_sat = res
            else:
                res_sat = np.concatenate((res_sat,res))
            
    
PlotMagSat('gri',res_sat)
"""

#################Fake Observations

# this is to generate observations : list of (nexp, exptime)
nexp_expt = [(1,5),(1,15),(1,30)]
obs = Observations('../DB_Files','descddf_v1.4_10yrs')
cadence_obs = 1
# plot the seeing
#obs.plotSeeings()

# make observations
#obs.make_all_obs(cadence=cadence_obs)

#perform simulation on Observations_* files
nexp_expt = [(1,15),(1,30)]
x1_color = [(0.0,0.0)]
seasons = [2]

Simulations(nexp_expt=nexp_expt ,x1_color=x1_color,seasons=seasons,cadence=cadence_obs)

#estimate the saturation time here
estimateSaturationTime('Output_Simu',x1_color,seasons,nexp_expt,cadence_obs,nproc=8)


df = pd.DataFrame(np.load('TimeSat_{}.npy'.format(cadence_obs),allow_pickle=True))

plotTimeSaturation(0.0,0.0,df)


plt.show()
