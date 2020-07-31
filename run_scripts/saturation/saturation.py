from sn_saturation.psf_pixels import PSF_pixels,PlotPixel,PlotMaxFrac
from sn_saturation.mag_saturation import MagToFlux,MagSaturation,PlotMagSat
from sn_saturation.observations import Observations
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd

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

    
    
"""
seeing = 0.5
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

obs = Observations('../DB_Files','descddf_v1.4_10yrs')

#obs.plotSeeings()
obs.make_all_obs()
plt.show()
