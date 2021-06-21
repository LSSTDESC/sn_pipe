from sn_saturation.psf_pixels import PSF_pixels, PlotPixel, PlotMaxFrac
from sn_saturation.mag_saturation import MagToFlux, MagSaturation, plotMagContour, plotDeltamagContour
from sn_saturation.observations import Observations, prepareYaml
from sn_saturation.sn_sat_effects import SaturationTime, plotTimeSaturation, plot_gefficiency, plotTimeSaturationContour
import numpy as np
import time
#import matplotlib.pyplot as plt
import pandas as pd
import os
from sn_saturation import plt
from sn_tools.sn_utils import multiproc


class Simulations:
    """
    class to perform LC simulations

    Parameters
    ---------------
    nexp_expt: list(tuple), opt
      list of (nexp, exposuretime) (default: [(1, 5), (1, 15), (1, 30)])
    dbDir: str, opt
       location dir of OS files (default: '../../DB_Files')
    dbName: str, opt
      OS name to get seeing values (default:  'baseline_nexp1_v1.7_10yrs')
    cadence: int, opt
      cadence of observation (default: 3 days)


    """

    def __init__(self, nexp_expt=[(1, 1), (1, 5), (1, 10), (1, 15), (1, 20), (1, 30), (1, 40)],
                 dbDir='../../DB_Files',
                 dbName='baseline_nexp1_v1.7_10yrs',
                 cadence=3,
                 x1_color=[(0.0, 0.0)],
                 seasons=[2]):

        self.nexp_expt = nexp_expt
        self.dbDir = dbDir
        self.dbName = dbName
        self.cadence = cadence
        self.x1_color = x1_color
        self.seasons = seasons

        # generate fake obs
        self.generateFakes()

        # make simulations here
        self.simulations()

    def generateFakes(self):
        """
        Method to generate fake observations
        """
        # Fake Observations

        # this is to generate observations : list of (nexp, exptime)
        obs = Observations(self.dbDir, self.dbName)

        # plot the seeing
        # obs.plotSeeings()
        # plt.show()
        # make observations
        obs.make_all_obs(nexp_expt=self.nexp_expt, cadence=self.cadence)

    def simulations(self):
        """
        Method to perform a set of simulations using multiprocessing

        """

        nproc = 4
        # input_yaml = 'input/saturation/param_simulation_gen.yaml'

        for (x1, color) in self.x1_color:
            for season in self.seasons:
                params = {}
                params['x1'] = x1
                params['color'] = color
                params['season'] = season
                params['cadence'] = self.cadence
                params['dbName'] = self.dbName
                multiproc(self.nexp_expt, params, self.simulationIndiv, nproc)

    def simulationIndiv(self, nexp_expt, params, j=0, output_q=None):
        """
        Method to perform simulations on a subset of parameters

        Parameters
        ---------------
        nexp_expt: list(tuple)
          list of (nexp, exptime)
        params: dict
          parameter dict of this function
        j: int, opt
          multiproc number (default: 0)
        output_q: multiprocessing queue, opt
          queue for multiprocessing (default: None)

        """
        cadence = params['cadence']
        x1 = params['x1']
        color = params['color']
        season = params['season']
        dbName = params['dbName']

        for (nexp, expt) in nexp_expt:

            dbName = 'Observations_{}_{}_{}'.format(nexp, expt, cadence)
            prodid = 'Saturation_{}_{}_{}_{}_{}_{}'.format(
                nexp, expt, x1, color, cadence, season)
            cmd = 'python run_scripts/simulation/run_simulation.py --npixels 1'
            cmd += ' --radius 0.01'
            cmd += ' --RAmin 0.0'
            cmd += ' --RAmax 0.1'
            cmd += ' --SN_x1_type unique --SN_x1_min {}'.format(x1)
            cmd += ' --SN_color_type unique --SN_color_min {}'.format(
                color)
            cmd += ' --SN_z_type uniform --SN_z_min 0.01 --SN_z_max 0.05 --SN_z_step 0.001'
            #cmd += ' --SN_z_type uniform --SN_z_min 0.01 --SN_z_max 0.011 --SN_z_step 0.001'
            cmd += ' --SN_daymax_type uniform --SN_daymax_step 3.'
            #cmd += ' --SN_daymax_type unique --SN_daymax_step 50.'
            cmd += ' --SN_ebvofMW 0.'

            cmd += ' --dbName {} --dbExtens npy --dbDir .'.format(dbName)
            cmd += ' --ProductionIDSimu {}'.format(prodid)
            cmd += ' --Observations_fieldtype Fake --Observations_coadd 0'
            cmd += ' --Simulator_name sn_simulator.sn_fast'
            cmd += ' --Observations_season {}'.format(season)
            #cmd += ' --Display_LC_display 1'
            os.system(cmd)

        if output_q is not None:
            return output_q.put({j: None})
        else:
            return None


def estimate_magsat(psf_types=['single_gauss'], exptimes=np.arange(1., 62., 2.), full_wells=[90000., 120000.]):
    """
    Method to estimate saturated magnitude

    Parameters
    ---------------
    psf_types: list(str), opt
      list of psf models to consider (default: ['single_gauss'])
    exptimes: array, opt
      array of exposure times (default: np.arange(1.,62.,2.)
    full_wells: list(float), opt
      list of full wells (default: [90000, 120000] pe)

    """
    res_sat = None
    for psf_type in psf_types:
        mag_sat = MagSaturation(psf_type=psf_type)
        for exptime in exptimes:
            for full_well in full_wells:
                res = mag_sat(exptime, full_well)
                print(res)
                if res_sat is None:
                    res_sat = res
                else:
                    res_sat = np.concatenate((res_sat, res))

    np.save(mag_sat_file, np.copy(res_sat))


class PixelPSFSeeing:
    """
    class to estimate the flux fraction max as a function of the seeing depending on PSF

    Parameters
    --------------
    psf_type: str
      psf profile

    """

    def __init__(self, psf_type):

        self.seeings = np.arange(0.3, 2.6, 0.01)
        self.psf_type = psf_type
        prefix = 'PSF_pixel'

        resfi = self.loop_seeing()

        # save the results in a file
        np.save('{}_{}.npy'.format(prefix, psf_type), np.copy(resfi))

        summary = self.summary(resfi)
        np.save('{}_{}_summary.npy'.format(prefix, psf_type), summary)

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

        df = df.round({'xpixel': 2, 'ypixel': 2,
                       'xc': 2, 'yc': 2, 'seeing': 2})

        # take the pixel in (0,0)
        idx = np.abs(df['xpixel']) < 1.e-5
        idx &= np.abs(df['ypixel']) < 1.e-5
        df = df[idx]

        grp = df.groupby(['seeing']).apply(lambda x: pd.DataFrame({'pixel_frac_max': [x['pixel_frac'].max()],
                                                                   'pixel_frac_min': [x['pixel_frac'].min()],
                                                                   'pixel_frac_med': [x['pixel_frac'].median()]})).reset_index()

        finalresu = grp[['seeing', 'pixel_frac_max',
                         'pixel_frac_min', 'pixel_frac_med']].to_records(index=False)

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
            toproc = PSF_pixels(seeing, self.psf_type)
            res = toproc.get_flux_map()

            if resfi is None:
                resfi = res
            else:
                resfi = np.concatenate((resfi, res))

        return resfi


def estimateSaturationTime(dirFile, nexp_expt_simu, x1_color, seasons, nexp_expt, cadence_obs, band, nproc):
    """
    Function to estimate saturation time vs z

    Parameters
    --------------
    dirFile: str
      location dir of the simulation
    nexp_expt_simu: tuple
      nexp and exposure time for ref simu file
    x1_color: list(tuple)
      list of (x1,color) combi
    season: list(int)
      list of the seasons to gen
    nexp_expt: list(tuples)
      list of (nexp, exptime) combis
    cadence: int
      cadence of observations
    band: str
      band to process
    nproc: int
     number of proc for multiprocessing

    """
    full_wells = [90000, 120000]
    
    timesat = pd.DataFrame()
    
    for (x1, color) in x1_color:
        for season in seasons:
            sat = SaturationTime(dirFile, x1, color,
                             nexp_expt_simu[0], nexp_expt_simu[1], season, cadence_obs, band)
            for (nexp, expt) in nexp_expt:
                for full_well in full_wells:
                    res = sat.multi_time(full_well, nexp, expt, npp=nproc)
                    timesat = pd.concat((timesat, res))
                    print(res)

    np.save('TimeSat_{}_{}.npy'.format(cadence_obs, band),
            timesat.to_records(index=False))


"""
seeing = 0.2355/3.
# for psf_type in ['single_gauss','double_gauss']:
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


time_ref = time.time()
psf_type = 'single_gauss'

# PixelPSFSeeing(psf_type)

print('done', time.time()-time_ref)

# PlotMaxFrac()
#PlotMaxFrac(psf_type=psf_type, title='Single gaussian profile')
"""
PlotMaxFrac(psf_type=psf_type, title='')

PlotPixel(0.7, 'single_gauss', 'xpixel', 0., 'ypixel', 0., 'xc',
          'yc', 'Single Gaussian profile', type_plot='contour')
"""

# Saturation mag
"""
mag_flux = MagToFlux()

mag_flux.plot()

mag_sat_file = 'mag_sat.npy'

if not os.path.isfile(mag_sat_file):
    estimate_magsat()

#plotMagSat('gri', res_sat)
plotMagContour(mag_sat_file)
plotDeltamagContour()
plt.show()
"""


# make LC simulation here
"""
nexp_expt = []
for expt in range(1, 64, 4):
    nexp_expt.append((1, expt))
"""    
cadence_obs = 1

nexp_expt=[(1,30)]
#Simulations(dbDir='../DB_Files',nexp_expt=nexp_expt, cadence=cadence_obs)

#print(test)
#cadence_obs = 3
# estimate the saturation time here
band = 'g'
nexp_expt = []
for expt in range(0, 70, 10):
    nexp_expt.append((1, expt))

nexp_expt[0] = (1,1)


print(nexp_expt)
#estimateSaturationTime('Output_Simu', nexp_expt_simu=(1,30),x1_color=[(0.0, 0.0)], seasons=[2],
#                       nexp_expt=nexp_expt, cadence_obs=cadence_obs, nproc=8, band=band)


plotTimeSaturationContour(0.0, 0.0)

plt.show()


df = pd.DataFrame(
    np.load('TimeSat_{}_{}.npy'.format(cadence_obs, band), allow_pickle=True))

plotTimeSaturation(0.0, 0.0, df)
plot_gefficiency(0.0, 0.0, df)

plt.show()
