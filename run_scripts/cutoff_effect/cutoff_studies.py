import sncosmo
from astropy.cosmology import w0waCDM
import numpy as np
from lsst.sims.catUtils.dust import EBV
import os
from scipy.interpolate import griddata
from sn_tools.sn_telescope import Telescope
from lsst.sims.photUtils import Bandpass, Sed
from astropy import units as u
import pandas as pd


def SALT2Templates(SALT2Dir='SALT2.Guy10_UV2IR', blue_cutoff=3800.):

    for vv in ['salt2_template_0', 'salt2_template_1']:
        fName = '{}/{}_orig.dat'.format(SALT2Dir, vv)
        data = np.loadtxt(fName, dtype={'names': ('phase', 'wavelength', 'flux'),
                                        'formats': ('f8', 'i4', 'f8')})
        print(data)
        data['flux'][data['wavelength'] <= blue_cutoff] = 0.0

        print(data)
        np.savetxt('{}/{}.dat'.format(SALT2Dir, vv),
                   data, fmt=['%1.2f', '%4d', '%.7e', ])


class Cutoffs:

    def __init__(self, x1=-2.0, color=0.2, daymax=0.0, blue_cutoff=380., redcutoff=800., mjdCol='observationStartMJD', filterCol='filter', SALT2Dir=''):

        model = 'salt2-extended'
        version = '1.0'

        source = sncosmo.get_source(model, version=version)

        if SALT2Dir != '':
            source = sncosmo.SALT2Source(modeldir=SALT2Dir)

        dustmap = sncosmo.OD94Dust()

        lsstmwebv = EBV.EBVbase()

        self.mjdCol = mjdCol
        self.filterCol = filterCol
        self.x1 = x1
        self.color = color
        self.daymax = daymax
        # SN model
        self.SN = sncosmo.Model(source=source,
                                effects=[dustmap, dustmap],
                                effect_names=['host', 'mw'],
                                effect_frames=['rest', 'obs'])

        # SN parameters
        self.SN.set(t0=daymax)
        self.SN.set(c=color)
        self.SN.set(x1=x1)

        # SN normalisation
        # cosmology
        H0 = 72.0
        Omega_m = 0.3
        Omega_l = 0.70
        w0 = -1.0
        wa = 0.0
        self.cosmo = self.cosmology(H0, Omega_m, Omega_l, w0, wa)

        # x0 normalisation
        self.x0_grid = self.x0(-19.0906)
        self.x0_from_grid = griddata((self.x0_grid['x1'], self.x0_grid['color']),
                                     self.x0_grid['x0_norm'], (x1, color),  method='nearest')

        # wavelength for the model
        wave_min = 3000.
        wave_max = 11501.

        self.wave = np.arange(wave_min, wave_max, 1.)

        # telescope
        self.telescope = Telescope(airmass=1.2)

        lambda_min = dict(zip('grizy', [300., 670., 300., 300., 300.]))
        # band registery in sncosmo
        for band in 'grizy':
            throughput = self.telescope.atmosphere[band]
            print(band, lambda_min[band])
            idx = throughput.wavelen <= lambda_min[band]
            # throughput.sb[idx] = 0.

            bandcosmo = sncosmo.Bandpass(
                throughput.wavelen, throughput.sb, name='LSST::'+band, wave_unit=u.nm)
            sncosmo.registry.register(bandcosmo, force=True)

    def x0(self, absMag):
        """
        Method to load x0 data

        Parameters
        ---------------
        config: dict
          parameters to load and (potentially) regenerate x0s

        Returns
        -----------

        """
        # check whether X0_norm file exist or not (and generate it if necessary)
        x0normFile = 'reference_files/X0_norm_{}.npy'.format(absMag)
        if not os.path.isfile(x0normFile):
            # if this file does not exist, grab it from a web server
            check_get_file(config['Web path'], 'reference_files',
                           'X0_norm_{}.npy'.format(absMag))

        return np.load(x0normFile)

    def cosmology(self, H0, Omega_m, Omega_l, w0, wa):

        cosmo = w0waCDM(H0=H0,
                        Om0=Omega_m,
                        Ode0=Omega_l,
                        w0=w0, wa=wa)
        return cosmo

    def set_x0(self, z):

        # luminosity distance
        lumidist = self.cosmo.luminosity_distance(z).value*1.e3

        x0 = self.x0_from_grid / lumidist ** 2
        alpha = 0.13
        beta = 3.
        x0 *= np.power(10., 0.4*(alpha * self.x1 - beta * self.color))
        print('x0 normalisation', x0)
        self.SN.set(x0=x0)

    def __call__(self, obs, z):

        # no dust
        ebvofMW = 0.0
        self.SN.set(mwebv=ebvofMW)

        # z val
        self.SN.set(z=z)
        # x0 normalisation
        self.set_x0(z)

        # Select obs depending on min and max phases
        # blue and red cutoffs applied

        """
        obs = self.cutoff(obs, self.sn_parameters['daymax'],
                          self.sn_parameters['z'],
                          self.sn_parameters['min_rf_phase'],
                          self.sn_parameters['max_rf_phase'],
                          self.sn_parameters['blue_cutoff'],
                          self.sn_parameters['red_cutoff'])
        """

        obs = self.selectObsPhase(obs, z)
        # Get the fluxes (vs wavelength) for each obs
        print(obs[self.mjdCol])

        obsdf = pd.DataFrame(obs)
        obsdf[self.filterCol] = 'LSST::'+obsdf[self.filterCol]

        print('hh', obsdf)
        fluxes_cosmo = self.SN.bandflux(
            obsdf[self.filterCol], obsdf[self.mjdCol], zpsys='ab', zp=2.5*np.log10(3631))
        fluxcov_cosmo = self.SN.bandfluxcov(
            obsdf[self.filterCol], obsdf[self.mjdCol], zpsys='ab', zp=2.5*np.log10(3631))

        cov = np.sqrt(np.diag(fluxcov_cosmo[1]))
        print('fluxes', cov/fluxes_cosmo)

        # flux estimates

        throughput = self.telescope.atmosphere
        """
        band = 'r'
        lambda_min = 700.


        """

        fluxes = 10.*self.SN.flux(obs[self.mjdCol], self.wave)
        self.estimateFluxes(self.wave/10., fluxes, obs, throughput)

        self.plot(fluxes, z, throughput)

    def estimateFluxes(self, wavelength, fluxes, obs, throughput):

        wavelength = np.repeat(wavelength[np.newaxis, :], len(fluxes), 0)
        SED_time = Sed(wavelen=wavelength, flambda=fluxes)

        fluxes = []
        transes = []
        nvals = range(len(SED_time.wavelen))
        print('jjj', nvals)
        # Arrays of SED, transmissions to estimate integrated fluxes

        seds = [Sed(wavelen=SED_time.wavelen[i], flambda=SED_time.flambda[i])
                for i in nvals]
        transes = np.asarray([throughput[obs[self.filterCol][i]]
                              for i in nvals])
        int_fluxes = np.asarray(
            [seds[i].calcFlux(bandpass=transes[i]) for i in nvals])

        print(int_fluxes, obs[self.filterCol])

    def selectObsPhase(self, obs, z):
        obs_sel = None
        for b in 'grizy':
            idx = obs[self.filterCol] == b
            sel = obs[idx]
            if len(sel) > 0:
                phases = (sel[self.mjdCol]-self.daymax)/(1.+z)
                idxa = np.argmin(np.abs(phases))
                if obs_sel is None:
                    obs_sel = np.array(sel[idxa])
                else:
                    obs_sel = np.hstack([obs_sel, np.array(sel[idxa])])
        return obs_sel

    def plot(self, fluxes, z, throughput):

        import matplotlib.pyplot as plt
        self.pltDef(plt)
        fig, ax = plt.subplots()
        fig.suptitle('z = {}'.format(z))
        for bb in 'grizy':
            ax.plot(
                10.*throughput[bb].wavelen, throughput[bb].sb)
        axa = ax.twinx()
        # axa.plot(self.wave, fluxes[0, :], color='k')
        for fflux in fluxes:
            idx = fflux > 10e-25
            axa.plot(self.wave[idx], fflux[idx], color='k')
            axa.fill_between(self.wave[idx], 0., fflux[idx], alpha=0.05)

        ax.set_ylim([0., None])
        axa.set_ylim([0., None])
        ax.set_xlabel('wavelength [nm]')
        ax.set_ylabel('sb (0-1)')
        axa.set_ylabel('Flux [ergs / s / cm$^2$ / Angstrom]')
        plt.show()

    def pltDef(self, plt):

        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['legend.fontsize'] = 12
        plt.rcParams['font.size'] = 12


fake_data = 'Fake_DESC.npy'

if not os.path.isfile(fake_data):
    # data do not exist -> have to generate them
    fake_config = 'input/Fake_cadence/Fake_cadence.yaml'
    cmd = 'python run_scripts/fakes/make_fake.py --config {} --output {}'.format(
        fake_config, fake_data.split('.')[0])
    os.system(cmd)


z = 0.77
lambda_g_min = 6700.
SALT2Dir = 'SALT2.Guy10_UV2IR'
blue_cutoff = lambda_g_min/(1.+z)
blue_cutoff = 3600.
# make the SALT2 model with this cutoff
SALT2Templates(SALT2Dir=SALT2Dir, blue_cutoff=blue_cutoff)

mysimu = Cutoffs(SALT2Dir=SALT2Dir)

obs = np.load('Fake_DESC.npy')

mysimu(obs, z)
