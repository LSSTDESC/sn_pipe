from sn_tools.sn_io import loopStack
import os
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd


class SNR_m5:
    """
    Class to estimate, for each band and each considered LC
    the Signal-to-Noise Ratio vs fiveSigmaDepth

    Parameters
    ---------------
    inputDir: str
      input directory where the LC file is located
    refFile: str
      name of the LC file

    """

    def __init__(self, inputDir, refFile):

        # load the reference file

        refdata = pd.DataFrame(np.copy(self.load(inputDir, refFile)))
        print(refdata.columns)
        refdata['band'] = refdata['band'].map(lambda x: x.decode()[-1])

        # load the gamma file
        gamma = self.load('reference_files', 'gamma.hdf5')

        # print(gamma)

        # load mag to flux corresp

        mag_to_flux = np.load('reference_files/Mag_to_Flux_SNCosmo.npy')
        print(mag_to_flux.dtype)

        # select exposure time of 30s and
        idx = np.abs(gamma['exptime']-30.) < 1.e-5
        selgamma = gamma[idx]

        bands = 'grizy'

        # get interpolators for gamma and magflux
        gammadict = {}
        magfluxdict = {}

        for b in bands:
            io = selgamma['band'] == b
            ib = mag_to_flux['band'] == b
            gammadict[b] = interp1d(
                selgamma[io]['mag'], selgamma[io]['gamma'], bounds_error=False, fill_value=0.)
            magfluxdict[b] = interp1d(
                mag_to_flux[ib]['m5'], mag_to_flux[ib]['flux_e'], bounds_error=False, fill_value=0.)

        # SNR vs m5 estimation
        resdf = pd.DataFrame()
        for b in 'grizy':
            idx = refdata['band'] == b
            datab = refdata[idx]
            res = self.process(datab, gammadict[b], magfluxdict[b])
            resdf = pd.concat((resdf, res))

        # save the result in a numpy array
        np.save('SNR_m5.npy', resdf.to_records(index=False))

        self.plot(resdf)

    def process(self, data, gamma, magtoflux):
        """
        Method to estimate SNR vs m5

        Parameters
        ---------------
        data: pandas df
          LC DataFrame
        gamma: interp1d
           gamma values interpolator (m5->gamma)
        magtoflux: interp1d
           mag to flux interpolator (mag -> flux) 

        Returns
        ----------
        pandas df with the following cols:
        band, z, SNR, SNR_bd, m5 where
        - SNR = 1/srand
        - SNR_bd = SNR if background dominated (sigma = sigma_5 = flux_5/5. 
        where flux_5 is estimated from m5)

        """
        res = pd.DataFrame()
        datab = data.copy()
        for m5 in np.arange(23., 27., 0.01):
            datab.loc[:, 'm5'] = m5
            datab.loc[:, 'gamma'] = gamma(m5)
            datab.loc[:, 'SNR'] = 1. / \
                self.srand(datab['gamma'],
                           datab['mag'], datab['m5'])

            datab.loc[:, 'SNR_bd'] = datab['flux_e_sec']/(magtoflux(m5)/5.)
            grp = datab.groupby(['band', 'z']).apply(
                lambda x: self.calc(x, m5)).reset_index()
            res = pd.concat((res, grp))
        return res

    def calc(self, grp, m5):
        """
        Method to estimate some quantities per group (1 group = 1LC/z/band)

        Parameters
        --------------
        grp: pandas grp
          data to process
        m5: float
          fiveSigmaDepth value

        Returns
        -----------
        pandas df with the following cols:
         - SNR = np.sqrt(np.sum(SNR*SNR)
        - SNR_bd = np.sqrt(np.sum(SNR_bd*SNR_bd)
        - m5: fiveSigmaDepth

        """
        SNR = np.sqrt(np.sum(grp['SNR']**2))
        SNR_bd = np.sqrt(np.sum(grp['SNR_bd']**2))

        return pd.DataFrame({'SNR': [SNR],
                             'SNR_bd': [SNR_bd],
                             'm5': [m5]})

    def load(self, theDir, fname):
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

    def srand(self, gamma, mag, m5):
        """
        Method to estimate sigma_rand = 1./SNR
        see eq (5) of 
        LSST: from Science Drivers to Reference Design and Anticipated Data Products
        (arXiv:0805.2366 [astro-ph])

        Parameters
        ---------------
        gamma: array (float)
          gamma parameter values
        mag: array (float)
          magnitude
        m5: array (float)
          fiveSigmaDepth values

        Returns
        -----------
        sigma_rand = (0.04-gamma)x-gammax**2
        with x = 10**(0.4*(mag-m5))

        """
        x = 10**(0.4*(mag-m5))
        return np.sqrt((0.04-gamma)*x+gamma*x**2)

    def plot(self, data, zref=0.7):
        """
        Method to plot SNR vs m5

        """

        import matplotlib.pyplot as plt
        fontsize = 15
        for b in 'grizy':
            idx = data['band'] == b
            sel = data[idx]
            fig, ax = plt.subplots()
            fig.suptitle('{} band - z = {}'.format(b, zref), fontsize=fontsize)
            idxb = np.abs(sel['z']-zref) < 1.e-5
            selb = sel[idxb]
            ax.plot(selb['m5'], selb['SNR'],  color='k',
                    label='1./$\sigma_{rand}$')
            ax.plot(selb['m5'], selb['SNR_bd'], color='r',
                    label='background dominated')

            ax.set_xlabel('m$_{5}$ [mag]', fontsize=fontsize)
            ax.set_ylabel('SNR', fontsize=fontsize)
            ax.legend(fontsize=fontsize)
            ax.yaxis.set_tick_params(labelsize=15)
            ax.xaxis.set_tick_params(labelsize=15)

        plt.show()


inputDir = 'input/sn_studies'
refFile = 'Fakes_NSNMetric_Fake_lc_nside_64_coadd_0_0.0_360.0_-1.0_-1.0_0.hdf5'

snr = SNR_m5(inputDir, refFile)
