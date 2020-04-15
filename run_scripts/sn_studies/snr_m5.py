import os
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd
from sn_design_dd_survey.utils import m5_to_flux5, srand, gamma, load


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
    x1: float
      SN stretch
    color: float
      x1 color

    """

    def __init__(self, inputDir, refFile, x1=-2.0, color=0.2):

        outfile = 'SNR_m5.npy'

        if not os.path.isfile(outfile):
            self.process_main(inputDir, refFile, x1, color)

        resdf = pd.DataFrame(np.copy(np.load(outfile, allow_pickle=True)))
        self.get_m5(resdf)
        self.plot(resdf)

    def process_main(self, inputDir, refFile, x1, color):

        # load the reference file

        refdata = pd.DataFrame(np.copy(load(inputDir, refFile)))
        refdata['band'] = refdata['band'].map(lambda x: x.decode()[-1])

        idc = (refdata['x1']-x1) < 1.e-5
        idc &= (refdata['color']-color) < 1.e-5
        refdata = refdata[idc]

        # load the gamma file
        #gamma = self.load('reference_files', 'gamma.hdf5')

        # load mag to flux corresp

        # mag_to_flux = np.load('reference_files/Mag_to_Flux_SNCosmo.npy')
        mag_to_flux = m5_to_flux5('grizy')

        # print(mag_to_flux.dtype)

        # select exposure time of 30s and
        #idx = np.abs(gamma['exptime']-30.) < 1.e-5
        #selgamma = gamma[idx]

        bands = 'grizy'

        # get interpolators for gamma and magflux
        gammadict = gamma(bands)
        # magfluxdict = {}
        """
        for b in bands:
            io = selgamma['band'] == b
            gammadict[b] = interp1d(
                selgamma[io]['mag'], selgamma[io]['gamma'], bounds_error=False, fill_value=0.)
           
        """
        # SNR vs m5 estimation
        resdf = pd.DataFrame()
        # for b in 'grizy':
        zref = 0.7
        for b in 'grizy':
            idx = refdata['band'] == b
            #idx &= np.abs(refdata['z']-zref) < 1.e-5
            datab = refdata[idx]
            res = self.process(datab, gammadict[b], mag_to_flux[b])
            resdf = pd.concat((resdf, res))

        # save the result in a numpy array
        np.save('SNR_m5.npy', resdf.to_records(index=False))

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
        for m5 in np.arange(15., 28., 0.01):
            datab.loc[:, 'm5'] = m5
            datab.loc[:, 'gamma'] = gamma(m5)
            datab.loc[:, 'SNR'] = 1. / \
                srand(datab['gamma'],
                      datab['mag'], datab['m5'])

            datab.loc[:, 'SNR_bd'] = datab['flux_e_sec']/(magtoflux(m5)/5.)
            datab.loc[:, 'f5'] = magtoflux(m5)
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
        sumflux = np.sqrt(np.sum(grp['flux_e_sec']**2.))
        SNR = np.sqrt(np.sum(grp['SNR']**2))
        SNR_bd = np.sqrt(np.sum(grp['SNR_bd']**2))
        SNR_test = 5.*sumflux/grp['f5'].median()

        return pd.DataFrame({'SNR': [SNR],
                             'SNR_bd': [SNR_bd],
                             'SNR_test': [SNR_test],
                             'm5': [m5]})

    def plot(self, data, zref=0.7):
        """
        Method to plot SNR vs m5

        """

        import matplotlib.pyplot as plt
        fontsize = 15
        for b in 'grizy':
            idx = data['band'] == b
            sel = data[idx]
            fig, ax = plt.subplots(nrows=2)
            fig.suptitle('{} band - z = {}'.format(b, zref), fontsize=fontsize)
            idxb = np.abs(sel['z']-zref) < 1.e-5
            selb = sel[idxb]
            ax[0].plot(selb['m5'], selb['SNR'],  color='k',
                       label='1./$\sigma_{rand}$')
            ax[0].plot(selb['m5'], selb['SNR_bd'], color='r',
                       label='background dominated')
            ax[0].plot(selb['m5'], selb['SNR_test'], color='b',
                       label='background dominated - test')

            # ax[0].set_xlabel('m$_{5}$ [mag]', fontsize=fontsize)
            ax[0].set_ylabel('SNR', fontsize=fontsize)
            ax[0].legend(fontsize=fontsize)
            ax[0].yaxis.set_tick_params(labelsize=15)
            # ax[0].xaxis.set_tick_params(labelsize=15)

            ax[1].plot(selb['SNR'], selb['SNR']/selb['SNR_bd'],  color='k',
                       label='1./$\sigma_{rand}$')

            ax[1].set_xlabel('m$_{5}$ [mag]', fontsize=fontsize)
            ax[1].set_ylabel('SNR ratio', fontsize=fontsize)
            ax[1].legend(fontsize=fontsize)
            ax[1].yaxis.set_tick_params(labelsize=15)
            ax[1].xaxis.set_tick_params(labelsize=15)

        plt.show()

    def get_m5(self, data, SNR=dict(zip('grizy', [20., 20., 30., 30., 35.])), zref=0.7):

        for b in SNR.keys():
            idx = data['band'] == b
            idx &= np.abs(data['z']-zref) < 1.e-5
            sel = data[idx]
            if len(sel) > 0:
                myinterpa = interp1d(
                    sel['SNR'], sel['m5'], bounds_error=False, fill_value=0.)
                myinterpb = interp1d(
                    sel['SNR_bd'], sel['m5'], bounds_error=False, fill_value=0.)
                myinterpc = interp1d(
                    sel['SNR_test'], sel['m5'], bounds_error=False, fill_value=0.)
                print(b, SNR[b], myinterpa(SNR[b]),
                      myinterpb(SNR[b]), myinterpc(SNR[b]))


inputDir = 'input/sn_studies'
refFile = 'Fakes_NSNMetric_Fake_lc_nside_64_coadd_0_0.0_360.0_-1.0_-1.0_0.hdf5'

snr = SNR_m5(inputDir, refFile)
