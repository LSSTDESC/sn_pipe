import h5py
import matplotlib.pyplot as plt
from astropy.table import Table, vstack
import numpy as np
from scipy.interpolate import interp1d
import pandas as pd

filtercolors = dict(zip('ugrizy', ['b', 'c', 'g', 'y', 'r', 'm']))
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['font.size'] = 12


class FitPlots:
    """
    class to display some results from sn_fit_lc

    Parameters
    ---------------
    files: dict
      data to process
      the dict should have:
        - as keys: labels
        - as values: full path to the files

    """

    def __init__(self, files):

        self.SN_table = {}

        for key, val in files.items():
            self.SN_table[key] = self.load_file(val)

    def load_file(self, fileName):
        """
        Method  to load SN data

        Parameters
        ---------------
        fileName: str
          name of the file to load

        Returns
        -----------
        params: astropy table
        with SN

        """

        f = h5py.File(fileName, 'r')
        print(f.keys(), len(f.keys()))
        sn = Table()
        for i, key in enumerate(f.keys()):
            sni = Table.read(fileName, path=key)
            sn = vstack([sn, sni])

        return sn

    def plot2D(self, tabs, varx, vary, legx, legy):

        fig, ax = plt.subplots()

        self.plot2D_indiv(ax, tabs, varx, vary)

        ax.grid()
        ax.set_xlabel(legx)
        ax.set_ylabel(legy)
        ax.set_ylim([0., 0.1])
        ax.set_xlim([0.01, 0.78])
        ax.legend(loc='upper left')

    def plot2D_indiv(self, ax, tabs, varx, vary, label='', color_cut=0.04):

        dict_interp = {}
        for key, tab in tabs.items():

            idx = tab[vary] > 0
            sel = tab[idx]

            ax.plot(sel[varx], np.sqrt(sel[vary]), label=key)

            """
            interp = interp1d(
                np.sqrt(sel[vary]), sel[varx], bounds_error=False, fill_value=0.)

            dict_interp[key] = interp1d(sel[varx], np.sqrt(
                sel[vary]), bounds_error=False, fill_value=0.)

            zlim = interp(color_cut)
            ax.plot(ax.get_xlim(), [color_cut]*2,
                    linestyle='--', color='k')
            ax.plot([zlim]*2, [0., 0.08], linestyle='--', color='k')
            mystr = 'z$_{lim}$'
            ax.text(zlim-0.03, 0.085, '{}={}'.format(mystr, np.round(zlim, 2)))
            """
        # Compare variation
        if len(tabs) < 1:

            zplot = np.arange(0.05, 0.8, 0.01)
            df = pd.DataFrame(zplot.tolist(), columns=['z'])
            for key, val in dict_interp.items():
                kk = '_'.join(key.split('_')[:-1])
                print('kkkk', kk)
                df['sigC_{}'.format(kk)] = val(zplot)

            print(df)
            figb, axb = plt.subplots()
            axb.plot(df['z'], df['sigC_fast_cosmo'] /
                     df['sigC_cosmo_cosmo'], label='fc/cc')
            axb.plot(df['z'], df['sigC_fast_fast'] /
                     df['sigC_cosmo_cosmo'], label='ff/cc')
            axb.plot(df['z'], df['sigC_fast_fast'] /
                     df['sigC_fast_cosmo'], label='ff/fc')

            axb.legend()
            axb.grid()
            axb.set_xlabel('z')
            axb.set_ylabel('$\sigma_{Color}$ ratio')
            axb.set_ylim([0.95, 1.1])
            axb.set_xlim([0.01, 0.78])


dictfiles = {}

for bluecutoff in [360.0, 370.0, 380.0]:
    for ebvofMW in [0.0]:
        thedir = 'Output_Fit_{}_800.0_ebvofMW_{}'.format(bluecutoff, ebvofMW)
        fi_cosmo_cosmo = 'Fit_sn_cosmo_Fake_Fake_DESC_seas_-1_-2.0_0.2_{}_800.0_ebvofMW_{}_sn_cosmo.hdf5'.format(
            bluecutoff, ebvofMW)
        fi_fast_cosmo = 'Fit_sn_fast_Fake_Fake_DESC_seas_-1_-2.0_0.2_{}_800.0_ebvofMW_{}_sn_cosmo.hdf5'.format(
            bluecutoff, ebvofMW)
        fi_fast_fast = 'Fit_sn_fast_Fake_Fake_DESC_seas_-1_-2.0_0.2_{}_800.0_ebvofMW_{}_sn_fast.hdf5'.format(
            bluecutoff, ebvofMW)

        dictfiles['cosmo_cosmo_{}_{}'.format(bluecutoff,
                                             ebvofMW)] = '{}/{}'.format(thedir, fi_cosmo_cosmo)
        """
        dictfiles['fast_cosmo_{}_{}'.format(bluecutoff,
                                            ebvofMW)]='{}/{}'.format(thedir, fi_fast_cosmo)
        dictfiles['fast_fast_{}_{}'.format(bluecutoff,
                                           ebvofMW)]='{}/{}'.format(thedir, fi_fast_fast)
        """

fitplot = FitPlots(dictfiles)
fitplot.plot2D(fitplot.SN_table, 'z', 'Cov_colorcolor',
               '$z$', '$\sigma_{color}$')

plt.show()
"""
tab = load_file('{}/{}'.format(thedir, fi))

thedir = 'Output_Fit_noy'
tab_noy = load_file('{}/{}'.format(thedir, fi))


print(tab[['mbfit']])

fig, ax = plt.subplots()

plot(ax, tab, label='grizy: 1/7/50/23/119')
plot(ax, tab_noy, label='griz: 1/7/100/46')

ax.grid()
ax.set_xlabel('z')
ax.set_ylabel('$\sigma_C$')
ax.set_ylim([0., 0.1])
ax.set_xlim([0., 0.8])
ax.legend(loc='upper left')

plt.show()
"""
