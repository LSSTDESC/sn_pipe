import h5py
import matplotlib.pyplot as plt
from astropy.table import Table, vstack
import numpy as np
from scipy.interpolate import interp1d

filtercolors = dict(zip('ugrizy', ['b', 'c', 'g', 'y', 'r', 'm']))
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['font.size'] = 12


def load_file(paramFile):
    """
    Function to load simulation parameters

    Parameters
    ---------------
    paramFile: str
       name of the parameter file

    Returns
    -----------
    params: astropy table
       with simulation parameters

    """

    f = h5py.File(paramFile, 'r')
    print(f.keys(), len(f.keys()))
    params = Table()
    for i, key in enumerate(f.keys()):
        pars = Table.read(paramFile, path=key)
        params = vstack([params, pars])

    return params


def plot(ax, tab, label='', color_cut=0.04):

    ax.plot(tab['z'], np.sqrt(tab['Cov_colorcolor']), label=label)
    interp = interp1d(np.sqrt(tab['Cov_colorcolor']), tab['z'])

    zlim = interp(color_cut)
    ax.plot(ax.get_xlim(), [color_cut]*2,
            linestyle='--', color='k')
    ax.plot([zlim]*2, [0., 0.08], linestyle='--', color='k')
    mystr = 'z$_{lim}$'
    ax.text(zlim-0.03, 0.085, '{}={}'.format(mystr, np.round(zlim, 2)))


thedir = 'Output_Fit'
fi = 'Fit_sncosmo_Fake_Fake_DESC_seas_-1_-2.0_0.2.hdf5'

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
