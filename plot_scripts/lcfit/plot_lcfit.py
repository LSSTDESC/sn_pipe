import h5py
import matplotlib.pyplot as plt
from astropy.table import Table, vstack
import numpy as np
from scipy.interpolate import interp1d


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


thedir = '.'
fi = 'Fit_sncosmo_Fake_Fake_DESC_seas_-1_-2.0_0.2.hdf5'

tab = load_file('{}/{}'.format(thedir, fi))

print(tab[['mbfit', 'mb_recalc']])

fig, ax = plt.subplots()

color_cut = 0.04
ax.plot(tab['z'], np.sqrt(tab['Cov_colorcolor']))

interp = interp1d(np.sqrt(tab['Cov_colorcolor']), tab['z'])

zlim = interp(color_cut)
ax.plot(ax.get_xlim(), [color_cut]*2, linestyle='--', color='k')
ax.plot([zlim]*2, ax.get_ylim(), linestyle='--', color='k')
mystr = 'z$_{lim}$'
ax.text(0.4, 0.06, '{}={}'.format(mystr, np.round(zlim, 2)))
ax.grid()
ax.set_xlabel('z')
ax.set_ylabel('$\sigma_C$')
ax.set_ylim([0., 0.1])
ax.set_xlim([0., 0.7])

plt.show()
