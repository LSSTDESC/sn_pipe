import h5py
import matplotlib.pyplot as plt
from astropy.table import Table, vstack
import numpy as np


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

print(tab['mbfit'])

plt.plot(tab['z'], np.sqrt(tab['Cov_colorcolor']))

plt.show()
