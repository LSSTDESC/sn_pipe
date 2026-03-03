
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sn_plotter_tools.plot_tools import plot_xy
from sn_tools.sn_cosmo_model import cosmo_values
import time

plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['figure.titlesize'] = 20
plt.rcParams['legend.fontsize'] = 20
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['figure.titleweight'] = 'bold'
# plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 20


def plot_loop(res, yvar='w', yleg='$w_{DE}$'):
    """
    Function to plot cosmo results

    Parameters
    ----------
    res : pandas df
        Data to plot.
    yvar : str, optional
        y-axis variable. The default is 'w'.
    yleg : str, optional
        y-axis legend. The default is '$w_{DE}$'.

    Returns
    -------
    None.

    """

    configs_pl = res['config'].unique()

    fig, ax = plt.subplots(figsize=(12, 8))
    res = res.sort_values(by=['config'])

    mark = ['o', 's', '*', 'h']
    ls = ['solid', 'dotted', 'dashed', 'dashdot']
    col = ['b', 'r', 'g', 'k']

    for i, conf in enumerate(configs_pl):
        idx = res['config'] == conf
        sel = res[idx]
        sel = sel.sort_values(by=['z'])
        
        label = ('{} - ({})=({})'.format(conf, sel['de_params'].unique()[0],
                                        sel['de_values'].unique()[0]))
        plot_xy(sel,
                xvar='z',xleg='$z$',yvar=yvar,yleg=yleg,
                fig=fig,ax=ax,
                label=label,marker=mark[i], color=col[i], linestyle=ls[i])
        
    # ax.set_xscale('log')
    ax.set_xlim([0.01, 1.1])
    ax.set_xlabel(r'$z$')
    ax.set_ylabel(r'{}'.format(yleg))
    ax.grid(visible=True)
    ax.legend()



H0 = 70.0
Om0 = 0.3
Ode0=1.-Om0

cosmo_params = {}

de_params = dict(zip(['w0','wa'],[-1,0.]))
de_params_dde = dict(zip(['w0','wa'],[-0.6,0.4]))

vv = 'astropy'
cosmo_params[vv] = {}
cosmo_params[vv]['de_params'] = de_params
cosmo_params[vv]['de_class'] = 'w0waCDM'
cosmo_params[vv]['de_model'] = 'CPL'
cosmo_params[vv]['de_eos'] = 'w0+wa*z/(1+z)'
cosmo_params[vv]['H0'] = H0
cosmo_params[vv]['Om0'] = Om0
cosmo_params[vv]['Ode0'] = Ode0
cosmo_params[vv]['class_loc'] = 'astropy.cosmology'

vv = 'custom'
cosmo_params[vv] = {}
cosmo_params[vv]['de_params'] = de_params_dde
cosmo_params[vv]['de_class'] = 'DDE_FLRW'
cosmo_params[vv]['de_model'] = 'CPL'
cosmo_params[vv]['de_eos'] = 'w0+wa*z/(1+z)'
cosmo_params[vv]['H0'] = H0
cosmo_params[vv]['Om0'] = Om0
cosmo_params[vv]['Ode0'] = Ode0
cosmo_params[vv]['class_loc'] = 'sn_tools.sn_cosmo_model'

"""
vv = 'customb'
cosmo_params[vv] = {}
cosmo_params[vv]['de_params'] = de_params
cosmo_params[vv]['de_class'] = 'DDE_FLRW_symbol'
cosmo_params[vv]['de_model'] = 'CPL'
cosmo_params[vv]['de_eos'] = 'w0+wa*z/(1+z)'
cosmo_params[vv]['H0'] = H0
cosmo_params[vv]['Om0'] = Om0
cosmo_params[vv]['Ode0'] = Ode0
cosmo_params[vv]['class_loc'] = 'sn_tools.sn_cosmo_model'
"""
print(cosmo_params)

res = pd.DataFrame()
for key, vals in cosmo_params.items():
    time_ref = time.time()
    df = cosmo_values(vals)
    df['config'] = key
    res = pd.concat((res,df))
    print('after',time.time()-time_ref)
    
print(df.columns)
plot_loop(res)
plt.show()
