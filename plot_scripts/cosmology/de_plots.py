
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sn_tools.sn_cosmo_model import w0waDDE
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


def dist_mod(H0, Om0, w0, wa, config, z=np.arange(0.01, 1.101, 0.01), model=1):
    """
    Function to estimate the distance modulus

    Parameters
    ----------
    H0 : float
        Hubble constant value.
    Om0 : float
        Omega_matter value.
    w0 : float
        w0 DE parameter.
    wa : float
        wa DE parameter.
    config : str
        config name.
    z : numpy array, optional
        redshift range. The default is np.arange(0.01,1.101,0.01).

    Returns
    -------
    res : pandas df
        Estimated dL and w vs z.

    """

    cosmology = w0waDDE(H0=H0, Om0=Om0, Ode0=1.-Om0, w0=w0, wa=wa, model=model)

    distmod = cosmology.distmod(z).value
    lumidist = cosmology.luminosity_distance(z).value*1.e3
    wz = cosmology.w(z)

    res = pd.DataFrame(z, columns=['z'])
    res['mu'] = distmod
    res['w0'] = w0
    res['wa'] = wa
    res['config'] = config
    res['dl'] = lumidist
    res['wz'] = wz

    return res

def dist_mod_new(de_class,params, config, z=np.arange(0.01, 1.101, 0.05)):
    """
    Function to estimate the distance modulus

    Parameters
    ----------
    H0 : float
        Hubble constant value.
    Om0 : float
        Omega_matter value.
    w0 : float
        w0 DE parameter.
    wa : float
        wa DE parameter.
    config : str
        config name.
    z : numpy array, optional
        redshift range. The default is np.arange(0.01,1.101,0.01).

    Returns
    -------
    res : pandas df
        Estimated dL and w vs z.

    """
    global cosmology
    if de_class == 'astropy':
        to_eval = '{}('.format(params['de_class'])
        for vv in ['H0','Om0','Ode0']:
            to_eval += '{}={},'.format(vv,params[vv])
        for key,vals in params['de_params'].items():
            to_eval += '{}={},'.format(key,vals)
        to_eval += ')'
        #cosmology = w0waDDE(H0=H0, Om0=Om0, Ode0=1.-Om0, w0=w0, wa=wa, model=model)
        print(to_eval)
        from astropy.cosmology import w0waCDM
        cosmology = eval(to_eval)
        distmod = cosmology.distmod(z).value
        print(distmod)
    
    if de_class == 'custom':
        from sn_tools.sn_cosmo_model import DDE_FLRW
        to_eval = '{}('.format(params['de_class'])
        for vv in ['H0','Om0','Ode0']:
            to_eval += '{}={},'.format(vv,params[vv])
        to_eval += 'de_params={},'.format(params['de_params'])
        to_eval += 'de_eos=\"{}\"'.format(str(params['de_eos']))
        to_eval += ')'
        print(to_eval)
        cosmology = eval(to_eval)
        distmod = cosmology.distmod(z).value
        print(distmod)
        
    distmod = cosmology.distmod(z).value
    lumidist = cosmology.luminosity_distance(z).value*1.e3
    wz = cosmology.w(z)

    res = pd.DataFrame(z, columns=['z'])
    res['mu'] = distmod
    for key,vals in params['de_params']:
        res[key] = vals
    res['de_eos'] = params['de_eos']
    res['config'] = config
    res['dl'] = lumidist
    res['wz'] = wz

    return res

"""
def dist_mod_test(H0,Om0,w0,wa,config,z=np.arange(0.01,1.101,0.01)):

   cosmology = MyCosmo(H0=H0,Om0=Om0,Ode0=1.-Om0,w0=w0, wa=wa)

   distmod = cosmology.distmod(z).value
   lumidist = cosmology.luminosity_distance(z).value*1.e3
   wz = cosmology.w(z)
   
   res = pd.DataFrame(z, columns=['z'])
   res['mu'] = distmod
   res['w0'] = w0
   res['wa'] = wa
   res['config'] = config
   res['dl'] = lumidist
   res['wz'] = wz
   
   return res
"""


def plot_loop(res, yvar='wz', yleg='$w_{DE}$'):

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
        w0 = sel['w0'].unique().tolist()[0]
        wa = sel['wa'].unique().tolist()[0]
        label = '$(w_0,w_a)$=({},{})'.format(np.round(w0, 1), np.round(wa, 1))

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

def plot_xy(df, 
            xvar='z',xleg='$z$',
            yvar='wz',yleg='$w_{DE}$',
            fig=None,ax=None,
            label='',color='k',marker='o',
            linestyle='solid',figtit=''):
    """
    Generic plot function

    Parameters
    ----------
    df : pandas df
        Data to plot.
    xvar : str, optional
        x-axis variable. The default is 'z'.
    xleg : str, optional
        x-axis label. The default is '$z$'.
    yvar : str, optional
        y-axis variable. The default is 'wz'.
    yleg : str, optional
        y-axis label. The default is '$w_{DE}$'.
    fig : matplotlib figure, optional
        figure for the plot. The default is None.
    ax : matplotlib axis, optional
        axis for the plot. The default is None.
    label : str, optional
        plot label. The default is ''.
    color : str, optional
        plot color. The default is 'k'.
    marker : str, optional
        plot marker. The default is 'o'.
    linestyle : str, optional
        plot linestyle. The default is 'solid'.
    figtit : str, optional
        Figure title. The default is ''.

    Returns
    -------
    None.

    """
    
    
    fig_orig=True
    if fig is None:
        fig_orig=False
        fig, ax = plt.subplots(figsize=(12, 8))
        
    if figtit != '':
        fig.suptitle(figtit)
    
    ax.plot(df[xvar], df[yvar], label=label,
               marker=marker, color=color, linestyle=linestyle)
    
    if not fig_orig:
        ax.set_xlabel(r'{}'.format(xleg))
        ax.set_ylabel(r'{}'.format(yleg))
        ax.grid(visible=True)
        ax.legend()

H0 = 70.0
Om0 = 0.3
Ode0=1.-Om0

core_packs = ['astropy','custom']

cosmo_params = {}

de_params = dict(zip(['w0','wa'],[-0.6,0.1]))

vv = 'astropy'
cosmo_params[vv] = {}
cosmo_params[vv]['de_params'] = de_params
cosmo_params[vv]['de_class'] = 'w0waCDM'
cosmo_params[vv]['de_model'] = 'CPL'
cosmo_params[vv]['de_eos'] = 'w0+wa*z/(1+z)'
cosmo_params[vv]['H0'] = H0
cosmo_params[vv]['Om0'] = Om0
cosmo_params[vv]['Ode0'] = Ode0

vv = 'custom'
cosmo_params[vv] = {}
cosmo_params[vv]['de_params'] = de_params
cosmo_params[vv]['de_class'] = 'DDE_FLRW'
cosmo_params[vv]['de_model'] = 'CPL'
cosmo_params[vv]['de_eos'] = 'w0+wa*z/(1+z)'
cosmo_params[vv]['H0'] = H0
cosmo_params[vv]['Om0'] = Om0
cosmo_params[vv]['Ode0'] = Ode0
print(cosmo_params)

for key, vals in cosmo_params.items():
    time_ref = time.time()
    dist_mod_new(key,vals,'toto')
    print('after',time.time()-time_ref)
print(test)

"""
w0 = -0.9
wa = -0.5

test = w0waDDE(H0=H0, Om0=Om0, Ode0=1.-Om0,w0=w0,wa=wa)

z = np.arange(0.01,1.1,0.01)

rra = test.de_density_scale(z)
rrb = test.de_density_scale_int(z)

print(rra)
print(rrb)
Parameter(doc="Dark energy equation of state at z=0.", fvalidate="float")
print(toast)
"""
de_params = [(-1.0, 0.0, 1), (-0.84, -0.62, 1),
             (-0.67, -1.09, 1), (-0.72, 3.29, 2)]
# de_params=[(-1.0,0.0,1),(-0.84,-0.62,1),(-0.67,-1.09,1),(-0.75,-0.52,1)]

names = ['config1', 'config2', 'config3', 'config4']

configs = dict(zip(names, de_params))

res = pd.DataFrame()

for key, vals in configs.items():
    df_ = dist_mod(H0, Om0, vals[0], vals[1], key, model=vals[2])
    res = pd.concat((res, df_))

idx = res['config'] == 'config1'

ref = res[idx]

res = res.merge(ref, left_on=['z'], right_on=['z'], suffixes=['', '_ref'])

res['delta_mu'] = res['mu']-res['mu_ref']
res['ratio_dl'] = res['dl']/res['dl_ref']
res['flux_ratio'] = 10**(-0.4*res['delta_mu'])

plot_loop(res)

plot_loop(res, yvar='delta_mu', yleg='$\Delta \mu$ [mag]')
plot_loop(res, yvar='flux_ratio', yleg='$\\frac{\Delta flux}{flux}$')
plt.show()
