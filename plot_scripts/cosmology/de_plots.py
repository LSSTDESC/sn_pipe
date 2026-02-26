
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sn_tools.sn_cosmo_model import w0waDDE

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


def plot(res, yvar='wz', yleg='$w_{DE}$'):

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
        ax.plot(sel['z'], sel[yvar], label=label,
                marker=mark[i], color=col[i], linestyle=ls[i])

    # ax.set_xscale('log')
    ax.set_xlim([0.01, 1.1])
    ax.set_xlabel(r'$z$')
    ax.set_ylabel(r'{}'.format(yleg))
    ax.grid(visible=True)
    ax.legend()


H0 = 70.0
Om0 = 0.3

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

plot(res)

plot(res, yvar='delta_mu', yleg='$\Delta \mu$ [mag]')
plot(res, yvar='flux_ratio', yleg='$\\frac{\Delta flux}{flux}$')
plt.show()
