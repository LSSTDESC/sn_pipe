from astropy.cosmology import w0waCDM, FLRW
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy import exp
import astropy.units as u
from astropy.cosmology.parameter import Parameter
# from astropy.cosmology.utils import aszarr
from astropy.cosmology._src.utils import aszarr
from astropy.cosmology._src.flrw import scalar_inv_efuncs
from scipy.integrate import quad

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


class w0waDDE(FLRW):
    r"""FLRW cosmology with a CPL dark energy equation of state and curvature.

    The equation for the dark energy equation of state uses the
    CPL form as described in Chevallier & Polarski [1]_ and Linder [2]_:
    :math:`w(z) = w_0 + w_a (1-a) = w_0 + w_a z / (1+z)`.

    Parameters
    ----------
    H0 : float or scalar quantity-like ['frequency']
        Hubble constant at z = 0. If a float, must be in [km/sec/Mpc].

    Om0 : float
        Omega matter: density of non-relativistic matter in units of the
        critical density at z=0.

    Ode0 : float
        Omega dark energy: density of dark energy in units of the critical
        density at z=0.

    w0 : float, optional
        Dark energy equation of state at z=0 (a=1). This is pressure/density
        for dark energy in units where c=1.

    wa : float, optional
        Negative derivative of the dark energy equation of state with respect
        to the scale factor. A cosmological constant has w0=-1.0 and wa=0.0.

    Tcmb0 : float or scalar quantity-like ['temperature'], optional
        Temperature of the CMB z=0. If a float, must be in [K]. Default: 0 [K].
        Setting this to zero will turn off both photons and neutrinos
        (even massive ones).

    Neff : float, optional
        Effective number of Neutrino species. Default 3.04.

    m_nu : quantity-like ['energy', 'mass'] or array-like, optional
        Mass of each neutrino species in [eV] (mass-energy equivalency enabled).
        If this is a scalar Quantity, then all neutrino species are assumed to
        have that mass. Otherwise, the mass of each species. The actual number
        of neutrino species (and hence the number of elements of m_nu if it is
        not scalar) must be the floor of Neff. Typically this means you should
        provide three neutrino masses unless you are considering something like
        a sterile neutrino.

    Ob0 : float or None, optional
        Omega baryons: density of baryonic matter in units of the critical
        density at z=0.  If this is set to None (the default), any computation
        that requires its value will raise an exception.

    name : str or None (optional, keyword-only)
        Name for this cosmological object.

    meta : mapping or None (optional, keyword-only)
        Metadata for the cosmology, e.g., a reference.

    Examples
    --------
    >>> from astropy.cosmology import w0waCDM
    >>> cosmo = w0waCDM(H0=70, Om0=0.3, Ode0=0.7, w0=-0.9, wa=0.2)

    The comoving distance in Mpc at redshift z:

    >>> z = 0.5
    >>> dc = cosmo.comoving_distance(z)

    References
    ----------
    .. [1] Chevallier, M., & Polarski, D. (2001). Accelerating Universes with
           Scaling Dark Matter. International Journal of Modern Physics D,
           10(2), 213-223.
    .. [2] Linder, E. (2003). Exploring the Expansion History of the
           Universe. Phys. Rev. Lett., 90, 091301.
    """

    w0 = Parameter(doc="Dark energy equation of state at z=0.",
                   fvalidate="float")
    wa = Parameter(
        doc="Negative derivative of dark energy equation of state w.r.t. a.",
        fvalidate="float",
    )
    model = Parameter(doc="DDE model", fvalidate="float")

    def __init__(
        self,
        H0,
        Om0,
        Ode0,
        w0=-1.0,
        wa=0.0,
        model=1,
        Tcmb0=0.0 * u.K,
        Neff=3.04,
        m_nu=0.0 * u.eV,
        Ob0=None,
        *,
        name=None,
        meta=None
    ):
        super().__init__(
            H0=H0,
            Om0=Om0,
            Ode0=Ode0,
            Tcmb0=Tcmb0,
            Neff=Neff,
            m_nu=m_nu,
            Ob0=Ob0,
            name=name,
            meta=meta,
        )
        self.w0 = w0
        self.wa = wa
        self.model = model

        # Please see :ref:`astropy-cosmology-fast-integrals` for discussion
        # about what is being done here.
        if self.Tcmb0.value == 0:
            self._inv_efunc_scalar = scalar_inv_efuncs.w0wacdm_inv_efunc_norel
            self._inv_efunc_scalar_args = (
                self.Om0,
                self.Ode0,
                self.Ok0,
                self.w0,
                self.wa,
            )
        elif not self._massivenu:
            self._inv_efunc_scalar = scalar_inv_efuncs.w0wacdm_inv_efunc_nomnu
            self._inv_efunc_scalar_args = (
                self.Om0,
                self.Ode0,
                self.Ok0,
                self.Ogamma0 + self._Onu0,
                self.w0,
                self.wa,
            )
        else:
            self._inv_efunc_scalar = scalar_inv_efuncs.w0wacdm_inv_efunc
            self._inv_efunc_scalar_args = (
                self.Om0,
                self.Ode0,
                self.Ok0,
                self.Ogamma0,
                self.neff_per_nu,
                self.nmasslessnu,
                self.nu_y_list,
                self.w0,
                self.wa,
            )

    def w(self, z):
        r"""Returns dark energy equation of state at redshift ``z``.

        Parameters
        ----------
        z : Quantity-like ['redshift'], array-like, or `~numbers.Number`
            Input redshift.

        Returns
        -------
        w : ndarray or float
            The dark energy equation of state
            Returns `float` if the input is scalar.

        Notes
        -----
        The dark energy equation of state is defined as
        :math:`w(z) = P(z)/\rho(z)`, where :math:`P(z)` is the pressure at
        redshift z and :math:`\rho(z)` is the density at redshift z, both in
        units where c=1. Here this is
        :math:`w(z) = w_0 + w_a (1 - a) = w_0 + w_a \frac{z}{1+z}`.
        """
        z = aszarr(z)

        if self.model == 1:
            res = self.w0 + self.wa * z / (z + 1.0)

        if self.model == 2:
            res = -1. + self.w0*np.sin(self.wa*z)/(1.+z**2)

        return res

    def de_density_scale(self, z):
        r"""Evaluates the redshift dependence of the dark energy density.

        Parameters
        ----------
        z : Quantity-like ['redshift'], array-like, or `~numbers.Number`
            Input redshift.

        Returns
        -------
        I : ndarray or float
            The scaling of the energy density of dark energy with redshift.
            Returns `float` if the input is scalar.

        Notes
        -----
        The scaling factor, I, is defined by :math:`\rho(z) = \rho_0 I`,
        and in this case is given by

        .. math::

           I = \left(1 + z\right)^{3 \left(1 + w_0 + w_a\right)}
                     \exp \left(-3 w_a \frac{z}{1+z}\right)
        """
        z = aszarr(z)
        zp1 = z + 1.0  # (converts z [unit] -> z [dimensionless])

        if self.model == 1:
            res = zp1 ** (3 * (1 + self._w0 + self._wa)) * \
                exp(-3 * self._wa * z / zp1)
            return res
        else:
            return self.de_density_scale_int(z)

    def de_density_scale_int(self, z):
        """
        Method to estimate DE density from integral

        Parameters
        ----------
        z : numpy array
            redshift values.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """

        a = aszarr(z)

        r = []

        for zz in a:
            rr = self.de_density_scale_z(zz)
            r.append(rr)

        return np.asarray(r)

    def de_density_scale_z(self, z):
        """
        Method to estimate de density scale using integ

        Parameters
        ----------
        z : float
            redshift value.

        Returns
        -------
        float
            DE density.

        """

        res = quad(self.integrand, 0, z)[0]

        return np.exp(3.*res)

    def integrand(self, x):
        """
        Integrand for DE

        Parameters
        ----------
        x : float
            var to integrate.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """

        return (1.+self.w(x))/(1.+x)


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
