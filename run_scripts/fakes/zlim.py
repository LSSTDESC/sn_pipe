import numpy as np
from sn_tools.sn_io import loopStack
from scipy.interpolate import interp1d
import pandas as pd
from optparse import OptionParser
import os
import multiprocessing
from astropy.table import Table, vstack
import matplotlib.pyplot as plt


def zlimit(tab, covcc_col='Cov_colorcolor', z_col='z', sigmaC=0.04):
    """
    Function to estimate zlim for sigmaC value

    Parameters
    ---------------
    tab: astropy table
      data to process: columns covcc_col and z_col should be in this tab
    covcc_col: str, opt
        name of the column corresponding to the cov_colorcolor value (default: Cov_colorcolor)
    z_col: str, opt
       name of the column corresponding to the redshift value (default: z)
    sigmaC: float, opt
      sigma_color value to estimate zlimit from (default: 0.04)

    Returns
    ----------
    The zlimit value corresponding to sigmaC

    """
    interp = interp1d(np.sqrt(tab[covcc_col]),
                      tab[z_col], bounds_error=False, fill_value=0.)

    interpv = interp1d(tab[z_col], np.sqrt(tab[covcc_col]),
                       bounds_error=False, fill_value=0.)

    zvals = np.arange(0.2, 1.0, 0.005)

    colors = interpv(zvals)
    ii = np.argmin(np.abs(colors-sigmaC))
    print(colors)
    return np.round(zvals[ii], 3)


def SNR(tab, band, z):
    """
    Function to estimate the SNR for band and z

    Parameters
    ---------------
    tab: astropy table
      with SNR_b values
    band: str
      the band to consider
    z: float
      the redshift value to estimate SNR

    Returns
    ----------
    SNR(z) for the band

    """

    interp = interp1d(tab['z'], tab['SNR_{}'.format(band)],
                      bounds_error=False, fill_value=0.)

    return np.round(interp(z), 1)


def plot(tab, covcc_col='Cov_colorcolor', z_col='z', multiDaymax=False, stat=None, sigmaC=0.04):
    """
    Function to plot covcc vs z. A line corresponding to sigmaC is also drawn.

    Parameters
    ---------------
    tab: astropy table
      data to process: columns covcc_col and z_col should be in this tab
    covcc_col: str, opt
        name of the column corresponding to the cov_colorcolor value (default: Cov_colorcolor)
    z_col: str, opt
       name of the column corresponding to the redshift value (default: z)
    sigmaC: float, opt
      sigma_color value to estimate zlimit from (default: 0.04)

    """

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    tab.sort(z_col)

    xlims = [0.1, 0.91]
    ylims = [0.01, 0.08]

    mean_zlim = -1.
    std_zlim = -1.
    daymax_mean = -1.

    if stat is not None:
        idx = stat['zlim'] > 0.
        selstat = np.copy(stat[idx])
        mean_zlim = np.round(np.mean(selstat['zlim']), 2)
        std_zlim = np.round(np.std(selstat['zlim']), 2)
        idx = (np.abs(selstat['zlim'] - mean_zlim)).argmin()
        daymax_mean = selstat[idx]['daymax']

        selstat.sort(order=['zlim', 'daymax'])

    if multiDaymax:
        tab_bet = Table()
        idx = np.abs(tab['daymax']-selstat[0]['daymax']) < 1.e-5
        # plot_indiv(ax,tab[idx])
        tab_bet = vstack([tab_bet, tab[idx]])
        idx = np.abs(tab['daymax']-selstat[-1]['daymax']) < 1.e-5
        sol = tab[idx]
        sol.sort('z', reverse=True)
        tab_bet = vstack([tab_bet, sol])
        plot_indiv(ax, tab_bet, fill=True)
        idx = np.abs(tab['daymax'] - daymax_mean) < 0.01
        plot_indiv(ax, tab[idx], mean_zlim=mean_zlim, std_zlim=std_zlim)
        """
        for daymax in np.unique(tab['daymax']):
            ido = np.abs(tab['daymax']-daymax)<1.e-5
            plot_indiv(ax,tab[ido])
        """
    else:
        plot_indiv(ax, tab, mean_zlim=mean_zlim)
    ax.set_ylim(ylims)
    ax.set_xlim(xlims)
    ax.plot(ax.get_xlim(), [sigmaC]*2, color='r')
    ax.grid()
    ax.set_xlabel('z', fontsize=12)
    ax.set_ylabel('$\sigma_{C}$', fontsize=12)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)


def plot_indiv(ax, tab, covcc_col='Cov_colorcolor', z_col='z', fill=False, mean_zlim=-1, std_zlim=-1.):
    """
    Function to plot on ax

    Parameters
    --------------
    ax: matplotlib axis
      axes where to plot
    tab: pandas df
      data to plot
    covcc_col: str, opt
        name of the column corresponding to the cov_colorcolor value (default: Cov_colorcolor)
    z_col: str, opt
       name of the column corresponding to the redshift value (default: z)

    """
    if not fill:
        ax.plot(tab[z_col], np.sqrt(tab[covcc_col]), color='k')
        print('eeee', mean_zlim)
        if mean_zlim > 0.:
            zlimtxt = 'z$_{lim}$'
            txt = '{} = {} '.format(zlimtxt, mean_zlim)
            if std_zlim >= 0.:
                txt += '$\pm$ {}'.format(std_zlim)
            ax.text(0.3, 0.06, txt, fontsize=12, color='k')
    else:
        ax.fill_between(tab[z_col], np.sqrt(tab[covcc_col]), color='yellow')


def plot_SNR(sel, zlim):
    """
    Function to plot SNR vs z

    Parameters
    --------------
    sel: astropy table
       data to plot

    """

    fig, ax = plt.subplots()

    bands = 'grizy'
    colors = dict(zip('ugrizy', ['b', 'c', 'g', 'y', 'r', 'm']))

    for i, b in enumerate(bands):
        ax.plot(sel['z'], sel['SNR_{}'.format(b)], color=colors[b])
        print('SNR', SNR(sel, b, zlim))
        SNR_b = SNR(sel, b, zlim)
        ax.text(zlim+0.05, 60.-i*5, 'SNR - {} band = {}'.format(b,
                                                                SNR_b), color=colors[b], fontsize=12)

    ax.set_ylim(0.0,)
    if zlim > 0:
        ax.plot([zlim]*2, ax.get_ylim(), color='k', ls='dashed')
    ax.grid()
    ax.set_xlabel('z', fontsize=12)
    ax.set_ylabel('SNR', fontsize=12)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)


def ana_zlim(tagprod, simulator='sn_fast', fitter='sn_fast', x1=-2.0, color=0.2, ebvofMW=0.0, snrmin=1., error_model=1, errmodrel=-1., bluecutoff=380., redcutoff=800., multiDaymax=0, sigmaC=0.04, plot_this=0):
    """
    Function to analyze the output of the fit.
    The idea is to estimate the redshift limit

    Parameters
    ---------------
    tagprod: str
      tag for the production
    simulator: str, opt
      name of the simulator used (default: sn_fast)
    fitter: str, opt
      fitter used (default: sn_fast)
    x1: float, opt
      SN x1 (default: -2.0)
    color: float, opt
      SN color (default: 0.2)
    ebvofMW: float, opt
      E(B-V)  (default: 0.0)
    snrmin: float, opt
      min SNR for LC points to be considered for the fit (default: 1.)
    error_model: int, opt
      error_model included (1) or not (0) (default: 1)
    errmodrel: float, opt
      max flux relative error model (default:-1)
    bluecutoff: float, opt
      bluecutoff if error_model=0 (default: 380.)
    redcutoff: float, opt
     red cutoff if error_model=0 (default: 800.)
    sigmaC: float, opt
      sigmaC value to estimate zlim (default: 0.04)
    plot_this: int, opt
      to plot the results (default: 0)

    Returns
    -----------
    The redshift limit.

    """

    cutoff = 'error_model'
    if error_model < 1:
        cutoff = '{}_{}'.format(bluecutoff, redcutoff)

    dirFile = 'Output_Fit_{}_ebvofMW_{}_snrmin_{}_errmodrel_{}'.format(
        cutoff, ebvofMW, int(snrmin), np.round(errmodrel, 2))

    fName = 'Fit_{}_Fake_{}_{}_{}_ebvofMW_{}_{}_{}.hdf5'.format(
        simulator, x1, color, cutoff, ebvofMW, tagprod, fitter)

    fullName = '{}/{}'.format(dirFile, fName)

    # if the file does not exist: no fit performed -> zlim=-1.
    if not os.path.isfile(fullName):
        print('warning file missing', fullName)
        return -1.0, -1.0

    tab = loopStack([fullName], 'astropyTable')

    idx = tab['z'] >= 0.05
    sel = tab[idx]

    # print(sel.columns,zlim(sel))

    rz = []

    ro = []
    if multiDaymax:
        for daymax in np.unique(sel['daymax']):
            ido = np.abs(sel['daymax']-daymax) < 1.e-5
            if len(sel[ido]) >= 2:
                zzlim = zlimit(sel[ido], sigmaC=sigmaC)
                rz.append(zzlim)
                ro.append((daymax, zzlim))
        if plot_this:
            plot(sel, multiDaymax=multiDaymax,
                 stat=np.rec.fromrecords(ro, names=['daymax', 'zlim']))
            plt.show()
    else:
        zzlim = zlimit(sel, sigmaC=sigmaC)
        rz.append(zzlim)
        ro.append((np.mean(sel['daymax']), zzlim))
        if plot_this:
            plot(sel, stat=np.rec.fromrecords(ro, names=['daymax', 'zlim']))
            plot_SNR(sel, zzlim)
            plt.show()
    return np.mean(rz), np.std(rz)


def simufit_cmd(tagprod, x1=-2.0, color=0.2,
                ebvofMW=0.0,
                simulator='sn_fast',
                fitter='sn_fast',  snrmin=1.0,
                error_model=1,
                errmodrel=-1.,
                bluecutoff=380.,
                redcutoff=800.,
                Nvisits=dict(zip('grizy', [10, 20, 20, 26, 20])),
                m5=dict(zip('grizy', [24.51, 24.06, 23.62, 23.0, 22.17])),
                cadence=3, bands='grizy', multiDaymax=0, m5File=None, healpixID=-1, season=1):
    """
    Function to simulate and fit SN light curves

    Parameters
    ---------------
    tagprod: str
      tag for the production
     x1: float, opt
      SN x1 (default: -2.0)
    color: float, opt
      SN color (default: 0.2)
    ebvofMW: float, opt
      E(B-V)  (default: 0.0)
      simulator: str, opt
      name of the simulator used (default: sn_fast)
    fitter: str, opt
      fitter used (default: sn_fast)
    snrmin: float, opt
      min SNR for LC points to be considered for the fit (default: 1.)
    error_model: int, opt
      error_model included (1) or not (0) (default: 1)
    errmodrel: float, opt
       max error model relative flux (default: -1)
    bluecutoff: float, opt
      bluecutoff if error_model=0 (default: 380.)
    redcutoff: float, opt
     red cutoff if error_model=0 (default: 800.)
    Nvisits: dict, opt
      number of visits per band (default: dict(zip('grizy',[10,20,20,26,20])))
    m5: dict, opt
       m5 per band (single visit) (default: dict(zip('grizy',[24.51,24.06,23.62,23.0,22.17])))
    cadence: float, opt
      cadence of observations (default: 3)
    bands: str, opt
      bands to consider (default: grizy)
    multiDaymax: bool, opt
      to simulate/fit multi daymax SN
    m5File: str, opt
      m5 file name
    healpixID: int,opt
      healpixID to get m5 values (default: -1)
    """
    cmd = 'python run_scripts/fakes/loop_full_fast.py'
    cmd += ' --x1 {}'.format(x1)
    cmd += ' --color {}'.format(color)
    cmd += ' --ebv {}'.format(ebvofMW)
    cmd += ' --simus {}'.format(simulator)
    cmd += ' --{}_simu_fitter {}'.format(simulator, fitter)
    cmd += ' --error_model {}'.format(error_model)
    cmd += ' --bluecutoff {}'.format(bluecutoff)
    cmd += ' --redcutoff {}'.format(redcutoff)
    cmd += ' --tagprod {}'.format(tagprod)
    cmd += ' --snrmin {}'.format(snrmin)
    cmd += ' --m5File {}'.format(m5File)
    cmd += ' --multiDaymax {}'.format(multiDaymax)
    cmd += ' --healpixID {}'.format(healpixID)
    cmd += ' --seasons {}'.format(season)
    cmd += ' --errmodrel {}'.format(errmodrel)

    for b in bands:
        cmd += ' --Nvisits_{} {}'.format(b, Nvisits[b])
        cmd += ' --m5_{} {}'.format(b, m5[b])
        cmd += ' --cadence_{} {}'.format(b, cadence[b])

    return cmd


def multiproc(conf, bands, multiDaymax=False, m5File='NoData', healpixID_m5=False, sigmaC=0.04, action='all', plot=0, nproc=8):
    """
    Function to process data using multiprocessing

    Parameters
    --------------
    conf: pandas df
      configuration file
    bands: str
      list of the filters to consider
    multiDaymax: bool, opt
     to simulate/fit multi daymax SN (default: False)
    m5File: str,opt
      m5 file to get values from (default: 'NoData')
    healpixID_m5: bool, opt
      healpixID to get m5 values (default: False)
    nproc: int, opt
      number of procs for multiprocessing (default: 8)
    sigmaC: float, opt
      error on the color to estimate zlim (default: 0.04)
    action: str, opt
      what you have to do (all/simufit/zlim) (default: all)
    plot: bool, opt
      to plots the results (default: 0)

    Returns
    -----------
    original DataFrame plus zlim appended
    """

    nz = len(conf)
    t = np.linspace(0, nz, nproc+1, dtype='int')
    result_queue = multiprocessing.Queue()

    procs = [multiprocessing.Process(name='Subprocess-'+str(j), target=process,
                                     args=(conf[t[j]:t[j+1]], bands, multiDaymax, m5File, healpixID_m5, sigmaC, action, plot, j, result_queue))
             for j in range(nproc)]

    for p in procs:
        p.start()

    resultdict = {}
    # get the results in a dict

    for i in range(nproc):
        resultdict.update(result_queue.get())

    for p in multiprocessing.active_children():
        p.join()

    restot = pd.DataFrame()

    # gather the results
    for key, vals in resultdict.items():
        restot = pd.concat((restot, vals), sort=False)
        """
            if restot is None:
                restot = vals
            else:
                restot = np.concatenate((restot, vals))
        """
    return restot


def process(config, bands, multiDaymax, m5File, healpixID_m5, sigmaC, action, plot=0, j=0, output_q=None):
    """
    Function to process data

    Parameters
    --------------
    conf: pandas df
      configuration file
    bands: str
      list of the filters to consider
    multiDaymax: bool
      to have multiple daymax for simulation/fit
    m5File: str
      m5 file (from simulation usually)  
    healpixID_m5: bool
      healpixID to get m5 values 
    sigmaC: float
      color error requested to estimate zlim
    action: str
      what to do
    plot: int, opt
      to display the results (default: 0)
    j: int, opt
      multiprocessing number(default: 0)
    output_q: multiprocessing queue (default: None)


    Returns
    -----------
    original DataFrame plus zlim appended

    """

    rz = []
    rstd = []
    confres = pd.DataFrame(config)
    for index, conf in config.iterrows():
        # first step: simulate and fit
        tagprod = '{}_{}'.format(conf['tagprod'], conf['season'])
        if action == 'all' or action == 'simufit':
            simufit(conf, bands, multiDaymax, m5File, healpixID_m5, tagprod)
        if action == 'all' or action == 'zlim':
            rzo, rstdo = zlim_estimate(
                conf, tagprod, multiDaymax, sigmaC, plot)
            rz += rzo
            rstd += rstdo

    if action == 'all' or action == 'zlim':
        confres['zlim_mean'] = rz
        confres['zlim_std'] = rstd

    if output_q is not None:
        return output_q.put({j: confres})
    else:
        return confres


def simufit(conf, bands, multiDaymax, m5File, healpixID_m5, tagprod):

    Nvisits = {}
    m5 = {}
    cadence = {}

    for b in bands:
        Nvisits[b] = conf['N{}'.format(b)]
        m5[b] = conf['m5_{}'.format(b)]
        cadence[b] = conf['cadence_{}'.format(b)]

    healpixID = -1
    if healpixID_m5:
        healpixID = conf['tagprod']
    cmd = simufit_cmd(tagprod, conf['x1'], conf['color'], conf['ebvofMW'],
                      conf['simulator'], conf['fitter'], conf['snrmin'],
                      conf['error_model'], conf['errmodrel'], conf['bluecutoff'], conf['redcutoff'],
                      Nvisits, m5, cadence, bands, multiDaymax, m5File, healpixID, conf['season'])
    print(cmd)
    os.system(cmd)


def zlim_estimate(conf, tagprod, multiDaymax, sigmaC, plot):

    rz = []
    rstd = []
    print('alllll', sigmaC)
    zlim_mean, zlim_std = ana_zlim(tagprod,
                                   simulator=conf['simulator'],
                                   fitter=conf['fitter'],
                                   x1=conf['x1'],
                                   color=conf['color'],
                                   ebvofMW=conf['ebvofMW'],
                                   snrmin=conf['snrmin'],
                                   error_model=conf['error_model'],
                                   errmodrel=conf['errmodrel'],
                                   bluecutoff=conf['bluecutoff'],
                                   redcutoff=conf['redcutoff'],
                                   multiDaymax=multiDaymax,
                                   sigmaC=sigmaC,
                                   plot_this=plot)
    print(zlim_mean)
    rz.append(np.round(zlim_mean, 2))
    rstd.append(np.round(zlim_std, 2))

    return rz, rstd


parser = OptionParser()

parser.add_option('--config', type='str', default='run_scripts/fakes/config.csv',
                  help='config file to use[%default]')
parser.add_option('--outName', type='str', default='config_out.csv',
                  help='output file name [%default]')
parser.add_option('--m5File', type=str, default='NoData',
                  help='m5 file [%default]')
parser.add_option('--multiDaymax', type=int, default=0,
                  help='to have multi T0 simulated/fitted [%default]')
parser.add_option('--healpixID_m5', type=int, default=0,
                  help='to get m5 from a healpixel [%default]')
parser.add_option('--sigmaC', type=float, default=0.04,
                  help='sigma color to estimate zlim [%default]')
parser.add_option('--action', type=str, default='all',
                  help='what to do: all, simu_fit, zlim [%default]')
parser.add_option('--plot', type=int, default=0,
                  help='to make plots of the results [%default]')


opts, args = parser.parse_args()

config = pd.read_csv(opts.config, comment='#')

bands = 'grizy'
res = multiproc(config, bands, opts.multiDaymax, opts.m5File,
                opts.healpixID_m5, opts.sigmaC, opts.action, opts.plot, nproc=1)
res.to_csv(opts.outName, index=False)
