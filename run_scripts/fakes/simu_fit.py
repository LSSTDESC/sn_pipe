from optparse import OptionParser
#from sn_tools.sn_io import make_dict_from_config, make_dict_from_optparse
#from sn_tools.sn_cadence_tools import GenerateFakeObservations
from sn_tools.sn_io import make_dict_from_config
from fake_utils import FakeObservations, add_option, config
from sn_tools.sn_io import check_get_dir
from sn_fit.mbcov import MbCov

import numpy.lib.recfunctions as rf
import sn_simu_input as simu_input
import sn_fit_input as fit_input
from sn_simu_wrapper.sn_wrapper_for_simu import SimuWrapper
from sn_fit.process_fit import Fitting
import time
from astropy.table import Table, vstack
import numpy as np
import multiprocessing
import pandas as pd
import os
import yaml
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


def plot(fig, ax, tab, covcc_col='Cov_colorcolor', z_col='z', multiDaymax=False, stat=None, sigmaC=0.04):
    """
    Function to plot covcc vs z. A line corresponding to sigmaC is also drawn.

    Parameters
    ---------------
    fig: matplotlib figure
    ax: matplotlib axis
    tab: astropy table
      data to process: columns covcc_col and z_col should be in this tab
    covcc_col: str, opt
        name of the column corresponding to the cov_colorcolor value (default: Cov_colorcolor)
    z_col: str, opt
       name of the column corresponding to the redshift value (default: z)
    sigmaC: float, opt
      sigma_color value to estimate zlimit from (default: 0.04)

    """

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

    # plt.show()


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
        if mean_zlim > 0.:
            zlimtxt = 'z$_{lim}$'
            txt = '{} = {} '.format(zlimtxt, mean_zlim)
            if std_zlim >= 0.:
                txt += '$\pm$ {}'.format(std_zlim)
            ax.text(0.3, 0.06, txt, fontsize=12, color='k')
    else:
        ax.fill_between(tab[z_col], np.sqrt(tab[covcc_col]), color='yellow')


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


def plot_SNR(fig, ax, sel, zlim):
    """
    Function to plot SNR vs z

    Parameters
    --------------
    fig: matplotlib figure
    ax: matplotlib axis
    sel: astropy table
       data to plot

    """

    bands = 'grizy'
    colors = dict(zip('ugrizy', ['b', 'c', 'g', 'y', 'r', 'm']))

    for i, b in enumerate(bands):
        ax.plot(sel['z'], sel['SNR_{}'.format(b)], color=colors[b])
        # print('SNR', SNR(sel, b, zlim))
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

    idx = tab[covcc_col] > 0.
    tab = tab[idx]
    interpv = interp1d(tab[z_col], np.sqrt(tab[covcc_col]),
                       bounds_error=False, fill_value=0.)

    zmin = np.min(tab[z_col])
    zmax = np.max(tab[z_col])

    zvals = np.arange(zmin, zmax, 0.005)

    colors = interpv(zvals)

    ii = np.argmin(np.abs(colors-sigmaC))
    # print(colors)

    return np.round(zvals[ii], 3)


class GenSimFit:
    """
    class to generate observations, simulate and fit fake observations

    Parameters
    ---------------
    config_fake: dict
      configuration dict to generate fakes
    config_simu: dict
      configuration dict for simulation
    config_fit: dict
      configuration dict for fit
    outputDir: str
      main output directory
    zlim_calc: bool
      to estimate zlim from fitted values
    tagprod: int, opt
      tag for production (default: -1: no tag)
    """

    def __init__(self, config_fake, config_simu, config_fit, outputDir, zlim_calc=False, nsn_calc=False, survey_area=0., display=False, tagprod=0):

        # grab config
        self.config_simu = config_simu
        self.config_fit = config_fit
        self.tagprod = tagprod
        self.outputDir = outputDir
        self.zlim_calc = zlim_calc
        self.nsn_calc = nsn_calc
        self.survey_area = survey_area
        self.simu_name = config_simu['Simulator']['name'].split('.')[-1]
        self.fitter_name = config_fit['Fitter']['name'].split('.')[-1]
        self.display = display

        # prepare for output
        self.save_simu = config_simu['OutputSimu']['save']
        self.save_fit = config_fit['OutputFit']['save']
        self.prepare_for_save()

        # simulator instance
        self.simu = SimuWrapper(self.config_simu)

        if self.save_simu:
            # save simu config file
            self.dump_dict_to_yaml(
                self.config_simu['OutputSimu']['directory'], self.config_simu['ProductionIDSimu'], self.config_simu)
        if self.save_fit:
            # save fit configuration file
            self.dump_dict_to_yaml(
                self.config_fit['OutputFit']['directory'], self.config_fit['ProductionIDFit'], self.config_fit)
            if not self.save_simu:
                # save simu config file
                self.dump_dict_to_yaml(
                    self.config_fit['OutputFit']['directory'], self.config_simu['ProductionIDSimu'], self.config_simu)

        # fitter instance
        covmb = None
        mbCalc = config_fit['mbcov']['estimate']

        if mbCalc:
            # for this we need to have the SALT2 dir and files
            # if it does not exist get it from the web

            salt2Dir = config_fit['mbcov']['directory']
            webPath = config_fit['WebPathFit']
            check_get_dir(webPath, salt2Dir, salt2Dir)
            covmb = MbCov(salt2Dir, paramNames=dict(
                zip(['x0', 'x1', 'color'], ['x0', 'x1', 'c'])))

        self.fit = Fitting(self.config_fit, covmb)

        # nproc for multiproc
        self.nproc = config_fit['MultiprocessingFit']['nproc']

        self.config_fake = config_fake

        # this is to estimate nsn
        self.nsn_eval = None
        if self.nsn_calc:
            from sn_tools.sn_rate import NSN
            self.nsn_eval = NSN()

        """
        print(config_simu)
        print(config_fit)
        """

    def __call__(self, params):
        """
        call function.
        The output is a list of fitted SN that can be copied on disk.

        Parameters
        ---------------
        params: dict
          configuration dict for fake obs

        Returns
        -----------
        astropy table with a list of fitted SN

        """
        self.simu.metric.season = -1
        restot = Table()
        for i, row in params.iterrows():
            config_fake = self.getconfig(row)
            restot = vstack([restot, self.runSequence(
                config_fake, row['season_length'])])

        if self.save_fit:
            outName = '{}/{}.hdf5'.format(
                self.config_fit['OutputFit']['directory'], self.config_fit['ProductionIDFit'])
            restot.write(outName, 'fitlc', compression=True)

        r = []
        # print('res', len(restot), self.zlim_calc)
        if self.zlim_calc and len(restot) >= 1:
            restot.sort('z')
            if 'sigma_mu' in restot.columns:
                print(restot[['z', 'sigma_mu']], np.sqrt(
                    restot['Cov_mbmb']), np.sqrt(restot['Cov_colorcolor']))
            for tagprod in np.unique(restot['tagprod']):
                idx = restot['tagprod'] == tagprod
                # print('hhh', np.sqrt(restot[idx]['Cov_colorcolor']),
                #      restot[idx]['z', 'daymax'], restot.columns)

                vv = restot[idx]
                zlimit_val = zlimit(vv)

                #cad_z = int(np.mean(vv['cadence_z']))
                #season = np.mean(vv['season'])

                print('zlimit', tagprod, self.simu_name,
                      self.fitter_name, zlimit_val)
                numSN = -1
                if self.nsn_calc:
                    season_length = np.mean(vv['season_length'])
                    numSN = self.nsn_eval(
                        0.01, zlimit_val, 0.01, season_length, self.survey_area)
                r.append(
                    (tagprod, zlimit_val, numSN))
                if self.display:
                    import matplotlib.pyplot as plt

                    fig, ax = plt.subplots()
                    plot(fig, ax, vv)
                    print(restot.dtype.names)
                    if 'SNR_i' in restot.dtype.names:
                        fig, ax = plt.subplots()
                        plot_SNR(fig, ax, vv, zlimit_val)
                    plt.show()

        zlimdf = pd.DataFrame(r, columns=['tagprod', 'zlim', 'nsn_exp'])

        return restot, zlimdf

    def runSequence(self, config_fake, season_length):
        """
        Method to perform the complete sequence: observation generation, simulation and fit

        Parameters
        ---------------
        config_fake: dict
          configuration dict for fake obs.

        Returns
        -----------
        fitted LC (astropy table)
        """
        # generate fake obs
        fakeData = FakeObservations(config_fake).obs
        print('fake Data', len(fakeData))

        # simulate LCs
        list_lc = self.simu.run(fakeData)

        print('simu LC', len(list_lc))
        # add the tag prod to lc metadata

        for lc in list_lc:
            lc.meta['tagprod'] = config_fake['tagprod']
            lc.meta['season_length'] = season_length

        # fit LCs
        res = self.fit_loop(list_lc)

        print(res[['z', 'Cov_colorcolor']])
        print('fit lc', len(res))
        return res

    def fit_loop(self, list_lc):
        """
        Method to fit a list of light curves.
        This method uses multiprocessing.

        Parameters
        ---------------
        list_lc: list
          list of light curves (astropy tables)

        Returns
        ----------
        astropy table: each row corresponds to a fitted LC (hence a supernova)
        """

        # multiprocessing parameters
        nz = len(list_lc)
        t = np.linspace(0, nz, self.nproc+1, dtype='int')
        # print('multi', nz, t)
        result_queue = multiprocessing.Queue()

        procs = [multiprocessing.Process(name='Subprocess-'+str(j), target=self.fit_lc,
                                         args=(list_lc[t[j]:t[j+1]], j, result_queue))
                 for j in range(self.nproc)]

        for p in procs:
            p.start()

        resultdict = {}
        # get the results in a dict

        for i in range(self.nproc):
            resultdict.update(result_queue.get())

        for p in multiprocessing.active_children():
            p.join()

        restot = Table()

        # gather the results
        for key, vals in resultdict.items():
            restot = vstack([restot, vals])
        return restot

    def fit_lc(self, list_lc, j=0, output_q=None):
        """
        Method to fit a list of light curves.
        This method is used by the multiprocessing (fit_loop).

        Parameters
        ---------------
        list_lc: list
          list of light curves (astropy tables)

        Returns
        ----------
        astropy table: each row corresponds to a fitted LC (hence a supernova)
        """

        tabfit = Table()
        for lc in list_lc:
            resfit = self.fit(lc)
            tabfit = vstack([tabfit, resfit])

        if output_q is not None:
            return output_q.put({j: tabfit})
        else:
            return tabfit

    def getconfig(self, row):
        """
        Method to make a config dict for fake obs.

        Parameters
        ---------------
        row: df row
          set of parameters to build the dict from

        Returns
        -----------
        the configuration dict.

        """
        config = self.config_fake.copy()

        for band in 'grizy':
            config['cadence'][band] = row['cadence_{}'.format(band)]
            config['m5'][band] = row['m5_{}'.format(band)]
            config['Nvisits'][band] = row['N{}'.format(band)]

        config['tagprod'] = row['tagprod']
        config['seasons'] = str(row['season'])
        return config

    def prepare_for_save(self):
        """
        Method defining a set of infos to save data: output names for dir and files

        """
        errormod = self.config_simu['Simulator']['errorModel']
        cutoff = '{}_{}'.format(
            self.config_simu['SN']['blueCutoffg'], self.config_simu['SN']['redCutoffg'])
        ebv = self.config_simu['SN']['ebvofMW']

        if errormod:
            cutoff = 'error_model'

        outDir_simu = '{}/Output_Simu_{}_ebvofMW_{}'.format(self.outputDir,
                                                            cutoff, ebv)

        self.config_simu['OutputSimu']['directory'] = outDir_simu

        snrmin = self.config_fit['LCSelection']['snrmin']
        outDir_fit = '{}/Output_Fit_{}_ebvofMW_{}_snrmin_{}'.format(self.outputDir,
                                                                    cutoff, ebv, int(snrmin))
        if errormod:
            errmodrel = self.config_fit['LCSelection']['errmodrel']
            outDir_fit += '_errmodrel_{}'.format(np.round(errmodrel, 2))

        self.config_fit['OutputFit']['directory'] = outDir_fit

        sn_model = self.config_simu['Simulator']['model']
        simu = self.config_simu['Simulator']['name'].split('.')[-1]
        fitter = self.config_fit['Fitter']['name'].split('.')[-1].split('_')
        fitter = '_'.join([vv for vv in fitter[1:]])
        sn_type = self.config_simu['SN']['type']
        x1 = self.config_simu['SN']['x1']['min']
        color = self.config_simu['SN']['color']['min']

        fname = '{}_{}'.format(sn_type, sn_model)
        if 'salt2' in sn_model:
            fname = '{}_{}'.format(x1, color)
        tag = '{}_Fake_{}_{}_ebvofMW_{}'.format(
            simu, fname, cutoff, ebv)
        if self.tagprod >= 0:
            tag += '_{}'.format(self.tagprod)

        self.config_simu['ProductionIDSimu'] = tag
        self.config_fit['ProductionIDFit'] = 'Fit_{}_{}'.format(tag, fitter)

        # create dirs if necessary
        if self.save_simu or self.save_fit:
            if not os.path.exists(self.outputDir):
                os.mkdir(self.outputDir)
        if self.save_simu:
            if not os.path.exists(outDir_simu):
                os.mkdir(outDir_simu)

        if self.save_fit:
            if not os.path.exists(outDir_fit):
                os.mkdir(outDir_fit)

    def dump_dict_to_yaml(self, theDir, theName, theDict):
        """
        Method to dump a dict to a yaml file

        Parameters
        ---------------
        theDir: str
           output directory
        theName: str
           output file name
        theDict: dict
          dict to dump
        """
        outputyaml = '{}/{}.yaml'.format(theDir, theName)
        with open(outputyaml, 'w') as file:
            documents = yaml.dump(theDict, file)


# this is to load option for fake cadence
path = 'input/Fake_cadence'
confDict_fake = make_dict_from_config(path, 'config_cadence.txt')
# get all possible simulation parameters and put in a dict
path = simu_input.__path__
confDict_simu = make_dict_from_config(path[0], 'config_simulation.txt')
# get all possible simulation parameters and put in a dict
path = fit_input.__path__
confDict_fit = make_dict_from_config(path[0], 'config_fit.txt')


parser = OptionParser()

# add option for Fake data here
add_option(parser, confDict_fake)
add_option(parser, confDict_simu)
add_option(parser, confDict_fit)

parser.add_option(
    '--outputDir', help='main output directory [%default]', default='/sps/lsst/users/gris/config_zlim', type=str)
parser.add_option(
    '--config', help='config file of parameters [%default]', default='config_z_0.8.csv', type=str)
parser.add_option('--zlim_calc', type=int, default=0,
                  help='to estimate zlim or not [%default]')
parser.add_option(
    '--nsn_calc', help='to estimate nsn or not [%default]', default=0, type=int)
parser.add_option(
    '--survey_area', help='area for nsn estimation in deg2 [%default]', default=0.21, type=float)
parser.add_option('--tagprod', type=int, default=-1,
                  help='tag for outputfile [%default]')
parser.add_option('--plot', type=int, default=0,
                  help='to display some results [%default]')


opts, args = parser.parse_args()

time_ref = time.time()
# make the config files here
config_fake = config(confDict_fake, opts)
config_simu = config(confDict_simu, opts)
config_fit = config(confDict_fit, opts)
zlim_calc = opts.zlim_calc
nsn_calc = opts.nsn_calc
survey_area = opts.survey_area
tagprod = opts.tagprod
display = opts.plot

# instance process here
process = GenSimFit(config_fake, config_simu, config_fit,
                    opts.outputDir, tagprod=tagprod,
                    zlim_calc=zlim_calc, nsn_calc=nsn_calc,
                    survey_area=survey_area, display=display)

# run
params = pd.read_csv(opts.config, comment='#')
res, zlimdf = process(params)

params = params.merge(zlimdf, left_on=['tagprod'], right_on=[
    'tagprod'])

params.to_csv(opts.config, index=False)
print('params', params)
# print(res.columns)
"""

fakeData = FakeObservations(newDict_fake).obs
print(fakeData)

# instance for simulation
simu = SimuWrapper(newDict_simu)

# simulate here

list_lc = simu.run(fakeData)

# fit instance here
fit = Fitting(newDict_fit)

for lc in list_lc:
    print(lc.meta)
    resfit = fit(lc)
    print(resfit)
"""
print('Elapsed time', time.time()-time_ref)


"""
for key, vals in confDict.items():
    newval = eval('opts.{}'.format(key))
    newDict[key]=(vals[0],newval)

# new dict with configuration params
yaml_params = make_dict_from_optparse(newDict)
"""
