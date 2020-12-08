import numpy as np
import os
from optparse import OptionParser
from sn_tools.sn_io import make_dict_from_config,make_dict_from_optparse
import yaml
from astropy.table import Table,vstack
from sn_fit.mbcov import MbCov
from sn_tools.sn_io import loopStack,check_get_dir
import glob

class SimFit:
    """
    class to simulate and fit type Ia supernovae

    Parameters
    ---------------
    x1: float, opt
      SN strech(default: -2.0)
    color: float
      SN color(default: 0.2)
    Nvisits: dict, opt
      number of visits for each band(default: 'grizy', [10, 20, 20, 26, 20])
    m5: dict, opt
      fiveSigmaDepth single visit for each band(default: 'grizy', [24.51, 24.06, 23.62, 23.0, 22.17])
    cadence: dict, opt
      cadence of observation(per band)(default='grizy', [3., 3., 3., 3.3.])
    error_model: int, opt
      to use error model or not (default: 1)
    bluecutoff: float, opt
      blue cutoff to apply(if error_model=0)(default: 380.)
    redcutoff: float, opt
      red cutoff to apply(if error_model=0)(default: 800.)
    simulator: str, opt
      simulator to use(default: sn_fast)
    fitters: list(str), opt
      list of fitters to use(defaulf: ['sn_fast','sn_cosmo'])          
    outDir_simu: str, opt
       location dir of simus (default: zlim_simu)
    outDir_fit: str, opt
       location dir of fitted values (default: zlim_fit)
    outDir_obs: str, opt
       location dir of obs (default: zlim_obs)
    snrmin: float, opt
       SNR min for LC point fits (default : 1.0)
    sigma_mu: bool, opt
       to estimate sigma_mu (default: False)
    tag: str, opt
      tag for the production(default: test)

    """

    def __init__(self, x1=-2.0, color=0.2,
                 error_model=1,
                 bluecutoff=380.,
                 redcutoff=800.,
                 simulator='sn_fast',
                 fitters=['sn_fast','sn_cosmo'],
                 outDir_simu = 'zlim_simu',
                 outDir_fit = 'zlim_fit',
                 outDir_obs = 'zlim_obs',
                 snrmin=1.0,
                 sigma_mu=False,
                 tag='test'):

        self.x1 = x1
        self.color = color
        self.tag = tag
        self.sigma_mu = sigma_mu

        # simulation parameters
        self.error_model = error_model
        self.bluecutoff = bluecutoff
        self.redcutoff = redcutoff
        if self.error_model:
            self.cutoff = 'error_model'
        else:
            self.cutoff = '{}_{}'.format(self.bluecutoff, self.redcutoff)
        self.ebvofMW = 0.
        self.simulator = simulator
        self.zmin = 0.01
        self.zmax = 1.0
        self.zstep = 0.05

        # fit parameters
        self.snrmin = snrmin
        self.nbef = 4
        self.naft = 10
        self.nbands = 0
        self.fitters = fitters

        # define some directory for the production of LC+Simu

        self.outDir_simu = outDir_simu
        self.outDir_fit = outDir_fit
        self.outDir_obs = outDir_obs

        # check whether output dir exist - create if necessary
        self.check_create(self.outDir_simu)
        self.check_create(self.outDir_fit)
        self.check_create(self.outDir_obs)

        self.fake_name = 'Fakes_{}'.format(self.tag)
        self.fake_config = '{}/fake_config_{}.yaml'.format(
            self.outDir_obs, tag)
        self.fake_data = '{}/{}'.format(self.outDir_obs, self.fake_name)

        # prepare for sigma_mu calc
        self.covmb = None
        if self.sigma_mu:
            salt2Dir = 'SALT2_Files'
            webPath = 'https://me.lsst.eu/gris/DESC_SN_pipeline'
            check_get_dir(webPath,salt2Dir, salt2Dir)

            self.covmb = MbCov(salt2Dir, paramNames=dict(
                zip(['x0', 'x1', 'color'], ['x0', 'x1', 'c'])))
        
    def check_create(self, dirname):
        """
        Method to create a dir if it does not exist

        Parameters
        ---------------
        dirname: str
          directory name

        """
        if not os.path.exists(dirname):
            os.system('mkdir -p {}'.format(dirname))

    def process(self, Nvisits=dict(zip('grizy', [10, 20, 20, 26, 20])),
                m5=dict(zip('grizy', [24.51, 24.06, 23.62, 23.0, 22.17])),
                cadence=dict(zip('grizy', [3., 3., 3., 3., 3.])),
                zmin=0.01, zmax=0.9, zstep=0.05):
        """
        Method to process data in three steps:
        - generate fake obs
        - generate LCs from fake obs
        - fit LC generated from fake obs

        """

        # generate observations
        self.generate_obs(Nvisits, m5, cadence)

        # simulation(fast) of LCs
        self.simulate_lc(zmin, zmax, zstep)

        # fit (fast) these LCs
        for fitter in self.fitters:
            self.fit_lc(fitter)
            if self.sigma_mu:
                self.calc_sigma_mu(fitter)

    def generate_obs(self, Nvisits, m5, cadence):
        """
        Method to generate fake observations

        """

        bands = Nvisits.keys()
        # generate fake_config file
        cmd = 'python run_scripts/make_yaml/make_yaml_fakes.py'
        for b in bands:
            cmd += ' --cadence_{} {}'.format(b, cadence[b])
            cmd += ' --m5_{} {}'.format(b, m5[b])
            cmd += ' --Nvisits_{} {}'.format(b, int(np.round(Nvisits[b], 0)))
        cmd += ' --fileName {}'.format(self.fake_config)
        os.system(cmd)

        # create fake data from yaml configuration file
        cmd = 'python run_scripts/fakes/make_fake.py --config {} --output {}'.format(
            self.fake_config, self.fake_data)
        os.system(cmd)

    def simulate_lc(self, zmin, zmax, zstep):
        """
        Method to simulate LC

        """
        cmd = 'python run_scripts/simulation/run_simulation.py --dbDir .'
        cmd += ' --dbDir {}'.format(self.outDir_obs)
        cmd += ' --dbName {}'.format(self.fake_name)
        cmd += ' --dbExtens npy'
        cmd += ' --SN_x1_type unique'
        cmd += ' --SN_x1_min {}'.format(self.x1)
        cmd += ' --SN_color_type unique'
        cmd += ' --SN_color_min {}'.format(self.color)
        cmd += ' --SN_z_type uniform'
        cmd += ' --SN_z_min {}'.format(zmin)
        cmd += ' --SN_z_max {}'.format(zmax)
        cmd += ' --SN_z_step {}'.format(zstep)
        cmd += ' --SN_daymax_type unique'
        cmd += ' --Observations_fieldtype Fake'
        cmd += ' --Observations_coadd 0'
        cmd += ' --radius 0.01'
        cmd += ' --Output_directory {}'.format(self.outDir_simu)
        cmd += ' --Simulator_name sn_simulator.{}'.format(self.simulator)
        cmd += ' --Multiprocessing_nproc 1'
        cmd += ' --RAmin 0.0'
        cmd += ' --RAmax 0.1'
        cmd += '  --ProductionID {}'.format(self.tag)
        cmd += ' --SN_blueCutoff {}'.format(self.bluecutoff)
        cmd += ' --SN_redCutoff {}'.format(self.redcutoff)
        cmd += ' --SN_ebvofMW {}'.format(self.ebvofMW)
        cmd += ' --npixels -1'
        cmd += ' --Simulator_errorModel {}'.format(self.error_model)

        os.system(cmd)

    def fit_lc(self, fitter):
        """
        Method to fit light curves

        """
        cmd = 'python run_scripts/fit_sn/run_sn_fit.py'
        cmd += ' --Simulations_dirname {}'.format(self.outDir_simu)
        cmd += ' --Simulations_prodid {}_0'.format(self.tag)
        cmd += ' --mbcov_estimate 0 --Multiprocessing_nproc 1'
        cmd += ' --Output_directory {}'.format(self.outDir_fit)
        cmd += ' --LCSelection_snrmin {}'.format(self.snrmin)
        cmd += ' --LCSelection_nbef {}'.format(self.nbef)
        cmd += ' --LCSelection_naft {}'.format(self.naft)
        cmd += ' --LCSelection_nbands {}'.format(self.nbands)
        cmd += ' --Fitter_name sn_fitter.fit_{}'.format(fitter)
        cmd += ' --ProductionID {}_{}'.format(self.tag, fitter)
        os.system(cmd)

    def calc_sigma_mu(self, fitter):

        tag = '{}_{}'.format(self.tag, fitter)
        inputName = glob.glob('{}/Fit_{}*.hdf5'.format(self.outDir_fit,tag))[0]

        sn = loopStack([inputName], 'astropyTable')

        fires = Table()
        for resu in sn:
            resfit = Table(resu)
            #print(resfit.columns)
            if  resfit['fitstatus'] == 'fitok':
                covDict = self.mbcovCalc(self.covmb,resfit)
                #print(covDict)
                for key in covDict.keys():
                    resfit[key] = [covDict[key]]
            else:
                pp = ['Cov_x0mb', 'Cov_x1mb', 'Cov_colormb',
              'Cov_mbmb', 'mb_recalc', 'sigma_mu']
                for key in pp:
                    resfit[key] = [-1.]
            fires = vstack([fires,resfit])         

        print(type(fires),fires.columns)
        print(fires.info)
        outName = inputName.replace('.hdf5','_sigma_mu.hdf5')
        fires.write(outName, 'lc_fit_sigma_mu', compression=True)
            

    def mbcovCalc(self,covmb,vals,alpha=0.14, beta=3.1):
        """
        Method to estimate mb covariance data

        Parameters
        ---------------
        alpha: float, opt
          alpha param (default: 0.14)
        beta: float, opt
          beta param (default: 3.1, opt)
        covmb: Mbcov class
          class to estimate mb covariance data
        vals: astropy table
          fitted parameters

        Returns
        ----------
        dict with the following keys:
        Cov_x0mb,Cov_x1mb,Cov_colormb,Cov_mbmb,mb_recalc,sigma_mu

        """
        import numpy as np

        cov = np.ndarray(shape=(3, 3), dtype=float, order='F')
        cov[0, 0] = vals['Cov_x0x0'].data
        cov[1, 1] = vals['Cov_x1x1'].data
        cov[2, 2] = vals['Cov_colorcolor'].data
        cov[0, 1] = vals['Cov_x0x1'].data
        cov[0, 2] = vals['Cov_x0color'].data
        cov[1, 2] = vals['Cov_x1color'].data
        cov[2, 1] = cov[1, 2]
        cov[1, 0] = cov[0, 1]
        cov[2, 0] = cov[0, 2]

        params = dict(zip(['x0', 'x1', 'c'], [vals['x0_fit'].data,
                                              vals['x1_fit'].data, vals['color_fit'].data]))

        resu = self.covmb.mbCovar(params, cov, ['x0', 'x1', 'c'])
        sigmu_sq = resu['Cov_mbmb']
        sigmu_sq += alpha**2 * vals['Cov_x1x1'].data + \
            beta**2 * vals['Cov_colorcolor'].data
        sigmu_sq += 2.*alpha*resu['Cov_x1mb']
        sigmu_sq += -2.*alpha*beta*vals['Cov_x1color'].data
        sigmu_sq += -2.*beta*resu['Cov_colormb']
        sigmu = np.array([0.])
        if sigmu_sq >= 0.:
            sigmu = np.sqrt(sigmu_sq)

        resu['sigma_mu'] = sigmu.item()

        return resu

        
    def getSN(self):
        """
        Method to load SN from file

        Returns
        -----------
        sn: astropy table

        """
        fName = '{}/Fit_{}_{}.hdf5'.format(self.outDir_fit,
                                           self.tag, self.fitter)
        fFile = h5py.File(fName, 'r')
        keys = list(fFile.keys())
        sn = Table()
        for key in keys:
            tab = Table.read(fName, path=key)
            sn = vstack([sn, tab])

        return sn

    def getLC(self, z=0.5):
        """"
        Method to load the light curve from file

        Returns
        -----------
        the light curve (format: astropyTable)

        """
        fName = '{}/LC_{}_0.hdf5'.format(self.outDir_simu,
                                         self.tag)
        fFile = h5py.File(fName, 'r')
        keys = list(fFile.keys())
        lc = Table()
        for key in keys:
            tab = Table.read(fName, path=key)
            if np.abs(tab.meta['z']-z) < 1.e-8:
                lc = vstack([lc, tab])

        return lc

    def zlim(self, color_cut=0.04):
        """
        Method to estimate the redshift limit

        Parameters
        ---------------
        color_cut: float, opt
           sigmaColor cut(default: 0.04)

        """

        if 'sn' not in globals():
            self.sn = self.getSN()

        # make interpolation
        from scipy.interpolate import interp1d
        interp = interp1d(np.sqrt(self.sn['Cov_colorcolor']),
                          self.sn['z'], bounds_error=False, fill_value=0.)

        zlim = np.round(interp(color_cut), 2)
        return zlim

    def plot(self, color_cut=0.04):
        """
        Method to plot sigmaC vs z

        Parameters
        ---------------
        color_cut: float, opt
          sigmaColor value to plot(default: 0.04)

        """

        if 'sn' not in globals():
            self.sn = self.getSN()

        import matplotlib.pyplot as plt
        from scipy.interpolate import interp1d

        zlim = self.zlim(color_cut)
        fig, ax = plt.subplots()
        ax.plot(self.sn['z'], np.sqrt(self.sn['Cov_colorcolor']),
                label='zlim={}'.format(zlim), color='r')

        ax.plot(ax.get_xlim(), [color_cut]*2,
                linestyle='--', color='k')

        zmin = 0.2
        zmax = zlim+0.1
        interp = interp1d(self.sn['z'], np.sqrt(self.sn['Cov_colorcolor']),
                          bounds_error=False, fill_value=0.)
        ax.set_xlim([zmin, zmax])
        ax.set_ylim([interp(zmin), interp(zmax)])
        ax.grid()
        ax.set_xlabel('z')
        ax.set_ylabel('$\sigma_{color}$')
        ax.legend(loc='upper left')
        plt.show()


def plot_sigmaC_z(sn, zlim, color_cut=0.04):
    """
    Method to plot sigmaC vs z

    Parameters
    ---------------
    color_cut: float, opt
    sigmaColor value to plot(default: 0.04)

    """

    import matplotlib.pyplot as plt
    from scipy.interpolate import interp1d

    fig, ax = plt.subplots()
    ax.plot(sn['z'], np.sqrt(sn['Cov_colorcolor']),
            label='zlim={}'.format(zlim), color='r')

    ax.plot(ax.get_xlim(), [color_cut]*2,
            linestyle='--', color='k')

    zmin = 0.2
    zmax = zlim+0.1
    interp = interp1d(sn['z'], np.sqrt(sn['Cov_colorcolor']),
                      bounds_error=False, fill_value=0.)
    ax.set_xlim([zmin, zmax])
    ax.set_ylim([interp(zmin), interp(zmax)])
    ax.grid()
    ax.set_xlabel('z')
    ax.set_ylabel('$\sigma_{color}$')
    ax.legend(loc='upper left')
    plt.show()

# this is to load option for fake cadence
path = 'input/Fake_cadence'
confDict = make_dict_from_config(path,'config_cadence.txt')

parser = OptionParser()

parser.add_option("--x1", type=float, default=-2.0,
                  help="SN x1 [%default]")
parser.add_option("--color", type=float, default=0.2,
                  help="SN color [%default]")
parser.add_option("--zmax", type=float, default=0.8,
                  help="max redshift for simulated data [%default]")
parser.add_option("--ebv", type=float, default=0.0,
                  help="ebvofMW value[%default]")
parser.add_option("--bluecutoff", type=float, default=380.,
                  help="blue cutoff for SN spectrum[%default]")
parser.add_option("--redcutoff", type=float, default=800.,
                  help="red cutoff for SN spectrum[%default]")
parser.add_option("--simus", type=str, default='sn_fast,sn_cosmo',
                  help=" simulator to use[%default]")
parser.add_option("--snrmin", type=float, default=1.,
                  help="SNR min for LC points (fit)[%default]")
parser.add_option("--nbef", type=int, default=4,
                  help="min n LC points before max (fit)[%default]")
parser.add_option("--naft", type=int, default=10,
                  help="min n LC points after max (fit)[%default]")
parser.add_option("--nbands", type=int, default=0,
                  help="min number of bands with at least 2 points with SNR>5[%default]")
parser.add_option("--error_model", type=str, default='0,1',
                  help="error model to consider[%default]")
parser.add_option("--sigma_mu", type=int, default=0,
                  help="to estimate sigma mu[%default]")


#add option for Fake data here
for key, vals in confDict.items():
    vv = vals[1]
    if vals[0] != 'str':
        vv = eval('{}({})'.format(vals[0],vals[1]))
    parser.add_option('--{}'.format(key),help='{} [%default]'.format(vals[2]),default=vv,type=vals[0],metavar='')

parser.add_option('--fake_config',help='output file name [%default]',default='Fake_cadence.yaml',type='str')

opts, args = parser.parse_args()


# make the fake config file here
newDict = {}
for key, vals in confDict.items():
    newval = eval('opts.{}'.format(key))
    newDict[key]=(vals[0],newval)

dd = make_dict_from_optparse(newDict)
# few changes to be made here: transform some of the input to list
for vv in ['seasons','seasonLength']:
    what = dd[vv]
    if '-' not in what or what[0] == '-':
        nn = list(map(int,what.split(',')))
        print('ici',nn)
    else:
        nn = list(map(int,what.split('-')))
        nn = range(np.min(nn),np.max(nn))
    dd[vv] = nn

for vv in ['MJDmin']:
    what = dd[vv]
    if '-' not in what or what[0] == '-':
        nn = list(map(float,what.split(',')))
    else:
        nn = list(map(float,what.split('-')))
        nn = range(np.min(nn),np.max(nn))
    dd[vv] = nn


#print('boo',yaml.safe_dump(dd))

#print('config',dd)
with open(opts.fake_config, 'w') as f:
    data = yaml.safe_dump(dd, f)

#fake_config = 'input/Fake_cadence/Fake_cadence.yaml'

x1 = opts.x1
color = opts.color
zmax = opts.zmax
bluecutoff = opts.bluecutoff
redcutoff = opts.redcutoff
ebv = opts.ebv
snrmin = opts.snrmin
nbef = opts.nbef
naft = opts.naft
nbands = opts.nbands

simus = list(map(str,opts.simus.split(',')))

error_models = list(map(int, opts.error_model.split(',')))

fitters = dict(zip(['sn_cosmo','sn_fast'],[['sn_cosmo'],['sn_cosmo','sn_fast']]))

for simu in simus:
    for errormod in error_models:
        cutoff = '{}_{}'.format(bluecutoff,redcutoff)
        if errormod:
            cutoff = 'error_model'
        outDir_simu = 'Output_Simu_{}_ebvofMW_{}'.format(cutoff,ebv)
        outDir_fit = 'Output_Fit_{}_ebvofMW_{}_snrmin_{}'.format(cutoff,ebv,int(snrmin))
        outDir_obs =  'Output_obs_{}_ebvofMW_{}_snrmin_{}'.format(cutoff,ebv,int(snrmin))
        tag = '{}_Fake_{}_{}_{}_ebvofMW_{}'.format(simu,x1,color,cutoff,ebv)
        sim_fit = SimFit(x1=x1, color=color,
                         error_model=errormod,
                         bluecutoff=bluecutoff,
                         redcutoff=redcutoff,
                         simulator=simu,
                         fitters=fitters[simu],
                         outDir_simu = outDir_simu,
                         outDir_fit = outDir_fit,
                         outDir_obs = outDir_obs,
                         snrmin=snrmin,
                         sigma_mu=opts.sigma_mu,
                         tag=tag)

        sim_fit.process()

"""
for errmod in error_models:
    run(x1,color,simus,ebv,bluecutoff,redcutoff,errmod,opts.fake_config,snrmin,nbef,naft,nbands,zmax)
"""
"""
# case error_model=1
run(x1,color,simus,ebv,bluecutoff,redcutoff,1,opts.fake_config,snrmin,nbef,naft,zmax)
# case error_model=0
#run(x1,color,simus,ebv,bluecutoff,redcutoff,0,opts.fake_config,snrmin,nbef,naft,zmax)
"""
