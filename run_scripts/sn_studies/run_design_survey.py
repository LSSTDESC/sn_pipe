from sn_design_dd_survey.wrapper import Data, Nvisits_cadence, Mod_z
from sn_design_dd_survey.budget import DD_Budget
from sn_design_dd_survey.snr import SNR, Nvisits_z_plot
from sn_design_dd_survey.signal_bands import RestFrameBands
from sn_design_dd_survey.showvisits import ShowVisits
from sn_design_dd_survey.templateLC import templateLC
from sn_design_dd_survey import plt
from sn_design_dd_survey.snr_m5 import SNR_m5
from sn_design_dd_survey.ana_combi import CombiChoice, Visits_Cadence
from sn_design_dd_survey.zlim_visits import RedshiftLimit


# from sn_DD_opti.showvisits import GUI_Visits
# from sn_DD_opti.budget import GUI_Budget

import os
import multiprocessing
import pandas as pd
import numpy as np
import glob
import time


def chk_cr(dirname):
    """
    Function to create a dir if does not exist

    Parameters
    ---------------
    dirname: str
      name of the directory
    """
    if not os.path.exists(dirname):
        os.makedirs(dirname)


class TemplateData:
    """
    class to generate usefull data: template LC, snr_m5, ...

    Parameters
    ---------------
    x1: float, opt
      SN x1 (default=-2.0)
    color: float, opt
       SN color (default: 0.2)
    bands: str, opt
      bands to consider (default: grizy)
    dirStudy: str, opt
      main directory where files will be produced (default: dd_design)
    dirTemplates: str, opt
      sub dir where template LC will be placed
    dirSNR_m5: str, opt
      sub dir where SNR<->m5 files will be placed

    """

    def __init__(self, x1=-2., color=0.2,
                 bands='grizy',
                 dirStudy='dd_design',
                 dirTemplates='Templates',
                 dirSNR_m5='SNR_m5'):

        self.x1 = x1
        self.color = color
        self.bands = bands

        self.dirTemplates = '{}/{}'.format(dirStudy, dirTemplates)
        self.dirSNR_m5 = '{}/{}'.format(dirStudy, dirSNR_m5)

    def templates(self,
                  zmin=0,
                  zmax=1.1,
                  zstep=0.01,
                  error_model=1,
                  bluecutoff=380.,
                  redcutoff=800.,
                  ebvofMW=0.,
                  simulator='sn_fast',
                  cadence=3.):
        """
        Method to generate templates with a defined regular cadence

        Parameters
        --------------
        zmin :  float,opt
          min redshift value (default: 0)
        zmax: float,opt
          max redshift value (default: 1.1)
        zstep: float, opt
          step redshift value (default:0.01)
        error_model: int, opt
           error model for the simulation (default:1)
        bluecutoff: float, opt
          blue cutoff if error_model=0 (default:380)
        redcutoff: float, opt
          red cutoff if error_model=0 (default: 800.)
        ebvofMW: float, opt
          ebv of MW (default:0)
        simulator: str, opt
          simulator to use to produce LCs (default: 'sn_fast')
        cadences: float, opt
          cadence of observation (the same filter for each filter) (default: 3.)
        """

        cutof = self.cutoff(error_model, bluecutoff, redcutoff)

        templid = '{}_{}_{}_ebv_{}_{}_cad_{}'.format(
            simulator, self.x1, self.color, ebv, cutof, int(cadence))
        fname = 'LC_{}_0.hdf5'.format(templid)
        cadences = dict(zip(self.bands, [cadence]*len(self.bands)))
        # generate template here
        templateLC(self.x1, self.color, simulator, ebv, bluecutoff, redcutoff,
                   error_model, zmin, zmax, zstep, self.dirTemplates, self.bands, cadences, templid)

    def cutoff(self, error_model, bluecutoff, redcutoff):

        cuto = '{}_{}'.format(bluecutoff, redcutoff)
        if error_model:
            cuto = 'error_model'

        return cuto

    def snr_m5(self, snrmin=1.):
        """
        Method to produce SNR vs m5 files

        Parameters
        --------------
        snrmin: float, opt
          SNR min selection

        """

        template_list = glob.glob('{}/LC*.hdf5'.format(self.dirTemplates))
        for lc in template_list:
            lcName = lc.split('/')[-1]
            outName = '{}/{}'.format(self.dirSNR_m5,
                                     lcName.split('.hdf5')[0].replace('LC', 'SNR_m5'))
            SNR_m5(self.dirTemplates, lcName, '{}.npy'.format(outName), snrmin)


class DD_SNR:

    def __init__(self, x1=-2., color=0.2, bands='grizy',
                 dirStudy='dd_design',
                 dirTemplates='Templates',
                 dirSNR_m5='SNR_m5',
                 dirm5='m5_files',
                 dirSNR_combi='SNR_combi',
                 dirSNR_opti='SNR_opti',
                 cadence=3,
                 error_model=1,
                 bluecutoff=380., redcutoff=800.,
                 ebvofMW=0.,
                 sn_simulator='sn_fast',
                 m5_file='medValues_flexddf_v1.4_10yrs_DD.npy'):

        self.x1 = x1
        self.color = color
        self.bands = bands

        self.dirTemplates = '{}/{}'.format(dirStudy, dirTemplates)
        self.dirSNR_m5 = '{}/{}'.format(dirStudy, dirSNR_m5)
        self.dirm5 = '{}/{}'.format(dirStudy, dirm5)
        self.dirSNR_combi = '{}/{}'.format(dirStudy, dirSNR_combi)
        self.dirSNR_opti = '{}/{}'.format(dirStudy, dirSNR_opti)

        # get data
        self.data = self.grab_data(cadence,
                                   error_model,
                                   bluecutoff, redcutoff,
                                   ebvofMW,
                                   sn_simulator,
                                   m5_file)

    def grab_data(self, cadence,
                  error_model,
                  bluecutoff, redcutoff,
                  ebvofMW,
                  sn_simulator,
                  m5_file):
        """
        Method to grab Data corresponding to a given cadence

        Parameters
        --------------
        cadence: int, opt
          cadence of the data to get (
        error_model: int, opt
           error model for the simulation (default:1)
        bluecutoff: float, opt
          blue cutoff if error_model=0 (default:380)
        redcutoff: float, opt
          red cutoff if error_model=0 (default: 800.)
        ebvofMW: float, opt
          ebv of MW (default:0)
        simulator: str, opt
          simulator to use to produce LCs (default: 'sn_fast')
        m5_file: str,opt
          m5 file (default: 'medValues_flexddf_v1.4_10yrs_DD.npy')

        """

        cutof = self.cutoff(error_model, bluecutoff, redcutoff)
        lcName = 'LC_{}_{}_{}_ebv_{}_{}_cad_{}_0.hdf5'.format(
            simulator, self.x1, self.color, ebvofMW, cutof, int(cadence))
        m5Name = '{}/{}'.format(self.dirm5, m5_file)
        return Data(self.dirTemplates, lcName, m5Name, self.x1, self.color, bluecutoff, redcutoff, error_model, bands=self.bands)

    def cutoff(self, error_model, bluecutoff, redcutoff):

        cuto = '{}_{}'.format(bluecutoff, redcutoff)
        if error_model:
            cuto = 'error_model'

        return cuto

    def plot_data(self, bluecutoff=380., redcutoff=800.):
        """
        Method to display useful plots related to data

        Parameters
        --------------
        bluecutoff: float, opt
          blue cutoff if error_model=0 (default:380)
        redcutoff: float, opt
          red cutoff if error_model=0 (default: 800.)

        """

        self.data.plotzlim()
        self.data.plotFracFlux()
        self.data.plot_medm5()

        # this is to plot restframebands cutoff
        mybands = RestFrameBands(blue_cutoff=bluecutoff,
                                 red_cutoff=redcutoff)
        mybands.plot()
        plt.show()

    def SNR_combi(self,
                  SNR_par=dict(
                      zip(['max', 'step', 'choice'], [70., 1., 'Nvisits'])),
                  zmin=0.1,
                  zmax=1.1,
                  zstep=0.05,
                  nproc=8):
        """
        Method to estimate SNR combinations

        Parameters
        --------------
        SNR_par: dict, opt
          parameters for SNR combi estimation
        zmin: float, opt
          min redshift (default: 0.1)
        zmax: float, opt
          max redshift (default: 1.1)
        zstep: float, opt
         step for redshift (default: 0.05)
        nproc: int, opt
          number of procs for multiprocessing

        """

        zref = np.round(np.arange(zmin, zmax+zstep, zstep), 2)

        SNR_name = (self.data.lcName
                    .replace('LC', 'SNR_m5')
                    .replace('.hdf5', '.npy')
                    )
        SNR_m5_file = '{}/{}'.format(self.dirSNR_m5, SNR_name)

        snr_calc = SNR(self.dirSNR_combi, self.data, SNR_par,
                       SNR_m5_file=SNR_m5_file, zref=zref,
                       save_SNR_combi=True, verbose=False, nproc=nproc)


class OptiCombi:
    """
    class to choose the optimal SNR combination

    """

    def __init__(self, fracSignalBand, dirStudy='dd_design',
                 dirSNR_combi='SNR_combi',
                 dirSNR_opti='SNR_opti',
                 snr_opti_file='opti_combi.npy'):
        """
        Method to select optimal (wrt a certain criteria) SNR combinations

        Parameters
        ---------------
        fracSignalBands: numpy array
          fraction of signal per band
        dirStudy: str, opt
          main dir for the study (default: dd_design)
        dirSNR_combi: str, opt
           SNR combi dir (default: SNR_combi)
        dirSNR_opti: str, opt
          location dir of the opti combi output file
        snr_opti_file: str, opt
          name of the output file containing optimal combinations
        """

        combi = CombiChoice(fracSignalBand, dirSNR_combi)

        resdf = pd.DataFrame()
        snr_dirs = glob.glob('{}/*'.format(dirSNR_combi))

        for fi in snr_dirs:
            z = (
                fi.split('/')[-1]
                .split('_')[-1]
            )
            z = np.round(float(z), 2)
            res = combi(z)
            if res is not None:
                resdf = pd.concat((resdf, res))

        np.save('{}/{}'.format(dirSNR_opti, snr_opti_file),
                resdf.to_records(index=False))


class DD_Design_Survey:
    """
    class to optimize the DD survey

    Parameters
    --------------
    x1: float, opt
       SN stretch (default: -2.0)
    color: float, opt
       SN color (default: 0.2)
    bands: str, opt
      bands to be considered (default: grizy)

    """

    def __init__(self, x1=-2., color=0.2, bands='grizy', dirStudy='dd_design'):

        self.x1 = x1
        self.color = color
        self.bands = bands

        self.ch_cr(dirStudy)
        self.dirTemplates = '{}/Templates'.format(dirStudy)
        self.dirSNR_m5 = '{}/SNR_m5'.format(dirStudy)
        self.dirm5 = '{}/m5_files'.format(dirStudy)
        self.dirSNR_combi = '{}/SNR_combi'.format(dirStudy)
        self.dirOpti = '{}/SNR_opti'.format(dirStudy)

        self.ch_cr(self.dirTemplates)
        self.ch_cr(self.dirSNR_m5)
        self.ch_cr(self.dirSNR_combi)
        self.ch_cr(self.dirOpti)

    def ch_cr(self, dirname):
        """
        Method to create a dir if does not exist

        Parameters
        ---------------
        dirname: str
          name of the directory
        """
        if not os.path.exists(dirname):
            os.makedirs(dirname)

    def templates(self,
                  zmin=0,
                  zmax=1.1,
                  zstep=0.01,
                  error_model=1,
                  bluecutoff=380.,
                  redcutoff=800.,
                  ebvofMW=0.,
                  simulator='sn_fast',
                  cadence=3.):
        """
        Method to generate templates with a defined regular cadence

        Parameters
        --------------
        zmin :  float,opt
          min redshift value (default: 0)
        zmax: float,opt
          max redshift value (default: 1.1)
        zstep: float, opt
          step redshift value (default:0.01)
        error_model: int, opt
           error model for the simulation (default:1)
        bluecutoff: float, opt
          blue cutoff if error_model=0 (default:380)
        redcutoff: float, opt
          red cutoff if error_model=0 (default: 800.)
        ebvofMW: float, opt
          ebv of MW (default:0)
        simulator: str, opt
          simulator to use to produce LCs (default: 'sn_fast')
        cadences: float, opt
          cadence of observation (the same filter for each filter) (default: 3.)
        """

        cutof = self.cutoff(error_model, bluecutoff, redcutoff)

        templid = '{}_{}_{}_ebv_{}_{}_cad_{}'.format(
            simulator, self.x1, self.color, ebv, cutof, int(cadence))
        fname = 'LC_{}_0.hdf5'.format(templid)
        cadences = dict(zip(self.bands, [cadence]*len(self.bands)))
        # generate template here
        templateLC(self.x1, self.color, simulator, ebv, bluecutoff, redcutoff,
                   error_model, zmin, zmax, zstep, self.dirTemplates, self.bands, cadences, templid)

    def cutoff(self, error_model, bluecutoff, redcutoff):

        cuto = '{}_{}'.format(blue_cutoff, red_cutoff)
        if error_model:
            cuto = 'error_model'

        return cuto

    def snr_m5(self, snrmin=1.):
        """
        Method to produce SNR vs m5 files

        Parameters
        --------------
        snrmin: float, opt
          SNR min selection

        """

        template_list = glob.glob('{}/LC*.hdf5'.format(self.dirTemplates))
        for lc in template_list:
            lcName = lc.split('/')[-1]
            outName = '{}/{}'.format(self.dirSNR_m5,
                                     lcName.split('.hdf5')[0].replace('LC', 'SNR_m5'))
            SNR_m5(self.dirTemplates, lcName, '{}.npy'.format(outName), snrmin)

    def data(self, cadence,
             error_model=1,
             bluecutoff=380., redcutoff=800.,
             ebvofMW=0.,
             sn_simulator='sn_fast',
             m5_file='medValues_flexddf_v1.4_10yrs_DD.npy'):
        """
        Method to grab Data corresponding to a given cadence

        Parameters
        --------------
        cadence: int, opt
          cadence of the data to get (
        error_model: int, opt
           error model for the simulation (default:1)
        bluecutoff: float, opt
          blue cutoff if error_model=0 (default:380)
        redcutoff: float, opt
          red cutoff if error_model=0 (default: 800.)
        ebvofMW: float, opt
          ebv of MW (default:0)
        simulator: str, opt
          simulator to use to produce LCs (default: 'sn_fast')
        m5_file: str,opt
          m5 file (default: 'medValues_flexddf_v1.4_10yrs_DD.npy')

        """

        cutof = self.cutoff(error_model, bluecutoff, redcutoff)
        lcName = 'LC_{}_{}_{}_ebv_{}_{}_cad_{}_0.hdf5'.format(
            simulator, self.x1, self.color, ebvofMW, cutof, int(cadence))
        m5Name = '{}/{}'.format(self.dirm5, m5_file)
        return Data(self.dirTemplates, lcName, m5Name, self.x1, self.color, bluecutoff, redcutoff, error_model, bands=self.bands)

    def plot_data(self, data, bluecutoff=380., redcutoff=800.):
        """
        Method to display useful plots related to data

        Parameters
        --------------
        bluecutoff: float, opt
          blue cutoff if error_model=0 (default:380)
        redcutoff: float, opt
          red cutoff if error_model=0 (default: 800.)

        """

        data.plotzlim()
        data.plotFracFlux()
        data.plot_medm5()

        # this is to plot restframebands cutoff
        mybands = RestFrameBands(blue_cutoff=bluecutoff,
                                 red_cutoff=redcutoff)
        mybands.plot()
        plt.show()

    def SNR_combi(self, data,
                  SNR_par=dict(
                      zip(['max', 'step', 'choice'], [70., 1., 'Nvisits'])),
                  zmin=0.1,
                  zmax=1.1,
                  zstep=0.05,
                  nproc=8):
        """
        Method to estimate SNR combinations

        Parameters
        --------------
        data: Data
          data to use for processing
        SNR_par: dict, opt
          parameters for SNR combi estimation
        zmin: float, opt
          min redshift (default: 0.1)
        zmax: float, opt
          max redshift (default: 1.1)
        zstep: float, opt
         step for redshift (default: 0.05)
        nproc: int, opt
          number of procs for multiprocessing

        """

        zref = np.round(np.arange(zmin, zmax+zstep, zstep), 2)

        SNR_name = (data.lcName
                    .replace('LC', 'SNR_m5')
                    .replace('.hdf5', '.npy')
                    )
        SNR_m5_file = '{}/{}'.format(self.dirSNR_m5, SNR_name)

        snr_calc = SNR(self.dirSNR_combi, data, SNR_par,
                       SNR_m5_file=SNR_m5_file, zref=zref,
                       save_SNR_combi=True, verbose=False, nproc=nproc)

    def optiCombi(self, fracSignalBand, snr_opti_file='opti_combi.npy'):
        """
        Method to select optimal (wrt a certain criteria) SNR combinations

        Parameters
        ---------------
        fracSignalBands: numpy array
          fraction of signal per band
        snr_opti_file: str, opt
          name of the output file containing optimal combinations
        """
        combi = CombiChoice(fracSignalBand, self.dirSNR_combi)

        resdf = pd.DataFrame()
        snr_dirs = glob.glob('{}/*'.format(self.dirSNR_combi))

        for fi in snr_dirs:
            z = (
                fi.split('/')[-1]
                .split('_')[-1]
            )
            z = np.round(float(z), 2)
            res = combi(z)
            if res is not None:
                resdf = pd.concat((resdf, res))

        np.save('{}/opti_combi.npy'.format(self.dirOpti),
                resdf.to_records(index=False))

    def nvisits_cadence(self, m5_single, snr_opti_file):
        """
        Method to estimate the number of visits
        as a function of the cadence.
        The idea is that the SNR optimized for a given cadence
        is the same for other cadences

        Three ingredients needed:
        - m5 single visit (file used to make SNR combis)
        - SNR_m5 vs cadence
        - SNR_opti file

        Parameters
        ---------------
        m5_single: numpy array
          m5 single band (median per filter over seasons)
        snr_opti: str
          file name corresponding to SNR opti
        """

        # load SNR_opti file
        snr_opti_df = pd.DataFrame(
            np.load('{}/{}'.format(self.dirOpti, snr_opti_file), allow_pickle=True))

        cadvis = Visits_Cadence(snr_opti_df, m5_single)
        bands = np.unique(m5_single['filter'])
        bb = []
        for b in bands:
            bb.append('Nvisits_{}'.format(b))
        # load SNR_m5 files for various cadences
        fis = glob.glob('{}/*'.format(self.dirSNR_m5))

        res = pd.DataFrame()
        for fi in fis:
            fib = (
                fi.split('/')[-1]
                .split('_')
            )
            idx = fib.index('cad')
            cadence = int(fib[idx+1])
            # load m5 single file
            m5_cad = pd.DataFrame(np.load(fi, allow_pickle=True))
            nv_cad = cadvis(m5_cad)
            nv_cad['cadence'] = cadence
            res = pd.concat((res, nv_cad))

        np.save('Nvisits_z_med.npy', res.to_records(index=False))

        # transform the data to have a format compatible with GUI
        TransformData(res, 'Nvisits_z_med', grlist=['z', 'cadence'])


class Nvisits_Cadence_Fields:

    def __init__(self, x1=-2.0, color=0.2,
                 error_model=1,
                 bluecutoff=380., redcutoff=800.,
                 ebvofMW=0.,
                 sn_simulator='sn_fast',
                 dirTemplates='dd_design/Templates'):
        """
        class  to estimate the number of visits for DD fields depending on cadence
        from a number of visits defined with median m5 values

        Parameters
        ---------------
        x1 : float, opt
          SN x1 (default: -2.0)
        color: float, opt
          SN color (default : 0.2)
        error_model: int, opt
          error model for LC (default: 1)
        bluecutoff: float,opt
          blue cutoff (if error_model=0) (default: 380.)
        redcutoff: float, opt
          red cutoff (if error_model=0) (default: 800.)
        ebvofMW: float, opt
         ebv of MW (default: 0.)
        sn_simulator: str, opt
         simulator to generate the template (default: sn_fast)
        dirTemplates: str, opt
          location dir of the templates (default: dd_design/Templates)

        """
        self.x1 = x1
        self.color = color
        self.error_model = error_model
        self.bluecutoff = bluecutoff
        self.redcutoff = redcutoff
        self.ebvofMW = ebvofMW
        self.sn_simulator = sn_simulator
        self.dirTemplates = dirTemplates

        # load nvisits_ref
        self.nvisits_ref = np.load('Nvisits_z_med.npy', allow_pickle=True)

        restot = self.multiproc()

        # restot = pd.DataFrame(
        #    np.load('Nvisits_z_fields.npy', allow_pickle=True))

        TransformData(restot, 'Nvisits_z_fields', grlist=[
            'z', 'cadence', 'fieldname', 'season'])

    def multiproc(self):

        resdf = pd.DataFrame()

        time_ref = time.time()
        result_queue = multiprocessing.Queue()

        cadences = range(2, 5)
        nproc = len(cadences)
        for j, cadence in enumerate(cadences):
            p = multiprocessing.Process(name='Subprocess-'+str(j), target=self.nvisits_single_cadence,
                                        args=(cadence, j, result_queue))
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

        # transformation to get the apprriate format for GUI

        #np.save('Nvisits_z_fields.npy', restot.to_records(index=False))

        print('end of processing', time.time()-time_ref)

        return restot

    def nvisits_single_cadence(self, cadence,
                               j=0, output_q=None):

        resdf = pd.DataFrame()
        red = RedshiftLimit(self.x1, self.color,
                            cadence=cadence,
                            error_model=self.error_model,
                            bluecutoff=self.bluecutoff, redcutoff=self.redcutoff,
                            ebvofMW=self.ebvofMW,
                            sn_simulator=self.sn_simulator,
                            lcDir=self.dirTemplates)

        idx = np.abs(self.nvisits_ref['cadence']-cadence) < 1.e-5
        sela = self.nvisits_ref[idx]
        for min_par in np.unique(sela['min_par']):
            idb = sela['min_par'] == min_par
            sel_visits = sela[idb]
            respar = red(sel_visits)
            respar = respar.reset_index(drop=True)
            resdf = pd.concat((resdf, respar))

        if output_q is not None:
            return output_q.put({j: resdf})
        else:
            return resdf


class TransformData:
    """
    class to transform a set of rows to a unique one

    Parameters
    ---------------
    df: pandas df 
     data to transform
    fi: str
      npy file to transform
    grlist: list(str)
      used for the groupby df
    """

    def __init__(self, df=None, outName='', grlist=['z', 'cadence']):

        print(df.columns)
        for min_par in np.unique(df['min_par']):
            idx = df['min_par'] == min_par
            gr = df[idx].groupby(grlist).apply(
                lambda x: self.transform(x)).reset_index()

            if 'season' not in gr.columns:
                gr['season'] = 0
            if 'fieldname' not in gr.columns:
                gr['fieldname'] = 'all'

            if 'zlim' in gr.columns:
                gr['z'] = gr['zlim']
                gr = gr.drop(columns=['zlim'])

            np.save('{}_{}.npy'.format(outName, min_par),
                    gr.to_records(index=False))

    def transform(self, grp):
        """
        Method to transform a set of rows to a unique one

        Parameters
        ---------------
        grp: pandas df group
         data to modify

        Returns
        ----------
        pandas df of the data transformed

        """
        r = []
        names = []

        for b in np.unique(grp['band']):
            idx = grp['band'] == b
            r.append(grp[idx]['Nvisits'].values.item())
            names.append('Nvisits_{}'.format(b))

        r.append(grp['Nvisits'].sum())
        names.append('Nvisits')

        if 'zlim' in grp.columns:
            r.append(grp['zlim'].median())
            names.append('zlim')

        return pd.DataFrame(np.rec.fromrecords([r], names=names))


# Step 1: Load the data needed for analysis
# ----------------------------------------
bluecutoff = 380.
redcutoff = 800.
x1 = -2.0
color = 0.2
simulator = 'sn_fast'
ebv = 0.
error_model = 1
bands = 'grizy'
zmin = 0.0
zmax = 1.0
zstep = 0.01
bands = 'grizy'
cadence = 3.
cadences = dict(zip(bands, [cadence]*len(bands)))
cutoff = '{}_{}'.format(bluecutoff, redcutoff)
if error_model:
    cutoff = 'error_model'

# define dir here

dirStudy = 'dd_design_new'
dirTemplates = 'Templates'
dirSNR_m5 = 'SNR_m5'
dirm5 = 'm5_files'
dirSNR_combi = 'SNR_combi'
dirSNR_opti = 'SNR_opti'


chk_cr('{}/{}'.format(dirStudy, dirTemplates))
chk_cr('{}/{}'.format(dirStudy, dirSNR_m5))
chk_cr('{}/{}'.format(dirStudy, dirm5))
chk_cr('{}/{}'.format(dirStudy, dirSNR_combi))
chk_cr('{}/{}'.format(dirStudy, dirSNR_opti))

# generate template data (LC+SNR_m5) here

"""
# class instance
templ = TemplateData(x1=x1, color=color, bands='grizy',
                     dirStudy=dirStudy,
                     dirTemplates=dirTemplates,
                     dirSNR_m5=dirSNR_m5)
# create template LC for cadence = 1 to 4 days

for cadence in range(1, 5):
    templ.templates(cadence=cadence)

# estimate SNR vs m5 for the above generated templates
templ.snr_m5()
"""

m5_file = 'medValues_flexddf_v1.4_10yrs_DD.npy'
cadence_for_opti = 3
dd_snr = DD_SNR(x1=x1, color=color,
                bands=bands,
                dirStudy=dirStudy,
                dirTemplates=dirTemplates,
                dirSNR_m5=dirSNR_m5,
                dirm5=dirm5,
                dirSNR_combi=dirSNR_combi,
                cadence=cadence_for_opti,
                error_model=error_model,
                bluecutoff=bluecutoff,
                redcutoff=redcutoff,
                ebvofMW=ebv,
                sn_simulator=simulator,
                m5_file=m5_file)

# dd_snr.plot_data()

dd_snr.SNR_combi()
print(test)
process = DD_Design_Survey()

# create template LC for cadence = 1 to 4 days

"""
for cadence in range(1, 5):
    process.templates(cadence=cadence)
"""

# estimate SNR vs m5 for the above generated templates
# process.snr_m5()

# get data corresponding to cadence = 3
# m5_file = 'medValues_flexddf_v1.4_10yrs_DD.npy'
# data = process.data(cadence=3, m5_file=m5_file)

# usefull cross-check plots
# process.plot_data(data)

# estimate SNR combinations
# process.SNR_combi(data, nproc=4)


# analyze SNR combinations - get the optimal ones
# fracSignalBand = data.fracSignalBand.fracSignalBand
# process.optiCombi(fracSignalBand, snr_opti_file='opti_combi.npy')

# propagate optimisation to all cadences

# process.nvisits_cadence(data.m5_Band, snr_opti_file='opti_combi.npy')
# Nvisits_z_plot('Nvisits_z_med.npy')

# get the number of visits per field
Nvisits_Cadence_Fields()
# create snr_m5 files for above-generated templates


print(test)


plot_input = False
plot_snr = False
plot_nvisits = False
plot = False

theDir = 'input/sn_studies'
# simulation of template LC here
prodid = '{}_{}_{}_ebv_{}_{}_cad_{}'.format(
    simulator, x1, color, ebv, cutoff, int(cadence))
fname = 'LC_{}_0.hdf5'.format(prodid)
if not os.path.isfile('{}/{}'.format(theDir, fname)):
    templateLC(x1, color, simulator, ebv, blue_cutoff, red_cutoff,
               error_model, zmin, zmax, zstep, theDir, bands, cadences, prodid)

m5file = 'medValues_flexddf_v1.4_10yrs_DD.npy'

data = Data(theDir, fname, m5file, x1, color,
            blue_cutoff, red_cutoff, error_model, bands=bands)

print(type(data))

if plot_input:
    # few plots related to data

    data.plotzlim()
    data.plotFracFlux()
    data.plot_medm5()

    # this is to plot restframebands cutoff
    mybands = RestFrameBands(blue_cutoff=blue_cutoff,
                             red_cutoff=red_cutoff)
    print(mybands.zband)
    mybands.plot()
    plt.show()

fracSB = 'fracSignalBand.npy'
if not os.path.isfile(fracSB):
    fracSignalBand = data.fracSignalBand.fracSignalBand
    np.save(fracSB, fracSignalBand.to_records(index=False))

# SNR vs m5 reference file needed
SNR_m5_file = 'SNR_m5_error_model_snrmin_1_cad_{}.npy'.format(int(cadence))
if not os.path.isfile(SNR_m5_file):
    cmd = 'python run_scripts/sn_studies/snr_m5.py --inputDir {} --refLC {} --outFile {}'.format(
        theDir, fname, SNR_m5_file)
    os.system(cmd)

# Step 2: get the SNR requirements (minimal per band) with sigma_C<0.04
# ----------------------------------------------------------------------

# Create dir for SNR output
# ---------------------------
SNRDir = 'SNR_files'
if not os.path.isdir(SNRDir):
    os.makedirs(SNRDir)

# criteria used for SNR choice
# can be :
# Nvisits, Nvisits_y -> minimal number of visits (total) or in the y band
# fracflux -> SNR distribution (per band) according to the flux distribution
# -------------------------------------------------------------------------


# choose one SNR distribution
SNR_par = dict(zip(['max', 'step', 'choice'], [70., 1., 'Nvisits']))

zref = np.round(np.arange(0.1, np.max(data.lc['z'])+0.1, 0.05), 2)
# zref = np.round(np.arange(0.1, np.max(data.lc['z'])+0.1, 0.1), 2)
# zref = np.round(np.array([0.75,0.9]), 2)

snr_calc = SNR(SNRDir, data, SNR_par,
               SNR_m5_file=SNR_m5_file, zref=zref,
               save_SNR_combi=False, verbose=False, nproc=8)

# plot the results

# load m5 reference values - here: med value per field per filter per season
# m5_type = 'median_m5_field_filter_season'  # m5 values
plot_snr = True
if plot_snr:
    snrplot = SNR_plot('SNR_files', -2.0, 0.2, 1.0, cutoff,
                       3., theDir, m5file, 'median_m5_filter')
    snrplot.plotSummary()
    # snrplot.plotSummary_band()
    # for combi in ['fracflux_rizy', 'Nvisits_rizy', 'Nvisits_y_rizy']:
    """
    for combi in ['Nvisits_grizy']:
        snrplot.plotIndiv(combi)
        snrplot.plotIndiv(combi, legy='Filter allocation')
    """
    plt.show()


# snr_calc.plot()
# plt.show()

# Step 3: Estimate the DD-DESC budget
# ----------------------------------------------------------------------

cadence = -1  # define the cadence -1 = all cadences

# load m5 reference values - here: med value per field per filter per season
m5_type = 'median_m5_field_filter_season'  # m5 values
myvisits_seasons = Nvisits_cadence(
    snr_calc.SNR, cadence, theDir, m5file, m5_type, 'Nvisits', bands)

# this is to plot the variation of the number visits vs season

if plot_nvisits:
    print('plot number of visits vs season')
    myvisits_seasons.plot()

    plt.show()

# load m5 reference file - here median per filter (over fields and seasons)
m5_type = 'median_m5_filter'

myvisits_ref = Nvisits_cadence(
    snr_calc.SNR, cadence, theDir, m5file, m5_type, 'Nvisits', bands)


# visualization of the results is done with the sn_DD_Opti package
nvisits_cadence = 'Nvisits_cadence_Nvisits_median_m5_filter.npy'
nvisits_cadence_season = 'Nvisits_cadence_Nvisits_median_m5_field_filter_season.npy'

dir_config = 'sn_DD_Opti/input'
dir_config = '.'

# this is to display the number of visits vs z for a given cadence
# GUI_Visits(nvisits_cadence, cadence=3, dir_config=dir_config)

# this is to display budget vs zlim (and filter allocation)
"""
GUI_Budget(nvisits_cadence,
           nvisits_cadence_season,
           runtype='Nvisits_single', dir_config='.')

plt.show()
"""
