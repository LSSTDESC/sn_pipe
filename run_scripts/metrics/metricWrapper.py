import numpy as np
from sn_metrics.sn_snr_metric import SNSNRMetric
from sn_metrics.sn_cadence_metric import SNCadenceMetric
from sn_metrics.sn_snrrate_metric import SNSNRRateMetric
from sn_metrics.sn_nsn_metric import SNNSNMetric
from sn_metrics.sn_sl_metric import SLSNMetric
from sn_tools.sn_cadence_tools import ReferenceData
from sn_tools.sn_utils import GetReference
import os


class CadenceMetricWrapper:
    def __init__(self, season=-1, coadd=True, fieldtype='DD'):

        self.metric = SNCadenceMetric(coadd=coadd)
        self.name = 'CadenceMetric_{}'.format(fieldtype)

    def run(self,obs,filterCol='filter'):
        
        return self.metric.run(obs)


class SNRMetricWrapper:
    def __init__(self, z=0.3, x1=-2.0, color=0.2, names_ref=['SNCosmo'], coadd=False, dirfiles='reference_files', dirFakes='input', shift=10., season=-1):

        self.z = z
        self.coadd = coadd
        self.names_ref = names_ref
        self.season = season
        self.shift = shift
        self.name = 'SNRMetric'
        self.fake_file = '{}/{}.yaml'.format(dirFakes, 'Fake_cadence')
        Li_files = []
        mag_to_flux_files = []
        for name in names_ref:
            Li_files.append(
                '{}/Li_{}_{}_{}.npy'.format(dirfiles, name, x1, color))
            mag_to_flux_files.append(
                '{}/Mag_to_Flux_{}.npy'.format(dirfiles, name))

        bands = 'grizy'
        self.lim_sn = {}
        for band in bands:
            self.lim_sn[band] = ReferenceData(
                Li_files, mag_to_flux_files, band, z)

    def run(self, band, obs, filterCol='filter'):
        idx = obs[filterCol] == band
        sel = obs[idx]
        metric = SNSNRMetric(lim_sn=self.lim_sn[band], fake_file=self.fake_file, coadd=self.coadd,
                             names_ref=self.names_ref, shift=self.shift, season=self.season, z=self.z)
        return metric.run(np.copy(sel))


class SNRRateMetricWrapper:
    def __init__(self, z=0.3, x1=-2.0, color=0.2, names_ref=['SNCosmo'], coadd=False, dirfiles='reference_files', season=-1, bands='gri'):

        self.z = z
        self.coadd = coadd
        self.names_ref = names_ref
        self.season = season
        self.name = 'SNRRateMetric'
        Li_files = []
        mag_to_flux_files = []
        for name in names_ref:
            Li_files.append(
                '{}/Li_{}_{}_{}.npy'.format(dirfiles, name, x1, color))
            mag_to_flux_files.append(
                '{}/Mag_to_Flux_{}.npy'.format(dirfiles, name))

        self.bands = bands
        self.lim_sn = {}
        for band in self.bands:
            self.lim_sn[band] = ReferenceData(
                Li_files, mag_to_flux_files, band, z)

        SNR = [30., 40., 30.]  # WFD SNR cut to estimate sum(Li**2)

        self.snr_ref = dict(zip(self.bands, SNR))

    def run(self, obs, filterCol='filter'):
        metric = SNSNRRateMetric(lim_sn=self.lim_sn, names_ref=self.names_ref,
                                 coadd=self.coadd, season=self.season, z=self.z, bands=self.bands, snr_ref=self.snr_ref)
        return metric.run(np.copy(obs))


class NSNMetricWrapper:
    def __init__(self, fieldtype='DD', nside=64, pixArea=9.6, season=-1, templateDir='', ploteffi=False, verbose=False):

        zmax = 1.3

        self.name = 'NSNMetric_{}_zlim_nside_{}'.format(fieldtype, nside)

        Instrument = {}
        Instrument['name'] = 'LSST'  # name of the telescope (internal)
        # dir of throughput
        Instrument['throughput_dir'] = 'LSST_THROUGHPUTS_BASELINE'
        Instrument['atmos_dir'] = 'THROUGHPUTS_DIR'  # dir of atmos
        Instrument['airmass'] = 1.2  # airmass value
        Instrument['atmos'] = True  # atmos
        Instrument['aerosol'] = False  # aerosol

        lc_reference = {}
        gamma_reference = 'reference_files/gamma.hdf5'

        print('Loading reference files',)
        for (x1, color) in [(-2.0, 0.2), (0.0, 0.0)]:
            fname = '{}/LC_{}_{}_vstack.hdf5'.format(templateDir, x1, color)

            lc_reference[(x1, color)] = GetReference(
                fname, gamma_reference, Instrument)

        print('Reference data loaded')

        # LC selection criteria

        if fieldtype == 'DD':
            N_bef = 5
            N_aft = 10
            snr_min = 5.
            N_phase_min = 1
            N_phase_max = 1

        if fieldtype == 'WFD':
            N_bef = 4
            N_aft = 10
            snr_min = 0.
            N_phase_min = 0
            N_phase_max = 0

        # metric instance

        self.metric = SNNSNMetric(
            lc_reference, season=season, zmax=zmax, pixArea=pixArea, verbose=verbose, ploteffi=ploteffi, N_bef=N_bef, N_aft=N_aft, snr_min=snr_min, N_phase_min=N_phase_min, N_phase_max=N_phase_max)

    def run(self, obs):
        return self.metric.run(obs)


class SLMetricWrapper:
    def __init__(self, season=-1, nside=64):

        self.name = 'SLMetric'
        self.metric = SLSNMetric(season=season, nside=nside)

    def run(self, obs):
        return self.metric.run(obs)
