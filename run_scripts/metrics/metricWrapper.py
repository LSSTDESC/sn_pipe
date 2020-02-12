import numpy as np
from sn_metrics.sn_snr_metric import SNSNRMetric
from sn_metrics.sn_cadence_metric import SNCadenceMetric
from sn_metrics.sn_obsrate_metric import SNObsRateMetric
from sn_metrics.sn_nsn_metric import SNNSNMetric
from sn_metrics.sn_sl_metric import SLSNMetric
from sn_tools.sn_cadence_tools import ReferenceData
from sn_tools.sn_utils import GetReference
import os
import multiprocessing
import healpy as hp


class MetricWrapper:
    def __init__(self, name='Cadence', season=-1, coadd=True, fieldType='DD', nside=64, ramin=0., ramax=360., decmin=-1.0, decmax=-1.0):

        self.name = '{}Metric_{}_nside_{}_coadd_{}_{}_{}_{}_{}'.format(name,
                                                                       fieldType, nside, coadd, ramin, ramax, decmin, decmax)

        self.metric = None

    def run(self, obs):
        return self.metric.run(obs)


class CadenceMetricWrapper(MetricWrapper):
    def __init__(self, name='Cadence', season=-1, coadd=True, fieldType='DD', nside=64, ramin=0., ramax=360., decmin=-1.0, decmax=-1.0, metadata={}):
        super(CadenceMetricWrapper, self).__init__(
            name=name, season=season, coadd=coadd, fieldType=fieldType,
            nside=nside, ramin=ramin, ramax=ramax, decmin=decmin, decmax=decmax)

        self.metric = SNCadenceMetric(
            coadd=coadd, nside=nside, verbose=metadata.verbose)


class SNRMetricWrapper(MetricWrapper):
    def __init__(self, name='SNR', season=-1, coadd=True, fieldType='DD', nside=64, ramin=0., ramax=360., decmin=-1.0, decmax=-1.0, metadata={}):
        super(SNRMetricWrapper, self).__init__(
            name=name, season=season, coadd=coadd, fieldType=fieldType,
            nside=nside, ramin=ramin, ramax=ramax, decmin=decmin, decmax=decmax)

        shift = 10.
        x1 = -2.0
        color = 0.2

        fake_file = '{}/{}.yaml'.format(metadata.dirFakes, 'Fake_cadence')

        Li_files = []
        mag_to_flux_files = []

        #names_ref = list(metadata.names_ref)
        for name in [metadata.names_ref]:
            Li_files.append(
                '{}/Li_{}_{}_{}.npy'.format(metadata.dirRefs, name, x1, color))
            mag_to_flux_files.append(
                '{}/Mag_to_Flux_{}.npy'.format(metadata.dirRefs, name))

        lim_sn = ReferenceData(
            Li_files, mag_to_flux_files, metadata.band, metadata.z)

        self.metric = SNSNRMetric(lim_sn=lim_sn, fake_file=fake_file, coadd=coadd,
                                  names_ref=[metadata.names_ref], shift=shift, season=season, z=metadata.z, verbose=metadata.verbose)


class ObsRateMetricWrapper(MetricWrapper):
    def __init__(self, name='ObsRate', season=-1, coadd=True, fieldType='DD', nside=64, ramin=0., ramax=360., decmin=-1.0, decmax=-1.0, metadata={}):
        super(ObsRateMetricWrapper, self).__init__(
            name=name, season=season, coadd=coadd, fieldType=fieldType,
            nside=nside, ramin=ramin, ramax=ramax, decmin=decmin, decmax=decmax)

        x1 = -2.0
        color = 0.2

        Li_files = []
        mag_to_flux_files = []
        for name in [metadata.names_ref]:
            Li_files.append(
                '{}/Li_{}_{}_{}.npy'.format(metadata.dirRefs, name, x1, color))
            mag_to_flux_files.append(
                '{}/Mag_to_Flux_{}.npy'.format(metadata.dirRefs, name))

        #self.bands = bands
        bands = 'gri'
        lim_sn = {}
        for band in bands:
            lim_sn[band] = ReferenceData(
                Li_files, mag_to_flux_files, band, metadata.z)

        SNR = [30., 40., 30.]  # WFD SNR cut to estimate sum(Li**2)

        snr_ref = dict(zip(bands, SNR))

        self.metric = SNObsRateMetric(lim_sn=lim_sn, names_ref=[metadata.names_ref],
                                      coadd=coadd, season=season, z=metadata.z, bands=bands, snr_ref=snr_ref, verbose=metadata.verbose)


class NSNMetricWrapper(MetricWrapper):
    def __init__(self, name='NSN', season=-1, coadd=True, fieldType='DD',
                 nside=64, ramin=0., ramax=360., decmin=-1.0,
                 decmax=-1.0, metadata={}):
        super(NSNMetricWrapper, self).__init__(
            name=name, season=season, coadd=coadd, fieldType=fieldType,
            nside=nside, ramin=ramin, ramax=ramax, decmin=decmin, decmax=decmax)

        verbose = True
        zmax = 1.1
        if fieldType == 'WFD':
            zmax = 0.6

        self.Instrument = {}
        self.Instrument['name'] = 'LSST'  # name of the telescope (internal)
        # dir of throughput
        self.Instrument['throughput_dir'] = 'LSST_THROUGHPUTS_BASELINE'
        self.Instrument['atmos_dir'] = 'THROUGHPUTS_DIR'  # dir of atmos
        self.Instrument['airmass'] = 1.2  # airmass value
        self.Instrument['atmos'] = True  # atmos
        self.Instrument['aerosol'] = False  # aerosol

        lc_reference = {}
        self.gamma_reference = 'reference_files/gamma.hdf5'

        x1_colors = [(-2.0, -0.2), (-2.0, 0.0), (-2.0, 0.2),
                     (0.0, -0.2), (0.0, 0.0), (0.0, 0.2),
                     (2.0, -0.2), (2.0, 0.0), (2.0, 0.2)]
        # (2.0, -0.2)] #(2.0, 0.0), (2.0, 0.2)]

        if metadata.proxy_level > 0:
            x1_colors = [(-2.0, 0.2), (0.0, 0.0)]

        print('Loading reference files')
        result_queue = multiprocessing.Queue()

        for j in range(len(x1_colors)):
            x1 = x1_colors[j][0]
            color = x1_colors[j][1]
            fname = '{}/LC_{}_{}_vstack.hdf5'.format(
                metadata.templateDir, x1, color)
            p = multiprocessing.Process(
                name='Subprocess_main-'+str(j), target=self.load, args=(fname, j, result_queue))
            p.start()

        resultdict = {}
        for j in range(len(x1_colors)):
            resultdict.update(result_queue.get())

        for p in multiprocessing.active_children():
            p.join()

        for j in range(len(x1_colors)):
            if resultdict[j] is not None:
                lc_reference[x1_colors[j]] = resultdict[j]

        print('Reference data loaded', lc_reference.keys())

        # LC selection criteria

        if fieldType == 'DD':
            N_bef = 2
            N_aft = 5
            snr_min = 5.
            N_phase_min = 1
            N_phase_max = 1

        if fieldType == 'WFD':
            N_bef = 2
            N_aft = 5
            snr_min = 0.
            N_phase_min = 0
            N_phase_max = 0

        if fieldType == 'Fake':
            N_bef = 0
            N_aft = 0
            snr_min = 0.
            N_phase_min = 0
            N_phase_max = 0

        # load x1_color_dist

        x1_color_dist = np.genfromtxt('reference_files/Dist_X1_Color_JLA_high_z.txt', dtype=None,
                                      names=('x1', 'color', 'weight_x1', 'weight_x1', 'weight_tot'))

        # metric instance
        pixArea = hp.nside2pixarea(nside, degrees=True)

        self.metric = SNNSNMetric(
            lc_reference, season=season, zmax=zmax, pixArea=pixArea,
            verbose=metadata.verbose, ploteffi=metadata.ploteffi,
            N_bef=N_bef, N_aft=N_aft,
            snr_min=snr_min,
            N_phase_min=N_phase_min,
            N_phase_max=N_phase_max,
            outputType=metadata.outputType,
            proxy_level=metadata.proxy_level,
            x1_color_dist=x1_color_dist,
            coadd=coadd, lightOutput=metadata.lightOutput, T0s=metadata.T0s)

    def load(self, fname, j=-1, output_q=None):

        lc_ref = GetReference(
            fname, self.gamma_reference, self.Instrument)

        if output_q is not None:
            output_q.put({j: lc_ref})
        else:
            return tab_tot


class SLMetricWrapper(MetricWrapper):
    def __init__(self, name='SL', season=-1, coadd=0, nside=64, fieldType='WFD', ramin=0., ramax=360., decmin=-1.0, decmax=-1.0, metadata={}):
        super(SLMetricWrapper, self).__init__(
            name=name, season=season, coadd=coadd, fieldType=fieldType,
            nside=nside, ramin=ramin, ramax=ramax, decmin=decmin, decmax=decmax)

        self.metric = SLSNMetric(
            season=season, nside=nside, coadd=coadd, verbose=metadata.verbose)
