import numpy as np
from sn_metrics.sn_snr_metric import SNSNRMetric
from sn_metrics.sn_cadence_metric import SNCadenceMetric
from sn_metrics.sn_obsrate_metric import SNObsRateMetric
from sn_metrics.sn_nsn_metric import SNNSNMetric
from sn_metrics.sn_saturation_metric import SNSaturationMetric
from sn_metrics.sn_sl_metric import SLSNMetric
from sn_tools.sn_cadence_tools import ReferenceData
from sn_tools.sn_utils import GetReference, LoadDust
from sn_tools.sn_telescope import Telescope
from sn_tools.sn_io import check_get_file
import os
import multiprocessing
import healpy as hp
import yaml


class MetricWrapper:
    def __init__(self, name='Cadence', season=-1,
                 coadd=True, fieldType='DD', nside=64,
                 RAmin=0., RAmax=360.,
                 Decmin=-1.0, Decmax=-1.0,
                 npixels=0, metadata={}, outDir='', ebvofMW=-1.0):

        self.name = '{}Metric_{}_nside_{}_coadd_{}_{}_{}_{}_{}_npixels_{}_ebvofMW_{}'.format(name,
                                                                                             fieldType, nside, coadd, RAmin, RAmax, Decmin, Decmax, npixels, ebvofMW)

        self.metric = None

        self.metadata = vars(metadata)

        # select values to dump
        self.metaout = ['name', 'seasons', 'coadd', 'fieldType',
                        'nside', 'RAmin', 'RAmax', 'Decmin', 'Decmax', 'metric', 'Output dir', 'remove_dithering', 'ebvofMW']

        self.metadata['name'] = self.name
        self.metadata['metric'] = name
        self.metadata['Output dir'] = outDir
        self.outDir = outDir

    def run(self, obs):
        return self.metric.run(obs)

    def saveConfig(self):
        ti = dict(zip(self.metaout, [self.metadata[k] for k in self.metaout]))
        nameOut = '{}/{}_conf.yaml'.format(self.outDir, self.name)
        print('Saving configuration file', nameOut)
        with open(nameOut, 'w') as file:
            yaml.dump(ti, file)


class CadenceMetricWrapper(MetricWrapper):
    def __init__(self, name='Cadence', season=-1,
                 coadd=True, fieldType='DD', nside=64,
                 RAmin=0., RAmax=360.,
                 Decmin=-1.0, Decmax=-1.0,
                 npixels=0,
                 metadata={}, outDir='', ebvofMW=-1.0, bluecutoff=380.0, redcutoff=800.0):
        super(CadenceMetricWrapper, self).__init__(
            name=name, season=season, coadd=coadd, fieldType=fieldType,
            nside=nside, RAmin=RAmin, RAmax=RAmax,
            Decmin=Decmin, Decmax=Decmax, npixels=npixels, metadata=metadata, outDir=outDir, ebvofMW=ebvofMW)

        self.metric = SNCadenceMetric(
            coadd=coadd, nside=nside, verbose=metadata.verbose)

        self.saveConfig()


class SNRMetricWrapper(MetricWrapper):
    def __init__(self, name='SNR', season=-1,
                 coadd=True, fieldType='DD', nside=64,
                 RAmin=0., RAmax=360.,
                 Decmin=-1.0, Decmax=-1.0,
                 npixels=0,
                 metadata={}, outDir='', ebvofMW=-1.0, bluecutoff=380.0, redcutoff=800.0):
        super(SNRMetricWrapper, self).__init__(
            name=name, season=season, coadd=coadd, fieldType=fieldType,
            nside=nside, RAmin=RAmin, RAmax=RAmax,
            Decmin=Decmin, Decmax=Decmax, npixels=npixels,
            metadata=metadata, outDir=outDir, ebvofMW=ebvofMW)

        self.metaout += ['x1', 'color', 'dirFake', 'dirRefs', 'band', 'z']
        web_path = 'https://me.lsst.eu/gris/DESC_SN_pipeline'

        shift = 10.
        x1 = metadata.x1
        color = metadata.color

        print(metadata)
        fake_file = '{}/{}.yaml'.format(metadata.dirFake,
                                        'Fake_cadence_snrmetric')

        Li_files = []
        mag_to_flux_files = []

        # names_ref = list(metadata.names_ref)
        for name in [metadata.names_ref]:
            Li_name = 'Li_{}_{}_{}.npy'.format(name, x1, color)
            mag_name = 'Mag_to_Flux_{}.npy'.format(name)
            Li_files.append(
                '{}/{}'.format(metadata.dirRefs, Li_name))
            mag_to_flux_files.append(
                '{}/{}'.format(metadata.dirRefs, mag_name))
            check_get_file(web_path, metadata.dirRefs, Li_name)
            check_get_file(web_path, metadata.dirRefs, mag_name)

        lim_sn = ReferenceData(
            Li_files, mag_to_flux_files, metadata.band, metadata.z)

        self.metric = SNSNRMetric(lim_sn=lim_sn, fake_file=fake_file, coadd=coadd,
                                  names_ref=[metadata.names_ref], shift=shift, season=season, z=metadata.z, verbose=metadata.verbose)
        self.saveConfig()


class ObsRateMetricWrapper(MetricWrapper):
    def __init__(self, name='ObsRate', season=-1,
                 coadd=True, fieldType='DD', nside=64,
                 RAmin=0., RAmax=360.,
                 Decmin=-1.0, Decmax=-1.0,
                 npixels=0,
                 metadata={}, outDir='', ebvofMW=-1.0, bluecutoff=380.0, redcutoff=800.0):
        super(ObsRateMetricWrapper, self).__init__(
            name=name, season=season, coadd=coadd, fieldType=fieldType,
            nside=nside, RAmin=RAmin, RAmax=RAmax,
            Decmin=Decmin, Decmax=Decmax,
            npixels=npixels,
            metadata=metadata, outDir=outDir, ebvofMW=ebvofMW)

        self.metaout += ['x1', 'color', 'dirRefs', 'z', 'bands', 'SNR']

        x1 = metadata.x1
        color = metadata.color
        bands = 'gri'
        SNR = [30., 40., 30.]  # WFD SNR cut to estimate sum(Li**2)
        self.metadata['bands'] = bands
        self.metadata['SNR'] = SNR

        Li_files = []
        mag_to_flux_files = []
        for name in [metadata.names_ref]:
            Li_files.append(
                '{}/Li_{}_{}_{}.npy'.format(metadata.dirRefs, name, x1, color))
            mag_to_flux_files.append(
                '{}/Mag_to_Flux_{}.npy'.format(metadata.dirRefs, name))

        # self.bands = bands

        lim_sn = {}
        for band in bands:
            lim_sn[band] = ReferenceData(
                Li_files, mag_to_flux_files, band, metadata.z)

        snr_ref = dict(zip(bands, SNR))

        self.metric = SNObsRateMetric(lim_sn=lim_sn, names_ref=[metadata.names_ref],
                                      coadd=coadd, season=season, z=metadata.z, bands=bands, snr_ref=snr_ref, verbose=metadata.verbose)

        self.saveConfig()


class NSNMetricWrapper(MetricWrapper):
    def __init__(self, name='NSN', season=-1, coadd=True, fieldType='DD',
                 nside=64, RAmin=0., RAmax=360.,
                 Decmin=-1.0, Decmax=-1.0,
                 npixels=0,
                 metadata={}, outDir='', ebvofMW=-1.0, bluecutoff=380.0, redcutoff=800.0, error_model=1):
        super(NSNMetricWrapper, self).__init__(
            name=name, season=season, coadd=coadd, fieldType=fieldType,
            nside=nside, RAmin=RAmin, RAmax=RAmax,
            Decmin=Decmin, Decmax=Decmax,
            npixels=npixels,
            metadata=metadata, outDir=outDir, ebvofMW=ebvofMW)

        zmin = 0.
        zmax = 1.1
        if fieldType == 'WFD':
            zmax = 0.6

        tel_par = {}
        tel_par['name'] = 'LSST'  # name of the telescope (internal)
        # dir of throughput
        tel_par['throughput_dir'] = 'LSST_THROUGHPUTS_BASELINE'
        tel_par['atmos_dir'] = 'THROUGHPUTS_DIR'  # dir of atmos
        tel_par['airmass'] = 1.2  # airmass value
        tel_par['atmos'] = True  # atmos
        tel_par['aerosol'] = False  # aerosol

        self.telescope = Telescope(name=tel_par['name'],
                                   throughput_dir=tel_par['throughput_dir'],
                                   atmos_dir=tel_par['atmos_dir'],
                                   atmos=tel_par['atmos'],
                                   aerosol=tel_par['aerosol'],
                                   airmass=tel_par['airmass'])
        lc_reference = {}

        templateDir = 'Template_LC'
        gammaDir = 'reference_files'
        gammaName = 'gamma.hdf5'
        web_path = 'https://me.lsst.eu/gris/DESC_SN_pipeline'
        # loading dust file
        dustDir = 'Template_Dust'
        dustcorr = {}

        x1_colors = [(-2.0, -0.2), (-2.0, 0.0), (-2.0, 0.2),
                     (0.0, -0.2), (0.0, 0.0), (0.0, 0.2),
                     (2.0, -0.2), (2.0, 0.0), (2.0, 0.2)]
        # (2.0, -0.2)] #(2.0, 0.0), (2.0, 0.2)]

        if metadata.proxy_level == 2:
            x1_colors = [(-2.0, 0.2), (0.0, 0.0)]

        print('Loading reference files')
        result_queue = multiprocessing.Queue()

        wave_cutoff = 'error_model'
        errmodrel = -1.
        if error_model:
            errmodrel = 0.1

        if not error_model:
            wave_cutoff = '{}_{}'.format(bluecutoff, redcutoff)
        for j in range(len(x1_colors)):
            x1 = x1_colors[j][0]
            color = x1_colors[j][1]

            fname = 'LC_{}_{}_{}_ebvofMW_0.0_vstack.hdf5'.format(
                x1, color, wave_cutoff)
            if ebvofMW < 0.:
                dustFile = 'Dust_{}_{}_{}.hdf5'.format(
                    x1, color, wave_cutoff)
                dustcorr[x1_colors[j]] = LoadDust(
                    dustDir, dustFile, web_path).dustcorr
            else:
                dustcorr[x1_colors[j]] = None
            p = multiprocessing.Process(
                name='Subprocess_main-'+str(j), target=self.load, args=(templateDir, fname, gammaDir, gammaName, web_path, j, result_queue))
            p.start()

        resultdict = {}
        for j in range(len(x1_colors)):
            resultdict.update(result_queue.get())

        for p in multiprocessing.active_children():
            p.join()

        for j in range(len(x1_colors)):
            if resultdict[j] is not None:
                lc_reference[x1_colors[j]] = resultdict[j]

        print('Reference data loaded', lc_reference.keys(), fieldType)

        # LC selection criteria

        if fieldType == 'DD':
            n_bef = 4
            n_aft = 10
            snr_min = 1.
            n_phase_min = 1
            n_phase_max = 1
            zlim_coeff = 0.95

        if fieldType == 'WFD':
            n_bef = 4
            n_aft = 10
            snr_min = 0.
            n_phase_min = 1
            n_phase_max = 1
            zlim_coeff = 0.85

        if fieldType == 'Fake':
            n_bef = 0
            n_aft = 0
            snr_min = 0.
            n_phase_min = 0
            n_phase_max = 0
            zlim_coeff = 0.95

        # load x1_color_dist

        fName = 'Dist_x1_color_JLA_high_z.txt'
        fDir = 'reference_files'
        check_get_file(web_path, fDir, fName)
        x1_color_dist = np.genfromtxt('{}/{}'.format(fDir, fName), dtype=None,
                                      names=('x1', 'color', 'weight_x1', 'weight_c', 'weight_tot'))

        # print(x1_color_dist)

        if metadata.proxy_level == 1:
            x1vals = np.arange(-3., 5., 2.)
            cvals = np.arange(-0.3, 0.5, 0.2)

            r = []
            for ix in range(len(x1vals)-1):
                ii = x1_color_dist['x1'] >= x1vals[ix]
                ii &= x1_color_dist['x1'] < x1vals[ix+1]
                x1med = np.median([x1vals[ix], x1vals[ix+1]])
                for ic in range(len(cvals)-1):
                    iib = x1_color_dist['color'] >= cvals[ic]
                    iib &= x1_color_dist['color'] < cvals[ic+1]
                    cmed = np.median([cvals[ic], cvals[ic+1]])
                    print(x1med, np.round(cmed, 1), np.sum(
                        x1_color_dist[ii & iib]['weight_tot']))
                    r.append((np.round(x1med, 1), np.round(cmed, 1),
                              np.sum(x1_color_dist[ii & iib]['weight_tot'])))

            x1_color_dist = np.rec.fromrecords(
                r, names=['x1', 'color', 'weight_tot'])

        pixArea = hp.nside2pixarea(nside, degrees=True)

        # metric instance
        self.metric = SNNSNMetric(
            lc_reference, dustcorr, season=season, zmin=zmin,
            zmax=zmax, pixArea=pixArea,
            verbose=metadata.verbose, timer=metadata.timer,
            ploteffi=metadata.ploteffi,
            n_bef=n_bef, n_aft=n_aft,
            snr_min=snr_min,
            n_phase_min=n_phase_min,
            n_phase_max=n_phase_max,
            errmodrel=errmodrel,
            outputType=metadata.outputType,
            proxy_level=metadata.proxy_level,
            x1_color_dist=x1_color_dist,
            coadd=coadd, lightOutput=metadata.lightOutput,
            T0s=metadata.T0s, zlim_coeff=zlim_coeff, ebvofMW=ebvofMW)

        self.metadata['n_bef'] = n_bef
        self.metadata['n_aft'] = n_aft
        self.metadata['snr_min'] = snr_min
        self.metadata['n_phase_min'] = n_phase_min
        self.metadata['n_phase_max'] = n_phase_max
        self.metadata['zlim_coeff'] = zlim_coeff
        self.metadata['error_model'] = error_model
        self.metadata['errmodrel'] = errmodrel

        self.metaout += ['ploteffi', 'outputType',
                         'proxy_level', 'lightOutput', 'T0s',
                         'n_bef', 'n_aft', 'snr_min', 'n_phase_min', 'n_phase_max', 'error_model', 'errmodrel', 'zlim_coeff']
        self.saveConfig()

    def load(self, templateDir, fname, gammaDir, gammaName, web_path, j=-1, output_q=None):

        lc_ref = GetReference(templateDir,
                              fname, gammaDir, gammaName, web_path, self.telescope)

        if output_q is not None:
            output_q.put({j: lc_ref})
        else:
            return tab_tot


class SaturationMetricWrapper(MetricWrapper):
    def __init__(self, name='Saturation', season=-1, coadd=False, fieldType='DD',
                 nside=64, RAmin=0., RAmax=360.,
                 Decmin=-1.0, Decmax=-1.0,
                 npixels=0,
                 metadata={}, outDir='', ebvofMW=-1.0, bluecutoff=380.0, redcutoff=800.0, error_model=1):
        super(SaturationMetricWrapper, self).__init__(
            name=name, season=season, coadd=coadd, fieldType=fieldType,
            nside=nside, RAmin=RAmin, RAmax=RAmax,
            Decmin=Decmin, Decmax=Decmax,
            npixels=npixels,
            metadata=metadata, outDir=outDir, ebvofMW=ebvofMW)

        zmin = 0.015
        zmax = 0.025

        tel_par = {}
        tel_par['name'] = 'LSST'  # name of the telescope (internal)
        # dir of throughput
        tel_par['throughput_dir'] = 'LSST_THROUGHPUTS_BASELINE'
        tel_par['atmos_dir'] = 'THROUGHPUTS_DIR'  # dir of atmos
        tel_par['airmass'] = 1.2  # airmass value
        tel_par['atmos'] = True  # atmos
        tel_par['aerosol'] = False  # aerosol

        self.telescope = Telescope(name=tel_par['name'],
                                   throughput_dir=tel_par['throughput_dir'],
                                   atmos_dir=tel_par['atmos_dir'],
                                   atmos=tel_par['atmos'],
                                   aerosol=tel_par['aerosol'],
                                   airmass=tel_par['airmass'])
        lc_reference = {}

        templateDir = 'Template_LC'
        refDir = 'reference_files'
        gammaName = 'gamma_extended.hdf5'
        web_path = 'https://me.lsst.eu/gris/DESC_SN_pipeline'
        # loading dust file
        dustDir = 'Template_Dust'
        dustcorr = {}

        x1_colors = [(0.0, 0.0)]

        print('Loading reference files')
        result_queue = multiprocessing.Queue()

        wave_cutoff = 'error_model'
        errmodrel = -1.
        if error_model:
            errmodrel = 0.1

        if not error_model:
            wave_cutoff = '{}_{}'.format(bluecutoff, redcutoff)
        for j in range(len(x1_colors)):
            x1 = x1_colors[j][0]
            color = x1_colors[j][1]

            fname = 'LC_{}_{}_{}_ebvofMW_0.0_vstack.hdf5'.format(
                x1, color, wave_cutoff)
            if ebvofMW < 0.:
                dustFile = 'Dust_{}_{}_{}.hdf5'.format(
                    x1, color, wave_cutoff)
                dustcorr[x1_colors[j]] = LoadDust(
                    dustDir, dustFile, web_path).dustcorr
            else:
                dustcorr[x1_colors[j]] = None
            p = multiprocessing.Process(
                name='Subprocess_main-'+str(j), target=self.load, args=(templateDir, fname, refDir, gammaName, web_path, j, result_queue))
            p.start()

        resultdict = {}
        for j in range(len(x1_colors)):
            resultdict.update(result_queue.get())

        for p in multiprocessing.active_children():
            p.join()

        for j in range(len(x1_colors)):
            if resultdict[j] is not None:
                lc_reference[x1_colors[j]] = resultdict[j]

        print('Reference data loaded', lc_reference.keys(), fieldType)

        # LC selection criteria

        if fieldType == 'DD':
            n_bef = 4
            n_aft = 10
            snr_min = 1.
            n_phase_min = 1
            n_phase_max = 1
            zlim_coeff = 0.95
            coadd = True
            saturationLevel = 0.5

        if fieldType == 'WFD':
            n_bef = 4
            n_aft = 10
            snr_min = 0.
            n_phase_min = 1
            n_phase_max = 1
            zlim_coeff = 0.85
            coadd = False
            saturationLevel = 0.99

        if fieldType == 'Fake':
            n_bef = 0
            n_aft = 0
            snr_min = 0.
            n_phase_min = 0
            n_phase_max = 0
            zlim_coeff = 0.95
            saturationLevel = 0.99

        pixArea = hp.nside2pixarea(nside, degrees=True)

        # load file to estimate saturation here
        fracpixelName = 'PSF_pixel_single_gauss_summary.npy'
        check_get_file(web_path, refDir, fracpixelName)

        fracpixel = np.load('{}/{}'.format(refDir, fracpixelName))

        # metric instance
        self.metric = SNSaturationMetric(
            lc_reference, dustcorr, season=season, zmin=zmin,
            zmax=zmax,
            verbose=metadata.verbose, timer=metadata.timer,
            plotmetric=metadata.ploteffi,
            snr_min=snr_min,
            coadd=coadd, lightOutput=False,
            ebvofMW=ebvofMW,
            fracpixel=fracpixel, saturationLevel=saturationLevel)

        self.metadata['snr_min'] = snr_min
        self.metaout += ['ploteffi', 'lightOutput', 'snr_min']
        self.saveConfig()

    def load(self, templateDir, fname, gammaDir, gammaName, web_path, j=-1, output_q=None):

        lc_ref = GetReference(templateDir,
                              fname, gammaDir, gammaName, web_path, self.telescope)

        if output_q is not None:
            output_q.put({j: lc_ref})
        else:
            return tab_tot


class SLMetricWrapper(MetricWrapper):
    def __init__(self, name='SL', season=-1,
                 coadd=0, nside=64, fieldType='WFD',
                 RAmin=0., RAmax=360.,
                 Decmin=-1.0, Decmax=-1.0,
                 npixels=0,
                 metadata={}, outDir='', ebvofMW=-1.0):
        super(SLMetricWrapper, self).__init__(
            name=name, season=season, coadd=coadd, fieldType=fieldType,
            nside=nside, RAmin=RAmin, RAmax=RAmax,
            Decmin=Decmin, Decmax=Decmax,
            npixels=npixels,
            metadata=metadata, outDir=outDir, ebvofMW=ebvofMW)

        self.metric = SLSNMetric(
            season=season, nside=nside, coadd=coadd, verbose=metadata.verbose)

        self.saveConfig()
