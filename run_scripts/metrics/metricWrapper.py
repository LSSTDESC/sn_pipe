import numpy as np
from sn_tools.sn_cadence_tools import ReferenceData
from sn_tools.sn_calcFast import GetReference, LoadDust
from sn_tools.sn_io import check_get_file
import multiprocessing
import healpy as hp
import yaml


class MetricWrapper:
    def __init__(self, **params):

        self.metric = None

        name = params['metric']
        npixels = params['npixels']
        fieldType = params['fieldType']
        nside = params['nside']
        coadd = params['coadd']
        RAmin = params['RAmin']
        RAmax = params['RAmax']
        Decmin = params['Decmin']
        Decmax = params['Decmax']
        npixels = params['npixels']
        ebvofMW = params['ebvofMW']

        metadata = dict(params)

        self.name = '{}Metric_{}_nside_{}_coadd_{}'.format(name, fieldType,
                                                           nside, coadd)
        self.name += '{}_{}_{}_{}_npixels_{}_ebvofMW_{}'.format(RAmin, RAmax,
                                                                Decmin, Decmax,
                                                                npixels,
                                                                ebvofMW)
        # self.metadata = vars(metadata)
        self.metadata = metadata
        # select values to dump
        self.metaout = ['name', 'seasons', 'coadd', 'fieldType',
                        'nside', 'RAmin', 'RAmax', 'Decmin', 'Decmax',
                        'metric', 'Output dir', 'remove_dithering', 'ebvofMW']

        self.metadata['name'] = self.name
        # self.metadata['metric'] = name
        self.metadata['Output dir'] = params['outDir']
        self.outDir = params['outDir']

    def run(self, obs, imulti=-1):
        return self.metric.run(obs, imulti=imulti)

    def saveConfig(self, params={}):
        # ti = dict(zip(self.metaout, [self.metadata[k]
        # for k in self.metaout]))
        if params:
            nameOut = '{}/{}_conf.yaml'.format(self.outDir, self.name)
            #print('Saving configuration file', nameOut)
            with open(nameOut, 'w') as file:
                yaml.dump(params, file)


class CadenceMetricWrapper(MetricWrapper):
    def __init__(self, name='Cadence', season=-1,
                 coadd=True, fieldType='DD', nside=64,
                 RAmin=0., RAmax=360.,
                 Decmin=-1.0, Decmax=-1.0,
                 npixels=0,
                 metadata={}, outDir='', ebvofMW=-1.0, bluecutoff=380.0,
                 redcutoff=800.0):
        super(CadenceMetricWrapper, self).__init__(
            name=name, season=season, coadd=coadd, fieldType=fieldType,
            nside=nside, RAmin=RAmin, RAmax=RAmax,
            Decmin=Decmin, Decmax=Decmax, npixels=npixels, metadata=metadata,
            outDir=outDir, ebvofMW=ebvofMW)

        from sn_metrics.sn_cadence_metric import SNCadenceMetric
        self.metric = SNCadenceMetric(
            coadd=coadd, nside=nside, verbose=metadata.verbose)

        self.saveConfig()


class SNRMetricWrapper(MetricWrapper):
    def __init__(self, name='SNR', season=-1,
                 coadd=True, fieldType='DD', nside=64,
                 RAmin=0., RAmax=360.,
                 Decmin=-1.0, Decmax=-1.0,
                 npixels=0,
                 metadata={}, outDir='', ebvofMW=-1.0,
                 bluecutoff=380.0, redcutoff=800.0):
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

        from sn_metrics.sn_snr_metric import SNSNRMetric
        self.metric = SNSNRMetric(lim_sn=lim_sn, fake_file=fake_file,
                                  coadd=coadd,
                                  names_ref=[metadata.names_ref], shift=shift,
                                  season=season, z=metadata.z,
                                  verbose=metadata.verbose)
        self.saveConfig()


class ObsRateMetricWrapper(MetricWrapper):
    def __init__(self, name='ObsRate', season=-1,
                 coadd=True, fieldType='DD', nside=64,
                 RAmin=0., RAmax=360.,
                 Decmin=-1.0, Decmax=-1.0,
                 npixels=0,
                 metadata={}, outDir='', ebvofMW=-1.0, bluecutoff=380.0,
                 redcutoff=800.0):
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

        from sn_metrics.sn_obsrate_metric import SNObsRateMetric
        self.metric = SNObsRateMetric(lim_sn=lim_sn,
                                      names_ref=[metadata.names_ref],
                                      coadd=coadd, season=season,
                                      z=metadata.z, bands=bands,
                                      snr_ref=snr_ref,
                                      verbose=metadata.verbose)

        self.saveConfig()


class NSNMetricWrapper(MetricWrapper):
    def __init__(self, **params):
        super(NSNMetricWrapper, self).__init__(**params)

        gammaName = params['gamma_file']
        bluecutoff = params['bluecutoff']
        redcutoff = params['redcutoff']
        error_model = params['error_model']
        nside = params['nside']

        lc_reference, dustcorr = load_reference(
            error_model, 0.0, [(-2.0, 0.2), (0.0, 0.0)],
            bluecutoff, redcutoff, gammaName=gammaName)
        pixArea = np.round(hp.nside2pixarea(nside, degrees=True), 3)
        templateLC = None

        if params['ploteffi']:
            templateLC = loadTemplateLC(error_model, 0, x1_colors=[
                                        (-2.0, 0.2), (0.0, 0.0)])

        params['lc_reference'] = lc_reference
        params['dustcorr'] = dustcorr
        params['pixArea'] = float(pixArea)
        params['templateLC'] = templateLC

        print('Reference data loaded',
              lc_reference.keys(), params['fieldType'])

        errmodrel = -1.
        if error_model:
            errmodrel = 0.05

        # metric instance
        from sn_metrics.sn_nsn_metric import SNNSNMetric
        self.metric = SNNSNMetric(**params)

        params.pop('lc_reference')
        params.pop('dustcorr')
        params.pop('templateLC')
        self.saveConfig(params)


class NSNYMetricWrapper(MetricWrapper):
    def __init__(self, **params):
        super(NSNYMetricWrapper, self).__init__(**params)

        gammaName = params['gamma_file']
        error_model = params['error_model']
        sn_model = params['sn_model']
        sn_version = params['sn_version']
        nside = params['nside']
        verbose = params['verbose']

        lc_reference, dustcorr = load_reference(
            error_model, 0.0, [(-2.0, 0.2), (0.0, 0.0)],
            sn_model, sn_version,
            gammaName=gammaName)

        if verbose:
            print('Reference data loaded',
                  lc_reference.keys(), params['fieldType'])

        errmodrel = -1.
        if error_model:
            errmodrel = 0.05

        pixArea = hp.nside2pixarea(nside, degrees=True)
        templateLC = None
        if params['ploteffi'] or params['fig_for_movie']:
            templateLC = loadTemplateLC(error_model, 0, x1_colors=[
                                        (-2.0, 0.2), (0.0, 0.0)])

        params['templateLC'] = templateLC
        params['pixArea'] = float(np.round(pixArea, 4))
        params['lc_reference'] = lc_reference
        params['dustcorr'] = dustcorr

        # loading zps
        zp_airmass = load_zp(params['WebPathSimu'],
                             params['zpDir'],
                             params['zpFile'])
        bands = zp_airmass['band'].tolist()
        slope = zp_airmass['slope'].tolist()
        intercept = zp_airmass['intercept'].tolist()

        params['zp_slope'] = dict(zip(bands, slope))
        params['zp_intercept'] = dict(zip(bands, intercept))

        # metric instance
        from sn_metrics.sn_nsn_yearly_metric import SNNSNYMetric
        self.metric = SNNSNYMetric(**params)

        params.pop('lc_reference')
        params.pop('dustcorr')
        params.pop('templateLC')

        self.saveConfig(params)


class SNRTimeMetricWrapper(MetricWrapper):
    def __init__(self, name='NSN', season=-1, coadd=True, fieldType='DD',
                 nside=64, RAmin=0., RAmax=360.,
                 Decmin=-1.0, Decmax=-1.0,
                 npixels=0,
                 metadata={}, outDir='', ebvofMW=-1.0,
                 bluecutoff=380.0, redcutoff=800.0, error_model=0):
        super(SNRTimeMetricWrapper, self).__init__(
            name=name, season=season, coadd=coadd, fieldType=fieldType,
            nside=nside, RAmin=RAmin, RAmax=RAmax,
            Decmin=Decmin, Decmax=Decmax,
            npixels=npixels,
            metadata=metadata, outDir=outDir, ebvofMW=ebvofMW)

        zmin = 0.01
        zmax = 1.1
        bands = 'grizy'
        fig_for_movie = False
        if fieldType == 'WFD':
            zmin = 0.1
            zmax = 0.50
            bands = 'griz'
            fig_for_movie = False

        self.telescope = telescope_def()

        lc_reference, dustcorr = load_reference(
            error_model, 0.0, [(0.0, 0.0)], bluecutoff, redcutoff)

        print('Reference data loaded', lc_reference.keys(), fieldType)

        print('hello', metadata)
        # LC selection criteria

        if fieldType == 'DD':
            n_bef = 4
            n_aft = 10
            snr_min = 1.
            n_phase_min = 1
            n_phase_max = 1
            zlim_coeff = 0.95

        if fieldType == 'WFD':
            n_bef = 3
            n_aft = 8
            snr_min = 1.
            n_phase_min = 1
            n_phase_max = 1
            zlim_coeff = 0.85

        if fieldType == 'Fake':
            n_bef = 4
            n_aft = 10
            snr_min = 1.
            n_phase_min = 1
            n_phase_max = 1
            zlim_coeff = 0.95

        errmodrel = -1.
        if error_model:
            errmodrel = 0.05

        pixArea = hp.nside2pixarea(nside, degrees=True)

        templateLC = None
        if metadata.ploteffi:
            templateLC = loadTemplateLC(error_model, 0, x1_colors=[
                                        (-2.0, 0.2), (0.0, 0.0)])
        # metric instance
        from sn_metrics.sn_snr_time_metric import SNSNRTIMEMetric
        self.metric = SNSNRTIMEMetric(
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
            coadd=coadd, lightOutput=metadata.lightOutput,
            T0s=metadata.T0s, zlim_coeff=zlim_coeff,
            ebvofMW=ebvofMW, bands=bands,
            fig_for_movie=fig_for_movie,
            templateLC=templateLC, dbName=metadata.dbName)

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


"""
    def load(self, templateDir, fname, gammaDir, gammaName, web_path, j=-1, output_q=None):

        lc_ref = GetReference(templateDir,
                              fname, gammaDir, gammaName, web_path, self.telescope)

        if output_q is not None:
            output_q.put({j: lc_ref})
        else:
            return tab_tot
 """


class SaturationMetricWrapper(MetricWrapper):
    def __init__(self, name='Saturation', season=-1, coadd=False, fieldType='DD',
                 nside=64, RAmin=0., RAmax=360.,
                 Decmin=-1.0, Decmax=-1.0,
                 npixels=0,
                 metadata={}, outDir='', ebvofMW=-1.0, bluecutoff=380.0, redcutoff=800.0, error_model=0):
        super(SaturationMetricWrapper, self).__init__(
            name=name, season=season, coadd=coadd, fieldType=fieldType,
            nside=nside, RAmin=RAmin, RAmax=RAmax,
            Decmin=Decmin, Decmax=Decmax,
            npixels=npixels,
            metadata=metadata, outDir=outDir, ebvofMW=ebvofMW)

        zmin = 0.020
        zmax = 0.021

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
            print('loading', fname)
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
            n_bef = 3
            n_aft = 5
            snr_min = 1.
            n_phase_min = 1
            n_phase_max = 1
            zlim_coeff = 0.85
            coadd = False
            saturationLevel = 0.99

        if fieldType == 'Fake':
            n_bef = 3
            n_aft = 5
            snr_min = 1.
            n_phase_min = 1
            n_phase_max = 1
            zlim_coeff = 0.95
            saturationLevel = 0.99

        pixArea = hp.nside2pixarea(nside, degrees=True)

        # load file to estimate saturation here
        fracpixelName = 'PSF_pixel_single_gauss_summary.npy'
        check_get_file(web_path, refDir, fracpixelName)

        fracpixel = np.load('{}/{}'.format(refDir, fracpixelName))

        # metric instance
        from sn_metrics.sn_saturation_metric import SNSaturationMetric
        self.metric = SNSaturationMetric(
            lc_reference, dustcorr, season=season, zmin=zmin,
            zmax=zmax,
            verbose=metadata.verbose, timer=metadata.timer,
            plotmetric=metadata.ploteffi,
            snr_min=snr_min,
            coadd=coadd, lightOutput=False,
            ebvofMW=ebvofMW,
            fracpixel=fracpixel, saturationLevel=saturationLevel, telescope=self.telescope)

        self.metadata['snr_min'] = snr_min
        self.metaout += ['ploteffi', 'lightOutput', 'snr_min']
        self.saveConfig()

    def load(self, templateDir, fname, gammaDir, gammaName, web_path, j=-1, output_q=None):

        lc_ref = GetReference(templateDir,
                              fname, gammaDir, gammaName, web_path)

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

        from sn_metrics.sn_sl_metric import SLSNMetric
        self.metric = SLSNMetric(
            season=season, nside=nside, coadd=coadd, verbose=metadata.verbose)

        self.saveConfig()


def telescope_def():
    """
    Method to define a telescope


    """
    from sn_tools.sn_telescope import Telescope
    tel_par = {}
    tel_par['name'] = 'LSST'  # name of the telescope (internal)
    # dir of throughput
    tel_par['throughput_dir'] = 'LSST_THROUGHPUTS_BASELINE'
    tel_par['atmos_dir'] = 'THROUGHPUTS_DIR'  # dir of atmos
    tel_par['airmass'] = 1.2  # airmass value
    tel_par['atmos'] = True  # atmos
    tel_par['aerosol'] = False  # aerosol

    telescope = Telescope(name=tel_par['name'],
                          throughput_dir=tel_par['throughput_dir'],
                          atmos_dir=tel_par['atmos_dir'],
                          atmos=tel_par['atmos'],
                          aerosol=tel_par['aerosol'],
                          airmass=tel_par['airmass'])

    return telescope


def load_reference(error_model=1, ebvofMW=-1,
                   x1_colors=[(-2.0, 0.2), (0.0, 0.0)],
                   sn_model='salt3', sn_version='1.0',
                   gammaName='gamma_WFD.hdf5'):
    """
    Method to load reference files (LC, ...)

    Parameters
    ---------------
    error_model: int, opt
     use error_model (1) or not (0) (default: 1)
    ebvofMW: float, opt
      E(B-V) (default -1: loaded from dustmap)
    x1_colors: list(pair(float)), opt
     (x1,color) pairs for template loading (default: [(-2.0, 0.2), (0.0, 0.0)])

    Returns
    -----------
    dict of lc reference

    """
    lc_reference = {}

    templateDir = 'Template_LC'
    gammaDir = 'reference_files'
    web_path = 'https://me.lsst.eu/gris/DESC_SN_pipeline'
    # loading dust file
    dustDir = 'Template_Dust'
    dustcorr = {}

    #print('Loading reference files')
    result_queue = multiprocessing.Queue()

    wave_cutoff = 'error_model'

    if not error_model:
        #wave_cutoff = '{}_{}'.format(bluecutoff, redcutoff)
        wave_cutoff = 'cutoff'

    for j in range(len(x1_colors)):
        x1 = x1_colors[j][0]
        color = x1_colors[j][1]

        fname = 'LC_{}_{}_{}_{}_{}_ebvofMW_0.0_vstack.hdf5'.format(
            x1, color, wave_cutoff, sn_model, sn_version)
        if ebvofMW < 0.:
            dustFile = 'Dust_{}_{}_{}.hdf5'.format(
                x1, color, wave_cutoff)
            dustcorr[x1_colors[j]] = LoadDust(
                dustDir, dustFile, web_path).dustcorr
        else:
            dustcorr[x1_colors[j]] = None
        p = multiprocessing.Process(
            name='Subprocess_main-'+str(j), target=loadFile,
            args=(templateDir, fname, gammaDir, gammaName, web_path,
                  j, result_queue))
        p.start()

    resultdict = {}
    for j in range(len(x1_colors)):
        resultdict.update(result_queue.get())

    for p in multiprocessing.active_children():
        p.join()

    for j in range(len(x1_colors)):
        if resultdict[j] is not None:
            lc_reference[x1_colors[j]] = resultdict[j]

    return lc_reference, dustcorr


def loadFile(templateDir, fname, gammaDir, gammaName, web_path,
             j=-1, output_q=None):

    lc_ref = GetReference(templateDir,
                          fname, gammaDir, gammaName, web_path)

    if output_q is not None:
        output_q.put({j: lc_ref})
    else:
        return lc_ref


def load_x1_color_dist():

    fName = 'Dist_x1_color_JLA_high_z.txt'
    fDir = 'reference_files'
    check_get_file(web_path, fDir, fName)
    x1_color_dist = np.genfromtxt('{}/{}'.format(fDir, fName), dtype=None,
                                  names=('x1', 'color', 'weight_x1',
                                         'weight_c', 'weight_tot'))

    # print(x1_color_dist)

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

    return x1_color_dist


def loadTemplateLC(error_model=1, ebvofMW=-1,
                   x1_colors=[(-2.0, 0.2), (0.0, 0.0)],
                   bluecutoff=380., redcutoff=800.):
    """
    Method to load reference files (LC, ...)

    Parameters
    ---------------
    error_model: int, opt
     use error_model (1) or not (0) (default: 1)
    ebvofMW: float, opt
      E(B-V) (default -1: loaded from dustmap)
    x1_colors: list(pair(float)), opt
     (x1,color) pairs for template loading (default: [(-2.0, 0.2), (0.0, 0.0)])
    bluecutoff: float, opt
      blue cutoff (default: 380.)
    redcutoff: float, opt
      red cutoff (default: 800.)

    Returns
    -----------
    dict of lc reference

    """
    import h5py
    from astropy.table import Table, vstack, Column
    import pandas as pd

    templateDir = 'Template_LC'

    wave_cutoff = 'error_model'

    if not error_model:
        wave_cutoff = '{}_{}'.format(bluecutoff, redcutoff)

    templLC = {}
    for (x1, color) in x1_colors:

        lcName = 'LC_{}_{}_{}_ebvofMW_0.0_vstack.hdf5'.format(
            x1, color, wave_cutoff)
        # Load the file - lc reference
        lcFullName = '{}/{}'.format(templateDir, lcName)
        f = h5py.File(lcFullName, 'r')
        keys = list(f.keys())
        # lc_ref_tot = Table.read(filename, path=keys[0])
        templLC[(x1, color)] = Table.from_pandas(pd.read_hdf(lcFullName))

    return templLC


def load_zp(web_path, templateDir, fName):
    """
    Function to load zp_airmass results

    Parameters
    ----------
    web_path : str
          web path of original files.
    templateDir : str
          location dir
    fName : str
          file name

    Returns
    -------
    res : record array
          array with data.

    """

    from sn_tools.sn_io import check_get_file
    check_get_file(web_path, templateDir, fName)
    fullName = '{}/{}'.format(templateDir, fName)

    res = np.load(fullName)

    return res
