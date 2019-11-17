import numpy as np
from sn_metrics.sn_snr_metric import SNSNRMetric
from sn_metrics.sn_cadence_metric import SNCadenceMetric
from sn_metrics.sn_snrrate_metric import SNSNRRateMetric
from sn_metrics.sn_nsn_metric import SNNSNMetric
from sn_metrics.sn_sl_metric import SLSNMetric
from sn_tools.sn_cadence_tools import ReferenceData
from sn_tools.sn_utils import GetReference
import os
import multiprocessing

class CadenceMetricWrapper:
    def __init__(self, season=-1, coadd=True, fieldType='DD',nside=64,ramin=0.,ramax=360.,decmin=-1.0,decmax=-1.0):

        self.metric = SNCadenceMetric(coadd=coadd,nside=nside)
        self.name = 'CadenceMetric_{}_nside_{}_coadd_{}_{}_{}_{}_{}'.format(fieldType,nside,coadd,ramin,ramax,decmin,decmax)

    def run(self,obs,filterCol='filter'):

        return self.metric.run(obs)


class SNRMetricWrapper:
    def __init__(self, z=0.2, x1=-2.0, color=0.2, names_ref=['SNCosmo'], coadd=False, dirfiles='reference_files', dirFakes='input/Fake_cadence', shift=10., season=-1,nside=64,band='g',fieldType='WFD',ramin=0.,ramax=360.,decmin=-1.0,decmax=-1.0):

        self.z = z
        self.coadd = coadd
        self.names_ref = names_ref
        self.season = season
        self.shift = shift
        self.name = 'SNR{}Metric_{}_nside_{}_coadd_{}_{}_{}_{}_{}'.format(band,fieldType,nside,coadd,ramin,ramax,decmin,decmax)
        self.fake_file = '{}/{}.yaml'.format(dirFakes, 'Fake_cadence')
        self.band = band
       
        Li_files = []
        mag_to_flux_files = []
        for name in names_ref:
            Li_files.append(
                '{}/Li_{}_{}_{}.npy'.format(dirfiles, name, x1, color))
            mag_to_flux_files.append(
                '{}/Mag_to_Flux_{}.npy'.format(dirfiles, name))

        bands = 'griz'
        self.lim_sn = {}
        for band in bands:
            self.lim_sn[band] = ReferenceData(
                Li_files, mag_to_flux_files, band, z)

    def run(self, obs, filterCol='filter'):
        idx = obs[filterCol] == self.band
        sel = obs[idx]
        metric = SNSNRMetric(lim_sn=self.lim_sn[self.band], fake_file=self.fake_file, coadd=self.coadd,
                             names_ref=self.names_ref, shift=self.shift, season=self.season, z=self.z)
        return metric.run(np.copy(sel))


class SNRRateMetricWrapper:
    def __init__(self, z=0.3, x1=-2.0, color=0.2, names_ref=['SNCosmo'],fieldType='WFD',nside=64,coadd=False, dirfiles='reference_files', season=-1, bands='gri',ramin=0.,ramax=360.,decmin=-1.0,decmax=-1.0):

        
        self.z = z
        self.coadd = coadd
        self.names_ref = names_ref
        self.season = season
        self.name = 'SNRRateMetric_{}_nside_{}_coadd_{}_{}_{}_{}_{}'.format(fieldType,nside,coadd,ramin,ramax,decmin,decmax)
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
    def __init__(self, fieldType='DD', nside=64, pixArea=9.6, season=-1, templateDir='', ploteffi=False, verbose=False, coadd=0,outputType='zlims', proxy_level=0,ramin=0.,ramax=360.,decmin=-1.0,decmax=-1.0):

        zmax = 1.3
        if fieldType == 'WFD':
            zmax = 0.6

        self.name = 'NSNMetric_{}_{}_nside_{}_coadd_{}_{}_{}_{}_{}'.format(fieldType,outputType,nside,coadd,ramin,ramax,decmin,decmax)
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
                     #(2.0, -0.2)] #(2.0, 0.0), (2.0, 0.2)]
                     
        if proxy_level >0:
            x1_colors = [(-2.0, 0.2), (0.0, 0.0)]

        print('Loading reference files')
        result_queue = multiprocessing.Queue()
       
        for j in range(len(x1_colors)):
            x1 = x1_colors[j][0]
            color = x1_colors[j][1]
            fname = '{}/LC_{}_{}_vstack.hdf5'.format(templateDir, x1, color)
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
                
        print('Reference data loaded',lc_reference.keys())

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

        x1_color_dist = np.genfromtxt('reference_files/Dist_X1_Color_JLA_high_z.txt',dtype=None,
                                      names=('x1','color','weight_x1','weight_x1','weight_tot'))
        

        # metric instance

        self.metric = SNNSNMetric(
            lc_reference, season=season, zmax=zmax, pixArea=pixArea, verbose=verbose, ploteffi=ploteffi, N_bef=N_bef, N_aft=N_aft, snr_min=snr_min, N_phase_min=N_phase_min, N_phase_max=N_phase_max,outputType=outputType, proxy_level=proxy_level,x1_color_dist=x1_color_dist,coadd=coadd)

    def run(self, obs):
        return self.metric.run(obs)

    def load(self,fname,j=-1, output_q=None):

        lc_ref = GetReference(
                fname, self.gamma_reference, self.Instrument)
        
        if output_q is not None:
            output_q.put({j: lc_ref})
        else:
            return tab_tot

class SLMetricWrapper:
    def __init__(self, season=-1, coadd=0,nside=64,fieldType='WFD',ramin=0.,ramax=360.,decmin=-1.0,decmax=-1.0):

        #self.name = 'SLMetric_{}'.format(fieldType)
        self.name = 'SLMetric_{}_nside_{}_coadd_{}_{}_{}_{}_{}'.format(fieldType,nside,coadd,ramin,ramax,decmin,decmax)
        self.metric = SLSNMetric(season=season, nside=nside, coadd=coadd)

    def run(self, obs):
        return self.metric.run(obs)
