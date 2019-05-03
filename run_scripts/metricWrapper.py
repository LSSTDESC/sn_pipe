import numpy as np
from sn_metrics.sn_snr_metric import SNSNRMetric
from sn_metrics.sn_cadence_metric import SNCadenceMetric
from sn_tools.sn_cadence_tools import ReferenceData
import os

class CadenceMetricWrapper:
    def __init__(self,season=-1,coadd=False):
        
        self.metric = SNCadenceMetric(coadd=coadd)
        self.name = 'CadenceMetric'


    def run(self, band, obs, filterCol='filter'):
        idx = obs[filterCol] == band
        sel = obs[idx]
        return self.metric.run(sel)

class SNRMetricWrapper:
    def __init__(self,z=0.3, x1=-2.0, color=0.2,names_ref=['SNCosmo'], coadd=False, dirfiles='reference_files', shift=10., season=-1):
 
        self.z = z
        self.coadd = coadd
        self.names_ref = names_ref
        self.season = season
        self.shift = shift
        self.name = 'SNRMetric'
 
        Li_files = []
        mag_to_flux_files = []
        for name in names_ref:
            Li_files.append('{}/Li_{}_{}_{}.npy'.format(dirfiles,name,x1,color))
            mag_to_flux_files.append('{}/Mag_to_Flux_{}.npy'.format(dirfiles,name))

        bands = 'grizy'
        self.lim_sn = {}
        for band in bands:
            self.lim_sn[band] = ReferenceData(
                Li_files,mag_to_flux_files,band, z)
        
    def run(self, band, obs, filterCol='filter'):
        idx = obs[filterCol] == band
        sel = obs[idx]
        metric = SNSNRMetric(lim_sn=self.lim_sn[band], coadd=self.coadd, names_ref=self.names_ref,shift=self.shift,season=self.season,z=self.z)
        return metric.run(np.copy(sel))

    
