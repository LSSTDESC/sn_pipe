from sn_design_dd_survey.wrapper import Data, Nvisits_cadence, Mod_z
from sn_design_dd_survey.budget import DD_Budget
from sn_design_dd_survey.snr import SNR, SNR_plot
from sn_design_dd_survey.signal_bands import RestFrameBands
from sn_design_dd_survey.showvisits import ShowVisits
from sn_design_dd_survey.templateLC import templateLC
from sn_design_dd_survey import plt
from sn_design_dd_survey.snr_m5 import SNR_m5

#from sn_DD_opti.showvisits import GUI_Visits
#from sn_DD_opti.budget import GUI_Budget

import os
import multiprocessing
import pandas as pd
import numpy as np
import glob


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
    def __init__(self, x1=-2.,color=0.2,bands='grizy',dirStudy='dd_design'):

        self.x1 = x1
        self.color = color
        self.bands = bands
        
        self.ch_cr(dirStudy)
        self.dirTemplates = '{}/Templates'.format(dirStudy)
        self.dirSNR_m5 = '{}/SNR_m5'.format(dirStudy)
        self.dirm5 = '{}/m5_files'.format(dirStudy)
        self.dirSNR_combi =  '{}/SNR_combi'.format(dirStudy)

        self.ch_cr(self.dirTemplates)
        self.ch_cr(self.dirSNR_m5)
        self.ch_cr(self.dirSNR_combi)

        
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
                  cadence = 3.):

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

        cutof = self.cutoff(error_model,blue_cutoff, red_cutoff)
        
        templid = '{}_{}_{}_ebv_{}_{}_cad_{}'.format(simulator, self.x1, self.color, ebv, cutof, int(cadence))
        fname = 'LC_{}_0.hdf5'.format(templid)
        cadences = dict(zip(self.bands,[cadence]*len(self.bands)))
        #generate template here
        templateLC(self.x1, self.color, simulator, ebv, bluecutoff, redcutoff,
                   error_model, zmin, zmax, zstep, self.dirTemplates, self.bands, cadences, templid)

    def cutoff(self, error_model,bluecutoff,redcutoff):

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
            outName = '{}/{}'.format(self.dirSNR_m5,lcName.split('.hdf5')[0].replace('LC','SNR_m5'))
            SNR_m5(self.dirTemplates, lcName,'{}.npy'.format(outName),snrmin)

    def data(self, cadence,
             error_model=1,
             bluecutoff=380.,redcutoff=800.,
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
    
        cutof = self.cutoff(error_model,blue_cutoff, red_cutoff)
        lcName = 'LC_{}_{}_{}_ebv_{}_{}_cad_{}_0.hdf5'.format(simulator,self.x1,self.color,ebvofMW,cutof,int(cadence))
        m5Name = '{}/{}'.format(self.dirm5,m5_file)
        return Data(self.dirTemplates, lcName, m5Name, self.x1, self.color,bluecutoff, redcutoff, error_model, bands=self.bands)

    def plot_data(self, data,bluecutoff=380.,redcutoff=800.):
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


    def SNR_combi(self,data,
                  SNR_par = dict(zip(['max', 'step', 'choice'], [70., 1., 'Nvisits'])),
                  zmin=0.1,
                  zmax=1.1,
                  zstep=0.05):

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

        zref = np.round(np.arange(zmin, zmax+zstep,zstep), 2)

        SNR_name = (data.lcName
                    .replace('LC','SNR_m5')
                    .replace('.hdf5','.npy')
                    )
        SNR_m5_file = '{}/{}'.format(self.dirSNR_m5,SNR_name)
        
        snr_calc = SNR(self.dirSNR_combi,data, SNR_par,
                       SNR_m5_file=SNR_m5_file, zref=zref,
                       save_SNR_combi=True, verbose=False, nproc=8)

# Step 1: Load the data needed for analysis
# ----------------------------------------

blue_cutoff = 380.
red_cutoff = 800.
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
cutoff = '{}_{}'.format(blue_cutoff, red_cutoff)
if error_model:
    cutoff = 'error_model'


process = DD_Design_Survey()

# create template LC for cadence = 1 to 4 days

"""
for cadence in range(1,5):
    process.templates(cadence=cadence)
"""
                                  
#process.snr_m5()

# get data corresponding to cadence = 3

data = process.data(cadence=3)

# usefull cross-check plots
#process.plot_data(data)

# estimate SNR combinations
process.SNR_combi(data)



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
#zref = np.round(np.arange(0.1, np.max(data.lc['z'])+0.1, 0.1), 2)
#zref = np.round(np.array([0.75,0.9]), 2)

snr_calc = SNR(SNRDir, data, SNR_par,
               SNR_m5_file=SNR_m5_file, zref=zref,
               save_SNR_combi=False, verbose=False, nproc=8)

# plot the results

# load m5 reference values - here: med value per field per filter per season
# m5_type = 'median_m5_field_filter_season'  # m5 values
plot_snr = True
if plot_snr:
    snrplot = SNR_plot('SNR_files', -2.0, 0.2, 1.0, cutoff, 3., theDir, m5file, 'median_m5_filter')
    snrplot.plotSummary()
    #snrplot.plotSummary_band()
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
#GUI_Visits(nvisits_cadence, cadence=3, dir_config=dir_config)

# this is to display budget vs zlim (and filter allocation)
"""
GUI_Budget(nvisits_cadence,
           nvisits_cadence_season,
           runtype='Nvisits_single', dir_config='.')

plt.show()
"""
