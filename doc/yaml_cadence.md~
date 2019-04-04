ProductionID: DD_baseline2018a
SN parameters:
   X1: -2.0        ![#f03c15](# stretch)
   Color: 0.2   ![#f03c15](#color)
Observations: 
   filename: ../flatiron/maf_local/sims_maf_contrib/tutorials/baseline2018a.db  # Name of db obs file (full path)
   fieldtype: DD #DD or WFD
   coadd: True # this is the coaddition per night
   season: -1 #season to simulate (-1 = all seasons)
   bands: 'riz' #bands to consider
   #bands: 'griz'
   SNR: [25., 25., 30., 35.] #DDF SNR cut to estimate sum(Li**2)
   #SNR:  [30., 40., 30., 20.]#WFD SNR cut to estimate sum(Li**2)
   mag_range: [23., 27.5] #DDF mag range
   #mag_range: [21., 25.5] #WFD mag range
   dt_range: [0.5, 25.] # DDF dt range
   #dt_range: [0.5, 30.]  # WFD dt range
Pixelisation:
    nside: 64
#Li file : ['SN_MAF/Reference_Files/Li_SNSim_-2.0_0.2.npy']
#Mag_to_flux file : ['SN_MAF/Reference_Files/Mag_to_Flux_SNSim.npy']
#names_ref: ['SNSim']
#Li file : ['SN_MAF/Reference_Files/Li_SNCosmo_-2.0_0.2.npy']
#Mag_to_flux file : ['SN_MAF/Reference_Files/Mag_to_Flux_SNCosmo.npy']
#names_ref: ['SNCosmo']
Li file : ['SN_MAF/Reference_Files/Li_SNSim_-2.0_0.2.npy','SN_MAF/Reference_Files/Li_SNCosmo_-2.0_0.2.npy']
Mag_to_flux file : ['SN_MAF/Reference_Files/Mag_to_Flux_SNSim.npy','SN_MAF/Reference_Files/Mag_to_Flux_SNCosmo.npy']
names_ref: ['SNSim','SNCosmo']
Metric: SN_Cadence_Metric