SN parameters:  
   &nbsp; X1: -2.0        **# stretch**   
   &nbsp; Color: 0.2   **#color**  
Observations:   
   &nbsp; filename: ../flatiron/maf_local/sims_maf_contrib/tutorials/baseline2018a.db  **# Name of db obs file (full path)**  
   &nbsp; fieldtype: DD **#DD or WFD**  
   &nbsp; coadd: True **# this is the coaddition per night**  
   &nbsp; season: [1] **#season to simulate (-1 = all seasons)**  
   &nbsp; bands: 'riz' **#bands to consider**  
   &nbsp; z: 0.7 **#redshift**  
Pixelisation:  
    &nbsp; nside: 64  
Li file : ['SN_MAF/Reference_Files/Li_SNSim_-2.0_0.2.npy','SN_MAF/Reference_Files/Li_SNCosmo_-2.0_0.2.npy'] **# Flux template files**  
Mag_to_flux file : ['SN_MAF/Reference_Files/Mag_to_Flux_SNSim.npy','SN_MAF/Reference_Files/Mag_to_Flux_SNCosmo.npy']**#mag to flux files**  
names_ref: ['SNSim','SNCosmo'] **# reference names - should fit the two previous items**    
Metric: SN_SNR_Metric **#name of the metric**  
Fake_file: SN_MAF/input/Fake_cadence.yaml **#This is to superimpose 'fake' observations**  
Display_Processing: False **# display SNR estimation while processing**  