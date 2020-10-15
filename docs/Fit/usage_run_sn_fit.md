### Usage: run_sn_fit.py [options] ###
<pre>

Options:
  -h, --help            show this help message and exit
  --ProductionID=PRODUCTIONID
                        Production Id [prodid]
  --Instrument_name=INSTRUMENT_NAME
                        instrument name [LSST]
  --Instrument_throughputDir=INSTRUMENT_THROUGHPUTDIR
                        instrument throughput dir [LSST_THROUGHPUTS_BASELINE]
  --Instrument_atmosDir=INSTRUMENT_ATMOSDIR
                        instrument atmos dir [THROUGHPUTS_DIR]
  --Instrument_airmass=INSTRUMENT_AIRMASS
                        instrument airmass [1.2]
  --Instrument_atmos=INSTRUMENT_ATMOS
                        instrument atmos [1]
  --Instrument_aerosol=INSTRUMENT_AEROSOL
                        instrument aerosol [0]
  --Simulations_prodid=SIMULATIONS_PRODID
                         Name of simulation  file [prodid]
  --Simulations_dirname=SIMULATIONS_DIRNAME
                         dir of LC files [dbDir]
  --Fitter_name=FITTER_NAME
                         fitter name: sncosmo,snfast,...
                        [sn_fitter.fit_sn_cosmo]
  --Fitter_model=FITTER_MODEL
                         spectra model [salt2-extended]
  --Fitter_version=FITTER_VERSION
                        version [1.0]
  --LCSelection_snrmin=LCSELECTION_SNRMIN
                        min SNR for LC points [1.0]
  --LCSelection_nbef=LCSELECTION_NBEF
                        number of LC points before max [4]
  --LCSelection_naft=LCSELECTION_NAFT
                        number of LC points after max [10]
  --Display=DISPLAY     to display fit result 'on-line' [0]
  --Output_directory=OUTPUT_DIRECTORY
                        Output directory [Output_Fit]
  --Output_save=OUTPUT_SAVE
                        output save file [1]
  --Multiprocessing_nproc=MULTIPROCESSING_NPROC
                        multiprocessing number of procs [1]
  --mbcov_estimate=MBCOV_ESTIMATE
                        to activate estimation of mbcov [0]
  --mbcov_directory=MBCOV_DIRECTORY
                         directory where to find files to estimate mbcov
                        [SALT2_Files]
  --WebPath=WEBPATH     web path for reference files
                        [https://me.lsst.eu/gris/DESC_SN_pipeline]

</pre>

## Example
<ul>
<li> Let us suppose that some simulations have been run using the sn_simulation package. In the output simulation directory, three files should be available: prodid.yaml, Simu_prodid_num.hdf5, LC_prodid_num.hdf5, where prodid has been defined in the yaml file used for the simulation. num is an integer corresponding to the multiprocessing configuration used at the simulation stage.
<li> ls Output_Simu yields to
prodid.yaml
LC_prodid_0.hdf5
Simu_prodid_0.hdf5

<li> In that case, the fit of the light curves will be made using:
     <ul>
     <li> python run_scripts/fit_sn/run_sn_fit.py --Simulations_prodid prodid --Simulations_dirname Output_Simu
     </ul>
All other parameters are the default ones.

<li> Output directory after processing:
<ul>
<li> ls Output_Fit
<li> Fit_prodid.yaml : parameters used for the fit
<li>Fit_prodid.hdf5 : SN with fitted parameters (astropy tables)
</ul>
</ul>