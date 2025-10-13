## Usage: run_scripts/sim_to_fit/run_sim_to_fit.py [options]

<pre>
Options:
  -h, --help            show this help message and exit
  --fit_remove_sat=FIT_REMOVE_SAT
                        to fit w/o saturated points  [0]
  --dbName=DBNAME       db name [alt_sched]
  --dbExtens=DBEXTENS   db extension [npy]
  --dbDir=DBDIR         db dir  [/sps/lsst/cadence/LSST_SN_CADENCE/cadence_db]
  --templateDir=TEMPLATEDIR
                        template dir
                        [/sps/lsst/data/dev/pgris/Templates_final_new]
  --nproc=NPROC         number of proc [8]
  --nproc_pixels=NPROC_PIXELS
                        number of proc to process pixels [8]
  --remove_dithering=REMOVE_DITHERING
                        remove dithering for DDF [0]
  --simuType=SIMUTYPE   flag for new simulations [0]
  --saveData=SAVEDATA   flag to dump data on disk [0]
  --dirRefs=DIRREFS     dir of reference files for the metric
                        [reference_files]
  --dirFake=DIRFAKE     dir of fake files for the metric [input/Fake_cadence]
  --pixelmap_dir=PIXELMAP_DIR
                        dir where to find pixel maps [None]
  --nclusters=NCLUSTERS
                        number of clusters in data (DD only) [0]
  --radius=RADIUS       radius around clusters (DD and Fakes) [4.0]
  --pixelList=PIXELLIST
                        list of healpixIds to process [None]
  --RAmin=RAMIN         RA min for obs area - for WDF only [0.0]
  --RAmax=RAMAX         RA max for obs area - for WDF only [360.0]
  --Decmin=DECMIN       Dec min for obs area - for WDF only [-90.0]
  --Decmax=DECMAX       Dec max for obs area - for WDF only [40.0]
  --npixels=NPIXELS     number of pixels to process [-1]
  --FoV=FOV             telescope field of view [9.6]
  --telrot=TELROT       telescope rotation angle [0]
  --fp_level=FP_LEVEL   fp level(raft,ccd,sensor) [ccd]
  --display=DISPLAY     display results [0]
  --fieldType=FIELDTYPE
                        field type DD or WFD [DD]
  --nside=NSIDE         healpix nside [64]
  --DD_list=DD_LIST     list of DDFs [COSMOS,CDFS,EDFS,ELAISS1,XMM-LSS]
  --fieldName=FIELDNAME
                        fieldName - for DD only [COSMOS]
  --ebvofMW_pixel=EBVOFMW_PIXEL
                        E(B-V) for pixel processing [-1.0]
  --lookup_ddf=LOOKUP_DDF
                        DDF lookup table [input/simulation/lookup_ddf.csv]
  --noteCol=NOTECOL     obs col to grab field name [target_name]
  --code=CODE           code to use (old/new) [new]
  --ProductionIDSimu=PRODUCTIONIDSIMU
                        Production Id [prodid]
  --SN_Id=SN_ID         SN Id [100]
  --SN_type=SN_TYPE     SN type [SN_Ia]
  --SN_modelPar_nameFile=SN_MODELPAR_NAMEFILE
                        SN dist filename [x1_color_G10.csv]
  --SN_modelPar_dirFile=SN_MODELPAR_DIRFILE
                        SN dist model ref dir [reference_files]
  --SN_modelPar_x1sigma=SN_MODELPAR_X1SIGMA
                        shift for x1 distrib parameter [0]
  --SN_modelPar_colorsigma=SN_MODELPAR_COLORSIGMA
                        shift for color distrib parameter [0]
  --SN_modelPar_mbsigma=SN_MODELPAR_MBSIGMA
                        shift for mb SN parameter [0]
  --SN_modelPar_mbsigmafile=SN_MODELPAR_MBSIGMAFILE
                        file for mbsigma application
                        [sigma_mb_from_simu_Ny_40.hdf5]
  --SN_x1_type=SN_X1_TYPE
                        SN x1 type [unique]
  --SN_x1_min=SN_X1_MIN
                        SN x1 min [-2.0]
  --SN_x1_max=SN_X1_MAX
                        SN x1 max [0.2]
  --SN_x1_step=SN_X1_STEP
                        SN x1 step [0.1]
  --SN_color_type=SN_COLOR_TYPE
                        SN color type [unique]
  --SN_color_min=SN_COLOR_MIN
                        SN color min [0.2]
  --SN_color_max=SN_COLOR_MAX
                        SN color max [1.0]
  --SN_color_step=SN_COLOR_STEP
                        SN color step [0.1]
  --SN_z_type=SN_Z_TYPE
                        SN z type [unique]
  --SN_z_min=SN_Z_MIN   SN z min [0.01]
  --SN_z_max=SN_Z_MAX   SN z max [1.0]
  --SN_z_step=SN_Z_STEP
                        SN z step [0.01]
  --SN_z_rate=SN_Z_RATE
                        SN z rate [combined]
  --SN_z_weight=SN_Z_WEIGHT
                        SN z weight(sn_rate/flat) [sn_rate]
  --SN_z_maxsimu=SN_Z_MAXSIMU
                         SN max z for rate [1.1]
  --SN_z_minsimu=SN_Z_MINSIMU
                         SN max z for rate [0.01]
  --SN_z_sigmaz=SN_Z_SIGMAZ
                         redshift error to be propagated to zsim [1e-05]
  --SN_daymax_type=SN_DAYMAX_TYPE
                        SN daymax type [unique]
  --SN_daymax_step=SN_DAYMAX_STEP
                        SN daymax step [1.0]
  --SN_daymax_restricted=SN_DAYMAX_RESTRICTED
                        to restrict daymax values [0]
  --SN_minRFphase=SN_MINRFPHASE
                        SN min rf phase [-20.0]
  --SN_maxRFphase=SN_MAXRFPHASE
                        SN max rf phase [60.0]
  --SN_minRFphaseQual=SN_MINRFPHASEQUAL
                        SN min rf phase qual [-10.0]
  --SN_maxRFphaseQual=SN_MAXRFPHASEQUAL
                        SN max rf phase qual [35.0]
  --SN_absmag=SN_ABSMAG
                        SN absmag [-19.0906]
  --SN_band=SN_BAND     SN band [bessellB]
  --SN_magsys=SN_MAGSYS
                        SN magsys [vega]
  --SN_differentialFlux=SN_DIFFERENTIALFLUX
                        SN diff flux [0]
  --SN_salt2Dir=SN_SALT2DIR
                        SN SALT2 dir [SALT2_Files]
  --SN_blueCutoffu=SN_BLUECUTOFFU
                        SN blue cutoff u-band [380.0]
  --SN_redCutoffu=SN_REDCUTOFFU
                        SN red cutoff u-band [800.0]
  --SN_blueCutoffg=SN_BLUECUTOFFG
                        SN blue cutoff g-band [380.0]
  --SN_redCutoffg=SN_REDCUTOFFG
                        SN red cutoff g-band [800.0]
  --SN_blueCutoffr=SN_BLUECUTOFFR
                        SN blue cutoff r-band [380.0]
  --SN_redCutoffr=SN_REDCUTOFFR
                        SN red cutoff    r-band [800.0]
  --SN_blueCutoffi=SN_BLUECUTOFFI
                        SN blue cutoff i-band [360.0]
  --SN_redCutoffi=SN_REDCUTOFFI
                        SN red cutoff    i-band [800.0]
  --SN_blueCutoffz=SN_BLUECUTOFFZ
                        SN blue cutoff z-band [380.0]
  --SN_redCutoffz=SN_REDCUTOFFZ
                        SN red cutoff z-band [800.0]
  --SN_blueCutoffy=SN_BLUECUTOFFY
                        SN blue cutoff y-band [380.0]
  --SN_redCutoffy=SN_REDCUTOFFY
                        SN red cutoff y-band [800.0]
  --SN_ebvofMW=SN_EBVOFMW
                        SN E(B-V) [-1.0]
  --SN_NSNfactor=SN_NSNFACTOR
                        NSN*factor for simulation [1]
  --SN_NSNabsolute=SN_NSNABSOLUTE
                        absolute number of SN to produce [-1]
  --SN_sigmaInt=SN_SIGMAINT
                        SN intrinsic dispersion [0.0]
  --SN_nspectra=SN_NSPECTRA
                        Number of spectra to generate [0]
  --SN_smearFlux=SN_SMEARFLUX
                        LC flux smearing [1]
  --SN_alpha=SN_ALPHA   nuisance parameter (alpha) [0.13]
  --SN_beta=SN_BETA     nuisance parameter (beta) [3.1]
  --SN_x0_griddata=SN_X0_GRIDDATA
                        to extract x0 from griddata [0]
  --SN_simuFile=SN_SIMUFILE
                         simulation parameter file [None]
  --Cosmology_Model=COSMOLOGY_MODEL
                        cosmology model [w0waCDM]
  --Cosmology_Om=COSMOLOGY_OM
                        cosmology Omegam [0.3]
  --Cosmology_Ol=COSMOLOGY_OL
                        cosmology Omegall [0.7]
  --Cosmology_H0=COSMOLOGY_H0
                        cosmology H0 [70.0]
  --Cosmology_w0=COSMOLOGY_W0
                        cosmology w0 [-1.0]
  --Cosmology_wa=COSMOLOGY_WA
                        cosmology wa [0.0]
  --InstrumentSimu_name=INSTRUMENTSIMU_NAME
                        instrument name [LSST]
  --InstrumentSimu_telescope_dir=INSTRUMENTSIMU_TELESCOPE_DIR
                        main telescope dir [throughputs]
  --InstrumentSimu_telescope_tag=INSTRUMENTSIMU_TELESCOPE_TAG
                        throughputs tag version [1.9]
  --InstrumentSimu_throughputDir=INSTRUMENTSIMU_THROUGHPUTDIR
                        instrument throughput dir [baseline]
  --InstrumentSimu_atmosDir=INSTRUMENTSIMU_ATMOSDIR
                        instrument atmos dir [atmos]
  --InstrumentSimu_atmosType=INSTRUMENTSIMU_ATMOSTYPE
                        instrument atmos/const/dep [const]
  --InstrumentSimu_airmass=INSTRUMENTSIMU_AIRMASS
                        instrument airmass [1.2]
  --InstrumentSimu_round_airmass=INSTRUMENTSIMU_ROUND_AIRMASS
                        instrument airmass rounding [1]
  --InstrumentSimu_aerosol=INSTRUMENTSIMU_AEROSOL
                        instrument aerosol [0.1]
  --InstrumentSimu_round_aerosol=INSTRUMENTSIMU_ROUND_AEROSOL
                        instrument aerosol rounding [1]
  --InstrumentSimu_pwv=INSTRUMENTSIMU_PWV
                        instrument precipit. water vapor [4.0]
  --InstrumentSimu_round_pwv=INSTRUMENTSIMU_ROUND_PWV
                        instrument precipit. water vapor rounding [1]
  --InstrumentSimu_ozone=INSTRUMENTSIMU_OZONE
                        ozone [300.0]
  --InstrumentSimu_round_ozone=INSTRUMENTSIMU_ROUND_OZONE
                        ozone rounding [1]
  --InstrumentSimu_sigma_aerosol=INSTRUMENTSIMU_SIGMA_AEROSOL
                        instrument aerosol [0.01]
  --InstrumentSimu_sigma_pwv=INSTRUMENTSIMU_SIGMA_PWV
                        instrument precipit. water vapor [0.1]
  --InstrumentSimu_sigma_ozone=INSTRUMENTSIMU_SIGMA_OZONE
                        ozone [3.0]
  --Observations_filename=OBSERVATIONS_FILENAME
                        observation file name [fullDbName]
  --Observations_fieldtype=OBSERVATIONS_FIELDTYPE
                        observations field type [WFD]
  --Observations_fieldname=OBSERVATIONS_FIELDNAME
                        observations field name (DD only) [all]
  --Observations_coadd=OBSERVATIONS_COADD
                        observations coaddition per night [1]
  --Observations_season=OBSERVATIONS_SEASON
                        observations seasons  [-1]
  --Simulator_name=SIMULATOR_NAME
                        simulator name [sn_simulator.sn_cosmo]
  --Simulator_model=SIMULATOR_MODEL
                        simulator model [salt3]
  --Simulator_version=SIMULATOR_VERSION
                        simulator version [2.0]
  --Simulator_errorModel=SIMULATOR_ERRORMODEL
                        simulator error model [0]
  --ReferenceFiles_TemplateDir=REFERENCEFILES_TEMPLATEDIR
                        dir for templates ref files  [Template_LC]
  --ReferenceFiles_GammaDir=REFERENCEFILES_GAMMADIR
                        dir for gamma ref files [reference_files]
  --ReferenceFiles_GammaFile=REFERENCEFILES_GAMMAFILE
                        gamma ref file name [gamma.hdf5]
  --ReferenceFiles_DustCorrDir=REFERENCEFILES_DUSTCORRDIR
                        dir for template dust files [Template_Dust]
  --ReferenceFiles_fluxpixelDir=REFERENCEFILES_FLUXPIXELDIR
                        dir for template dust files [reference_files]
  --Host=HOST           Host [0]
  --Display_LC_display=DISPLAY_LC_DISPLAY
                        display LC [0]
  --Display_LC_time=DISPLAY_LC_TIME
                        display LC persistency time [5.0]
  --OutputSimu_directory=OUTPUTSIMU_DIRECTORY
                        Output directory [Output_Simu]
  --OutputSimu_save=OUTPUTSIMU_SAVE
                        output save file in sn_simulator [1]
  --OutputSimu_savefromwrapper=OUTPUTSIMU_SAVEFROMWRAPPER
                        output save file in simuwrapper [0]
  --OutputSimu_throwempty=OUTPUTSIMU_THROWEMPTY
                        do not save empty LC [1]
  --OutputSimu_throwafterdump=OUTPUTSIMU_THROWAFTERDUMP
                        remove LC after dumping on file [1]
  --OutputSimu_clean=OUTPUTSIMU_CLEAN
                        to clean output dir before processing [1]
  --MultiprocessingSimu_nproc=MULTIPROCESSINGSIMU_NPROC
                        multiprocessing number of procs [1]
  --Pixelisation_nside=PIXELISATION_NSIDE
                        pixelisation nside Healpix [64]
  --WebPathSimu=WEBPATHSIMU
                        web path for reference files
                        [https://me.lsst.eu/gris/DESC_SN_pipeline]
  --saturation_effect=SATURATION_EFFECT
                        to include saturation effects [0]
  --saturation_psf=SATURATION_PSF
                        PSF for saturation effects [single_gauss]
  --saturation_ccdfullwell=SATURATION_CCDFULLWELL
                        ccd full well value  [90000.0]
  --selection_params=SELECTION_PARAMS
                        csv file of LC selection
                        [input/lc_selection/light_curve_selection.csv]
  --nproc_sel=NPROC_SEL
                         number of procs for multiprocessing [8]
  --ProductionIDFit=PRODUCTIONIDFIT
                        Production Id [prodid]
  --InstrumentFit_name=INSTRUMENTFIT_NAME
                        instrument name [LSST]
  --InstrumentFit_telescope_dir=INSTRUMENTFIT_TELESCOPE_DIR
                        main telescope dir [throughputs]
  --InstrumentFit_telescope_tag=INSTRUMENTFIT_TELESCOPE_TAG
                        throughputs tag version [1.9]
  --InstrumentFit_throughputDir=INSTRUMENTFIT_THROUGHPUTDIR
                        instrument throughput dir [baseline]
  --InstrumentFit_atmosDir=INSTRUMENTFIT_ATMOSDIR
                        instrument atmos dir [atmos_new]
  --InstrumentFit_airmass=INSTRUMENTFIT_AIRMASS
                        instrument airmass [1.2]
  --InstrumentFit_aerosol=INSTRUMENTFIT_AEROSOL
                        instrument aerosol [0.0]
  --InstrumentFit_pwv=INSTRUMENTFIT_PWV
                        instrument precipit. water vapor [4.0]
  --InstrumentFit_ozone=INSTRUMENTFIT_OZONE
                        ozone  [300.0]
  --Simulations_prodid=SIMULATIONS_PRODID
                         Name of simulation  file [prodid]
  --Simulations_dirname=SIMULATIONS_DIRNAME
                         dir of LC files [dbDir]
  --Fitter_name=FITTER_NAME
                         fitter name: sncosmo,snfast,...
                        [sn_fitter.fit_sn_cosmo]
  --Fitter_model=FITTER_MODEL
                         spectra model [salt3]
  --Fitter_version=FITTER_VERSION
                        version [2.0]
  --Fitter_parnames=FITTER_PARNAMES
                         parameters to fit [t0,x1,c,x0]
  --Fitter_sigmaz=FITTER_SIGMAZ
                         redshift error for LC fit [1e-05]
  --LCSelection_snrmin=LCSELECTION_SNRMIN
                        min SNR for LC points [1.0]
  --fit_selected=FIT_SELECTED
                         to fit only selected LC [0]
  --fit_coadded=FIT_COADDED
                         to fit coadded lc points [0]
  --Display=DISPLAY     to display fit result 'on-line' [0]
  --OutputFit_directory=OUTPUTFIT_DIRECTORY
                        Output directory [Output_Fit]
  --OutputFit_save=OUTPUTFIT_SAVE
                        output save file [1]
  --MultiprocessingFit_nproc=MULTIPROCESSINGFIT_NPROC
                        multiprocessing number of procs [1]
  --MultiprocessingFit_nmaxBatch=MULTIPROCESSINGFIT_NMAXBATCH
                        max number of LC fit for a sub-proc [50]
  --MultiprocessingFit_timeout=MULTIPROCESSINGFIT_TIMEOUT
                        time out in case of crash [500.0]
  --mbcov_estimate=MBCOV_ESTIMATE
                        to activate estimation of mbcov [0]
  --mbcov_directory=MBCOV_DIRECTORY
                         directory where to find files to estimate mbcov
                        [SALT2_Files]
  --WebPathFit=WEBPATHFIT
                        web path for reference files
                        [https://me.lsst.eu/gris/DESC_SN_pipeline]

</pre>