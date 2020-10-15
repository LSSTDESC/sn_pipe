### Usage: run_simulation.py [options] ###
<pre>
Options:
   -h, --help            show this help message and exit
  --dbName=DBNAME       db name [descddf_v1.4_10yrs]
  --dbDir=DBDIR         db dir [ /sps/lsst/cadence/LSST_SN_CADENCE/cadence_db]
  --dbExtens=DBEXTENS   db extension [npy]
  --nodither=NODITHER   to remove dithering [0]
  --RAmin=RAMIN         RA min for obs area - for WDF only[0.0]
  --RAmax=RAMAX         RA max for obs area - for WDF only[360.0]
  --Decmin=DECMIN       Dec min for obs area - for WDF only[-1.0]
  --Decmax=DECMAX       Dec max for obs area - for WDF only[-1.0]
  --remove_dithering=REMOVE_DITHERING
                        remove dithering for DDF [0]
  --pixelmap_dir=PIXELMAP_DIR
                        dir where to find pixel maps[]
  --npixels=NPIXELS     number of pixels to process[-1]
  --nclusters=NCLUSTERS
                        number of clusters in data (DD only)[0]
  --radius=RADIUS       radius around clusters (DD and Fakes)[4.0]
  --nproc=NPROC         number of procs to run[1]
  --ProductionID=PRODUCTIONID
                        Production Id [prodid]
  --SN_Id=SN_ID         SN Id [100]
  --SN_type=SN_TYPE     SN type [SN_Ia]
  --SN_modelPar_name=SN_MODELPAR_NAME
                        SN dist model par name [x1_color]
  --SN_modelPar_rate=SN_MODELPAR_RATE
                        SN dist model rate name [JLA]
  --SN_modelPar_dirFile=SN_MODELPAR_DIRFILE
                        SN dist model ref dir [reference_files]
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
                        SN z rate [Perrett]
  --SN_daymax_type=SN_DAYMAX_TYPE
                        SN daymax type [unique]
  --SN_daymax_step=SN_DAYMAX_STEP
                        SN daymax step [1.0]
  --SN_minRFphase=SN_MINRFPHASE
                        SN min rf phase [-20.0]
  --SN_maxRFphase=SN_MAXRFPHASE
                        SN max rf phase [60.0]
  --SN_minRFphaseQual=SN_MINRFPHASEQUAL
                        SN min rf phase qual [-15.0]
  --SN_maxRFphaseQual=SN_MAXRFPHASEQUAL
                        SN max rf phase qual [45.0]
  --SN_absmag=SN_ABSMAG
                        SN absmag [-19.0906]
  --SN_band=SN_BAND     SN band [bessellB]
  --SN_magsys=SN_MAGSYS
                        SN magsys [vega]
  --SN_differentialFlux=SN_DIFFERENTIALFLUX
                        SN diff flux [0]
  --SN_salt2Dir=SN_SALT2DIR
                        SN SALT2 dir [SALT2_Files]
  --SN_blueCutoff=SN_BLUECUTOFF
                        SN blue cutoff [380.0]
  --SN_redCutoff=SN_REDCUTOFF
                        SN red cutoff [800.0]
  --SN_ebvofMW=SN_EBVOFMW
                        SN E(B-V) [-1.0]
  --SN_NSNfactor=SN_NSNFACTOR
                        NSN*factor for simulation [1]
  --Cosmology_Model=COSMOLOGY_MODEL
                        cosmology model [w0waCDM]
  --Cosmology_Om=COSMOLOGY_OM
                        cosmology Omegam [0.3]
  --Cosmology_Ol=COSMOLOGY_OL
                        cosmology Omegall [0.7]
  --Cosmology_H0=COSMOLOGY_H0
                        cosmology H0 [72.0]
  --Cosmology_w0=COSMOLOGY_W0
                        cosmology w0 [-1.0]
  --Cosmology_wa=COSMOLOGY_WA
                        cosmology wa [0.0]
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
  --Observations_filename=OBSERVATIONS_FILENAME
                        observation file name [fullDbName]
  --Observations_fieldtype=OBSERVATIONS_FIELDTYPE
                        observations field type [WFD]
  --Observations_fieldname=OBSERVATIONS_FIELDNAME
                        observations field name (DD only) [all]
  --Observations_coadd=OBSERVATIONS_COADD
                        observations coaddition per night [1]
  --Observations_season=OBSERVATIONS_SEASON
                        observations seasons [-1]
  --Simulator_name=SIMULATOR_NAME
                        simulator name [sn_simulator.sn_cosmo]
  --Simulator_model=SIMULATOR_MODEL
                        simulator model [salt2-extended]
  --Simulator_version=SIMULATOR_VERSION
                        simulator version [1.0]
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
  --Host=HOST           Host [0]
  --Display_LC_display=DISPLAY_LC_DISPLAY
                        display LC [0]
  --Display_LC_time=DISPLAY_LC_TIME
                        display LC persistency time [5.0]
  --Output_directory=OUTPUT_DIRECTORY
                        Output directory [Output_Simu]
  --Output_save=OUTPUT_SAVE
                        output save file [1]
  --Multiprocessing_nproc=MULTIPROCESSING_NPROC
                        multiprocessing number of procs [1]
  --Pixelisation_nside=PIXELISATION_NSIDE
                        pixelisation nside Healpix [64]
  --WebPath=WEBPATH     web path for reference files
                        [https://me.lsst.eu/gris/DESC_SN_pipeline]

</pre>

### Examples ###
<ul>
<li>  Simulation using one proc, nside_healpix =64, (x1,color) = (-2.0,0.2), z in range [0.01, 1.] (random), random daymax, on 1 pixel of the sky, for the OS file descddf_v1.4_10yrs.db located in ../../DB_Files, on WFD
      <ul>
     <li>python run_scripts/simulation/run_simulation.py  --Multiprocessing_nproc 1 --Pixelisation_nside 64 --SN_x1_type random --SN_x1_min -2.0 --SN_x1_max -2.0 --SN_color_type random --SN_color_min 0.2 --SN_color_max 0.2 --SN_z_type random --SN_z_min 0.01 --SN_z_max 0.6 --SN_z_step 0.01 --SN_daymax_type random --SN_daymax_step 1. --npixels 1 --dbName descddf_v1.4_10yrs --dbDir ../../DB_Files --dbExtens db --Observations_fieldtype WFD --RAmin 36. --RAmax 40 --Decmin -41. --Decmax -37 --radius 4. </li>
     </ul>
     </li>

</li>
