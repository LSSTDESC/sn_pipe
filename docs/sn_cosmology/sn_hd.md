# Dark Energy parameters estimation from a Hubble diagram fit

## Cosmology from SNe Ia production

## Usage: run_scripts/cosmology/cosmology.py

<pre>

Script to fit cosmology parameters

Options:
  -h, --help            show this help message and exit
  --dataDir_DD=DATADIR_DD
                        data dir DD
                        [../Output_SN_sigmaInt_0.0_Hounsell_G10_JLA]
  --dbName_DD=DBNAME_DD
                        db name DD [DDF_Univ_WZ]
  --dataDir_WFD=DATADIR_WFD
                        data dir WFD
                        [../Output_SN_WFD_sigmaInt_0.0_Hounsell_G10_JLA]
  --dbName_WFD=DBNAME_WFD
                        db name WFD [baseline_v3.0_10yrs]
  --selconfig=SELCONFIG
                        sel config [G10_JLA]
  --outDir=OUTDIR       output dir [../cosmo_fit]
  --outName=OUTNAME     output file name [cosmo_fit]
  --survey=SURVEY       survey to use
                        [input/cosmology/scenarios/survey_scenario.csv]
  --hosteffiDir=HOSTEFFIDIR
                        host effi dir [input/cosmology/host_effi]
  --footprintDir=FOOTPRINTDIR
                        footprint dir [input/cosmology/footprints]
  --max_sigma_mu=MAX_SIGMA_MU
                        Max sigma_mu defining low sigma_mu sample [0.12]
  --test_mode=TEST_MODE
                        test mode run [0]
  --plot_test=PLOT_TEST
                        test mode run+of the program and plots [0]
  --low_z_optimize=LOW_Z_OPTIMIZE
                        maximize the lowz sample [1]
  --sigmaInt=SIGMAINT   sigmaInt for SN [0.12]
  --surveyDir=SURVEYDIR
                        to dump surveys on disk [None]
  --timescale=TIMESCALE
                        timescale for the cosmology (year or season) [year]
  --fields_for_stat=FIELDS_FOR_STAT
                        field list for stat (fit) [COSMOS,XMM-
                        LSS,ELAISS1,CDFS,EDFSa,EDFSb]
  --simu_norm_factor=SIMU_NORM_FACTOR
                        norm factors for simu
                        [input/cosmology/simuinfo/normfactor.csv]
  --seasons_cosmo=SEASONS_COSMO
                        seasons to estimate cosmology params [1-10]
  --nrandom=NRANDOM     number of random sample (per season/year) to generate
                        [50]
  --nproc=NPROC         number of procs to use [8]
  --wfd_tagsurvey=WFD_TAGSURVEY
                        tag for the WFD survey [notag]
  --dd_tagsurvey=DD_TAGSURVEY
                        tag for the DD survey [notag]
  --fitparam_names=FITPARAM_NAMES
                        fit parameter names [w0,wa,Om0]
  --fitparam_values=FITPARAM_VALUES
                        fit parameter init values [-1.0,0.0,0.3]
  --prior=PRIOR         prior for the fit [1]
  --select_DDF=SELECT_DDF
                        to select DDF SN for the fit [0]
  --select_WFD=SELECT_WFD
                        to select WFD SN for the fit [0]
  --H0=H0               cosmo par [70.0]
  --Om0=OM0             cosmo par [0.3]
  --Ode0=ODE0           cosmo par [0.7]
  --w0=W0               cosmo par [-1.0]
  --wa=WA               cosmo par [0.0]
  --alpha=ALPHA         nuisance par [0.13]
  --beta=BETA           nuisance par [3.1]
  --recalc_sigmu=RECALC_SIGMU
                        recalc sigma_mu [0]

</pre>

## Cosmology from SNe Ia LSST surveys

This step requires to generate SNe Ia LSST surveys as describe [here](docs/gen_survey/sn_gen_survey.md).

### Interactive estimation

#### script to fit cosmology parameters on SNe Ia surveys

### run_scripts/cosmology/cosmology_survey.py

### Usage: script to fit cosmology parameters on SNe Ia surveys

<pre>
Options:
  -h, --help            show this help message and exit
  --fitparam_names=FITPARAM_NAMES
                        fit parameter names [w0,wa,Om0]
  --fitparam_values=FITPARAM_VALUES
                        fit parameter init values [-1.0,0.0,0.3]
  --prior=PRIOR         prior for the fit [1]
  --H0=H0               cosmo par [70.0]
  --Om0=OM0             cosmo par [0.3]
  --Ode0=ODE0           cosmo par [0.7]
  --w0=W0               cosmo par [-1.0]
  --wa=WA               cosmo par [0.0]
  --alpha=ALPHA         nuisance par [0.13]
  --beta=BETA           nuisance par [3.1]
  --recalc_sigmu=RECALC_SIGMU
                        recalc sigma_mu [1]
  --prior_varname=PRIOR_VARNAME
                        prior varname list [Om0]
  --prior_refvalue=PRIOR_REFVALUE
                        prior refvalue list [0.3]
  --prior_sigma=PRIOR_SIGMA
                        prior sigma list [0.0073]
  --dataDir=DATADIR     SN data directory [../sn_surveys]
  --dbName_DD=DBNAME_DD
                        OS to consider - DDF [baseline_v4.3.1_10yrs]
  --dbName_WFD=DBNAME_WFD
                        OS to consider - WFD [baseline_v4.3.1_10yrs]
  --yearmax=YEARMAX     year max for cosmology measurements [10]
  --nproc=NPROC         number of procs for multiprocessing [8]
  --outDir=OUTDIR       output directory [../cosmo_fit_test]

</pre>

### Batch estimation

#### [Usage] for_batch/scripts/cosmo/loop_cosmofit_survey.py --help

#### script to fit LSST SN surveys

<pre>
Options:
  -h, --help            show this help message and exit
  --dbList=DBLIST       db list to process [list_OS_new_wfd.csv]
  --outDir=OUTDIR       output directory [/sps/lsst/users/gris/cosmo_fit_last]
  --dataDir=DATADIR     input directory for survey
                        files[/sps/lsst/users/gris/sn_surveys]
  --fitparam_names=FITPARAM_NAMES
                        fit parameter names [w0,wa,Om0]
  --fitparam_values=FITPARAM_VALUES
                        fit parameter values [-1.0,0.0,0.3]
  --prior=PRIOR         prior for the fit [1]
  --prior_varname=PRIOR_VARNAME
                        prior varname list
  --prior_refvalue=PRIOR_REFVALUE
                        prior refvalue list
  --prior_sigma=PRIOR_SIGMA
                        prior sigma list

</pre>