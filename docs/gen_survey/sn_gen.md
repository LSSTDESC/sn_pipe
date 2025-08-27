# LSST SN survey realization

## python run_scripts/cosmology/gen_survey.py

Usage: script to generate LSST SN surveys

<pre>
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
  --outName=OUTNAME     output file name [cosmo_fit]
  --surveyFile=SURVEYFILE
                        survey to use
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
                        to dump surveys on disk [../sn_surveys]
  --timescale=TIMESCALE
                        timescale for the cosmology (year or season) [year]
  --simu_norm_factor=SIMU_NORM_FACTOR
                        norm factors for simu
                        [input/cosmology/simuinfo/normfactor.csv]
  --seasons=SEASONS     seasons to generate sn surveys [1-10]
  --nrandom=NRANDOM     number of random sample (per season/year) to generate
                        [50]
  --nproc=NPROC         number of procs for multiprocessing [8]
  --wfd_tagsurvey=WFD_TAGSURVEY
                        tag for the WFD survey [notag]
  --dd_tagsurvey=DD_TAGSURVEY
                        tag for the DD survey [notag]
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
                        recalc sigma_mu [1]
  --save_full_survey=SAVE_FULL_SURVEY
                        to save the survey with no spectro selection [0]
  --n_random_survey=N_RANDOM_SURVEY
                        number of random surveys to generate [1]
  --analyze_survey=ANALYZE_SURVEY
                        to analyze the survey online [0]

</pre>

## Example

python run_scripts/cosmology/gen_survey.py --dataDir_DD=../Output_SN_DD_sigmaInt_0.0_Hounsell_z_smflux_notelrot_airmass_G10_JLA --dataDir_WFD=../Output_SN_WFD_sigmaInt_0.0_Hounsell_z_smflux_notelrot_airmass_G10_JLA --dbName_DD=baseline_v4.3.1_10yrs --dbName_WFD=baseline_v4.3.1_10yrs --surveyFile=survey_scenario_spectroz_TiDES_10_new.csv --seasons=1-10 --save_full_survey=0 --n_random_survey=50

## Input files

[survey_scenario_spectroz_TiDES_10_new.csv](survey_scenario_spectroz_TiDES_10_new.csv): spectroscopic scenario