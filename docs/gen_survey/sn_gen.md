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
                        survey to use [input/cosmology/scenarios/survey_scenar
                        io_spectroz_TiDES_5.csv]
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
  --deparams=DEPARAMS   cosmo param names [w0,wa]
  --devalues=DEVALUES   cosmo param values [-1.,0.]
  --declass=DECLASS     DE class to use (w0waCDM/DDE_FLRW) [w0waCDM]
  --classloc=CLASSLOC   DE class
                        location(astropy.cosmology/sn_tools.sn_cosmo_model)
                        [astropy.cosmology]
  --demodel=DEMODEL     DE eos model name [CPL]
  --deeos=DEEOS         DE eos model [w0+wa*z/(1+z)]
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

## Batch production
It is possible to launch a batch production (for example: at ccin2p3) using the following script

Usage: python for_batch/scripts/cosmo/loop_gen_survey.py [options]

<pre>
Options:
  -h, --help            show this help message and exit
  --dataDir_DD=DATADIR_DD
                         DD data dir [/sps/lsst/users/gris/DD_dir]
  --dataDir_WFD=DATADIR_WFD
                         WFD data dir [/sps/lsst/users/gris/WFD_dir]
  --surveyFile=SURVEYFILE
                        survey file [input/cosmology/scenarios/survey_scenario
                        _spectroz_TiDES_5.csv]
  --seasons=SEASONS      seasons/years to consider [1-10]
  --save_full_survey=SAVE_FULL_SURVEY
                         to save the full survey [0]
  --n_random_survey=N_RANDOM_SURVEY
                         number of random surveys to generate [50]
  --select_WFD=SELECT_WFD
                         to select WFD SNe Ia [1]
  --surveyDir=SURVEYDIR
                         output directory [/sps/lsst/users/gris/sn_surveys]
  --low_z_optimize=LOW_Z_OPTIMIZE
                         to buil an optimize low-z WFD sample [0]
  --dbList=DBLIST        list of db to process [list_OS_new_wfd.csv]
  --tagName=TAGNAME      tag for the script [taga]
  --H0=H0               cosmo par [70.0]
  --Om0=OM0             cosmo par [0.3]
  --Ode0=ODE0           cosmo par [0.7]
  --deparams=DEPARAMS   cosmo param names [w0,wa]
  --devalues=DEVALUES   cosmo param values [-1.,0.]
  --declass=DECLASS     DE class to use (w0waCDM/DDE_FLRW) [w0waCDM]
  --classloc=CLASSLOC   DE class
                        location(astropy.cosmology/sn_tools.sn_cosmo_model)
                        [astropy.cosmology]
  --demodel=DEMODEL     DE eos model name [CPL]
  --deeos=DEEOS         DE eos model [w0+wa*z/(1+z)]

</pre>