## script run_scripts/scheduler/build_realistic_ddf_survey.py

## Usage: build_realistic_ddf_survey.py [options]

<pre>
Script to build a realistic ddf survey

Options:
  -h, --help            show this help message and exit
  --dirFiles=DIRFILES   Location dir of the ddf summary file
                        [../sn_ddf_scheduler]
  --ddf_rubin_scheduler_data=DDF_RUBIN_SCHEDULER_DATA
                        Location dir of the ddf summary file
                        [../../rubin_sim_data/scheduler/ddf_grid.npz]
  --mjd_min=MJD_MIN     survey start [60980]
  --inputconfigDir=INPUTCONFIGDIR
                        input config dir of the survey [input/scheduler]
  --ddf_survey=DDF_SURVEY
                        survey to implement [ddf_desc_0.70_sn]
  --udf=UDF             ultra-deep fields [COSMOS,XMM_LSS]
  --ddf=DDF             deep fields [ELAISS1,ECDFS,EDFS_a,EDFS_b]
  --outDir=OUTDIR       output dir of the produced files
                        [../observations_sn_scheduler]

</pre>