## script run_scripts/scheduler/ddf_rubin_scheduler_auto.py

## Usage: ddf_rubin_scheduler_auto.py [options]

<pre>
Script to produce DDF observations for the LSST scheduler

Options:
  -h, --help            show this help message and exit
  --inputDir=INPUTDIR   Location dir of input files [../survey_lsst_scheduler]
  --outputDir=OUTPUTDIR
                        output dir of the produced files
                        [../desc_ddf_deep_rolling_auto]
  --survey=SURVEY       config file for visits [ddf_desc_0.70_sn.npy]

</pre>

The input file (../survey_lsst_scheduler/ddf_desc_0.70_sn.npy as default) is can be generated using the script [generate_survey.py](usage_generate_survey.cmd)