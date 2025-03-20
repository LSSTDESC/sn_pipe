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

The input file (../survey_lsst_scheduler/ddf_desc_0.70_sn.npy by default) can be generated using the script [generate_survey.py](usage_generate_survey.md)

The output file (../desc_ddf_deep_rolling_auto/ddf_desc_0.70_sn.npy by default) may be analyzed using the script [plot_scripts/scheduler/ana_ddf_scheduler.py](plot_ana_ddf.md)]