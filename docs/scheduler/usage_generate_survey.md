## script run_scripts/scheduler/generate_survey.py

## Usage: generate_survey.py [options]

<pre>
Script to produce DDF input file to generate DDF tables for the LSST scheduler

Options:
  -h, --help            show this help message and exit
  --inputDir=INPUTDIR   Location dir of input files [input/scheduler]
  --outputDir=OUTPUTDIR
                        output dir of the produced files
                        [../survey_lsst_scheduler]
  --ddf_survey_fields=DDF_SURVEY_FIELDS
                        config file for fields [deep_rolling_survey.csv]
  --ddf_survey_visits=DDF_SURVEY_VISITS
                        config file for visits [ddf_desc_0.70_sn.csv]

</pre>

Two input files are required: a configuration file for the fields (ex: [deep_rolling_survey.csv](deep_rolling_survey.csv)) and a file with the expected number of visits per observing night (ex: [ddf_desc_0.70_sn.csv](ddf_desc_0.70_sn.csv)).

The results of the script is a numpy array saved in a file (with the same name as ddf_survey_visits) located in outputDir.
