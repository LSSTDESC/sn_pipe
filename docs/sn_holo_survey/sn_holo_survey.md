# Holo survey design for SNe Ia

Goal: identify a set of Gaia stars close to DDF position. These stars could be observed with AuxTel to measure atmospheric parameters simultaneous to LSST pointings

## Get star catalogs

Two noteboks may be used (notebooks directory)

 ###  gaia_golden_sample.ipynb

### cat_analysis.ipynb

## Match stars with LSST pointings

### run_scripts/sn_holo_survey/holo_survey.py

<pre>
Usage: holo_survey.py [options]

Script build an AuxTel survey

Options:
  -h, --help            show this help message and exit
  --dbDir=DBDIR         Data dir [../DB_Files]
  --dbName=DBNAME       OS name [baseline_v4.3.1_10yrs]
  --nside=NSIDE         Healpix nside parameter [512]
  --deltaRA=DELTARA     RA width around pointing center [10]
  --deltaDec=DELTADEC   Dec width around pointing center [10]
  --fp_level=FP_LEVEL   FP granularity level (ccd,raft,sensor) [raft]
  --targetDir=TARGETDIR
                        targets data dir [~/Bureau]
  --targetFile=TARGETFILE
                        targets data file [gaia_source_file_ddf_v0.parquet]
  --nproc=NPROC         number of procs for multiprocessing [8]
  --show_Plot=SHOW_PLOT
                        to show plot [0]
  --running_mode=RUNNING_MODE
                        running mode all_pointings/mean_pointings
                        [mean_pointings]
  --outDir=OUTDIR       output directory [../sn_holo_survey]
  --outName=OUTNAME     output directory [holo_survey_mean_pointings.hdf5]

</pre>

## Get nearby stars to target

### plot_scripts/holo_survey/ana_holo.py

<pre>

Usage: ana_holo.py [options]

Script to analyze the holo survey

Options:
  -h, --help            show this help message and exit
  --fileDir=FILEDIR     OS file dir [../sn_holo_survey]
  --dbName=DBNAME       OS name [baseline_v4.3.1_10yrs]
  --surveyName=SURVEYNAME
                        survey name [holo_survey_mean_pointings]
  --outDir=OUTDIR       output dir [../sky_map_holo]

</pre>

## Analysis of the survey: get the final list

### plot_scripts/holo_survey/plot_targets.py
<pre>

sage: plot_targets.py [options]

Script to analyse (Gaia) stars matching DDFs

Options:
  -h, --help       show this help message and exit
  --theDir=THEDIR  file directory [../sky_map_holo/baseline_v4.3.1_10yrs]

</pre>