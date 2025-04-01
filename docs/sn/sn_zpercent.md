# z-percentiles estimation

## Data analysis

### Usage: run_scripts/sn_analysis/zpercentile_ddf.py [options]

<pre>
Script to estimate z_0.8 and z_0.9

Options:
  -h, --help            show this help message and exit
  --dbDir=DBDIR         OS location
                        dir[../Output_SN_sigmaInt_0.0_Hounsell_G10_JLA]
  --config=CONFIG       OS DD list[input/plots/config_ana.csv]
  --norm_factor=NORM_FACTOR
                        normalization factor [30]
  --runType=RUNTYPE     run type  [spectroz]
  --timescale=TIMESCALE
                        timescale of the files to process [year]
  --timeslots=TIMESLOTS
                        time slot (season or year) to process [1-10]
  --dataType=DATATYPE   data type [DataFrame]
  --fields=FIELDS       data type [COSMOS,XMM-LSS,CDFS,ELAISS1,EDFS_a,EDFS_b]

</pre>

### Example

python run_scripts/sn_analysis/zpercentile_ddf.py --dbDir=../Output_SN_DD_sigmaInt_0.0_Hounsell_z_smflux_notelrot_G10_JLA --config=[config_zper.csv](config_zper.csv) --fields=COSMOS

--> result: a file zpercentiles_ddf.hdf5

## Plot the results

### Usage: plot_scripts/sn/sn_plot_zpercent_ddf.py [options]

<pre>
Script to plot zpercentiles for DDF

Options:
  -h, --help            show this help message and exit
  --fileName=FILENAME   data to plot [zpercentiles_ddf.hdf5]
  --config=CONFIG       OS DD list[input/plots/config_ana.csv]
  --timescale=TIMESCALE
                        timescale of the files to process [year]
  --fields=FIELDS       data type [COSMOS,XMM-LSS,CDFS,ELAISS1,EDFS_a,EDFS_b]

</pre>

[z_0.8 and z_0.9 vs year - COSMOS](plotd1.png)

[z_0.8 and z_0.9 vs year - XMM-LSS](plotd2.png)