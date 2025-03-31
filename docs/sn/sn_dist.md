# Usage: plot_scripts/sn/sn_plot_dist_ddf.py [options]

<pre>
Script to plot nsn vs pixel dist

Options:
  -h, --help            show this help message and exit
  --dbDir=DBDIR         OS location
                        dir[../Output_SN_sigmaInt_0.0_Hounsell_G10_JLA]
  --dbName=DBNAME       OS to process[baseline_v4.3.1_10yrs]
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

# Example

python plot_scripts/sn/sn_plot_dist_ddf.py --dbDir=../Output_SN_DD_sigmaInt_0.0_Hounsell_z_smflux_notelrot_G10_JLA --fields=COSMOS --dbName=desc_ddf_gen_0.70_sn_v4.3.1_10yrs

[NSN vs dist](plotb1.png)

[NSN(z>=0.8) vs dist](plotb2.png)

[NSN(z>=0.8,sigmaC<= 0.04) vs dist](plotb3.png)