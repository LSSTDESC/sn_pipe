# Usage: plot_scripts/sn/sn_plot_feature_ddf.py [options]

<pre>
Script to plot DDF features

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

python plot_scripts/sn/sn_plot_feature_ddf.py --dbDir=../Output_SN_DD_sigmaInt_0.0_Hounsell_z_smflux_notelrot_G10_JLA --fields=COSMOS --dbName=desc_ddf_gen_0.80_sn_v4.3.1_10yrs

[sigma_mu vs z per year - COSMOS](plotc1.png)

[nsn vs z per year - COSMOS](plotc2.png)

[frac nsn(sigmac<=0.04) vs z per year - COSMOS](plotc3.png)