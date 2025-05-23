# zlimit for sigmaC <= 0.04

## Data analysis

### Usage: run_scripts/sn_analysis/sn_zlim_sigmaC.py [options]

<pre>
Script to analyze zlim for sigmaC<=0.04

Options:
  -h, --help            show this help message and exit
  --dbDir=DBDIR         OS location dir[../Output_SN_DD_sigmaInt_0.0_Hounsell_
                        z_smflux_notelrot_zfaint_G10_JLA]
  --dbList=DBLIST       OS name [ddf_list.csv]
  --runType=RUNTYPE     run type [DDF_spectroz]
  --timescale=TIMESCALE
                        timescale [season]
  --sigmaC=SIGMAC       sigma color max value for selection
  --outDir=OUTDIR       output directory
  --nproc=NPROC         nproc for multiprocessing

</pre>

## Plots

### Usage: plot_scripts/sn/sn_plot_zlim_sigmaC.py [options]

<pre>

Script to plot zlim for sigmaC<=0.04

Options:
  -h, --help       show this help message and exit
  --dbDir=DBDIR    OS location dir[../sn_zlim_sigmaC]
  --dbList=DBLIST  OS name [config_ana_selplot.csv]

</pre>

[mean zlimit (sigmaC<=0.04) vs season - COSMOS](plotzlim1.png)

[mean zlimit (sigmaC<=0.04) vs season - XMM-LSS](plotzlim2.png)
