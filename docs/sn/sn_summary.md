# Data processing

## Usage: run_scripts/sn_analysis/nsn_ddf.py [options]

<pre>

Script to estimate SN - DDF after selection

Options:
  -h, --help            show this help message and exit
  --dbDir=DBDIR         OS location dir[../Output_SN_DD_sigmaInt_0.0_Hounsell_
                        z_smflux_notelrot_G10_JLA]
  --list_os=LIST_OS     OS DD list[ddf_list.csv]
  --norm_factor=NORM_FACTOR
                        normalization factor [30]
  --runType=RUNTYPE     run type  [spectroz]
  --timescale=TIMESCALE
                        timescale of the files to process [year]
  --timeslots=TIMESLOTS
                        time slot (season or year) to process [1-10]
  --dataType=DATATYPE   data type [DataFrame]
  --nside=NSIDE         nside healpix parameter [128]
  --outputDir=OUTPUTDIR
                        output dir for the file to store [../sn_summary_ddf]
  --fileName=FILENAME   sn file name to draw [sn_summary_ddf.hdf5]

</pre>

# Make figures

## Usage: plot_scripts/sn/sn_analyzer_sel_ddf.py [options]

<pre>

Script to analyze SN - DDF after selection

Options:
  -h, --help            show this help message and exit
  --config=CONFIG       OS DD list[input/plots/config_ana.csv]
  --plots=PLOTS         plots to draw [nsn_all,nsn_ud]
  --print_nsn=PRINT_NSN
                        to print nsn as a latex table [0]
  --ud_fields=UD_FIELDS
                        UD fields to consider [COSMOS,XMM-LSS]
  --dd_fields=DD_FIELDS
                        DD fields to consider [CDFS,ELAISS1,EDFS_a,EDFS_b]
  --inputDir=INPUTDIR   input dir for the file to draw [../sn_summary_ddf]
  --fileName=FILENAME   sn file name to draw [sn_summary_ddf.hdf5]

</pre>

## Example

### python plot_scripts/sn/sn_analyzer_sel_ddf.py --dbDir=../Output_SN_DD_sigmaInt_0.0_Hounsell_z_smflux_notelrot_G10_JLA --config=[config_ana_selplot.csv](config_ana_selplot.csv)

[sum(NSN) vs year - all DDFs](plota1.png)

[cumulative sum(NSN) vs year - all DDFs](plota2.png)

[sum(NSN)(z>=0.8, sigmaC<=0.04) vs year - all DDFs](plota3.png)

[cumulative sum(NSN)(z>=0.8, sigmaC<=0.04) vs year - all DDFs](plota4.png)

[Ratio  NSN(z>=0.8, sigmaC<=0.04)/NSN(z>=0.8) vs year - all DDFs](plota5.png)


[sum(NSN) vs year - COSMOS+XMM-LSS](plota6.png)

[cumulative sum(NSN) vs year - COSMOS+XMM-LSS](plota7.png)

[sum(NSN)(z>=0.8, sigmaC<=0.04) vs year - COSMOS+XMM-LSS](plota8.png)

[cumulative sum(NSN)(z>=0.8, sigmaC<=0.04) vs year - COSMOS+XMM-LSS](plota9.png)

[Ratio  NSN(z>=0.8, sigmaC<=0.04)/NSN(z>=0.8) vs year - COSMOS+XMM-LSS](plota10.png)

[NSN/deg2 (z>=0.8, sigmaC<=0.04)/NSN(z>=0.8) vs year - COSMOS+XMM-LSS](plota11.png)
### the option --print_nsn=1 lead to the printing of (nsn, err_nsn) for each OS/year in latex format.

\begin{table}[!htbp]
\begin{center}
\caption{mycaption}\label{tab:mylabel}
\begin{tabular}{l|c|c|c|c|c|c|c|c|c|c}
\hline
\hline
 year & baseline_v4.3.1_10yrs& desc_ddf_gen_0.70_co_v4.3.1_10yrs& desc_ddf_gen_0.70_sn_v4.3.1_10yrs& desc_ddf_gen_0.70_wz_v4.3.1_10yrs& desc_ddf_gen_0.75_co_v4.3.1_10yrs& desc_ddf_gen_0.75_sn_v4.3.1_10yrs& desc_ddf_gen_0.75_wz_v4.3.1_10yrs& desc_ddf_gen_0.80_co_v4.3.1_10yrs& desc_ddf_gen_0.80_sn_v4.3.1_10yrs& desc_ddf_gen_0.80_wz_v4.3.1_10yrs \\
\hline
1 & 645 \pm 20 & 347 \pm 27 & 325 \pm 25 & 329 \pm 26 & 354 \pm 28 & 357 \pm 28 & 344 \pm 27 & 331 \pm 26 & 326 \pm 25 & 349 \pm 27\\
2 & 2174 \pm 71 & 1333 \pm 53 & 1210 \pm 46 & 1512 \pm 57 & 1362 \pm 53 & 1256 \pm 49 & 1497 \pm 53 & 1463 \pm 51 & 1345 \pm 50 & 1532 \pm 53\\
3 & 2293 \pm 66 & 1473 \pm 53 & 1377 \pm 51 & 1669 \pm 57 & 1484 \pm 51 & 1375 \pm 49 & 1669 \pm 53 & 1544 \pm 51 & 1473 \pm 50 & 1685 \pm 53\\
4 & 1876 \pm 70 & 1270 \pm 45 & 1179 \pm 40 & 1432 \pm 51 & 1302 \pm 43 & 1218 \pm 39 & 1453 \pm 49 & 992 \pm 64 & 846 \pm 63 & 1120 \pm 66\\
5 & 1751 \pm 73 & 1255 \pm 51 & 1107 \pm 42 & 1401 \pm 57 & 760 \pm 60 & 577 \pm 54 & 1045 \pm 67 & 939 \pm 66 & 768 \pm 62 & 1057 \pm 67\\
\hline
1-5 & 8741 \pm 302 & 5681 \pm 232 & 5201 \pm 206 & 6346 \pm 250 & 5264 \pm 238 & 4786 \pm 220 & 6010 \pm 252 & 5271 \pm 260 & 4760 \pm 253 & 5746 \pm 269
\hline
6 & 1671 \pm 73 & 800 \pm 58 & 606 \pm 52 & 1079 \pm 65 & 818 \pm 61 & 603 \pm 52 & 1144 \pm 68 & 991 \pm 65 & 854 \pm 65 & 1098 \pm 67\\
7 & 1943 \pm 78 & 768 \pm 60 & 572 \pm 51 & 1069 \pm 66 & 770 \pm 58 & 568 \pm 51 & 1073 \pm 66 & 970 \pm 66 & 773 \pm 62 & 1009 \pm 63\\
8 & 1641 \pm 74 & 747 \pm 59 & 517 \pm 48 & 1089 \pm 67 & 734 \pm 58 & 499 \pm 46 & 1034 \pm 66 & 924 \pm 65 & 708 \pm 58 & 1016 \pm 65\\
9 & 1954 \pm 75 & 802 \pm 62 & 626 \pm 56 & 1106 \pm 67 & 802 \pm 61 & 621 \pm 56 & 1083 \pm 68 & 946 \pm 66 & 797 \pm 64 & 1075 \pm 67\\
10 & 2015 \pm 76 & 855 \pm 63 & 622 \pm 53 & 1155 \pm 67 & 844 \pm 61 & 607 \pm 52 & 1158 \pm 69 & 1030 \pm 68 & 842 \pm 64 & 1183 \pm 71\\
\hline
1-10 & 17967 \pm 681 & 9656 \pm 536 & 8148 \pm 469 & 11846 \pm 585 & 9233 \pm 540 & 7687 \pm 481 & 11504 \pm 592 & 10134 \pm 591 & 8737 \pm 569 & 11128 \pm 604
\hline
\end{tabular}
\end{center}
\end{table}