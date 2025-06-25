## Usage: run_scripts/os_info/nvisits_m5.py

<pre>

Script to estimate cumulative Nvisits and m5 to check requirements from PZ,WL from pointings

Options:
  -h, --help         show this help message and exit
  --dirFile=DIRFILE  file directory [../DB_Files]
  --dd_list=DD_LIST  OS DD list[ddf_list.csv]
  --fields=FIELDS    DD fields to consider [DD:COSMOS,DD:XMM_LSS,DD:ECDFS,DD:E
                     LAISS1,DD:EDFS_a,DD:EDFS_b]
  --outDir=OUTDIR    output dir [../nvisits_m5]

</pre>

## Example

By default, python run_scripts/os_info/nvisits_m5.py will process OS listed in ddf_list and located in dirFile for all the DDFs. For each OS an output file is created in outDir.