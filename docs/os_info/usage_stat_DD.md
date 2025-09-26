## Usage:  run_scripts/metrics/stat_DD.py

<pre>
Options:
-h, --help            show this help message and exit
--dbList=DBLIST       db name [List.csv]
--outName=OUTNAME     data location dir [Summary_DD_pointings.hdf5]
--save_nightly=SAVE_NIGHTLY
                        to save nightly results[0]
 --outDir=OUTDIR       data outputdir [../summary_DD_pointings]
</pre>

where the [List.csv](List.csv) configuration file contains the list of OS to process.

## example
Using run_scripts/metrics/stat_DD.py with no option will generate a set of Summary_DD_pointings.hdf5 files for each of OS of the list List.csv (eg outDir/dbName/Summary_DD_pointings.hdf5).
