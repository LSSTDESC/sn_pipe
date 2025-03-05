## Usage:  run_scripts/metrics/stat_DD.py

<pre>
Options:
-h, --help            show this help message and exit
--dbList=DBLIST       db name [List.csv]
--outName=OUTNAME     data location dir [Summary_DD_pointings.hdf5]
--save_nightly=SAVE_NIGHTLY
                        to save nightly results[0]
</pre>

where the [List.csv](List.csv) configuration file contains the list of OS to process.

## example
Using run_scripts/metrics/stat_DD.py with no option will generate an output file named Summary_DD_pointings.hdf5 from the processing of Os of the list List.csv.
