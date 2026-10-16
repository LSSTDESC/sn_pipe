## Data production: match ref stars with LSST pointings

### Usage: run_scripts/os_calib/add_star_ref.py [options]

<pre>
add (smeared) atmos. params to OS strategy

Options:
  -h, --help            show this help message and exit
  --dbDir=DBDIR         OS dir [../DB_Files]
  --dbName=DBNAME       OS name [baseline_v5.3.0_10yrs]
  --dbExtens=DBEXTENS   OS file extens [npy]
  --star_calib_file=STAR_CALIB_FILE
                        star used for atmos calib
                        [input/scheduler/calspec_v2.csv]
  --outDir=OUTDIR       output directory [../DB_Files_star_calib]

</pre>

## Analysis

Usage: plot_scripts/os_calib/plot_star_calib.py [options]

<pre>
figures related to obs and calib stars

Options:
  -h, --help           show this help message and exit
  --dataDir=DATADIR    data dir [../DB_Files_star_calib]
  --dbName=DBNAME      OS name [baseline_v5.3.0_10yrs_star_calib]
  --dbExtens=DBEXTENS  OS extens  [npy]

</pre>