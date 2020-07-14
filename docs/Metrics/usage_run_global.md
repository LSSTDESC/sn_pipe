## Usage: run_global_metric.py [options]

<pre>
Options:
  -h, --help           show this help message and exit
  --dbName=DBNAME      db name [alt_sched]
  --dbExtens=DBEXTENS  db extension [npy]
  --dbDir=DBDIR        db dir []
  --outDir=OUTDIR      output directory []
  --nproc=NPROC        nproc for multiprocessing [1]
</pre>

## Example
 - run the global metric using observations from OS baseline_v1.4_10yrs.db located in ../../DB_Files using 4 procs. Results will be copied in the directory GlobalOutput: 
   - python run_scripts/metrics/run_global_metric.py --dbName baseline_v1.4_10yrs --dbExtens db --dbDir ../../DB_Files --outDir GlobalOutput --nproc 4