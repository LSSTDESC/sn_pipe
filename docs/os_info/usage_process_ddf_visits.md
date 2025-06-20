## Usage: run_scripts/os_info/ddf_visits_night.py [options]

<pre>
Script to study DDF visits on a nightly basis from pointings

Options:
  -h, --help            show this help message and exit
  --dbDir=DBDIR         file directory [../DB_Files]
  --dbList=DBLIST       liof OS to process [dbList.csv]
  --ddf_list=DDF_LIST   list of ddf [DD:COSMOS,DD:ECDFS,DD:EDFS_a,DD:EDFS_b,DD
                        :ELAISS1,DD:XMM_LSS]
  --outDir=OUTDIR       output directory [../ddf_visits_night]
  --nproc=NPROC         nproc for multiprocessing [8]

  </pre>

## example

Running by default python ddf_visits_night.py will generate output files in ../ddf_visits_night corresponding to the processing of the [observing strategies](dbList.csv).