## DD pixels

### Usage: for_batch/scripts/os_pixels/run_dd_pixels.py [options]

<pre>
Script to launch a set of batches for DD pixels

Options:
  -h, --help           show this help message and exit
  --dbList=DBLIST      list of DBs to process [DD_fbs_4.3.1.csv]
  --outDir=OUTDIR      dir where to save data [/sps/lsst/users/gris/dd_pixels]
  --proctime=PROCTIME  max processing time [05:00:00]
  --procmem=PROCMEM    mem for processing [20G]
  --fields=FIELDS      DDF to process [COSMOS,XMM-
                       LSS,CDFS,ELAISS1,EDFS_a,EDFS_b]
  --procmode=PROCMODE  mode of processing: batch/interact [batch]
  --nproc=NPROC        nproc for multiprocessing [8]
  --shDir=SHDIR        dir for sh scripts [sh_scripts]

</pre>

## WFD pixels

### Usage: for_batch/scripts/os_pixels/run_wfd_pixels.py [options]

<pre>
Script to launch a set of batches for WFD pixels

Options:
  -h, --help           show this help message and exit
  --dbList=DBLIST      list of DBs to process [WFD_fbs_4.3.1.csv]
  --outDir=OUTDIR      dir where to save data
                       [/sps/lsst/users/gris/wfd_pixels]
  --proctime=PROCTIME  max processing time [05:00:00]
  --procmem=PROCMEM    mem for processing [20G]
  --procmode=PROCMODE  mode of processing: batch/script_only [batch]
  --nproc=NPROC        nproc for multiprocessing [8]

</pre>