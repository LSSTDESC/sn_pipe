## Usage: ddf_visits_night.py [options]

<pre>
Script to study DDF visits on a nightly basis from pointings

Options:
  -h, --help            show this help message and exit
  --dbDir=DBDIR         file directory [../DB_Files]
  --dbName=DBNAME       OS name [desc_ddf_v4.2.1_10yrs]
  --ddf_list=DDF_LIST   list of ddf [DD:COSMOS,DD:ECDFS,DD:EDFS_a,DD:EDFS_b,DD
                        :ELAISS1,DD:XMM_LSS]
  --configDir=CONFIGDIR
                        input config dir [input/scheduler]
  --configName=CONFIGNAME
                        input config name [ddf_desc_0.70_sn]
  --outDir=OUTDIR       output directory [../ddf_visits_night]
  </pre>

## example

Running by default python ddf_visits_night.py will generate an output file in ../ddf_visits_night named desc_ddf_v4.2.1_10yrs.hdf5.