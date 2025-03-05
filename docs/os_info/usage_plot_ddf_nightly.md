## Usage: plot_ddf_nightly.py [options]

<pre>
Script to plot nightly filter dist for DDFs

Options:
  -h, --help         show this help message and exit
  --dbDir=DBDIR      file directory [../DB_Files]
  --dbName=DBNAME    OS name [desc_ddf_v4.2.1_10yrs]
  --fields=FIELDS    list of fields [DD:COSMOS,DD:XMM_LSS,DD:ECDFS,DD:ELAISS1,
                     DD:EDFS_a,DD:EDFS_b]
  --colors=COLORS    colors  [k,r,b,m,orange,purple]
  --colName=COLNAME  name to tag DDFs  [target_name]
 </pre>

## example

Choosing the night number 28 after running 'python plot_ddf_nightly.py' will lead to the plot of the DDF filter allocation corresponding to the night 28 for the obsreving strategy desc_ddf_v4.2.1_10yrs (located in ../DB_Files).

[filter allocation - night 28 - DDF](plotc1.png)