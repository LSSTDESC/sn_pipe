## Usage: plot_scripts/os_info/plot_ddf_visits_night.py [options]

<pre>
Script to analyse DDF visits on a nightly basis from pointings

Options:
  -h, --help       show this help message and exit
  --dbDir=DBDIR    file directory [../ddf_visits_night]
  --dbName=DBNAME  OS name [desc_ddf_v4.2.1_10yrs]

</pre>

##example
python plot_scripts/os_info/plot_ddf_visits_night.py will lead to a set of figures corresponding to the file ../ddf_visits_night/desc_ddf_v4.2.1_10yrs.hdf5:

[Nnights with nvisits(expected)/nvisits(simulated)=1 per field](plotb1.png)

[Nnights with nvisits(expected)/nvisits(simulated)<1 per field](plotb2.png)

[Nnights with nvisits(expected)/nvisits(simulated)>1 per field](plotb3.png)

[Nnights with nvisits(expected)/nvisits(simulated)<1 per band](plotb4.png)

[DDF obs. time and NDDF observed vs night](plotb5.png)