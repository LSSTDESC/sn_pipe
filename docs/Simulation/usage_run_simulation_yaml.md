### Usage: run_simulation_from_yaml.py [options] ###
<pre>
Options:
  -h, --help            show this help message and exit
  --RAmin=RAMIN         RA min for obs area - for WDF only[0.0]
  --RAmax=RAMAX         RA max for obs area - for WDF only[360.0]
  --Decmin=DECMIN       Dec min for obs area - for WDF only[-1.0]
  --Decmax=DECMAX       Dec max for obs area - for WDF only[-1.0]
  --remove_dithering=REMOVE_DITHERING
                        remove dithering for DDF [0]
  --pixelmap_dir=PIXELMAP_DIR
                        dir where to find pixel maps[]
  --npixels=NPIXELS     number of pixels to process[0]
  --nclusters=NCLUSTERS
                        number of clusters in data (DD only)[0]
  --radius=RADIUS       radius around clusters (DD and Fakes)[4.0]
  --config_yaml=CONFIG_YAML
                        input yaml config file[]
</pre>


### Examples ###
<ul>
<li>  Simulation on one pixel in the sky area [0.,4.]x[-24.,-20] in (RA,Dec) (radius: 4 deg) with the yaml configuration file input/simulation/param_simulation_example.yaml
      <ul>
     <li>python run_scripts/simulation/run_simulation_from_yaml.py --RAmin 0. --RAmax 4. --Decmin -24. --Decmax -20. --npixels 1 --radius 4. --config_yaml input/simulation/param_simulation_example.yaml
   </li>
     </ul>
     </li>
After this processing, three files should be available in the output directory (see Output/directory keyword in the yaml file; by default: Output_Simu):
  <ul>
     <li> prodid.yaml : yaml file with the parameters used for simulation
     <li> LC_prodid_0.hdf5 : simulated light curves (astropy tables)
     <li> Simu_prodid_0.hdf5 : SN simulation parameters (astropy tables) 
     </ul>
 prodid is a tag for the production: it can be chosen by changing using the ProductionID keyword in the yaml file. The "_0" in LC_* and Simu_* files correspond to the proc number in the multiprocessing.
</li>
