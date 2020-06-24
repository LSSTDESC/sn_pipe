### Usage: run_simulation.py [options] ###
<pre>
Options:
  -h, --help            show this help message and exit
  --dbName=DBNAME       db name [descddf_v1.4_10yrs]
  --dbDir=DBDIR         db dir [ /sps/lsst/cadence/LSST_SN_CADENCE/cadence_db]
  --dbExtens=DBEXTENS   db extension [npy]
  --outDir=OUTDIR       output dir [Output_Simu]
  --nside=NSIDE         healpix nside [64]
  --nproc=NPROC         number of proc  [8]
  --diffflux=DIFFFLUX   flag for diff flux[0]
  --season=SEASON       season to process[-1]
  --fieldType=FIELDTYPE
                        field - DD or WFD[DD]
  --x1Type=X1TYPE       x1 type (unique,random,uniform) [unique]
  --x1min=X1MIN         x1 min if x1Type=unique (x1val) or uniform[-2.0]
  --x1max=X1MAX         x1 max - if x1Type=uniform[2.0]
  --x1step=X1STEP       x1 step - if x1Type=uniform[0.1]
  --colorType=COLORTYPE
                        color type (unique,random,uniform) [unique]
  --colormin=COLORMIN   color min if colorType=unique (colorval) or
                        uniform[0.2]
  --colormax=COLORMAX   color max - if colorType=uniform[0.3]
  --colorstep=COLORSTEP
                        color step - if colorType=uniform[0.1]
  --zType=ZTYPE          zcolor type (unique,uniform,random) [uniform]
  --daymaxType=DAYMAXTYPE
                        daymax type (unique,uniform,random) [unique]
  --daymaxstep=DAYMAXSTEP
                        daymax step [1]
  --zmin=ZMIN           min redshift [0.0]
  --zmax=ZMAX           max redshift [1.0]
  --zstep=ZSTEP         max redshift [0.02]
  --saveData=SAVEDATA   to save data [0]
  --nodither=NODITHER   to remove dithering [0]
  --coadd=COADD         to coadd or not[1]
  --RAmin=RAMIN         RA min for obs area - for WDF only[0.0]
  --RAmax=RAMAX         RA max for obs area - for WDF only[360.0]
  --Decmin=DECMIN       Dec min for obs area - for WDF only[-1.0]
  --Decmax=DECMAX       Dec max for obs area - for WDF only[-1.0]
  --remove_dithering=REMOVE_DITHERING          remove dithering for DDF [0]
  --pixelmap_dir=PIXELMAP_DIR   dir where to find pixel maps[]
  --npixels=NPIXELS     number of pixels to process[0]
  --nclusters=NCLUSTERS  number of clusters in data (DD only)[0]
  --radius=RADIUS       radius around clusters (DD and Fakes)[4.0]
  --simulator=SIMULATOR  simulator to use[sn_cosmo]
  --prodid=PRODID       prod id tag[Test]
  --ebvofMW=EBVOFMW     ebvofMW value[-1.0]
  --bluecutoff=BLUECUTOFF  blue cutoff for SN[380.0]
  --redcutoff=REDCUTOFF    red cutoff for SN[800.0]

</pre>

### Examples ###
<ul>
<li>  Simulation using one proc, nside_healpix =64, (x1,color) = (-2.0,0.2), z in range [0.01, 1.] (random), random daymax, on 1 pixel of the sky, for the OS file baseline_v1.4_10yrs.db located in ../../DB_Files, on WFD
      <ul>
     <li>python run_scripts/simulation/run_simulation.py --nproc 1 --nside 64 --x1Type random --x1min -2.0 --x1max -2.0 --colorType random --colormin 0.2 --colormax 0.2 --zType random --zmin 0.01 --zmax 0.6 --zstep 0.01 --daymaxType random --daymaxstep 1. --npixel 1 --dbName baseline_v1.4_10yrs --dbDir ../../DB_Files --dbExtens db --fieldType WFD --RAmin 36. --RAmax 40 --Decmin -41. --Decmax -37 --radius 4. </li>
     </ul>
     </li>

</li>
