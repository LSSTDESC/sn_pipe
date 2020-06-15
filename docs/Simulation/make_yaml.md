### Usage: make_yaml.py [options] ###
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
  --simulator=SIMULATOR
                        simulator to use[sn_cosmo]
  --prodid=PRODID       prodid of the simulation [Test]
</pre>