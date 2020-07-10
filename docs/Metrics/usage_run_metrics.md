### Usage: run_metrics.py [options] ###
<pre>
Options:
  -h, --help            show this help message and exit
  --dbName=DBNAME       db name [alt_sched]
  --dbExtens=DBEXTENS   db extension [npy]
  --dbDir=DBDIR         db dir [/sps/lsst/cadence/LSST_SN_CADENCE/cadence_db]
  --outDir=OUTDIR       output dir [MetricOutput]
  --templateDir=TEMPLATEDIR
                        template dir
                        [/sps/lsst/data/dev/pgris/Templates_final_new]
  --nside=NSIDE         healpix nside [64]
  --nproc=NPROC         number of proc  [8]
  --fieldType=FIELDTYPE
                        field type DD or WFD[DD]
  --zmax=ZMAX           zmax for simu [1.2]
  --remove_dithering=REMOVE_DITHERING
                        remove dithering for DDF [0]
  --simuType=SIMUTYPE   flag for new simulations [0]
  --saveData=SAVEDATA   flag to dump data on disk [0]
  --metric=METRIC       metric to process [cadence]
  --coadd=COADD         nightly coaddition [1]
  --RAmin=RAMIN         RA min for obs area - for WDF only[0.0]
  --RAmax=RAMAX         RA max for obs area - for WDF only[360.0]
  --Decmin=DECMIN       Dec min for obs area - for WDF only[-1.0]
  --Decmax=DECMAX       Dec max for obs area - for WDF only[-1.0]
  --proxy_level=PROXY_LEVEL
                        proxy level for the metric[2]
  --T0s=T0S             T0 values to consider[all]
  --lightOutput=LIGHTOUTPUT
                        light LC output[0]
  --outputType=OUTPUTTYPE
                        outputType of the metric[zlims]
  --seasons=SEASONS     seasons to process[-1]
  --verbose=VERBOSE     verbose mode for the metric[0]
  --timer=TIMER         timer mode for the metric[0]
  --ploteffi=PLOTEFFI   plot efficiencies for the metric[0]
  --z=Z                 redshift for the metric[0.3]
  --band=BAND           band for the metric[r]
  --dirRefs=DIRREFS     dir of reference files for the metric[reference_files]
  --dirFake=DIRFAKE     dir of fake files for the metric[input/Fake_cadence]
  --names_ref=NAMES_REF
                        ref name for the ref files for the metric[SNCosmo]
  --x1=X1               Supernova stretch[-2.0]
  --color=COLOR         Supernova color[0.2]
  --pixelmap_dir=PIXELMAP_DIR
                        dir where to find pixel maps[]
  --npixels=NPIXELS     number of pixels to process[0]
  --nclusters=NCLUSTERS
                        number of clusters in data (DD only)[0]
  --radius=RADIUS       radius around clusters (DD and Fakes)[4.0]

</pre>

### Examples ###

 - run the **(nSN, zlim) metric** for DD fieds, OS name=descddf_v1.4_10yrs.db located in ../../DB_Files, nclusters(=nDDF)=5, using 4 procs:
   - python run_scripts/metrics/run_metrics.py --dbDir ../../DB_Files --dbName descddf_v1.4_10yrs --dbExtens npy --fieldType DD --metric NSN --nclusters 5 --nproc 4 
 - run the **(nSN, zlim) metric** for WFD fieds, OS name=descddf_v1.4_10yrs.db located in ../../DB_Files, using 4 procs, on one pixel located in a region defined in RA in [36.,40.] and Dec in [-41.,-37.] 
   - python run_scripts/metrics/run_metrics.py --dbDir ../../DB_Files --dbName descddf_v1.4_10yrs --dbExtens npy --fieldType WFD --metric NSN --nproc 4 --npixels 1 --RAmin 36. --RAmax 40 --Decmin -41. --Decmax -37 --radius 4. --verbose 1 --ebvofMW 0.0
 - run the **cadence metric** for WFD fields, OS name=descddf_v1.5_10yrs.db located in ../../DB_Files, on 10 pixels licated in (RA,Dec): [20,24]x[-20,-24], r-band (default)
   - python run_scripts/metrics/run_metrics.py --dbDir ../../DB_Files --dbName descddf_v1.5_10yrs --dbExtens db --fieldType WFD --metric Cadence --nproc 1 --npixels 10 --RAmin 20. --RAmax 24. --Decmin -24. --Decmax -20 --radius 4. --saveData 1