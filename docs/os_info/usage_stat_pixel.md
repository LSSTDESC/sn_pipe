## Usage:

run_scripts/os_info/run_obs_strat_pixels.py

<pre>
Options:
  -h, --help            show this help message and exit
  --dbName=DBNAME       db name [alt_sched]
  --dbExtens=DBEXTENS   db extension [npy]
  --dbDir=DBDIR         db dir  [/sps/lsst/cadence/LSST_SN_CADENCE/cadence_db]
  --templateDir=TEMPLATEDIR
                        template dir
                        [/sps/lsst/data/dev/pgris/Templates_final_new]
  --nproc=NPROC         number of proc [8]
  --nproc_pixels=NPROC_PIXELS
                        number of proc to process pixels [8]
  --remove_dithering=REMOVE_DITHERING
                        remove dithering for DDF [0]
  --simuType=SIMUTYPE   flag for new simulations [0]
  --saveData=SAVEDATA   flag to dump data on disk [0]
  --dirRefs=DIRREFS     dir of reference files for the metric
                        [reference_files]
  --dirFake=DIRFAKE     dir of fake files for the metric [input/Fake_cadence]
  --pixelmap_dir=PIXELMAP_DIR
                        dir where to find pixel maps [None]
  --nclusters=NCLUSTERS
                        number of clusters in data (DD only) [0]
  --radius=RADIUS       radius around clusters (DD and Fakes) [4.0]
  --pixelList=PIXELLIST
                        list of healpixIds to process [None]
  --RAmin=RAMIN         RA min for obs area - for WDF only [0.0]
  --RAmax=RAMAX         RA max for obs area - for WDF only [360.0]
  --Decmin=DECMIN       Dec min for obs area - for WDF only [-90.0]
  --Decmax=DECMAX       Dec max for obs area - for WDF only [40.0]
  --npixels=NPIXELS     number of pixels to process [-1]
  --FoV=FOV             telescope field of view [9.6]
  --telrot=TELROT       telescope rotation angle [0]
  --fp_level=FP_LEVEL   fp level(raft,ccd,sensor) [ccd]
  --display=DISPLAY     display results [0]
  --fieldType=FIELDTYPE
                        field type DD or WFD [DD]
  --nside=NSIDE         healpix nside [64]
  --DD_list=DD_LIST     list of DDFs [COSMOS,CDFS,EDFS,ELAISS1,XMM-LSS]
  --fieldName=FIELDNAME
                        fieldName - for DD only [COSMOS]
  --ebvofMW_pixel=EBVOFMW_PIXEL
                        E(B-V) for pixel processing [-1.0]
  --lookup_ddf=LOOKUP_DDF
                        DDF lookup table [input/simulation/lookup_ddf.csv]
  --code=CODE           code to use (old/new) [new]
  --outDir=OUTDIR        output directory [../test_metric]
  --prodID=PRODID        output name [test]
</pre>

## example

python run_scripts/os_info/run_obs_strat_pixels.py --dbName=baseline_v4.0_10yrs --dbDir=../DB_Files --prodID=baseline_v4.0_10yrs --nside=128

will process the observing strategy baseline_v4.0_10yrs located in the directory ../DB_Files with the healpix parameter nside=128. The output file (pandas df format) is to be found in the ../test_metric directory (default value for the script) and is named baseline_v4.0_10yrs.hdf5.
By default Deep Drilling fields are processed (--fieldType parameter) and only COSMOS (--fieldName parameter) is considered.
