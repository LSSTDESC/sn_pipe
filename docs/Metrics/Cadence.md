# Running the cadence metric

``
python run_scripts/metrics/run_metrics_fromnpy.py --dbDir /sps/lsst/cadence/LSST_SN_PhG/cadence_db/fbs_1.4/db --dbName twi_filters_5_v1.4_10yrs --dbExtens db --nproc 5 --nside 128 --simuType 1 --outDir /sps/lsst/users/gris/MetricOutput --fieldType DD --saveData 1 --metric Cadence --coadd 1 --ramin 0.0 --ramax 360.0 --decmin -1.0 --decmax -1.0
``