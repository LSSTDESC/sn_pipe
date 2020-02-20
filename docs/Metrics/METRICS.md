# Installing, running and analyzing the metrics

## Installation of the metrics package

```
python install_sn_pack.py --package=metrics --gitbranch=thebranch
```

## Running the metrics

There are currently four metrics implemented. They can all be processed using the script run_scripts/metrics/run_metrics_fromnpy.py. Arguments used with this script will depend on the metric to run:

| Metric| command for processing|
|----|----|
| Cadence | python run_scripts/metrics/run_metrics_fromnpy.py --dbDir /sps/lsst/cadence/LSST_SN_PhG/cadence_db/fbs_1.4/db --dbName twi_filters_5_v1.4_10yrs --dbExtens db --nproc 5 --nside 128 --simuType 1 --outDir /sps/lsst/users/gris/MetricOutput --fieldType DD --saveData 1 --metric Cadence --coadd 1 --ramin 0.0 --ramax 360.0 --decmin -1.0 --decmax -1.0 |


If you have comments, suggestions or questions, please [write us an issue](https://github.com/LSSTDESC/sn_pipe/issues).


