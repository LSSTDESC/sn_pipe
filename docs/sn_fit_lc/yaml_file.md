# yaml file for fit: parameter definition

The input yaml file contains all the settings to fit SN light curves. It is composed of a dictionnary with key/values given below.

## Instrument (key: Instrument)

| key | value | definition |
|---|---|---|
|name |LSST | name of the instrument|
|throughputDir|LSST_THROUGHPUTS_BASELINE|throughput directory|
|atmosDir|THROUGHPUTS_DIR|instrument atmos dir|
|airmass|1.2|instrument airmass|
|atmos|1|instrument atmos|
|aerosol|0|instrument aerosol|

## Simulations info (key: Simulations)

| key | value | definition |
|---|---|---|
|prodid|prodid|Name of simulation  file|
|dirname|dbDir|dir of LC files|

## Fitter infos (key: Fitter)

| key | value | definition |
|---|---|---|
|name|sn_fitter.fit_sn_cosmo|fitter name: sn_cosmo,sn_fast,...|
|model|salt2-extended|spectra model|
|version|1.0|model version|

## Light curve selection : only selected LC are fitted (key: LCSelection)

| key | value | definition |
|---|---|---|
|snrmin|1.0|min SNR for LC points|
|nbef|4|number of LC points before max|
|naft|10|number of LC points after max|

## Output infos (key: Output)

| key | value | definition |
|---|---|---|
|directory|Output_Fit|Output directory|
|save|1|save file or not|

## mb cov matrix estimate (key: mbcov)
| key | value | definition |
|---|---|---|
|estimate|0|to activate estimation of mbcov|
|directory|SALT2_Files|directory where to find files to estimate mbcov|

## multiprocessing option (key: Multiprocessing)
| key | value | definition |
|---|---|---|
|nproc|1|number of procs to be used for the fitting procedure|

## Miscellaneous

| key | value | definition |
|---|---|---|
|ProductionID| prodid| tag for the production |
|Display|0|to display the fit 'on-line'|
|WebPath|https://me.lsst.eu/gris/DESC_SN_pipeline|web path for reference files|

