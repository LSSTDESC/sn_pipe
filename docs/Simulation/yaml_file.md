# yaml file for simulation: parameter definition

The input yaml file contains all the settings to run the simulation. It is composed of a dictionnary with key/values given below.

## Supernovae parameters (key: SN parameters)
 
| keya | keyb | value | definition |
|---|---|---|---|
| Id |   | 100 | Id of the first SN | 
| x1_color | | | stretch and color distribution |  
| | rate | JLA | | type of distribution chosen |  
| |dirFile | 'reference_files' | location dir of the files |  
|x1 | | | SN x1 |
| | type | unique | parameter choice: unique/random/uniform |
|| min | 0.0 | min x1 value |
|| max | 2.0 | max x1 value |
|| step | 0.1 | x1 step value |
|color | | | SN color |
| | type | unique | parameter choice: unique/random/uniform |
|| min | 0.0 | min color value |
|| max | 2.0 | max color value |
|| step | 0.1 | color step value |
|z | | | SN redshift |
| | type | unique | parameter choice: unique/random/uniform |
|| min | 0.0 | min z value |
|| max | 2.0 | max z value |
|| step | 0.1 | z step value |
|| rate |Perrett | Type Ia volumetric rate : Perrett, Ripoche, Dilday |
|daymax | | | max lumi day |
| | type | unique | parameter choice: unique/random/uniform |
|| step | 1. | daymax step value [day] |
|min_rf_phase ||  -20.  | obs min phase (rest frame)|
|max_rf_phase ||  60.  |obs max phase (rest frame)
|min_rf_phase_qual || -15. |obs min phase (rest frame) (qual cuts)|
|max_rf_phase_qual || 45. |obs max phase (rest frame) (qual cuts)|
|absmag ||  -19.0906|peak abs mag |
|band || bessellB|band for absmag|
|magsys || vega |magsys for absmag |
|differential_flux|| 0 | to estimate differential flux (0/1)|
|salt2Dir ||  SALT2_Files |dir where SALT2 files are located |
|blue_cutoff || 380. | blue cutoff value [nm]|
|red_cutoff ||  800. | red cutoff value [nm] |
|ebvofMW || -1.0 | E(B-V) for dust: if -1.0: taken from a dust map | 
|NSN factor || 1 | scale factor for production |

## Cosmology (key: Cosmology)

|key | value | definition |
|---|---|---|
|Model | w0waCDM |Cosmological model |
|Omega_m | 0.30 |Omega_m |
| Omega_l | 0.70 |Omega_l|
| H0 | 72.0 | H0 |
|w0|  -1.0 | w0 |
|wa | 0.0  | wa |

## Telescope (key: Instrument)

|key | value | definition |
|---|---|---|
|name | LSST |name of the telescope (internal) |
|throughput_dir | LSST_THROUGHPUTS_BASELINE  |dir of throughput |
|atmos_dir | THROUGHPUTS_DIR  | dir of atmos |
|airmass | 1.2   | airmass value |
|atmos | True  |atmos |
|aerosol | False  |aerosol |

## Observations (key: Observations)
|key | value | definition |
|---|---|---|
|filename|/sps/lsst/cadence/LSST_SN_PhG/cadence_db/opsim_db/kraken_2026.db | Name of db obs file (full path) |
| fieldtype | WFD |  field type (DD or WFD) |
|coadd | 1 | coaddition per night (0=no/1=yes)
|season | [3,4] | season to simulate (-1 = all seasons) |

## Simulator (key: Simulator)

|keya | keyb | value | definition |
|---|---|---|---|
|name || sn_simulator.sn_cosmo  |Simulator name: sn_cosmo,sn_fast
|model | |salt2-extended  | spectra model |
|version | |1.0 | version |
|Template Dir || Template_LC |loc. dir of LC templates (sn_fast)|
|Gamma Dir | |reference_files |loc. dir of gamma files ||
|Gamma File | |gamma.hdf5 |gamma file name|
|DustCorr Dir ||Template_Dust |loc. dir of dust templates (sn_fast)|
|Host Parameters ||None  |Host parameters |
|Display_LC  ||| display during LC simulations|
||display | False | display (True) or not (False) |
|| time | 1 | display during time (sec) before closing|

## Generic parameters

| keya | keyb | value | definition |
|---|---|---| ---|
| ProductionID | | prodid | production Id |
| Output ||| Output infos |
||directory | outputDir | output directory (full path) |
||save |  True | to copy data on disk (True) or not (False) |
|Multiprocessing ||| multiprocessing infos |
|| nproc | 3 | number of proc chosen |
|Pixelisation ||| sky pixelisation |
||nside | 64 | healpix nside value |
|Web path |  | https://me.lsst.eu/gris/DESC_SN_pipeline | url where files needed to perform simulations are located|