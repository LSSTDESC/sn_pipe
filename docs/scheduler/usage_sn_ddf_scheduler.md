## Usage: sn_ddf_scheduler.py [options]

<pre>
Script to estimate the schedule of a field

Options:
  -h, --help            show this help message and exit
  --target_list=TARGET_LIST
                        list of targets [input/scheduler/ddf.csv]
  --mjd_min=MJD_MIN     survey start [60980]
  --num_years=NUM_YEARS
                        number of years [10]
  --year_length=YEAR_LENGTH
                        year length [days] [365]
  --outDir=OUTDIR       output dir for the results [../sn_ddf_scheduler]
  --sun_alt_night=SUN_ALT_NIGHT
                        max sun alt (in deg.) for a night to be defined
                        [-18.0]
  --alt_min=ALT_MIN     min star alt (in deg.) for observation [25.0]
  --alt_max=ALT_MAX     max star alt (in deg.) for observation [86.5]
  --airmass_max=AIRMASS_MAX
                        max airmass for observation [2.5]
  --nproc=NPROC         nproc for multiprocessing [8]
  --nseasons=NSEASONS   number of seasons [10]

<pre>

## example

Running the script by default will lead to the generation of a set of pandas df (located in ../sn_ddf_scheduler) with scheduler infos (day, mjd, night_duration, ...) corresponding to the selection criteria as given by the script (by default: sun_alt_night < -18 deg, 25 < alt_star < 86.5, airmass < 2.5) 