Usage: star_alt_night.py [options]

<pre>
Script to estimate the schedule of a set of fields

Options:
  -h, --help            show this help message and exit
  --target_list=TARGET_LIST
                        list of targets [input/scheduler/ddf.csv]
  --year=YEAR           year of observation [2025]
  --month=MONTH         month of observation [11]
  --day=DAY             day of observation [1]

</pre>

where the [input/scheduler/ddf.csv](ddf.csv) configuration file contains the list of targets (ra,dec)

## example
The run python star_alt_night.py with default parameters will lead to the display of the DDFs alt vs time for november 1st, 2025:

[star alt vs time](star_alt.png)
