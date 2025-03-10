## script plot_scripts/scheduler/ana_ddf_scheduler.py

## Usage: ana_ddf_scheduler.py [options]

<pre>
Script to analyze the file produced for the LSST scheduler

Options:
  -h, --help            show this help message and exit
  --dirFiles=DIRFILES   Location dir of the ddf obs file
                        [../desc_ddf_deep_rolling]
  --ddf_survey=DDF_SURVEY
                        survey to analyze [ddf_desc_0.70_sn]
  --mjd_min=MJD_MIN     survey start [60980]

</pre>

## Example

The following plots are displayed by default.

[season length vs season](plotb1.png)

[cadence of observation vs season](plotb2.png)
