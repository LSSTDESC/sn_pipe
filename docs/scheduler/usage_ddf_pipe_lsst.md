# Build DDF scheduler tables using sn_pipe tools and LSST scheduler tools

The LSST scheduler takes as input a table of observation times for the DDF which corresponds to the list of mjds where the fields are observable. It also takes into account observing conditions (such as fivesigma depth values per band/mjd).
We have adapted the set of scripts from Peter Yoachim to produce DDF tables corresponding to deep rolling surveys, a set of scenarios proposed in  Philippe Gris et al 2024 ApJS 275 21.

The DDF tables are generated in two steps:

[Build a realistic survey from sn_pipe scheduler files](usage_build_realistic.md)
This scripts takes into account the moon phase to define properly the bands to observe.

[Generate the DDF tables for the scheduler](usage_sn_lsst_tables.md)
Will generate a table of DDF observations in the correct format for the LSST scheduler.

[Analysis of the produced file](plot_ana_ddf.md)
Analyze the file produced for the LSST scheduler