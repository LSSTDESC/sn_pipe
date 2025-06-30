#!/bin/bash

#rolling 2 with all reqs satisfied (calib: yearly)
python run_scripts/lsst_ddf_cohesive_strategy/lsst_cohesive_strategy.py --ddf_scenario=input/lsst_ddf_cohesive_strategy/ddf_survey_allreqs_2.csv --survey_type=realistic --survey_output_name=ddf_survey_rolling_2_all_reqs.csv

#rolling 2 with all reqs satisfied except for UDF yearly
python run_scripts/lsst_ddf_cohesive_strategy/lsst_cohesive_strategy.py --ddf_scenario=input/lsst_ddf_cohesive_strategy/ddf_survey_rolling_2.csv --survey_type=realistic --survey_output_name=ddf_survey_rolling_2.csv

#rolling 5 with all reqs satisfied (calib: yearly)
python run_scripts/lsst_ddf_cohesive_strategy/lsst_cohesive_strategy.py --ddf_scenario=input/lsst_ddf_cohesive_strategy/ddf_survey_rolling_5.csv --survey_type=realistic --survey_output_name=ddf_survey_rolling_5_all_reqs.csv

#rolling 5 with all reqs satisfied (calib: 10 years)
python run_scripts/lsst_ddf_cohesive_strategy/lsst_cohesive_strategy.py --ddf_scenario=input/lsst_ddf_cohesive_strategy/ddf_survey_rolling_5.csv --survey_type=science_fiction --survey_output_name=ddf_survey_rolling_5.csv
