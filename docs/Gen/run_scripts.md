# run_scripts 

Set of scripts to run the Survey Strategy Support pipeline

 * fakes: to generate fake data and process through the pipeline
   * full_simulation_fit.py
   * make_fake.py
   * sigmaC_z.py
 * fit_sn: to fit supernovae light curves
   * [run_sn_fit.py](../Fit/usage_run_sn_fit.md)
 * io_transform: to perform some io transformation 
   * extract.py
 * obs_pixelize: to associate (simulated) observations to pixels in the sky
   * gime_npixels.py
   * run_obs_to_pixels.py
 * postprocessing
   * convert_to_npy.py
   * Loop_convert.py
 * simulation: to perform supernova simulation
   * [run_simulation.py](../Simulation/usage_run_simulation.md)
   * run_simulation_MAF.py
   * [make_yaml.py](../Simulation/make_yaml.md)
   * simuWrapper.py
 * sn_studies: some studies around supernovae
   * check_snr.py
   * run_design_survey.py
   * run_sn_reshuffle.py
   * snr_m5.py
 * templates: to produce template LC files (used by the fast simulator)
   * run_diffflux.py
   * run_simulation_template.py
 * utils
   * add_DD_Fields.py
   * calcGamma.py
   * reshuffling.py
 * visu_cadence: to visualize observing strategy output
   * run_visu_cadence.py
 * metrics: to process metrics through the pipeline
     * config_metric_template.yaml
     * estimate_DDFrac.py
     * metricWrapper.py
     * run_cadence_metric.py
     * run_global_metric.py
     * [run_metrics.py](../Metrics/usage_run_metrics.md)
     * run_ObsRate_metric.py
     * run_sl_metric.py
     * run_snr_metric.py