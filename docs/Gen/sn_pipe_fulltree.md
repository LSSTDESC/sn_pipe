```bash
|-- docs
|   |-- Figures
|   |   |-- LC_1.png
|   |   |-- nsn_metric_cadence.png
|   |   |-- nsn_metric_ebv.png
|   |   |-- nsn_metric_nsn.png
|   |   |-- nsn_metric_seasonlength.png
|   |   |-- nsn_metric_zlim.png
|   |   |-- nsn_WFD_summary.png
|   |   |-- NSN_zlim_DDF_1.png
|   |   |-- NSN_zlim_DDF_2.png
|   |   |-- NSN_zlim_DDF_3.png
|   |   |-- sigmaC.png
|   |-- Fit
|   |   |-- Fit.md
|   |   |-- usage_run_sn_fit.md
|   |-- Gen
|   |   |-- for_batch.md
|   |   |-- plot_scripts.md
|   |   |-- run_scripts.md
|   |   |-- sn_pipe_fulltree.md
|   |   |-- sn_pipe.md
|   |   |-- test.md
|   |   |-- usage_pip_sn_pack.md
|   |   |-- usage_setup_release.md
|   |-- Metrics
|   |   |-- Cadence.md
|   |   |-- Cadence_run.md
|   |   |-- Metrics.md
|   |   |-- nsn_metric.md
|   |   `-- usage_run_metrics.md
|   |-- Plots
|   |   |-- usage_plot_lcfit.md
|   |   |-- usage_plot_nsn_metric_DD_summary.md
|   |   |-- usage_plot_nsn_metric_OS.md
|   |   |-- usage_plot_nsn_metric_WFD_summary.md
|   |   |-- usage_plot_simu.md
|   |-- Simulation
|   |   |-- make_yaml.md
|   |   |-- Simulation.md
|   |   |-- usage_run_simulation.md
|   |   |-- usage_run_simulation_yaml.md
|   |   |-- yaml_file.md
|   |-- sn_pipe_scheme.png
|   |-- Templates
|   |   |-- usage_run_template_LC.md
|   |   |-- usage_run_template_vstack.md
|   |-- tree_md.sh
|-- for_batch
|   |-- input
|   |   |-- DD_fbs13.txt
|   |   |-- DD_fbs_14.csv
|   |   |-- DD_fbs14.txt
|   |   |-- DD_fbs_15.csv
|   |   |-- global.csv
|   |   |-- make_files.py
|   |   |-- WFD_fbs13.txt
|   |   |-- WFD_fbs14.csv
|   |   |-- WFD_fbs14.txt
|   |   |-- WFD_fbs15.csv
|   |   |-- WFD_fbs15.txt
|   |   |-- WFD.txt
|   |-- scripts
|       |-- batch_fit.py
|       |-- batch_global.py
|       |-- batch_metrics.py
|       |-- batch_obsTopixels.py
|       |-- batch_simulations.py
|       |-- batch_templates.py
|       |-- check_batch.py
|       `-- check_template_prod.py
|-- input
|   |-- Fake_cadence
|   |   |-- Fake_cadence_seqs.yaml
|   |   |-- Fake_cadence.yaml
|   |-- fit_sn
|   |   |-- param_fit_gen.yaml
|   |-- metrics
|   |   |-- param_cadence_metric.yaml
|   |   |-- param_obsrate_metric.yaml
|   |   |-- param_sl_metric.yaml
|   |   |-- param_snr_metric.yaml
|   |-- plots
|   |   |-- podids_fit.csv
|   |-- reshuffling
|   |   |-- scen1.csv
|   |   |-- scen2.csv
|   |-- simulation
|   |   |-- param_fakesimulation.yaml
|   |   |-- param_simulation_example.yaml
|   |   |-- param_simulation_gen.yaml
|   |   |-- param_simulation_nb.yaml
|   |   |-- param_simulation.yaml
|   |-- sn_studies
|   |   |-- DD_scen1.yaml
|   |   |-- DD_scen2.yaml
|   |   |-- DD_scen3.yaml
|   |   |-- Fakes_NSNMetric_Fake_lc_nside_64_coadd_0_0.0_360.0_-1.0_-1.0_0.hdf5
|   |   |-- medValues_flexddf_v1.4_10yrs_DD.npy
|   |   |-- medValues.npy
|   |-- templates
|       |-- Fake_cadence_template.yaml
|       |-- param_fakesimulation_template.yaml
|-- LICENCE
|-- notebooks
|   |-- SNCadence.ipynb
|   |-- SNFitLC.ipynb
|   |-- SNSimulation.ipynb
|   |-- SNSNR.ipynb
|-- pip_sn_pack.py
|-- plot_scripts
|   |-- input
|   |   |-- cadenceCustomize_fbs13.csv
|   |   |-- cadenceCustomize_fbs14.csv
|   |   |-- cadenceCustomize.txt
|   |   |-- WFD_test.csv
|   |-- lcfit
|   |   |-- plot_lcfit.py
|   |-- metrics
|   |   |-- plot_cadence_metric_DD.py
|   |   |-- plot_cadence_metric.py
|   |   |-- plot_global.py
|   |   |-- plot_LC.py
|   |   |-- plot_nsn_metric_DD_summary.py
|   |   |-- plot_nsn_metric_OS.py
|   |   |-- plot_nsn_metric_WFD_summary.py
|   |   |-- plot_snr_metric.py
|   |   |-- plot_summary.py
|   |-- simulation
|       |-- plot_simu.py
|-- README.md
|-- requirements.txt
|-- run_scripts
|   |-- cutoff_effect
|   |   |-- cutoff_studies.py
|   |-- dust_for_fast
|   |   |-- displayDustMap.py
|   |   |-- dustImpact.py
|   |   |-- dustImpact_Templates.py
|   |-- fakes
|   |   |-- full_simulation_fit.py
|   |   |-- full_simulation.py
|   |   |-- loop_full_fast.py
|   |   |-- make_fake.py
|   |   |-- sigmaC_z.py
|   |-- fit_sn
|   |   |-- run_sn_fit.py
|   |-- io_transform
|   |   |-- extract.py
|   |-- metrics
|   |   |-- config_metric_template.yaml
|   |   |-- estimate_DDFrac.py
|   |   |-- metricWrapper.py
|   |   |-- run_cadence_metric.py
|   |   |-- run_global_metric.py
|   |   |-- run_metrics.py
|   |   |-- run_ObsRate_metric.py
|   |   |-- run_sl_metric.py
|   |   |-- run_snr_metric.py
|   |-- obs_pixelize
|   |   |-- gime_npixels.py
|   |   |-- run_obs_to_pixels.py
|   |-- postprocessing
|   |   |-- convert_to_npy.py
|   |   |-- Loop_convert.py
|   |-- simulation
|   |   |-- make_yaml.py
|   |   |-- run_simulation_from_yaml.py
|   |   |-- run_simulation_MAF.py
|   |   |-- run_simulation.py
|   |   |-- simuWrapper.py
|   |-- sn_studies
|   |   |-- check_snr.py
|   |   |-- run_design_survey.py
|   |   |-- run_sn_reshuffle.py
|   |   |-- snr_m5.py
|   |-- templates
|   |   |-- run_diffflux.py
|   |   |-- run_simulation_template.py
|   |   |-- run_template_LC.py
|   |   |-- run_template_vstack.py
|   |   |-- simuLoop.py
|   |-- utils
|   |   |-- add_DD_Fields.py
|   |   |-- calcGamma.py
|   |   |-- reshuffling.py
|   |-- visu_cadence
|       |-- run_visu_cadence.py
|-- setup.py
|-- setup_release.sh

```
