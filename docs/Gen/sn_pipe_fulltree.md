```bash
├── doc
│   ├── cadence_m5_r.png
│   ├── snr_z_season_1.png
│   ├── yaml_cadence.md
│   ├── yaml_cadence.md~
│   ├── yaml_snr.md
│   └── zlim_r.png
├── docs
│   ├── Gen
│   │   ├── for_batch.md
│   │   ├── sn_pipe.md
│   │   ├── test.md
│   │   ├── usage_pip_sn_pack.md
│   │   └── usage_setup_release.md
│   ├── Metrics
│   │   ├── Cadence.md
│   │   ├── Cadence_run.md
│   │   └── Metrics.md
│   ├── sn_pipe_scheme.png
│   └── tree_md.sh
├── for_batch
│   ├── input
│   │   ├── DD_fbs13.txt
│   │   ├── DD_fbs14.txt
│   │   ├── make_files.python
│   │   ├── WFD_fbs13.txt
│   │   ├── WFD_fbs14.txt
│   │   └── WFD.txt
│   └── scripts
│       ├── batch_fit.py
│       ├── batch_global.py
│       ├── batch_metrics.py
│       ├── batch_obsTopixels.py
│       ├── batch_simulations.py
│       └── batch_templates.py
├── input
│   ├── Fake_cadence
│   │   ├── Fake_cadence_seqs.yaml
│   │   └── Fake_cadence.yaml
│   ├── fit_sn
│   │   └── param_fit_gen.yaml
│   ├── metrics
│   │   ├── param_cadence_metric.yaml
│   │   ├── param_obsrate_metric.yaml
│   │   ├── param_sl_metric.yaml
│   │   └── param_snr_metric.yaml
│   ├── reshuffling
│   │   ├── scen1.csv
│   │   └── scen2.csv
│   ├── simulation
│   │   ├── param_fakesimulation.yaml
│   │   ├── param_simulation_gen.yaml
│   │   ├── param_simulation_nb.yaml
│   │   └── param_simulation.yaml
│   ├── sn_studies
│   │   ├── DD_scen1.yaml
│   │   ├── DD_scen2.yaml
│   │   ├── DD_scen3.yaml
│   │   ├── Fakes_NSNMetric_Fake_lc_nside_64_coadd_0_0.0_360.0_-1.0_-1.0_0.hdf5
│   │   ├── medValues_flexddf_v1.4_10yrs_DD.npy
│   │   └── medValues.npy
│   └── templates
│       ├── Fake_cadence_template.yaml
│       └── param_fakesimulation_template.yaml
├── LICENCE
├── notebooks
│   ├── SNCadence.ipynb
│   ├── SNSimulation.ipynb
│   └── SNSNR.ipynb
├── pip_sn_pack.py
├── plot_scripts
│   ├── cadenceCustomize_fbs13.csv
│   ├── cadenceCustomize_fbs14.csv
│   ├── cadenceCustomize.txt
│   ├── plot_cadence_metric_DD.py
│   ├── plot_cadence_metric.py
│   ├── plot_global.py
│   ├── plot_LC.py
│   ├── plot_nsn_metric_DD.py
│   ├── plot_nsn_metric_WFD.py
│   ├── plot_snr_metric.py
│   └── plot_summary.py
├── README.md
├── reference_files
│   ├── Dist_X1_Color_JLA_high_z.txt
│   ├── Dist_X1_Color_JLA_low_z.txt
│   ├── gamma.hdf5
│   ├── LC_Ref_-2.0_0.2.hdf5
│   ├── Li_SNCosmo_-2.0_0.2.npy
│   ├── Li_SNSim_-2.0_0.2.npy
│   ├── Mag_to_Flux_SNCosmo.npy
│   ├── Mag_to_Flux_SNSim.npy
│   ├── pixel_max_median_double_gaussian_profile.npy
│   ├── pixel_max_median_single_gaussian_profile.npy
│   ├── SNR_m5.npy
│   └── X0_norm_-19.0906.npy
├── requirements.txt
├── run_scripts
│   ├── fakes
│   │   ├── make_fake.py
│   │   └── sigmaC_z.py
│   ├── fit_sn
│   │   └── run_sn_fit.py
│   ├── io_transform
│   │   └── extract.py
│   ├── metrics
│   │   ├── config_metric_template.yaml
│   │   ├── estimate_DDFrac.py
│   │   ├── metricWrapper.py
│   │   ├── run_cadence_metric.py
│   │   ├── run_global_metric.py
│   │   ├── run_metrics_fromnpy.py
│   │   ├── run_ObsRate_metric.py
│   │   ├── run_sl_metric.py
│   │   └── run_snr_metric.py
│   ├── obs_pixelize
│   │   ├── gime_npixels.py
│   │   └── run_obs_to_pixels.py
│   ├── postprocessing
│   │   ├── convert_to_npy.py
│   │   └── Loop_convert.py
│   ├── simulation
│   │   ├── run_simulation_fromnpy.py
│   │   └── run_simulation.py
│   ├── sn_studies
│   │   ├── run_design_survey.py
│   │   ├── run_sn_reshuffle.py
│   │   └── snr_m5.py
│   ├── templates
│   │   ├── run_diffflux.py
│   │   └── run_simulation_template.py
│   ├── utils
│   │   ├── add_DD_Fields.py
│   │   ├── calcGamma.py
│   │   └── reshuffling.py
│   └── visu_cadence
│       └── run_visu_cadence.py
├── SALT2_Files
│   ├── Instruments
│   │   ├── Landolt
│   │   │   ├── CVS
│   │   │   │   ├── Entries
│   │   │   │   ├── Repository
│   │   │   │   └── Root
│   │   │   ├── darksky.dat
│   │   │   ├── FilterWheel
│   │   │   ├── instrument.cards
│   │   │   ├── instrument.cards.~1.1.1.1.~
│   │   │   ├── README
│   │   │   ├── sb_-41A.dat
│   │   │   ├── si_-25A.dat
│   │   │   ├── sr_-21A.dat
│   │   │   ├── sux_modified.dat
│   │   │   └── sv_-27A.dat
│   │   └── SNLS3-Landolt-model
│   │       └── sb-shifted.dat
│   ├── MagSys
│   │   ├── bd_17d4708_stisnic_002.ascii
│   │   ├── bd_17d4708_stisnic_002.ascii~
│   │   ├── bd_17d4708_stisnic_002.ascii.orig
│   │   └── VegaBD17-2008-11-28.dat
│   ├── RatInt_for_mb.npy
│   ├── RatInt_for_mb_old.npy
│   └── snfit_data
│       └── salt2-4
│           ├── salt2_template_0.dat
│           └── salt2_template_1.dat
├── setup.py
└── setup_release.sh
```