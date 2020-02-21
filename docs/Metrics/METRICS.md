# Installing, running and analyzing the metrics

## Installation of the metrics package

```
python install_sn_pack.py --package=metrics --gitbranch=thebranch
```

## Run and analyze the metrics

| Metric| Process| Analyze | Output plots|
|----|----|----|----|
| [Cadence](Cadence.md) | [Cadence metric run](Cadence_run.md) | [Cadence metric analysis](Cadence_plot.md)| cadence vs m5; redshift limit|
| [SNR](SNR.md) | [SNR metric run](SNR_run.md) | [SNR metric analysis](SNR_plot.md)| SN rate with SNR>SNR_min|
| [ObsRate](ObsRate.md)| [ObsRate metric run ](ObsRate_run.md) | [ObsRate metric analysis](ObsRate_plot.md)| SN observing rate|
| [NSN](NSN.md)| [NSN metric run](NSN_run.md) | [NSN metric analysis](NSN_plot.md)| NSN vs redshift limit|



If you have comments, suggestions or questions, please [write us an issue](https://github.com/LSSTDESC/sn_pipe/issues).


