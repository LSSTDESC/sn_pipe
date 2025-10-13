
## Usage: for_batch/scripts/sim_to_fit/prodIt.py [options]

<pre>
Script to produce SN for WFD and DDF surveys

Options:
  -h, --help            show this help message and exit
  --runType=RUNTYPE     type of run: DDF, WFD, DDF+WFD [DDF]
  --dbList_DD=DBLIST_DD
                        dbList DD to process  [DD_fbs_5.0.0.csv]
  --dbList_WFD=DBLIST_WFD
                        dbList WFD to process  [WFD_fbs_5.0.0.csv]
  --outDir_main=OUTDIR_MAIN
                        Main output dir
                        [/sps/lsst/groups/cadence/LSST_SN_PhG/prod_simu]
  --outDir_DD=OUTDIR_DD
                        output dir for DDF [Output_SN_DD_sigmaInt_0.0_Hounsell
                        _z_smflux_notelrot_airmass]
  --outDir_WFD=OUTDIR_WFD
                        output dir for WFD [Output_SN_WFD_sigmaInt_0.0_Hounsel
                        l_z_smflux_notelrot_airmass]
  --SN_smearFlux=SN_SMEARFLUX
                        SN flux smearing [1]
  --Fitter_sigmaz=FITTER_SIGMAZ
                        sigma_z for the fitter [1e-05]
  --Observations_coadd=OBSERVATIONS_COADD
                        coadd observations [1]
  --saturation_effect=SATURATION_EFFECT
                        to include saturation effects [0]
</pre>