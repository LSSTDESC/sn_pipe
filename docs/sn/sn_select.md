
## Usage: for_batch/scripts/select/select_sn.py [options]

<pre>
script to select SNe Ia

Options:
  -h, --help            show this help message and exit
  --dataDir=DATADIR     data dir [/sps/lsst/groups/cadence/LSST_SN_PhG/prod_si
                        mu/Output_SN_WFD_sigmaInt_0.0_Hounsell_z_smflux_notelr
                        ot_airmass]
  --listFields=LISTFIELDS
                        list of fields to process [WFD]
  --fieldType=FIELDTYPE
                        Type of fields to process [WFD]
  --dbList=DBLIST       List of OS to process [list_OS_new_wfd.csv]
  --timescale=TIMESCALE
                        timescale for output files [year]
  --selconfig=SELCONFIG
                        selection criteria [G10_JLA]
  --outDir_pre=OUTDIR_PRE
                        main output directory [/sps/lsst/users/gris/Output_SN_
                        WFD_sigmaInt_0.0_Hounsell_z_smflux_notelrot_airmass]
  --scriptName=SCRIPTNAME
                        output sh script [select_wfd.sh]
  --runIt=RUNIT         to run the sh script [1]

</pre>
