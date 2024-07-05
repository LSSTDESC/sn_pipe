import os


listdb = ['WFD_fbs_3.2.csv', 'WFD_fbs_3.4.csv']
listdb = ['WFD_fbs_3.4_sat.csv']
psfs = ['moffat', 'single_gauss']
ccdfulls = [80000, 90000, 100000, 130000]

for dbName in listdb:
    for psf in psfs:
        for ccdfull in ccdfulls:
            cmd = 'python for_batch/scripts/sn_prod/prod_sn_dd_wfd.py'
            cmd += ' --outDir_DD=\'\' '
            cmd += ' --outDir_WFD=/sps/lsst/users/gris/Output_SN_WFD_sigmaInt_0.0_Hounsell_z_smflux_sat'
            cmd += ' --dbList_DD=\'\''
            cmd += ' --dbList_WFD={}'.format(dbName)
            cmd += ' --SN_smearFlux=1 --Fitter_sigmaz=1e-05 --SN_z_max=0.035 --SN_NSNfactor_WFD=1000'
            cmd += ' --saturation_effect=1 --saturation_psf={} --saturation_ccdfullwell={}'.format(
                psf, ccdfull)
            cmd += ' --Observations_coadd=0 --fit_remove_sat=0,1'

            # print(cmd)
            os.system(cmd)
