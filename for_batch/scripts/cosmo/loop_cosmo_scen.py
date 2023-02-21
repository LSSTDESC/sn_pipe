import pandas as pd
import os
from optparse import OptionParser

def cmd(row,nproc,outDir,fileDir,Ny,fit_parameters,fit_prior,sigma_mu_photoz, sigma_mu_bias_x1_color,nsn_bias_simu,nsn_WFD_yearly,zspectro_only,nsn_spectro_ultra_yearly,nsn_spectro_ultra_tot,nsn_spectro_deep_yearly,nsn_spectro_deep_tot,nsn_spectro_tuned,pfs_current_strategy,scr='python sn_studies/sn_fom/fom.py'):
    
    fparams = fit_parameters.split(',')
    fparams = '_'.join(ff for ff in fparams)

    fDir = '{}_Ny_{}'.format(fileDir,Ny)
    oDir = '{}_Ny_{}_{}'.format(outDir,Ny,fparams)

    cmd_ = scr
    for vv in ['dbName', 'fields', 'nseasons', 'npointings', 'configName']:
        cmd_ += ' --{} {}'.format(vv, row[vv])

    cmd_ += ' --year_survey {}'.format(row['year'])

    cmd_ += ' --add_WFD SN_WFD_100000'
    cmd_ += ' --nproc {}'.format(nproc)
    cmd_ += ' --dirFit {}'.format(oDir)
    cmd_ += ' --fileDir {}'.format(fDir)
    cmd_ += ' --Ny {}'.format(Ny)
    cmd_ += ' --fit_parameters {}'.format(fit_parameters)
    cmd_ += ' --fit_prior {}'.format(fit_prior)
    if sigma_mu_photoz != 'None':
        cmd_ += ' --sigma_mu_photoz {}'.format(sigma_mu_photoz)
    cmd_ += ' --sigma_mu_bias_x1_color {}'.format(sigma_mu_bias_x1_color)
    cmd_ += ' --nsn_bias_simu {}'.format(nsn_bias_simu)
    cmd_ += ' --nsn_WFD_yearly {}'.format(nsn_WFD_yearly)
    cmd_ += ' --zspectro_only {}'.format(zspectro_only)
    cmd_ += ' --nsn_spectro_ultra_yearly {}'.format(nsn_spectro_ultra_yearly)
    cmd_ += ' --nsn_spectro_ultra_tot {}'.format(nsn_spectro_ultra_tot)
    cmd_ += ' --nsn_spectro_deep_yearly {}'.format(nsn_spectro_deep_yearly)
    cmd_ += ' --nsn_spectro_deep_tot {}'.format(nsn_spectro_deep_tot)
    cmd_ += ' --nsn_spectro_tuned {}'.format(nsn_spectro_tuned)
    cmd_ += ' --pfs_current_strategy {}'.format(pfs_current_strategy)
    print(cmd_)
    return cmd_


def prepareOut(tag):
    """                                                                                        
    Function to prepare for the batch                                                            

    directories for scripts and log files are defined here.                                    

    """

    cwd = os.getcwd()
    dirScript = cwd + "/scripts"

    if not os.path.isdir(dirScript):
        os.makedirs(dirScript)

    dirLog = cwd + "/logs"
    if not os.path.isdir(dirLog):
        os.makedirs(dirLog)

    id = '{}'.format(tag)

    name_id = 'cosmofit_{}'.format(id)
    log = dirLog + '/'+name_id+'.log'
    errlog = dirLog + '/'+name_id+'.err'

    return dirScript, name_id, log, errlog, cwd

def process(row,batch,nproc,outDir,fileDir,Ny,fit_parameters,fit_prior,sigma_mu_photoz, sigma_mu_bias_x1_color, nsn_bias_simu,nsn_WFD_yearly,zspectro_only,nsn_spectro_ultra_yearly,nsn_spectro_ultra_tot,nsn_spectro_deep_yearly,nsn_spectro_deep_tot,nsn_spectro_tuned,pfs_current_strategy,tagscript):

    tag = '{}_Ny_{}'.format(row['configName'],Ny)
    if tagscript != '':
        tag += '_{}'.format(tagscript)
    
    dirScript, name_id, log, errlog, cwd = prepareOut(tag)
    # qsub command
    #qsub = 'qsub -P P_lsst -l sps=1,ct=3:00:00,h_vmem=16G -j y -o {} -pe multicores {} <<EOF'.format(
    #    log, nproc)

    dict_batch = {}
    dict_batch['--account'] = 'lsst'
    dict_batch['-L'] = 'sps'
    dict_batch['--time'] = '3:00:00'
    dict_batch['--mem'] = '16G'
    dict_batch['--output'] = log
    #dict_batch['--cpus-per-task'] = str(nproc)                                                                    
    dict_batch['-n'] = 8
    dict_batch['--error'] = errlog
    #dict_batch['-p'] = 'hpc'

    scriptName = dirScript+'/'+name_id+'.sh'

    # fill the script
    script = open(scriptName, "w")
    script.write("#!/bin/env bash\n")
    if batch:
        for key, vals in dict_batch.items():
            script.write("#SBATCH {} {} \n".format(key,vals))

    if batch:
        script.write(" cd " + cwd + "\n")
        script.write(" echo 'sourcing setups' \n")
        script.write(" source setup_release.sh Linux -5\n")
        script.write("echo 'sourcing done' \n")

    # need this to limit the number of multithread
    script.write(" export MKL_NUM_THREADS=1 \n")
    script.write(" export NUMEXPR_NUM_THREADS=1 \n")
    script.write(" export OMP_NUM_THREADS=1 \n")
    script.write(" export OPENBLAS_NUM_THREADS=1 \n")

    cmd_ = cmd(row,nproc,outDir,fileDir,Ny,fit_parameters,fit_prior,sigma_mu_photoz, sigma_mu_bias_x1_color,nsn_bias_simu,nsn_WFD_yearly,zspectro_only,nsn_spectro_ultra_yearly,nsn_spectro_ultra_tot,nsn_spectro_deep_yearly,nsn_spectro_deep_tot,nsn_spectro_tuned,pfs_current_strategy)
    script.write(cmd_+ "\n")
    
    script.close()

    if batch:
        os.system("sbatch "+scriptName)
        print('go man')

parser = OptionParser()

parser.add_option("--fileName", type="str", default='config_cosmoSN.csv',
                  help="dbName to process  [%default]")
parser.add_option("--outDir", type="str",
                  default='/sps/lsst/users/gris/fake/Fit', help="output directory [%default]")
parser.add_option("--fileDir", type="str",
                  default='/sps/lsst/users/gris/Fakes_nosigmaInt/Fit', help="file directory [%default]")
parser.add_option("--batch", type=int, default=1,
                  help="batch running mode [%default]")
parser.add_option("--nproc", type=int, default=8,
                  help="number of proc to use[%default]")
parser.add_option("--Ny", type=int, default=20,help="y-band max visits at z=0.9 [%default]")
parser.add_option("--fit_parameters", type=str, default='Om,w0',
                  help="parameters to fit [%default]")
parser.add_option("--fit_prior", type=int, default=0,
                  help="prior for the fit [%default]")
parser.add_option("--sigma_mu_photoz", type=str, default='',
                  help="mu error from photoz [%default]")
parser.add_option("--sigma_mu_bias_x1_color", type=str, default='sigma_mu_bias_x1_color_1_sigma',
                  help="mu error bias from x1 and color n-sigma variation [%default]")
parser.add_option("--nsn_bias_simu", type=str, default='nsn_bias_Ny_40',
                  help="nsn_bias file for distance moduli simulation [%default]")
parser.add_option("--tagscript", type=str, default='',
                  help="tag for the script [%default]")
parser.add_option("--nsn_WFD_yearly", type=int, default=-1,
                  help="number of WFD SN per year (-1=full sample) [%default]")
parser.add_option("--zspectro_only", type=int, default=0,
                  help="select SN with z spectro only [%default]")
parser.add_option("--nsn_spectro_ultra_yearly", type=int, default=200,
                  help="number of spectro-z host for ultradeep fields (per year) [%default]")
parser.add_option("--nsn_spectro_ultra_tot", type=int, default=2000,
                  help="number of spectro-z host for ultradeep fields (total) [%default]")
parser.add_option("--nsn_spectro_deep_yearly", type=int, default=500,
                  help="number of spectro-z host for deep fields (per year) [%default]")
parser.add_option("--nsn_spectro_deep_tot", type=int, default=2500,
                  help="number of spectro-z host for deep fields (total) [%default]")
parser.add_option("--nsn_spectro_tuned", type=int, default=0,
                  help="number of spectro-z host for ud abs [%default]")
parser.add_option("--pfs_current_strategy", type=int, default=0,
                  help="to activate PFS current strategy [%default]")

opts, args = parser.parse_args()

cosmo_scen = pd.read_csv(opts.fileName, delimiter=';', comment='#')

for i, row in cosmo_scen.iterrows():
    process(row,opts.batch,opts.nproc,
            opts.outDir,opts.fileDir,opts.Ny,
            opts.fit_parameters,
            opts.fit_prior,
            opts.sigma_mu_photoz,
            opts.sigma_mu_bias_x1_color,
            opts.nsn_bias_simu,
            opts.nsn_WFD_yearly,
            opts.zspectro_only,
            opts.nsn_spectro_ultra_yearly,
            opts.nsn_spectro_ultra_tot,
            opts.nsn_spectro_deep_yearly,
            opts.nsn_spectro_deep_tot,
            opts.nsn_spectro_tuned,
            opts.pfs_current_strategy,
            opts.tagscript)
    
# print(cosmo_scen)