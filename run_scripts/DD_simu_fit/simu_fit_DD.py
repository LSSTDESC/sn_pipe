import glob
import os
from optparse import OptionParser

def simuSN(dbDir, dbName, dbExtens,outputDir_simu,
           nclusters,fieldname,
           SN_x1_type,SN_x1_min,SN_x1_max,SN_x1_step,
           SN_color_type,SN_color_min,SN_color_max,SN_color_step,
           SN_z_type,SN_z_min, SN_z_max,SN_z_step,
           SN_NSNfactor,  prodid, error_model,nproc):
    
    cmd = 'python run_scripts/simulation/run_simulation.py '
    cmd += ' --dbDir {}'.format(dbDir)
    cmd += ' --dbName {}'.format(dbName)
    cmd += ' --dbExtens {}'.format(dbExtens)
    cmd += ' --Output_directory {}'.format(outputDir_simu)
    cmd += '--Observations_fieldtype DD --nclusters {}'.format(nclusters)
    cmd += ' --nproc {}'.format(nproc)
    for vv in ['x1','color','z']:
        for tt in ['type','min','max','step']:
            val = eval('SN_{}_{}'.format(vv,tt))
            cmd += ' --SN_{}_{} {}'.format(vv,tt,val)

    cmd += ' --ProductionID {}'.format(prodid)
    cmd += ' --Simulator_errorModel {}'.format(error_model)
            
    print(cmd)


def fitSN(dirFiles,prodid,outDir_fit,nbef,naft,snrmin,fitter,nproc):
    fitter_name= 'sn_fitter.fit_{}'.format(fitter)
    prodid_fit = '{}_{}'.format(prodid,fitter)
    cmd = 'python run_scripts/fit_sn/run_sn_fit.py'
    cmd += ' --Simulations_dirname {} --ProductionID {}'.format(dirFiles,prodid_fit)
    cmd += ' --Multiprocessing_nproc {}'.format(nproc)
    cmd += ' --Output_directory {}'.format(outDir_fit)
    cmd += ' --LCSelection_naft {}'.format(naft)
    cmd += ' --Simulations_prodid {}'.format(prodid)
    cmd += ' --Fitter_name {}'.format(fitter_name)
    cmd += ' --LCSelection_snrmin 5.'
    print(cmd)
    


def fitLoop(dirFiles,prodid,outDir_fit,nbef,naft,snrmin,fitter,nproc):
    files = glob.glob('{}/Simu_{}*.hdf5'.format(dirFiles,prodid))
    for j,fi in enumerate(files):
        ff = fi.split('Simu_')[-1]
        ff = ff.split('.hdf5')[0]
        if 'error_model' in ff:
            print(ff)
            fitSN(dirFiles,ff,outDir_fit,nbef,naft,snrmin,fitter,nproc)
            break
    
    
    
parser = OptionParser()

parser.add_option("--dbDir", type=str, default='',
                  help="db dir [%default]")
parser.add_option("--dbName", type=str, default='descddf_v1.4_10yrs',
                  help="db name [%default]")
parser.add_option("--dbExtens", type=str, default='npy',
                  help="db extens [%default]")
parser.add_option("--outputDir_simu", type=str, default='/sps/lsst/users/gris/DD/Simu',
                  help="output dir for simulations [%default]")
parser.add_option("--outputDir_fit", type=str, default='/sps/lsst/users/gris/DD/Fit',
                  help="output dir for fit [%default]")
parser.add_option("--nclusters", type=int, default=6,
                  help="number of DD clusters in data[%default]")
parser.add_option("--fieldname", type=str, default='COSMOS',
                  help="DD field to process [%default]")
parser.add_option("--nsn_factor", type=int, default=10,
                  help="factor for sn simulation (NSN=factor*NSN_rate)[%default]")
parser.add_option("--nproc", type=int, default=8,
                  help="number of proc[%default]")
parser.add_option("--snrmin", type=float, default=5.,
                  help="SNR min for LC points (fit)[%default]")
parser.add_option("--nbef", type=int, default=4,
                  help="min n LC points before max (fit)[%default]")
parser.add_option("--naft", type=int, default=10,
                  help="min n LC points after max (fit)[%default]")

opts, args = parser.parse_args()

# simulation of faint SN to get zlims

simuSN(dbDir=opts.dbDir, dbName=opts.dbName, dbExtens=opts.dbExtens,
       outputDir_simu=opts.outputDir_simu,
       nclusters=opts.nclusters,fieldname=opts.fieldname,
           SN_x1_type='unique',SN_x1_min=-2.0,SN_x1_max=2.0,SN_x1_step=0.1,
           SN_color_type='unique',SN_color_min=0.2,SN_color_max=0.3,SN_color_step=0.01,
           SN_z_type='random',SN_z_min=0.01, SN_z_max=0.7,SN_z_step=0.01,
           SN_NSNfactor=opts.nsn_factor, prodid='{}_{}_faintSN'.format(opts.dbName,opts.fieldname),
       error_model=1,nproc=opts.nproc)

# simulation of random SN 

simuSN(dbDir=opts.dbDir, dbName=opts.dbName, dbExtens=opts.dbExtens,
       outputDir_simu=opts.outputDir_simu,
       nclusters=opts.nclusters,fieldname=opts.fieldname,
           SN_x1_type='random',SN_x1_min=-2.0,SN_x1_max=2.0,SN_x1_step=0.1,
           SN_color_type='random',SN_color_min=0.2,SN_color_max=0.3,SN_color_step=0.01,
           SN_z_type='random',SN_z_min=0.01, SN_z_max=1.2,SN_z_step=0.01,
           SN_NSNfactor=opts.nsn_factor, prodid='{}_{}_allSN'.format(opts.dbName,opts.fieldname),
       error_model=1,nproc=opts.nproc)

# fit those LC
prodid = '{}_{}_faintSN'.format(opts.dbName,opts.fieldname)
fitLoop(opts.outputDir_simu,prodid,opts.outputDir_fit,opts.nbef,opts.naft,opts.snrmin,'sn_cosmo',opts.nproc)

