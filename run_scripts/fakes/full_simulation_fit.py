import os
from optparse import OptionParser


parser = OptionParser()

parser.add_option("--fake_config", type="str", default='Fake_cadence.yaml',
                  help="config file name for fake obs[%default]")
parser.add_option("--fake_output", type="str", default='Fake_DESC',
                  help="output file namefor fake_obs[%default]")
parser.add_option("--RAmin", type=float, default=-0.05,
                  help="RA min for obs area [%default]")
parser.add_option("--RAmax", type=float, default=0.05,
                  help="RA max for obs area [%default]")
parser.add_option("--Decmin", type=float, default=-0.05,
                  help="Dec min for obs area [%default]")
parser.add_option("--Decmax", type=float, default=0.05,
                  help="Dec max for obs area [%default]")
parser.add_option("--outDir_simu", type=str, default='Output_Simu',
                  help="output dir for simulation results[%default]")
parser.add_option("--outDir_fit", type=str, default='Output_Fit',
                  help="output dir for fit results [%default]")


opts, args = parser.parse_args()

fake_config = opts.fake_config
fake_output = opts.fake_output
RAmin = opts.RAmin
RAmax = opts.RAmax
Decmin = opts.Decmin
Decmax = opts.Decmax
outDir_simu = opts.outDir_simu
outDir_fit = opts.outDir_fit


# first step: create fake data from yaml configuration file
cmd = 'python run_scripts/fakes/make_fake.py --config {} --output {}'.format(
    fake_config, fake_output)

os.system(cmd)

# now run the full simulation on these data
cmd = 'python run_scripts/simulation/run_simulation_fromnpy.py --dbDir .'
cmd += ' --dbName {}'.format(opts.fake_output)
cmd += ' --dbExtens npy --x1 -2.0 --color 0.2 --fieldType Fake'
cmd += ' --coadd 0 --radius 0.1 --saveData 1'
cmd += ' --outDir {}'.format(outDir_simu)
print(cmd)
os.system(cmd)

# now fit these light curves
cmd = 'python run_scripts/fit_sn/run_sn_fit.py'
cmd += ' --dirFiles {}'.format(outDir_simu)
cmd += ' --prodid {}_seas_-1_-2.0_0.2'.format(fake_output)
cmd += ' --prefix sncosmo_Fake --mbcov 0 --nproc 1'
cmd += ' --outDir {}'.format(outDir_fit)

print(cmd)
os.system(cmd)
