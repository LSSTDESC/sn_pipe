from optparse import OptionParser
from analysis import loadData, zlimit

parser = OptionParser()

parser.add_option("--dbDir", type='str', default='../../Fit_sncosmo',
                  help="location dir of the files to process[%default]")
parser.add_option("--dbName", type='str', default='baseline_nexp2_v1.7_10yrs',
                  help="name of the file to process [%default]")
parser.add_option("--tagName_zlim", type='str', default='faintSN',
                  help="tag name for the production to get zlim [%default]")
parser.add_option("--tagName_cosmo", type='str', default='allSN',
                  help="tag name for the production to get cosmology [%default]")


opts, args = parser.parse_args()

# load data -  faintSN
tab_faint = loadData(opts.dbDir, opts.dbName, opts.tagName_zlim)

print(len(tab_faint))

# estimate redshift completeness
zcomp = zlimit(tab_faint)

zcomplete = zcomp()

# load data - allSN
tab_all = loadData(opts.dbDir, opts.dbName, opts.tagName_cosmo)

print(tab_all.columns)
