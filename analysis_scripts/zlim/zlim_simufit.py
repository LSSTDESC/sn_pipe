from optparse import OptionParser
from analysis import loadData, zcompleteness

parser = OptionParser()

parser.add_option("--dbDir", type='str', default='../../Fit_sncosmo',
                  help="location dir of the files to process[%default]")
parser.add_option("--dbName", type='str', default='baseline_nexp2_v1.7_10yrs',
                  help="name of the file to process [%default]")
parser.add_option("--tagName", type='str', default='faintSN',
                  help="tag name for the production to analyze [%default]")


opts, args = parser.parse_args()


# load file
tab = loadData(opts.dbDir, opts.dbName, opts.tagName)

print(len(tab))

zcomp = zcompleteness(tab)

print(zcomp())
