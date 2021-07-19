import glob
from optparse import OptionParser

def transform(fi):

    fa = fi.replace('logs','scripts')
    fa = fa.replace('.log','.sh')

    return fa

parser = OptionParser()

parser.add_option("--logDir", type="str", default='logs',
                  help="log directory [%default]")
parser.add_option("--relnum", type="str", default='v1.7',
                  help="release number [%default]")
parser.add_option("--metric", type="str", default='NSN',
                  help="metric  [%default]")

opts, args = parser.parse_args()

dirLog = opts.logDir
relnum = opts.relnum
metric = opts.metric

search_path = '{}/*{}_10yrs*{}*.log'.format(dirLog,relnum,metric)
fis = glob.glob(search_path)

print(fis)

what = 'end of processing'
outFile = None
for fi in fis:
    ff = open(fi)
    data = ff.read()
    n_what = data.count(what)
    if n_what != 8:
        print('to reprocess:',fi,transform(fi))
        if not outFile:
            outFile = open('toreprocess_{}.sh'.format(relnum),'w')
            outFile.write("#!/bin/bash \n")
        else:
            outFile.write('sh {} \n'.format(transform(fi)))

if outFile:
    outFile.close()
