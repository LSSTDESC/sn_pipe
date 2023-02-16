import os
from optparse import OptionParser


def launch(src, dbList, outDir_main, fieldNames, sel,itag):
    """
    Function to launch batches for DD studies

    Parameters
    ----------------
    src: str 
      script to use
    dbList: str
      list of dbNames to process.
    outDir: str
      output directory name
    fieldNames: str
      DD field Names to process.
    sel: str
      selection string.

    """
    
    VRO_FPs = ['circular','realistic']
    VRO_FPs = ['realistic']
    project_FPs = ['gnomonic']

    telrots = [0,1,-1]
    telrots = [1]
    nsides = [128,256]
    nsides= [128]

    for nside in nsides:
        for VRO_FP in VRO_FPs:
            for project_FP in project_FPs:
                for telrot in telrots:
                    cmd = '{} --dbList {} --fieldNames {} {}'.format(src,dbList,fieldNames,sel)
                    cmd += ' --outDir {}_{}_{}_{}_{}_new'.format(outDir_main,nside,project_FP,VRO_FP,telrot)
                    cmd += ' --pixelmap_dir None'
                    cmd += ' --nside {}'.format(nside)
                    cmd += ' --project_FP {}'.format(project_FP)
                    cmd += ' --VRO_FP {}'.format(VRO_FP)
                    cmd += ' --telrot {}'.format(telrot)
                    cmd += ' --addInfo 1'
                    cmd += ' --tagName {}'.format(itag)
                    print(cmd)
                    os.system(cmd)

parser = OptionParser()

parser.add_option("--dbLists", type="str", default='DD_fbs_2.99.csv,DD_fbs_2.1.csv',
                  help="main dir where the files are located [%default]")

opts, args = parser.parse_args()

src = 'python for_batch/scripts/nsn_metric_DD.py'

outDir_main = '/sps/lsst/users/gris/MetricOutput_DD'
fieldNames = ['COSMOS,XMM-LSS,ELAISS1','EDFSa,EDFSb,CDFS']
sel = ' --select_epochs 1 --n_bef 4 --n_aft 10'
dbLists = opts.dbLists.split(',')
#fields = ','.join(fieldNames)

for dbList in dbLists:
    for itag,fields in enumerate(fieldNames):
        launch(src,dbList,outDir_main,fields,sel,itag)


