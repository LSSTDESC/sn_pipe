from optparse import OptionParser
import yaml

def decrypt_parser(parser):
    """
    Method to decrypt the parser help

    Parameters
    ---------------
    parser: optparse parser


    Returns
    ----------
    dict with decrypted infos

    """
    
    file_name = 'help_script.txt'
    file_object  = open(file_name,'w')
    parser.print_help(file_object)
    file_object.close()

    file = open(file_name, 'r') 
    line = file.read().splitlines()

    params = {}
    for i,ll in enumerate(line):
        lolo = ' '.join(ll.split(' ')).split()
        if lolo and lolo[0][:2]=='--':
            key = lolo[0].split('--')[1]
            key = key.split('=')
            params[key[0]]= (key[1].split('/')[0],key[1].split('/')[1])

    return params

def make_dict(thedict,key,what,val):
    """
    Method to append to a dict infos
    The goal is in fine to create a yaml file

    Parameters
    --------------
    thedict: dict
      the dict to append to
    key: str
     key used to append
    what: (str,str)
      name and type
    val: str
      value

    Returns
    ----------
    thedict: dict
      resulting dict

    """
      
    
    keym = key.split('_')[0]
    keys = ''
    if '_' in key:
        keys = '_'.join(key.split('_')[1:])

    names = what[0].split(',')
    dtypes = what[1].split(',')
    val = val.split(',')
    
    valb = [val[i] if dtypes[i]=='str' else eval('{}({})'.format(dtypes[i],val[i]))  for i in range(len(dtypes))]
    
    if keys=='':
        if names[0] != 'value':
            thedict[keym]=dict(zip(names,valb))
        else:
            thedict[keym]=valb[0]
    else:
        if keym not in thedict.keys():
            thedict[keym] = {}
        if names[0] != 'value':
            thedict[keym][keys]=dict(zip(names,valb))  
        else:
            thedict[keym][keys]=valb[0] 
            
    return thedict

parser = OptionParser()

# ProductionID: prodid
parser.add_option('--ProductionID',help='Id for the production  [%default]',default='prodid',metavar='value/str')

# Supernova parameters
parser.add_option('--SN_Id',help='SN Id [%default]',default='100',metavar='value/int')
parser.add_option('--SN_type',help='SN type [%default]',default='SN_Ia',metavar='value/str')
parser.add_option('--SN_x1_color',help='SN x1 and color distributions',default='JLA,reference_files',metavar='rate,dirFile/str,str')
parser.add_option('--SN_x1',help='x1 SN[%default]',default='unique,-2.0,1.0,0.1',metavar='type,min,max,step/str,float,float,float')
parser.add_option('--SN_color',help='color SN[%default]',default='unique,0.2,1.0,0.1',metavar='type,min,max,step/str,float,float,float')
parser.add_option('--SN_z',help='redshift SN[%default]',default='unique,0.01,1.0,0.1,Perrett',metavar='type,min,max,step,rate/str,float,float,float,str')
parser.add_option('--SN_daymax',help='daymax SN[%default]',default='unique,1.',metavar='type,step/str,float')
parser.add_option('--SN_min_rf_phase',help='min rf phase SN[%default]',default='-20.',metavar='value/float')
parser.add_option('--SN_max_rf_phase',help='max rf phase SN[%default]',default='60.',metavar='value/float')
parser.add_option('--SN_min_rf_phase_qual',help='min rf phase qual SN[%default]',default='-15.',metavar='value/float')
parser.add_option('--SN_max_rf_phase_qual',help='max rf phase qual SN[%default]',default='45. ',metavar='value/float')
parser.add_option('--SN_absmag',help='absmag SN[%default]',default='-19.0906',metavar='value/float')
parser.add_option('--SN_band',help='band SN[%default]',default='bessellB',metavar='value/str')
parser.add_option('--SN_magsys',help='magsys SN[%default]',default='vega',metavar='value/str')
parser.add_option('--SN_differential_flux',help='diff flux SN[%default]',default='0',metavar='value/int')
parser.add_option('--SN_salt2Dir',help='SALT2DIR[%default]',default='SALT2_Files',metavar='value/str')
parser.add_option('--SN_blue_cutoff',help='blue cutoff SN[%default]',default='380.',metavar='value/float')
parser.add_option('--SN_red_cutoff',help='red cutoff SN[%default]',default='800.',metavar='value/float')
parser.add_option('--SN_ebvofMW',help='E(B-V) of MW[%default]',default='-1.',metavar='value/float')
parser.add_option('--SN_NSN_factor',help='factor for simulation[%default]',default='1',metavar='value/int')

## Cosmology parameters
parser.add_option('--Cosmology',help='cosmology parameters[%default]',
                  default='w0waCDM,0.30,0.70,72.0,-1.0,0.0',
                  metavar='Model,Omega_m,Omega_l,H0,w0,wa/str,float,float,float,float,float')

## Instrument
parser.add_option('--Instrument',help='instrument config[%default]',
                  default='LSST,LSST_THROUGHPUTS_BASELINE,THROUGHPUTS_DIR,1.2,1,0',
                  metavar='name,throughput_dir,atmos_dir,airmass,atmos,aerosol/str,str,str,float,int,int')

##Observations
parser.add_option('--Observations',help='observations config[%default]',
                  default='fullDbName,WFD,1,-1',
                  metavar='filename,fieldtype,coadd,season/str,str,int,int')

## Simulator
parser.add_option('--Simulator',help='simulator config[%default]',
                  default='sn_simulator.sn_cosmo,salt2-extended,1.0,1',
                  metavar='name,model,version,error_model/str,str,float,int')

## Reference_Files
parser.add_option('--ReferenceFiles',help='ref files [%default]',
                  default='Template_LC,reference_files,gamma.hdf5,Template_Dust',
                  metavar='Template_Dir,Gamma_Dir,Gamma_File,DustCorr_Dir/str,str,str,str')

## Host
parser.add_option('--Host',help='host parameters[%default]',default='0',metavar='value/int')
                  
## Display LC : #display during LC simulations
parser.add_option('--Display_LC',help='display LC parameters[%default]',default='0,5',metavar='display,time/int,float')

##Output:
parser.add_option('--Output',help='output dir parameters[%default]',default='Output_Simu,1',metavar='directory,save/str,int')

#Multiprocessing
parser.add_option('--Multiprocessing',help='multiprocessing parameters[%default]',default='1',metavar='nproc/int')

# Pixelisation
parser.add_option('--Pixelisation',help='sky pixelisation parameters[%default]',default='64',metavar='nside/int')
  
#Web path
parser.add_option('--WebPath',help='web path for config parameters[%default]',default='https://me.lsst.eu/gris/DESC_SN_pipeline',metavar='value/str')

# configname
parser.add_option('--configName',help='name for the produced yaml file [%default]',default='config.yaml',metavar='value/str')

opts, args = parser.parse_args()


params = decrypt_parser(parser)
print(params)
               

conf = {}
for key, vals in params.items():
    conf = make_dict(conf,key,vals,eval('opts.{}'.format(key)))


print(conf)
#dumping in a yaml file
with open(opts.configName, 'w') as file:
    documents = yaml.dump(conf, file)
