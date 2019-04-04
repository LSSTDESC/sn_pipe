#!/bin/bash

location=$1
script_loc=$2

declare -A arr
#arr['NERSC']='/global/common/software/lsst/cori-haswell-gcc/stack/setup_w_2018_13-sims_2_7_0.sh'
arr['NERSC']='/global/common/software/lsst/cori-haswell-gcc/stack/setup_w_2018_19-sims_2_8_0.sh'
arr['CCIN2P3']='/cvmfs/sw.lsst.eu/linux-x86_64/lsst_sims/sims_2_8_0/loadLSST.bash'
arr['mylaptop']='/cvmfs/sw.lsst.eu/linux-x86_64/lsst_sims/sims_2_8_0/loadLSST.bash'

if [ ! -z ${script_loc} ]
then
    arr[$1]=$2
fi
#echo ${arr[@]}

if [ -z ${location} ]
then
    echo 'This script requires at least one  parameter' ${location}
    echo 'Possible values are: ' ${!arr[@]}
    echo 'Please rerun this script with one of these values or use:'
    echo 'python SN_MAF/setups/setup_metric.sh MYENV full_path_to_release_stack'
    echo 'where the stack release should include the lsst_sim package'
    return
fi


if [[ ! ${arr[$location]} ]]
then
    echo 'Problem here. We do not know where you would like to run'
    echo 'and which setup (stack) script you would like to use' 
    echo 'Try the following command'
    echo 'python sn_maf/setups/setup_metric.sh MYENV full_path_to_release_stack'
    echo 'where the stack release should include the lsst_sim package'
else
    thescript=${arr[$location]}
    source ${thescript}
    setup lsst_sims
fi
