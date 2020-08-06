#!/bin/bash

DESC_SIMS_VER=w_2020_15

CORISTR='cori'
NODENAME=`uname -n`
if [[ "$NODENAME" == *"$CORISTR"* ]]; then ON_CORI=1; fi

if [[ $ON_CORI != 1 ]]; then
    echo "This jupyter kernel set up is to be run at NERSC."
    exit 1
fi

echo "Setting up lsst_sims " $DESC_SIMS_VER 
INST_DIR=$PWD
source /global/common/software/lsst/cori-haswell-gcc/stack/setup_any_sims.sh $DESC_SIMS_VER
jupyter kernelspec install $INST_DIR/nersc/desc-sn-pipe --user

