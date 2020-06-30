#!/bin/bash

OS=$1

if [ $OS != 'Linux' ] && [ $OS != 'Mac' ]
then
    echo 'Unknown Operating System'
    echo 'Possibilities are Linux or Mac'
    echo 'Can not go further'
    set -e
fi


dir_rel=''
if [ $OS == 'Linux' ] 
then
    dir_rel='/cvmfs/sw.lsst.eu/linux-x86_64'
fi

if [ $OS == 'Mac' ] 
then
    dir_rel='/cvmfs/sw.lsst.eu/darwin-x86_64'
fi

#get the shell
myshell=$(echo $SHELL | rev | cut -d "/" -f1  | rev)

echo 'the shell is' $myshell

myshell='sh'

echo $myshell
if [ $myshell == 'sh' ] 
then
    myshell='bash'
fi

echo $myshell

# get distribs
array=($(ls -dtr $dir_rel/lsst_sims/*))

# print possible distribs

len=${#array[*]}

i=0
while [ $i -lt $len ]; do
echo "$i: ${array[$i]}"
let i++
done

# choose the latest-1 release installed

release=${array[$len-6]}

#Fill a file with the release chosen

echo "release: ${release}" >& current_release.yaml

# make the setups
echo 'Setting-up' ${release}
source ${release}/loadLSST.$myshell
setup lsst_sims
