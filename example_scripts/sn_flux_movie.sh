#!/bin/bash

x1=${1:-0}
color=${2:-0}
z=${3:-0.8}

echo "processing data"
cmd="python run_scripts/simulation/run_flux_spectra_sn.py --x1=$x1 --color=$color --z=$z --sed=1 --outDirDisplay=../plot_flux_spectra --phases_sed=None"

echo $cmd
$cmd

echo "creating movie"

cmd='python run_scripts/utils/make_movie_from_png.py --figDir=../plot_flux_spectra --prefix=flux_spectra --outName=flux_spectra --extens=png'

$cmd

echo "displaying movie"

cmd="vlc movies/flux_spectra.mp4"

$cmd
