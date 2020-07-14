#!/bin/bash

#File: tree-md

#tree=$(tree -L $1 -tf --noreport -I '*~' --charset ascii $2 |
#       sed -e 's/| \+/  /g' -e 's/[|`]-\+/ */g' -e 's:\(* \)\(\(.*/\)\([^/]\+\)\):\1[\4](\2):g')

tree=$(tree -L $2 -tf --noreport -I '*~' --charset ascii $3 |
	      sed -e 's/| \+/  /g' -e 's/[|`]-\+/ */g' -e 's:\(* \)\(\(.*/\)\([^/]\+\)\):\1\4:g')

printf "# $1 \n\n${tree}"
