#!/usr/bin/env bash

source load_modules.sh

# domain sizes
declare -a sizes=(256 512 1024 2048 4096)

# generate input files
for size in ${sizes[*]} 
do
  echo "Generating input data ${size}x${size}..."
  ../build/data_generator -o input_data_${size}.h5 -n ${size} -H 100 -C 20
done
