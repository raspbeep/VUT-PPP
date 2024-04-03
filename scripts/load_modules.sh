#!/usr/bin/env bash

ml CMake/3.22.1-GCCcore-11.2.0 intel/2021b HDF5/1.12.1-intel-2021b-parallel Score-P/8.0-iimpi-2021b

# export OpenMP_ROOT=$(brew --prefix)/opt/libomp
# cmake -DCMAKE_CXX_COMPILER=clang++ -DLOGIN=xkrato61 -Bbuild -S.


# build on ubuntu
# rm -rf build && cmake -DCMAKE_CXX_COMPILER=g++ -DCMAKE_BUILD_TYPE=Debug -DLOGIN=xkrato61 -Bbuild -S. && cmake --build build --config Debug
# ./build/data_generator && mpiexec -np 4 ./build/ppp_proj01 -m 1 -d -n 100 -i ppp_input_data.h5 -o output.h5

# disable asking for supersuer
# echo 0| sudo tee /proc/sys/kernel/yama/ptrace_scope