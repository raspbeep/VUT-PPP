#!/usr/bin/env bash
#SBATCH --account=DD-23-135
#SBATCH --job-name=PPP_PROJ01_PROF
#SBATCH -p qcpu
#SBATCH -t 01:00:00
#SBATCH -N 4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --distribution=block:cyclic:cyclic,NoPack

source load_modules.sh

STDOUT_FILE="run_scorep_profile_out.csv"
STDERR_FILE="run_scorep_profile_err.txt"
BINARY_PATH="../build_prof/ppp_proj01"

# Clear the stdout and stderr files
rm -f $STDOUT_FILE $STDERR_FILE

DISK_WRITE_INTENSITY=50

export OMP_NUM_THREADS=8

export SCOREP_ENABLE_PROFILING=true
export SCOREP_TOTAL_MEMORY=2G
srun -n 16 $BINARY_PATH -B    -n 100 -t $OMP_NUM_THREADS -m 1 -w $DISK_WRITE_INTENSITY -i input_data_1024.h5 >> $STDOUT_FILE 2>> $STDERR_FILE
srun -n 16 $BINARY_PATH -b -g -n 100 -t $OMP_NUM_THREADS -m 1 -w $DISK_WRITE_INTENSITY -i input_data_1024.h5 >> $STDOUT_FILE 2>> $STDERR_FILE

export SCOREP_ENABLE_TRACING=true
export SCOREP_FILTERING_FILE=ppp_scorep_filter.flt
srun -n 16 $BINARY_PATH -B    -n 100 -t $OMP_NUM_THREADS -m 1 -w $DISK_WRITE_INTENSITY -i input_data_1024.h5 >> $STDOUT_FILE 2>> $STDERR_FILE
srun -n 16 $BINARY_PATH -b -g -n 100 -t $OMP_NUM_THREADS -m 1 -w $DISK_WRITE_INTENSITY -i input_data_1024.h5 >> $STDOUT_FILE 2>> $STDERR_FILE
