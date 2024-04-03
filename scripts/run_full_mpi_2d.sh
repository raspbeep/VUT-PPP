#!/usr/bin/env bash
#SBATCH --account=DD-23-135
#SBATCH --job-name=PPP_PROJ01_MPI
#SBATCH -p qcpu
#SBATCH -t 05:00:00
#SBATCH -N 4
#SBATCH --ntasks-per-node=32
#SBATCH --distribution=block:block:block,Pack

source load_modules.sh

declare -a SIZES=(256 512 1024 2048 4096)
declare -a PROCESSES=(1 16 32 64 128)

STDOUT_FILE="run_full_mpi_2d_out.csv"
STDERR_FILE="run_full_mpi_2d_err.txt"
BINARY_PATH="../build/ppp_proj01"

# Clear the stdout and stderr files
rm -f $STDOUT_FILE $STDERR_FILE

USER_SCRATCH_PATH=/scratch/project/dd-23-135/$USER

mkdir -p $USER_SCRATCH_PATH

OUT_FILE_PATH=$USER_SCRATCH_PATH/$SLURM_JOBID

mkdir -p $OUT_FILE_PATH

# Doplnte vhodne nastavenie Lustre file system #
################################################

################################################

DISK_WRITE_INTENSITY=50

for procs in ${PROCESSES[*]}; do
    for size in ${SIZES[*]}; do
        B="-b"
        
        if [ "$procs" -eq 1 ]; then
            nnodes=1
            n_iters=`expr $((2000000/$size))`
            modeP2P=0
            modeRMA=0
            if [ "$size" -eq 256 ]; then
                B="-B"
            fi
        else
            nnodes=$((procs/SLURM_NTASKS_PER_NODE))
            if [ "$nnodes" -eq 0 ]; then
                nnodes=1
            fi

            n_iters=`expr $((20000000/$size))`
            modeP2P=1
            modeRMA=2
        fi

        INPUT=input_data_$size.h5
        OUTPUT=$OUT_FILE_PATH/${size}x${size}_out_mpi_2d.h5
        
        srun -N $nnodes -n $procs $BINARY_PATH $B -g    -n $n_iters -m $modeP2P -w $DISK_WRITE_INTENSITY -i $INPUT            >> $STDOUT_FILE 2>> $STDERR_FILE
        srun -N $nnodes -n $procs $BINARY_PATH -b -g    -n $n_iters -m $modeP2P -w $DISK_WRITE_INTENSITY -i $INPUT -o $OUTPUT >> $STDOUT_FILE 2>> $STDERR_FILE
        srun -N $nnodes -n $procs $BINARY_PATH -b -g -p -n $n_iters -m $modeP2P -w $DISK_WRITE_INTENSITY -i $INPUT -o $OUTPUT >> $STDOUT_FILE 2>> $STDERR_FILE

        srun -N $nnodes -n $procs $BINARY_PATH -b -g    -n $n_iters -m $modeRMA -w $DISK_WRITE_INTENSITY -i $INPUT            >> $STDOUT_FILE 2>> $STDERR_FILE
        srun -N $nnodes -n $procs $BINARY_PATH -b -g    -n $n_iters -m $modeRMA -w $DISK_WRITE_INTENSITY -i $INPUT -o $OUTPUT >> $STDOUT_FILE 2>> $STDERR_FILE
        srun -N $nnodes -n $procs $BINARY_PATH -b -g -p -n $n_iters -m $modeRMA -w $DISK_WRITE_INTENSITY -i $INPUT -o $OUTPUT >> $STDOUT_FILE 2>> $STDERR_FILE

        rm -f $OUTPUT
    done
done

rm -rf $OUT_FILE_PATH
