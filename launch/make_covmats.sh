#!/bin/bash --login

###
#job name
#SBATCH --job-name=null_examples
#job stdout file
#SBATCH --output=null_examples.out.%J
#job stderr file
#SBATCH --error=null_examples.err.%J
#maximum job time in D-HH:MM
#SBATCH --time=0-72:00
#number of parallel processes (tasks) you are requesting - maps to MPI processes
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=191000
#SBATCH -p c_compute_physics1
#SBATCH -A scw1361
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=harrisoni@cardiff.ac.uk
###

module load Miniforge3

conda deactivate
conda activate /shared/home1/c.c1025819/miniforge3/envs/actxdes

export ACTDIR=/scratch/SCWF00082/shared/harrisoni

export OMP_NUM_THREADS=8

which python >> $SLURM_SUBMIT_DIR/pyver.txt
python $SLURM_SUBMIT_DIR/scripts/measure_sims_spectra.py
