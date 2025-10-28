#!/bin/bash
#SBATCH -A p32595
#SBATCH -p short
#SBATCH -t 04:00:00
#SBATCH -N 1
#SBATCH --mem=2G
#SBATCH --ntasks-per-node=1
#SBATCH --error=error_%j.txt
#SBATCH --output=out_%j.txt

# Parameters
num_jobs=100  # Number of submissions
program_name='run_simulation.py'

for ((job=1; job<=num_jobs; job++)); do
    # Generate the submission script for the job
    ./generate_submission.sh $job $program_name
done
