#!/bin/bash
#SBATCH -A p32595              # Allocation
#SBATCH -p short                # Queue
#SBATCH -t 04:00:00             # Walltime/duration of the job
#SBATCH -N 1                    # Number of Nodes
#SBATCH --mem=2G               # Memory per node in GB needed for a job. Also see --mem-per-cpu
#SBATCH --ntasks-per-node=1     # Number of Cores (Processors)
#SBATCH --error=error.txt                                                       
#SBATCH --output=out.txt      
#SBATCH --job-name DEMO                                                   
echo started at `date`>> time.log
python3 $1
echo finished at `date`>> time.log
