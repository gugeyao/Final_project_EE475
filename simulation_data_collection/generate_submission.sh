#!/bin/bash

# Input parameters
jobID=$1
program_name=$2

# Create the submission script
submission_script="demo_${jobID}.sh"
cat <<EOF > $submission_script
#!/bin/bash
#SBATCH -A p32595
#SBATCH -p short
#SBATCH -t 04:00:00
#SBATCH -N 1
#SBATCH --mem=2G
#SBATCH --ntasks-per-node=1
#SBATCH --error=error.txt
#SBATCH --output=out.txt
#SBATCH --job-name ${jobID}

# Load environment
echo "Job started at \$(date)" >> "time_${jobID}.log"

# Run 4 simulations with different parameters
for ((sim=1; sim<=10; sim++)); do
    # Generate a unique seed
    seed=\$(date +%s%N)
    # Generate random parameters using the seed
    speed=\$(awk -v seed=\$seed 'BEGIN{srand(seed); printf "%.3f", 1 + rand() * 4}')  # Speed: 1 to 5
    sleep 1
    seed=\$(date +%s%N)
    turning_rate_degree=\$(awk -v seed=\$((seed + 1)) 'BEGIN{srand(seed); printf "%.3f", 20 + rand() * 40}')  # Turning rate: 20 to 60
    sleep 1
    seed=\$(date +%s%N)
    repulsion_radius=\$(awk -v seed=\$((seed + 2)) 'BEGIN{srand(seed); printf "%.3f", rand() * 4}')  # ZOR: 0 to 4
    sleep 1
    seed=\$(date +%s%N)
    orientation_width=\$(awk -v seed=\$((seed + 3)) 'BEGIN{srand(seed); printf "%.3f", rand() * 8}')  # ZOO: 0 to 8

    # Create a directory for this simulation's results
    sim_folder="speed_\${speed}_turn_\${turning_rate_degree}_zor_\${repulsion_radius}_zoo_\${orientation_width}"
    mkdir -p \$sim_folder
    if [[ ! -d \$sim_folder ]]; then
        echo "Failed to create directory: \$sim_folder" >> "time_${jobID}.log"
        continue
    fi
    cp $program_name \$sim_folder
    cd \$sim_folder || { echo "Failed to enter directory: \$sim_folder"; exit 1; }
    # Run the simulation (use env python)
    python3 $program_name \$speed \$turning_rate_degree \$repulsion_radius \$orientation_width > output.txt 2> error.txt
    cd ..
    echo "Simulation \$sim finished with parameters: speed=\${speed}, turning=\${turning_rate_degree}, zor=\${repulsion_radius}, zoo=\${orientation_width}" >> "time_${jobID}.log"
done

echo "Job finished at \$(date)" >> "time_${jobID}.log"
EOF

# Make the script executable and submit the job
chmod +x $submission_script
job_id=$(sbatch $submission_script)
echo "Submitted job $job_id for $jobID"
