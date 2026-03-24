# ============================
# SGE Job Configuration Block
# ============================

# Hard Runtime Limit: Job will be terminated if it runs longer than this.
# Format: [hours]:[minutes]:[seconds]. Longer runtimes may result in lower
# scheduling priority.
#$ -l h_rt=10:0:0

# Memory Request: Estimated total memory (RAM) usage.
# Used by the scheduler for resource allocation. Jobs requesting less memory
# are prioritized.
#$ -l tmem=10G

# Working Directory: Set to the directory from which the job is submitted.
#$ -wd /home/smartine/scripts/example/

# Job Name: Useful for tracking and log file naming
# (e.g., TestPythonScript.o<JOB_ID>).
#$ -N TestPythonScript

# GPU usage: Request a GPU. Safe to leave in any case.
$ -l gpu=true

# Node Selection: Run the job on a specific node (minella).
# Do not modify unless explicitly required.
$ -l hostname=minella

# Project Account: Assigns the job to the PNLab project group.
# Do not modify unless you are part of a different project.
$ -P pnlab

# Output Handling: Merge standard output (stdout) and error (stderr) into a
# single file.
#$ -j y

# Shell Specification: Use Bash to interpret this script.
$ -S /bin/bash

# ============================
# Execution Section
# ============================

# Print the name of the compute node (useful for confirmation).
hostname

# Log the job start time.
date

# GPU Information: Run if your job uses a GPU. Safe to leave in any case.
nvidia-smi


# Load the required Python environment.
# Note: Python 3.9 is slightly outdated. Consider a newer version if supported.
# See available modules at: https://hpc.cs.ucl.ac.uk/software/
# source /share/apps/source_files/python/python-3.9.5.source

source dl_env/bin/activate

# Execute your Python script.
python3 nn_main.py

# Log the job completion time.
date
