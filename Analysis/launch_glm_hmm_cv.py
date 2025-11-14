import os
import subprocess
import argparse
import shutil

def generate_batch_file(n_states):
    batch_content = f"""#!/bin/bash
#
#SBATCH -N 1   # number of nodes
#SBATCH -n 10   # number of cores
#SBATCH --mem 128 # memory pool for all cores
#SBATCH -o glm_hmm_cv_{n_states}.out # STDOUT
#SBATCH -e glm_hmm_cv_{n_states}.err # STDERR
#
hostname
source ~/.bashrc
conda activate sound_cat
python -u glm_gmm_cv.py --n_states {n_states} --nb_rand_initilizations 100
exit
"""
    with open(f"batch_{n_states}.sh", "w") as batch_file:
        batch_file.write(batch_content)

def launch_jobs(start_state, end_state):
    for n in range(start_state, end_state + 1):
        generate_batch_file(n)
        subprocess.run(["sbatch", f"batch_{n}.sh"])
    copy_batch_files(start_state, end_state)

def copy_batch_files(start_state, end_state):
    for n in range(start_state, end_state + 1):
        shutil.copy(f"batch_{n}.sh", "../Data")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and launch batch files for GLM-HMM cross-validation.")
    parser.add_argument("--start-state", type=int, help="Starting number of states")
    parser.add_argument("--end-state", type=int, help="Ending number of states")
    args = parser.parse_args()

    launch_jobs(args.start_state, args.end_state)

