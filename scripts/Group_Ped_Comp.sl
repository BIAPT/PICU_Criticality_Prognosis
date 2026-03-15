#!/bin/bash -l
#SBATCH --job-name=NewPedeatricComplex
#SBATCH --output=/home/dnewman/...
#SBATCH --error=/home/dnewman/...
#SBATCH --account=def-sblain
#SBATCH --time=0-01:45:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=30G
#SBATCH --mail-user=x6s...
#SBATCH --mail-type=ALL
#SBATCH --array=0-331

module load python/3.10.2

CURRENT_DIR=$(pwd)
source ${CURRENT_DIR}/neural_complexity/bin/activate

# Prepare to process files
FILES=(${CURRENT_DIR}/preprocessing/epochs_2/epochs/*.fif)
FILE_TO_PROCESS=${FILES[$SLURM_ARRAY_TASK_ID]}

# Check if file exists and then process
if [ -f "$FILE_TO_PROCESS" ]; then
    FILENAME=$(basename -- "$FILE_TO_PROCESS")
    CONDITION="${FILENAME::-4}"  # Removes the last 4 characters (e.g., '.fif')

    # Execute the Python script with specified options
    python ${CURRENT_DIR}/scripts/neural_complexity.py $FILE_TO_PROCESS --save --out_dir ${CURRENT_DIR}/source/source_2/ --condition "$CONDITION" --standardize --impute --entropy --complexity --fractal --dfa --power --verbose --PDF

else
    echo "File for TASK ID $SLURM_ARRAY_TASK_ID does not exist."
fi