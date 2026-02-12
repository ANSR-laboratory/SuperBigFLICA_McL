#!/bin/bash
#SBATCH --job-name=SupBigFLICA         # Job name
#SBATCH --output=%x_%j.out             # Standard output file name (%x: job name, %j: job ID)
#SBATCH --error=%x_%j.err              # Standard error file name
#SBATCH --cpus-per-task=2              # Number of CPUs per task (reduced based on low CPU utilization)
#SBATCH --time=72:00:00                # Time limit (hh:mm:ss)
#SBATCH --mem=100G                     # Memory requirement (increased based on observed utilization)
#SBATCH --mail-type=ALL                # Email notifications for job start, end, and failure
#SBATCH --mail-user=ycheng23@mgh.harvard.com

# Set up output directory
BASE_OUTPUT_DIR="/data/qnilab/AD_NPS_R01_2022/ADNI_Data_Fusion/SBF_Outputs/SBF_ADNI_CDR/CDRSB" # Replace with your desired directory for output
OUTPUT_DIR="$BASE_OUTPUT_DIR/job_$(date +'%Y%m%d_%H%M%S')"
mkdir -p "$OUTPUT_DIR"  # Create the directory if it doesn't exist
echo "All outputs will be saved to: $OUTPUT_DIR"

# Generate a log file name with the current date and time
LOG_DIR="/home/ycheng23/SuperBigFLICA_python/Latest_Versions_14Nov2024/time_logs"  # Replace with your desired directory for logs
mkdir -p "$LOG_DIR"  # Create the directory if it doesn't exist
LOG_FILE="$LOG_DIR/run_log_$(date +'%Y%m%d_%H%M%S').log"
ERROR_FILE="$OUTPUT_DIR/error_log$(date +'%Y%m%d_%H%M%S').err"

# Record the start time
START_TIME=$(date +"%Y-%m-%d %H:%M:%S")
echo "Script started at: $START_TIME" | tee -a "$LOG_FILE"

# Activate virtual environment
# VENV_DIR="/data/qnilab/Virt_Envs/bigflica"  
VENV_DIR="/data/qnilab/AD_NPS_R01_2022/ADNI_Data_Fusion/.myenv"  

source "$VENV_DIR/bin/activate"
echo "Virtual environment activated: $VENV_DIR" | tee -a "$LOG_FILE"

# Load required module
# module purge
# module add fsl              # Load FSL (version 6.0.7.4 assumed)
# module add freesurfer       # Load FreeSurfer (version 7.3.2 assumed)
echo "Modules fsl and freesurfer loaded." | tee -a "$LOG_FILE"

# Start a timer and run the Python script
SECONDS=0  # Initialize a timer

# Uncomment the Python script you want to run
python3 -u SBF.py


# Record the end time
END_TIME=$(date +"%Y-%m-%d %H:%M:%S")
echo "Script ended at: $END_TIME" | tee -a "$LOG_FILE"

# Calculate and log the elapsed time
ELAPSED_TIME=$SECONDS
echo "Total computing time: $(($ELAPSED_TIME / 3600)) hours, $((($ELAPSED_TIME / 60) % 60)) minutes, $(($ELAPSED_TIME % 60)) seconds" | tee -a "$LOG_FILE"

# Notify the user where the log file is saved
echo "Log file saved to: $LOG_FILE"
