#!/bin/bash

# Directory to clean up
DIR_TO_CLEAN="/home/icb/yufan.xia/milad.bassil/tmp"

# Count how many actual jobs are running (excluding the header)
JOB_COUNT=$(squeue --me | awk 'NR>1' | wc -l)

if [ "$JOB_COUNT" -eq 0 ]; then
    echo "No jobs are running. Proceeding to clean the directory."

    # Delete all contents in the folder but not the folder itself
    rm -rf "$DIR_TO_CLEAN"/*
    echo "Directory cleaned."
else
    echo "You have $JOB_COUNT running job(s). Aborting cleanup."
fi
