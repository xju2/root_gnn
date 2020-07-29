#!/bin/bash

function get_code(){
	NUMBER=$(echo $@ | tr -dc '0-9')
	echo $NUMBER
}

nJobs=5

EXE="run.sh"

start=$(sbatch ${EXE})
JOBID=$(get_code ${start})
echo "Submitted batch job ${JOBID}"

for ((i=0; i < ${nJobs}; i+=1)); do
	JOBID=$(get_code $(sbatch --dependency=afterok:${JOBID} ${EXE}))
	echo "Submitted batch job ${JOBID}"
done
