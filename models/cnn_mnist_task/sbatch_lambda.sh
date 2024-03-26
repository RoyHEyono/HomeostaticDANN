#!/usr/bin/env bash

lmbda=(0 1e-2 1e-1)

for lda in "${lmbda[@]}"
do
    sbatch lambda_job.sh $lda
done