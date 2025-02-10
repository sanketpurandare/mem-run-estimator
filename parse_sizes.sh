#!/bin/bash

# assumes relevant modules have been loaded, e.g., via initconda.sh
HOME="/n/holylabs/LABS/idreos_lab/Users/azhao"

DATA_DIR=$HOME/mem-run-estimator/
FINAL_DIR=$DATA_DIR/final_models

rm -rf $FINAL_DIR
mkdir -p $FINAL_DIR

SCRIPT_DIR=$HOME/gpu_profiling/scripts

DATA_SOURCES=("conv_next_sizes" "vit_sizes")
OP_TYPES=("addmm" "bmm" "mm" "sdpea" "sdpfa" "conv")

# Iterate over each data source and operation type
for source in "${DATA_SOURCES[@]}"; do
    for op in "${OP_TYPES[@]}"; do
        echo "Processing op_type=$op from source=$source"
        python3 "$SCRIPT_DIR/parse.py" --op_type "$op" --path "$DATA_DIR/$source" --save_path "$FINAL_DIR" --overwrite "a"
    done
done
