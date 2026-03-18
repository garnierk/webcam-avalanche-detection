#!/usr/bin/bash

set -euo pipefail

#####################
# Usage
#####################
# ./architecture_comparison.sh /path/to/data /path/to/project
# ou
# DATADIR=/path/to/data ./architecture_comparison.sh

DATADIR="${1:-${DATADIR:-.data}}"
PROJECTDIR="${2:-${PROJECTDIR:-.}}"
#####################
# Training constants
#####################
PYTHON=python3
device=0
batch=16
workers=8
epochs=180
patience=25
project=paper_yolo
hyp=hyp.scratch_withConf.yaml
train=segmentation/train.py
WEIGHTS_DIR="$PROJECTDIR/weights"
SAVE_DIR="$PROJECTDIR/segmentation/experiments/architecture/runs"

mkdir -p "$SAVE_DIR"

# Script de préparation dataset/labels/splits
PREP_SCRIPT="prepare_yolo_data.py"

#####################
# Comet config
#####################
export COMET_PROJECT_NAME="paper_yolo"
# export COMET_API_KEY="YOUR_API_KEY"
# export COMET_WORKSPACE="YOUR_WORKSPACE"
export COMET_AUTO_LOG_PARAMETERS=1
export COMET_AUTO_LOG_METRICS=1
export COMET_AUTO_LOG_MODEL=1

#####################
# Préparation dataset
#####################
echo "==> DATADIR: $DATADIR"
$PYTHON "$PREP_SCRIPT" --data-dir "$DATADIR"

#####################
# Common train params
#####################
trainParams="--device $device --batch-size $batch --workers $workers --epochs $epochs --project $project --hyp $hyp --patience $patience"

run_exp () {
    local model_tag=$1
    local weights=$2
    local imSize=$3
    local seed=$4

    local name="${model_tag}_${imSize}_p${patience}s${seed}"
    local dataset_yaml="${DATADIR}/yolo_split/avalanchesplit${seed}.yaml"

    export COMET_EXPERIMENT_NAME="$name"

    mkdir -p "$SAVE_DIR/$name"

    $PYTHON "$train" \
        --name "$name" \
        --weights "$WEIGHTS_DIR/$weights" \
        $trainParams \
        --img "$imSize" \
        --data "$dataset_yaml"\
        --save-dir "$SAVE_DIR"
}

for imSize in 448 896
do
    for seed in 50 100 150
    do
        # YOLOv3
        run_exp "yolo3_norm" "yolov3.pt" "$imSize" "$seed"
        run_exp "yolo3_spp"  "yolov3-spp.pt" "$imSize" "$seed"
        run_exp "yolo3_tiny" "yolov3-tiny.pt" "$imSize" "$seed"

        # YOLOv5-seg
        run_exp "yolo5_nseg" "yolov5n-seg.pt" "$imSize" "$seed"
        run_exp "yolo5_sseg" "yolov5s-seg.pt" "$imSize" "$seed"
        run_exp "yolo5_mseg" "yolov5m-seg.pt" "$imSize" "$seed"
    done
done
