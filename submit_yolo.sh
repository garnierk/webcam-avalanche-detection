#!/bin/bash
#SBATCH --job-name=webcam-yolo-bench
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-41%8
#SBATCH --cpus-per-task=8
#SBATCH --mem=62G
#SBATCH --time=0-5:00:00
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1

set -euo pipefail

#####################
# Environment setup
#####################
SOURCEDIR="$HOME/links/projects/def-boakes/garnierk/webcam-avalanche-detection"
export TORCH_HOME="$SOURCEDIR/torch_cache"
export OMP_NUM_THREADS=1

mkdir -p "$SOURCEDIR/logs"
mkdir -p "$SLURM_TMPDIR"
cd "$SLURM_TMPDIR"

module purge
module load gcc opencv/4.8.1 python/3.10 scipy-stack/2024a cuda cudnn httpproxy

virtualenv --no-download .venv
source .venv/bin/activate

cd "$SOURCEDIR"
python -m pip install --no-index -r requirements.txt
python -m pip install -e .

mkdir -p "$SLURM_TMPDIR/data"
unzip -q -d "$SLURM_TMPDIR/data" "$SOURCEDIR/uibk_avalanches.zip"

python utils/train_test_split.py --source-dir $SLURM_TMPDIR/data --output-dir $SLURM_TMPDIR/data
#####################
# User config
#####################
DATADIR="${DATADIR:-$SLURM_TMPDIR/data}"
WEIGHTS_DIR="${WEIGHTS_DIR:-$SOURCEDIR/weights}"

PYTHON="$SLURM_TMPDIR/.venv/bin/python"
TRAIN_SCRIPT="$SOURCEDIR/segmentation/train.py"
PREP_SCRIPT="$SOURCEDIR/segmentation/data/scripts/generate_yolo_annotations.py"

PROJECT="paper_yolo"
HYP="$SOURCEDIR/segmentation/data/hyps/hyp.scratch_withConf.yaml"

DEVICE=0
BATCH=8
WORKERS=8
EPOCHS=180
PATIENCE=25

#####################
# Comet
#####################
export COMET_PROJECT_NAME="paper_yolo"
export COMET_AUTO_LOG_PARAMETERS=1
export COMET_AUTO_LOG_METRICS=1
export COMET_AUTO_LOG_MODEL=1

#####################
# Prepare labels + splits on the cluster
#####################
cd "$DATADIR"
mkdir -p "$SLURM_TMPDIR/yolo_split"

DATASET_YAML_DIR="$SLURM_TMPDIR/segmentation/data/"

mkdir -p $DATASET_YAML_DIR

"$PYTHON" "$PREP_SCRIPT" --data-dir "$DATADIR" --output-dir "$SLURM_TMPDIR/yolo_split" --source-dir "$SLURM_TMPDIR"

echo "=== CHECK YAML ==="
cat "$SLURM_TMPDIR/segmentation/data/avalanchesplit100.yaml"

echo "=== CHECK CLASSES IN LABELS ==="
find "$SLURM_TMPDIR" -name "*.txt" -path "*labels*" -print0 | \
xargs -0 awk '{print $1}' | sort -n | uniq -c

#####################
# Experiment grid
#####################
MODELS=(
  "yolo3_norm:yolov3.pt"
  "yolo3_spp:yolov3-spp.pt"
  "yolo3_tiny:yolov3-tiny.pt"
  "yolo5_nseg:yolov5n-seg.pt"
  "yolo5_sseg:yolov5s-seg.pt"
  "yolo5_mseg:yolov5m-seg.pt"
  "yolo5_lseg:yolov5l-seg.pt"
)

IMSIZES=(448 896)
SEEDS=(50 100 150)

N_MODELS=${#MODELS[@]}
N_IMSIZES=${#IMSIZES[@]}
N_SEEDS=${#SEEDS[@]}

TASK_ID=${SLURM_ARRAY_TASK_ID}

MODEL_IDX=$(( TASK_ID / (N_IMSIZES * N_SEEDS) ))
REM=$(( TASK_ID % (N_IMSIZES * N_SEEDS) ))
IMSIZE_IDX=$(( REM / N_SEEDS ))
SEED_IDX=$(( REM % N_SEEDS ))

MODEL_ENTRY="${MODELS[$MODEL_IDX]}"
MODEL_TAG="${MODEL_ENTRY%%:*}"
WEIGHT_FILE="${MODEL_ENTRY##*:}"

IMSIZE="${IMSIZES[$IMSIZE_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"

RUN_NAME="${MODEL_TAG}_${IMSIZE}_p${PATIENCE}s${SEED}"
WEIGHTS_PATH="$WEIGHTS_DIR/$WEIGHT_FILE"

DATASET_YAML="$SLURM_TMPDIR/segmentation/data/avalanchesplit${SEED}.yaml"
echo "TASK_ID       : $TASK_ID"
echo "MODEL_TAG     : $MODEL_TAG"
echo "WEIGHTS_PATH  : $WEIGHTS_PATH"
echo "IMSIZE        : $IMSIZE"
echo "SEED          : $SEED"
echo "RUN_NAME      : $RUN_NAME"
echo "DATASET_YAML  : $DATASET_YAML"

if [ ! -f "$WEIGHTS_PATH" ]; then
    echo "ERROR: missing weights: $WEIGHTS_PATH"
    exit 1
fi

if [ ! -f "$DATASET_YAML" ]; then
    echo "ERROR: missing dataset yaml: $DATASET_YAML"
    find "$DATADIR" -maxdepth 3 -name "avalanchesplit*.yaml" -print || true
    exit 1
fi

export COMET_EXPERIMENT_NAME="$RUN_NAME"

cd "$SOURCEDIR"
"$PYTHON" "$TRAIN_SCRIPT" \
    --name "$RUN_NAME" \
    --weights "$WEIGHTS_PATH" \
    --device "$DEVICE" \
    --batch-size "$BATCH" \
    --workers "$WORKERS" \
    --epochs "$EPOCHS" \
    --project "$PROJECT" \
    --hyp "$HYP" \
    --patience "$PATIENCE" \
    --img "$IMSIZE" \
    --data "$DATASET_YAML"
