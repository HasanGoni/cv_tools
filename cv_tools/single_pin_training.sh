#!/bin/bash

INIT_MODEL_PATH="/home/i4a_dev.work/Easy_pin_detection/models"
INITIAL_MODEL_NAME="time_20_38_06_val_frGrnd0.9416_epoch_194.h5"

# where to save current model
MODEL_PATH="/home/hasan/Schreibtisch/projects/Infineon/projects/part2/Current_training_data20240111/models"
MODEL_NAME="sn_pin_data"

IMAGE_PATH="/home/hasan/Schreibtisch/projects/Infineon/projects/part2/Current_training_data20240111/sn_images"
MASK_PATH="/home/hasan/Schreibtisch/projects/Infineon/projects/part2/Current_training_data20240111/sn_masks"
BATCH_SIZE=200
IMAGE_HEIGHT=64
IMAGE_WIDTH=64

EPOCH_NUMBER=200
TRN_LAYER_NUM=-1

START_LEARNING_RATE=0.002
MAX_LEARNING_RATE=0.02
MIN_LEARNING_RATE=0.000002
RAMPUP_EPOCHS=8
SUSTAIN_EPOCHS=5


p_script="python single_pin_training.py\
            --initial_model_name "$INITIAL_MODEL_NAME"\
            --initial_model_path "$INIT_MODEL_PATH"\
            --model_path "$MODEL_PATH"\
            --model_name "$MODEL_NAME"\
            --image_path "$IMAGE_PATH"\
            --mask_path "$MASK_PATH"\
            --image_height "$IMAGE_HEIGHT"\
            --image_width "$IMAGE_WIDTH"\
            --batch_size "$BATCH_SIZE"
            --start_learning_rate "$START_LEARNING_RATE"\
            --max_learning_rate "$MAX_LEARNING_RATE"\
            --minimum_learning_rate "$MIN_LEARNING_RATE"\
            --rampup_epochs "$RAMPUP_EPOCHS"\
            --sustain_epochs "$SUSTAIN_EPOCHS"\
            "
# --pretrained
#bsub -Is -q gpu -gpu "num=1:j_exclusive=yes" -R "osrel==70 && ui==aiml-python" -P ma_ai -n 8 ${p_script}
$p_script