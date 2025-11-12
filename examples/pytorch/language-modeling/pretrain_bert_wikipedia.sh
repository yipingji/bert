#!/bin/bash

# BERT Pretraining on English Wikipedia
# Uses 4 GPUs with bf16 mixed precision

# Select which 4 GPUs to use (change these numbers as needed)
# Options: 0,1,2,3 or 2,3,4,5 or 4,5,6,7, etc.
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Number of GPUs (should match CUDA_VISIBLE_DEVICES count)
NUM_GPUS=8

# Training hyperparameters
# For pretraining from scratch, we use model_type instead of loading a checkpoint
# We still need the tokenizer from a pretrained model
TOKENIZER_NAME="google-bert/bert-base-uncased"
MODEL_TYPE="bert"
DATASET_NAME="wikimedia/wikipedia"
DATASET_CONFIG="20231101.en"
OUTPUT_DIR="../../bert-wikipedia-pretrained"

# Batch size settings
PER_DEVICE_BATCH_SIZE=32
GRAD_ACCUM_STEPS=1
# Effective batch size = 32 * 1 * 8 = 256

# Training settings
MAX_SEQ_LENGTH=512
MLM_PROBABILITY=0.15
LEARNING_RATE=1e-4
MAX_STEPS=800000  # Train for 100k steps instead of epochs
# NUM_EPOCHS=40  # Disabled: using max_steps instead
WARMUP_STEPS=10000

# Logging and saving
LOGGING_STEPS=500
SAVE_STEPS=20000
SAVE_TOTAL_LIMIT=10

echo "========================================="
echo "BERT Pretraining from Scratch on Wikipedia"
echo "========================================="
echo "GPUs: $CUDA_VISIBLE_DEVICES ($NUM_GPUS GPUs)"
echo "Model type: $MODEL_TYPE (training from scratch)"
echo "Tokenizer: $TOKENIZER_NAME"
echo "Dataset: $DATASET_NAME ($DATASET_CONFIG)"
echo "Output: $OUTPUT_DIR"
echo "Learning rate: $LEARNING_RATE"
echo "Max training steps: $MAX_STEPS"
echo "Batch size per device: $PER_DEVICE_BATCH_SIZE"
echo "Gradient accumulation: $GRAD_ACCUM_STEPS"
echo "Effective batch size: $((PER_DEVICE_BATCH_SIZE * GRAD_ACCUM_STEPS * NUM_GPUS))"
echo "========================================="

# Run training with torchrun
torchrun --nproc_per_node=$NUM_GPUS run_mlm.py \
    --model_type $MODEL_TYPE \
    --tokenizer_name $TOKENIZER_NAME \
    --dataset_name $DATASET_NAME \
    --dataset_config_name $DATASET_CONFIG \
    --do_train \
    --do_eval \
    --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
    --per_device_eval_batch_size 128 \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --learning_rate $LEARNING_RATE \
    --max_steps $MAX_STEPS \
    --max_seq_length $MAX_SEQ_LENGTH \
    --mlm_probability $MLM_PROBABILITY \
    --output_dir $OUTPUT_DIR \
    --bf16 \
    --warmup_steps $WARMUP_STEPS \
    --save_steps $SAVE_STEPS \
    --logging_steps $LOGGING_STEPS \
    --save_total_limit $SAVE_TOTAL_LIMIT \
    --dataloader_num_workers 4 \
    --preprocessing_num_workers 32 \
    --weight_decay 0.01 \
    --save_strategy steps \
    --eval_strategy steps \
    --eval_steps 5000 \
    --load_best_model_at_end \
    --metric_for_best_model eval_loss \
    --report_to tensorboard

echo ""
echo "========================================="
echo "Training complete!"
echo "Model saved to: $OUTPUT_DIR"
echo "========================================="
