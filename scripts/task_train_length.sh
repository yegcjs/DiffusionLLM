CACHE_DIR="cache"
HALF_PREC="--bf16 --bf16_full_eval" # replace this with "--fp16 --fp16_full_eval" if your device does not support bf16
METRIC="bleu"
while [ $# -gt 0 ]; do
  case "$1" in
    --data_path=*)
        DATA_PATH="${1#*=}"
        ;;
    --pretrained=*)
        PRETRAINED_MODEL="${1#*=}"
        ;;
    --cache_dir=*)
        CACHE_DIR="${1#*=}"
        ;;
    --output_dir=*)
        OUTPUT_DIR="${1#*=}"
        ;;
    --train_steps=*)
        MAX_TRAINING_STEPS="${1#*=}"
        ;;
    --eval_steps=*)
        EVAL_STEPS="${1#*=}"
        ;;
    --mini_bsz=*)
        MINI_BSZ="${1#*=}"
        ;;
    --metric=*)
        METRIC="${1#*=}"
        ;;
    --accum_step=*)
        GRADIENT_ACCUMULATION_STEP="${1#*=}"
        ;;
    --ds_config=*)
        DEEPSPEED_CONFIG="${1#*=}"
        ;;
    --ckpt=*)
        CHECKPOINT="${1#*=}"
        ;;
    --extra=*)
        EXTRA="${1#*=}"
        ;;
    *)
      printf "***************************\n"
      printf "* Error: Invalid argument '${*}'\n"
      printf "***************************\n"
      exit 1
  esac
  shift
done


python3 -m torch.distributed.launch --use-env --nproc_per_node=8 --nnodes=1 \
    --node_rank=0 --master_addr=127.0.0.1 --master_port=12345 \
    src/train.py --dataset_type fairseq \
    --data_path $DATA_PATH --oracle_length --pretrained $PRETRAINED_MODEL \
    --cache_dir $CACHE_DIR --output_dir $OUTPUT_DIR \
    --per_device_eval_batch_size $MINI_BSZ \
    --per_device_train_batch_size $MINI_BSZ \
    --max_steps $MAX_TRAINING_STEPS  --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEP \
    --evaluation_strategy "steps" --eval_steps $EVAL_STEPS  --eval_metric $METRIC --load_best_model_at_end --metric_for_best_model $METRIC --greater_is_better True \
    --logging_steps 100 --save_steps $EVAL_STEPS --report_to tensorboard \
    --log_level info --argmax_decoding --save_total_limit 1 --remove_unused_column False \
    --learning_rate 5e-5 --weight_decay 0.01 --adam_beta1 0.9 --adam_beta2 0.98 \
    --lr_scheduler_type inverse_sqrt --warmup_ratio 0.01 \
    $HALF_PREC --label_smoothing 0 --deepspeed $DEEPSPEED_CONFIG \
    --max_length 512 --train_length --finetune_from_model $CHECKPOINT $EXTRA