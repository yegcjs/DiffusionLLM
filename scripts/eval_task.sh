CACHE_DIR="cache"
METRIC="bleu"
LB="1"
MBR="1"
BLEU_TOKENIZE="13a"
MAX_ITER="50"
HALF_PREC="--bf16 --bf16_full_eval" # replace this with "--fp16 --fp16_full_eval" if your device does not support bf16
while [ $# -gt 0 ]; do
  case "$1" in
    --data_path=*)
        DATA_PATH="${1#*=}"
        ;;
    --ckpt=*)
        CKPT_PATH="${1#*=}"
        ;;
    --output_dir=*)
        OUTPUT_DIR="${1#*=}"
        ;;
    --mini_bsz=*)
        MINI_BSZ="${1#*=}"
        ;;
    --metric=*)
        METRIC="${1#*=}"
        ;;
    --length_beam=*)
        LB="${1#*=}"
        ;;
    --max_iter=*)
        MAX_ITER="${1#*=}"
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

CKPT=`ls $CKPT_PATH | grep checkpoint-* | tail -1`

python3 -m torch.distributed.launch --use-env --nproc_per_node=8 --nnodes=1 \
    --node_rank=0 --master_addr=127.0.0.1 --master_port=12345 \
    src/evaluate.py --ckpt_args_file $CKPT_PATH/args.json --resume_from_checkpoint $CKPT_PATH/$CKPT \
    --prediction_write_to $OUTPUT_DIR --max_length 512 \
    --output_dir $CKPT_PATH --per_device_eval_batch_size $MINI_BSZ \
    --remove_unused_column False $HALF_PREC \
    --dataset_type fairseq --data_path $DATA_PATH --eval_metric $METRIC \
    --mbr $MBR --length_beam $LB --argmax_decoding --bleu_tokenize $BLEU_TOKENIZE \
    --return_history --max_iterations $MAX_ITER $EXTRA 