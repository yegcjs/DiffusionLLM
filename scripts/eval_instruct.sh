CACHE_DIR="cache"
LB="1"
MBR="1"
MAX_ITER="50"
HALF_PREC="--bf16 --bf16_full_eval" # replace this with "--fp16 --fp16_full_eval" if your device does not support bf16
while [ $# -gt 0 ]; do
  case "$1" in
    --data_paths=*)
        DATA_PATHS="${1#*=}"
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
    --dataset_type pair --data_path $DATA_PATHS \
    --mbr $MBR --length_beam $LB --argmax_decoding --report_to none \
    --return_history --max_iterations $MAX_ITER $EXTRA 

for file in mmlu_0-shot mmlu_2-shot \
            bbh-nlp_0-shot bbh-nlp_2-shot \
            iwslt14_0-shot iwslt14_2-shot \
            tydiqa_0-shot tydiqa_1-shot \
            mgsm_0-shot_en mgsm_3-shot_en \
            mgsm_0-shot_de mgsm_3-shot_de
do
    path=`ls $OUTPUT_DIR | grep $file`
    if [ -f $OUTPUT_DIR/$path ]; then
        data=`echo $file | tr "_" "\n" | head -1`
        shot=`echo $file | tr "_" "\n" | head -2 | tail -1`
        echo $shot
        printf "****** $file ******\n"
        lang="_"
        case $data in
            mmlu)
                data_path=data/instruct/mmlu/${shot}/val
                ;;
            bbh-nlp)
                data_path=data/instruct/bbh-nlp/${shot}/validation
                ;;
            iwslt14)
                data_path=data/instruct/iwslt14/$shot/test/deen.jsonl
                ;;
            tydiqa)
                data_path=data/instruct/tydiqa/${shot}/validation
                ;;
            mgsm)
                lang=`echo $file | tr "_" "\n" | head -3` 
                data_path=data/instruct/mgsm/${shot}/${lang}.jsonl
                ;;
        esac
        python3 scripts/metric_instruct.py --data $data --path $OUTPUT_DIR/$path \
            --data_path $data_path --result_path $OUTPUT_DIR/${data}.${lang}.${shot}.json
    fi
done
