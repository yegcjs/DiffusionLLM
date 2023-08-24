while [ $# -gt 0 ]; do
  case "$1" in
    --arg_0=*)
      ARG_0="${1#*=}"
      ;;
    --arg_1=*)
      ARG_1="${1#*=}"
      ;;
    *)
      printf "***************************\n"
      printf "* Error: Invalid argument '${*}'\n"
      printf "***************************\n"
      exit 1
  esac
  shift
done


for file in mmlu_0-shot mmlu_2-shot \
            bbh_nlp_0-shot bbh_nlp_2-shot \
            iwslt14_0-shot iwslt14_2-shot \
            tydiqa_0-shot tydiqa_1-shot \
            mgsm_0-shot_en mgsm_3-shot_en \
            mgsm_0-shot_de mgsm_3-shot_de
do
  echo $file
done