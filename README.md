# Diffusion Language Models Can Perform Many Tasks with Scaling and Instruction-Finetuning

This repository contains official code for the paper [Diffusion Language Models Can Perform Many Tasks with Scaling and Instruction-Finetuning](https://arxiv.org/abs/2308.12219)

## Abstract

The recent surge of generative AI has been fueled by the generative power of diffusion probabilistic models and the scalable capabilities of large language models. Despite their potential, it remains elusive whether diffusion language models can solve general language tasks comparable to their autoregressive counterparts. This paper demonstrates that scaling diffusion models w.r.t. data, sizes, and tasks can effectively make them strong language learners. We build competent diffusion language models at scale by first acquiring knowledge from massive data via masked language modeling pretraining thanks to their intrinsic connections. We then reprogram pretrained masked language models into diffusion language models via diffusive adaptation, wherein task-specific finetuning and instruction finetuning are explored to unlock their versatility in solving general language tasks. Experiments show that scaling diffusion language models consistently improves performance across downstream language tasks. We further discover that instruction finetuning can elicit zero-shot and few-shot in-context learning abilities that help tackle many unseen tasks by following natural language instructions, and show promise in advanced and challenging abilities such as reasoning.

## Dependency

You can install the dependency with

```bash
pip3 install -r src/requirements.txt
pip3 install -e transformers
```

## Task Specific Tuning
### Data Preparation
You can obtain preprocessed data from the following urls.
```bash
mkdir -p data && cd data
pip3 install gdown
# IWSLT14
gdown 1I3Qo2hAY93dwHWLqKVGfLyKOfU-kaOAx
tar -xzvf iwslt14.de-en.prompted.xlm-r-dict.bin.tar.gz
# WMT14
gdown 1O3haSqpSbEipnYszwiNkSUiZ6fE5pezQ
tar -xzvf wmt14.en-de.prompted.xlm-r-dict.bin.tar.gz
# Gigaword-10K
gdown 16pjGW00Acn0LSKCkfx1z1p-YgS9DRU2M
tar -xzvf gigaword-10k.prompted.xlm-r-dict.bin.tar.gz
```

### Training and Evaluation

> The following commands assume working on a 8 A100 machine. For other machine configurations, please modify the mini batch sizes and `torch.distributed.launch` lines in the scripts accordingly. 

Task specific tune XLM-R-XXL with the following commands
<details open>
<summary>IWSLT14</summary>

```bash
# train the diffusion model
bash scripts/task_train.sh --data_path=data/iwslt14.de-en.prompted.xlm-r-dict.bin --pretrained=facebook/xlm-roberta-xxl --cache_dir=cache --output_dir=outputs/ckpts/iwslt14.xxl --train_steps=30000 --eval_steps=5000 --mini_bsz=2048 --accum_step=8 --ds_config=scripts/ds_config_zero2.json --extra="--src_lang de --tgt_lang en"

CKPT=`ls outputs/ckpts/iwslt14.xxl | grep checkpoint* | tail -1`

# train length predictor
bash scripts/task_train_length.sh  --data_path=data/iwslt14.de-en.prompted.xlm-r-dict.bin --pretrained=facebook/xlm-roberta-xxl --cache_dir=cache --output_dir=outputs/ckpts/iwslt14.xxl.length --train_steps=30000 --eval_steps=5000 --mini_bsz=4 --accum_step=8 --ds_config=scripts/ds_config_zero2.json --ckpt=outputs/ckpts/iwslt14.xxl/$CKPT --extra="--src_lang de --tgt_lang en" 

# evaluate
bash scripts/eval_task.sh --data_path=data/iwslt14.de-en.prompted.xlm-r-dict.bin --ckpt=outputs/ckpts/iwslt14.xxl --metric=bleu --output_dir=outputs/ckpts/iwslt14.xxl/eval --extra="--src_lang en --tgt_lang de --oracle_length"   

bash scripts/eval_task.sh --data_path=data/iwslt14.de-en.prompted.xlm-r-dict.bin --ckpt=outputs/ckpts/iwslt14.xxl.length --metric=bleu --output_dir=outputs/ckpts/iwslt14.xxl.length/eval --length_beam=10 --extra="--src_lang en --tgt_lang de"
```

</details>

<details>
<summary>WMT14</summary>

```bash
# train the diffusion model
bash scripts/task_train.sh --data_path=data/wmt14.en-de.prompted.xlm-r-dict.bin --pretrained=facebook/xlm-roberta-xxl --cache_dir=cache --output_dir=outputs/ckpts/wmt14.xxl --train_steps=100000 --eval_steps=10000 --mini_bsz=1024 --accum_step=16 --ds_config=scripts/ds_config_zero2.json --extra="--src_lang en --tgt_lang de --bleu_tokenize intl"

CKPT=`ls outputs/ckpts/wmt14.xxl | grep checkpoint* | tail -1`

# train length predictor
bash scripts/task_train_length.sh --data_path=data/wmt14.en-de.prompted.xlm-r-dict.bin --pretrained=facebook/xlm-roberta-xxl --cache_dir=cache --output_dir=outputs/ckpts/wmt14.xxl.length --train_steps=30000 --eval_steps=5000 --mini_bsz=4 --accum_step=8 --ds_config=scripts/ds_config_zero2.json --ckpt=outputs/ckpts/wmt14.xxl/$CKPT --extra="--src_lang en --tgt_lang de --bleu_tokenize intl" 

# evaluate
bash scripts/eval_task.sh --data_path=data/wmt14.en-de.prompted.xlm-r-dict.bin --ckpt=outputs/ckpts/wmt14.xxl --metric=bleu --output_dir=outputs/ckpts/wmt14.xxl/out --extra="--src_lang en --tgt_lang de --oracle_length"
bash scripts/eval_task.sh --data_path=data/wmt14.en-de.prompted.xlm-r-dict.bin --ckpt=outputs/ckpts/wmt14.xxl.length --metric=bleu --output_dir=outputs/ckpts/wmt14.xxl.length/out --length_bema=10 --extra="--src_lang en --tgt_lang de"
```

</details>


<details>
<summary>Gigaword-10K</summary>

```bash
# train the diffusion model
bash scripts/task_train.sh --data_path=data/gigaword-10k.prompted.xlm-r-dict.bin --pretrained=facebook/xlm-roberta-xxl --cache_dir=cache --output_dir=outputs/ckpts/gigaword-10k.xxl --train_steps=1000 --eval_steps=100 --mini_bsz=2048 --accum_step=1 --ds_config=scripts/ds_config_zero2.json --metric=rouge --extra="--src_lang src --tgt_lang tgt"

CKPT=`ls outputs/ckpts/gigaword-10k.xxl | grep checkpoint* | tail -1`

# train length predictor
bash scripts/task_train_length.sh --data_path=data/gigaword-10k.prompted.xlm-r-dict.bin --pretrained=facebook/xlm-roberta-xxl --cache_dir=cache --output_dir=outputs/ckpts/gigaword-10k.xxl.length --train_steps=30000 --eval_steps=5000 --mini_bsz=4 --accum_step=8 --ds_config=scripts/ds_config_zero2.json --ckpt=outputs/ckpts/gigword-10k.xxl/$CKPT --metric=rouge --extra="--src_lang src --tgt_lang tgt " 

# evaluate
bash scripts/eval_task.sh --data_path=data/gigaword-10k.prompted.xlm-r-dict.bin --ckpt=outputs/ckpts/gigaword-10k.xxl --metric=rouge --output_dir=outputs/ckpts/wmt14.xxl/out --extra="--src_lang src --tgt_lang tgt --oracle_length"
bash scripts/eval_task.sh --data_path=data/gigaword-10k.prompted.xlm-r-dict.bin --ckpt=outputs/ckpts/gigaword-10k.xxl.length --metric=rouge --output_dir=outputs/ckpts/wmt14.xxl.length/out --length_bema=10 --extra="--src_lang src --tgt_lang tgt"
```

</details>

## Instruction tuning
### Data Preparation
Run the following commands to download `FLAN 2022`.
```bash
bash scripts/download_flan_2022.sh
```

### Training and Evaluatoin
Instruction tune XLM-R-XXL with the following commands

```bash
# train diffusion model
bash scripts/instruction_tuning.sh --data_path=data/flan_2022 --pretrained=facebook/xlm-roberta-xxl --output_dir=outputs/ckpts/flan_2022.xxl --cache_dir=cache --train_steps=4000 --mini_bsz=2 --accum_step=128 --ds_config=scripts/ds_config_zero2.json


# train length predictor
CKPT=`ls outputs/ckpts/flan_2022.xxl | grep checkpoint* | tail -1`
bash scripts/instruction_train_length.sh --data_path=data/flan_2022 --pretrained=facebook/xlm-roberta-xxl --output_dir=outputs/ckpts/flan_2022.xxl.length --cache_dir=cache --train_steps=16000 --mini_bsz=16 --accum_step=2 --ds_config=scripts/ds_config_zero2.json --ckpt=outputs/ckpts/flan_2022.xxl/$CKPT


# evaluation
bash scripts/eval_instruct.sh --data_paths="data/instruct/mmlu/0-shot/val/full.jsonl data/instruct/mmlu/2-shot/val/full.jsonl data/instruct/bbh-nlp/0-shot/validation/full.jsonl instruct/bbh-nlp/2-shot/validation/full.jsonl" --ckpt=outputs/ckpts/flan_2022.xxl --output_dir=outputs/ckpts/flan_2022.xxl/out --mini_bsz=8 --max_iter=1 --extra="--oracle_length"

bash scripts/eval_instruct.sh --data_paths="data/instruct/iwslt14/0-shot/test/deen.jsonl data/instruct/iwslt14/2-shot/test/deen.jsonl" --ckpt=outputs/ckpts/flan_2022.xxl --output_dir=outputs/ckpts/flan_2022.xxl/out --mini_bsz=8 --max_iter=50 --extra="--oracle_length"

bash scripts/eval_instruct.sh --data_paths="data/instruct/iwslt14/0-shot/test/deen.jsonl data/instruct/iwslt14/2-shot/test/deen.jsonl" --ckpt=outputs/ckpts/flan_2022.xxl.length --output_dir=outputs/ckpts/flan_2022.xxl.length/out --mini_bsz=8 --max_iter=50 --length_beam=10

bash scripts/eval_instruct.sh --data_paths="data/instruct/tydiqa/0-shot/validation/flan2022.jsonl data/instruct/tydiqa/1-shot/validation/flan2022.jsonl" --ckpt=outputs/ckpts/flan_2022.xxl --output_dir=outputs/ckpts/flan_2022.xxl/out --mini_bsz=8 --max_iter=10 --extra="--oracle_length"

bash scripts/eval_instruct.sh --data_paths="data/instruct/tydiqa/0-shot/validation/flan2022.jsonl data/instruct/tydiqa/1-shot/validation/flan2022.jsonl" --ckpt=outputs/ckpts/flan_2022.xxl.length --output_dir=outputs/ckpts/flan_2022.xxl.length/out --mini_bsz=8 --max_iter=10 --length_beam=3

bash scripts/eval_instruct.sh --data_paths="data/instruct/mgsm/0-shot/en.jsonl data/instruct/mgsm/3-shot/en.jsonl" --ckpt=outputs/ckpts/flan_2022.xxl --output_dir=outputs/ckpts/flan_2022.xxl/out --mini_bsz=8 --max_iter=50 --extra="--oracle_length"

```

The evaluation scripts will produce inference results under the `--output_dir`. 
> examplary generation results can be found [here](https://gofile.io/d/bNZ8MM)

## Checkpoints

Model checkpoints are available at

|[Flan XLM-R-BASE](https://gofile.io/d/DLAkl3)|[Flan XLM-R-LARGE](https://gofile.io/d/RvqQyS)|[Flan XLM-R-XL](https://gofile.io/d/4F7vzk)|[Flan XLM-R-XXL](https://gofile.io/d/d5vka1)|

## Citation

```
@article{ye2023diffusionllm,
    title={Diffusion Language Models Can Perform Many Tasks with Scaling and Instruction-Finetuning},
    author={Ye, Jiasheng and Zheng, Zaixiang and Bao, Yu and Qian, Lihua and Gu, Quanquan},
    journal={arXiv preprint arXiv:2308.12219},
    year={2023}
}
```
