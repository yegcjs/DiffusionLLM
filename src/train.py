import torch

import transformers 
from transformers.utils import logging
from transformers.trainer_utils import get_last_checkpoint

import os

from model.dd_model import DiscreteDiffusionModelArguments, DiscreteDiffusionLlamaModel
from model.llama import LlamaNonCausalModel
from trainer.dd_trainer import DiscreteDiffusionTrainingArguments
from dd_generator import DiscreteDiffusionGenerator, DiscreteDiffusionGeneratorArguments, MergeBLEU, MergeRouge
from trainer.dd_trainer import DiscreteDiffusionTrainer, DiscreteDiffusionLengthTrainer
from utils import load_ckpt, is_master, argument_filter, load_model_tokenizer
from data.dd_data import (
    DiscreteDiffusionDataArguments, DiscreteDiffusionDataCollator,
    FairseqLangPairDataset, FlanInstructionTuningDataset, FlanV2Dataset,
    PairDataset, MemoryMapTokensDataset
)

import json

def parse_args():
    parser = transformers.HfArgumentParser((
        DiscreteDiffusionDataArguments,   # data
        DiscreteDiffusionModelArguments,   # model
        DiscreteDiffusionTrainingArguments,
        DiscreteDiffusionGeneratorArguments,   # generation
    ))
    data_args, model_args, train_args, gen_args = parser.parse_args_into_dataclasses()
    if train_args.resume_from_checkpoint is None:
        if os.path.exists(train_args.output_dir):
            train_args.resume_from_checkpoint = get_last_checkpoint(train_args.output_dir)
        
    # dump the arguments
    if is_master():
        if not os.path.exists(train_args.output_dir):
            os.makedirs(train_args.output_dir)
        d = {
            "data": argument_filter(data_args.__dict__),
            "generator": argument_filter(gen_args.__dict__),
            "model": argument_filter(model_args.__dict__),
            "train": argument_filter(train_args.__dict__)
        }
        if train_args.deepspeed is not None:
            with open(train_args.deepspeed, "r") as f:
                ds = json.load(f)
            d["deepspeed"] = ds
        with open(f"{train_args.output_dir}/args.json", "w") as f:
            json.dump(d, f, indent=4)
    return data_args, model_args, train_args, gen_args

def main():
    data_args, model_args, train_args, gen_args = parse_args()
    # init model
    model, tokenizer = load_model_tokenizer(model_args, do_train=True)
    
    # load datasets 
    # FIXME: merge them
    if data_args.dataset_type == "fairseq":
        dictionary, (train_set, valid_set, _) = FairseqLangPairDataset.load_data(data_args)
        if train_args.batch_by_tokens:
            setattr(train_args, "max_length", data_args.max_length)
        collator = DiscreteDiffusionDataCollator(bos_id=dictionary.bos(), eos_id=dictionary.eos(), pad_id=dictionary.pad())
        generator = DiscreteDiffusionGenerator(gen_args, dictionary=dictionary) 
        # remove redundant tokens in model
        model.remove_redundant_embeddings(dictionary)
    elif data_args.dataset_type == "flan_v2":
        setattr(data_args, "cache_dir", model_args.cache_dir)
        setattr(data_args, "pretrained", model_args.pretrained)
        setattr(data_args, "per_device_batch_size", train_args.per_device_train_batch_size)
        (train_set, valid_set, _) = FlanV2Dataset.load_data(data_args, tokenizer); assert valid_set is None
        collator = DiscreteDiffusionDataCollator(bos_id=tokenizer.bos_token_id, eos_id=tokenizer.eos_token_id, pad_id=tokenizer.pad_token_id)
        generator = DiscreteDiffusionGenerator(gen_args, tokenizer=tokenizer) 
    elif data_args.dataset_type == "mmap":
        (train_set, valid_set, _) = MemoryMapTokensDataset.load_data(data_args, tokenizer); assert valid_set is None
        collator = DiscreteDiffusionDataCollator(bos_id=tokenizer.bos_token_id, eos_id=tokenizer.eos_token_id, pad_id=tokenizer.pad_token_id)
        generator = DiscreteDiffusionGenerator(gen_args, tokenizer=tokenizer)  
    
    # resume checkpoint
    if train_args.finetune_from_model is not None:
        model = load_ckpt(model, train_args.finetune_from_model)
    
    # build trainer
    metric = None
    if train_args.eval_metric == "bleu":
        metric = MergeBLEU()
    elif train_args.eval_metric ==  "rouge":
        metric = MergeRouge()
    Trainer = DiscreteDiffusionTrainer if not train_args.train_length else DiscreteDiffusionLengthTrainer
    trainer = Trainer(
        model=model, args=train_args, 
        train_dataset=train_set, eval_dataset=valid_set,
        generator=generator,
        data_collator=collator,
        compute_metrics=metric
    )
    # train!
    trainer.train(resume_from_checkpoint=train_args.resume_from_checkpoint)

if __name__ == '__main__':
    main()