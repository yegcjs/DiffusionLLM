import transformers
import dataclasses
from dataclasses import dataclass, field

import json

from model.dd_model import DiscreteDiffusionModelArguments, DiscreteDiffusionLlamaModel
from data.dd_data import DiscreteDiffusionDataArguments, PairDataset, DiscreteDiffusionDataCollator, FairseqLangPairDataset
from trainer.dd_trainer import DiscreteDiffusionArguments, DiscreteDiffusionTrainingArguments, DiscreteDiffusionTrainer
from dd_generator import DiscreteDiffusionGeneratorArguments, DiscreteDiffusionGenerator, MergeBLEU

from copy import deepcopy
from typing import List

from utils import load_model_tokenizer, load_ckpt

import os

@dataclass
class DiscreteDiffusionEvalArguments:
    ckpt_args_file: str = field(
        default="",
        metadata={"help": "args file to load config"}
    )
    no_compute_loss: bool = field(
        default=False,
        metadata={"help": "whether to ignore computing loss"}
    )
    prediction_write_to: str = field(
        default=None
    )


@dataclass
class DiscreteDiffusionEvalDataArguments(DiscreteDiffusionDataArguments):
    data_path: List[str] = field(
        default_factory=lambda: []
    )
    
def main():
    parser = transformers.HfArgumentParser((
        DiscreteDiffusionEvalArguments,
        DiscreteDiffusionTrainingArguments, 
        DiscreteDiffusionEvalDataArguments,
        DiscreteDiffusionGeneratorArguments
    ))
    eval_args, train_args, data_args, gen_args = parser.parse_args_into_dataclasses()

    with open(eval_args.ckpt_args_file, "r") as f:
        config = json.load(f)
    model_args = config['model']
    acceptable_model_args_keys = {item.name for item in dataclasses.fields(DiscreteDiffusionModelArguments)}
    for key in list(model_args.keys()):
        if key not in acceptable_model_args_keys:
            del model_args[key]
    model_args = DiscreteDiffusionModelArguments(**model_args)

    model, tokenizer = load_model_tokenizer(model_args, do_train=False)
    
    
    metric = {
        "none": None,
        "bleu": MergeBLEU()
    }[train_args.eval_metric]
    
    if data_args.dataset_type != "fairseq": # FIXME: abandon fairseq dataset in the future
        model = load_ckpt(model, train_args.resume_from_checkpoint) # should we use a args from eval?
    else:
        # delay loading model util dataset is loaded because we need dictionary to remove redundant embeddings
        pass
    
    if eval_args.prediction_write_to is not None:
        os.makedirs(eval_args.prediction_write_to, exist_ok=True)
        
    for data_path in data_args.data_path:
        data_item_args_dict = deepcopy(data_args.__dict__)
        data_item_args_dict["data_path"] = data_path
        data_item_args = DiscreteDiffusionDataArguments(**data_item_args_dict)
        if data_args.dataset_type == "fairseq":
            assert len(data_args.data_path) == 1
            dictionary, (_, _, testset) = FairseqLangPairDataset.load_data(data_item_args, train=False, valid=False, test=True)
            collator = DiscreteDiffusionDataCollator(bos_id=dictionary.bos(), eos_id=dictionary.eos(), pad_id=dictionary.pad())
            generator = DiscreteDiffusionGenerator(gen_args, dictionary=dictionary) 
            # remove redundant tokens in model
            model.remove_redundant_embeddings(dictionary)
            model = load_ckpt(model, train_args.resume_from_checkpoint)
        elif data_item_args.dataset_type == "pair":
            _, _, testset = PairDataset.load_data(data_item_args, tokenizer, train=False, valid=False, test=True)
            collator = DiscreteDiffusionDataCollator(bos_id=tokenizer.bos_token_id, eos_id=tokenizer.eos_token_id, pad_id=tokenizer.pad_token_id)
            generator = DiscreteDiffusionGenerator(gen_args, tokenizer=tokenizer) 
        

        trainer = DiscreteDiffusionTrainer(
            model=model, args=train_args, 
            generator=generator,
            data_collator=collator,
            compute_metrics=metric
        )
        trainer.set_eval_compute_loss(~eval_args.no_compute_loss)
        write_to_file_name = data_path.replace('/', '_') + ".txt"
        write_to = (
            f"{eval_args.prediction_write_to}/{write_to_file_name}"
            if eval_args.prediction_write_to is not None
            else None
        )
        trainer.begin_write_prediction(write_to) 
        result = trainer.evaluate(testset)
        trainer.end_write_prediction()
    
        print(result)
    
if __name__ == '__main__':
    main()