import torch
import torch.distributed as dist

from peft import PeftModel, LoraConfig, TaskType, get_peft_model

from model.dd_model import DiscreteDiffusionModelArguments, DiscreteDiffusionLlamaModel, DiscreteDiffusionXLMRModel
from model.llama import LlamaMaskedLM 

from transformers import AutoTokenizer, LlamaForCausalLM, AutoModelForMaskedLM
from transformers.utils import logging
from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.modeling_utils import _load_state_dict_into_model

import os 

from typing import Dict, List, Union, get_args

import math

logger = logging.get_logger(__name__)


def is_master():
    return (not dist.is_initialized()) or (dist.get_rank()==0)

def serialized_func(enable=False):
    def _serialized_func(func):
        def wrapped_func(*args, **kwargs):
            local_rank = int(os.environ["LOCAL_RANK"])  if dist.is_initialized() else 0
            local_world_size = int(os.environ["LOCAL_WORLD_SIZE"]) if dist.is_initialized() else 1
            ret = None
            for i in range(local_world_size):
                if dist.is_initialized() and enable:
                    dist.barrier()
                if i == local_rank:
                    ret = func(*args, **kwargs)
                    logger.info(f"Local rank {local_rank} is done")
            return ret
        return wrapped_func
    return _serialized_func

def mean_ds(x, dim=None):
    return (
        x.float().mean().type_as(x)
        if dim is None
        else x.float().mean(dim).type_as(x)
    )

def argument_filter(arguments):
    if isinstance(arguments, List):
        arg_list = []
        for item in arguments:
            if isinstance(item, get_args(Union[int, float, str])):
                arg_list.append(item)
            elif isinstance(item, get_args(List)):
                arg_list.append(argument_filter(item))
        return arg_list
    elif isinstance(arguments, Dict):
        arg_dict = {}
        for key, value in arguments.items():
            assert type(key) == str
            if isinstance(value, get_args(Union[int, float, str])):
                arg_dict[key] = value
            elif isinstance(value, get_args(Union[Dict, List])):
                arg_dict[key] = argument_filter(value)
        return arg_dict

@serialized_func(not is_deepspeed_zero3_enabled())
def load_ckpt(model, ckpt_path, do_train=False):
    files = os.listdir(ckpt_path)
    # lora
    if 'adapter_model.bin' in files:
        model = PeftModel.from_pretrained(model, ckpt_path, is_trainable=do_train)
    # pytorch_model.bin 
    else:
        if 'pytorch_model.bin' in files:
            state_dict = torch.load(f"{ckpt_path}/pytorch_model.bin", map_location="cpu")
        # deepspeed
        else:
            from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
            state_dict = get_fp32_state_dict_from_zero_checkpoint(ckpt_path)
        # err_msg = _load_state_dict_into_model(model, state_dict, "")
        if isinstance(model, DiscreteDiffusionXLMRModel) and state_dict['model.lm_head.decoder.weight'].shape == torch.Size([0]):
            state_dict['model.lm_head.decoder.weight'] = state_dict['model.roberta.embeddings.word_embeddings.weight']
        
        incompatible = model.load_state_dict(state_dict, strict=False)
        logger.info(incompatible)
    return model# , tokenizer

def _get_missing_special_tokens(tokenizer, tokenizer_pad_to_multiple):
    # add special tokens
    special_token_dict, padding_tokens = dict(), []
    if tokenizer.pad_token is None:
        special_token_dict["pad_token"] = "<pad>"
    if tokenizer.bos_token is None:
        special_token_dict["bos_token"] = "<s>"
    if tokenizer.eos_token is None:
        special_token_dict["eos_token"] = "<s>"
    if tokenizer.unk_token is None:
        special_token_dict["unk_token"] = "<unk>"
    if tokenizer.mask_token is None:
        special_token_dict["mask_token"] = "<mask>"
    current_vocab_size = len(tokenizer.get_vocab()) + len(special_token_dict)
    target_vocab_size = math.ceil(current_vocab_size / tokenizer_pad_to_multiple) * tokenizer_pad_to_multiple  
    for i in range(target_vocab_size - current_vocab_size):
        assert (f"<unused{i}>" not in tokenizer.get_vocab()), f"unused_{i} already exists in the vocabulary"
        padding_tokens.append(f"<unused{i}>")
    return special_token_dict, padding_tokens
    

# @serialized_func 
def load_model_tokenizer(model_args, do_train):
    pretrained, config = model_args.pretrained, model_args.config
    model_type = pretrained if pretrained is not None else config
    model_type = "llama" if "llama" in model_type.lower() else "xlm-roberta"
    if pretrained is not None:
        model = {
            "llama": LlamaForCausalLM if model_args.attention_strategy == "causal" else LlamaMaskedLM,
            "xlm-roberta": AutoModelForMaskedLM
        }[model_type].from_pretrained(pretrained, cache_dir=model_args.cache_dir)
        tokenizer = AutoTokenizer.from_pretrained(pretrained, padding_side="right", use_fast=False, cache_dir=model_args.cache_dir)
    else:
        model = {
            "llama": LlamaForCausalLM if model_args.attention_strategy == "causal" else LlamaMaskedLM,
            "xlm-roberta": AutoModelForMaskedLM
        }[model_type].from_config(config, cache_dir=model_args.cache_dir)
        tokenizer = AutoTokenizer.from_pretrained(config, padding_side="right", use_fast=False, cache_dir=model_args.cache_dir)
    
    if model_type == "llama":   # previous xlm-robert models does not prepare for this
        extra_special_token_dict, padding_tokens = _get_missing_special_tokens(tokenizer, model_args.vocab_pad_to_multiple)
        tokenizer.add_special_tokens(extra_special_token_dict)
        tokenizer.add_tokens(padding_tokens)
        model.resize_token_embeddings(len(tokenizer))
    
        num_new_tokens = len(extra_special_token_dict)
        if num_new_tokens > 0:
            input_embeddings = model.get_input_embeddings().weight.data
            output_embeddings = model.get_output_embeddings().weight.data

            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

            input_embeddings[-num_new_tokens:] = input_embeddings_avg
            output_embeddings[-num_new_tokens:] = output_embeddings_avg
    
    dd_model = {
        "llama": DiscreteDiffusionLlamaModel,
        "xlm-roberta": DiscreteDiffusionXLMRModel
    }[model_type](model_args, tokenizer, model)
    
    if model_args.lora:
        lora_config = LoraConfig(
            TaskType.TOKEN_CLS, r=model_args.lora_rank, lora_alpha=model_args.lora_alpha, 
            target_modules=model_args.lora_target_modules, bias=model_args.lora_bias,
            lora_dropout=model_args.lora_dropout,
            inference_mode=(not do_train)
        )
        dd_model = get_peft_model(dd_model, lora_config)
    
    return dd_model, tokenizer