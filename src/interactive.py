import json 

import deepspeed

import torch

from model.dd_model import DiscreteDiffusionModelArguments, DiscreteDiffusionLlamaModel
from utils import load_model_tokenizer, load_ckpt
from dd_generator import DiscreteDiffusionGeneratorArguments, DiscreteDiffusionGenerator, MergeBLEU

from transformers.utils import logging

logger = logging.get_logger(__name__)

class InteractiveDiffusion:
    def __init__(self, model_ckpt_config_path, model_ckpt_path):
        with open(model_ckpt_config_path, "r") as f:
            model_args = DiscreteDiffusionModelArguments(**json.load(f)["model"])
        model, tokenizer = load_model_tokenizer(model_args, False)
        if model_ckpt_path is not None:
            logger.info(f"Loading checkpoint from {model_ckpt_path}")
            model = load_ckpt(model, model_ckpt_path)
        
        # self.model = deepspeed.init_inference(
        #     model, replace_with_kernel_inject=True# , dtype=torch.bfloat16
        # )
        self.model = model.cuda().to(torch.bfloat16)
        logger.info("using bfloat16, please make sure that your device supports it")
        self.tokenizer = tokenizer 
    
    
    @torch.no_grad()
    def sample(self, prompt, lengths, **kwargs):
        gen_args = DiscreteDiffusionGeneratorArguments(**kwargs, oracle_length=True)
        generator = DiscreteDiffusionGenerator(gen_args, tokenizer=self.tokenizer) 
        full_output = self.tokenizer.encode(prompt)
        src_length = len(full_output)
        lengths = list(map(int, lengths.split()))
        for length in lengths:
            masks = [self.tokenizer.mask_token_id] * length
            input_src_tokens = torch.tensor([full_output + masks], device='cuda')
            partial_masks = torch.zeros_like(input_src_tokens, device="cuda", dtype=torch.bool)
            partial_masks[:, :src_length] = True
            prefix_masks = input_src_tokens.ne(self.tokenizer.mask_token_id)
            inputs = {
                "net_input": {
                    "src_tokens": input_src_tokens,
                    "partial_masks": partial_masks,
                    "prefix_masks": prefix_masks
                }
            }
            # FIXME:
            # assert (partial_masks == prefix_masks).all()
            for i, step_out in enumerate(generator.stepwise_generate(self.model, inputs)):
                output = step_out.output_tokens[~partial_masks]
                full_output = step_out.output_tokens.tolist()[0]
                yield self.tokenizer.decode(output)
            
            yield("-------")
    
