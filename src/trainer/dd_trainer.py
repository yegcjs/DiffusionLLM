import transformers
from transformers import Trainer
from transformers.utils import logging, is_torch_tpu_available
from transformers.deepspeed import deepspeed_init, is_deepspeed_zero3_enabled
from transformers.trainer_utils import EvalPrediction, EvalLoopOutput, seed_worker, has_length
from transformers.trainer_pt_utils import find_batch_size
from transformers.trainer_callback import TrainerCallback
from transformers.modeling_utils import PreTrainedModel
from transformers.data.data_collator import DataCollator
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


from typing import List, Dict, Any, Union, Optional, Tuple, Callable

from data.dd_data import TokenSizeDistributedLengthGroupSampler
from dd_generator import DiscreteDiffusionGenerator

from dataclasses import dataclass, field
from typing import Optional

from utils import mean_ds

from tqdm import tqdm

import math

import os

from transformers import TrainingArguments

from dataclasses import dataclass, field

from typing import List

@dataclass
class DiscreteDiffusionArguments(TrainingArguments):
    batch_by_tokens: bool = field(
        default=False
    )
    eval_metric: str = field(
        default="none"
    )
    weighting: str = field(
        default="linear",
        metadata={"help": "weighting for training losses"}
    )
    mask_on_source: bool = field(
        default=False,
        metadata={"help": "whether masking is performed on source side"}
    )
    mask_on_paddings: bool = field(
        default=False,
        metadata={"help":"whether apply masking on paddings"}
    )

@dataclass
class DiscreteDiffusionTrainingArguments(DiscreteDiffusionArguments):
    finetune_from_model: str = field(
        default=None,
        metadata={"help": "results from previous stage, used for multiple stage training"}
    )
    mask_ratio_sampler: str = field(
        default="diffusion",
        metadata={"help": "diffusion|fixed[mask-ratio]. to decide whether fixed mask ratio mlm or diffusion trianing"}
    )
    train_length: bool = field(
        default=False
    )



class DiscreteDiffusionTrainer(Trainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: DiscreteDiffusionTrainingArguments = None,
        generator: DiscreteDiffusionGenerator = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ):
        super().__init__(
            model, args, data_collator, train_dataset, eval_dataset, tokenizer, model_init, compute_metrics, 
            callbacks, optimizers, preprocess_logits_for_metrics
        )
        self.generator = generator 
        self.eval_compute_loss = True
        # self.dictionary = generator.dictionary 
        
    def get_token_batched_dataloader(self, dataset, train=False):
        lengths = [dataset.size(i) for i in tqdm(range(len(dataset)))]
        batch_sampler = TokenSizeDistributedLengthGroupSampler(
            self.args.train_batch_size if train else self.args.eval_batch_size, # max_tokens
            self.args.max_length,
            dataset=dataset,
            num_replicas=self.args.world_size,
            rank=self.args.process_index,
            model_input_name=None,
            lengths=lengths,
            infinite=train
        )
        dataloader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            worker_init_fn=seed_worker
        )
        return dataloader

    def get_train_dataloader(self):
        # self.train_dataset.set_max_length(self.args.max_length)
        if self.args.batch_by_tokens:
            return self.get_token_batched_dataloader(self.train_dataset, train=True)
        else:
            return super().get_train_dataloader()

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        # if eval_dataset is not None:
        #     eval_dataset.set_max_length(self.args.max_length)
        # else:
        #     self.eval_dataset.set_max_length(self.args.max_length)
        if self.args.batch_by_tokens:
            eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
            return self.get_token_batched_dataloader(eval_dataset)
        else:
            return super().get_eval_dataloader(eval_dataset)

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        # test_dataset.set_max_length(self.args.max_length)
        if self.args.batch_by_tokens:
            return self.get_token_batched_dataloader(test_dataset)
        else:
            return super().get_test_dataloader(test_dataset)
    
    def compute_loss(self, model, inputs, return_outputs=False):
        raw_model = model.module if hasattr(model, "module") else model
        target = inputs["net_input"]["src_tokens"]
        partial_masks = (
            inputs["net_input"]["partial_masks"] 
            if not self.args.mask_on_source
            else torch.zeros_like(target).bool()
        )
        
        # couple
        if self.args.mask_ratio_sampler == "diffusion":
            t1, t2 = torch.randint(
                1, raw_model.args.num_diffusion_timesteps + 1, (2 * target.size(0), ), device=target.device
            ).chunk(2)
        elif self.args.mask_ratio_sampler.startswith("fixed"):
            ratio = float(self.args.mask_ratio_sampler[5:])
            t1, t2 = torch.tensor(
                [math.ceil(ratio * raw_model.args.num_diffusion_timesteps)] * (2 * target.size(0)),
                device=target.device, dtype=torch.long
            ).chunk(2)

        maskable_mask = (~partial_masks)
        if not self.args.mask_on_paddings: 
            maskable_mask = maskable_mask & target.ne(self.generator.pad_id)
        x_t, t, loss_mask = list(
            raw_model.q_sample_coupled(
                target, t1, t2,
                maskable_mask=maskable_mask
            ).values()
        )
        target = target.repeat(2, 1)
        partial_masks = partial_masks.repeat(2, 1)
        
        attention_mask = torch.ones_like(x_t) if self.args.mask_on_paddings else None
        logits = model(x_t, partial_masks, attention_mask=attention_mask, loss_mask=loss_mask)
        
        num_timesteps = raw_model.args.num_diffusion_timesteps
        weight = {
            "linear": (num_timesteps - (t - 1)),    # num_timesteps * (1 - (t-1)/num_timesteps)
            "constant": num_timesteps
        }[self.args.weighting][:, None].float()
        weight = weight.expand(loss_mask.size())[loss_mask]
        # cnt_weight = loss_mask.sum(-1)[:, None].expand(loss_mask.size())[loss_mask]
        # cnt_weight = x_t.size(-1)
        cnt_weight = maskable_mask.repeat(2, 1).sum(dim=-1)[:, None].expand(loss_mask.size())[loss_mask]
        ce = F.cross_entropy(logits, target[loss_mask], reduction="none").float()   # num_masked samples
        ce = (ce * weight / cnt_weight).sum() / x_t.size(0) 
        # /  mean_ds(ce.mean(-1) * weight * num_timesteps)
        ls = self.args.label_smoothing_factor
        if ls > 0:
            logit_loss = -F.log_softmax(logits, dim=-1).mean(dim=-1).float()
            logit_loss = num_timesteps * (logit_loss / cnt_weight).sum() / x_t.size(0)
            
            diffusion_loss = ((1 - ls) * ce + ls * logit_loss)
        else:
            diffusion_loss = ce
        return (diffusion_loss, logits) if return_outputs else diffusion_loss
    
    def set_eval_compute_loss(self, value):
        self.eval_compute_loss = value
    
    def begin_write_prediction(self, prediction_write_to):
        if prediction_write_to is None:
            return
        assert not hasattr(self, "write_to"), "writo file already exists"
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            file_name = f"{prediction_write_to}.{rank}"
        else:
            file_name = prediction_write_to
        self.prediction_write_to = prediction_write_to
        self.write_to = open(file_name, "w")
    
    def end_write_prediction(self):
        if not hasattr(self, "write_to"):
            return 
        self.write_to.close()
        delattr(self, "write_to")
        if not torch.distributed.is_initialized():
            return 
        torch.distributed.barrier()
        if torch.distributed.get_rank() == 0:   # aggregate all 
            lines = []
            for i in range(torch.distributed.get_world_size()):
                with open(f"{self.prediction_write_to}.{i}", "r") as f:
                    lines = lines + [line.strip() for line in f]
                os.remove(f"{self.prediction_write_to}.{i}")
            
            with open(self.prediction_write_to, "w") as f:
                f.write('\n'.join(lines))
        
        delattr(self, "prediction_write_to")
 
         
    @torch.no_grad()
    def prediction_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], prediction_loss_only: bool, ignore_keys: Optional[List[str]] = None) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        inputs = self._prepare_inputs(inputs)
        if not self.eval_compute_loss:
            loss = torch.tensor([0.]).to(inputs['target'].device)
        else:
            loss = self.compute_loss(model, inputs)

        hyps, history = self.generator.generate(model, inputs)        
        refs = inputs["target"]
        
        if hasattr(self, "write_to") or not prediction_loss_only:
            hyps_seqs = self.generator.decode(hyps)
            refs_seqs = self.generator.decode(refs)

            if hasattr(self, "write_to"):
                inputs_seqs = self.generator.decode(inputs["net_input"]["src_tokens"])
                if history is not None:
                    # import ipdb; ipdb.set_trace()
                    history_seqs = [
                        [self.generator.decode(step[None, :], preserve_special=True)[0] for step in his] 
                        for his in history
                    ]
                    for index, src, hyp, ref, his in zip(inputs["id"], inputs_seqs, hyps_seqs, refs_seqs, history_seqs):
                        index = index.item()
                        self.write_to.write(f"SRC-{index}\t{src}\nHYP-{index}\t{hyp}\nREF-{index}\t{ref}\n") 
                        for i, his_seq in enumerate(his):
                            self.write_to.write(f"STEP{i}-{index}\t{his_seq}\n")
                else:
                    for index, src, hyp, ref in zip(inputs["id"], inputs_seqs, hyps_seqs, refs_seqs):
                        index = index.item()
                        self.write_to.write(f"SRC-{index}\t{src}\nHYP-{index}\t{hyp}\nREF-{index}\t{ref}\n")
                    
            if (not prediction_loss_only) and (self.compute_metrics is not None):
                if self.args.eval_metric == "bleu":
                    bleu = self.generator.compute_bleu(hyps_seqs, refs_seqs)
                    
                    sys_stat = torch.tensor([*bleu.counts, bleu.sys_len]).to(loss)
                    ref_stat = torch.tensor([*bleu.totals, bleu.ref_len]).to(loss)
                elif self.args.eval_metric == "rouge":
                    rouge = self.generator.compute_rouge(hyps_seqs, refs_seqs)
                    sys_stat = torch.tensor([rouge]).to(loss)
                    ref_stat = torch.tensor([len(hyps_seqs)]).to(loss)
                    
                return (loss, sys_stat, ref_stat)
        return (loss, None, None)

class DiscreteDiffusionLengthTrainer(DiscreteDiffusionTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # global global_step_step 
        # global_step_step += 1
        # if global_step_step % 2 == 0:
        #     return super().compute_loss(model, inputs)  
        raw_model = model.module if hasattr(model, "module") else model
        partial_masks = inputs["net_input"]["partial_masks"] 
        partial_masks[:, 0] = True
        input_tokens = inputs["net_input"]["src_tokens"].masked_fill(~partial_masks, raw_model.pad_id)
        max_index = input_tokens.ne(raw_model.pad_id).sum(dim=-1).max()
        input_tokens = input_tokens[:, :max_index]
        
        target = (
            (~partial_masks) &
            inputs["net_input"]["src_tokens"].ne(raw_model.pad_id)
        ).sum(-1).clamp(2) - 2 # -eos, 1->0
        logits = raw_model.forward_length(input_tokens)
        loss = F.cross_entropy(logits, target)
        return (loss, logits) if return_outputs else loss 