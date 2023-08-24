import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from transformers import PreTrainedModel
from fairseq.data import Dictionary

import dataclasses
from dataclasses import dataclass, field

from collections import namedtuple

import math

from typing import List

from .llama import LlamaNonCausalModel

import warnings

decoder_out_t = namedtuple(
    "decoder_out_t",
    ["output_tokens", "output_scores", "output_masks", "non_fixed_sym_masks", "attn", "step", "max_step", "history"],
)

@dataclass
class DiscreteDiffusionModelArguments:
    num_diffusion_timesteps: int = field(
        default=50,
        metadata={"help": "number of total timesteps for this diffusion model"}
    )
    diffusion_type: str = field(
        default="absorbing",
        metadata={"help": "diffusion type"}
    )
    pretrained: str = field(
        default=None
    )
    cache_dir: str = field(
        default="/mnt/bn/research/cache"
    )
    config: str = field(
        default=None
    )
    prefix_lm: bool = field(
        default=False,
        metadata={"help": "deprecated"}
    )
    attention_strategy: str = field(
        default="full"
    )
    vocab_pad_to_multiple: int = field(
        default=1 # 64
    )
    lora: bool = field(
        default=False,
        metadata={"help": "whether to traing with lora"}
    )
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["query", "value"]
    )
    lora_alpha: float = field(
        default=16
    )
    lora_rank: int = field(
        default=16
    )
    lora_bias: str = field(
        default="none"
    )
    lora_dropout: float = field(
        default=0
    )
       
    def __post_init__(self):
        if self.prefix_lm:
            Warning("option prefix_lm is deprecated, use attention_strategy=\"prefix_lm\" instead.")
            self.attention_strategy = "prefix_lm"


class DiscreteDiffusionBase(nn.Module):    
    def __init__(self, args, tokenizer) -> None:
        super().__init__()
        self.args = args
        
        self.tokenizer = tokenizer 
        self.mask_id = tokenizer.mask_token_id
        assert self.mask_id is not None, "mask token not found in tokenizer"
        self.bos_id = tokenizer.bos_token_id
        assert self.bos_id is not None, "bos token not found in tokenizer"
        self.eos_id = tokenizer.eos_token_id
        assert self.eos_id is not None, "eos token not found in tokenizer"
        self.pad_id = tokenizer.pad_token_id
        assert self.pad_id is not None, "pad token not found in tokenizer"
        self.line_splitter_id = tokenizer.encode('\n', add_special_tokens=True)[-1]
    
    def gradient_checkpointing_enable(self):
        assert hasattr(self, "model"), "self.model is not set"
        self.model.gradient_checkpointing_enable()
    
    def add_fake_layer(self):
        assert hasattr(self, "config"), "could not infer embedding dimension because self.config is not found."
        self.fake_layer = nn.Parameter(torch.zeros((self.config.hidden_size, )))
        
    def q_sample_coupled(self, x_0, t1, t2, maskable_mask):
        assert self.args.diffusion_type == "absorbing", "we only support absorbing diffusion temporarily"
        # partial mask: True for the part should not be mask
        t1_eq_t2_mask = (t1 == t2)
        t1, t2 = torch.maximum(t1, t2).float(), torch.minimum(t1, t2).float()
        
        # sample t1
        u = torch.rand_like(x_0, dtype=torch.float)
        t1_mask = (u < (t1 / self.args.num_diffusion_timesteps)[:, None]) & maskable_mask
        x_t1 = x_0.masked_fill(t1_mask, self.mask_id)
        
        # sample t2
        u = torch.rand_like(x_0, dtype=torch.float)
        t2_mask = t1_mask & (u > ((t1 - t2) / t1)[:, None])
        u = torch.rand_like(x_0[t1_eq_t2_mask], dtype=torch.float) 
        t2_mask[t1_eq_t2_mask] = (u < (t1[t1_eq_t2_mask] / self.args.num_diffusion_timesteps)[:, None]) & (maskable_mask[t1_eq_t2_mask])
        x_t2 = x_0.masked_fill(t2_mask, self.mask_id)
        
        return {
            "x_t": torch.cat([x_t1, x_t2], dim=0),
            "t": torch.cat([t1, t2]),
            "mask_mask": torch.cat([t1_mask, t2_mask], dim=0)
        }
    
    # FIXME: to design a more flexible interface for canvas initialization! 
    # The flexible version should only require source tokens and prompt formats!
    def initialize_decode_samples(self, tokens, partial_masks, prefix_masks, oracle_length=False, length_beam=1, mbr=1):
        # if tokens is None, set the length of prediction as maximum length
        # if tokens is not None, set the length of prediction as oracle length temporarily. 
        # TODO: Handle length predition
        if tokens is None:
            raise NotImplementedError
        else:
            if not oracle_length:
                inputs_tokens = tokens.masked_fill(~prefix_masks, self.pad_id)
                src_length = inputs_tokens.ne(self.pad_id).sum(dim=-1)
                inputs_tokens = inputs_tokens[:, :src_length.max()]
                length_logits = self.forward_length(inputs_tokens)
                length = (
                    torch.min(
                        length_logits.topk(length_beam, dim=-1).indices + 1, # at least one token
                        self.config.max_position_embeddings - 2 - src_length[:, None] - 1    # max_length - src_length - eos
                    )
                )
                output_tokens = []
                for i, token in enumerate(inputs_tokens):
                    for b in range(length_beam):
                        for m in range(mbr):
                            output_tokens.append(
                                torch.cat([
                                    token[:src_length[i]], 
                                    torch.tensor([self.mask_id] * length[i][b] + [self.eos_id]).to(token)
                                ])
                            )
                output_tokens = pad_sequence(output_tokens, batch_first=True, padding_value=self.pad_id)
                output_mask = output_tokens.eq(self.mask_id)
                # partial_masks = output_tokens.ne(self.mask_id) & output_tokens.ne(self.eos_id) & output_tokens.ne(self.pad_id)
            else:
                output_tokens = torch.stack([token for token in tokens for m in range(mbr)])
                partial_masks = torch.stack([mask for mask in partial_masks for m in range(mbr)])
                prefix_masks = torch.stack([mask for mask in prefix_masks for m in range(mbr)])
                output_mask = (
                    output_tokens.ne(self.pad_id) &
                    output_tokens.ne(self.bos_id) &
                    output_tokens.ne(self.eos_id) &
                    ~prefix_masks
                )
                output_tokens = output_tokens.masked_fill(output_mask, self.mask_id)
            output_scores = torch.zeros_like(output_tokens, dtype=torch.float)
            
            return partial_masks, decoder_out_t(
                output_tokens=output_tokens,
                output_scores=output_scores,
                output_masks=output_mask,
                non_fixed_sym_masks=output_mask.clone(),
                attn=None,
                step=0,
                max_step=math.inf,
                history=None
            )
            

class DiscreteDiffusionLlamaModel(DiscreteDiffusionBase):
    def __init__(self, args, tokenizer, model):
        super().__init__(args, tokenizer)
        # need some model specific trick
        if model.config.tie_word_embeddings:
            model.lm_head.weight = model.model.embed_tokens.weight
        self.model = model
        self.config = model.config
        if args.lora:
            self.add_fake_layer()
        # self.fake_layer = nn.Parameter(torch.zeros((self.config.hidden_size, ))) # to support lora with gradient checkpointing
    
    def forward(self, prev_output_tokens, partial_mask, attention_mask=None, loss_mask=None, cache=None):
        input_ids = prev_output_tokens
        if attention_mask is None:
            attention_mask = prev_output_tokens.ne(self.pad_id).int()
        embeddings = self.model.model.embed_tokens(input_ids)
        if hasattr(self, "fake_layer") and self.training:
            self.fake_layer.requires_grad = True
            embeddings = embeddings + self.fake_layer * 0   # trick to support lora + gradient checkpointing
        
        if self.args.attention_strategy == "full" or self.args.attention_strategy == "causal":
            if self.args.attention_strategy == "causal":
                assert (not isinstance(self.model.model, LlamaNonCausalModel))
            outputs = self.model.model(inputs_embeds=embeddings, attention_mask=attention_mask)[0]
        else:
            # TODO: support cache
            ext_partial_mask = partial_mask.float()
            ext_partial_mask = torch.bmm(ext_partial_mask[:, :, None], ext_partial_mask[:, None, :]).int()  # B, T, T
            # attention_mask = attention_mask.float()
            ext_mask = attention_mask[:, None, :].repeat(1, attention_mask.size(-1), 1)
            # ext_mask = torch.bmm(attention_mask[:, :, None], attention_mask[:, None, :]).int()    # B, T, T
            
            try:
                assert (ext_mask.sum(dim=-1) != 0).all()
            except:
                import ipdb;ipdb.set_trace()
            
            if self.args.attention_strategy == "blockwise":
                splitter_mask = input_ids.eq(self.line_splitter_id)
                shifted_splitter_mask = splitter_mask.clone()
                shifted_splitter_mask[:, 1:] = splitter_mask[:, :-1]
                shifted_splitter_mask[:, 0] = False
                block_id = shifted_splitter_mask.cumsum(dim=-1)
                ext_mask = ext_mask * ((block_id[:, None, :] <= block_id[:, :, None]).int())
            # import ipdb;ipdb.set_trace()
            ext_mask[partial_mask] = ext_partial_mask[partial_mask]
                
            outputs = self.model.model(inputs_embeds=embeddings, attention_mask=ext_mask)[0]
                
        
        assert (~torch.isnan(outputs)).all()
        
        outputs = outputs[loss_mask] if loss_mask is not None else outputs
        return self.model.lm_head(outputs)
    
class DiscreteDiffusionXLMRModel(DiscreteDiffusionBase):
    _is_tokenizer_index_correct = True
    
    def __init__(self, args, tokenizer, model) -> None:
        super().__init__(args, tokenizer)
        if model.config.tie_word_embeddings:
            model.lm_head.decoder.weight  = model.roberta.embeddings.word_embeddings.weight
        self.model = model
        self.config = model.config
        if args.lora:
            self.add_fake_layer() 
        # length predictor
        self.length_trm = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.config.hidden_size, 
                nhead=self.config.num_attention_heads,
                dim_feedforward=self.config.intermediate_size,
                batch_first=True
            ),
            num_layers=1,
        )
        self.length_predictor = nn.Sequential(
            nn.Linear(self.config.hidden_size , self.config.intermediate_size),
            nn.Tanh(),
            nn.Linear(self.config.intermediate_size, self.config.max_position_embeddings)
        )
    
    def remove_redundant_embeddings(self, dictionary):
        assert self._is_tokenizer_index_correct
        self._is_tokenizer_index_correct = False
        
        tokenizer = self.tokenizer
        vocab = tokenizer.get_vocab()
        vocab[dictionary.bos_word] = tokenizer.bos_token_id
        vocab[dictionary.pad_word] = tokenizer.pad_token_id
        vocab[dictionary.eos_word] = tokenizer.eos_token_id
        vocab[dictionary.unk_word] = tokenizer.unk_token_id
        subdict2modeldict = {}
        for i in range(len(dictionary)):
            token = dictionary[i]
            subdict2modeldict[i] = (
                vocab[token] 
                if token in vocab
                else tokenizer.unk_token_id
            )
        ori_embedding_weight = self.model.roberta.embeddings.word_embeddings.weight
        ori_out_linaer_weight = self.model.lm_head.decoder.weight
        ori_out_linear_bias = self.model.lm_head.decoder.bias
        new_embedding = torch.stack(
            [
                ori_embedding_weight[subdict2modeldict[i]] 
                for i in range(len(subdict2modeldict))
            ], dim=0
        )
        new_out_linear_weight = torch.stack(
            [
                ori_out_linaer_weight[subdict2modeldict[i]] 
                for i in range(len(subdict2modeldict))
            ], dim=0
        )
        new_out_linear_bias = torch.stack(
            [
                ori_out_linear_bias[subdict2modeldict[i]] 
                for i in range(len(subdict2modeldict))
            ], dim=0
        )
        self.model.roberta.embeddings.word_embeddings = nn.Embedding.from_pretrained(
            new_embedding, freeze=False, padding_idx=dictionary.pad()
        )
        new_out_linear = nn.Linear(new_out_linear_weight.size(0), new_out_linear_weight.size(1))
        new_out_linear.weight.data = new_out_linear_weight
        new_out_linear.bias.data = new_out_linear_bias
        self.model.lm_head.decoder = new_out_linear
        
        self.mask_id = dictionary.mask_index
        self.bos_id = dictionary.bos()
        self.eos_id = dictionary.eos()
        self.pad_id = dictionary.pad()
        self.line_splitter_id = dictionary.index('\n')
    
    def forward_lm_head(self, features):
        features = self.model.lm_head.dense(features)
        features = self.model.lm_head.layer_norm(gelu(features))
        return self.model.lm_head.decoder(features)
   
    def forward_length(self, input_ids):
        attention_mask = input_ids.ne(self.pad_id).int()
        with torch.no_grad():
            _feature = self.model.roberta(input_ids, attention_mask=attention_mask)[0]
        feature = self.length_trm(_feature, src_key_padding_mask=(1-attention_mask).bool())
        if not (~feature.isnan()).all():
            import ipdb; ipdb.set_trace()
        length = attention_mask.sum(dim=-1)
        pooled_feature = feature.masked_fill((attention_mask==0)[:, :, None], 0).float().sum(1) / length[:, None]
        # assert len(feature.size()) == 2
        length_logits = self.length_predictor(pooled_feature.to(feature))
        return length_logits
    
    def forward(self, prev_output_tokens, partial_mask, attention_mask=None, loss_mask=None, cache=None):
        input_ids = prev_output_tokens
        if attention_mask is None:
            attention_mask = prev_output_tokens.ne(self.pad_id).int()        
        
        embeddings = self.model.roberta.embeddings.word_embeddings(input_ids)
        
        # a trick to avoid the confliction between peft's LoRA implemention and gradient checkpointing 
        if hasattr(self, "fake_layer") and self.training:
            self.fake_layer.requires_grad = True
            embeddings = embeddings + self.fake_layer * 0   # trick to support lora + gradient checkpointing
            # self.model.roberta.embeddings.word_embeddings.weight.requires_grad = True
        
        
        if self.args.prefix_lm:
            # TODO: support cache
            ext_partial_mask = partial_mask.float()
            ext_partial_mask = torch.bmm(ext_partial_mask[:, :, None], ext_partial_mask[:, None, :]).int()  # B, T, T
            # attention_mask = attention_mask.float()
            ext_mask = attention_mask[:, None, :].repeat(1, attention_mask.size(-1), 1)
            # ext_mask = torch.bmm(attention_mask[:, :, None], attention_mask[:, None, :]).int()    # B, T, T
            
            ext_mask[partial_mask] = ext_partial_mask[partial_mask]
            try:
                assert (ext_mask.sum(dim=-1) != 0).all()
            except:
                import ipdb;ipdb.set_trace()
            # outputs = self.model.roberta(input_ids, attention_mask=ext_mask)[0]
            outputs = self.model.roberta(inputs_embeds=embeddings, attention_mask=ext_mask)[0]
        else:
            outputs = self.model.roberta(inputs_embeds=embeddings, attention_mask=attention_mask)[0]
            # outputs = self.model.roberta(input_ids, attention_mask=attention_mask)[0]
        
        if not (~torch.isnan(outputs)).all():
            outputs.masked_fill_(outputs.isnan(), 0)
            print("nan bug!")
        
        outputs = outputs[loss_mask] if loss_mask is not None else outputs
        return self.model.lm_head(outputs)
