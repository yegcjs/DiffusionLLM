from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.distributions as dists
from torch.nn.utils.rnn import pad_sequence

import numpy as np

import math


from fairseq.data import Dictionary

import sacrebleu

from rouge import Rouge

@dataclass
class DiscreteDiffusionGeneratorArguments:
    max_iterations: int = field(
        default=10
    )
    mbr: int = field(
        default=1
    )
    length_beam: int = field(
        default=1
    )
    oracle_length: bool = field(
        default=False
    )
    strategy: str = field(
        default="reparam-uncond-deterministic-cosine"
    )
    argmax_decoding: bool = field(
        default=False
    )
    bpe: str = field(
        default="sentencepiece"
    )
    bleu_tokenize: str = field(
        default="13a"
    )
    return_history: bool = field(
        default=False
    )
    temperature: float = field(
        default=1
    )



def topk_masking(scores, cutoff_len, stochastic=False, temp=1.0):
    """
    scores: [b, n]
    cutoff_len: [b, 1]
    stochastic: bool, whether to add noise to select top_k or not
    returns:
        mask: [b, n], with 1 if the token is in top-k lowest scores, 0 otherwise
    """
    if stochastic:
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(scores) + 1e-8) + 1e-8)
        _scores = scores + temp * gumbel_noise
    else:
        _scores = scores
    sorted_index = _scores.sort(-1)[0]
    cutoff = sorted_index.gather(dim=-1, index=cutoff_len) # + 1e-10
    # cutoff_len = k -> select k + 1 tokens
    masking = _scores < cutoff
    try:
        assert (~(cutoff_len == 0).all()) | (~masking).all()
    except:
        import ipdb;ipdb.set_trace()
    return masking


class MergeBLEU(object):
    def __call__(self, evalpreds):
        # if torch.distributed.get_rank() == 0:
        #     import ipdb; ipdb.set_trace()
        # else:
        #     import time; time.sleep(120)
        import inspect
        sys_stats, ref_stats = evalpreds[0], evalpreds[1]
        
        sys_stats = sys_stats.reshape(-1, 5).astype('long').sum(0).tolist()
        ref_stats = ref_stats.reshape(-1, 5).astype('long').sum(0).tolist()
        try:
            from sacrebleu.metrics import BLEU
            comp_bleu = BLEU.compute_bleu
        except ImportError:
            comp_bleu = sacrebleu.compute_bleu
        fn_sig = inspect.getfullargspec(comp_bleu)[0]
        if "smooth_method" in fn_sig:
            smooth = {"smooth_method": "exp"}
        else:
            smooth = {"smooth": "exp"}
        return {
            "bleu": comp_bleu(
                correct=sys_stats[:4], 
                total=ref_stats[:4],
                sys_len=sys_stats[-1],
                ref_len=ref_stats[-1],
                **smooth
            ).score
        }

class MergeRouge(object):
    def __call__(self, evalpreds):
        # if torch.distributed.get_rank() == 0:
        #     import ipdb; ipdb.set_trace()
        # else:
        #     import time; time.sleep(120)
        import inspect
        # sys
        avg_rouge, batch_size = evalpreds[0], evalpreds[1]
        
        rouge = (avg_rouge * batch_size).sum() / batch_size.sum()
        
        return {
            "rouge": rouge
        }
        

class DiscreteDiffusionGenerator:
    def __init__(self, args, dictionary=None, tokenizer=None) -> None:
        self.args = args 
        self.dictionary = dictionary
        self.tokenizer = tokenizer
        self.write_prediction = None
    
        assert (dictionary is not None) or (tokenizer is not None)
        assert (dictionary is None) ^ (tokenizer is None)
        
        self.retain_history = args.return_history
        
        if dictionary is not None:
            self.pad_id = dictionary.pad()
            self.bos_id = dictionary.bos()
            self.eos_id = dictionary.eos()
            self.mask_id = dictionary.mask_index
        else:
            self.pad_id = tokenizer.pad_token_id
            self.bos_id = tokenizer.bos_token_id
            self.eos_id = tokenizer.eos_token_id
            self.mask_id = tokenizer.mask_token_id
    
        self.rouge = Rouge(["rouge-l"])
    
    def set_write_to(self, path):
        self.write_prediction = path
    
    def _reparam_decoding(
        self, 
        output_tokens, 
        output_scores, 
        cur_tokens,
        cur_scores,
        decoding_strategy,
        xt_neq_x0, 
        non_special_sym_mask, 
        t,
        max_step,
        noise
    ):
        """
            This function is used to perform reparameterized decoding.
        """
        # output_tokens: [B, N]
        # output_scores: [B, N]
        # cur_tokens: [B, N]
        # cur_scores: [B, N]
        # xt_neq_x0: equivalent to not_b_t [B, N]
        # non_special_sym_mask: [B, N]
        # noise: either [B, N] or scalar (if using the mask noise)
        
        # decoding_strategy needs to take the form of "reparam-<conditioning>-<topk_mode>-<schedule>"
        _, condition, topk_mode, schedule = decoding_strategy.split("-")

        # first set the denoising rate according to the schedule
        if schedule == "linear":
            rate = 1 - t / max_step
        elif schedule == "cosine":
            rate = np.cos(t / max_step * np.pi * 0.5)
        else:
            raise NotImplementedError

        # compute the cutoff length for denoising top-k positions
        cutoff_len = (
            non_special_sym_mask.sum(1, keepdim=True).type_as(output_scores) * rate
            ).long()
        # set the scores of special symbols to a large value so that they will never be selected
        _scores_for_topk = cur_scores.masked_fill(~non_special_sym_mask, 1000.0)
        
        # the top-k selection can be done in two ways: stochastic by injecting Gumbel noise or deterministic
        if topk_mode.startswith("stochastic"):
            noise_scale = float(topk_mode.replace("stochastic", ""))
            lowest_k_mask = topk_masking(_scores_for_topk, cutoff_len, stochastic=True, temp=noise_scale * rate)
        elif topk_mode == "deterministic":
            lowest_k_mask = topk_masking(_scores_for_topk, cutoff_len, stochastic=False)
        else:
            raise NotImplementedError
        
        # Various choices to generate v_t := [v1_t, v2_t].
        # Note that 
        #   v1_t governs the outcomes of tokens where b_t = 1,
        #   v2_t governs the outcomes of tokens where b_t = 0.
        
        # #### the `uncond` mode ####
        # In our reparameterized decoding, 
        # both v1_t and v2_t can be fully determined by the current token scores .
        
        # #### the `cond` mode ####
        # However, we can also impose some conditional constraints on v1_t so that
        # the decoding can be performed in a more conservative manner.
        # For example, we can set v1_t = 0 only when 
        # (the newly output tokens are the same as previous denoised results, AND
        # the current token score becomes lower, AND
        # the current token score is not in the top-k share among all tokens).
        if condition == "cond":
            not_v1_t = (cur_tokens == output_tokens) & (cur_scores < output_scores) & lowest_k_mask
        elif condition == "uncond":
            not_v1_t = lowest_k_mask
        else:
            raise NotImplementedError
        
        # for b_t = 0, the token is set to noise if it is in the lowest k scores.
        not_v2_t = lowest_k_mask

        masked_to_noise = (~xt_neq_x0 & not_v1_t) | (xt_neq_x0 & not_v2_t)
        if isinstance(noise, torch.Tensor):
            output_tokens.masked_scatter_(masked_to_noise, noise[masked_to_noise])
        elif isinstance(noise, (int, float)):
            output_tokens.masked_fill_(masked_to_noise, noise)
        else:
            raise NotImplementedError("noise should be either a tensor or a scalar")
        output_scores.masked_fill_(masked_to_noise, -math.inf)

        masked_to_x0 = xt_neq_x0 & ~not_v2_t
        output_tokens.masked_scatter_(masked_to_x0, cur_tokens[masked_to_x0])
        output_scores.masked_scatter_(masked_to_x0, cur_scores[masked_to_x0])
        # b_{t} = (b_{t+1} & u_t) | v_t
        # For convenience, save the NOT of b_t for the next iteration
        # NOT_b_{t} = (NOT_b_{t+1} | not_v1_t) & not_v2_t
        new_xt_neq_x0 = (xt_neq_x0 | not_v1_t) & not_v2_t
        return new_xt_neq_x0
    
    def denoise_step(self, model, decoder_out, partial_masks):
        output_tokens = decoder_out.output_tokens
        output_scores = decoder_out.output_scores
        prev_step, cur_step = decoder_out.step, decoder_out.step + 1 
        max_step = decoder_out.max_step
        temperature = self.args.temperature
        # temperature = (
        #     -0.05 * (cur_step / (max_step - 1)) + 0.5
        #     if self.temperature_annealing
        #     else self.temperature
        # )
        
        # t = torch.LongTensor(
        #     [(max_step - prev_step) * (model.num_diffusion_timesteps // max_step)] * output_tokens.size(0)
        # ).to(output_tokens)
        logits = model(output_tokens, partial_masks)
        
        logits[..., self.mask_id] = -math.inf
        scores = torch.log_softmax(logits, dim=-1)
        
        
        if self.args.strategy == "cmlm":
            # get the mask
            # <bos>, <eos> are ignored in this case since
            # they are not equal to unk.
            output_masks = output_tokens.eq(self.mask_id)
            unmask_prob = 1 / (max_step - prev_step)
            # where to unmask
            changes = torch.rand(output_tokens.shape, device=output_tokens.device) < unmask_prob
            # don't unmask somewhere already unmasked
            changes = torch.bitwise_and(changes, output_masks)

            if self.args.argmax_decoding:
                output_scores, new_tokens = scores.max(-1)
            else:
                new_tokens = dists.Categorical(logits=scores / temperature).sample()
                output_scores = torch.gather(scores, -1, new_tokens.unsqueeze(-1)).squeeze(-1)
            output_tokens[changes] = new_tokens[changes]
        elif self.args.strategy == "ar":
            output_masks = output_tokens.eq(self.mask_id)
            unmask_indices = (output_tokens.ne(self.mask_id) & output_tokens.ne(self.eos_id) & output_tokens.ne(self.pad_id)).sum(dim=-1)
            indices = torch.arange(output_tokens.size(-1)).expand(output_tokens.shape).to(output_masks.device)
            if self.args.argmax_decoding:
                output_scores, new_tokens = scores.max(-1)
            else:
                new_tokens = dists.Categorical(logits=scores / temperature).sample()
                output_scores = torch.gather(scores, -1, new_tokens.unsqueeze(-1)).squeeze(-1)
            output_tokens[unmask_indices[:, None]==indices] = new_tokens[unmask_indices[:, None]==indices]
            # output_tokens[changes] = new_tokens[changes]
        else:
            if self.args.argmax_decoding:
                cur_scores, cur_tokens = scores.max(-1)
            else:
                cur_tokens = dists.Categorical(logits=scores / temperature).sample()
                cur_scores = torch.gather(scores, -1, cur_tokens.unsqueeze(-1)).squeeze(-1)
            cur_scores = cur_scores.to(output_scores)
            
            output_masks = self._reparam_decoding(
                output_tokens=output_tokens,
                output_scores=output_scores,
                cur_tokens=cur_tokens,
                cur_scores=cur_scores,
                decoding_strategy=self.args.strategy,
                xt_neq_x0=decoder_out.output_masks,
                non_special_sym_mask=decoder_out.non_fixed_sym_masks,
                t=cur_step,
                max_step=max_step,
                noise=self.mask_id
            )
        if self.retain_history:
            history = ([] if decoder_out.history is None else decoder_out.history) + [output_tokens.clone()]
        else:
            history = None
        # history = (
        #     decoder_out.history + [output_tokens.clone()]
        #     if self.retain_history
        #     else None 
        # )
        return decoder_out._replace(
            step=cur_step,
            output_tokens=output_tokens,
            output_scores=output_scores,
            output_masks=output_masks,
            history=history,
        )

    
    def decode(self, seqs_tensors, preserve_special=False):
        seqs_tensors[seqs_tensors < 0] = self.pad_id
        if self.dictionary is not None:
            seqs = [
                self.dictionary.string(seq, self.args.bpe).strip()
                for seq in seqs_tensors
            ] 
            if not preserve_special:
                seqs = [seq.replace(self.dictionary.pad_word, '') for seq in seqs]
        else:
            seqs = self.tokenizer.batch_decode(seqs_tensors, skip_special_tokens=(not preserve_special))
        return [seq.lower() for seq in seqs]
    
    def compute_bleu(self, hyps, refs):
        if isinstance(hyps, torch.Tensor):
            hyps = self.decode(hyps)
        if isinstance(refs, torch.Tensor):
            refs = self.decode(refs)
        return sacrebleu.corpus_bleu(hyps, [refs], tokenize=self.args.bleu_tokenize)
   
    def compute_rouge(self, hyps, refs):
        if isinstance(hyps, torch.Tensor):
            hyps = self.decode(hyps)
        if isinstance(refs, torch.Tensor):
            refs = self.decode(refs) 
        return self.rouge.get_scores(hyps, [[ref] for ref in refs])['rouge-l']['f']
    
    def stepwise_generate(self, model, inputs):
        src_tokens = inputs["net_input"]["src_tokens"]
        partial_masks = inputs["net_input"]["partial_masks"]
        # assert src_tokens.size(-1) < 514
        # assert partial_masks.size(-1) < 514
        # target = inputs["target"]
        raw_model = model.module if hasattr(model, "module") else model
        if "prefix_masks" in inputs["net_input"]:
            prefix_masks = inputs["net_input"]["prefix_masks"]
        else:
            prefix_masks = partial_masks
        # TODO: FIXME: to support general blockwise generation.
        partial_masks, prev_decoder_out = raw_model.initialize_decode_samples(
            src_tokens, partial_masks, prefix_masks, oracle_length=self.args.oracle_length, length_beam=self.args.length_beam, mbr=self.args.mbr
        )
        prev_decoder_out = prev_decoder_out._replace(
            step=0, max_step=self.args.max_iterations
        )
        for step in range(self.args.max_iterations):
            prev_decoder_out = self.denoise_step(model, prev_decoder_out, partial_masks)
            yield prev_decoder_out
            
    @torch.no_grad()
    def generate(self, model, inputs):
        src_tokens = inputs["net_input"]["src_tokens"]
        partial_masks = inputs["net_input"]["partial_masks"]
        # assert src_tokens.size(-1) < 514
        # assert partial_masks.size(-1) < 514
        # target = inputs["target"]
        # TODO: FIXME: to support general blockwise generation.
        if "prefix_masks" in inputs["net_input"]:
            prefix_masks = inputs["net_input"]["prefix_masks"]
        else:
            prefix_masks = partial_masks 
        partial_masks, prev_decoder_out = model.initialize_decode_samples(
            src_tokens, partial_masks, prefix_masks, oracle_length=self.args.oracle_length, length_beam=self.args.length_beam, mbr=self.args.mbr
        )
        prev_decoder_out = prev_decoder_out._replace(
            step=0, max_step=self.args.max_iterations
        )
        
        for step in range(self.args.max_iterations):
            prev_decoder_out = self.denoise_step(model, prev_decoder_out, partial_masks)            
            
        def finalized_hypos(tokens, scores, partial_mask, history=None):
            cutoff = (
                tokens.ne(self.pad_id) & 
                tokens.ne(self.bos_id) & 
                tokens.ne(self.eos_id) & 
                (~partial_mask)
            )
            tokens = tokens[cutoff]
            if scores is None:
                score = None
            else:
                scores = scores[cutoff]
                score = scores.mean().item()
            ret_dict = {
                "tokens": tokens,
                "positional_scores": scores,
                "score": score,
                "alignment": None
            }
            if history is not None:
                ret_dict["history"] = [
                    finalized_hypos(history_tokens, None, partial_mask, history=None)
                    for history_tokens in history
                ]
            return ret_dict
        
        def mbr_select(hyps):
            index = np.argmax(np.array(
                [self.rouge.get_scores([hyps[i]], [[hyps[j]]])['rouge-l']['f']
                 for j in range(len(hyps)) if i != j]
            ).mean() for i in range(len(hyps)))
            return hyps[index]
        
        def score_select(hyps):
            index = np.argmax([hyp["score"] for hyp in hyps])
            return hyps[index]
        
        output_tokens, output_scores = prev_decoder_out.output_tokens, prev_decoder_out.output_scores
        if self.retain_history:
            full_history = prev_decoder_out.history 
            histories = [[full_history[j][i] for j in range(self.args.max_iterations)] for i in range(output_tokens.size(0))]
            hyps = []
            for tokens, scores, partial_mask, history in zip(output_tokens, output_scores, partial_masks, histories):
                hyps.append(finalized_hypos(tokens, scores, partial_mask, history))
            # hyps = [
            #     finalized_hypos(tokens, scores, partial_mask, history)
            #     for tokens, scores, partial_mask, history in zip(output_tokens, output_scores, partial_masks, histories)
            # ]
        else:
            hyps = [
                finalized_hypos(tokens, scores, partial_mask, None) 
                for tokens, scores, partial_mask in zip(output_tokens, output_scores, partial_masks)
            ]
        repeatition = self.args.mbr * self.args.length_beam
        if repeatition > 1:
            hyps = [score_select(hyps[i:i+repeatition])for i in range(0, len(hyps), repeatition)]
            # hyps = [mbr_select(hyps[i:i+repeatition])for i in range(0, len(hyps), repeatition)]
            
        finalized = pad_sequence([h["tokens"] for h in hyps ], batch_first=True, padding_value=self.pad_id)
        history = [[item["tokens"] for item in h["history"]] for h in hyps] if self.retain_history else None
        return finalized, history