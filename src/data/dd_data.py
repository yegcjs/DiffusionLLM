import torch
from torch.utils.data import Dataset, IterableDataset, BatchSampler, DistributedSampler

from fairseq.data import Dictionary
from fairseq.tasks.translation import load_langpair_dataset
from fairseq.data.language_pair_dataset import collate
from fairseq.data import data_utils, FairseqDataset, Dictionary

from transformers import AutoTokenizer
from transformers.trainer_pt_utils import DistributedLengthGroupedSampler, get_length_grouped_indices

# from src.task.partial_discrete_diffusion_task import PartialDiffusionLanguagePairDataset, concat_func

from typing import Any, Dict, Iterator, List, Optional, Union

from dataclasses import dataclass, field

from functools import partial

import math

import numpy as np

from datasets import load_dataset, load_from_disk

from tqdm import tqdm

import json

import multiprocessing as mp


@dataclass
class DiscreteDiffusionDataArguments:
    dataset_type: str = field(
        default="fairseq"   # fairseq | flan | flanv2 | pair
    )
    data_path: str = field(
        default=""
    )
    src_lang: str = field(
        default=""
    )
    tgt_lang: str = field(
        default=""
    )
    prompt_built: bool = field(
        default=False
        # help="only for flan"
    )
    # batch_by_tokens: bool = field(
    #     default=False
    # )
    max_length: int = field(
        default=2048
    )
    packing: bool = field(
        default=False,
        metadata={"help": "whether to pack the output data"}
    )


class PromptDataset(Dataset):
    def __init__(self, args, raw_data, tokenizer):
        super().__init__()
        self.args = args
        self.raw_data = raw_data
        self.tokenizer = tokenizer
        self.item_size = {}
        self.max_length = args.max_length
    
    # def set_max_length(self, max_length):
    #     self.max_length = max_length
        
    def __len__(self):
        return len(self.raw_data)
    
    def size(self, index):
        if index not in self.item_size:
            item = self.__getitem__(index)
            self.item_size[index] = len(item["source"])
        return self.item_size[index]
    
    def ordered_indices(self):
        raise NotImplementedError
    
    @staticmethod
    def load_data(args, tokenizer, train=True, valid=True, test=False):
        raise NotImplementedError

    def build_prompt(self, item):
        raise NotImplementedError
    
    def __getitem__(self, index):
        raise NotImplementedError
    
def concat_func(dictionary, src_tokens, tgt_tokens):
    if src_tokens[-1] == dictionary.eos():
        src_tokens = src_tokens[:-1]
    if tgt_tokens[0] == dictionary.bos():
        tgt_tokens = tgt_tokens[1:]
    return torch.cat([src_tokens, tgt_tokens], dim=-1)

class PartialDiffusionLanguagePairDataset(FairseqDataset):
    def __init__(self, language_pair_dataset, concat_func):
        super().__init__()
        self.dataset = language_pair_dataset
        self.concat_func = concat_func
        self.eos_id = self.dataset.tgt_dict.eos()
        self.bos_id = self.dataset.tgt_dict.bos()
        self.pad_id = self.dataset.tgt_dict.pad()
    
    def __getitem__(self, index):
        sample = self.dataset[index]
        sample["source"] = self.concat_func(sample["source"], sample["target"])
        return sample
    
    def collater(self, samples):
        samples = self.dataset.collater(samples)
        if samples == {}:
            return samples
        try:
            source = samples["net_input"]["src_tokens"]
        except:
            import pdb
            pdb.set_trace()
        
        full_length = (source.ne(self.pad_id)).sum(-1)
        ori_src_length = full_length - (
            samples["target"].ne(self.pad_id) &
            samples["target"].ne(self.bos_id)
        ).sum(-1)

        samples["net_input"]["partial_masks"] = torch.arange(source.size(-1)).expand_as(source) < ori_src_length[:, None]
        return samples
         
    def __len__(self):
        return len(self.dataset)
    
    def get_batch_shapes(self):
        return self.dataset.get_batch_shapes()
    
    def filter_indices_by_size(self, indices, max_sizes):
        return self.dataset.filter_indices_by_size(indices, max_sizes)

    def size(self, index):
        return len(self[index]["source"])
    
    def batch_by_size(self, indices, max_tokens=None, max_sentences=None, required_batch_size_multiple=1):
        return self.dataset.batch_by_size(indices, max_tokens, max_sentences, required_batch_size_multiple)
    
    def num_tokens(self, index):
        return self.dataset.num_tokens(index)
    
    def num_tokens_vec(self, indices):
        return self.dataset.num_tokens_vac(indices)
    
    def ordered_indices(self):
        return self.dataset.ordered_indices()
    
    @property
    def supports_prefetch(self):
        return self.dataset.supports_prefetch
    
    def prefetch(self, indices):
        return self.dataset.prefetch(indices)
    
    @property
    def supports_fetch_outside_dataloader(self):
        return self.dataset.supports_fetch_outside_dataloader
 
class FairseqLangPairDataset(PromptDataset):
    def size(self, index):
        return self.raw_data.size(index)
    
    def ordered_indices(self):
        return self.raw_data.ordered_indices()
    
    def __getitem__(self, index):
        return self.raw_data[index]
         
    @staticmethod
    def load_data(args, train=True, valid=True, test=False):
        # raise NotImplementedError
        # dictionary
        dictionary = Dictionary.load(f"{args.data_path}/dict.{args.src_lang}.txt")
        setattr(dictionary, "mask_index", dictionary.index("<mask>"))    
        
        # dataset
        datasets = []
        load_split = {"train": train, "valid": valid, "test": test}
        for split in ["train", "valid", "test"]:
            datasets.append(
                FairseqLangPairDataset(
                    args,
                    PartialDiffusionLanguagePairDataset(
                        load_langpair_dataset(
                            args.data_path, split, args.src_lang, dictionary, args.tgt_lang, dictionary,
                            combine=False, dataset_impl=None, upsample_primary=-1,
                            left_pad_source=False, left_pad_target=False,
                            max_source_positions=2048, max_target_positions=2048, # filter in sampelr
                            prepend_bos=True
                        ),
                        partial(concat_func, dictionary)
                    ),
                    tokenizer=None
                ) if load_split[split] else None
            ) 
        return dictionary, datasets
    

class PairDataset(PromptDataset):
    def __getitem__(self, index):
        item = self.raw_data[index]
        src = self.tokenizer.encode(item["inputs"])
        tgt = self.tokenizer.encode(item["targets"])[-self.max_length:]
        return {
            "id": index,
            "source": torch.tensor(src[:-1] + tgt[1:])[-self.max_length:],
            "target": torch.tensor(tgt)
        }
    
    @staticmethod    
    def load_data(args, tokenizer, train=True, valid=True, test=False):
        # assert (not train and not valid and test)   # test only
        # tokenizer = AutoTokenizer.from_pretrained(args.pretrained, cache_dir=args.cache_dir)
        with open(args.data_path, "r") as f:
            raw_data = [json.loads(line) for line in f]
        if train:
            return (PairDataset(args, raw_data, tokenizer), None, None)
        else:
            return (None, None, PairDataset(args, raw_data, tokenizer))

class FlanInstructionTuningDataset(PromptDataset):
    def __init__(self, args, raw_data, tokenizer):
        super().__init__(args, raw_data, tokenizer)
        raise NotImplementedError
        skip_range = set(range(1174950, 1204949)) | set(range(2561000, 2591000))   # wmt_ende
        self.valid_indices = [i for i in range(len(raw_data)) if i not in skip_range]

    def __len__(self):
        return len(self.valid_indices)
    
    @staticmethod
    def load_data(args, tokenizer, train=True, valid=True, test=False):
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        num_proc = mp.cpu_count() / world_size
        try:
            datasets = load_dataset(args.data_path, cache_dir=args.cache_dir, num_proc=num_proc)
        except:
            datasets = load_from_disk(args.data_path) 
        # tokenizer = AutoTokenizer.from_pretrained(args.pretrained, cache_dir=args.cache_dir)
        splits = [
            FlanInstructionTuningDataset(args, datasets["train"], tokenizer) if train else None,
            FlanInstructionTuningDataset(args, datasets["validation"], tokenizer) if valid else None,
            FlanInstructionTuningDataset(args, datasets["test"], tokenizer) if test else None
        ]
        return splits
    
    def build_prompt(self, item):
        prompt = item["inputs"].strip()
        if not self.args.prompt_built:
            if prompt[-1] not in {'.', '?'}:
                prompt = prompt + '.'
            prompt += " Answer:"
        return prompt
    
    def __getitem__(self, index):
        index2 = self.valid_indices[index]
        item = self.raw_data[index2]
        src_with_prompt = self.build_prompt(item)
        src = self.tokenizer.encode(src_with_prompt, return_tensors='pt')[0]
        tgt = self.tokenizer.encode(item["targets"], return_tensors='pt')[0, -self.max_length:]
        return {
            "id": index,
            "source": torch.cat([src[:-1], tgt[1:]])[-self.max_length:],
            "target": tgt
        }


class FlanV2Dataset(IterableDataset):
    def __init__(self, args, data_path, tokenizer) -> None:
        super().__init__()
        self._full_file_name = []
        with open(f"{data_path}/ratio.json", "r") as f:
            ratios = json.load(f)
        self.sampler = torch.distributions.Categorical(torch.tensor([ratio for _, ratio in ratios.items()]))
        self._full_file_name = [f"{data_path}/{file}" for file in ratios]
        self.idx2dataset = [open(file, "r") for file in self._full_file_name]
        self.args = args
        self.tokenizer = tokenizer
        
        self.counter = 0
        
        self.rank = rank =0 if not torch.distributed.is_initialized() else  torch.distributed.get_rank()
        world_size = 1 if not torch.distributed.is_initialized() else torch.distributed.get_world_size() 

        self.global_batch_size = args.per_device_batch_size * world_size
        self.read_step_size = world_size
        self.counter_range = set(range(
            rank * args.per_device_batch_size, (rank + 1) * args.per_device_batch_size
        ))
        
        for dataset in self.idx2dataset:
            for _ in range(rank):
                dataset.readline()       
    
    def read_step(self, dataset=None, index=None):
        # select a dataset first
        if dataset is None:
            index = self.sampler.sample()
            dataset = self.idx2dataset[index]
        for _ in range(self.read_step_size):
            line = dataset.readline()
            if not line:
                dataset.close()
                dataset = self.idx2dataset[index] = open(self._full_file_name[index], "r")
                line = dataset.readline()
        # tokenize
        item = json.loads(line)
        if "inputs_ids" in item and "targets_ids" in item:
            src = item["inputs_ids"] 
            tgt = item["targets_ids"]
        else:
            src = self.tokenizer.encode(item["inputs"])
            tgt = self.tokenizer.encode(item["targets"])

        if len(src) + len(tgt) - 2 > self.args.max_length:
            # lets do some concat
            tgt = tgt[-self.args.max_length:]
            # return self.read_step(dataset, index)
        if src[-1] == self.tokenizer.eos_token_id:
            src = src[:-1]
        return {
            "id": self.counter,
            "source": torch.tensor(src+ tgt[1:])[-self.args.max_length:],
            "target": torch.tensor(tgt)
        }
                
    @staticmethod
    def load_data(args, tokenizer, train=True, valid=True, test=False):
        assert not test
        return (FlanV2Dataset(args, args.data_path, tokenizer=tokenizer), None, None)
    
    def __iter__(self) -> Iterator:
        # lets do some hacking
        # hf trainer wrap the dataset with IterableDatasetShard to avoid duplicas in DDP
        # this iter should only yield valid sample during the range its sample will be used
        # and it should skip N 
        
        # assume single worker 
        assert torch.utils.data.get_worker_info() is None
        while True:
            if (self.counter % self.global_batch_size) not in self.counter_range:
                yield None
            else:
                yield self.read_step()
            self.counter += 1


@dataclass
class MemoryMapTokensDataset(Dataset):
    def __init__(self, args, data_path, tokenizer):
        super().__init__()
        self.args = args
        self.tokens = np.memmap(data_path, dtype="ushort", mode="r")
        self.num_total_tokens = self.tokens.shape[0]
        self.length = args.max_length
      
    def __len__(self):
        return self.num_total_tokens // self.length
    
    def __getitem__(self, index):
        start, end = index * self.length, (index + 1) * self.length
        data = np.array(self.tokens[start:end], dtype=int)
        return {
            "id": index,
            "source": torch.tensor(data),
            "target": torch.tensor(data)
        }
      
    @staticmethod
    def load_data(args, tokenizer, train=True, valid=False, test=False):
        assert (not test)
        return (MemoryMapTokensDataset(args, args.data_path, tokenizer=tokenizer), None, None)
        
         
@dataclass
class DiscreteDiffusionDataCollator(object):
    
    bos_id: int
    eos_id: int
    pad_id: int
    
    def __call__(self, samples):
        for sample in samples:
            assert sample is not None
        samples = collate(
            samples, self.pad_id, self.eos_id,
            left_pad_source=False, left_pad_target=False
        )
        if samples == {}:
            return samples
        try:
            source = samples["net_input"]["src_tokens"]
        except:
            import pdb
            pdb.set_trace()
        full_length = (source.ne(self.pad_id)).sum(-1)
        ori_src_length = full_length - (
            samples["target"].ne(self.pad_id) &
            samples["target"].ne(self.bos_id)
        ).sum(-1)

        samples["net_input"]["partial_masks"] = torch.arange(source.size(-1)).expand_as(source) < ori_src_length[:, None]
        return samples

# batch sampler
# class TokenSizeDistributedLengthGroupSampler(BatchSampler):
#     pass 

# class DistributedBatchSampler(BatchSampler):
#     def __init__(self, batch_sampler, **kwargs):
#         self.batch_sampler = batch_sampler
#         self.kwargs = kwargs

#     def __iter__(self):
#         for batch in self.batch_sampler:
#             yield list(DistributedSampler(batch, **self.kwargs))

#     def __len__(self):
#         return len(self.batch_sampler)
    
class TokenSizeDistributedLengthGroupSampler(DistributedLengthGroupedSampler):
    def __init__(
        self,
        batch_size: int,
        max_length: int,
        dataset: Optional[Dataset],
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        seed: int = 0,
        drop_last: bool = False,
        lengths: Optional[List[int]] = None,
        model_input_name: Optional[str] = None,
        infinite: bool = False
    ):
        super().__init__(batch_size, dataset, num_replicas, rank, seed, drop_last, lengths, model_input_name)
        self.max_length = max_length
        self.dataset = dataset
        self.infinite = infinite
        
        self.num_batches = None
    
    def __len__(self):
        return self.num_batches if self.num_batches is not None else 0x7fffffff 
        
    def __iter__(self) -> Iterator:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = self.dataset.ordered_indices()
        # indices, _ = self.dataset.filter_indices_by_size(indices, self.max_length)
        indices = [index for index in indices if self.lengths[index] <= self.max_length]
        batches = data_utils.batch_by_size(
            indices, num_tokens_fn=None, num_tokens_vec=[self.lengths[index] for index in indices],
            max_tokens=self.batch_size
        )

        num_good_batches = math.floor(len(batches) / self.num_replicas) * self.num_replicas
        total_batches = math.ceil(len(batches) / self.num_replicas) * self.num_replicas
        
        while sum(len(batch) for batch in batches[num_good_batches:]) < total_batches - num_good_batches:
            num_good_batches -= self.num_replicas
        
        new_batches = batches[:num_good_batches]
        reallocate_batches = batches[num_good_batches:]
        reallocate_batches.extend([[] for _ in range(total_batches - len(batches))])
        
        i, j = 0, len(reallocate_batches) - 1
        while i < j:
            while len(reallocate_batches[i]) <= 1 and i < j:
                i = i + 1
            while len(reallocate_batches[j]) > 0 and i < j:
                j = j - 1
            if i >= j:
                break 
            reallocate_batches[j] = [reallocate_batches[i][0]]
            reallocate_batches[i] = reallocate_batches[i][1:]
        new_batches.extend(reallocate_batches)
        assert (len(new_batches) % self.num_replicas == 0)
        batches = new_batches[self.rank : len(new_batches) : self.num_replicas]
        i, num_batches = 0, len(batches)
        self.num_batches = num_batches

        while True:
            yield batches[i]
            i = (i + 1) % num_batches
            if not self.infinite and i <= 0:
                break
            