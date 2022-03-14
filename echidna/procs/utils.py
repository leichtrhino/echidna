
from datetime import datetime
import typing as tp
import json
import random
import numpy
import torch

from ..data.datasets import collate_fn
from ..models.utils import match_length
from ..metrics.composite import get_loss_name

from . import trainings
from . import validations

def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    torch.manual_seed(worker_seed)
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

def build_dataloader(dataset,
                     sample_size : int,
                     batch_size : int,
                     num_workers : int,
                     shuffle : bool,
                     seed : int):

    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
        numpy.random.seed(seed)
        random.seed(seed)

    if sample_size and len(dataset) > sample_size and shuffle:
        all_indices = list(range(len(dataset)))
        random.shuffle(all_indices)
        indices = all_indices[:sample_size]
        pass
    elif sample_size and len(dataset) > sample_size and not shuffle:
        indices = list(range(sample_size))
    else:
        indices = list(range(len(dataset)))
    dataset = torch.utils.data.Subset(dataset, indices)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        worker_init_fn=_seed_worker,
        generator=generator,
        collate_fn=collate_fn,
    )
    return loader



