
import typing as tp
import math
import random
import numpy
import torch

from .datasets import Dataset

def build_dataloader(dataset : Dataset,
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

    if sample_size and shuffle:
        repeats = math.ceil(sample_size / len(dataset))
        base_indices = list(range(len(dataset)))
        extra_indices = list(range(len(dataset)))
        random.shuffle(extra_indices)
        indices = (base_indices * (repeats-1) + extra_indices)[:sample_size]
    elif sample_size and not shuffle:
        # +0.5 to avoid uneven sampling when |dataset| << sample_size
        indices = numpy.linspace(
            0.5, (len(dataset)-1)+0.5, sample_size, dtype=int).tolist()
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

def collate_fn(l : tp.List[tp.Tuple[tp.Dict[str, torch.Tensor], tp.Dict]]):
    data_tuple, metadata_tuple = zip(*l)
    collate_data = {
        'waves': torch.stack([d['waves'] for d in data_tuple], dim=0),
        'sheets': None
    }
    collate_metadata = {
        'index': [d['index'] for d in metadata_tuple],
        'sample': [d['sample'] for d in metadata_tuple],
        'augmentation': [d['augmentation'] for d in metadata_tuple],
        'mixture': [d['mixture'] for d in metadata_tuple],
    }

    return collate_data, collate_metadata

def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    torch.manual_seed(worker_seed)
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

