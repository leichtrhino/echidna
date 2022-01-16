
import os
import typing as tp
import logging
import math
import json
import random
import multiprocessing
from datetime import datetime

import torch
import torchaudio
import numpy

from .transforms import Resample
from .utils import merge_activation

class Datasource(object):
    """
    Metadata for single raw wave and sheet pair

    Attributes
    ----------
    id : str
    wave_path : str
    sheet_path : str, optional
    category : str, optional
    track : str, optional
    fold : str, optional
    """
    def __init__(self,
                 id : str,
                 wave_path : str,
                 sheet_path : str,
                 category : str,
                 track : str=None,
                 fold : str=None):
        self.id = id
        self.wave_path = wave_path
        self.sheet_path = sheet_path
        self.category = category
        self.track = track
        self.fold = fold

    @classmethod
    def from_dict(cls, d : tp.Dict):
        """
        """
        return cls(id=d['id'],
                   wave_path=d['wave_path'],
                   sheet_path=d.get('sheet_path', None),
                   category=d.get('category', None),
                   track=d.get('track', None),
                   fold=d.get('fold', None))

    def to_dict(self):
        """
        """
        return {
            'id': self.id,
            'wave_path': str(self.wave_path) if self.wave_path else None,
            'sheet_path': str(self.sheet_path) if self.sheet_path else None,
            'category': self.category,
            'track': self.track,
            'fold': self.fold,
        }

class Sample(object):
    def __init__(self,
                 path : str,
                 categories : tp.List[str],
                 tracks : tp.List[str],
                 folds : str,
                 sample_rate : int):
        self.path = path
        self.categories = categories
        self.tracks = tracks
        self.folds = folds
        self.sample_rate = sample_rate

    @classmethod
    def from_dict(cls, d : tp.Dict):
        return cls(
            path=d['path'],
            categories=d['categories'],
            tracks=d.get('tracks', [None] * len(d['categories'])),
            folds=d.get('folds', [None] * len(d['categories'])),
            sample_rate=d['sample_rate'],
        )

    @classmethod
    def from_list(cls, l : tp.List):
        return [cls.from_dict(d) for d in l]

    def to_dict(self):
        return {
            'path': str(self.path) if self.path else None,
            'categories': self.categories,
            'tracks': self.tracks,
            'folds': self.folds,
            'sample_rate': self.sample_rate,
        }

class SampleSpec(object):
    """
    Metadata for datasource sampling

    Attributes
    ----------
    datasources : list
        list of Datasource objects
    sample_size : int
    source_per_category : int
    sample_rate : int
    duration : float
    seed : int
    """

    def __init__(self,
                 datasources : tp.List[Datasource],
                 fold : str,
                 sample_size : int,
                 source_per_category : int,
                 source_by_category : dict,
                 sample_rate : int,
                 duration : float,
                 seed : int,
                 metadata_path : str,
                 data_dir : str,
                 journal_path : str,
                 log_path : str,
                 log_level : str,
                 jobs : int=None):

        self.datasources = datasources
        self.fold = fold
        self.sample_size = sample_size
        self.source_per_category = source_per_category
        self.source_by_category = source_by_category
        self.sample_rate = sample_rate
        self.duration = duration
        self.seed = seed
        self.metadata_path = metadata_path
        self.data_dir = data_dir
        self.journal_path = journal_path
        self.log_path = log_path
        self.log_level = log_level
        self.jobs = jobs

    @classmethod
    def from_dict(cls, d : tp.Dict):
        return cls(
            datasources=[Datasource.from_dict(s) for s in d['datasources']],
            fold=d['fold'],
            sample_size=d['sample_size'],
            source_per_category=d['source_per_category'],
            source_by_category=d.get('source_by_category'),
            sample_rate=d['sample_rate'],
            duration=d['duration'],
            seed=d['seed'],
            metadata_path=d['metadata_path'],
            data_dir=d['data_dir'],
            journal_path=d.get('journal_path', None),
            log_path=d.get('log_path'),
            log_level=d.get('log_level'),
            jobs=d.get('jobs', None),
        )

    def to_dict(self):
        return {
            'datasources': [s.to_dict() for s in self.datasources],
            'fold': self.fold,
            'sample_size': self.sample_size,
            'source_per_category': self.source_per_category,
            'source_by_category': self.source_by_category,
            'sample_rate': self.sample_rate,
            'duration': self.duration,
            'seed': self.seed,
            'metadata_path': str(self.metadata_path) if self.metadata_path else None,
            'data_dir': str(self.data_dir) if self.data_dir else None,
            'journal_path': str(self.journal_path) if self.journal_path else None,
            'log_path': str(self.log_path) if self.log_path else None,
            'log_level': self.log_level,
            'jobs': self.jobs,
        }

    def save_samples(self):
        _save_sample(self)

class SampleJournal(object):
    """
    Journal class for a sample

    Attributes
    ----------
    created_at : datetime
    seed : int
    datasources : tp.List[tp.List[str]]
    length : int
    offsets : tp.List[int]
    sample : Sample)
    """

    def __init__(self,
                 created_at : datetime,
                 seed : int,
                 datasources : tp.List[tp.List[str]],
                 length : int,
                 offsets : tp.List[int],
                 sample : Sample):
        self.created_at = created_at
        self.seed = seed
        self.datasources = datasources
        self.offsets = offsets
        self.length = length
        self.sample = sample

    @classmethod
    def from_dict(cls, d : tp.Dict):
        return cls(
            created_at=datetime.fromisoformat(d['created_at']),
            seed=d['seed'],
            datasources=d.get('datasources', []),
            length=d['length'],
            offsets=d['offsets'],
            sample=Sample.from_dict(d['sample'])
        )

    def to_dict(self):
        return {
            'created_at': self.created_at.isoformat(),
            'seed': self.seed,
            'datasources': self.datasources,
            'length': self.length,
            'offsets': self.offsets,
            'sample': self.sample.to_dict(),
        }

class SamplesJournal(object):
    """
    Journal class of samples

    Attributes
    ----------
    process_start : timestamp
    process_finish : timestamp
    metadata_path : str
    spec : SampleSpecification
    seed : int
    samples : tp.List[SampleJournal])

    """
    def __init__(self,
                 process_start : datetime,
                 process_finish : datetime,
                 metadata_path : str,
                 log_path : str,
                 spec,
                 seed : int,
                 sample_journals : tp.List[SampleJournal]):
        self.process_start = process_start
        self.process_finish = process_finish
        self.metadata_path = metadata_path
        self.log_path = log_path
        self.spec = spec
        self.seed = seed
        self.sample_journals = sample_journals

    @classmethod
    def from_dict(cls, d : tp.Dict):
        return cls(
            process_start=datetime.fromisoformat(d['process_start']),
            process_finish=datetime.fromisoformat(d['process_finish']),
            metadata_path=d['metadata_path'],
            log_path=d.get('log_path'),
            seed=d['seed'],
            spec=SampleSpec.from_dict(d['spec']),
            sample_journals=[
                SampleJournal.from_dict(j)
                for j in d['sample_journals']
            ],
        )

    def to_dict(self):
        return {
            'process_start': self.process_start.isoformat(),
            'process_finish': self.process_finish.isoformat(),
            'metadata_path': str(self.metadata_path) if self.metadata_path else None,
            'log_path': str(self.log_path) if self.log_path else None,
            'seed': self.seed,
            'spec': self.spec.to_dict(),
            'sample_journals': [j.to_dict() for j in self.sample_journals]
        }

def _save_sample(spec : SampleSpec):
    """
    Save sample to files

    """
    process_start = datetime.now()

    # datasources to select
    fold = [spec.fold] if type(spec.fold) == str else spec.fold
    datasources = [
        ds for ds in spec.datasources
        if fold is None or ds.fold in fold
    ]

    # prepare log
    logger = None
    if spec.log_path:
        if not os.path.exists(os.path.dirname(spec.log_path)):
            os.makedirs(os.path.dirname(spec.log_path))
        logger = logging.getLogger(__name__)
        logger.setLevel(spec.log_level)
        handler = logging.FileHandler(str(spec.log_path))
        handler.setFormatter(
            logging.Formatter('[%(levelname)s] %(message)s'))
        logger.addHandler(handler)

    # args for worker function
    random_ = random.Random(spec.seed)
    rel_paths = [
        os.path.join(f'{i//1000:03d}', f'{i%1000:03d}.pth')
        for i in range(spec.sample_size)
    ]
    args = [(
        datasources,
        spec.source_per_category,
        spec.source_by_category,
        spec.sample_rate,
        spec.duration,
        spec.metadata_path,
        os.path.join(spec.data_dir, rel_path),
        random_.randrange(2**32), # seed
    ) for rel_path in rel_paths]

    # create map function
    if spec.jobs is not None:
        pool = multiprocessing.Pool(spec.jobs)
        map_fn = pool.imap_unordered
    else:
        map_fn = map

    if logger:
        logger.info(json.dumps({
            'type': 'start_sampling',
            'timestamp': datetime.now().isoformat(),
            'sample_size': spec.sample_size,
            'source_per_category': spec.source_per_category,
            'source_by_category': spec.source_by_category,
            'sample_rate': spec.sample_rate,
            'duration': spec.duration,
            'seed': spec.seed,
            'jobs': spec.jobs,
        }))

    # sample
    metadata_list = []
    journal_list = []
    for metadata, journal in map_fn(_save_single_sample, args):
        metadata_list.append(metadata)
        journal_list.append(journal)
        if logger:
            logger.info(json.dumps({
                'type': 'made_sample',
                'timestamp': datetime.now().isoformat(),
                'sample_path': journal.sample.path,
                'channels': len(journal.sample.categories),
            }))

    # close map function
    if spec.jobs is not None:
        pool.close()

    process_finish = datetime.now()

    # save metadata
    with open(spec.metadata_path, 'w') as fp:
        json.dump([m.to_dict() for m in metadata_list], fp)

    if logger:
        logger.info(json.dumps({
            'type': 'save_samples',
            'timestamp': datetime.now().isoformat(),
            'metadata_path': str(spec.metadata_path),
            'sample_size': len(metadata_list),
        }))

    if spec.journal_path:
        journals = SamplesJournal(
            process_start=process_start,
            process_finish=process_finish,
            metadata_path=os.path.relpath(
                spec.metadata_path,
                os.path.dirname(spec.journal_path)
            ),
            log_path=os.path.relpath(
                spec.log_path,
                os.path.dirname(spec.journal_path)
            ) if spec.log_path else None,
            spec=spec,
            seed=spec.seed,
            sample_journals=journal_list
        )
        with open(spec.journal_path, 'w') as fp:
            json.dump(journals.to_dict(), fp)

        if logger:
            logger.info(json.dumps({
                'type': 'save_samples_journal',
                'timestamp': datetime.now().isoformat(),
                'journal_path': str(spec.journal_path),
            }))

    # finish epoch and close log handler
    if logger:
        logger.info(json.dumps({
            'type': 'finish_sampling',
            'timestamp': datetime.now().isoformat(),
        }))
        handlers = logger.handlers[:]
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()


def _save_single_sample(args):
    """

    Parameters
    ----------
    datasources : tp.List[Datasource],
    source_per_category : int,
    source_by_category : dict,
    sample_rate : int,
    duration : float,
    metadata_path : str,
    data_path : str,
    rel_path : str,
    seed : int
    """
    datasources, source_per_category, source_by_category, \
        sample_rate, duration, \
        metadata_path, data_path, seed = args
    wavelength = int(sample_rate * duration)

    random.seed(seed)
    torch.manual_seed(seed)
    numpy.random.seed(seed)

    # select track
    tracks = dict()
    for ds in datasources:
        if ds.track not in tracks:
            tracks[ds.track] = []
        tracks[ds.track].append(ds)
    if len(tracks) == 1 and None in tracks:
        track = None
    else:
        track = random.choice([t for t in tracks.keys() if t is not None])

    # calculate category list
    categories = dict()
    for ds in tracks[track] \
        + (tracks.get(None, []) if track is not None else []):
        if ds.category not in categories:
            categories[ds.category] = []
        categories[ds.category].append(ds)

    for ds_list in categories.values():
        random.shuffle(ds_list)

    # loop
    waves = []
    sheets = []
    datasources = []
    for category, ds_list in categories.items():
        if type(source_by_category) == dict and \
           source_by_category.get(category, 0) > 0:
            max_category_num = source_by_category[category]
        else:
            max_category_num = source_per_category
        source_i = 0
        source_num = 0
        # NOTE: Lagging could be occur if tracked and no-tracked waves
        # are in single category
        while source_num < max_category_num \
              and source_i < len(ds_list):
            wave = None
            sheet = None
            selected_ds = []
            while (wave is None or wave.shape[-1] < wavelength) \
                  and source_i < len(ds_list):
                ds = ds_list[source_i]
                source_i += 1
                selected_ds.append(ds)
                # load wav and concat
                w, orig_sr = torchaudio.load(ds.wave_path)
                w = w.mean(dim=0)
                w = Resample(orig_sr, sample_rate)(w)
                wave = w if wave is None else torch.cat((wave, w), dim=-1)
                # load midi
                if ds.sheet_path:
                    sheet = None
                    raise NotImplementedError()

            # append
            waves.append(wave)
            sheets.append(sheet)
            datasources.append(selected_ds)
            source_num += 1

    # calculate activation
    track_activations = dict(
        (ds[0].track, [(0, w.shape[-1], [])])
        for ds, w in zip(datasources, waves)
        if ds[0].track is not None
    )
    source_activations = [
        track_activations.get(ds[0].track, [(0, w.shape[-1], [])])
        for ds, w in zip(datasources, waves)
    ]
    for i, (a, w) in enumerate(zip(source_activations, waves)):
        merge_activation(a, w, i)

    # choose activation
    track_activation = dict()
    for k, a in track_activations.items():
        weights = [
            (end-start)*(10**(len(tags)-1)) if len(tags) else 0
            for start, end, tags in a
        ]
        track_activation[k] =\
            random.choices(a, weights=weights, k=1)[0]

    source_activation = []
    for i, (ds, a) \
        in enumerate(zip(datasources, source_activations)):
        if i not in set(tag for _, _, tags in a for tag in tags):
            source_activation.append(None)
        elif ds[0] in track_activations:
            source_activation.append(track_activation[ds[0]])
        else:
            weights = [
                end - start if len(tags) > 0 else 0
                for start, end, tags in a
            ]
            source_activation.append(
                random.choices(a, weights=weights, k=1)[0]
            )

    # crop waveforms and sheets
    crop_waves = []
    crop_sheets = []
    crop_datasources = []
    crop_offsets = []
    for ds, w, s, a in zip(datasources, waves, sheets, source_activation):
        if a is None:
            continue
        start, end, _ = a
        start = max(0, start - wavelength // 4)
        end = min(w.shape[-1], end + wavelength // 4)
        end = max(start, end - wavelength)
        offset = random.randint(start, end)
        pad = max(0, offset + wavelength - w.shape[-1])
        w = torch.cat((
            w[..., offset:min(offset+wavelength, w.shape[-1])],
            torch.zeros((*w.shape[:-1], pad), device=w.device)
        ), dim=-1)
        if s is not None:
            raise NotImplementedError()
        crop_waves.append(w)
        crop_sheets.append(s)
        crop_datasources.append(ds)
        crop_offsets.append(offset)

    # save
    dirname = os.path.dirname(data_path)
    if not os.path.isdir(dirname):
        os.makedirs(dirname, exist_ok=True)
    torch.save({
        'waves': torch.stack(crop_waves),
        'sheets': crop_sheets,
    }, data_path)

    # create metadata
    sample_metadata = Sample(
        path=os.path.relpath(data_path, os.path.dirname(metadata_path)),
        categories=[dss[0].category for dss in crop_datasources],
        tracks=[dss[0].track for dss in crop_datasources],
        folds=[dss[0].fold for dss in crop_datasources],
        sample_rate=sample_rate
    )

    journal_metadata = SampleJournal(
        created_at=datetime.now(),
        seed=seed,
        datasources=[
            [ds.id for ds in dss]
            for dss in crop_datasources
        ],
        length=wavelength,
        offsets=crop_offsets,
        sample=sample_metadata,
    )

    return sample_metadata, journal_metadata

