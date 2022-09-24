
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

from .transforms import Resample, TimeStretch, MultiPointScale
from .utils import merge_activation
from .datanodes import DataNode, EmptyNode, add_datanode_cls

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

class Channel(object):
    def __init__(self,
                 path : str,
                 category : str,
                 track : str,
                 fold : str,
                 sample_rate : int):
        self.path = path
        self.category = category
        self.track = track
        self.fold = fold
        self.sample_rate = sample_rate

    @classmethod
    def from_dict(cls, d : tp.Dict):
        return cls(
            path=d['path'],
            category=d['category'],
            track=d.get('track', None),
            fold=d.get('fold', None),
            sample_rate=d['sample_rate'],
        )

    @classmethod
    def from_list(cls, l : tp.List):
        return [cls.from_dict(d) for d in l]

    def to_dict(self):
        return {
            'path': str(self.path) if self.path else None,
            'category': self.category,
            'track': self.track,
            'fold': self.fold,
            'sample_rate': self.sample_rate,
        }

class SampleNode(DataNode):
    def __init__(self, channels, rel_path, children=None, metrics=None, device='cpu'):
        super().__init__(children, metrics)
        self.channels = channels
        self.rel_path = rel_path
        self.device = device

    def process_partial(self, channel_index):
        if channel_index is None:
            channel_index = list(range(self.sample.channels))
        return [
            dict(
                (k, v.to(self.device) if v is not None else None)
                for k, v in torch.load(
                        os.path.join(self.rel_path, self.channels[ci].path)
                ).items()
            )
            for ci in channel_index
        ]

    def require_channel_index_partial(self):
        return list(range(len(self.channels)))

    @property
    def super_type(self):
        return 'source'

    @classmethod
    def from_dict_args(cls,
                       obj : dict,
                       context : dict=None):
        return cls(
            channels=[Channel.from_dict(c) for c in obj['channels']],
            rel_path=context.get('rel_path', ''),
            device=context.get('device') or 'cpu',
        )

    def to_dict_args(self):
        return {
            'channels': [c.to_dict() for c in self.channels],
        }

add_datanode_cls('sample', SampleNode)


class SampleSpec(object):
    """
    Metadata for datasource sampling

    Attributes
    ----------
    datasources : list
        list of Datasource objects
    fold : str
    sample_size : int
    category_map : dict
        key: target category name
        value: dict of
            'max_channels': int (default: 1)
            'sources': dict of following:
                key: source category name (c.f. datasource)
                value: source selector configuration:
                    'weight': float (default: 1.0)

    sample_rate : int
    duration : float
    target_db : float
    seed : int

    metadata_path : str
    data_dir : str
    journal_path : str
    log_path : str
    log_level : str
    jobs : int
    """

    def __init__(self,
                 datasources : tp.List[Datasource],
                 fold : str,
                 sample_size : int,
                 category_map : dict,
                 sample_rate : int,
                 duration : float,
                 target_db : float,
                 seed : int,
                 metadata_path : str,
                 data_dir : str,
                 journal_path : str,
                 log_path : str,
                 log_level : str,
                 jobs : int=None,
                 device : str='cpu'):

        if type(category_map) != dict:
            raise ValueError('type of category map must be dict')
        for k1, v1 in category_map.items():
            if type(v1) != dict:
                raise ValueError('type of values of category map must be dict')
            if 'sources' not in v1 or type(v1['sources']) != dict or\
               not v1['sources']:
                raise ValueError('value of category map must include source')
            if 'max_samples' not in v1:
                v1['max_samples'] = 1

            for k2, v2 in v1['sources'].items():
                if v2 is not None and type(v2) != dict:
                    raise ValueError('type of values of values of '
                                     'category map must be dict or None')
                if v2 is None:
                    v2 = dict()
                    v1[k2] = v2
                if 'weight' not in v2:
                    v2['max_samples'] = 1.0

        self.datasources = datasources
        self.fold = fold
        self.sample_size = sample_size
        self.category_map = category_map
        self.sample_rate = sample_rate
        self.duration = duration
        self.target_db = target_db
        self.seed = seed
        self.metadata_path = metadata_path
        self.data_dir = data_dir
        self.journal_path = journal_path
        self.log_path = log_path
        self.log_level = log_level
        self.jobs = jobs
        self.device = device

    @classmethod
    def from_dict(cls, d : tp.Dict):
        return cls(
            datasources=[Datasource.from_dict(s) for s in d['datasources']],
            fold=d['fold'],
            sample_size=d['sample_size'],
            category_map=d['category_map'],
            sample_rate=d['sample_rate'],
            duration=d['duration'],
            target_db=d.get('target_db'),
            seed=d['seed'],
            metadata_path=d['metadata_path'],
            data_dir=d['data_dir'],
            journal_path=d.get('journal_path', None),
            log_path=d.get('log_path'),
            log_level=d.get('log_level'),
            jobs=d.get('jobs', None),
            device=d.get('device') or 'cpu',
        )

    def to_dict(self):
        return {
            'datasources': [s.to_dict() for s in self.datasources],
            'fold': self.fold,
            'sample_size': self.sample_size,
            'category_map': self.category_map,
            'sample_rate': self.sample_rate,
            'duration': self.duration,
            'target_db': self.target_db,
            'seed': self.seed,
            'metadata_path': str(self.metadata_path) if self.metadata_path else None,
            'data_dir': str(self.data_dir) if self.data_dir else None,
            'journal_path': str(self.journal_path) if self.journal_path else None,
            'log_path': str(self.log_path) if self.log_path else None,
            'log_level': self.log_level,
            'jobs': self.jobs,
            'device': self.device,
        }

    def save_samples(self):
        _save_sample(self)


class ChannelJournal(object):
    """
    Journal class for a channel

    Attributes
    ----------
    created_at : datetime
    seed : int
    datasources : tp.List[str]
    length : int
    offset : int
    channel : Channel
    """
    def __init__(self,
                 created_at : datetime,
                 seed : int,
                 datasources : tp.List[str],
                 length : int,
                 offset : int,
                 channel : Channel):
        self.created_at = created_at
        self.seed = seed
        self.datasources = datasources
        self.length = length
        self.offset = offset
        self.channel = channel

    @classmethod
    def from_dict(cls, d : tp.Dict):
        return cls(
            created_at=datetime.fromisoformat(d['created_at']),
            seed=d['seed'],
            datasources=[
                Datasource.from_dict(ds)
                for ds in d.get('datasources', [])
            ],
            length=d['length'],
            offset=d['offset'],
            channel=Channel.from_dict(d['channel']),
        )

    def to_dict(self):
        return {
            'created_at': self.created_at.isoformat(),
            'seed': self.seed,
            'datasources': [ds.to_dict() for ds in self.datasources],
            'length': self.length,
            'offset': self.offset,
            'channel': self.channel.to_dict(),
        }


class SampleJournal(object):
    """
    Journal class for a sample

    Attributes
    ----------
    created_at : datetime
    seed : int
    channel_journals : tp.List[ChannelJournal]
    sample : Sample
    """

    def __init__(self,
                 created_at : datetime,
                 seed : int,
                 channel_journals : tp.List[ChannelJournal],
                 sample : SampleNode):
        self.created_at = created_at
        self.seed = seed
        self.channel_journals = channel_journals
        self.sample = sample

    @classmethod
    def from_dict(cls, d : tp.Dict):
        return cls(
            created_at=datetime.fromisoformat(d['created_at']),
            seed=d['seed'],
            channel_journals=[
                ChannelJournal.from_dict(j) for j in d['channel_journals']],
            sample=SampleNode.from_dict(d['sample'])
        )

    def to_dict(self):
        return {
            'created_at': self.created_at.isoformat(),
            'seed': self.seed,
            'channel_journals':[j.to_dict() for j in self.channel_journals],
            'sample': self.sample.to_dict(),
        }

class SampleSetJournal(object):
    """
    Journal class of sample set

    Attributes
    ----------
    process_start : timestamp
    process_finish : timestamp
    metadata_path : str
    spec : SampleSpecification
    seed : int
    sample_journals : tp.List[SampleJournal])
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
            'metadata_path': str(self.metadata_path),
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
    rel_path_prefixs = [
        os.path.join(f'{i//1000:03d}', f'{i%1000:03d}')
        for i in range(spec.sample_size)
    ]
    args = [(
        sample_i,
        datasources,
        spec.category_map,
        spec.sample_rate,
        spec.duration,
        spec.target_db,
        spec.metadata_path,
        os.path.join(spec.data_dir, rel_path_prefix),
        random_.randrange(2**32), # seed
        spec.device,
    ) for sample_i, rel_path_prefix in enumerate(rel_path_prefixs)]

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
            'category_map': spec.category_map,
            'sample_rate': spec.sample_rate,
            'duration': spec.duration,
            'seed': spec.seed,
            'jobs': spec.jobs,
        }))

    # sample
    metadata_list = [None for _ in range(len(args))]
    journal_list = [None for _ in range(len(args))]
    for sample_i, metadata, journal in map_fn(_save_single_sample, args):
        metadata_list[sample_i] = metadata
        journal_list[sample_i] = journal
        if logger:
            logger.info(json.dumps({
                'type': 'made_sample',
                'timestamp': datetime.now().isoformat(),
                'sample_i': sample_i,
                'channel_size': len(metadata.channels)
            }))

    # close map function
    if spec.jobs is not None:
        pool.close()

    process_finish = datetime.now()

    # save metadata
    metadata = EmptyNode(children=metadata_list)
    with open(spec.metadata_path, 'w') as fp:
        json.dump(metadata.to_dict(), fp)

    if logger:
        logger.info(json.dumps({
            'type': 'save_samples',
            'timestamp': datetime.now().isoformat(),
            'metadata_path': str(spec.metadata_path),
            'sample_size': len(metadata_list),
        }))

    if spec.journal_path:
        if not os.path.exists(os.path.dirname(spec.journal_path)):
            os.makedirs(os.path.dirname(spec.journal_path))
        journals = SampleSetJournal(
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
    category_map : dict,
    sample_rate : int,
    duration : float,
    target_db : float,
    metadata_path : str,
    data_path_prefix : str,
    rel_path : str,
    seed : int
    """
    sample_i, datasources, category_map, \
        sample_rate, duration, target_db, \
        metadata_path, data_path_prefix, seed, device = args
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

    # calculate source category list
    source_categories = dict()
    for ds in tracks[track] \
        + (tracks.get(None, []) if track is not None else []):
        if ds.category not in source_categories:
            source_categories[ds.category] = []
        source_categories[ds.category].append(ds)
    for ds_list in source_categories.values():
        random.shuffle(ds_list)
    source_category_index = dict((k, 0) for k in source_categories)

    # loop
    waves = []
    sheets = []
    datasources = []
    target_categories = []
    for target_category, category_config in category_map.items():

        s_categories = category_config['sources']
        max_category_num = category_config.get('max_samples') or 1

        source_num = 0
        # NOTE: Lagging could be occur if tracked and no-tracked waves
        # are in single category
        while source_num < max_category_num \
              and any(source_category_index[c] < len(source_categories[c])
                      for c in s_categories):

            wave = None
            sheet = None
            selected_ds = []
            while (wave is None or wave.shape[-1] < wavelength) \
                  and any(source_category_index[c] < len(source_categories[c])
                          for c in s_categories):
                # select source
                c_population = [
                    c for c in s_categories
                    if source_category_index[c] < len(source_categories[c])
                ]
                c_weights = [
                    s_categories[c].get('weight', 1.0) for c in c_population
                ]
                c = random.choices(c_population, c_weights, k=1)[0]
                ds_list = source_categories[c]
                ds = ds_list[source_category_index[c]]
                source_category_index[c] += 1

                # load wav and concat
                w, orig_sr = torchaudio.load(ds.wave_path)
                w = w.to(device)
                w = w.mean(dim=0)
                w = Resample(orig_sr, sample_rate)(w)
                if target_db is not None:
                    w = MultiPointScale(
                        scales=[10 ** (target_db / 20)],
                        fractions=[],
                        normalize=True
                    )(w)
                # if no activation is found, skip this datasource
                activation = [(0, w.shape[-1], [])]
                top_db = 60 if target_db is None else -target_db*10
                merge_activation(activation, w, 'tag', top_db=top_db)
                if all(not tags for _, _, tags in activation):
                    continue
                if ds.track is None:
                    # shrink silence
                    w = torch.cat([
                        w[..., s:t] if tags else
                        w[..., s:t] if t-s < sample_rate * 0.5 else
                        torch.cat((
                            w[..., s:s+int(sample_rate*0.5)//2],
                            w[..., t-int(sample_rate*0.5)//2:t],
                        ), dim=-1)
                        for s, t, tags in activation
                    ], dim=-1)
                selected_ds.append(ds)
                wave = w if wave is None else torch.cat((wave, w), dim=-1)
                # load midi
                if ds.sheet_path:
                    sheet = None
                    raise NotImplementedError()

            # append
            if selected_ds:
                waves.append(wave)
                sheets.append(sheet)
                datasources.append(selected_ds)
                target_categories.append(target_category)
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
        top_db = 60 if target_db is None else -target_db*10
        merge_activation(a, w, i, top_db=top_db)

    # choose activation
    track_activation = dict()
    for k, a in track_activations.items():
        weights = [
            (end-start)*(10**(len(tags)-1)) if len(tags) else 0
            for start, end, tags in a
        ]
        if len(weights) == 0 or sum(weights) <= 0:
            track_activation[k] = None
        else:
            track_activation[k] =\
                random.choices(a, weights=weights, k=1)[0]

    source_activation = []
    for i, (ds, a) \
        in enumerate(zip(datasources, source_activations)):
        if i not in set(tag for _, _, tags in a for tag in tags):
            source_activation.append(None)
        elif ds[0].track in track_activations:
            source_activation.append(
                track_activation[ds[0].track]
                if i in track_activation[ds[0].track][2] else None
            )
        else:
            weights = [
                end - start if len(tags) > 0 else 0
                for start, end, tags in a
            ]
            if len(weights) == 0 or sum(weights) <= 0:
                source_activation.append(None)
            else:
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
    dirname = os.path.dirname(data_path_prefix)
    if not os.path.isdir(dirname):
        os.makedirs(dirname, exist_ok=True)
    for c_i, (crop_wave, crop_sheet) in \
        enumerate(zip(crop_waves, crop_sheets)):
        torch.save({
            'wave': crop_wave.to('cpu'),
            'sheet': crop_sheet,
        }, str(data_path_prefix)+f'.{c_i:02d}.pth')

    # create metadata
    channel_metadata_list = []
    channel_journal_list = []
    for c_i, (dss, offset) in enumerate(zip(crop_datasources, crop_offsets)):
        channel_metadata = Channel(
            path=os.path.relpath(
                str(data_path_prefix)+f'.{c_i:02d}.pth',
                os.path.dirname(metadata_path)
            ),
            category=target_categories[c_i],
            track=dss[0].track,
            fold=dss[0].fold,
            sample_rate=sample_rate,
        )
        channel_journal = ChannelJournal(
            created_at=datetime.now(),
            seed=seed,
            datasources=dss,
            length=wavelength,
            offset=offset,
            channel=channel_metadata,
        )
        channel_metadata_list.append(channel_metadata)
        channel_journal_list.append(channel_journal)

    sample_metadata = SampleNode(
        channels=channel_metadata_list,
        rel_path=os.path.dirname(metadata_path),
    )

    sample_journal = SampleJournal(
        created_at=datetime.now(),
        seed=seed,
        channel_journals=channel_journal_list,
        sample=sample_metadata,
    )

    return sample_i, sample_metadata, sample_journal

