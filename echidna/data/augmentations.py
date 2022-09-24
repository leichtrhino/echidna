
import typing as tp
import os
import logging
import json
import random
import multiprocessing
from datetime import datetime

import torch

from .utils import merge_activation
from .transforms import build_transform, Crop
from .datanodes import DataNode, add_datanode_cls

class ChannelAugmentation(object):
    def __init__(self,
                 offset : int,
                 time_stretch_rate : float,
                 pitch_shift_rate : float,
                 scale_amounts : tp.List[float],
                 scale_fractions : tp.List[float],
                 ):
        self.sample_augmentation = None
        self.offset = offset
        self.time_stretch_rate = time_stretch_rate
        self.pitch_shift_rate = pitch_shift_rate
        self.scale_amounts = scale_amounts
        self.scale_fractions = scale_fractions

    def process_partial(self, data):
        transform = build_transform(
            normalize=False,
            source_sample_rate=self.sample_augmentation.source_sample_rate,
            target_sample_rate=self.sample_augmentation.target_sample_rate,
            time_stretch_rate=self.time_stretch_rate,
            pitch_shift_rate=self.pitch_shift_rate,
            scale_amounts=self.scale_amounts,
            scale_fractions=self.scale_fractions,
            offset=self.offset,
            waveform_length=self.sample_augmentation.waveform_length,
            n_fft=self.sample_augmentation.n_fft,
            hop_length=self.sample_augmentation.hop_length,
            win_length=self.sample_augmentation.win_length,
        ).to(data['wave'].device)
        wave = transform(data['wave'])
        data = {
            'wave': wave,
            'sheet': None,
        }
        return data

    @classmethod
    def from_dict(cls, d : dict):
        return cls(
            offset=d.get('offset', 0),
            time_stretch_rate=d.get('time_stretch_rate', 1.),
            pitch_shift_rate=d.get('pitch_shift_rate', 1.),
            scale_amounts=d.get('scale_amounts', [1.0]),
            scale_fractions=d.get('scale_fractions', []),
        )

    def to_dict(self):
        return {
            'offset': self.offset,
            'time_stretch_rate': self.time_stretch_rate,
            'pitch_shift_rate': self.pitch_shift_rate,
            'scale_amounts': self.scale_amounts,
            'scale_fractions': self.scale_fractions,
        }

class AugmentationNode(DataNode):
    def __init__(self,
                 source_sample_rate : int,
                 target_sample_rate : int,
                 waveform_length : int,
                 channel_augmentations,
                 n_fft : int=2048,
                 hop_length : int=512,
                 win_length : int=2048,
                 children=None,
                 metrics=None,
                 ):
        super().__init__(children, metrics)

        self.source_sample_rate = source_sample_rate
        self.target_sample_rate = target_sample_rate
        self.waveform_length = waveform_length

        self.channel_augmentations = channel_augmentations
        for a in self.channel_augmentations:
            a.sample_augmentation = self

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

    @property
    def super_type(self):
        return 'filter'

    def require_channel_index_partial(self):
        return None

    def process_partial(self, data):
        if len(data) != len(self.channel_augmentations):
            raise ValueError('length of input and augmentation differ')
        post_data = [
            a.process_partial(d) if d is not None and a is not None
            else None
            for a, d in zip(self.channel_augmentations, data)
        ]
        return post_data

    @classmethod
    def from_dict_args(cls, d : dict, c : dict=None):
        return cls(
            source_sample_rate=d['source_sample_rate'],
            target_sample_rate=d['target_sample_rate'],
            waveform_length=d['waveform_length'],
            channel_augmentations=[
                ChannelAugmentation.from_dict(a)
                for a in d['channel_augmentations']
            ],
            n_fft=d.get('n_fft', 2048),
            hop_length=d.get('hop_length', 512),
            win_length=d.get('win_length', 2048),
        )

    def to_dict_args(self):
        return {
            'source_sample_rate': self.source_sample_rate,
            'target_sample_rate': self.target_sample_rate,
            'waveform_length': self.waveform_length,
            'channel_augmentations': [
                a.to_dict()
                for a in self.channel_augmentations
            ],
            'n_fft': self.n_fft,
            'hop_length': self.hop_length,
            'win_length': self.win_length,
        }

add_datanode_cls('augmentation', AugmentationNode)

class ChannelAugmentationJournal(object):
    def __init__(self,
                 created_at : datetime,
                 channel_augmentation : ChannelAugmentation):
        self.created_at = created_at
        self.channel_augmentation = channel_augmentation

    @classmethod
    def from_dict(cls, d : tp.Dict):
        return cls(
            created_at=datetime.fromisoformat(d['created_at']),
            channel_augmentation=ChannelAugmentation.from_dict(
                d['channel_augmentation']),
        )

    def to_dict(self):
        return {
            'created_at': self.created_at.isoformat(),
            'channel_augmentation': self.channel_augmentation.to_dict(),
        }

class SampleAugmentationJournal(object):
    def __init__(self,
                 created_at : datetime,
                 seed : int,
                 channel_journals : tp.List[ChannelAugmentationJournal],
                 sample_augmentation : AugmentationNode,
                 ):
        self.created_at = created_at
        self.seed = seed
        self.channel_journals = channel_journals
        self.sample_augmentation = sample_augmentation

    @classmethod
    def from_dict(cls, d : tp.Dict):
        return cls(
            created_at=datetime.fromisoformat(d['created_at']),
            seed=d['seed'],
            channel_journals=[ChannelAugmentationJournal.from_dict(j)
                for j in d['channel_journals']],
            sample_augmentation=AugmentationNode.from_dict(
                d['sample_augmentation']),
        )

    def to_dict(self):
        return {
            'created_at': self.created_at.isoformat(),
            'seed': self.seed,
            'channel_journals': [
                j.to_dict() for j in self.channel_journals],
            'sample_augmentation': self.sample_augmentation.to_dict(),
        }

class AugmentationNodeJournal(object):
    def __init__(self,
                 created_at : datetime,
                 seed : int,
                 sample_journals : tp.List[SampleAugmentationJournal],
                 augmentations : tp.List[AugmentationNode],
                 ):
        self.created_at = created_at
        self.seed = seed
        self.sample_journals = sample_journals
        self.augmentations = augmentations

    @classmethod
    def from_dict(cls, d : tp.Dict):
        return cls(
            created_at=datetime.fromisoformat(d['created_at']),
            seed=d['seed'],
            sample_journals=[SampleAugmentationJournal.from_dict(j)
                for j in d['sample_journals']],
            augmentations=[AugmentationNode.from_dict(n)
                for n in d['augmentations']],
        )

    def to_dict(self):
        return {
            'created_at': self.created_at.isoformat(),
            'seed': self.seed,
            'sample_journals': [
                j.to_dict() for j in self.sample_journals],
            'augmentations': [
                n.to_dict_nochildren()
                for n in self.augmentations
            ],
        }

class AugmentationSetJournal(object):
    def __init__(self,
                 process_start : datetime,
                 process_finish : datetime,
                 metadata_path : str,
                 log_path : str,
                 spec,
                 node_journals : tp.List[AugmentationNodeJournal],
                 ):
        self.process_start = process_start
        self.process_finish = process_finish
        self.metadata_path = metadata_path
        self.log_path = log_path
        self.spec = spec
        self.node_journals = node_journals

    @classmethod
    def from_dict(cls, d : tp.Dict):
        return cls(
            process_start=datetime.fromisoformat(d['process_start']),
            process_finish=datetime.fromisoformat(d['process_finish']),
            metadata_path=d['metadata_path'],
            log_path=d.get('log_path'),
            spec=AugmentationSpec.from_dict(d['spec']),
            node_journals=[
                AugmentationNodeJournal.from_dict(j)
                for j in d['node_journals']
            ]
        )

    def to_dict(self):
        return {
            'process_start': self.process_start.isoformat(),
            'process_finish': self.process_finish.isoformat(),
            'metadata_path': str(self.metadata_path)
            if self.metadata_path else None,
            'log_path': str(self.log_path)
            if self.log_path else None,
            'spec': self.spec.to_dict(),
            'node_journals': [
                j.to_dict() for j in self.node_journals]
        }

class AugmentationSpec(object):
    def __init__(self,
                 # from random augmentation parameter
                 source_sample_rate : int,
                 target_sample_rate : int,
                 waveform_length : int,

                 scale_range : tp.Tuple[int],
                 scale_point_range : tp.Tuple[int],
                 time_stretch_range : tp.Tuple[int],
                 pitch_shift_range : tp.Tuple[int],

                 # from general parameter
                 input_metadata_path : str,
                 output_metadata_path : str,
                 seed : int,
                 augmentation_per_parent : int,
                 journal_path : str,
                 log_path : str,
                 log_level : str,
                 jobs : int=None,
                 device : str='cpu',

                 n_fft : int=2048,
                 hop_length : int=512,
                 win_length : int=2048,
                 ):

        self.source_sample_rate = source_sample_rate
        self.target_sample_rate = target_sample_rate
        self.waveform_length = waveform_length

        self.scale_range = scale_range
        self.scale_point_range = scale_point_range
        self.time_stretch_range = time_stretch_range
        self.pitch_shift_range = pitch_shift_range

        self.input_metadata_path = input_metadata_path
        self.output_metadata_path = output_metadata_path
        self.seed = seed
        self.augmentation_per_parent = augmentation_per_parent
        self.journal_path = journal_path
        self.log_path = log_path
        self.log_level = log_level
        self.jobs = jobs
        self.device = device

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

    @classmethod
    def from_dict(cls, d : tp.Dict):
        return cls(
            source_sample_rate=d['source_sample_rate'],
            target_sample_rate=d['target_sample_rate'],
            waveform_length=d['waveform_length'],

            scale_range=d['scale_range'],
            scale_point_range=d['scale_point_range'],
            time_stretch_range=d['time_stretch_range'],
            pitch_shift_range=d['pitch_shift_range'],

            n_fft=d.get('n_fft', 2048),
            hop_length=d.get('hop_length', 512),
            win_length=d.get('win_length', 2048),

            input_metadata_path=d['input_metadata_path'],
            output_metadata_path=d['output_metadata_path'],
            seed=d['seed'],
            augmentation_per_parent=d['augmentation_per_parent'],
            journal_path=d['journal_path'],
            log_path=d.get('log_path'),
            log_level=d.get('log_level'),
            jobs=d.get('jobs', None),
            device=d.get('device') or 'cpu',
        )

    def to_dict(self):
        return {
            'source_sample_rate': self.source_sample_rate,
            'target_sample_rate': self.target_sample_rate,
            'waveform_length': self.waveform_length,

            'scale_range': self.scale_range,
            'scale_point_range': self.scale_point_range,
            'time_stretch_range': self.time_stretch_range,
            'pitch_shift_range': self.pitch_shift_range,

            'n_fft': self.n_fft,
            'hop_length': self.hop_length,
            'win_length': self.win_length,

            'input_metadata_path': str(self.input_metadata_path)
            if self.input_metadata_path else None,
            'output_metadata_path': str(self.output_metadata_path)
            if self.output_metadata_path else None,
            'seed': self.seed,
            'augmentation_per_parent': self.augmentation_per_parent,
            'journal_path': str(self.journal_path)
            if self.journal_path else None,
            'log_path': str(self.log_path)
            if self.log_path else None,
            'log_level': self.log_level,
            'jobs': self.jobs,
            'device': self.device,
        }

    def save_augmentation(self):
        _save_augmentation(self)

def _save_augmentation(spec : AugmentationSpec):
    """
    """

    process_start = datetime.now()

    # setup seed
    random_ = random.Random(spec.seed)

    # prepare log
    logger = None
    if spec.log_path:
        if not os.path.exists(os.path.dirname(spec.log_path)):
            os.makedirs(os.path.dirname(spec.log_path), exist_ok=True)
        logger = logging.getLogger(__name__)
        logger.setLevel(spec.log_level)
        handler = logging.FileHandler(str(spec.log_path))
        handler.setFormatter(
            logging.Formatter('[%(levelname)s] %(message)s'))
        logger.addHandler(handler)

    # load metadata
    with open(spec.input_metadata_path, 'r') as fp:
        datanode = DataNode.from_dict(
            json.load(fp),
            context={
                'rel_path': os.path.dirname(spec.input_metadata_path),
                'device': spec.device,
            },
        )
    chain_lists = [
        datanode.get_single_chain(i) for i in range(len(datanode))
    ]

    # prepare arguments
    args = [(
        sample_i,
        chain_list,
        {
            'source_sample_rate': spec.source_sample_rate,
            'target_sample_rate': spec.target_sample_rate,
            'waveform_length': spec.waveform_length,

            'scale_range': spec.scale_range,
            'scale_point_range': spec.scale_point_range,
            'time_stretch_range': spec.time_stretch_range,
            'pitch_shift_range': spec.pitch_shift_range,

            'n_fft': spec.n_fft,
            'hop_length': spec.hop_length,
            'win_length': spec.win_length,
        }, # augmentation_spec_params,
        spec.augmentation_per_parent,
        random_.randrange(2**32), #seed,
    ) for sample_i, chain_list in enumerate(chain_lists)]

    # map func
    if spec.jobs is not None:
        pool = multiprocessing.Pool(spec.jobs)
        map_fn = pool.imap_unordered
    else:
        map_fn = map

    if logger:
        logger.info(json.dumps({
            'type': 'start_augmentation',
            'timestamp': datetime.now().isoformat(),
            'sample_size': len(chain_lists),
            'augmentation_per_parent': spec.augmentation_per_parent,
            'seed': spec.seed,
            'jobs': spec.jobs,
        }))

    # iterate over dataset and find mixtures
    augmentation_list = [None for _ in range(len(chain_lists))]
    journal_list = [None for _ in range(len(chain_lists))]
    for i, augmentations, journals in map_fn(
            _make_augmentations_for_sample, args):
        augmentation_list[i] = augmentations
        journal_list[i] = journals
        if logger:
            for augmentation in augmentations:
                logger.info(json.dumps({
                    'type': 'made_augmentation',
                    'timestamp': datetime.now().isoformat(),
                    'sample_index': i,
                    'augmentation_size': len(augmentation or []),
                }))

    # close map function
    if spec.jobs is not None:
        pool.close()

    # append all mixtures to leaf node
    leaf_nodes = datanode.list_leaf_node()
    assert len(leaf_nodes) == len(augmentation_list)
    for n, m in zip(leaf_nodes, augmentation_list):
        if not m:
            continue
        n.children = m
        for _m in m:
            _m.parent = n
    datanode.balance_by_remove()

    process_finish = datetime.now()

    # save metadata
    if not os.path.exists(os.path.dirname(spec.output_metadata_path)):
        os.makedirs(os.path.dirname(spec.output_metadata_path))
    with open(spec.output_metadata_path, 'w') as fp:
        json.dump(datanode.to_dict(), fp)

    if logger:
        logger.info(json.dumps({
            'type': 'save_augmentations',
            'timestamp': datetime.now().isoformat(),
            'metadata_path': str(spec.output_metadata_path),
            'augmentation_size': len(datanode),
        }))

    # save journal
    if spec.journal_path is not None:
        if not os.path.exists(os.path.dirname(spec.journal_path)):
            os.makedirs(os.path.dirname(spec.journal_path))
        augmentations_journal = AugmentationSetJournal(
            process_start=process_start,
            process_finish=process_finish,
            metadata_path=os.path.relpath(
                spec.output_metadata_path,
                os.path.dirname(spec.journal_path)
            ),
            log_path=os.path.relpath(
                spec.log_path,
                os.path.dirname(spec.log_path)
            ) if spec.log_path else None,
            spec=spec,
            node_journals=journal_list,
        )
        with open(spec.journal_path, 'w') as fp:
            json.dump(augmentations_journal.to_dict(), fp)

        if logger:
            logger.info(json.dumps({
                'type': 'save_augmentations_journal',
                'timestamp': datetime.now().isoformat(),
                'journal_path': str(spec.journal_path),
            }))

    # finish epoch and close log handler
    if logger:
        logger.info(json.dumps({
            'type': 'finish_augmentation',
            'timestamp': datetime.now().isoformat(),
        }))
        handlers = logger.handlers[:]
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()

def _make_augmentations_for_sample(args):
    sample_i, node_list, augmentation_spec_params, \
        augmentation_per_sample, seed = args

    '''
        sample_i,
        chain_list,
        {
            'source_sample_rate': spec.source_sample_rate,
            'target_sample_rate': spec.target_sample_rate,
            'waveform_length': spec.waveform_length,

            'scale_range': spec.scale_range,
            'scale_point_range': spec.scale_point_range,
            'time_stretch_range': spec.time_stretch_range,
            'pitch_shift_range': spec.pitch_shift_range,

            'n_fft': spec.n_fft,
            'hop_length': spec.hop_length,
            'win_length': spec.win_length,
        }, # augmentation_spec_params,
        spec.augmentation_per_sample,
        random_.randrange(2**32), #seed,
    '''

    random_ = random.Random(seed)

    data, metadata = DataNode.process_single_chain(node_list)
    if metadata[0].super_type != 'source':
        logger.error(json.dumps({
            'type': 'InvalidNodeType',
            'messge': 'the first element of chain is not source',
            'sample_i': sample_i,
            'seed': seed,
            'metadata': [md.to_dict() for md in metadata]
        }))

    augmentations = []
    journals = []
    for augmentation_node, journal in [
            _make_param_set(
                data,
                source_metadata=metadata[0],
                seed=random_.randrange(2**32),
                **augmentation_spec_params,
            )
            for _ in range(augmentation_per_sample)
    ]:
        augmentations.append(augmentation_node)
        journals.append(journal)

    return sample_i, augmentations, AugmentationNodeJournal(
        created_at=datetime.now(),
        seed=seed,
        sample_journals=journals,
        augmentations=augmentations,
    )

def _make_param_set(data,
                    source_metadata,
                    seed : int,

                    source_sample_rate : int,
                    target_sample_rate : int,
                    waveform_length : int,

                    scale_range : tp.Tuple[int],
                    scale_point_range : tp.Tuple[int],
                    time_stretch_range : tp.Tuple[int],
                    pitch_shift_range : tp.Tuple[int],

                    n_fft : int=2048,
                    hop_length : int=512,
                    win_length : int=2048,
                    ) -> tp.List[tp.Dict[str, object]]:

    assert len(scale_range) == 2
    assert len(scale_point_range) == 2
    assert len(time_stretch_range) == 2
    assert len(pitch_shift_range) == 2

    random.seed(seed)
    torch.manual_seed(seed)

    # make time-dependent parameter for each track
    track_param = dict()
    track_activation = dict()
    for track in set(
            c.track for c in source_metadata.channels if c.track is not None):
        track_param[track] = {
            'time_stretch_rate': random.uniform(*time_stretch_range),
        }
        track_activation[track] = [(0, None, [])]

    # make transform parameter without offset
    channel_params = []
    channel_activations = []
    transformed_waves = []
    common_params = {
        'source_sample_rate': source_sample_rate,
        'target_sample_rate': target_sample_rate,
        'waveform_length': None, # this param is overwritten at the end
        'n_fft': n_fft,
        'hop_length': hop_length,
        'win_length': win_length,
    }

    for channel_i, (_data, _channel_metadata) in enumerate(
            zip(data, source_metadata.channels)):
        wave = _data['wave']
        track = _channel_metadata.track

        # build transform
        time_stretch_rate = track_param.get(
            track,
            {'time_stretch_rate' : random.uniform(*time_stretch_range)}
        )['time_stretch_rate']
        pitch_shift_rate = random.uniform(*pitch_shift_range)
        scale_points = random.randint(*scale_point_range)
        scale_amounts = [
            random.uniform(*scale_range) for _ in range(scale_points)
        ]
        if random.random() < 0.5 \
           and len(scale_amounts) > 1 and scale_amounts[0] != 1:
            scale_amounts = [-s for s in scale_amounts]
        scale_fractions = [
            random.random() for _ in range(scale_points-1)
        ]

        params = {
            'time_stretch_rate': time_stretch_rate,
            'pitch_shift_rate': pitch_shift_rate,
            'scale_amounts': scale_amounts,
            'scale_fractions': scale_fractions,
            'offset': None,
        }

        tf = build_transform(
            **common_params, **params, normalize=True).to(wave.device)
        transformed_wave = tf(wave)
        transformed_waves.append(transformed_wave)

        # find activatio
        activation = track_activation.get(
            track, [(0, transformed_wave.shape[-1], [])]
        )
        merge_activation(activation, transformed_wave, channel_i)
        channel_params.append(params)
        channel_activations.append(activation)

    # get offset
    result_waves = []
    for track, wave, param, activations in \
        zip([c.track for c in source_metadata.channels],
            transformed_waves, channel_params, channel_activations):
        if 'offset' in track_param.get(track, dict()):
            offset = track_param[track]['offset']
        else: # calculate offset from activation
            weights = [
                (end-start)*(10**(len(tags)-1)) if len(tags) else 0
                for start, end, tags in activations
            ]
            activation = random.choices(activations,
                                        weights=weights, k=1)[0]

            start, end, _ = activation
            start = max(0, start - waveform_length // 4)
            end = min(wave.shape[-1], end + waveform_length // 4)
            end = max(start, end - waveform_length)
            offset = random.randint(start, end)

            if track in track_param: # assign offset to track
                track_param[track]['offset'] = offset

        param['offset'] = offset
        result_waves.append(Crop(waveform_length, offset).to(wave.device)(wave))

    channel_augmentations = [ChannelAugmentation(**p) for p in channel_params]
    sample_augmentation = AugmentationNode(
        source_sample_rate=source_sample_rate,
        target_sample_rate=target_sample_rate,
        waveform_length=waveform_length,
        channel_augmentations=channel_augmentations,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
    )
    channel_journals = [ChannelAugmentationJournal(
        created_at=datetime.now(),
        channel_augmentation=c
    ) for c in channel_augmentations]
    sample_journal = SampleAugmentationJournal(
        created_at=datetime.now(),
        seed=seed,
        channel_journals=channel_journals,
        sample_augmentation=sample_augmentation
    )

    return sample_augmentation, sample_journal
