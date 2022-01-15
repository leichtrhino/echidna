
import os
import pathlib
import math
import random
import torch
import torchaudio

from echidna.data import samples

def prepare_datasources(tmpdir, seed):
    datasource_dir = pathlib.Path(tmpdir.name) / 'datasources'
    sample_dir = pathlib.Path(tmpdir.name) / 'samples'
    os.mkdir(datasource_dir)
    os.mkdir(sample_dir)

    # create waveform
    torchaudio.save(datasource_dir / 'ds001.wav',
                    torch.linspace(0, 2*math.pi, 8000*2)[None, :],
                    sample_rate=8000)
    torchaudio.save(datasource_dir / 'ds002.wav',
                    torch.linspace(0, 2*math.pi, 8000*2)[None, :],
                    sample_rate=8000)
    torchaudio.save(datasource_dir / 'ds003.wav',
                    torch.linspace(0, 2*math.pi, 8000)[None, :],
                    sample_rate=8000)
    torchaudio.save(datasource_dir / 'ds004.wav',
                    torch.linspace(0, 2*math.pi, 8000)[None, :],
                    sample_rate=8000)

    # datasources a: without tracks, folds
    datasources_a = [
        # category 1 (full length)
        samples.Datasource(id='ds001',
                           wave_path=datasource_dir / 'ds001.wav',
                           sheet_path=None,
                           category='ct001',
                           track=None,
                           fold=None),
        samples.Datasource(id='ds002',
                           wave_path=datasource_dir / 'ds002.wav',
                           sheet_path=None,
                           category='ct001',
                           track=None,
                           fold=None),
        # category 2 (short length)
        samples.Datasource(id='ds003',
                           wave_path=datasource_dir / 'ds003.wav',
                           sheet_path=None,
                           category='ct002',
                           track=None,
                           fold=None),
        samples.Datasource(id='ds004',
                           wave_path=datasource_dir / 'ds004.wav',
                           sheet_path=None,
                           category='ct002',
                           track=None,
                           fold=None),
    ]

    samplespec_a_1 = samples.SampleSpec(
        datasources=datasources_a,
        fold=None,
        sample_size=1,
        source_per_category=1,
        source_by_category=None,
        sample_rate=8000,
        duration=1.5,
        seed=seed,
        metadata_path=sample_dir/'a1'/'metadata.json',
        data_dir=sample_dir/'a1',
        journal_path=sample_dir/'a1'/'journal.json',
        log_path=sample_dir/'a1'/'log.txt',
        log_level='DEBUG',
    )
    samplespec_a_2 = samples.SampleSpec(
        datasources=datasources_a,
        fold=None,
        sample_size=1,
        source_per_category=2,
        source_by_category=None,
        sample_rate=8000,
        duration=1.5,
        seed=seed,
        metadata_path=sample_dir/'a2'/'metadata.json',
        data_dir=sample_dir/'a2',
        journal_path=sample_dir/'a2'/'journal.json',
        log_path=sample_dir/'a2'/'log.txt',
        log_level='DEBUG',
    )
    samplespec_a_3 = samples.SampleSpec(
        datasources=datasources_a,
        fold=None,
        sample_size=1,
        source_per_category=1,
        source_by_category={'ct001': 2},
        sample_rate=8000,
        duration=1.5,
        seed=seed,
        metadata_path=sample_dir/'a3'/'metadata.json',
        data_dir=sample_dir/'a3',
        journal_path=sample_dir/'a3'/'journal.json',
        log_path=sample_dir/'a3'/'log.txt',
        log_level='DEBUG',
    )

    # datasources b: with folds
    datasources_b = [
        samples.Datasource(id='ds001',
                           wave_path=datasource_dir / 'ds001.wav',
                           sheet_path=None,
                           category='ct001',
                           track=None,
                           fold='fl001'),
        samples.Datasource(id='ds002',
                           wave_path=datasource_dir / 'ds002.wav',
                           sheet_path=None,
                           category='ct001',
                           track=None,
                           fold='fl002'),
    ]

    samplespec_b_1 = samples.SampleSpec(
        datasources=datasources_b,
        fold='fl001',
        sample_size=1,
        source_per_category=1,
        source_by_category=None,
        sample_rate=8000,
        duration=1.5,
        seed=seed,
        metadata_path=sample_dir/'b1'/'metadata.json',
        data_dir=sample_dir/'b1',
        journal_path=sample_dir/'b1'/'journal.json',
        log_path=sample_dir/'b1'/'log.txt',
        log_level='DEBUG',
    )

    # datasources c: with tracks
    datasources_c = [
        # category 1 (full length)
        samples.Datasource(id='ds001',
                           wave_path=datasource_dir / 'ds001.wav',
                           sheet_path=None,
                           category='ct001',
                           track='tk001',
                           fold=None),
        samples.Datasource(id='ds002',
                           wave_path=datasource_dir / 'ds002.wav',
                           sheet_path=None,
                           category='ct001',
                           track='tk002',
                           fold=None),
        # category 2 (short length)
        samples.Datasource(id='ds003',
                           wave_path=datasource_dir / 'ds003.wav',
                           sheet_path=None,
                           category='ct002',
                           track=None,
                           fold=None),
        samples.Datasource(id='ds004',
                           wave_path=datasource_dir / 'ds004.wav',
                           sheet_path=None,
                           category='ct002',
                           track=None,
                           fold=None),
    ]

    samplespec_c_2 = samples.SampleSpec(
        datasources=datasources_c,
        fold=None,
        sample_size=1,
        source_per_category=2,
        source_by_category=None,
        sample_rate=8000,
        duration=1.5,
        seed=seed,
        metadata_path=sample_dir/'c2'/'metadata.json',
        data_dir=sample_dir/'c2',
        journal_path=sample_dir/'c2'/'journal.json',
        log_path=sample_dir/'c2'/'log.txt',
        log_level='DEBUG',
    )

    # datasources d: without tracks, folds
    datasources_d = [
        # category 1 (full length)
        samples.Datasource(id='ds001',
                           wave_path=datasource_dir / 'ds001.wav',
                           sheet_path=None,
                           category='ct001',
                           track=None,
                           fold=None),
        # category 2 (full length)
        samples.Datasource(id='ds002',
                           wave_path=datasource_dir / 'ds002.wav',
                           sheet_path=None,
                           category='ct002',
                           track=None,
                           fold=None),
        # category 3 (short length)
        samples.Datasource(id='ds003',
                           wave_path=datasource_dir / 'ds003.wav',
                           sheet_path=None,
                           category='ct003',
                           track=None,
                           fold=None),
        samples.Datasource(id='ds004',
                           wave_path=datasource_dir / 'ds004.wav',
                           sheet_path=None,
                           category='ct003',
                           track=None,
                           fold=None),
    ]

    samplespec_d_1 = samples.SampleSpec(
        datasources=datasources_d,
        fold=None,
        sample_size=1,
        source_per_category=1,
        source_by_category=None,
        sample_rate=8000,
        duration=1.5,
        seed=seed,
        metadata_path=sample_dir/'d1'/'metadata.json',
        data_dir=sample_dir/'d1',
        journal_path=sample_dir/'d1'/'journal.json',
        log_path=sample_dir/'d1'/'log.txt',
        log_level='DEBUG',
    )

    # datasources e: with tracks
    datasources_e = [
        # category 1 (full length)
        samples.Datasource(id='ds001',
                           wave_path=datasource_dir / 'ds001.wav',
                           sheet_path=None,
                           category='ct001',
                           track='tk001',
                           fold=None),
        # category 2 (full length)
        samples.Datasource(id='ds002',
                           wave_path=datasource_dir / 'ds002.wav',
                           sheet_path=None,
                           category='ct002',
                           track='tk001',
                           fold=None),
        # category 3 (short length)
        samples.Datasource(id='ds003',
                           wave_path=datasource_dir / 'ds003.wav',
                           sheet_path=None,
                           category='ct003',
                           track=None,
                           fold=None),
        samples.Datasource(id='ds004',
                           wave_path=datasource_dir / 'ds004.wav',
                           sheet_path=None,
                           category='ct003',
                           track=None,
                           fold=None),
    ]

    samplespec_e_1 = samples.SampleSpec(
        datasources=datasources_e,
        fold=None,
        sample_size=1,
        source_per_category=2,
        source_by_category=None,
        sample_rate=8000,
        duration=1.5,
        seed=seed,
        metadata_path=sample_dir/'e1'/'metadata.json',
        data_dir=sample_dir/'e1',
        journal_path=sample_dir/'e1'/'journal.json',
        log_path=sample_dir/'e1'/'log.txt',
        log_level='DEBUG',
    )

    return {
        'A1': samplespec_a_1,
        'A2': samplespec_a_2,
        'A3': samplespec_a_3,
        'B1': samplespec_b_1,
        'C2': samplespec_c_2,
        'D1': samplespec_d_1,
        'E1': samplespec_e_1,
    }

class ToyDataset(torch.utils.data.Dataset):
    def __init__(self, sample_size, val=False, seed=None):
        self.sample_size = sample_size
        self.factor = -1 if val else 1
        random.seed(seed)
        self.seed_list = [random.randrange(2**32) for _ in range(len(self))]

    def __len__(self):
        return self.sample_size

    def __getitem__(self, idx):
        torch.manual_seed(self.seed_list[idx])
        data = {
            'waves': torch.cat(
                (torch.full((3, 1), self.factor * idx),
                 torch.rand(3, 3999)),
                dim=-1
            ),
            'sheets': None
        }
        metadata = {
            'index': idx,
            'sample': f'sample{idx}',
            'augmentation': f'augmentation{idx}',
            'mixture': f'mixture{idx}',
        }
        return data, metadata

    def to_dict(self):
        return {
            'sample_size': self.sample_size,
            'factor': self.factor,
            'seed_list': self.seed_list,
        }

    @classmethod
    def from_dict(cls, d : dict):
        pass


