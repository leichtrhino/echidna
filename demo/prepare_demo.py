
from pathlib import Path
import itertools
import torch
import torchaudio

# prepare directories
root = Path(__file__).parent
# config directory
config_dir = root / 'config'
sample_dir = config_dir / 'samples'
mixtures_dir = config_dir / 'mixtures'
augmentations_dir = config_dir / 'augmentations'
trainings_dir = config_dir / 'trainings'
validations_dir = config_dir / 'validations'
separations_dir = config_dir / 'separations'
clusterings_dir = config_dir / 'clusterings'

for d in (config_dir,
          sample_dir,
          mixtures_dir,
          augmentations_dir,
          trainings_dir,
          validations_dir,
          separations_dir,
          clusterings_dir,
          ):
    if not d.exists():
        d.mkdir()

# data directory
data_dir = root / 'data'
datasources_dir = data_dir / 'datasources'
for d in (data_dir, datasources_dir):
    if not d.exists():
        d.mkdir()

########################################
# prepare wave of datasource
########################################
for (duration, duration_name), hz in itertools.product(
        [(1.01, 'long'), (0.51, 'short')],
        [220, 440, 880],
):
    channel = 1 if duration_name == 'short' else 2
    x = 0.8 * torch.sin(
        torch.linspace(0, hz*2*torch.pi*duration, int(duration*24000)))
    x = torch.stack([x]*channel, dim=0)
    torchaudio.save(datasources_dir/f'sin{hz}{duration_name}.wav', x, 24000)

########################################
# prepare csv of datasource
########################################
# notrack
with open(datasources_dir / 'datasource_notrack.csv', 'w') as fp:
    fp.write('id,wave_path,sheet_path,category,track,fold\n')
    for duration_name, hz in [('long', 220), ('short', 440), ('short', 880)]:
        category = 'c001' if hz == 220 else 'c002' if hz == 440 else 'c003'
        wave_path = datasources_dir / f'sin{hz}{duration_name}.wav'
        for fold in ['01', '02', '03']:
            for si in range(1, 6):
                id = 100 * (1 if hz == 220 else 2 if hz == 440 else 3) \
                    + 10 * int(fold) \
                    + si
                track = ''
                row = f'{id},{wave_path},,{category},{track},{fold}\n'
                fp.write(row)

# track
with open(datasources_dir / 'datasource_track.csv', 'w') as fp:
    fp.write('id,wave_path,sheet_path,category,track,fold\n')
    for duration_name, hz in [('long', 220), ('long', 440), ('short', 880)]:
        category = 'c001' if hz == 220 else 'c002' if hz == 440 else 'c003'
        wave_path = datasources_dir / f'sin{hz}{duration_name}.wav'
        for fold in ['01', '02', '03']:
            for si in range(1, 6):
                id = 100 * (1 if hz == 220 else 2 if hz == 440 else 3) \
                    + 10 * int(fold) \
                    + si
                track = f't{fold}{si}' if duration_name == 'long' else ''
                row = f'{id},{wave_path},,{category},{track},{fold}\n'
                fp.write(row)

########################################
# prepare yaml for samples
########################################
yaml_template = '''# sample config file for data samples
datasources: {datasources}
fold:
{folds}
sample_size: 16
source_per_category: 2
sample_rate: 48000
duration: 1.0
target_db: {target_db}
seed: 01410343
metadata_path: {metadata_path}
data_dir: {data_dir}
journal_path: {journal_path}
log_path: {log_path}
log_level: INFO
jobs: 4
'''

samples_conf_dir = config_dir / 'samples'
samples_dir = data_dir / 'samples'
datasources_dir = data_dir / 'datasources'
for samples_name, params in [
        ('notrack_training', {
            'datasources': datasources_dir / 'datasource_notrack.csv',
            'folds': '  - \'01\'\n  - \'02\'',
            'target_db': '-0.5',
        }),
        ('notrack_validation', {
            'datasources': datasources_dir / 'datasource_notrack.csv',
            'folds': '  - \'03\'',
            'target_db': '-0.5',
        }),
        ('track_training', {
            'datasources': datasources_dir / 'datasource_track.csv',
            'folds': '  - \'01\'\n  - \'02\'',
            'target_db': '-0.1',
        }),
        ('track_validation', {
            'datasources': datasources_dir / 'datasource_track.csv',
            'folds': '  - \'03\'',
            'target_db': '-0.1',
        }),
]:
    yaml_path = samples_conf_dir / f'{samples_name}.yaml'
    params['metadata_path'] = samples_dir / samples_name / 'metadata.json'
    params['data_dir'] = samples_dir / samples_name / 'data'
    params['journal_path'] = samples_dir / samples_name / 'journal.json'
    params['log_path'] = samples_dir / samples_name / 'log'
    with open(yaml_path, 'w') as fp:
        fp.write(yaml_template.format(**params))

########################################
# prepare yaml for mixtures
########################################
yaml_template = '''# sample config file for data mixtures
algorithm:
  type: category
  args:
    mix_category_list:
      -
        - 'c001'
        - 'c002'
    include_other: yes
seed: 01410343
mix_per_sample: 1
sample_metadata_path: {sample_metadata_path}
mixture_metadata_path: {mixture_metadata_path}
journal_path: {journal_path}
log_path: {log_path}
log_level: INFO
jobs: 4
'''

mixtures_conf_dir = config_dir / 'mixtures'
samples_dir = data_dir / 'samples'
mixtures_dir = data_dir / 'mixtures'
for mixtures_name, params in [
        ('notrack_training', {}),
        ('notrack_validation', {}),
        ('track_training', {}),
        ('track_validation', {}),
]:
    yaml_path = mixtures_conf_dir / f'{mixtures_name}.yaml'

    params['sample_metadata_path'] \
        = samples_dir / mixtures_name / 'metadata.json'
    params['mixture_metadata_path'] \
        = mixtures_dir / mixtures_name / 'metadata.json'
    params['journal_path'] = mixtures_dir / mixtures_name / 'journal.json'
    params['log_path'] = mixtures_dir / mixtures_name / 'log'
    with open(yaml_path, 'w') as fp:
        fp.write(yaml_template.format(**params))


########################################
# prepare yaml for augmentations
########################################
yaml_template = '''# sample config file for data augmentations
algorithm:
  type: random
  args:
    source_sample_rate: 48000
    target_sample_rate: 24000
    waveform_length: 24000
    normalize: no
{transform_params}    n_fft: 2048
    hop_length: 512
    win_length: 2048
seed: 01410343
augmentation_per_sample: 10
sample_metadata_path: {sample_metadata_path}
augmentation_metadata_path: {augmentation_metadata_path}
journal_path: {journal_path}
log_path: {log_path}
log_level: INFO
jobs: 4
'''

with_transform = '''
    scale_range:
      - 0.6
      - 1.4
    scale_point_range:
      - 2
      - 5
    time_stretch_range:
      - 0.8
      - 1.2
    pitch_shift_range:
      - 0.8
      - 1.2
'''

without_transform = '''
    scale_range:
      - 1.0
      - 1.0
    scale_point_range:
      - 1
      - 1
    time_stretch_range:
      - 1.0
      - 1.0
    pitch_shift_range:
      - 1.0
      - 1.0
'''


augmentations_conf_dir = config_dir / 'augmentations'
samples_dir = data_dir / 'samples'
augmentations_dir = data_dir / 'augmentations'
augmentation_type = 'random'
for samples_name, params in [
        ('notrack_training', {'transform_params': with_transform}),
        ('track_training', {'transform_params': with_transform}),
        ('notrack_validation', {'transform_params': without_transform}),
        ('track_validation', {'transform_params': without_transform}),
]:
    augmentation_data_dir = \
        augmentations_dir / f'{samples_name}_{augmentation_type}'
    yaml_path = \
        augmentations_conf_dir / f'{samples_name}_{augmentation_type}.yaml'

    params['sample_metadata_path'] = \
        samples_dir / samples_name / 'metadata.json'
    params['augmentation_metadata_path'] = \
        augmentation_data_dir / 'metadata.json'
    params['journal_path'] = augmentation_data_dir / 'journal.json'
    params['log_path'] = augmentation_data_dir / 'log'
    with open(yaml_path, 'w') as fp:
        fp.write(yaml_template.format(**params))

# for easy augmentation
yaml_template = '''# sample config file for data augmentations
algorithm:
  type: entropy
  args:
    source_sample_rate: 48000
    target_sample_rate: 24000
    waveform_length: 24000
    normalize: no
    scale_range:
      - 0.6
      - 1.4
    scale_point_range:
      - 2
      - 5
    time_stretch_range:
      - 0.8
      - 1.2
    pitch_shift_range:
      - 0.8
      - 1.2
    mixture_algorithm:
      type: category
      args:
        mix_category_list:
          -
            - 'c001'
            - 'c002'
        include_other: yes
    trials_per_augmentation: 10
    separation_difficulty: {separation_difficulty}
    n_fft: 2048
    hop_length: 512
    win_length: 2048
seed: 01410343
augmentation_per_sample: 10
sample_metadata_path: {sample_metadata_path}
augmentation_metadata_path: {augmentation_metadata_path}
journal_path: {journal_path}
log_path: {log_path}
log_level: INFO
jobs: 4
'''

augmentations_conf_dir = config_dir / 'augmentations'
samples_dir = data_dir / 'samples'
augmentations_dir = data_dir / 'augmentations'
for samples_name, separation_difficulty in itertools.product(
        ['notrack_training', 'track_training'],
        ['easy', 'hard']
):
    augmentation_data_dir = \
        augmentations_dir / f'{samples_name}_{separation_difficulty}'
    yaml_path = \
        augmentations_conf_dir / f'{samples_name}_{separation_difficulty}.yaml'

    params = {}
    params['sample_metadata_path'] = \
        samples_dir / samples_name / 'metadata.json'
    params['augmentation_metadata_path'] = \
        augmentation_data_dir / 'metadata.json'
    params['journal_path'] = augmentation_data_dir / 'journal.json'
    params['log_path'] = augmentation_data_dir / 'log'
    params['separation_difficulty'] = 0.0 if separation_difficulty == 'easy' \
        else 1.0
    with open(yaml_path, 'w') as fp:
        fp.write(yaml_template.format(**params))

########################################
# prepare yaml for training
########################################

# make set for easy training
# define yaml parts
checkpoint_encdec = '''  type: initial
  args:
    model:
      type: initial
      args:
        class: baseline_encdec
        hyperparameters:
          base:
            in_channel: 1
            out_channel: 2
            n_fft: 512
            hop_length: 128
            n_lstm: 2
            lstm_channel: 32
            magbook_size: 3
            phasebook_size: 8
            output_residual: yes
        seed: 01410343
    optimizer_class: adam
    optimizer_args:
      lr: 1.0e-4
    scheduler_class: reduce_on_plateau
    scheduler_args:
      patience: 5
      factor: 0.5
    seed: 01410343
'''

checkpoint_chimera = '''  type: initial
  args:
    model:
      type: initial
      args:
        class: baseline_chimera
        hyperparameters:
          base:
            in_channel: 1
            out_channel: 2
            n_fft: 512
            hop_length: 128
            n_lstm: 2
            lstm_channel: 32
            magbook_size: 3
            phasebook_size: 8
            output_residual: yes
          embd:
            embd_feature: 256
            embd_dim: 8
        seed: 01410343
    optimizer_class: adam
    optimizer_args:
      lr: 1.0e-4
    scheduler_class: reduce_on_plateau
    scheduler_args:
      patience: 5
      factor: 0.5
    seed: 01410343
'''

checkpoint_saved = '''  type: saved
  args:
    path: {checkpoint_path}
'''

dataset_template = '''  type: composite
  args:
    components:
{dataset_components}'''

basic_dataset_template = '''      -
        type: basic
        args:
          samples_metadata_path: {samples_metadata_path}
          augmentations_metadata_path: {augmentations_metadata_path}
          mixtures_metadata_path: {mixtures_metadata_path}'''

loss_chimera = '''  type: composite
  args:
    components:
      -
        func:
          type: negative_sdr
        weight: 1.0
      -
        func:
          type: spectrogram
          args:
            spectral_convergence_weight: 1.0
            spectral_magnitude_weight: 1.0
            spectral_magnitue_norm: 1
            spectral_magnitude_log: yes
        weight: 1.0
      -
        func:
          type: whitened_kmeans
        weight: 1.0e-3
'''

loss_encdec = '''  type: composite
  args:
    components:
      -
        func:
          type: negative_sdr
        weight: 1.0
      -
        func:
          type: spectrogram
          args:
            spectral_convergence_weight: 1.0
            spectral_magnitude_weight: 1.0
            spectral_magnitue_norm: 1
            spectral_magnitude_log: yes
        weight: 1.0
'''

template = '''# sample config file for training
# checkpoint
checkpoint:
{checkpoint}

# dataset params
training_dataset:
{training_dataset}
training_sample_size: 16

validation_dataset:
{validation_dataset}
validation_sample_size: 8

# output params
checkpoint_pattern: {checkpoint_pattern}
journal_pattern: {journal_pattern}
log_pattern: {log_pattern}
log_level: INFO

# misc. training param
loss_function:
{loss_function}

batch_size: 4
compute_batch_size: 4
training_epochs: 5
max_grad_norm: 1.0e+4
n_fft: 2048
hop_length: 512
seed: 01410343
device: cpu
jobs: 0
'''

trainings_conf_dir = config_dir / 'trainings'
samples_dir = data_dir / 'samples'
augmentations_dir = data_dir / 'augmentations'
mixtures_dir = data_dir / 'mixtures'
trainings_dir = root / 'trainings'

validation_dataset = dataset_template.format(
    dataset_components
    = basic_dataset_template.format(
        samples_metadata_path
        = samples_dir/'notrack_validation'/'metadata.json',
        augmentations_metadata_path
        = augmentations_dir/'notrack_validation_random'/'metadata.json',
        mixtures_metadata_path
        = mixtures_dir/'notrack_validation'/'metadata.json',
    ) + '\n'
    + basic_dataset_template.format(
        samples_metadata_path
        = samples_dir/'track_validation'/'metadata.json',
        augmentations_metadata_path
        = augmentations_dir/'track_validation_random'/'metadata.json',
        mixtures_metadata_path
        = mixtures_dir/'track_validation'/'metadata.json',
    )
)

training_datasets = dict((k, dataset_template.format(
    dataset_components
    = basic_dataset_template.format(
        samples_metadata_path
        = samples_dir/'notrack_training'/'metadata.json',
        augmentations_metadata_path
        = augmentations_dir/'notrack_training_random'/'metadata.json',
        mixtures_metadata_path
        = mixtures_dir/'notrack_training'/'metadata.json',
    ) + '\n'
    + basic_dataset_template.format(
        samples_metadata_path
        = samples_dir/'track_training'/'metadata.json',
        augmentations_metadata_path
        = augmentations_dir/'track_training_random'/'metadata.json',
        mixtures_metadata_path
        = mixtures_dir/'track_training'/'metadata.json',
    ) + '\n'
    + basic_dataset_template.format(
        samples_metadata_path
        = samples_dir/'notrack_training'/'metadata.json',
        augmentations_metadata_path
        = augmentations_dir/f'notrack_training_{k}'/'metadata.json',
        mixtures_metadata_path
        = mixtures_dir/'notrack_training'/'metadata.json',
    ) + '\n'
    + basic_dataset_template.format(
        samples_metadata_path
        = samples_dir/'track_training'/'metadata.json',
        augmentations_metadata_path
        = augmentations_dir/f'track_training_{k}'/'metadata.json',
        mixtures_metadata_path
        = mixtures_dir/'track_training'/'metadata.json',
    )
)) for k in ('easy', 'hard'))

for model_type, train_type in itertools.product(
        ['encdec', 'chimera'],
        ['easy', 'hard']
):


    trainings_path = trainings_dir / f'baseline_{model_type}'
    checkpoint_pattern = trainings_path/'checkpoints'/'{model_epoch:02d}.tar'
    journal_pattern = trainings_path/'journals'/'{model_epoch:02d}.json'
    log_pattern = trainings_path/'logs'/'{model_epoch:02}.txt'
    training_dataset = training_datasets[train_type]

    if model_type == 'encdec':
        checkpoint = checkpoint_encdec
        loss_function = loss_encdec
    elif model_type == 'chimera':
        checkpoint = checkpoint_chimera
        loss_function = loss_chimera

    if train_type == 'easy':
        pass
    elif train_type == 'hard':
        # overwrite to saved checkpoint
        checkpoint = checkpoint_saved.format(
            checkpoint_path=trainings_path/'checkpoints'/'05.tar'
        )

    yaml_content = template.format(
        checkpoint=checkpoint,
        training_dataset=training_dataset,
        validation_dataset=validation_dataset,
        checkpoint_pattern=checkpoint_pattern,
        journal_pattern=journal_pattern,
        log_pattern=log_pattern,
        loss_function=loss_function,
    )

    yaml_path = \
        trainings_conf_dir / f'baseline_{model_type}_{train_type}.yaml'
    with open(yaml_path, 'w') as fp:
        fp.write(yaml_content)


########################################
# prepare yaml for validation
########################################

# make set for easy training
# define yaml parts
template = '''# sample config file for validation
# model
model:
  type: checkpoint
  args:
    checkpoint:
      type: saved
      args:
        path: {checkpoint_path}

# dataset params
validation_dataset:
{validation_dataset}
validation_sample_size: 8

# output params
journal_pattern: {journal_pattern}
log_pattern: {log_pattern}
log_level: INFO

# misc. training param
loss_function:
{loss_function}

batch_size: 4
compute_batch_size: 4
n_fft: 2048
hop_length: 512
device: cpu
jobs: 0
'''

validations_conf_dir = config_dir / 'validations'
samples_dir = data_dir / 'samples'
augmentations_dir = data_dir / 'augmentations'
mixtures_dir = data_dir / 'mixtures'
trainings_dir = root / 'trainings'
validations_dir = root / 'validations'

for model_type, model_epoch in itertools.product(
        ['encdec', 'chimera'],
        ['05', '10']
):

    validations_path = validations_dir / f'baseline_{model_type}'
    journal_pattern = validations_path/'journals'/'{model_epoch:02d}.json'
    log_pattern = validations_path/'logs'/'{model_epoch:02}.txt'
    checkpoint_path = trainings_dir / \
        f'baseline_{model_type}' / 'checkpoints' / f'{model_epoch}.tar'

    if model_type == 'encdec':
        loss_function = loss_encdec
    elif model_type == 'chimera':
        loss_function = loss_chimera

    if train_type == 'easy':
        pass
    elif train_type == 'hard':
        # overwrite to saved checkpoint
        checkpoint = checkpoint_saved.format(
            checkpoint_path=validations_path/'checkpoints'/'05.tar'
        )

    yaml_content = template.format(
        checkpoint_path=checkpoint_path,
        validation_dataset=validation_dataset,
        journal_pattern=journal_pattern,
        log_pattern=log_pattern,
        loss_function=loss_function,
    )

    yaml_path = \
        validations_conf_dir / f'baseline_{model_type}_{model_epoch}.yaml'
    with open(yaml_path, 'w') as fp:
        fp.write(yaml_content)

########################################
# prepare yaml for separation
########################################
template = '''# sample config file for separation
# model
model:
  type: checkpoint
  args:
    checkpoint:
      type: saved
      args:
        path: {checkpoint_path}

# input/output params
input: {input}
output: {output}
journal_pattern: {journal_pattern}
log_pattern: {log_pattern}
log_level: INFO

sample_rate: 24000
duration: 0.25
overlap: 0.75
permutation_invariant: no
batch_size: 4
device: cpu
'''

separation_conf_dir = config_dir / 'separations'
separations_dir = root / 'separations'

for model_type, model_epoch in itertools.product(
        ['chimera'],
        ['05', '10']
):

    separations_path = separations_dir / f'baseline_{model_type}'
    separation_path = separations_dir / f'baseline_{model_type}'
    journal_pattern = separations_path/'journals'/'{model_epoch:02d}.json'
    log_pattern = separations_path/'logs'/'{model_epoch:02}.txt'
    checkpoint_path = trainings_dir / \
        f'baseline_{model_type}' / 'checkpoints' / f'{model_epoch}.tar'

    input = datasources_dir/f'sin440long.wav'
    output = f'''
  - {separation_path/"result"/("out_"+model_epoch+"_1.wav")}
  - {separation_path/"result"/("out_"+model_epoch+"_2.wav")}'''

    yaml_content = template.format(
        checkpoint_path=checkpoint_path,
        input=input,
        output=output,
        journal_pattern=journal_pattern,
        log_pattern=log_pattern,
    )

    yaml_path = \
        separation_conf_dir / f'baseline_{model_type}_{model_epoch}.yaml'
    with open(yaml_path, 'w') as fp:
        fp.write(yaml_content)

########################################
# prepare yaml for cluster
########################################
template = '''# sample config file for clustering
# model
model:
  type: checkpoint
  args:
    checkpoint:
      type: saved
      args:
        path: {checkpoint_path}

# input/output params
input: {input}
output: {output}
journal_pattern: {journal_pattern}
log_pattern: {log_pattern}
log_level: INFO

sample_rate: 24000
duration: 0.25
overlap: 0.75
n_fft: 512
hop_length: 128
batch_size: 4
device: cpu
'''

clustering_conf_dir = config_dir / 'clusterings'
clusterings_dir = root / 'clusterings'

for model_type, model_epoch in itertools.product(
        ['chimera'],
        ['05', '10']
):

    clusterings_path = clusterings_dir / f'baseline_{model_type}'
    clustering_path = clusterings_dir / f'baseline_{model_type}'
    journal_pattern = clusterings_path/'journals'/'{model_epoch:02d}.json'
    log_pattern = clusterings_path/'logs'/'{model_epoch:02}.txt'
    checkpoint_path = trainings_dir / \
        f'baseline_{model_type}' / 'checkpoints' / f'{model_epoch}.tar'

    input = datasources_dir/f'sin440long.wav'
    output = f'''
  - {clustering_path/"result"/("out_"+model_epoch+"_1.wav")}
  - {clustering_path/"result"/("out_"+model_epoch+"_2.wav")}
  - {clustering_path/"result"/("out_"+model_epoch+"_3.wav")}'''

    yaml_content = template.format(
        checkpoint_path=checkpoint_path,
        input=input,
        output=output,
        journal_pattern=journal_pattern,
        log_pattern=log_pattern,
    )

    yaml_path = \
        clustering_conf_dir / f'baseline_{model_type}_{model_epoch}.yaml'
    with open(yaml_path, 'w') as fp:
        fp.write(yaml_content)

