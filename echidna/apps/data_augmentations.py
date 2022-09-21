
import argparse

from .utils import make_config_action
from ..data.augmentations import AugmentationSpec

def attach_parser(parser):
    config_schema = {
        'type': 'object',
        'properties': {
            'source_sample_rate': {'type': 'number'},
            'target_sample_rate': {'type': 'number'},
            'waveform_length': {'type': 'number'},
            'scale_range': {'type': 'array'},
            'scale_point_range': {'type': 'array'},
            'time_stretch_range': {'type': 'array'},
            'pitch_shift_range': {'type': 'array'},
            'n_fft': {'type': 'integer'},
            'hop_length': {'type': 'integer'},
            'win_length': {'type': 'integer'},

            'seed': {'type': 'integer'},
            'augmentation_per_parent': {'type': 'integer'},
            'input_metadata_path': {'type': 'string'},
            'output_metadata_path': {'type': 'string'},
            'journal_path': {'type': 'string'},
            'log_path': {'type': 'string'},
            'log_level': {'type': 'string'},
            'jobs': {
                'Or': [
                    {'type': 'integer'},
                    {'type': 'null'},
                ]
            },
            'device': {
                'Or': [
                    {'type': 'string'},
                    {'type': 'null'},
                ]
            }
        },
        'additionalProperties': False,
    }

    parser.add_argument('-c', '--config',
                        help='configuration path',
                        action=make_config_action(config_schema))

    group = parser.add_argument_group(title='algorithm parameters')
    group.add_argument('--source-sample-rate')
    group.add_argument('--target-sample-rate')
    group.add_argument('--waveform-length')
    group.add_argument('--scale-range')
    group.add_argument('--scale-point-range')
    group.add_argument('--time-stretch-range')
    group.add_argument('--pitch-shift-range')
    group.add_argument('--n_fft', default=2048)
    group.add_argument('--hop_length', default=512)
    group.add_argument('--win_length', default=2048)

    group = parser.add_argument_group(title='augment parameters')
    group.add_argument('--seed')
    group.add_argument('--augmentation-per-parent')

    group = parser.add_argument_group(title='output parameters')
    group.add_argument('--input-metadata-path')
    group.add_argument('--output-metadata-path')
    group.add_argument('--journal-path')

    group = parser.add_argument_group(title='misc parameters')
    group.add_argument('--log-path')
    group.add_argument('--log-level')
    group.add_argument('--jobs')
    group.add_argument('--device')

def main(args):
    spec = AugmentationSpec(
        source_sample_rate=args.source_sample_rate,
        target_sample_rate=args.target_sample_rate,
        waveform_length=args.waveform_length,
        scale_range=args.scale_range,
        scale_point_range=args.scale_point_range,
        time_stretch_range=args.time_stretch_range,
        pitch_shift_range=args.pitch_shift_range,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,

        seed=args.seed,
        augmentation_per_parent=args.augmentation_per_parent,
        input_metadata_path=args.input_metadata_path,
        output_metadata_path=args.output_metadata_path,
        journal_path=args.journal_path,
        log_path=args.log_path,
        log_level=args.log_level,
        jobs=args.jobs,
        device=args.device,
    )
    spec.save_augmentation()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('echidna-data-augmentations')
    attach_parser(parser)
    args = parser.parse_args()
    main(args)
