
import argparse

from .utils import make_config_action, LoadJSONAction
from ..data.mixtures import MixSetSpec

def attach_parser(parser):
    config_schema = {
        'type': 'object',
        'properties': {
            'seed': {'type': 'integer'},
            'mix_category_list': {'type': 'array'},
            'mix_per_parent': {
                'Or': [
                    {'type': 'integer'},
                    {'type': 'null'},
                ]
            },
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
            },
        },
        'additionalProperties': False,
    }

    parser.add_argument('-c', '--config',
                        help='configuration path',
                        action=make_config_action(config_schema))

    group = parser.add_argument_group(title='mixture parameters')
    group.add_argument('--mix-category-list', action=LoadJSONAction)
    group.add_argument('--seed')
    group.add_argument('--mix-per-parent')

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
    spec = MixSetSpec(
        seed=args.seed,
        mix_category_list=args.mix_category_list,
        mix_per_parent=args.mix_per_parent,
        input_metadata_path=args.input_metadata_path,
        output_metadata_path=args.output_metadata_path,
        journal_path=args.journal_path,
        log_path=args.log_path,
        log_level=args.log_level,
        jobs=args.jobs,
        device=args.device,
    )
    spec.save_mixture()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('echidna-data-mixtures')
    attach_parser(parser)
    args = parser.parse_args()
    main(args)
