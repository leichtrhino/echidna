
import argparse

from .utils import make_config_action, make_structure_action
from ..data.mixtures import (
    MixAlgorithm,
    MixtureSpec,
)

def attach_parser(parser):
    config_schema = {
        'type': 'object',
        'properties': {
            'algorithm': {'type': 'object'},
            'seed': {'type': 'integer'},
            'mix_per_sample': {'type': 'integer'},
            'sample_metadata_path': {'type': 'string'},
            'mixture_metadata_path': {'type': 'string'},
            'journal_path': {'type': 'string'},
            'log_path': {'type': 'string'},
            'log_level': {'type': 'string'},
            'jobs': {'type': 'integer'},
        },
        'additionalProperties': False,
    }

    parser.add_argument('-c', '--config',
                        help='configuration path',
                        action=make_config_action(config_schema))

    group = parser.add_argument_group(title='mixture parameters')
    group.add_argument('--algorithm',
                       action=make_structure_action(MixAlgorithm))
    group.add_argument('--seed')
    group.add_argument('--mix-per-sample')

    group = parser.add_argument_group(title='output parameters')
    group.add_argument('--sample-metadata-path')
    group.add_argument('--mixture-metadata-path')
    group.add_argument('--journal-path')

    group = parser.add_argument_group(title='misc parameters')
    group.add_argument('--log-path')
    group.add_argument('--log-level')
    group.add_argument('--jobs')


def main(args):
    spec = MixtureSpec(
        algorithm=args.algorithm,
        seed=args.seed,
        mix_per_sample=args.mix_per_sample,
        sample_metadata_path=args.sample_metadata_path,
        mixture_metadata_path=args.mixture_metadata_path,
        journal_path=args.journal_path,
        log_path=args.log_path,
        log_level=args.log_level,
        jobs=args.jobs,
    )
    spec.save_mixture()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('echidna-data-mixtures')
    attach_parser(parser)
    args = parser.parse_args()
    main(args)
