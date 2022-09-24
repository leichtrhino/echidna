
import argparse

from ..data.samples import SampleSpec
from .utils import make_config_action, DatasourceAction, LoadJSONAction

def attach_parser(parser):
    config_schema = {
        'type': 'object',
        'properties': {
            'datasources': {'type': 'string'},
            'fold': {
                'type': 'array',
                'items': {'type': 'string'}
            },
            'sample_size': {'type': 'integer'},
            'category_map': {'type': 'object'},
            'sample_rate': {'type': 'integer'},
            'duration': {'type': 'number'},
            'target_db': {'type': 'number'},
            'seed': {'type': 'integer'},
            'metadata_path': {'type': 'string'},
            'data_dir': {'type': 'string'},
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

    group = parser.add_argument_group(title='datasource parameters',
                                      description='input')
    group.add_argument('--datasources', action=DatasourceAction)
    group.add_argument('--fold', nargs='+')

    group = parser.add_argument_group(title='sample parameters')
    group.add_argument('--sample-size')
    group.add_argument('--category-map', action=LoadJSONAction)
    group.add_argument('--sample-rate')
    group.add_argument('--duration')
    group.add_argument('--target-db', type=float)
    group.add_argument('--seed')

    group = parser.add_argument_group(title='output parameters')
    group.add_argument('--data-dir')
    group.add_argument('--metadata-path')
    group.add_argument('--journal-path')

    group = parser.add_argument_group(title='misc parameters')
    group.add_argument('--log-path')
    group.add_argument('--log-level')
    group.add_argument('--jobs')
    group.add_argument('--device')

def main(args):
    spec = SampleSpec(
        datasources=args.datasources,
        fold=args.fold,
        sample_size=args.sample_size,
        category_map=args.category_map,
        sample_rate=args.sample_rate,
        duration=args.duration,
        target_db=args.target_db,
        seed=args.seed,
        metadata_path=args.metadata_path,
        data_dir=args.data_dir,
        journal_path=args.journal_path,
        log_path=args.log_path,
        log_level=args.log_level,
        jobs=args.jobs,
        device=args.device,
    )
    spec.save_samples()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('echidna-data-samples')
    attach_parser(parser)
    args = parser.parse_args()
    main(args)
