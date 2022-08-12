
import argparse

from ..data.datasets import Dataset
from ..models.models import Model
from ..metrics.loss import Loss
from ..procs.validations import ValidationSpec
from .utils import make_config_action, make_structure_action

def attach_parser(parser):
    config_schema = {
        'tyue': 'object',
        'properties': {
            'model': {'type': 'object'},

            'validation_dataset': {'type': 'object'},
            'validation_sample_size': {'type': 'integer'},

            'journal_pattern': {'type': 'string'},
            'log_pattern': {'type': 'string'},
            'log_level': {'type': 'string'},

            'loss_function': {'type': 'object'},
            'batch_size': {'type': 'integer'},
            'compute_batch_size': {'type': 'integer'},
            'n_fft': {'type': 'integer'},
            'hop_length': {'type': 'integer'},
            'device': {'type': 'string', 'pattern': '^(cpu|cuda)$'},
            'jobs': {'type': 'integer'},
        },
        'additionalProperties': False,
    }

    parser.add_argument('-c', '--config',
                        help='configuration path',
                        action=make_config_action(config_schema))


    # checkpoint
    group = parser.add_argument_group(title='model parameter')
    group.add_argument('--model',
                       action=make_structure_action(Model))

    # dataset
    group = parser.add_argument_group(title='dataset parameters')
    group.add_argument('--validation-dataset',
                       action=make_structure_action(Dataset))
    group.add_argument('--validation-sample-size', type=int)

    # output
    group = parser.add_argument_group(title='output parameters')
    group.add_argument('--journal-pattern')
    group.add_argument('--log-pattern')
    group.add_argument('--log-level')

    # validation param
    group = parser.add_argument_group(title='validation parameters')
    group.add_argument('--loss-function',
                       action=make_structure_action(Loss))
    group.add_argument('--batch-size', type=int)
    group.add_argument('--compute-batch-size', type=int)
    group.add_argument('--n-fft', type=int)
    group.add_argument('--hop-length', type=int)
    group.add_argument('--device', choices=('cpu', 'cuda'))
    group.add_argument('--jobs', type=int)


def main(args):
    spec = ValidationSpec(
        # model
        model=args.model,

        # dataset params
        validation_dataset=args.validation_dataset,
        validation_sample_size=args.validation_sample_size,

        # output params
        journal_pattern=args.journal_pattern,
        log_pattern=args.log_pattern,
        log_level=args.log_level,

        # misc. training param
        loss_function=args.loss_function,
        batch_size=args.batch_size,
        compute_batch_size=args.compute_batch_size,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        device=args.device,
        jobs=args.jobs,
    )
    spec.validate()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('echidna-evaluate')
    attach_parser(parser)
    args = parser.parse_args()
    main(args)
