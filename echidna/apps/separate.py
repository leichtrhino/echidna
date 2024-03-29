
import argparse

from ..models.models import Model
from ..procs.separations import SeparationSpec
from .utils import make_config_action, make_structure_action

def attach_parser(parser):
    config_schema = {
        'type': 'object',
        'properties': {
            'model': {'type': 'object'},
            'input': {'type': 'string'},
            'output': {'type': 'array'},
            'journal_pattern': {'type': 'string'},
            'log_pattern': {'type': ['string', 'null']},
            'log_level': {'type': ['string', 'null']},
            'sample_rate': {'type': 'integer'},
            'duration': {'type': ['number', 'null']},
            'overlap': {'type': ['number', 'null']},
            'permutation_invariant': {'type': ['boolean', 'null']},
            'batch_size': {'type': ['integer', 'null']},
            'device': {'type': ['string', 'null']},
        }
    }
    parser.add_argument('-c', '--config',
                        help='configuration path',
                        action=make_config_action(config_schema))

    # model
    group = parser.add_argument_group(title='model')
    group.add_argument('--model',
                       action=make_structure_action(Model))

    # input/output
    group = parser.add_argument_group(title='input/output')
    group.add_argument('--input')
    group.add_argument('--output', nargs='+')
    group.add_argument('--journal-pattern')
    group.add_argument('--log-pattern')
    group.add_argument('--log-level')

    # separation quality parameters
    group = parser.add_argument_group(title='separation quality parameters')
    group.add_argument('--sample-rate', type=int)
    group.add_argument('--duration', type=float)
    group.add_argument('--overlap', type=float)
    group.add_argument('--permutation-invariant', type=bool)

    # separation process parameters
    group = parser.add_argument_group(title='separation process parameters')
    group.add_argument('--batch-size', type=int)
    group.add_argument('--device', choices=('cpu', 'cuda'))


def main(args):
    spec = SeparationSpec(
        # model
        model=args.model,
        # input/output
        input=args.input,
        output=args.output,
        journal_pattern=args.journal_pattern,
        log_pattern=args.log_pattern,
        log_level=args.log_level,
        # separation quality parameters
        sample_rate=args.sample_rate,
        duration=args.duration,
        overlap=args.overlap,
        permutation_invariant=args.permutation_invariant,
        # separation process parameters
        batch_size=args.batch_size,
        device=args.device,
    )
    spec.separate()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('echidna-separate')
    attach_parser(parser)
    args = parser.parse_args()
    main(args)
