
import argparse

from ..data.datasets import Dataset
from ..models.checkpoints import Checkpoint
from ..metrics.loss import Loss
from ..procs.trainings import TrainingSpec
from .utils import make_config_action, make_structure_action

def attach_parser(parser):
    config_schema = {
        'tyue': 'object',
        'properties': {
            'checkpoint': {'type': 'object'},

            'training_dataset': {'type': 'object'},
            'training_sample_size': {'type': ['integer', 'null']},
            'validation_dataset': {'type': ['object', 'null']},
            'validation_sample_size': {'type': ['integer', 'null']},

            'checkpoint_pattern': {'type': 'string'},
            'journal_pattern': {'type': 'string'},
            'log_pattern': {'type': 'string'},
            'log_level': {'type': 'string'},

            'loss_function': {'type': 'object'},
            'batch_size': {'type': 'integer'},
            'compute_batch_size': {'type': ['integer', 'null']},
            'training_epochs': {'type': 'integer'},
            'max_grad_norm': {'type': 'number'},
            'n_fft': {'type': 'integer'},
            'hop_length': {'type': 'integer'},
            'seed': {'type': 'integer'},
            'device': {'type': 'string', 'pattern': '^(cpu|cuda)$'},
            'jobs': {'type': 'integer'},
        },
        'additionalProperties': False,
    }

    parser.add_argument('-c', '--config',
                        help='configuration path',
                        action=make_config_action(config_schema))

    # checkpoint
    group = parser.add_argument_group(title='checkpoint parameters')
    group.add_argument('--checkpoint',
                       action=make_structure_action(Checkpoint))

    # dataset params
    group = parser.add_argument_group(title='dataset parameters')
    group.add_argument('--training-dataset',
                       action=make_structure_action(Dataset))
    group.add_argument('--training-sample-size', type=int)
    group.add_argument('--validation-dataset',
                       action=make_structure_action(Dataset))
    group.add_argument('--validation-sample-size', type=int)

    # output params
    group = parser.add_argument_group(title='output parameters')
    group.add_argument('--checkpoint-pattern')
    group.add_argument('--journal-pattern')
    group.add_argument('--log-pattern')
    group.add_argument('--log-level')

    # training param
    group = parser.add_argument_group(title='training parameters')
    group.add_argument('--loss-function',
                       action=make_structure_action(Loss))
    group.add_argument('--batch-size', type=int)
    group.add_argument('--compute-batch-size', type=int)
    group.add_argument('--training-epochs', type=int)
    group.add_argument('--max-grad-norm', type=float)
    group.add_argument('--n-fft', type=int)
    group.add_argument('--hop-length', type=int)
    group.add_argument('--seed', type=int)
    group.add_argument('--device', choices=('cpu', 'cuda'))
    group.add_argument('--jobs', type=int)


def main(args):
    spec = TrainingSpec(
        # checkpoint
        checkpoint=args.checkpoint,

        # dataset params
        training_dataset=args.training_dataset,
        training_sample_size=args.training_sample_size,
        validation_dataset=args.validation_dataset,
        validation_sample_size=args.validation_sample_size,

        # output params
        checkpoint_pattern=args.checkpoint_pattern,
        journal_pattern=args.journal_pattern,
        log_pattern=args.log_pattern,
        log_level=args.log_level,

        # misc. training param
        loss_function=args.loss_function,
        batch_size=args.batch_size,
        compute_batch_size=args.compute_batch_size,
        training_epochs=args.training_epochs,
        max_grad_norm=args.max_grad_norm,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        seed=args.seed,
        device=args.device,
        jobs=args.jobs,
    )
    spec.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('echidna-train')
    attach_parser(parser)
    args = parser.parse_args()
    main(args)
