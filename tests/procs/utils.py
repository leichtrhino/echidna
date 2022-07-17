
import random
import torch

from echidna.procs.trainings import TrainingSpec
from echidna.procs.validations import ValidationSpec
from echidna.metrics.waveform import L1WaveformLoss, L2WaveformLoss
from echidna.metrics.deepclustering import DeepClusteringLoss
from echidna.metrics.composite import CompositeLoss
from echidna.models.models import InitialModel
from echidna.models.checkpoints import InitialCheckpoint

#from ..models.utils import get_initial_model, get_initial_checkpoint
from ..data.utils import ToyDataset

def get_encdec_model():
    return InitialModel(
        klass='baseline_encdec',
        hyperparameters={
            'base': dict(
                in_channel=1,
                out_channel=3,
                n_lstm=2,
                lstm_channel=60,
                n_fft=128,
                hop_length=32,
                magbook_size=1,
                phasebook_size=1,
                output_residual=False
            ),
        },
        seed=1410343
    )

def get_chimera_model():
    return InitialModel(
        klass='baseline_chimera',
        hyperparameters={
            'base': dict(
                in_channel=1,
                out_channel=3,
                n_lstm=2,
                lstm_channel=60,
                n_fft=128,
                hop_length=32,
                magbook_size=1,
                phasebook_size=1,
                output_residual=False
            ),
            'embd': dict(
                embd_feature=64,
                embd_dim=16,
            )
        },
        seed=1410343
    )


def get_training_spec(chimera : bool,
                      composite_loss : bool,
                      dataset_size : int,
                      sample_size : int,
                      batch_size : int,
                      max_grad_norm : float,
                      compute_batch_size : int,
                      checkpoint_pattern,
                      journal_pattern,
                      log_pattern):

    checkpoint = InitialCheckpoint(
        model=get_chimera_model() if chimera else get_encdec_model(),
        optimizer_class='adam',
        optimizer_args={'lr': 1e-1},
        scheduler_class='reduce_on_plateau',
        scheduler_args={'factor': 0.5, 'patience': 5},
        seed=1410343,
    )

    training_dataset = ToyDataset(dataset_size)
    validation_dataset = ToyDataset(dataset_size, val=True)
    loss_module = CompositeLoss(
        [{'func': L1WaveformLoss(), 'weight': 1.0},
         {'func': L2WaveformLoss(), 'weight': 1.0},
         {'func': DeepClusteringLoss(), 'weight': 1.0}],
        permutation='none',
        reduction='mean',
    ) if composite_loss else L1WaveformLoss(reduction='mean')

    # initialize training and validation specs
    training_spec = TrainingSpec(
        # checkpoint
        checkpoint=checkpoint,

        # dataset params
        training_dataset=training_dataset,
        training_sample_size=sample_size,
        validation_dataset=validation_dataset,
        validation_sample_size=sample_size,

        # output params
        checkpoint_pattern=checkpoint_pattern,
        journal_pattern=journal_pattern,
        log_pattern=log_pattern,
        log_level='DEBUG',

        # misc. training param
        loss_function=loss_module,
        batch_size=batch_size,
        compute_batch_size=compute_batch_size,
        training_epochs=2,
        max_grad_norm=max_grad_norm,
        n_fft=2048,
        hop_length=512,
        seed=1410343,
        device='cpu'
    )

    return training_spec

def get_validation_spec(chimera : bool,
                        composite_loss : bool,
                        dataset_size : int,
                        sample_size : int,
                        batch_size : int,
                        compute_batch_size : int,
                        journal_pattern,
                        log_pattern):

    model = get_chimera_model() if chimera else get_encdec_model()
    validation_dataset = ToyDataset(dataset_size, val=True)
    loss_module = CompositeLoss(
        [{'func': L1WaveformLoss(), 'weight': 1.0},
         {'func': L2WaveformLoss(), 'weight': 1.0},
         {'func': DeepClusteringLoss(), 'weight': 1.0}],
        permutation='none',
        reduction='mean',
    ) if composite_loss else L1WaveformLoss(reduction='mean')

    # initialize validation and validation specs
    validation_spec = ValidationSpec(
        # model
        model=model,

        # dataset params
        validation_dataset=validation_dataset,
        validation_sample_size=sample_size,

        # output params
        journal_pattern=journal_pattern,
        log_pattern=log_pattern,
        log_level='DEBUG',

        # misc. validation param
        loss_function=loss_module,
        batch_size=batch_size,
        compute_batch_size=compute_batch_size,
        n_fft=2048,
        hop_length=512,
        device='cpu'
    )

    return validation_spec

