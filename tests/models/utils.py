
from echidna.models.models import InitialModel
from echidna.models.checkpoints import InitialCheckpoint

def get_initial_model():
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

def get_initial_checkpoint():
    return InitialCheckpoint(
        model=get_initial_model(),
        optimizer_class='adam',
        optimizer_args={'lr': 1e-1},
        scheduler_class='reduce_on_plateau',
        scheduler_args={'factor': 0.5, 'patience': 5},
        seed=1410343,
    )

