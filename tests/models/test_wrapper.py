
import unittest
import tempfile
import torch

from echidna.models.nest import BaselineEncDecModel
from echidna.models.multidomain.encdec import EncDecModel
from echidna.models.wrapper import (
    Model,
    InitialModel,
    SavedModel,
    Checkpoint,
    InitialCheckpoint,
    SavedCheckpoint,
    CheckpointModel,
)

def get_initial_model():
    return InitialModel(
        klass='baseline_encdec',
        hyperparameters={
            'base': dict(
                in_channel=2,
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

class TestInitialModel(unittest.TestCase):
    def test_init_model(self):
        m = get_initial_model()

        # create model
        torch_model = m.get_torch_model()
        self.assertEqual(type(torch_model), EncDecModel)
        self.assertEqual(m.get_class(), 'baseline_encdec')
        self.assertEqual(
            m.get_hyperparameters(),
            {
                'base': {
                    'in_channel': 2,
                    'out_channel': 3,
                    'n_lstm': 2,
                    'lstm_channel': 60,
                    'n_fft': 128,
                    'hop_length': 32,
                    'magbook_size': 1,
                    'phasebook_size': 1,
                    'output_residual': False,
                }
            }
        )

    def test_serialize_model(self):
        m = get_initial_model()

        # create dateset dict and evaluate
        d = m.to_dict()
        self.assertEqual(
            d,
            {
                'type': 'initial',
                'args': {
                    'class': 'baseline_encdec',
                    'hyperparameters': {
                        'base': {
                            'in_channel': 2,
                            'out_channel': 3,
                            'n_lstm': 2,
                            'lstm_channel': 60,
                            'n_fft': 128,
                            'hop_length': 32,
                            'magbook_size': 1,
                            'phasebook_size': 1,
                            'output_residual': False,
                        }
                    },
                    'seed': 1410343,
                }
            }
        )

        m_d = Model.from_dict(d)
        self.assertEqual(m.to_dict(), m_d.to_dict())

    def test_save_model(self):
        # save the model to temporary file
        with tempfile.NamedTemporaryFile() as tmpfile:
            m = get_initial_model()
            m.save_torch_model(tmpfile.name)
            d = torch.load(tmpfile.name)
            self.assertEqual(
                set(d.keys()),
                {'class', 'hyperparameters', 'state'}
            )

class TestSavedModel(unittest.TestCase):
    def setUp(self):
        self.tmpfile = tempfile.NamedTemporaryFile()
        self.init_m = get_initial_model()
        self.init_m.save_torch_model(self.tmpfile.name)

    def tearDown(self):
        self.tmpfile.close()

    def test_init_model(self):
        saved_model = SavedModel(self.tmpfile.name)
        saved_torch_model = saved_model.get_torch_model()
        torch_model = self.init_m.get_torch_model()
        self.assertEqual(type(saved_torch_model), type(torch_model))
        self.assertEqual(saved_model.get_class(), self.init_m.get_class())
        self.assertEqual(saved_model.get_hyperparameters(),
                         self.init_m.get_hyperparameters())

    def test_serialize(self):
        saved_model = SavedModel(self.tmpfile.name)

        d = saved_model.to_dict()
        self.assertEqual(
            d,
            {
                'type': 'saved',
                'args': {
                    'path': self.tmpfile.name,
                }
            }
        )

        m_d = Model.from_dict(d)
        self.assertEqual(saved_model.to_dict(), m_d.to_dict(),)


class TestInitialCheckpoint(unittest.TestCase):
    def test_init_checkpoint(self):
        checkpoint = get_initial_checkpoint()

        # get model
        model = checkpoint.get_model()
        self.assertEqual(model.get_class(), get_initial_model().get_class())
        self.assertEqual(model.get_hyperparameters(),
                         get_initial_model().get_hyperparameters())

        # create torch model
        torch_model = checkpoint.get_torch_model()
        self.assertEqual(type(torch_model), EncDecModel)

        # craete torch optimizer
        optim = checkpoint.get_torch_optimizer()
        self.assertEqual(type(optim), torch.optim.Adam)
        for p1, p2 in zip(
                torch_model.parameter_list(base_lr=0.1), optim.param_groups):
            self.assertEqual(p1.get('lr', 0.1), p2['lr'])

        # crate torch scheduler
        scheduler = checkpoint.get_torch_scheduler()
        self.assertEqual(type(scheduler),
                         torch.optim.lr_scheduler.ReduceLROnPlateau)
        self.assertEqual(scheduler.factor, 0.5)
        self.assertEqual(type(scheduler.optimizer), torch.optim.Adam)

    def test_serialize_checkpoint(self):
        checkpoint = get_initial_checkpoint()
        d = checkpoint.to_dict()
        self.assertEqual(
            d,
            {
                'type': 'initial',
                'args': {
                    'model': {
                        'type': 'initial',
                        'args': {
                            'class': 'baseline_encdec',
                            'hyperparameters': {
                                'base': {
                                    'in_channel': 2,
                                    'out_channel': 3,
                                    'n_lstm': 2,
                                    'lstm_channel': 60,
                                    'n_fft': 128,
                                    'hop_length': 32,
                                    'magbook_size': 1,
                                    'phasebook_size': 1,
                                    'output_residual': False,
                                }
                            },
                            'seed': 1410343,
                        },
                    },
                    'optimizer_class': 'adam',
                    'optimizer_args': {'lr': 0.1},
                    'scheduler_class': 'reduce_on_plateau',
                    'scheduler_args': {'factor': 0.5, 'patience': 5},
                    'seed': 1410343
                }
            }
        )

        c_d = Checkpoint.from_dict(d)
        self.assertEqual(d, c_d.to_dict())

    def test_save_checkpoint(self):
        # save the checkpoint to temporary file
        with tempfile.NamedTemporaryFile() as tmpfile:
            c = get_initial_checkpoint()
            c.save_torch_checkpoint(tmpfile.name)
            d = torch.load(tmpfile.name)
            self.assertEqual(
                set(d.keys()),
                {'class', 'hyperparameters', 'state',
                 'optimizer', 'scheduler'}
            )
            self.assertEqual(
                set(d['optimizer'].keys()),
                {'class', 'hyperparameters', 'state'}
            )
            self.assertEqual(
                set(d['scheduler'].keys()),
                {'class', 'hyperparameters', 'state'}
            )

class TestSavedCheckpoint(unittest.TestCase):
    def setUp(self):
        self.tmpfile = tempfile.NamedTemporaryFile()
        self.init_c = get_initial_checkpoint()
        self.init_c.save_torch_checkpoint(self.tmpfile.name)

    def tearDown(self):
        self.tmpfile.close()

    def test_init_checkpoint(self):
        saved_checkpoint = SavedCheckpoint(self.tmpfile.name)

        saved_torch_model = saved_checkpoint.get_torch_model()
        torch_model = self.init_c.get_torch_model()
        saved_torch_optimizer = saved_checkpoint.get_torch_optimizer()
        torch_optimizer = self.init_c.get_torch_optimizer()
        saved_torch_scheduler = saved_checkpoint.get_torch_scheduler()
        torch_scheduler = self.init_c.get_torch_scheduler()

        # get types of torch objects
        self.assertEqual(type(saved_torch_model),
                         type(torch_model))
        self.assertEqual(type(saved_torch_optimizer),
                         type(torch_optimizer))
        self.assertEqual(type(saved_torch_scheduler),
                         type(torch_scheduler))

        # get class and hyperparameters for each component
        self.assertEqual(saved_checkpoint.get_model_class(),
                         self.init_c.get_model_class())
        self.assertEqual(saved_checkpoint.get_model_hyperparameters(),
                         self.init_c.get_model_hyperparameters())
        self.assertEqual(saved_checkpoint.get_optimizer_class(),
                         self.init_c.get_optimizer_class())
        self.assertEqual(saved_checkpoint.get_optimizer_hyperparameters(),
                         self.init_c.get_optimizer_hyperparameters())
        self.assertEqual(saved_checkpoint.get_scheduler_class(),
                         self.init_c.get_scheduler_class())
        self.assertEqual(saved_checkpoint.get_scheduler_hyperparameters(),
                         self.init_c.get_scheduler_hyperparameters())

    def test_serialize(self):
        saved_checkpoint = SavedCheckpoint(self.tmpfile.name)

        d = saved_checkpoint.to_dict()
        self.assertEqual(
            d,
            {
                'type': 'saved',
                'args': {
                    'path': self.tmpfile.name,
                }
            }
        )

        m_d = Checkpoint.from_dict(d)
        self.assertEqual(saved_checkpoint.to_dict(), m_d.to_dict(),)

class TestCheckpointModel(unittest.TestCase):
    def test_init_model(self):
        checkpoint_model = CheckpointModel(get_initial_checkpoint())
        checkpoint_torch_model = checkpoint_model.get_torch_model()
        torch_model = get_initial_model().get_torch_model()
        self.assertEqual(type(checkpoint_torch_model), type(torch_model))
        self.assertEqual(checkpoint_model.get_class(),
                         get_initial_model().get_class())
        self.assertEqual(checkpoint_model.get_hyperparameters(),
                         get_initial_model().get_hyperparameters())

    def test_serialize_model(self):
        checkpoint_model = CheckpointModel(get_initial_checkpoint())
        d = checkpoint_model.to_dict()
        self.assertEqual(
            d,
            {
                'type': 'checkpoint',
                'args': {
                    'checkpoint': {
                        'type': 'initial',
                        'args': {
                            'model': {
                                'type': 'initial',
                                'args': {
                                    'class': 'baseline_encdec',
                                    'hyperparameters': {
                                        'base': {
                                            'in_channel': 2,
                                            'out_channel': 3,
                                            'n_lstm': 2,
                                            'lstm_channel': 60,
                                            'n_fft': 128,
                                            'hop_length': 32,
                                            'magbook_size': 1,
                                            'phasebook_size': 1,
                                            'output_residual': False,
                                        }
                                    },
                                    'seed': 1410343,
                                },
                            },
                            'optimizer_class': 'adam',
                            'optimizer_args': {'lr': 0.1},
                            'scheduler_class': 'reduce_on_plateau',
                            'scheduler_args': {'factor': 0.5, 'patience': 5},
                            'seed': 1410343
                        }
                    }
                }
            }
        )

        m_d = Model.from_dict(d)
        self.assertEqual(checkpoint_model.to_dict(), m_d.to_dict(),)

