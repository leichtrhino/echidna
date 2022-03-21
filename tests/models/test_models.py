
import inspect
import unittest
import tempfile
import torch

from echidna.models.nest import BaselineEncDecModel
from echidna.models.multidomain.encdec import EncDecModel
from echidna.models.models import (
    Model,
    InitialModel,
    SavedModel,
    CheckpointModel,
)
from echidna.models.checkpoints import (
    Checkpoint,
    InitialCheckpoint,
)

from .utils import get_initial_model, get_initial_checkpoint

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
                {'class', 'hyperparameters', 'epoch', 'state'}
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

class TestCheckpointModel(unittest.TestCase):
    def test_init_model(self):
        checkpoint = get_initial_checkpoint()
        if 'metrics' in inspect.signature(
                checkpoint.get_torch_scheduler().step).parameters:
            checkpoint.get_torch_scheduler().step(metrics=0)
        else:
            checkpoint.get_torch_scheduler().step()

        checkpoint_model = CheckpointModel(checkpoint)
        checkpoint_torch_model = checkpoint_model.get_torch_model()
        torch_model = get_initial_model().get_torch_model()
        self.assertEqual(type(checkpoint_torch_model), type(torch_model))
        self.assertEqual(checkpoint_model.get_class(),
                         get_initial_model().get_class())
        self.assertEqual(checkpoint_model.get_hyperparameters(),
                         get_initial_model().get_hyperparameters())
        self.assertEqual(checkpoint_model.get_epoch(), 1)

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

