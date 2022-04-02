
import unittest
import torch

from echidna.models.multidomain.encdec import EncDecModel
from .utils import ToyEncoder, ToyDecoder

class TestEncDec(unittest.TestCase):
    def test_encoderdecoder(self):
        m = EncDecModel(
            encoder_class=ToyEncoder,
            decoder_class=ToyDecoder,
            hyperparameters={
                'base': {
                    'out_channel': 2,
                }
            }
        )
        target_length = 1000
        input_length = m.reverse_wave_length(target_length)
        output_length = m.forward_wave_length(input_length)
        x = torch.zeros((8, 2, input_length))
        self.assertEqual(m(x)['waves'].shape, (8, 2, output_length))

