
import unittest
import torch

from echidna.models.encoderdecoder import EncoderDecoderModel
from .utils import ToyEncoder, ToyDecoder

class TestEncoderDecoder(unittest.TestCase):
    def test_encoderdecoder(self):
        m = EncoderDecoderModel(
            encoder_class=ToyEncoder,
            decoder_class=ToyDecoder,
            base_hyperparameters=dict(
                out_channel=2
            )
        )
        target_length = 1000
        input_length = m.reverse_wave_length(target_length)
        output_length = m.forward_wave_length(input_length)
        x = torch.zeros((8, 2, input_length))
        self.assertEqual(m(x).shape, (8, 2, output_length))

