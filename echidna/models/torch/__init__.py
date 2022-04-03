
from .encoderdecoder import EncoderDecoderModel
from .baseline import BaselineEncoder, BaselineDecoder

def BaselineEncoderDecoderModel(hyperparameters : dict):
    """
    """
    return EncoderDecoderModel(BaselineEncoder,
                               BaselineDecoder,
                               hyperparameters)

