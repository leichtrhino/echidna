
from .core import (
    BaselineEncoder,
    BaselineDecoder,
    WaveUNetEncoder,
    WaveUNetDecoder,
    ConvTasNetEncoder,
    ConvTasNetDecoder,
    DemucsEncoder,
    DemucsDecoder
)
from .multidomain import (
    EncDecModel,
    ChimeraNet,
)

def BaselineEncDecModel(hyperparameters : dict):
    """
    """
    return EncDecModel(BaselineEncoder,
                       BaselineDecoder,
                       hyperparameters)

def BaselineChimeraNet(hyperparameters : dict):
    """
    """
    return ChimeraNet(BaselineEncoder,
                      BaselineDecoder,
                      hyperparameters)

