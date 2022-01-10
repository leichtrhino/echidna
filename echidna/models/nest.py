
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

def WaveUNetEncDecModel(hyperparameters : dict):
    """
    """
    return EncDecModel(WaveUNetEncoder,
                       WaveUNetDecoder,
                       hyperparameters)

def WaveUNetChimeraNet(hyperparameters : dict):
    """
    """
    return ChimeraNet(WaveUNetEncoder,
                      WaveUNetDecoder,
                      hyperparameters)

def ConvTasNetEncDecModel(hyperparameters : dict):
    """
    """
    return EncDecModel(ConvTasNetEncoder,
                       ConvTasNetDecoder,
                       hyperparameters)

def ConvTasNetChimeraNet(hyperparameters : dict):
    """
    """
    return ChimeraNet(ConvTasNetEncoder,
                      ConvTasNetDecoder,
                      hyperparameters)

def DemucsEncDecModel(hyperparameters : dict):
    """
    """
    return EncDecModel(DemucsEncoder,
                       DemucsDecoder,
                       hyperparameters)

def DemucsChimeraNet(hyperparameters : dict):
    """
    """
    return ChimeraNet(DemucsEncoder,
                      DemucsDecoder,
                      hyperparameters)
