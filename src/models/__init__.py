"""
Neural network models with analog weight representation.
"""

from .lstm import AnalogLSTM, AnalogLSTMCell, create_lstm_lm
from .wideresnet import WideResNet, wideresnet16_4, wideresnet28_10
from .bert import (
    convert_bert_to_analog, 
    evaluate_bert_with_gdc,
    evaluate_bert_drift_stability,
    AnalogBertConfig,
    count_analog_layers
)

__all__ = [
    # LSTM
    'AnalogLSTM', 'AnalogLSTMCell', 'create_lstm_lm',
    # WideResNet
    'WideResNet', 'wideresnet16_4', 'wideresnet28_10',
    # BERT
    'convert_bert_to_analog', 
    'evaluate_bert_with_gdc',
    'evaluate_bert_drift_stability',
    'AnalogBertConfig',
    'count_analog_layers',
]
