"""
Hardware-Aware Training for Analog In-Memory Computing.

Replication of: Rasch et al. "Hardware-aware training for large-scale and 
diverse deep learning inference on analog in-memory computing"
Nature Electronics, 2023. arXiv:2302.08469

This package provides:
    - PCM physics simulation (programming noise, drift)
    - Analog neural network layers (AnalogLinear, AnalogConv2d)
    - HWA training techniques (noise ramping, drop-connect, caws)
    - Knowledge distillation training framework
    - Drift evaluation with Global Drift Compensation (GDC)

Quick start:
    from src import PCMPhysicsEngine, wideresnet16_4, HWATrainer
    
    physics = PCMPhysicsEngine(noise_scale=1.0)
    model = wideresnet16_4(physics=physics)
"""

__version__ = '0.1.0'
__author__ = 'HWA Research'

from .physics import PCMPhysicsEngine, NoiseScheduler, compute_gdc_factor
from .layers import (
    AnalogLinear, AnalogConv2d, StraightThroughEstimator,
    apply_caws, remap_all_weights, set_drop_connect_prob, set_inference_time
)
from .models import (
    AnalogLSTM, create_lstm_lm,
    WideResNet, wideresnet16_4, wideresnet28_10,
    convert_bert_to_analog, evaluate_bert_with_gdc, 
    evaluate_bert_drift_stability, AnalogBertConfig
)
from .training import TrainingConfig, HWATrainer, evaluate_drift_with_gdc
from .data import (
    get_cifar100_loaders, 
    WikiText2Corpus, batchify, get_lm_batch
)

__all__ = [
    # Version
    '__version__',
    
    # Physics
    'PCMPhysicsEngine', 'NoiseScheduler', 'compute_gdc_factor',
    
    # Layers
    'AnalogLinear', 'AnalogConv2d', 'StraightThroughEstimator',
    'apply_caws', 'remap_all_weights', 'set_drop_connect_prob', 'set_inference_time',
    
    # Models - LSTM
    'AnalogLSTM', 'create_lstm_lm',
    # Models - WideResNet
    'WideResNet', 'wideresnet16_4', 'wideresnet28_10',
    # Models - BERT
    'convert_bert_to_analog', 'evaluate_bert_with_gdc',
    'evaluate_bert_drift_stability', 'AnalogBertConfig',
    
    # Training
    'TrainingConfig', 'HWATrainer', 'evaluate_drift_with_gdc',
    
    # Data
    'get_cifar100_loaders',
    'WikiText2Corpus', 'batchify', 'get_lm_batch',
]
