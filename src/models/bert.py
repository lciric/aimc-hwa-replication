"""
Analog BERT for Natural Language Processing.

Converts HuggingFace BERT to analog by replacing nn.Linear with AnalogLinear.
Used for GLUE benchmark experiments (SST-2, MRPC, etc.).

Key insight from the paper: Transformers are robust to analog noise because
attention softmax and layer norms provide inherent noise tolerance.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple
import copy

from ..layers import AnalogLinear, set_inference_time
from ..physics import PCMPhysicsEngine, compute_gdc_factor


def convert_bert_to_analog(model: nn.Module, 
                           physics: PCMPhysicsEngine,
                           drop_connect_prob: float = 0.01) -> nn.Module:
    """
    Convert a HuggingFace BERT model to analog by replacing Linear layers.
    
    This is a "surgical" conversion that:
    1. Recursively finds all nn.Linear modules
    2. Replaces them with AnalogLinear (preserving weights)
    3. Connects them to the PCM physics engine
    
    Args:
        model: HuggingFace BertForSequenceClassification or similar
        physics: PCM physics engine for noise injection
        drop_connect_prob: probability of weight dropout (0.01 = 1%)
        
    Returns:
        Modified model with analog layers (in-place modification)
        
    Example:
        >>> from transformers import BertForSequenceClassification
        >>> model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        >>> physics = PCMPhysicsEngine(noise_scale=1.0)
        >>> analog_model = convert_bert_to_analog(model, physics)
    """
    replaced_count = 0
    
    def _replace_linear(module: nn.Module, prefix: str = ''):
        nonlocal replaced_count
        
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            
            if isinstance(child, nn.Linear):
                # Create analog equivalent
                analog_layer = AnalogLinear(
                    child.in_features,
                    child.out_features,
                    bias=(child.bias is not None),
                    physics=physics,
                    drop_connect_prob=drop_connect_prob
                )
                
                # Copy pretrained weights (warm start)
                analog_layer.weight.data = child.weight.data.clone()
                if child.bias is not None:
                    analog_layer.bias.data = child.bias.data.clone()
                
                # Replace in module tree
                setattr(module, name, analog_layer)
                replaced_count += 1
                
            else:
                # Recurse into submodules (attention blocks, etc.)
                _replace_linear(child, full_name)
    
    _replace_linear(model)
    print(f"[bert.py] Converted {replaced_count} Linear → AnalogLinear layers")
    
    return model


def count_analog_layers(model: nn.Module) -> int:
    """Count number of AnalogLinear layers in model."""
    return sum(1 for m in model.modules() if isinstance(m, AnalogLinear))


def evaluate_bert_with_gdc(model: nn.Module, 
                           eval_loader,
                           device: torch.device,
                           t_inference: float,
                           drift_nu: float = 0.06) -> float:
    """
    Evaluate BERT accuracy at a specific drift time with GDC.
    
    For BERT, GDC is applied as output logit scaling rather than
    per-layer hooks (simpler and works well for classification).
    
    Args:
        model: Analog BERT model
        eval_loader: DataLoader with tokenized examples
        device: torch device
        t_inference: time since programming in seconds
        drift_nu: drift exponent (0.06 for BERT, slightly different from vision)
        
    Returns:
        Accuracy as percentage
    """
    model.eval()
    
    # Set inference time on all analog layers
    set_inference_time(model, t_inference)
    
    # Compute GDC factor
    t0 = 20.0  # Reference time
    if t_inference <= t0:
        drift_factor = 1.0
        gdc_factor = 1.0
    else:
        drift_factor = (t_inference / t0) ** (-drift_nu)
        gdc_factor = 1.0 / drift_factor  # Compensate
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in eval_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            
            # Apply GDC to logits
            logits = outputs.logits * gdc_factor
            
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == batch["labels"]).sum().item()
            total += batch["labels"].size(0)
    
    # Reset inference time
    set_inference_time(model, 0.0)
    
    return 100.0 * correct / total


def evaluate_bert_drift_stability(model: nn.Module,
                                  eval_loader,
                                  device: torch.device,
                                  drift_times: Optional[list] = None,
                                  drift_nu: float = 0.06) -> Dict[str, Dict]:
    """
    Evaluate BERT accuracy across multiple drift times.
    
    Args:
        model: Analog BERT model  
        eval_loader: Validation DataLoader
        device: torch device
        drift_times: List of (seconds, label) tuples
        drift_nu: drift exponent
        
    Returns:
        Dictionary with results per time point
    """
    if drift_times is None:
        drift_times = [
            (1, "1 sec"),
            (3600, "1 hour"),
            (86400, "1 day"),
            (365 * 24 * 3600, "1 year")
        ]
    
    results = {}
    
    print(f"{'Time':>12} | {'Drift':>10} | {'GDC':>10} | {'Accuracy':>10}")
    print("-" * 50)
    
    for t_sec, label in drift_times:
        # Compute factors for display
        t0 = 20.0
        if t_sec <= t0:
            drift_factor = 1.0
            gdc_factor = 1.0
        else:
            drift_factor = (t_sec / t0) ** (-drift_nu)
            gdc_factor = 1.0 / drift_factor
        
        acc = evaluate_bert_with_gdc(model, eval_loader, device, t_sec, drift_nu)
        
        results[label] = {
            'time_sec': t_sec,
            'drift_factor': drift_factor,
            'gdc_factor': gdc_factor,
            'accuracy': acc
        }
        
        print(f"{label:>12} | {drift_factor:>10.4f} | {gdc_factor:>10.4f} | {acc:>9.2f}%")
    
    return results


class AnalogBertConfig:
    """Configuration for analog BERT training."""
    
    def __init__(
        self,
        task_name: str = "sst2",
        epochs: int = 3,
        batch_size: int = 32,
        learning_rate: float = 2e-5,
        max_length: int = 128,
        noise_scale: float = 3.0,
        noise_ramp_epochs: int = 1,
        drop_connect_prob: float = 0.01,
        drift_nu: float = 0.06,
        seed: int = 42
    ):
        self.task_name = task_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_length = max_length
        self.noise_scale = noise_scale
        self.noise_ramp_epochs = noise_ramp_epochs
        self.drop_connect_prob = drop_connect_prob
        self.drift_nu = drift_nu
        self.seed = seed


# Quick test
if __name__ == '__main__':
    print("[bert.py] Testing BERT conversion...")
    
    # This requires transformers to be installed
    try:
        from transformers import BertForSequenceClassification
        
        # Load base model
        model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased', num_labels=2
        )
        
        # Convert to analog
        physics = PCMPhysicsEngine(device='cpu', noise_scale=1.0)
        analog_model = convert_bert_to_analog(model, physics)
        
        n_analog = count_analog_layers(analog_model)
        print(f"[bert.py] Analog layers: {n_analog}")
        
        # Test forward pass
        dummy_input = {
            'input_ids': torch.randint(0, 1000, (2, 32)),
            'attention_mask': torch.ones(2, 32, dtype=torch.long)
        }
        output = analog_model(**dummy_input)
        print(f"[bert.py] Output shape: {output.logits.shape}")
        
    except ImportError:
        print("[bert.py] transformers not installed, skipping test")
