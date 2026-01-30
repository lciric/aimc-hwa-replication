#!/usr/bin/env python3
"""
Train LSTM language model on WikiText-2 with Hardware-Aware Training.

Two-phase training:
    Phase 1: Digital warmup - 5 epochs
    Phase 2: HWA fine-tuning - 5 epochs with PCM noise

Expected results (matching paper):
    - Digital baseline: ~330 PPL
    - HWA @ t=1s: ~260 PPL  
    - HWA @ t=1yr: ~260 PPL (drift stable with GDC)

Usage:
    python scripts/train_lstm.py
    python scripts/train_lstm.py --resume checkpoints/lstm_digital.pt
"""

import argparse
import logging
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import (
    PCMPhysicsEngine, create_lstm_lm, compute_gdc_factor,
    WikiText2Corpus, batchify, get_lm_batch,
    set_inference_time
)
from src.layers import AnalogLinear

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='HWA Training for LSTM LM')
    
    # Training
    parser.add_argument('--epochs-digital', type=int, default=5)
    parser.add_argument('--epochs-hwa', type=int, default=5)
    parser.add_argument('--lr', type=float, default=20.0)
    parser.add_argument('--batch-size', type=int, default=20)
    parser.add_argument('--bptt', type=int, default=35)
    parser.add_argument('--clip', type=float, default=0.25)
    
    # Model
    parser.add_argument('--embed-size', type=int, default=200)
    parser.add_argument('--hidden-size', type=int, default=200)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.5)
    
    # Checkpoints
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    
    # Hardware
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    
    return parser.parse_args()


def train_epoch(model, train_data, criterion, lr, bptt, clip, ntokens, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    hidden = model.init_hidden(train_data.size(1))
    start_time = time.time()
    
    for batch_idx, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_lm_batch(train_data, i, bptt)
        
        # Detach hidden state from history
        hidden = tuple(h.detach() for h in hidden) if isinstance(hidden, tuple) else hidden.detach()
        
        model.zero_grad()
        output, hidden = model(data, hidden)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()
        
        # Gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        # Manual SGD update
        for p in model.parameters():
            if p.grad is not None:
                p.data.add_(p.grad, alpha=-lr)
        
        total_loss += loss.item()
        
        # Progress logging
        if batch_idx % 200 == 0 and batch_idx > 0:
            cur_loss = total_loss / 200
            elapsed = time.time() - start_time
            logger.info(f'  batch {batch_idx:5d} | loss {cur_loss:.2f} | '
                       f'ppl {math.exp(cur_loss):.2f} | ms/batch {elapsed*1000/200:.1f}')
            total_loss = 0
            start_time = time.time()


def evaluate(model, data_source, criterion, bptt, ntokens):
    """Evaluate perplexity on data."""
    model.eval()
    total_loss = 0.0
    hidden = model.init_hidden(data_source.size(1))
    
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_lm_batch(data_source, i, bptt)
            output, hidden = model(data, hidden)
            hidden = tuple(h.detach() for h in hidden)
            total_loss += len(data) * criterion(output.view(-1, ntokens), targets).item()
    
    return total_loss / (len(data_source) - 1)


def evaluate_drift_with_gdc(model, test_data, criterion, bptt, ntokens, 
                            t_inference, physics):
    """
    Evaluate with drift and GDC compensation.
    
    GDC hooks amplify layer outputs to compensate for conductance decay.
    """
    model.eval()
    
    # GDC factor for this time
    t0 = physics.T0
    nu = physics.DRIFT_NU
    gdc = 1.0 if t_inference <= t0 else (t_inference / t0) ** nu
    
    # Install GDC hooks on analog layers
    handles = []
    
    def get_gdc_hook(gdc_val):
        def hook(module, inp, out):
            if hasattr(module, 'bias') and module.bias is not None:
                return (out - module.bias) * gdc_val + module.bias
            return out * gdc_val
        return hook
    
    for m in model.modules():
        if isinstance(m, AnalogLinear):
            m.set_inference_time(t_inference)
            handles.append(m.register_forward_hook(get_gdc_hook(gdc)))
    
    # Evaluate
    total_loss = 0.0
    hidden = model.init_hidden(test_data.size(1))
    
    with torch.no_grad():
        for i in range(0, test_data.size(0) - 1, bptt):
            data, targets = get_lm_batch(test_data, i, bptt)
            output, hidden = model(data, hidden)
            hidden = tuple(h.detach() for h in hidden)
            total_loss += len(data) * criterion(output.view(-1, ntokens), targets).item()
    
    # Cleanup hooks
    for h in handles:
        h.remove()
    set_inference_time(model, 0.0)
    
    return total_loss / (len(test_data) - 1), gdc


def main():
    args = parse_args()
    
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    # Create checkpoint directory
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info("Loading WikiText-2...")
    corpus = WikiText2Corpus()
    ntokens = len(corpus.dictionary)
    
    train_data = batchify(corpus.train, args.batch_size, device)
    val_data = batchify(corpus.valid, 10, device)
    test_data = batchify(corpus.test, 10, device)
    
    criterion = nn.CrossEntropyLoss()
    
    # =========================================================================
    # Phase 1: Digital Training
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 1: DIGITAL WARMUP")
    logger.info("=" * 60)
    
    # Digital model (no physics)
    model = create_lstm_lm(ntokens, physics=None, size='small').to(device)
    
    if args.resume:
        logger.info(f"Loading from {args.resume}")
        model.load_state_dict(torch.load(args.resume, map_location=device))
    else:
        lr = args.lr
        best_val_loss = None
        
        for epoch in range(1, args.epochs_digital + 1):
            logger.info(f"\n--- Epoch {epoch}/{args.epochs_digital} ---")
            train_epoch(model, train_data, criterion, lr, args.bptt, 
                       args.clip, ntokens, device)
            
            val_loss = evaluate(model, val_data, criterion, args.bptt, ntokens)
            logger.info(f'Validation PPL: {math.exp(val_loss):.2f}')
            
            if best_val_loss is None or val_loss < best_val_loss:
                torch.save(model.state_dict(), ckpt_dir / 'lstm_digital.pt')
                best_val_loss = val_loss
            else:
                lr /= 4.0
                logger.info(f'LR decay -> {lr}')
        
        model.load_state_dict(torch.load(ckpt_dir / 'lstm_digital.pt'))
    
    test_loss = evaluate(model, test_data, criterion, args.bptt, ntokens)
    logger.info(f"\nDigital Test PPL: {math.exp(test_loss):.2f}")
    
    # =========================================================================
    # Phase 2: HWA Fine-tuning
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 2: HWA FINE-TUNING (PCM noise)")
    logger.info("=" * 60)
    
    # Create analog model with physics
    physics = PCMPhysicsEngine(device=str(device), noise_scale=1.0)
    model_hwa = create_lstm_lm(ntokens, physics=physics, size='small').to(device)
    
    # Warm start from digital
    model_hwa.load_state_dict(model.state_dict(), strict=False)
    
    lr = args.lr / 4  # Lower LR for fine-tuning
    best_val_loss = None
    
    for epoch in range(1, args.epochs_hwa + 1):
        logger.info(f"\n--- HWA Epoch {epoch}/{args.epochs_hwa} ---")
        train_epoch(model_hwa, train_data, criterion, lr, args.bptt,
                   args.clip, ntokens, device)
        
        # Evaluate without noise
        physics.set_noise_scale(0.0)
        val_loss = evaluate(model_hwa, val_data, criterion, args.bptt, ntokens)
        physics.set_noise_scale(1.0)
        
        logger.info(f'Validation PPL: {math.exp(val_loss):.2f}')
        
        if best_val_loss is None or val_loss < best_val_loss:
            torch.save(model_hwa.state_dict(), ckpt_dir / 'lstm_hwa.pt')
            best_val_loss = val_loss
        else:
            lr /= 4.0
            logger.info(f'LR decay -> {lr}')
    
    model_hwa.load_state_dict(torch.load(ckpt_dir / 'lstm_hwa.pt'))
    
    # =========================================================================
    # Drift Evaluation
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("DRIFT ANALYSIS WITH GDC")
    logger.info("=" * 60)
    
    drift_times = [
        (1, "1 sec"),
        (3600, "1 hour"),
        (86400, "1 day"),
        (365 * 24 * 3600, "1 year")
    ]
    
    physics.set_noise_scale(1.0)  # Enable noise for inference
    
    results = []
    for t, label in drift_times:
        loss, gdc = evaluate_drift_with_gdc(
            model_hwa, test_data, criterion, args.bptt, ntokens, t, physics
        )
        ppl = math.exp(loss)
        results.append((label, gdc, ppl))
        logger.info(f"{label:>12} | GDC: {gdc:.4f} | PPL: {ppl:.2f}")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("FINAL RESULTS")
    logger.info("=" * 60)
    baseline_ppl = results[0][2]
    one_year_ppl = results[3][2]
    logger.info(f"Baseline (1 sec): {baseline_ppl:.2f} PPL")
    logger.info(f"1 year with GDC:  {one_year_ppl:.2f} PPL")
    logger.info(f"Drift degradation: +{one_year_ppl - baseline_ppl:.2f} PPL")
    
    if one_year_ppl - baseline_ppl < 1.0:
        logger.info("✓ SUCCESS: Excellent drift stability (<1 PPL degradation)")


if __name__ == '__main__':
    main()
