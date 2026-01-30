#!/usr/bin/env python3
"""
Train BERT on GLUE tasks with Hardware-Aware Training.

Fine-tunes bert-base-uncased on SST-2 (sentiment classification) with
analog noise injection for PCM deployment.

Expected results (matching paper Table 3):
    - Baseline accuracy: ~92-93% on SST-2 validation
    - Drift @ 1 year with GDC: ~92-93% (minimal degradation)

Usage:
    python scripts/train_bert.py
    python scripts/train_bert.py --task mrpc --epochs 5
    python scripts/train_bert.py --eval-only --checkpoint bert_hwa.pth
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='HWA Training for BERT')
    
    # Task
    parser.add_argument('--task', type=str, default='sst2',
                        choices=['sst2', 'mrpc', 'cola', 'qnli'],
                        help='GLUE task name')
    
    # Training
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--max-length', type=int, default=128)
    
    # HWA parameters
    parser.add_argument('--noise-scale', type=float, default=3.0)
    parser.add_argument('--noise-ramp-epochs', type=int, default=1)
    parser.add_argument('--drop-connect', type=float, default=0.01)
    parser.add_argument('--drift-nu', type=float, default=0.06,
                        help='Drift exponent (0.06 for BERT)')
    
    # Checkpoints
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default='checkpoints')
    parser.add_argument('--eval-only', action='store_true')
    
    # Hardware
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Imports that require transformers/datasets
    try:
        from transformers import BertForSequenceClassification, BertTokenizer
        from transformers import get_linear_schedule_with_warmup
        from datasets import load_dataset
    except ImportError:
        logger.error("Install transformers and datasets: pip install transformers datasets")
        return
    
    from src import (
        PCMPhysicsEngine, NoiseScheduler,
        convert_bert_to_analog, evaluate_bert_drift_stability,
        remap_all_weights, set_drop_connect_prob
    )
    
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # Load Model & Tokenizer
    # =========================================================================
    logger.info(f"Loading BERT for {args.task}...")
    
    # Task-specific num_labels
    num_labels = {'sst2': 2, 'mrpc': 2, 'cola': 2, 'qnli': 2}[args.task]
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased', num_labels=num_labels
    )
    
    # =========================================================================
    # Convert to Analog
    # =========================================================================
    logger.info("Converting to Analog BERT...")
    physics = PCMPhysicsEngine(device=str(device), noise_scale=args.noise_scale)
    model = convert_bert_to_analog(model, physics, drop_connect_prob=args.drop_connect)
    model = model.to(device)
    
    # Apply weight remapping for SNR optimization
    remap_all_weights(model)
    
    # Load checkpoint if provided
    if args.checkpoint:
        logger.info(f"Loading checkpoint: {args.checkpoint}")
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    
    # =========================================================================
    # Load Dataset
    # =========================================================================
    logger.info(f"Loading GLUE/{args.task} dataset...")
    
    dataset = load_dataset('glue', args.task)
    
    # Tokenization function (task-specific)
    if args.task == 'sst2':
        def tokenize_fn(examples):
            return tokenizer(examples['sentence'], 
                           padding='max_length', truncation=True, 
                           max_length=args.max_length)
        text_cols = ['sentence', 'idx']
    elif args.task in ['mrpc', 'qnli']:
        def tokenize_fn(examples):
            return tokenizer(examples['sentence1'], examples['sentence2'],
                           padding='max_length', truncation=True,
                           max_length=args.max_length)
        text_cols = ['sentence1', 'sentence2', 'idx']
    elif args.task == 'cola':
        def tokenize_fn(examples):
            return tokenizer(examples['sentence'],
                           padding='max_length', truncation=True,
                           max_length=args.max_length)
        text_cols = ['sentence', 'idx']
    
    tokenized = dataset.map(tokenize_fn, batched=True)
    tokenized = tokenized.remove_columns(text_cols)
    tokenized = tokenized.rename_column('label', 'labels')
    tokenized.set_format('torch')
    
    train_loader = DataLoader(tokenized['train'], shuffle=True, 
                              batch_size=args.batch_size)
    eval_loader = DataLoader(tokenized['validation'], 
                             batch_size=args.batch_size)
    
    logger.info(f"Train: {len(tokenized['train'])}, Val: {len(tokenized['validation'])}")
    
    # =========================================================================
    # Evaluation Only Mode
    # =========================================================================
    if args.eval_only:
        logger.info("\n" + "=" * 60)
        logger.info("DRIFT EVALUATION")
        logger.info("=" * 60)
        
        results = evaluate_bert_drift_stability(
            model, eval_loader, device,
            drift_nu=args.drift_nu
        )
        
        baseline = results['1 sec']['accuracy']
        one_year = results['1 year']['accuracy']
        logger.info(f"\nBaseline: {baseline:.2f}%")
        logger.info(f"1 year:   {one_year:.2f}%")
        logger.info(f"Drop:     {baseline - one_year:.2f}%")
        return
    
    # =========================================================================
    # Training
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("ANALOG BERT TRAINING")
    logger.info("=" * 60)
    
    optimizer = AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )
    noise_scheduler = NoiseScheduler(
        ramp_epochs=args.noise_ramp_epochs, 
        final_scale=args.noise_scale
    )
    
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        # Update noise scale
        current_noise = noise_scheduler.get_scale(epoch)
        physics.set_noise_scale(current_noise)
        
        # Training
        model.train()
        total_loss = 0
        
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(**batch)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            
            if step % 100 == 0:
                logger.info(f"Epoch {epoch+1} | Step {step:4d} | "
                           f"Loss: {loss.item():.3f} | Noise: {current_noise:.1f}x")
        
        # Evaluation
        model.eval()
        physics.set_noise_scale(0.0)  # Clean eval
        
        correct, total = 0, 0
        with torch.no_grad():
            for batch in eval_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                preds = torch.argmax(model(**batch).logits, dim=-1)
                correct += (preds == batch['labels']).sum().item()
                total += batch['labels'].size(0)
        
        acc = 100.0 * correct / total
        logger.info(f"Epoch {epoch+1} | Val Accuracy: {acc:.2f}%")
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), output_dir / 'bert_hwa_best.pth')
    
    # Save final
    torch.save(model.state_dict(), output_dir / 'bert_hwa_final.pth')
    logger.info(f"Best accuracy: {best_acc:.2f}%")
    
    # =========================================================================
    # Drift Evaluation
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("DRIFT EVALUATION WITH GDC")
    logger.info("=" * 60)
    
    # Load best model
    model.load_state_dict(torch.load(output_dir / 'bert_hwa_best.pth'))
    physics.set_noise_scale(1.0)  # Enable noise for inference
    
    results = evaluate_bert_drift_stability(
        model, eval_loader, device,
        drift_nu=args.drift_nu
    )
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("FINAL RESULTS")
    logger.info("=" * 60)
    baseline = results['1 sec']['accuracy']
    one_year = results['1 year']['accuracy']
    logger.info(f"Baseline (1 sec): {baseline:.2f}%")
    logger.info(f"1 year with GDC:  {one_year:.2f}%")
    logger.info(f"Drift degradation: {baseline - one_year:.2f}%")
    
    if baseline - one_year < 1.0:
        logger.info("✓ SUCCESS: Excellent drift stability (<1% degradation)")


if __name__ == '__main__':
    main()
