#!/usr/bin/env python3
"""
Train WideResNet-16-4 on CIFAR-100 with Hardware-Aware Training.

Two-phase training:
    Phase 1: Teacher (FP32) - 200 epochs
    Phase 2: Student (HWA) - 80 epochs with knowledge distillation

Expected results (matching paper Table 3):
    - Teacher accuracy: ~77-78% on CIFAR-100
    - Student @ t=1s: ~77% with GDC
    - Student @ t=1yr: ~77% with GDC (drift stable)

Usage:
    python scripts/train_wideresnet.py
    python scripts/train_wideresnet.py --teacher-only
    python scripts/train_wideresnet.py --resume checkpoints/teacher_best.pth
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import (
    PCMPhysicsEngine, wideresnet16_4, 
    TrainingConfig, HWATrainer, evaluate_drift_with_gdc,
    get_cifar100_loaders
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='HWA Training for WideResNet')
    
    # Training phases
    parser.add_argument('--teacher-only', action='store_true',
                        help='Only train teacher (Phase 1)')
    parser.add_argument('--student-only', action='store_true',
                        help='Only train student (Phase 2), requires --resume')
    
    # Checkpoints
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Directory for checkpoints')
    
    # Training hyperparameters
    parser.add_argument('--epochs-teacher', type=int, default=200)
    parser.add_argument('--epochs-student', type=int, default=80)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr-teacher', type=float, default=0.1)
    parser.add_argument('--lr-student', type=float, default=0.01)
    
    # HWA parameters
    parser.add_argument('--noise-scale', type=float, default=3.0,
                        help='Final noise multiplier for HWA')
    parser.add_argument('--noise-ramp-epochs', type=int, default=10,
                        help='Epochs to ramp noise from 0 to final')
    parser.add_argument('--drop-connect', type=float, default=0.01,
                        help='Drop-connect probability (0.01 = 1%)')
    
    # Knowledge distillation
    parser.add_argument('--distill-temp', type=float, default=4.0,
                        help='Temperature for distillation')
    parser.add_argument('--distill-alpha', type=float, default=0.9,
                        help='Weight for distillation loss (vs hard labels)')
    
    # Hardware
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Build config
    config = TrainingConfig(
        epochs_teacher=args.epochs_teacher,
        epochs_student=args.epochs_student,
        batch_size=args.batch_size,
        lr_teacher=args.lr_teacher,
        lr_student=args.lr_student,
        distill_temp=args.distill_temp,
        distill_alpha=args.distill_alpha,
        noise_scale=args.noise_scale,
        noise_ramp_epochs=args.noise_ramp_epochs,
        drop_connect_prob=args.drop_connect,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        seed=args.seed,
    )
    
    # Data loaders
    logger.info("Loading CIFAR-100...")
    train_loader, val_loader, test_loader = get_cifar100_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Initialize trainer
    trainer = HWATrainer(config)
    
    # =========================================================================
    # Phase 1: Teacher Training
    # =========================================================================
    if not args.student_only:
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 1: TEACHER TRAINING (FP32)")
        logger.info("=" * 60)
        
        # Teacher: standard WideResNet (no physics)
        teacher = wideresnet16_4(num_classes=100, physics=None)
        
        if args.resume and not args.teacher_only:
            logger.info(f"Loading teacher from {args.resume}")
            teacher.load_state_dict(torch.load(args.resume, map_location=device))
        else:
            teacher = trainer.train_teacher(teacher, train_loader, val_loader)
        
        # Save teacher
        torch.save(teacher.state_dict(), 
                   Path(config.checkpoint_dir) / 'teacher_final.pth')
        logger.info("Teacher saved to checkpoints/teacher_final.pth")
        
        if args.teacher_only:
            logger.info("Teacher-only mode: exiting after Phase 1")
            return
    else:
        # Load teacher for student training
        if args.resume is None:
            raise ValueError("--student-only requires --resume <teacher_checkpoint>")
        
        teacher = wideresnet16_4(num_classes=100, physics=None)
        teacher.load_state_dict(torch.load(args.resume, map_location=device))
        logger.info(f"Loaded teacher from {args.resume}")
    
    # =========================================================================
    # Phase 2: Student HWA Training
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 2: STUDENT HWA TRAINING")
    logger.info("=" * 60)
    
    # Create physics engine for student
    physics = PCMPhysicsEngine(device=args.device, noise_scale=0.0)
    
    # Student: WideResNet with analog physics
    student = wideresnet16_4(num_classes=100, physics=physics,
                             drop_connect_prob=args.drop_connect)
    
    # Train student with distillation
    student = trainer.train_student(
        teacher, student, train_loader, val_loader, physics
    )
    
    # Save final student
    torch.save(student.state_dict(),
               Path(config.checkpoint_dir) / 'student_final.pth')
    logger.info("Student saved to checkpoints/student_final.pth")
    
    # =========================================================================
    # Drift Evaluation
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("DRIFT EVALUATION WITH GDC")
    logger.info("=" * 60)
    
    drift_results = evaluate_drift_with_gdc(
        student, test_loader, physics, device,
        drift_times=[
            (1, "1 sec"),
            (3600, "1 hour"),
            (86400, "1 day"),
            (365 * 24 * 3600, "1 year")
        ],
        n_samples=3
    )
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("FINAL RESULTS")
    logger.info("=" * 60)
    baseline = drift_results["1 sec"]["accuracy_mean"]
    one_year = drift_results["1 year"]["accuracy_mean"]
    logger.info(f"Baseline (1 sec): {baseline:.2f}%")
    logger.info(f"1 year with GDC:  {one_year:.2f}%")
    logger.info(f"Drift degradation: {one_year - baseline:+.2f}%")
    
    if one_year >= 72.0:
        logger.info("✓ SUCCESS: Meets SOTA threshold (>72% at 1 year)")
    else:
        logger.info("✗ Below SOTA threshold")


if __name__ == '__main__':
    main()
