"""Main training script for respiratory rate estimation with cross-validation."""

import os
import argparse
import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
from pathlib import Path
from typing import Dict, List
import json

from data.datamodule import BIDMCDataModule, CrossValidationDataModule
from training.trainer import RespiratoryRateEstimator
from utils.metrics import print_metrics


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_callbacks(config: Dict, fold_idx: int = 0) -> List[pl.Callback]:
    """Setup PyTorch Lightning callbacks."""
    callbacks = []
    
    # Model checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints/fold_{fold_idx}",
        filename='{epoch}-{val/mae:.3f}',
        monitor='val/mae',
        mode='min',
        save_top_k=config['logging']['save_top_k'],
        save_last=True,
        verbose=True
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping
    early_stopping = EarlyStopping(
        monitor='val/mae',
        min_delta=config['training']['min_delta'],
        patience=config['training']['patience'],
        verbose=True,
        mode='min'
    )
    callbacks.append(early_stopping)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)
    
    return callbacks


def setup_logger(config: Dict, fold_idx: int = 0) -> TensorBoardLogger:
    """Setup TensorBoard logger."""
    logger = TensorBoardLogger(
        save_dir="logs",
        name=config['logging']['experiment_name'],
        version=f"fold_{fold_idx}",
        log_graph=True
    )
    return logger


def train_single_fold(datamodule: BIDMCDataModule, 
                     fold_idx: int, 
                     config: Dict,
                     stage: str = 'both') -> Dict[str, float]:
    """
    Train model for a single fold.
    
    Args:
        datamodule: DataModule for this fold
        fold_idx: Fold index
        config: Configuration dictionary
        stage: Training stage ('contrastive', 'supervised', 'both')
    
    Returns:
        Dictionary with test results
    """
    print(f"\n{'='*60}")
    print(f"Training Fold {fold_idx + 1} - Stage: {stage}")
    print(f"{'='*60}")
    
    # Setup callbacks and logger
    callbacks = setup_callbacks(config, fold_idx)
    logger = setup_logger(config, fold_idx)
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        accelerator=config['hardware']['accelerator'],
        devices=config['hardware']['devices'],
        precision=config['hardware']['precision'],
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=config['logging']['log_every_n_steps'],
        deterministic=True,
        enable_progress_bar=True,
        enable_model_summary=True
    )
    
    results = {}
    
    if stage in ['contrastive', 'both']:
        # Stage 1: Contrastive pretraining
        print(f"\n--- Stage 1: Contrastive Pretraining ---")
        
        # Set datamodule to contrastive mode
        datamodule.set_mode('contrastive')
        
        # Initialize model for contrastive learning
        model = RespiratoryRateEstimator(
            training_mode='contrastive',
            # Time branch parameters
            time_hidden_size=config['model']['time_branch']['hidden_size'],
            time_num_layers=config['model']['time_branch']['num_layers'],
            # Frequency branch parameters  
            freq_hidden_size=config['model']['freq_branch']['hidden_size'],
            freq_num_layers=config['model']['freq_branch']['num_layers'],
            n_fft=config['model']['freq_branch']['n_fft'],
            n_mels=config['model']['freq_branch']['n_mels'],
            # Fusion parameters
            fusion_hidden_size=config['model']['fusion']['hidden_size'],
            num_attention_heads=config['model']['fusion']['num_heads'],
            dropout=config['model']['time_branch']['dropout'],
            # Data parameters
            fs=config['data']['fs'],
            # Contrastive learning parameters
            temperature=float(config['contrastive']['temperature']),
            # Training parameters
            learning_rate=float(config['training']['learning_rate']),
            weight_decay=float(config['training']['weight_decay'])
        )
        
        # Train contrastive model
        trainer.fit(model, datamodule)
        
        # Save contrastive checkpoint
        contrastive_checkpoint = f"checkpoints/fold_{fold_idx}/contrastive_model.ckpt"
        trainer.save_checkpoint(contrastive_checkpoint)
        
        results['contrastive_epoch'] = trainer.current_epoch
        
        if stage == 'contrastive':
            return results
    
    if stage in ['supervised', 'both']:
        # Stage 2: Supervised fine-tuning
        print(f"\n--- Stage 2: Supervised Fine-tuning ---")
        
        # Set datamodule to supervised mode
        datamodule.set_mode('supervised')
        
        if stage == 'both':
            # Load pretrained contrastive model
            model = RespiratoryRateEstimator.load_from_checkpoint(
                contrastive_checkpoint,
                training_mode='supervised',
                # Time branch parameters
                time_hidden_size=config['model']['time_branch']['hidden_size'],
                time_num_layers=config['model']['time_branch']['num_layers'],
                # Frequency branch parameters  
                freq_hidden_size=config['model']['freq_branch']['hidden_size'],
                freq_num_layers=config['model']['freq_branch']['num_layers'],
                n_fft=config['model']['freq_branch']['n_fft'],
                n_mels=config['model']['freq_branch']['n_mels'],
                # Fusion parameters
                fusion_hidden_size=config['model']['fusion']['hidden_size'],
                num_attention_heads=config['model']['fusion']['num_heads'],
                dropout=config['model']['time_branch']['dropout'],
                # Data parameters
                fs=config['data']['fs'],
                # Training parameters
                learning_rate=float(config['training']['learning_rate']),
                weight_decay=float(config['training']['weight_decay'])
            )
            
            # Optionally freeze encoder for initial fine-tuning
            # model.freeze_encoder()
        else:
            # Initialize fresh model for supervised training
            model = RespiratoryRateEstimator(
                training_mode='supervised',
                # Time branch parameters
                time_hidden_size=config['model']['time_branch']['hidden_size'],
                time_num_layers=config['model']['time_branch']['num_layers'],
                # Frequency branch parameters  
                freq_hidden_size=config['model']['freq_branch']['hidden_size'],
                freq_num_layers=config['model']['freq_branch']['num_layers'],
                n_fft=config['model']['freq_branch']['n_fft'],
                n_mels=config['model']['freq_branch']['n_mels'],
                # Fusion parameters
                fusion_hidden_size=config['model']['fusion']['hidden_size'],
                num_attention_heads=config['model']['fusion']['num_heads'],
                dropout=config['model']['time_branch']['dropout'],
                # Data parameters
                fs=config['data']['fs'],
                # Training parameters
                learning_rate=float(config['training']['learning_rate']),
                weight_decay=float(config['training']['weight_decay'])
            )
        
        # Setup new trainer for supervised training
        supervised_callbacks = setup_callbacks(config, fold_idx)
        supervised_logger = TensorBoardLogger(
            save_dir="logs",
            name=f"{config['logging']['experiment_name']}_supervised",
            version=f"fold_{fold_idx}"
        )
        
        supervised_trainer = pl.Trainer(
            max_epochs=config['training']['max_epochs'],
            accelerator=config['hardware']['accelerator'],
            devices=config['hardware']['devices'],
            precision=config['hardware']['precision'],
            callbacks=supervised_callbacks,
            logger=supervised_logger,
            log_every_n_steps=config['logging']['log_every_n_steps'],
            deterministic=True
        )
        
        # Train supervised model
        supervised_trainer.fit(model, datamodule)
        
        # Test the model
        test_results = supervised_trainer.test(model, datamodule)
        
        # Extract test metrics
        if test_results:
            test_metrics = test_results[0]
            for key, value in test_metrics.items():
                if key.startswith('test/'):
                    metric_name = key.replace('test/', '')
                    results[metric_name] = value
        
        results['supervised_epoch'] = supervised_trainer.current_epoch
        
        # Save best model checkpoint path
        best_checkpoint = supervised_callbacks[0].best_model_path
        results['best_checkpoint'] = best_checkpoint
    
    return results


def run_cross_validation(config: Dict, stage: str = 'both') -> Dict[str, float]:
    """
    Run complete cross-validation experiment.
    
    Args:
        config: Configuration dictionary
        stage: Training stage ('contrastive', 'supervised', 'both')
    
    Returns:
        Average results across all folds
    """
    print(f"\n{'='*80}")
    print(f"Starting {config['cv']['n_folds']}-Fold Cross-Validation")
    print(f"Training Stage: {stage}")
    print(f"{'='*80}")
    
    # Setup cross-validation data module
    cv_datamodule = CrossValidationDataModule(
        data_dir=config['data']['data_dir'],
        window_size_sec=config['data']['window_size'],
        overlap=config['data']['overlap'],
        fs=config['data']['fs'],
        batch_size=config['training']['batch_size'],
        num_workers=4,
        n_folds=config['cv']['n_folds'],
        test_ratio=config['data']['test_ratio'],
        val_ratio=config['data']['val_ratio'],
        random_seed=config['cv']['random_seed'],
        augment_train=True,
        augment_val=False,
        augment_test=False,
        mode='supervised'  # Will be changed during training
    )
    print(cv_datamodule.get_fold_datamodule(0))
    print(stage)
    
    # Run training for all folds
    fold_results = cv_datamodule.run_all_folds(
        train_func=train_single_fold,
        config=config,
        stage=stage
    )
    
    # Calculate average results
    average_results = cv_datamodule.get_average_results()
    
    # Print final results
    print(f"\n{'='*80}")
    print(f"Cross-Validation Results ({config['cv']['n_folds']} folds)")
    print(f"{'='*80}")
    
    # Print individual fold results
    for i, result in enumerate(fold_results):
        print(f"\nFold {i + 1} Results:")
        if isinstance(result, dict):
            for key, value in result.items():
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value:.4f}")
    
    # Print average results
    print(f"\nAverage Results:")
    print("-" * 40)
    important_metrics = ['mae', 'rmse', 'r2', 'clinical_acc_2', 'correlation']
    
    for metric in important_metrics:
        mean_key = f"{metric}_mean"
        std_key = f"{metric}_std"
        if mean_key in average_results:
            mean_val = average_results[mean_key]
            std_val = average_results.get(std_key, 0.0)
            print(f"{metric.upper()}: {mean_val:.4f} ± {std_val:.4f}")
    
    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    results_file = results_dir / f"cv_results_{stage}.json"
    with open(results_file, 'w') as f:
        json.dump({
            'fold_results': fold_results,
            'average_results': average_results,
            'config': config
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    return average_results


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train respiratory rate estimation model')
    parser.add_argument('--config', type=str, default='src/config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--stage', type=str, default='both',
                       choices=['contrastive', 'supervised', 'both'],
                       help='Training stage')
    parser.add_argument('--fold', type=int, default=None,
                       help='Train specific fold only (for debugging)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    pl.seed_everything(args.seed)
    
    # Load configuration
    config = load_config(args.config)
    
    # Override seed if provided
    if args.seed != 42:
        config['cv']['random_seed'] = args.seed
    
    # Create directories
    Path("checkpoints").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    
    if args.fold is not None:
        # Train single fold (for debugging)
        print(f"Training single fold: {args.fold}")
        
        datamodule = BIDMCDataModule(
            data_dir=config['data']['data_dir'],
            window_size_sec=config['data']['window_size'],
            batch_size=config['training']['batch_size'],
            n_folds=config['cv']['n_folds'],
            current_fold=args.fold,
            num_workers=4,
            mode='supervised'
        )
        
        results = train_single_fold(datamodule, args.fold, config, args.stage)
        print(f"\nSingle fold results: {results}")
    else:
        print("in else")
        
        # Run full cross-validation
        results = run_cross_validation(config, args.stage)
        print(f"\nCross-validation completed!")


if __name__ == "__main__":
    main()
