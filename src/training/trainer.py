"""PyTorch Lightning module for training dual-branch RWKV model."""

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from typing import Dict, Any, Optional, Tuple
import numpy as np

from models.dual_branch_model import DualBranchRWKV
from utils.metrics import RespiratoryRateMetrics, LossFunction, ContrastiveLossFunction, print_metrics


class RespiratoryRateEstimator(pl.LightningModule):
    """
    PyTorch Lightning module for respiratory rate estimation with dual-branch RWKV.
    Supports both contrastive pretraining and supervised fine-tuning.
    """
    
    def __init__(self,
                 # Model parameters
                 time_hidden_size: int = 256,
                 time_num_layers: int = 6,
                 freq_hidden_size: int = 256,
                 freq_num_layers: int = 4,
                 fusion_hidden_size: int = 512,
                 num_attention_heads: int = 8,
                 dropout: float = 0.2,
                 
                 # Data parameters
                 fs: int = 125,
                 n_fft: int = 3750,
                 n_mels: int = 64,
                 
                 # Training parameters
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-5,
                 optimizer: str = 'adamw',
                 scheduler: str = 'plateau',
                 
                 # Loss parameters
                 mse_weight: float = 1.0,
                 mae_weight: float = 0.5,
                 huber_weight: float = 0.3,
                 range_penalty_weight: float = 0.1,
                 
                 # Contrastive learning parameters
                 contrastive_weight: float = 1.0,
                 infonce_weight: float = 1.0,
                 temporal_weight: float = 0.5,
                 temperature: float = 0.07,
                 
                 # Training mode
                 training_mode: str = 'supervised',  # 'contrastive', 'supervised', 'both'
                 
                 # Scheduler parameters
                 plateau_patience: int = 10,
                 plateau_factor: float = 0.5,
                 cosine_t_max: int = 50):
        
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize model
        self.model = DualBranchRWKV(
            time_hidden_size=time_hidden_size,
            time_num_layers=time_num_layers,
            freq_hidden_size=freq_hidden_size,
            freq_num_layers=freq_num_layers,
            fusion_hidden_size=fusion_hidden_size,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            fs=fs,
            n_fft=n_fft,
            n_mels=n_mels
        )
        
        # Initialize loss functions
        self.regression_loss = LossFunction(
            mse_weight=mse_weight,
            mae_weight=mae_weight,
            huber_weight=huber_weight,
            range_penalty_weight=range_penalty_weight
        )
        
        self.contrastive_loss = ContrastiveLossFunction(
            infonce_weight=infonce_weight,
            temporal_weight=temporal_weight,
            temperature=temperature
        )
        
        # Initialize metrics
        self.train_metrics = RespiratoryRateMetrics()
        self.val_metrics = RespiratoryRateMetrics()
        self.test_metrics = RespiratoryRateMetrics()
        
        # Training mode setup
        self.set_training_mode(training_mode)
        
        # Store training outputs for epoch-end processing
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
    
    def set_training_mode(self, mode: str):
        """Set training mode and configure model accordingly."""
        assert mode in ['contrastive', 'supervised', 'both']
        self.hparams.training_mode = mode
        
        if mode == 'contrastive':
            self.model.set_pretraining_mode(True)
        elif mode == 'supervised':
            self.model.set_pretraining_mode(False)
        else:  # both
            self.model.set_pretraining_mode(True)
    
    def forward(self, ppg_signal: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through the model."""
        return self.model(ppg_signal)
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Training step."""
        # Handle different data formats based on training mode
        if self.hparams.training_mode == 'contrastive':
            # Contrastive learning mode - use augmented views
            view1 = batch['view1']
            view2 = batch['view2']
            original = batch['original']
            resp_rate = batch['resp_rate']
            
            # Forward pass on both views and original
            outputs1 = self.forward(view1)
            outputs2 = self.forward(view2)
            outputs_orig = self.forward(original)
            
            total_loss = 0.0
            loss_dict = {}
            
            # Contrastive loss
            contrastive_losses = self.contrastive_loss(
                outputs1['projections'], outputs2['projections'], outputs_orig['projections'], resp_rate
            )
            total_loss += self.hparams.contrastive_weight * contrastive_losses['total_contrastive_loss']
            
            # Log contrastive losses
            for key, value in contrastive_losses.items():
                loss_dict[f'train/{key}'] = value
            
            # Also compute supervised loss on original for monitoring (if in 'both' mode)
            if self.hparams.training_mode == 'both':
                regression_losses = self.regression_loss(
                    outputs_orig['respiratory_rate'].squeeze(), resp_rate
                )
                # Add small weight to supervised loss during contrastive pretraining
                total_loss += 0.1 * regression_losses['total_loss']
                
                for key, value in regression_losses.items():
                    loss_dict[f'train/supervised_{key}'] = value
        
        else:
            # Supervised learning mode
            ppg_signal = batch['ppg']
            resp_rate = batch['resp_rate']
            
            # Forward pass
            outputs = self.forward(ppg_signal)
            
            total_loss = 0.0
            loss_dict = {}
            
            # Supervised loss
            regression_losses = self.regression_loss(outputs['respiratory_rate'].squeeze(), resp_rate)
            total_loss += regression_losses['total_loss']
            
            # Log regression losses
            for key, value in regression_losses.items():
                loss_dict[f'train/{key}'] = value
            
            # Update metrics
            self.train_metrics.update(outputs['respiratory_rate'].squeeze(), resp_rate)
            
        
        loss_dict['train/total_loss'] = total_loss
        
        # Log losses
        self.log_dict(loss_dict, on_step=True, on_epoch=True, prog_bar=True)
        
        # Store outputs for epoch-end processing
        if self.hparams.training_mode == 'contrastive':
            # In contrastive mode, store original predictions for monitoring
            self.training_step_outputs.append({
                'loss': total_loss,
                'predictions': outputs_orig['respiratory_rate'].detach(),
                'targets': resp_rate.detach()
            })
        else:
            # Supervised mode
            self.training_step_outputs.append({
                'loss': total_loss,
                'predictions': outputs['respiratory_rate'].detach(),
                'targets': resp_rate.detach()
            })
        
        return total_loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Validation step."""
        # Validation always uses supervised data format (BIDMCDataset)
        # regardless of training mode, for consistent evaluation
        ppg_signal = batch['ppg']
        resp_rate = batch['resp_rate']
        
        # Forward pass
        outputs = self.forward(ppg_signal)
        
        # Compute regression loss
        regression_losses = self.regression_loss(outputs['respiratory_rate'].squeeze(), resp_rate)
        
        # Update metrics
        self.val_metrics.update(outputs['respiratory_rate'].squeeze(), resp_rate)
        
        # Log validation losses
        val_loss_dict = {}
        for key, value in regression_losses.items():
            val_loss_dict[f'val/{key}'] = value
        
        self.log_dict(val_loss_dict, on_step=False, on_epoch=True, prog_bar=True)
        
        # Store outputs
        self.validation_step_outputs.append({
            'loss': regression_losses['total_loss'],
            'predictions': outputs['respiratory_rate'].detach(),
            'targets': resp_rate.detach()
        })
        
        return regression_losses['total_loss']
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Test step."""
        ppg_signal = batch['ppg']
        resp_rate = batch['resp_rate']
        
        # Forward pass
        outputs = self.forward(ppg_signal)
        
        # Compute regression loss
        regression_losses = self.regression_loss(outputs['respiratory_rate'].squeeze(), resp_rate)
        
        # Update metrics
        self.test_metrics.update(outputs['respiratory_rate'].squeeze(), resp_rate)
        
        # Store outputs
        self.test_step_outputs.append({
            'loss': regression_losses['total_loss'],
            'predictions': outputs['respiratory_rate'].detach(),
            'targets': resp_rate.detach(),
            'subject_ids': batch.get('subject_id', []),
            'outputs': outputs
        })
        
        return regression_losses['total_loss']
    
    def on_train_epoch_end(self):
        """Called at the end of training epoch."""
        if len(self.training_step_outputs) > 0:
            # Compute and log training metrics
            train_metrics = self.train_metrics.compute()
            
            # Log main metrics
            self.log('train/mae', train_metrics['mae'], on_epoch=True)
            self.log('train/rmse', train_metrics['rmse'], on_epoch=True)
            self.log('train/r2', train_metrics['r2'], on_epoch=True)
            self.log('train/clinical_acc_2', train_metrics['clinical_acc_2'], on_epoch=True)
            
            # Reset metrics
            self.train_metrics.reset()
        
        # Clear outputs
        self.training_step_outputs.clear()
    
    def on_validation_epoch_end(self):
        """Called at the end of validation epoch."""
        if len(self.validation_step_outputs) > 0:
            # Compute and log validation metrics
            val_metrics = self.val_metrics.compute()
            
            # Log main metrics
            self.log('val/mae', val_metrics['mae'], on_epoch=True, prog_bar=True)
            self.log('val/rmse', val_metrics['rmse'], on_epoch=True)
            self.log('val/r2', val_metrics['r2'], on_epoch=True, prog_bar=True)
            self.log('val/clinical_acc_2', val_metrics['clinical_acc_2'], on_epoch=True)
            self.log('val/correlation', val_metrics['correlation'], on_epoch=True)
            
            # Reset metrics
            self.val_metrics.reset()
        
        # Clear outputs
        self.validation_step_outputs.clear()
    
    def on_test_epoch_end(self):
        """Called at the end of test epoch."""
        if len(self.test_step_outputs) > 0:
            # Compute and log test metrics
            test_metrics = self.test_metrics.compute()
            
            # Log all test metrics
            for key, value in test_metrics.items():
                self.log(f'test/{key}', value, on_epoch=True)
            
            # Print detailed metrics
            print_metrics(test_metrics, "Test")
            
            # Reset metrics
            self.test_metrics.reset()
        
        # Clear outputs
        self.test_step_outputs.clear()
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizers and schedulers."""
        # Optimizer
        if self.hparams.optimizer.lower() == 'adamw':
            optimizer = optim.AdamW(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay
            )
        elif self.hparams.optimizer.lower() == 'adam':
            optimizer = optim.Adam(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.hparams.optimizer}")
        
        # Scheduler
        if self.hparams.scheduler.lower() == 'plateau':
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=self.hparams.plateau_factor,
                patience=self.hparams.plateau_patience
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val/mae',
                    'interval': 'epoch',
                    'frequency': 1
                }
            }
        elif self.hparams.scheduler.lower() == 'cosine':
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.hparams.cosine_t_max
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1
                }
            }
        else:
            return optimizer
    
    def freeze_encoder(self):
        """Freeze encoder for fine-tuning."""
        self.model.freeze_encoder()
    
    def unfreeze_encoder(self):
        """Unfreeze encoder."""
        self.model.unfreeze_encoder()


if __name__ == "__main__":
    # Test the Lightning module
    model = RespiratoryRateEstimator(
        time_hidden_size=128,  # Smaller for testing
        freq_hidden_size=128,
        fusion_hidden_size=256,
        training_mode='both'
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    batch_size = 4
    seq_len = 3750
    
    # Simulate batch
    batch = {
        'ppg': torch.randn(batch_size, seq_len),
        'resp_rate': torch.rand(batch_size) * 20 + 10,  # 10-30 breaths/min
        'contrastive_view1': torch.randn(batch_size, seq_len),
        'contrastive_view2': torch.randn(batch_size, seq_len)
    }
    
    print(f"Testing forward pass...")
    outputs = model(batch['ppg'])
    print(f"Output keys: {list(outputs.keys())}")
    print(f"Respiratory rate shape: {outputs['respiratory_rate'].shape}")
    
    print(f"Testing training step...")
    loss = model.training_step(batch, 0)
    print(f"Training loss: {loss.item():.4f}")
    
    print(f"Testing validation step...")
    val_loss = model.validation_step(batch, 0)
    print(f"Validation loss: {val_loss.item():.4f}")
    
    print("Lightning module test completed!")
