"""Metrics calculation utilities for respiratory rate estimation."""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torchmetrics


class RespiratoryRateMetrics:
    """Comprehensive metrics for respiratory rate estimation."""
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.reset()
    
    def reset(self):
        """Reset all accumulated values."""
        self.predictions = []
        self.targets = []
    
    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """Update metrics with new predictions and targets."""
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        
        # Flatten if needed
        preds = preds.flatten()
        targets = targets.flatten()
        
        # Filter out invalid values
        mask = (targets > 0) & (targets < 100) & ~np.isnan(preds) & ~np.isnan(targets)
        preds = preds[mask]
        targets = targets[mask]
        
        self.predictions.extend(preds.tolist())
        self.targets.extend(targets.tolist())
    
    def compute(self) -> Dict[str, float]:
        """Compute all metrics."""
        if len(self.predictions) == 0:
            return self._empty_metrics()
        
        preds = np.array(self.predictions)
        targets = np.array(self.targets)
        
        # Basic regression metrics
        mae = mean_absolute_error(targets, preds)
        mse = mean_squared_error(targets, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(targets, preds)
        
        # Percentage error metrics
        mape = np.mean(np.abs((targets - preds) / (targets + 1e-8))) * 100
        
        # Clinical accuracy metrics
        mae_clinical = self._clinical_accuracy(preds, targets, threshold=2.0)  # Within 2 breaths/min
        mae_clinical_5 = self._clinical_accuracy(preds, targets, threshold=5.0)  # Within 5 breaths/min
        
        # Correlation
        correlation = np.corrcoef(preds, targets)[0, 1] if len(preds) > 1 else 0.0
        
        # Range-specific metrics
        normal_mask = (targets >= 12) & (targets <= 20)  # Normal respiratory rate
        abnormal_mask = ~normal_mask
        
        normal_mae = mae if not np.any(normal_mask) else mean_absolute_error(
            targets[normal_mask], preds[normal_mask]
        )
        abnormal_mae = mae if not np.any(abnormal_mask) else mean_absolute_error(
            targets[abnormal_mask], preds[abnormal_mask]
        )
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'correlation': correlation,
            'clinical_acc_2': mae_clinical,
            'clinical_acc_5': mae_clinical_5,
            'normal_mae': normal_mae,
            'abnormal_mae': abnormal_mae,
            'n_samples': len(preds)
        }
    
    def _clinical_accuracy(self, preds: np.ndarray, targets: np.ndarray, 
                          threshold: float) -> float:
        """Calculate clinical accuracy (percentage within threshold)."""
        errors = np.abs(preds - targets)
        return np.mean(errors <= threshold) * 100
    
    def _empty_metrics(self) -> Dict[str, float]:
        """Return empty metrics dictionary."""
        return {
            'mae': float('inf'),
            'mse': float('inf'),
            'rmse': float('inf'),
            'r2': 0.0,
            'mape': float('inf'),
            'correlation': 0.0,
            'clinical_acc_2': 0.0,
            'clinical_acc_5': 0.0,
            'normal_mae': float('inf'),
            'abnormal_mae': float('inf'),
            'n_samples': 0
        }


class LossFunction(nn.Module):
    """Combined loss function for respiratory rate estimation."""
    
    def __init__(self, 
                 mse_weight: float = 1.0,
                 mae_weight: float = 0.5,
                 huber_weight: float = 0.3,
                 range_penalty_weight: float = 0.1):
        super().__init__()
        self.mse_weight = mse_weight
        self.mae_weight = mae_weight
        self.huber_weight = huber_weight
        self.range_penalty_weight = range_penalty_weight
        
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.huber_loss = nn.SmoothL1Loss(beta=1.0)
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute combined loss."""
        # Basic losses
        mse = self.mse_loss(predictions, targets)
        mae = self.mae_loss(predictions, targets)
        huber = self.huber_loss(predictions, targets)
        
        # Range penalty (penalize predictions outside physiological range)
        range_penalty = self._range_penalty(predictions)
        
        # Combined loss
        total_loss = (self.mse_weight * mse + 
                     self.mae_weight * mae + 
                     self.huber_weight * huber + 
                     self.range_penalty_weight * range_penalty)
        
        return {
            'total_loss': total_loss,
            'mse_loss': mse,
            'mae_loss': mae,
            'huber_loss': huber,
            'range_penalty': range_penalty
        }
    
    def _range_penalty(self, predictions: torch.Tensor) -> torch.Tensor:
        """Penalty for predictions outside physiological range (5-50 breaths/min)."""
        min_rate, max_rate = 5.0, 50.0
        
        # Penalty for values below minimum
        below_penalty = torch.relu(min_rate - predictions)
        
        # Penalty for values above maximum
        above_penalty = torch.relu(predictions - max_rate)
        
        return torch.mean(below_penalty**2 + above_penalty**2)


class ContrastiveLossFunction(nn.Module):
    """Loss function for contrastive learning phase."""
    
    def __init__(self, 
                 infonce_weight: float = 1.0,
                 temporal_weight: float = 0.5,
                 temperature: float = 0.07):
        super().__init__()
        self.infonce_weight = infonce_weight
        self.temporal_weight = temporal_weight
        self.temperature = temperature
    
    def infonce_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """InfoNCE loss for contrastive learning."""
        batch_size = z1.shape[0]
        
        # Normalize features
        z1 = nn.functional.normalize(z1, dim=-1)
        z2 = nn.functional.normalize(z2, dim=-1)
        
        # Concatenate both views
        z = torch.cat([z1, z2], dim=0)  # (2*batch_size, dim)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(z, z.T) / self.temperature
        
        # Create positive pairs mask
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        sim_matrix = sim_matrix[~mask].view(2 * batch_size, -1)
        
        # Positive pairs are (i, i+batch_size) and (i+batch_size, i)
        pos_indices = torch.cat([
            torch.arange(batch_size, 2 * batch_size),
            torch.arange(0, batch_size)
        ]).to(z.device)
        
        # --- DEBUG CHECKS ---
        if torch.any(pos_indices >= sim_matrix.shape[1]) or torch.any(pos_indices < 0):
            print(f"[InfoNCE] pos_indices out of bounds! pos_indices: {pos_indices}")
            print(f"[InfoNCE] sim_matrix shape: {sim_matrix.shape}")
            raise RuntimeError("InfoNCE: pos_indices out of bounds!")
        if sim_matrix.shape[0] != 2 * batch_size:
            print(f"[InfoNCE] sim_matrix shape mismatch: {sim_matrix.shape}, batch_size: {batch_size}")
            raise RuntimeError("InfoNCE: sim_matrix shape mismatch!")
        
        # Extract positive similarities
        pos_sim = sim_matrix[torch.arange(2 * batch_size), pos_indices].view(-1, 1)
        
        # Compute InfoNCE loss
        logits = torch.cat([pos_sim, sim_matrix], dim=1)
        labels = torch.zeros(2 * batch_size, dtype=torch.long, device=z.device)
        
        loss = nn.functional.cross_entropy(logits, labels)
        return loss
    
    def temporal_contrastive_loss(self, features: torch.Tensor, 
                                 labels: torch.Tensor) -> torch.Tensor:
        """Temporal contrastive loss based on respiratory rate similarity."""
        batch_size = features.shape[0]
        
        # Normalize features
        features = nn.functional.normalize(features, dim=-1)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Create positive pairs based on respiratory rate similarity
        rate_diff = torch.abs(labels.unsqueeze(0) - labels.unsqueeze(1))
        pos_mask = (rate_diff <= 2.0) & (rate_diff > 0)  # Within 2 breaths/min, exclude self
        
        if not pos_mask.any():
            return torch.tensor(0.0, device=features.device)
        
        # Positive and negative similarities
        pos_sim = sim_matrix[pos_mask]
        neg_mask = ~pos_mask
        neg_mask.fill_diagonal_(False)  # Exclude self-similarity
        neg_sim = sim_matrix[neg_mask]
        
        # Contrastive loss
        pos_loss = -torch.log(torch.exp(pos_sim).mean() + 1e-8)
        neg_loss = torch.log(torch.exp(neg_sim).mean() + 1e-8)
        
        return pos_loss + neg_loss
    
    def forward(self, z1: torch.Tensor, z2: torch.Tensor, 
                features: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Compute contrastive losses."""
        # InfoNCE loss
        infonce = self.infonce_loss(z1, z2)
        
        total_loss = self.infonce_weight * infonce
        
        losses = {
            'total_contrastive_loss': total_loss,
            'infonce_loss': infonce
        }
        
        # Temporal contrastive loss if features and labels provided
        if features is not None and labels is not None:
            temporal = self.temporal_contrastive_loss(features, labels)
            total_loss = total_loss + self.temporal_weight * temporal
            losses['total_contrastive_loss'] = total_loss
            losses['temporal_loss'] = temporal
        
        return losses


def compute_metrics_from_outputs(predictions: List[torch.Tensor], 
                                targets: List[torch.Tensor]) -> Dict[str, float]:
    """Compute metrics from lists of predictions and targets."""
    metrics = RespiratoryRateMetrics()
    
    for pred, target in zip(predictions, targets):
        metrics.update(pred, target)
    
    return metrics.compute()


def print_metrics(metrics: Dict[str, float], prefix: str = ""):
    """Pretty print metrics."""
    if prefix:
        print(f"\n{prefix} Metrics:")
    else:
        print("\nMetrics:")
    
    print("-" * 40)
    
    # Primary metrics
    print(f"MAE:           {metrics.get('mae', 0):.3f} breaths/min")
    print(f"RMSE:          {metrics.get('rmse', 0):.3f} breaths/min")
    print(f"R²:            {metrics.get('r2', 0):.3f}")
    print(f"Correlation:   {metrics.get('correlation', 0):.3f}")
    
    # Clinical accuracy
    print(f"Clinical Acc (±2): {metrics.get('clinical_acc_2', 0):.1f}%")
    print(f"Clinical Acc (±5): {metrics.get('clinical_acc_5', 0):.1f}%")
    
    # Range-specific
    if metrics.get('normal_mae', float('inf')) != float('inf'):
        print(f"Normal Range MAE:   {metrics.get('normal_mae', 0):.3f}")
    if metrics.get('abnormal_mae', float('inf')) != float('inf'):
        print(f"Abnormal Range MAE: {metrics.get('abnormal_mae', 0):.3f}")
    
    print(f"Samples:       {metrics.get('n_samples', 0)}")
    print("-" * 40)


if __name__ == "__main__":
    # Test metrics
    torch.manual_seed(42)
    
    # Simulate some predictions and targets
    batch_size = 100
    targets = torch.normal(16.0, 3.0, (batch_size,))  # Around 16 breaths/min
    predictions = targets + torch.normal(0.0, 2.0, (batch_size,))  # Add some noise
    
    # Ensure positive values
    targets = torch.clamp(targets, 5, 40)
    predictions = torch.clamp(predictions, 5, 40)
    
    print("Testing RespiratoryRateMetrics...")
    metrics = RespiratoryRateMetrics()
    metrics.update(predictions, targets)
    result = metrics.compute()
    print_metrics(result, "Test")
    
    print("\nTesting LossFunction...")
    loss_fn = LossFunction()
    loss_dict = loss_fn(predictions, targets)
    print(f"Total Loss: {loss_dict['total_loss'].item():.4f}")
    print(f"MSE Loss: {loss_dict['mse_loss'].item():.4f}")
    print(f"MAE Loss: {loss_dict['mae_loss'].item():.4f}")
    
    print("\nTesting ContrastiveLossFunction...")
    contrastive_loss_fn = ContrastiveLossFunction()
    z1 = torch.randn(batch_size, 128)
    z2 = torch.randn(batch_size, 128)
    features = torch.randn(batch_size, 256)
    
    contrastive_losses = contrastive_loss_fn(z1, z2, features, targets)
    print(f"Total Contrastive Loss: {contrastive_losses['total_contrastive_loss'].item():.4f}")
    print(f"InfoNCE Loss: {contrastive_losses['infonce_loss'].item():.4f}")
    if 'temporal_loss' in contrastive_losses:
        print(f"Temporal Loss: {contrastive_losses['temporal_loss'].item():.4f}")
    
    print("\nMetrics testing completed!")
