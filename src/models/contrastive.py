"""Contrastive learning module for self-supervised pretraining."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional


class ContrastiveProjectionHead(nn.Module):
    """Projection head for contrastive learning."""
    
    def __init__(self, input_dim: int, projection_dim: int = 128, hidden_dim: int = 512):
        super().__init__()
        
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project features to contrastive space."""
        return F.normalize(self.projection(x), dim=-1)


class PPGAugmentation(nn.Module):
    """PPG signal augmentation for contrastive learning."""
    
    def __init__(self, fs: int = 125):
        super().__init__()
        self.fs = fs
        
    def add_noise(self, signal: torch.Tensor, noise_std: float = 0.05) -> torch.Tensor:
        """Add Gaussian noise."""
        noise = torch.randn_like(signal) * noise_std
        return signal + noise
    
    def time_shift(self, signal: torch.Tensor, max_shift_sec: float = 1.0) -> torch.Tensor:
        """Apply random time shift."""
        batch_size, seq_len = signal.shape
        max_shift_samples = int(max_shift_sec * self.fs)
        
        augmented_signals = []
        for i in range(batch_size):
            shift = torch.randint(-max_shift_samples, max_shift_samples + 1, (1,)).item()
            if shift > 0:
                shifted = torch.cat([signal[i, shift:], signal[i, :shift]])
            elif shift < 0:
                shifted = torch.cat([signal[i, shift:], signal[i, :shift]])
            else:
                shifted = signal[i]
            augmented_signals.append(shifted)
        
        return torch.stack(augmented_signals)
    
    def amplitude_scale(self, signal: torch.Tensor, 
                       scale_range: Tuple[float, float] = (0.8, 1.2)) -> torch.Tensor:
        """Apply random amplitude scaling."""
        batch_size = signal.shape[0]
        scales = torch.empty(batch_size, 1, device=signal.device).uniform_(
            scale_range[0], scale_range[1]
        )
        return signal * scales
    
    def frequency_mask(self, signal: torch.Tensor, 
                      mask_freq_ratio: float = 0.1) -> torch.Tensor:
        """Apply frequency domain masking."""
        batch_size, seq_len = signal.shape
        
        # Convert to frequency domain
        fft_signal = torch.fft.fft(signal, dim=1)
        
        # Random frequency masking
        n_freq = seq_len
        n_mask = int(n_freq * mask_freq_ratio)
        
        for i in range(batch_size):
            mask_indices = torch.randperm(n_freq)[:n_mask]
            fft_signal[i, mask_indices] = 0
        
        # Convert back to time domain
        return torch.real(torch.fft.ifft(fft_signal, dim=1))
    
    def bandstop_filter(self, signal: torch.Tensor, 
                       stop_freq_range: Tuple[float, float] = (1.0, 2.0)) -> torch.Tensor:
        """Apply random bandstop filtering."""
        # Simplified bandstop filtering using frequency domain
        batch_size, seq_len = signal.shape
        
        fft_signal = torch.fft.fft(signal, dim=1)
        freqs = torch.fft.fftfreq(seq_len, 1/self.fs)
        
        # Random stop band
        stop_center = torch.empty(batch_size, 1).uniform_(
            stop_freq_range[0], stop_freq_range[1]
        ).to(signal.device)
        stop_width = 0.2  # 0.2 Hz width
        
        for i in range(batch_size):
            mask = (torch.abs(freqs) >= stop_center[i] - stop_width/2) & \
                   (torch.abs(freqs) <= stop_center[i] + stop_width/2)
            fft_signal[i, mask] = 0
        
        return torch.real(torch.fft.ifft(fft_signal, dim=1))
    
    def forward(self, signal: torch.Tensor, 
                augmentation_strength: float = 0.8) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply random augmentations to create two views."""
        # First view
        view1 = signal.clone()
        if torch.rand(1) < augmentation_strength:
            if torch.rand(1) < 0.7:
                view1 = self.add_noise(view1, 0.05)
            if torch.rand(1) < 0.4:
                view1 = self.time_shift(view1, 1.0)
            if torch.rand(1) < 0.5:
                view1 = self.amplitude_scale(view1, (0.8, 1.2))
            if torch.rand(1) < 0.3:
                view1 = self.frequency_mask(view1, 0.1)
        
        # Second view (different augmentations)
        view2 = signal.clone()
        if torch.rand(1) < augmentation_strength:
            if torch.rand(1) < 0.7:
                view2 = self.add_noise(view2, 0.05)
            if torch.rand(1) < 0.4:
                view2 = self.time_shift(view2, 1.0)
            if torch.rand(1) < 0.5:
                view2 = self.amplitude_scale(view2, (0.8, 1.2))
            if torch.rand(1) < 0.2:
                view2 = self.bandstop_filter(view2, (1.0, 2.0))
        
        return view1, view2


class InfoNCELoss(nn.Module):
    """InfoNCE loss for contrastive learning."""
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Compute InfoNCE loss.
        
        Args:
            z1: First view projections (batch_size, projection_dim)
            z2: Second view projections (batch_size, projection_dim)
        """
        batch_size = z1.shape[0]
        
        # Concatenate both views
        z = torch.cat([z1, z2], dim=0)  # (2*batch_size, projection_dim)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(z, z.T) / self.temperature  # (2*batch_size, 2*batch_size)
        
        # Create positive pairs mask
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        
        # Positive pairs: (i, i+batch_size) and (i+batch_size, i)
        pos_mask = torch.zeros_like(mask)
        pos_mask[:batch_size, batch_size:] = torch.eye(batch_size, dtype=torch.bool)
        pos_mask[batch_size:, :batch_size] = torch.eye(batch_size, dtype=torch.bool)
        
        # Remove self-similarities
        sim_matrix = sim_matrix[~mask].view(2 * batch_size, -1)
        pos_mask = pos_mask[~mask].view(2 * batch_size, -1)
        
        # Compute loss
        pos_sim = sim_matrix[pos_mask].view(2 * batch_size, 1)
        neg_sim = sim_matrix[~pos_mask].view(2 * batch_size, -1)
        
        logits = torch.cat([pos_sim, neg_sim], dim=1)
        labels = torch.zeros(2 * batch_size, dtype=torch.long, device=z.device)
        
        loss = F.cross_entropy(logits, labels)
        
        return loss


class TemporalContrastiveLoss(nn.Module):
    """Temporal contrastive loss for time series."""
    
    def __init__(self, temperature: float = 0.07, temporal_window: int = 5):
        super().__init__()
        self.temperature = temperature
        self.temporal_window = temporal_window
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Temporal contrastive loss based on respiratory rate similarity.
        
        Args:
            features: Encoded features (batch_size, feature_dim)
            labels: Respiratory rates (batch_size,)
        """
        batch_size = features.shape[0]
        
        # Normalize features
        features = F.normalize(features, dim=-1)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Create positive pairs based on respiratory rate similarity
        rate_diff = torch.abs(labels.unsqueeze(0) - labels.unsqueeze(1))
        pos_mask = rate_diff <= 2.0  # Within 2 breaths/min
        
        # Remove self-similarities
        pos_mask.fill_diagonal_(False)
        
        # Create negative mask
        neg_mask = ~pos_mask
        neg_mask.fill_diagonal_(False)
        
        # Compute positive and negative similarities
        pos_sim = sim_matrix[pos_mask]
        neg_sim = sim_matrix[neg_mask]
        
        if len(pos_sim) == 0:
            return torch.tensor(0.0, device=features.device)
        
        # Compute contrastive loss
        pos_loss = -torch.log(torch.exp(pos_sim).sum() + 1e-8)
        neg_loss = torch.log(torch.exp(neg_sim).sum() + 1e-8)
        
        loss = pos_loss + neg_loss
        
        return loss


class ContrastiveLearningModule(nn.Module):
    """Complete contrastive learning module."""
    
    def __init__(self, encoder_dim: int, projection_dim: int = 128, 
                 temperature: float = 0.07, fs: int = 125):
        super().__init__()
        
        # Projection head
        self.projection_head = ContrastiveProjectionHead(encoder_dim, projection_dim)
        
        # Augmentation module
        self.augmentation = PPGAugmentation(fs)
        
        # Loss functions
        self.infonce_loss = InfoNCELoss(temperature)
        self.temporal_loss = TemporalContrastiveLoss(temperature)
        
    def forward(self, features: torch.Tensor, 
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for contrastive learning.
        
        Args:
            features: Encoded features (batch_size, encoder_dim)
            labels: Optional respiratory rate labels
        """
        # Project to contrastive space
        projections = self.projection_head(features)
        
        outputs = {'projections': projections}
        
        # Compute temporal contrastive loss if labels provided
        if labels is not None:
            temporal_loss = self.temporal_loss(features, labels)
            outputs['temporal_loss'] = temporal_loss
        
        return outputs
    
    def compute_contrastive_loss(self, ppg_signals: torch.Tensor, 
                               encoder: nn.Module) -> torch.Tensor:
        """
        Compute contrastive loss with augmented views.
        
        Args:
            ppg_signals: Input PPG signals (batch_size, seq_len)
            encoder: Encoder model to extract features
        """
        # Create augmented views
        view1, view2 = self.augmentation(ppg_signals)
        
        # Encode both views
        with torch.cuda.amp.autocast(enabled=False):
            features1 = encoder(view1.unsqueeze(-1))  # Add channel dimension
            features2 = encoder(view2.unsqueeze(-1))
        
        # Project to contrastive space
        z1 = self.projection_head(features1)
        z2 = self.projection_head(features2)
        
        # Compute InfoNCE loss
        loss = self.infonce_loss(z1, z2)
        
        return loss


if __name__ == "__main__":
    # Test contrastive learning module
    batch_size = 8
    encoder_dim = 512
    projection_dim = 128
    seq_len = 3750
    
    # Create contrastive module
    contrastive_module = ContrastiveLearningModule(
        encoder_dim=encoder_dim,
        projection_dim=projection_dim,
        temperature=0.07,
        fs=125
    )
    
    # Test features
    features = torch.randn(batch_size, encoder_dim)
    labels = torch.rand(batch_size) * 20 + 10  # 10-30 breaths/min
    
    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Test forward pass
    with torch.no_grad():
        outputs = contrastive_module(features, labels)
        print(f"Projections shape: {outputs['projections'].shape}")
        print(f"Temporal loss: {outputs['temporal_loss'].item():.4f}")
    
    # Test augmentation
    ppg_signals = torch.randn(batch_size, seq_len)
    view1, view2 = contrastive_module.augmentation(ppg_signals)
    print(f"Original signal shape: {ppg_signals.shape}")
    print(f"Augmented view1 shape: {view1.shape}")
    print(f"Augmented view2 shape: {view2.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in contrastive_module.parameters())
    print(f"Contrastive module parameters: {total_params:,}")
