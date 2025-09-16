"""Dual-branch RWKV model with time and frequency domain processing."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from .optimized_rwkv import OptimizedRWKV, create_optimized_rwkv
from .frequency_branch import FrequencyBranch
from .contrastive import ContrastiveLearningModule


class CrossModalAttention(nn.Module):
    """Cross-modal attention between time and frequency domains."""
    
    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        # Ensure num_heads divides hidden_size evenly
        if hidden_size % num_heads != 0:
            num_heads = 8 if hidden_size >= 64 else 4
            while hidden_size % num_heads != 0 and num_heads > 1:
                num_heads -= 1
        self.num_heads = num_heads
        
        # Cross-attention layers
        self.time_to_freq_attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.freq_to_time_attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        
        # Layer normalization
        self.time_norm = nn.LayerNorm(hidden_size)
        self.freq_norm = nn.LayerNorm(hidden_size)
        
        # Feed-forward networks
        self.time_ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size)
        )
        
        self.freq_ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size)
        )
    
    def forward(self, time_features: torch.Tensor, 
                freq_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply cross-modal attention.
        
        Args:
            time_features: Time domain features (batch_size, hidden_size)
            freq_features: Frequency domain features (batch_size, hidden_size)
        """
        batch_size = time_features.shape[0]
        
        # Add sequence dimension for attention
        time_seq = time_features.unsqueeze(1)  # (batch_size, 1, hidden_size)
        freq_seq = freq_features.unsqueeze(1)  # (batch_size, 1, hidden_size)
        
        # Cross-attention: time attending to frequency
        time_attended, _ = self.time_to_freq_attention(
            time_seq, freq_seq, freq_seq
        )
        time_attended = time_attended.squeeze(1)  # (batch_size, hidden_size)
        
        # Cross-attention: frequency attending to time
        freq_attended, _ = self.freq_to_time_attention(
            freq_seq, time_seq, time_seq
        )
        freq_attended = freq_attended.squeeze(1)  # (batch_size, hidden_size)
        
        # Residual connections and layer norm
        time_enhanced = self.time_norm(time_features + time_attended)
        freq_enhanced = self.freq_norm(freq_features + freq_attended)
        
        # Feed-forward networks
        time_output = time_enhanced + self.time_ffn(time_enhanced)
        freq_output = freq_enhanced + self.freq_ffn(freq_enhanced)
        
        return time_output, freq_output


class AdaptiveFusion(nn.Module):
    """Adaptive fusion of time and frequency features."""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        
        # Gating network to learn fusion weights
        self.gate_network = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2),
            nn.Softmax(dim=-1)
        )
        
        # Feature transformation
        self.time_transform = nn.Linear(hidden_size, hidden_size)
        self.freq_transform = nn.Linear(hidden_size, hidden_size)
        
        # Final fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size)
        )
        
        self.output_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, time_features: torch.Tensor, 
                freq_features: torch.Tensor) -> torch.Tensor:
        """
        Adaptively fuse time and frequency features.
        
        Args:
            time_features: Time domain features (batch_size, hidden_size)
            freq_features: Frequency domain features (batch_size, hidden_size)
        """
        # Transform features
        time_transformed = self.time_transform(time_features)
        freq_transformed = self.freq_transform(freq_features)
        
        # Compute adaptive weights
        concatenated = torch.cat([time_transformed, freq_transformed], dim=-1)
        weights = self.gate_network(concatenated)  # (batch_size, 2)
        
        # Weighted fusion
        time_weight = weights[:, 0:1]  # (batch_size, 1)
        freq_weight = weights[:, 1:2]  # (batch_size, 1)
        
        fused = time_weight * time_transformed + freq_weight * freq_transformed
        
        # Apply final transformation
        output = self.fusion(fused)
        output = self.output_norm(output)
        
        return output


class DualBranchRWKV(nn.Module):
    """
    Dual-branch RWKV model for respiratory rate estimation.
    Combines time domain and frequency domain processing.
    """
    
    def __init__(self, 
                 time_hidden_size: int = 256,
                 time_num_layers: int = 6,
                 freq_hidden_size: int = 256,
                 freq_num_layers: int = 4,
                 fusion_hidden_size: int = 512,
                 num_attention_heads: int = 8,
                 dropout: float = 0.2,
                 fs: int = 125,
                 n_fft: int = 3750,
                 n_mels: int = 64):
        super().__init__()
        
        self.time_hidden_size = time_hidden_size
        self.freq_hidden_size = freq_hidden_size
        self.fusion_hidden_size = fusion_hidden_size
        
        # Time domain branch (optimized)
        self.time_branch = create_optimized_rwkv(
            input_size=1,  # Single channel PPG
            hidden_size=time_hidden_size,
            num_layers=time_num_layers,
            dropout=dropout,
            compile_kernels=True  # Enable CUDA acceleration
        )
        
        # Frequency domain branch
        self.frequency_branch = FrequencyBranch(
            fs=fs,
            hidden_size=freq_hidden_size,
            num_layers=freq_num_layers,
            n_fft=n_fft,
            n_mels=n_mels
        )
        
        # Cross-modal attention
        # Ensure the target size is compatible with num_attention_heads
        target_size = min(time_hidden_size, freq_hidden_size)
        if target_size % num_attention_heads != 0:
            # Adjust target size to be divisible by num_attention_heads
            target_size = (target_size // num_attention_heads) * num_attention_heads
            if target_size < 32:  # Minimum reasonable size
                target_size = 32
                num_attention_heads = min(num_attention_heads, target_size // 4)
        
        self.cross_attention = CrossModalAttention(
            hidden_size=target_size,
            num_heads=num_attention_heads,
            dropout=dropout
        )
        
        # Feature alignment (if different hidden sizes)
        if time_hidden_size != target_size:
            self.time_align = nn.Linear(time_hidden_size, target_size)
        else:
            self.time_align = nn.Identity()
            
        if freq_hidden_size != target_size:
            self.freq_align = nn.Linear(freq_hidden_size, target_size)
        else:
            self.freq_align = nn.Identity()
        
        # Store target size for later use
        self.target_size = target_size
        
        # Adaptive fusion
        self.adaptive_fusion = AdaptiveFusion(self.target_size)
        
        # Final regression head
        self.regression_head = nn.Sequential(
            nn.Linear(self.target_size, fusion_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_size, fusion_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_size // 2, 1),
            nn.ReLU()  # Ensure positive respiratory rate
        )
        
        # Contrastive learning module (for pretraining)
        self.contrastive_module = ContrastiveLearningModule(
            encoder_dim=self.target_size,
            projection_dim=128,
            temperature=0.07,
            fs=fs
        )
        
        # Training mode flag
        self.pretraining_mode = True
        
    def forward(self, ppg_signal: torch.Tensor, 
                return_features: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass through dual-branch model.
        
        Args:
            ppg_signal: PPG signal (batch_size, seq_len)
            return_features: Whether to return intermediate features
        """
        batch_size, seq_len = ppg_signal.shape
        
        # Time domain processing
        time_input = ppg_signal.unsqueeze(-1)  # (batch_size, seq_len, 1)
        time_features = self.time_branch(time_input)  # (batch_size, time_hidden_size)
        
        # Frequency domain processing
        freq_features = self.frequency_branch(ppg_signal)  # (batch_size, freq_hidden_size)
        
        # Align feature dimensions
        time_aligned = self.time_align(time_features)
        freq_aligned = self.freq_align(freq_features)
        
        # Cross-modal attention
        time_enhanced, freq_enhanced = self.cross_attention(time_aligned, freq_aligned)
        
        # Adaptive fusion
        fused_features = self.adaptive_fusion(time_enhanced, freq_enhanced)
        
        # Outputs dictionary
        outputs = {
            'fused_features': fused_features,
            'time_features': time_features,
            'freq_features': freq_features
        }
        
        # Regression prediction
        respiratory_rate = self.regression_head(fused_features)
        outputs['respiratory_rate'] = respiratory_rate
        
        # Contrastive learning outputs (if in pretraining mode)
        if self.pretraining_mode:
            contrastive_outputs = self.contrastive_module(fused_features)
            outputs.update(contrastive_outputs)
        
        if return_features:
            outputs.update({
                'time_enhanced': time_enhanced,
                'freq_enhanced': freq_enhanced,
                'time_aligned': time_aligned,
                'freq_aligned': freq_aligned
            })
        
        return outputs
    
    def compute_contrastive_loss(self, ppg_signals: torch.Tensor) -> torch.Tensor:
        """Compute contrastive loss for pretraining."""
        return self.contrastive_module.compute_contrastive_loss(
            ppg_signals, self.encode_features
        )
    
    def encode_features(self, ppg_signal: torch.Tensor) -> torch.Tensor:
        """Encode PPG signal to fused features (for contrastive learning)."""
        with torch.no_grad():
            outputs = self.forward(ppg_signal, return_features=False)
            return outputs['fused_features']
    
    def set_pretraining_mode(self, mode: bool):
        """Set pretraining mode."""
        self.pretraining_mode = mode
    
    def freeze_encoder(self):
        """Freeze encoder parameters for fine-tuning."""
        for param in self.time_branch.parameters():
            param.requires_grad = False
        for param in self.frequency_branch.parameters():
            param.requires_grad = False
        for param in self.cross_attention.parameters():
            param.requires_grad = False
        for param in self.adaptive_fusion.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self):
        """Unfreeze encoder parameters."""
        for param in self.parameters():
            param.requires_grad = True


if __name__ == "__main__":
    # Test dual-branch model
    batch_size = 4
    seq_len = 3750  # 30 seconds at 125 Hz
    
    model = DualBranchRWKV(
        time_hidden_size=256,
        time_num_layers=6,
        freq_hidden_size=256,
        freq_num_layers=4,
        fusion_hidden_size=512,
        fs=125
    )
    
    # Test input
    ppg_signal = torch.randn(batch_size, seq_len)
    
    print(f"Input PPG shape: {ppg_signal.shape}")
    
    # Forward pass
    with torch.no_grad():
        outputs = model(ppg_signal, return_features=True)
        
        print(f"Time features shape: {outputs['time_features'].shape}")
        print(f"Frequency features shape: {outputs['freq_features'].shape}")
        print(f"Fused features shape: {outputs['fused_features'].shape}")
        print(f"Respiratory rate shape: {outputs['respiratory_rate'].shape}")
        print(f"Projections shape: {outputs['projections'].shape}")
    
    # Test contrastive loss
    contrastive_loss = model.compute_contrastive_loss(ppg_signal)
    print(f"Contrastive loss: {contrastive_loss.item():.4f}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params:,}")
    
    # Count parameters by component
    time_params = sum(p.numel() for p in model.time_branch.parameters())
    freq_params = sum(p.numel() for p in model.frequency_branch.parameters())
    fusion_params = sum(p.numel() for p in model.cross_attention.parameters()) + \
                   sum(p.numel() for p in model.adaptive_fusion.parameters())
    regression_params = sum(p.numel() for p in model.regression_head.parameters())
    contrastive_params = sum(p.numel() for p in model.contrastive_module.parameters())
    
    print(f"\nParameter breakdown:")
    print(f"Time branch: {time_params:,}")
    print(f"Frequency branch: {freq_params:,}")
    print(f"Fusion layers: {fusion_params:,}")
    print(f"Regression head: {regression_params:,}")
    print(f"Contrastive module: {contrastive_params:,}")
