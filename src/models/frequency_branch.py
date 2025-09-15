"""Frequency domain processing branch for PPG signals."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import signal
from typing import Tuple, Optional, Dict
from .rwkv import RWKV


class SpectralFeatureExtractor(nn.Module):
    """Extract spectral features from PPG signals."""
    
    def __init__(self, fs: int = 125, n_fft: int = 3750, n_mels: int = 64):
        super().__init__()
        self.fs = fs
        self.n_fft = n_fft
        self.n_mels = n_mels
        
        # Create physiological mel-scale filterbank
        self.register_buffer('mel_filters', self._create_physio_mel_filters())
        
        # Spectral enhancement layers
        self.spectral_conv = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.spectral_norm = nn.BatchNorm1d(32)
        
        # Frequency band extractors
        self.respiratory_extractor = RespiratoryBandExtractor(fs)
        self.cardiac_extractor = CardiacBandExtractor(fs)
        
    def _create_physio_mel_filters(self) -> torch.Tensor:
        """Create physiological mel-scale filters optimized for PPG."""
        # Create a dummy filter that will be resized dynamically
        return torch.eye(self.n_mels)
    
    def create_mel_filters(self, freq_bins: int) -> torch.Tensor:
        """Create mel filters for given frequency bins."""
        # Define frequency points with higher resolution in respiratory/cardiac bands
        freq_points = []
        
        # Respiratory band: 0.1-0.5 Hz (6-30 breaths/min) - high resolution
        resp_points = np.linspace(0.1, 0.5, self.n_mels // 3)
        freq_points.extend(resp_points)
        
        # Transition: 0.5-0.8 Hz - medium resolution
        trans_points = np.linspace(0.5, 0.8, self.n_mels // 6)
        freq_points.extend(trans_points)
        
        # Cardiac band: 0.8-3.0 Hz (48-180 bpm) - high resolution
        cardiac_points = np.linspace(0.8, 3.0, self.n_mels // 2)
        freq_points.extend(cardiac_points)
        
        # Convert frequency points to mel filters
        mel_filters = torch.zeros(self.n_mels, freq_bins)
        freq_resolution = self.fs / 2 / freq_bins
        
        for i, freq in enumerate(freq_points[:self.n_mels]):
            # Create triangular filter centered at freq
            center_bin = int(freq / freq_resolution)
            width = max(1, int(0.1 / freq_resolution))  # 0.1 Hz width
            
            start_bin = max(0, center_bin - width)
            end_bin = min(freq_bins, center_bin + width + 1)
            
            # Triangular filter
            for j in range(start_bin, end_bin):
                if width > 0:
                    mel_filters[i, j] = 1.0 - abs(j - center_bin) / width
        
        return mel_filters
    
    def compute_spectrogram(self, ppg_signal: torch.Tensor) -> torch.Tensor:
        """Compute spectrogram of PPG signal."""
        batch_size, seq_len = ppg_signal.shape
        spectrograms = []
        
        # Use a reasonable window size (8 seconds) for better time resolution
        window_size = min(1000, seq_len // 4)  # 8 seconds at 125 Hz, or 1/4 of signal
        overlap = window_size // 2
        
        for i in range(batch_size):
            # Convert to numpy for scipy
            signal_np = ppg_signal[i].cpu().numpy()
            
            # Compute spectrogram with reasonable time resolution
            f, t, Sxx = signal.spectrogram(
                signal_np,
                fs=self.fs,
                nperseg=window_size,
                noverlap=overlap,
                window='hann'
            )
            
            # Convert to log scale and add to batch
            log_spec = np.log(Sxx + 1e-8)
            spectrograms.append(torch.from_numpy(log_spec).float())
        
        # Stack into batch tensor
        spectrograms = torch.stack(spectrograms).to(ppg_signal.device)
        return spectrograms
    
    def forward(self, ppg_signal: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract spectral features from PPG signal."""
        # Compute spectrogram
        spectrograms = self.compute_spectrogram(ppg_signal)  # (B, F, T)
        
        # Create mel filters for current spectrogram size
        freq_bins = spectrograms.shape[1]
        mel_filters = self.create_mel_filters(freq_bins).to(spectrograms.device)
        
        # Apply mel-scale transformation
        mel_spec = torch.matmul(mel_filters, spectrograms)  # (B, n_mels, T)
        
        # Extract spectral envelope
        spectral_envelope = torch.mean(spectrograms, dim=2, keepdim=True)  # (B, F, 1)
        # Transpose for 1D conv: (B, 1, F)
        spectral_envelope = spectral_envelope.transpose(1, 2)  # (B, 1, F)
        spectral_features = F.relu(self.spectral_norm(self.spectral_conv(spectral_envelope)))
        
        # Extract frequency band features
        resp_features = self.respiratory_extractor(spectrograms)
        cardiac_features = self.cardiac_extractor(spectrograms)
        
        return {
            'mel_spectrogram': mel_spec,
            'spectral_envelope': spectral_features.squeeze(-1),
            'respiratory_features': resp_features,
            'cardiac_features': cardiac_features,
            'full_spectrogram': spectrograms
        }


class RespiratoryBandExtractor(nn.Module):
    """Extract features from respiratory frequency band (0.1-0.5 Hz)."""
    
    def __init__(self, fs: int = 125):
        super().__init__()
        self.fs = fs
        self.freq_res = fs / 2  # Assuming n_fft gives this resolution
        
        # Peak detection layers
        self.peak_detector = nn.Conv1d(1, 16, kernel_size=5, padding=2)
        self.peak_enhance = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        
    def extract_respiratory_band(self, spectrograms: torch.Tensor) -> torch.Tensor:
        """Extract respiratory frequency band (0.1-0.5 Hz)."""
        freq_bins = spectrograms.shape[1]
        freq_resolution = self.fs / 2 / freq_bins
        
        start_idx = max(0, int(0.1 / freq_resolution))
        end_idx = min(freq_bins, int(0.5 / freq_resolution))
        
        return spectrograms[:, start_idx:end_idx, :]
    
    def forward(self, spectrograms: torch.Tensor) -> torch.Tensor:
        """Extract respiratory band features."""
        # Extract respiratory band
        resp_band = self.extract_respiratory_band(spectrograms)
        
        # Compute frequency profile (average over time)
        freq_profile = torch.mean(resp_band, dim=2, keepdim=True)  # (B, F_resp, 1)
        
        # Transpose for 1D conv: (B, 1, F_resp)
        freq_profile = freq_profile.transpose(1, 2)  # (B, 1, F_resp)
        
        # Check if we have enough frequency bins for convolution
        freq_bins = freq_profile.shape[-1]
        if freq_bins < 5:
            # Pad or expand to minimum size
            freq_profile = F.pad(freq_profile, (0, max(0, 5 - freq_bins)))
        
        # Enhance peaks
        enhanced = F.relu(self.peak_detector(freq_profile))
        peaks = self.peak_enhance(enhanced)
        
        # Global average pooling to fixed size
        features = F.adaptive_avg_pool1d(peaks, 1).squeeze(-1)  # (B, 32)
        
        return features


class CardiacBandExtractor(nn.Module):
    """Extract features from cardiac frequency band (0.8-3.0 Hz)."""
    
    def __init__(self, fs: int = 125):
        super().__init__()
        self.fs = fs
        
        # Cardiac pattern detection
        self.cardiac_conv = nn.Conv1d(1, 24, kernel_size=7, padding=3)
        self.cardiac_norm = nn.BatchNorm1d(24)
        
    def extract_cardiac_band(self, spectrograms: torch.Tensor) -> torch.Tensor:
        """Extract cardiac frequency band (0.8-3.0 Hz)."""
        freq_bins = spectrograms.shape[1]
        freq_resolution = self.fs / 2 / freq_bins
        
        start_idx = max(0, int(0.8 / freq_resolution))
        end_idx = min(freq_bins, int(3.0 / freq_resolution))
        
        return spectrograms[:, start_idx:end_idx, :]
    
    def forward(self, spectrograms: torch.Tensor) -> torch.Tensor:
        """Extract cardiac band features."""
        # Extract cardiac band
        cardiac_band = self.extract_cardiac_band(spectrograms)
        
        # Compute frequency profile
        freq_profile = torch.mean(cardiac_band, dim=2, keepdim=True)
        
        # Transpose for 1D conv: (B, 1, F_cardiac)
        freq_profile = freq_profile.transpose(1, 2)  # (B, 1, F_cardiac)
        
        # Check if we have enough frequency bins for convolution
        freq_bins = freq_profile.shape[-1]
        if freq_bins < 7:  # Kernel size for cardiac conv is 7
            freq_profile = F.pad(freq_profile, (0, max(0, 7 - freq_bins)))
        
        # Process through convolution
        features = F.relu(self.cardiac_norm(self.cardiac_conv(freq_profile)))
        
        # Global average pooling to get (B, 24)
        features = F.adaptive_avg_pool1d(features, 1).squeeze(-1)  # (B, 24)
        # Expand to match respiratory features dimension
        features = F.pad(features, (0, 8))  # Pad to (B, 32)
        
        return features


class MultiScaleFreqRWKV(nn.Module):
    """Multi-scale frequency processing with RWKV."""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        
        # Ensure divisible dimensions
        scale_dim = hidden_size // 4  # Leave some room for alignment
        if scale_dim < 16:
            scale_dim = 16
        
        # Different frequency resolution processors
        self.fine_scale_rwkv = RWKV(input_size, scale_dim, num_layers)
        self.medium_scale_rwkv = RWKV(input_size // 2, scale_dim, num_layers)
        self.coarse_scale_rwkv = RWKV(input_size // 4, scale_dim, num_layers)
        
        # Frequency pooling
        self.medium_pool = nn.AvgPool1d(2, stride=2)
        self.coarse_pool = nn.AvgPool1d(4, stride=4)
        
        # Feature projection to match hidden_size
        combined_size = scale_dim * 3
        self.feature_projection = nn.Linear(combined_size, hidden_size)
        
        # Scale fusion - ensure num_heads divides hidden_size
        num_heads = 8
        while hidden_size % num_heads != 0 and num_heads > 1:
            num_heads -= 1
        self.scale_fusion = nn.MultiheadAttention(hidden_size, num_heads=num_heads, batch_first=True)
        self.fusion_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, freq_features: torch.Tensor) -> torch.Tensor:
        """Process frequency features at multiple scales."""
        batch_size, seq_len, feature_dim = freq_features.shape
        
        # Fine scale (full resolution)
        fine_out = self.fine_scale_rwkv(freq_features)  # (B, scale_dim)
        
        # Medium scale (half resolution) - pool along frequency dimension
        if seq_len >= 2:
            medium_freq = self.medium_pool(freq_features.transpose(1, 2)).transpose(1, 2)
            # Ensure correct input dimension for medium scale RWKV
            if medium_freq.shape[-1] != feature_dim // 2:
                medium_freq = medium_freq[:, :, :feature_dim // 2]
            medium_out = self.medium_scale_rwkv(medium_freq)  # (B, scale_dim)
        else:
            medium_out = torch.zeros_like(fine_out)
        
        # Coarse scale (quarter resolution) - pool along frequency dimension  
        if seq_len >= 4:
            coarse_freq = self.coarse_pool(freq_features.transpose(1, 2)).transpose(1, 2)
            # Ensure correct input dimension for coarse scale RWKV
            if coarse_freq.shape[-1] != feature_dim // 4:
                coarse_freq = coarse_freq[:, :, :feature_dim // 4]
            coarse_out = self.coarse_scale_rwkv(coarse_freq)  # (B, scale_dim)
        else:
            coarse_out = torch.zeros_like(fine_out)
        
        # Combine scales and project to target dimension
        combined = torch.cat([fine_out, medium_out, coarse_out], dim=-1)  # (B, scale_dim * 3)
        projected = self.feature_projection(combined)  # (B, hidden_size)
        
        # Apply self-attention for scale fusion
        projected = projected.unsqueeze(1)  # (B, 1, hidden_size)
        attended, _ = self.scale_fusion(projected, projected, projected)
        attended = self.fusion_norm(attended.squeeze(1))  # (B, hidden_size)

        return attended


class FrequencyBranch(nn.Module):
    """Complete frequency domain processing branch."""
    
    def __init__(self, fs: int = 125, hidden_size: int = 256, num_layers: int = 4,
                 n_fft: int = 3750, n_mels: int = 64):
        super().__init__()
        self.fs = fs
        self.hidden_size = hidden_size
        
        # Spectral feature extraction
        self.spectral_extractor = SpectralFeatureExtractor(fs, n_fft, n_mels)
        
        # Multi-scale RWKV processing
        self.mel_rwkv = MultiScaleFreqRWKV(n_mels, hidden_size // 2, num_layers)
        
        # Feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Linear(64, hidden_size // 4),  # resp_features + cardiac_features (32 + 32)
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 4, hidden_size // 4)
        )
        
        # Final fusion
        self.final_fusion = nn.Sequential(
            nn.Linear(hidden_size // 2 + hidden_size // 4, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Output normalization
        self.output_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, ppg_signal: torch.Tensor) -> torch.Tensor:
        """Process PPG signal through frequency domain branch."""
        # Extract spectral features
        spectral_features = self.spectral_extractor(ppg_signal)
        
        # Process mel spectrogram through multi-scale RWKV
        mel_spec = spectral_features['mel_spectrogram']  # (B, n_mels, T)
        mel_spec = mel_spec.transpose(1, 2)  # (B, T, n_mels)
        mel_output = self.mel_rwkv(mel_spec)  # (B, hidden_size//2)
        
        # Fuse respiratory and cardiac features  
        resp_features = spectral_features['respiratory_features']  # (B, 32)
        cardiac_features = spectral_features['cardiac_features']  # (B, 32)
        
        band_features = torch.cat([resp_features, cardiac_features], dim=-1)  # (B, 64)
        band_output = self.feature_fusion(band_features)  # (B, hidden_size//4)
        
        # Final fusion
        combined = torch.cat([mel_output, band_output], dim=-1)
        output = self.final_fusion(combined)
        output = self.output_norm(output)
        
        return output


if __name__ == "__main__":
    # Test frequency branch
    batch_size = 4
    seq_len = 3750  # 30 seconds at 125 Hz
    fs = 125
    
    freq_branch = FrequencyBranch(fs=fs, hidden_size=256)
    ppg_signal = torch.randn(batch_size, seq_len)
    
    print(f"Input PPG shape: {ppg_signal.shape}")
    
    with torch.no_grad():
        output = freq_branch(ppg_signal)
        print(f"Frequency branch output shape: {output.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in freq_branch.parameters())
    print(f"Frequency branch parameters: {total_params:,}")
