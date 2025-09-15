"""Signal preprocessing utilities for PPG and respiratory signals."""

import numpy as np
import pandas as pd
import torch
from scipy import signal
from scipy.signal import butter, filtfilt, resample
from typing import Tuple, Optional, Dict, List
import warnings
warnings.filterwarnings("ignore")


class SignalPreprocessor:
    """Comprehensive signal preprocessing for PPG and respiratory signals."""
    
    def __init__(self, fs: int = 125):
        """
        Initialize preprocessor.
        
        Args:
            fs: Sampling frequency
        """
        self.fs = fs
        
        # Filter specifications
        self.ppg_lowpass = 15.0  # Hz
        self.ppg_highpass = 0.5  # Hz
        self.resp_lowpass = 1.0  # Hz
        self.resp_highpass = 0.05  # Hz
        
    def load_signals(self, signals_file: str) -> pd.DataFrame:
        """Load signals from CSV file."""
        try:
            df = pd.read_csv(signals_file)
            # Clean column names
            df.columns = df.columns.str.strip()
            return df
        except Exception as e:
            print(f"Error loading {signals_file}: {e}")
            return None
    
    def load_numerics(self, numerics_file: str) -> pd.DataFrame:
        """Load numeric values from CSV file."""
        try:
            df = pd.read_csv(numerics_file)
            # Clean column names
            df.columns = df.columns.str.strip()
            return df
        except Exception as e:
            print(f"Error loading {numerics_file}: {e}")
            return None
    
    def bandpass_filter(self, signal_data: np.ndarray, 
                       low_freq: float, high_freq: float,
                       order: int = 4) -> np.ndarray:
        """Apply bandpass filter to signal."""
        nyquist = self.fs / 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        # Ensure frequencies are in valid range
        low = max(0.001, min(low, 0.999))
        high = max(low + 0.001, min(high, 0.999))
        
        try:
            b, a = butter(order, [low, high], btype='band')
            filtered_signal = filtfilt(b, a, signal_data)
            return filtered_signal
        except Exception as e:
            print(f"Filter error: {e}")
            return signal_data
    
    def remove_artifacts(self, signal_data: np.ndarray, 
                        threshold_std: float = 4.0) -> np.ndarray:
        """Remove artifacts using statistical thresholding."""
        mean_val = np.mean(signal_data)
        std_val = np.std(signal_data)
        
        # Replace outliers with median value
        outlier_mask = np.abs(signal_data - mean_val) > (threshold_std * std_val)
        cleaned_signal = signal_data.copy()
        cleaned_signal[outlier_mask] = np.median(signal_data)
        
        return cleaned_signal
    
    def normalize_signal(self, signal_data: np.ndarray, 
                        method: str = 'zscore') -> np.ndarray:
        """Normalize signal using specified method."""
        if method == 'zscore':
            return (signal_data - np.mean(signal_data)) / (np.std(signal_data) + 1e-8)
        elif method == 'minmax':
            min_val, max_val = np.min(signal_data), np.max(signal_data)
            return (signal_data - min_val) / (max_val - min_val + 1e-8)
        elif method == 'robust':
            median_val = np.median(signal_data)
            mad = np.median(np.abs(signal_data - median_val))
            return (signal_data - median_val) / (mad + 1e-8)
        else:
            return signal_data
    
    def extract_ppg_features(self, ppg_signal: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract PPG-specific features."""
        features = {}
        
        # 1. Filtered PPG
        features['ppg_filtered'] = self.bandpass_filter(
            ppg_signal, self.ppg_highpass, self.ppg_lowpass
        )
        
        # 2. PPG envelope (amplitude modulation)
        analytic_signal = signal.hilbert(features['ppg_filtered'])
        features['ppg_envelope'] = np.abs(analytic_signal)
        
        # 3. PPG baseline (using moving average)
        window_size = int(self.fs * 5)  # 5-second window
        features['ppg_baseline'] = np.convolve(
            ppg_signal, np.ones(window_size)/window_size, mode='same'
        )
        
        # 4. Respiratory modulation (envelope - baseline)
        features['respiratory_modulation'] = features['ppg_envelope'] - features['ppg_baseline']
        
        # 5. Peak detection features
        peaks, _ = signal.find_peaks(features['ppg_filtered'], 
                                   height=np.std(features['ppg_filtered']),
                                   distance=int(self.fs * 0.4))  # Min 0.4s between peaks
        
        # Peak interval variability
        if len(peaks) > 1:
            peak_intervals = np.diff(peaks) / self.fs
            features['peak_intervals'] = peak_intervals
            features['hrv_rmssd'] = np.sqrt(np.mean(np.diff(peak_intervals)**2))
        else:
            features['hrv_rmssd'] = 0.0
        
        return features
    
    def create_windows(self, signal_data: np.ndarray, labels: np.ndarray,
                      window_size: int, overlap: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create overlapping windows from signal data.
        
        Args:
            signal_data: Input signal (samples,)
            labels: Corresponding labels (samples,)
            window_size: Window size in samples
            overlap: Overlap ratio (0-1)
            
        Returns:
            Tuple of (windowed_signals, windowed_labels)
        """
        step_size = int(window_size * (1 - overlap))
        n_windows = (len(signal_data) - window_size) // step_size + 1
        
        windows = []
        window_labels = []
        
        for i in range(n_windows):
            start_idx = i * step_size
            end_idx = start_idx + window_size
            
            if end_idx <= len(signal_data):
                # Extract window
                window = signal_data[start_idx:end_idx]
                
                # Get corresponding label (use median of the window)
                window_label_indices = np.arange(start_idx // self.fs, end_idx // self.fs)
                window_label_indices = window_label_indices[window_label_indices < len(labels)]
                
                if len(window_label_indices) > 0:
                    window_label = np.median(labels[window_label_indices])
                    windows.append(window)
                    window_labels.append(window_label)
        
        return np.array(windows), np.array(window_labels)
    
    def process_subject_data(self, signals_df: pd.DataFrame, 
                           numerics_df: pd.DataFrame,
                           window_size_sec: int = 30) -> Dict[str, np.ndarray]:
        """
        Process complete subject data.
        
        Args:
            signals_df: DataFrame with signal data
            numerics_df: DataFrame with numeric data (including RESP)
            window_size_sec: Window size in seconds
            
        Returns:
            Dictionary with processed data
        """
        result = {}
        
        # Extract PPG signal (PLETH column)
        if 'PLETH' in signals_df.columns:
            ppg_raw = signals_df['PLETH'].values
        else:
            print("PLETH column not found in signals")
            return None
        
        # Extract respiratory rate labels
        if 'RESP' in numerics_df.columns:
            resp_rate = numerics_df['RESP'].values
        else:
            print("RESP column not found in numerics")
            return None
        
        # Remove NaN values
        ppg_raw = ppg_raw[~np.isnan(ppg_raw)]
        resp_rate = resp_rate[~np.isnan(resp_rate)]
        
        if len(ppg_raw) == 0 or len(resp_rate) == 0:
            print("No valid data found")
            return None
        
        # Preprocess PPG signal
        ppg_cleaned = self.remove_artifacts(ppg_raw)
        ppg_normalized = self.normalize_signal(ppg_cleaned, method='zscore')
        
        # Extract PPG features
        ppg_features = self.extract_ppg_features(ppg_normalized)
        
        # Create windows
        window_size_samples = window_size_sec * self.fs
        
        # Main PPG signal windows
        ppg_windows, resp_windows = self.create_windows(
            ppg_normalized, resp_rate, window_size_samples
        )
        
        # Feature windows (respiratory modulation)
        if 'respiratory_modulation' in ppg_features:
            resp_mod_normalized = self.normalize_signal(ppg_features['respiratory_modulation'])
            resp_mod_windows, _ = self.create_windows(
                resp_mod_normalized, resp_rate, window_size_samples
            )
        else:
            resp_mod_windows = np.zeros_like(ppg_windows)
        
        result = {
            'ppg_windows': ppg_windows.astype(np.float32),
            'resp_mod_windows': resp_mod_windows.astype(np.float32),
            'resp_rate': resp_windows.astype(np.float32),
            'window_size': window_size_samples,
            'fs': self.fs
        }
        
        return result


class AugmentationTransforms:
    """Data augmentation transforms for contrastive learning."""
    
    def __init__(self, fs: int = 125):
        self.fs = fs
    
    def add_noise(self, signal: np.ndarray, noise_std: float = 0.05) -> np.ndarray:
        """Add Gaussian noise to signal."""
        noise = np.random.normal(0, noise_std, signal.shape)
        return signal + noise
    
    def time_shift(self, signal: np.ndarray, max_shift_sec: float = 1.0) -> np.ndarray:
        """Apply random time shift."""
        max_shift_samples = int(max_shift_sec * self.fs)
        shift = np.random.randint(-max_shift_samples, max_shift_samples + 1)
        
        if shift > 0:
            return np.concatenate([signal[shift:], signal[:shift]])
        elif shift < 0:
            return np.concatenate([signal[shift:], signal[:shift]])
        else:
            return signal
    
    def amplitude_scale(self, signal: np.ndarray, 
                       scale_range: Tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
        """Apply random amplitude scaling."""
        scale = np.random.uniform(scale_range[0], scale_range[1])
        return signal * scale
    
    def frequency_mask(self, signal: np.ndarray, 
                      mask_freq_ratio: float = 0.1) -> np.ndarray:
        """Apply frequency domain masking."""
        fft_signal = np.fft.fft(signal)
        n_freq = len(fft_signal)
        n_mask = int(n_freq * mask_freq_ratio)
        
        # Random frequency indices to mask
        mask_indices = np.random.choice(n_freq, n_mask, replace=False)
        fft_signal[mask_indices] = 0
        
        return np.real(np.fft.ifft(fft_signal))
    
    def apply_augmentation(self, signal: np.ndarray, 
                          augmentation_prob: float = 0.8) -> np.ndarray:
        """Apply random combination of augmentations."""
        augmented = signal.copy()
        
        if np.random.random() < augmentation_prob:
            # Randomly select augmentations to apply
            transforms = []
            
            if np.random.random() < 0.5:
                transforms.append(lambda x: self.add_noise(x, 0.05))
            
            if np.random.random() < 0.3:
                transforms.append(lambda x: self.time_shift(x, 1.0))
            
            if np.random.random() < 0.4:
                transforms.append(lambda x: self.amplitude_scale(x, (0.8, 1.2)))
            
            if np.random.random() < 0.2:
                transforms.append(lambda x: self.frequency_mask(x, 0.1))
            
            # Apply selected transforms
            for transform in transforms:
                augmented = transform(augmented)
        
        return augmented


if __name__ == "__main__":
    # Test preprocessing
    preprocessor = SignalPreprocessor(fs=125)
    
    # Test with sample data
    data_dir = "/Users/eli/VscodeProjects/idea2/data"
    signals_file = f"{data_dir}/bidmc_04_Signals.csv"
    numerics_file = f"{data_dir}/bidmc_04_Numerics.csv"
    
    print("Loading test data...")
    signals_df = preprocessor.load_signals(signals_file)
    numerics_df = preprocessor.load_numerics(numerics_file)
    
    if signals_df is not None and numerics_df is not None:
        print(f"Signals shape: {signals_df.shape}")
        print(f"Numerics shape: {numerics_df.shape}")
        print(f"Signal columns: {signals_df.columns.tolist()}")
        print(f"Numeric columns: {numerics_df.columns.tolist()}")
        
        print("\nProcessing subject data...")
        processed_data = preprocessor.process_subject_data(signals_df, numerics_df)
        
        if processed_data:
            print(f"PPG windows shape: {processed_data['ppg_windows'].shape}")
            print(f"Respiratory rate shape: {processed_data['resp_rate'].shape}")
            print(f"Sample respiratory rates: {processed_data['resp_rate'][:5]}")
    else:
        print("Failed to load test data")
