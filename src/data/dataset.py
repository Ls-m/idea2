"""Dataset classes for BIDMC respiratory rate estimation."""

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from data.preprocessing import SignalPreprocessor, AugmentationTransforms
from utils.data_utils import parse_subject_info, validate_subject_files
import time

class BIDMCDataset(Dataset):
    """Dataset class for BIDMC PPG and respiratory rate data."""
    
    def __init__(self, 
                 data_dir: str,
                 subject_ids: List[str],
                 window_size_sec: int = 30,
                 overlap: float = 0.5,
                 fs: int = 125,
                 augment: bool = False,
                 return_subject_info: bool = False):
        """
        Initialize BIDMC dataset.
        
        Args:
            data_dir: Directory containing BIDMC files
            subject_ids: List of subject IDs to include
            window_size_sec: Window size in seconds
            overlap: Overlap ratio for windows
            fs: Sampling frequency
            augment: Whether to apply data augmentation
            return_subject_info: Whether to return subject metadata
        """
        self.data_dir = Path(data_dir)
        self.subject_ids = subject_ids
        self.window_size_sec = window_size_sec
        self.overlap = overlap
        self.fs = fs
        self.augment = augment
        self.return_subject_info = return_subject_info
        
        # Initialize preprocessor
        self.preprocessor = SignalPreprocessor(fs=fs)
        
        # Initialize augmentation
        if augment:
            self.augmentation = AugmentationTransforms(fs=fs)
        
        # Load and process all data
        self.data_cache = {}
        self.samples = []
        self._load_all_data()
        
        print(f"Loaded {len(self.samples)} samples from {len(subject_ids)} subjects")
    
    def _load_all_data(self):
        """Load and preprocess all subject data."""
        for subject_id in self.subject_ids:
            try:
                # Validate files exist
                if not validate_subject_files(str(self.data_dir), subject_id):
                    print(f"Skipping {subject_id}: missing files")
                    continue
                
                # Load signals and numerics
                signals_file = self.data_dir / f"{subject_id}_Signals.csv"
                numerics_file = self.data_dir / f"{subject_id}_Numerics.csv"
                fix_file = self.data_dir / f"{subject_id}_Fix.txt"
                
                signals_df = self.preprocessor.load_signals(str(signals_file))
                numerics_df = self.preprocessor.load_numerics(str(numerics_file))
                
                if signals_df is None or numerics_df is None:
                    print(f"Skipping {subject_id}: failed to load data")
                    continue
                
                # Process subject data
                processed_data = self.preprocessor.process_subject_data(
                    signals_df, numerics_df, self.window_size_sec
                )
                
                if processed_data is None:
                    print(f"Skipping {subject_id}: failed to process data")
                    continue
                
                # Store processed data
                self.data_cache[subject_id] = processed_data
                
                # Parse subject info if needed
                if self.return_subject_info:
                    subject_info = parse_subject_info(str(fix_file))
                    self.data_cache[subject_id]['subject_info'] = subject_info
                
                # Create sample indices
                n_windows = processed_data['ppg_windows'].shape[0]
                for i in range(n_windows):
                    # Filter out invalid respiratory rates
                    resp_rate = processed_data['resp_rate'][i]
                    if 5 <= resp_rate <= 50:  # Valid range for human respiratory rate
                        self.samples.append((subject_id, i))
                
            except Exception as e:
                print(f"Error loading {subject_id}: {e}")
                continue
    
    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample by index."""
        start = time.time()
        subject_id, window_idx = self.samples[idx]
        data = self.data_cache[subject_id]
        
        # Get PPG window and respiratory rate
        ppg_window = data['ppg_windows'][window_idx].astype(np.float32)
        resp_rate = data['resp_rate'][window_idx].astype(np.float32)
        
        # Get respiratory modulation if available
        if 'resp_mod_windows' in data:
            resp_mod_window = data['resp_mod_windows'][window_idx].astype(np.float32)
        else:
            resp_mod_window = np.zeros_like(ppg_window)
        
        # Apply augmentation if enabled
        if self.augment and hasattr(self, 'augmentation'):
            ppg_window = self.augmentation.apply_augmentation(ppg_window)
        
        # Create sample dictionary
        sample = {
            'ppg': torch.tensor(ppg_window, dtype=torch.float32),
            'resp_mod': torch.tensor(resp_mod_window, dtype=torch.float32),
            'resp_rate': torch.tensor(resp_rate, dtype=torch.float32),
            'subject_id': subject_id,
            'window_idx': window_idx
        }
        
        # Add subject info if requested
        if self.return_subject_info and 'subject_info' in data:
            info = data['subject_info']
            sample.update({
                'age': torch.tensor(info.get('age', 60), dtype=torch.float32),
                'gender': info.get('gender', 'U'),
                'location': info.get('location', 'unknown')
            })
        end = time.time()
        print(f"Data loading time for {subject_id}: {end - start:.4f} seconds")
        return sample


class ContrastiveDataset(Dataset):
    """Dataset for contrastive learning (unlabeled or with labels)."""
    
    def __init__(self,
                 data_dir: str,
                 subject_ids: List[str],
                 window_size_sec: int = 30,
                 overlap: float = 0.5,
                 fs: int = 125,
                 augmentation_strength: float = 0.8):
        """
        Initialize contrastive dataset.
        
        Args:
            data_dir: Directory containing BIDMC files
            subject_ids: List of subject IDs
            window_size_sec: Window size in seconds
            overlap: Overlap ratio
            fs: Sampling frequency
            augmentation_strength: Strength of augmentation (0-1)
        """
        self.data_dir = Path(data_dir)
        self.subject_ids = subject_ids
        self.window_size_sec = window_size_sec
        self.overlap = overlap
        self.fs = fs
        self.augmentation_strength = augmentation_strength
        
        # Initialize preprocessor and augmentation
        self.preprocessor = SignalPreprocessor(fs=fs)
        self.augmentation = AugmentationTransforms(fs=fs)
        
        # Load data
        self.data_cache = {}
        self.samples = []
        self._load_all_data()
        
        print(f"Contrastive dataset: {len(self.samples)} samples from {len(subject_ids)} subjects")
    
    def _load_all_data(self):
        """Load all subject data for contrastive learning."""
        for subject_id in self.subject_ids:
            try:
                if not validate_subject_files(str(self.data_dir), subject_id):
                    continue
                
                signals_file = self.data_dir / f"{subject_id}_Signals.csv"
                numerics_file = self.data_dir / f"{subject_id}_Numerics.csv"
                
                signals_df = self.preprocessor.load_signals(str(signals_file))
                numerics_df = self.preprocessor.load_numerics(str(numerics_file))
                
                if signals_df is None or numerics_df is None:
                    continue
                
                processed_data = self.preprocessor.process_subject_data(
                    signals_df, numerics_df, self.window_size_sec
                )
                
                if processed_data is None:
                    continue
                
                self.data_cache[subject_id] = processed_data
                
                # Add all windows (including those with invalid resp rates for SSL)
                n_windows = processed_data['ppg_windows'].shape[0]
                for i in range(n_windows):
                    self.samples.append((subject_id, i))
                    
            except Exception as e:
                print(f"Error loading {subject_id} for contrastive learning: {e}")
                continue
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get contrastive sample with two augmented views."""
        subject_id, window_idx = self.samples[idx]
        data = self.data_cache[subject_id]
        
        # Get PPG window
        ppg_window = data['ppg_windows'][window_idx].astype(np.float32)
        resp_rate = data['resp_rate'][window_idx].astype(np.float32)
        
        # Create two augmented views
        view1 = self.augmentation.apply_augmentation(
            ppg_window, self.augmentation_strength
        )
        view2 = self.augmentation.apply_augmentation(
            ppg_window, self.augmentation_strength
        )
        
        sample = {
            'view1': torch.tensor(view1, dtype=torch.float32),
            'view2': torch.tensor(view2, dtype=torch.float32),
            'original': torch.tensor(ppg_window, dtype=torch.float32),
            'resp_rate': torch.tensor(resp_rate, dtype=torch.float32),
            'subject_id': subject_id,
            'window_idx': window_idx
        }
        
        return sample


class MultiTaskDataset(Dataset):
    """Dataset that supports both supervised and contrastive learning."""
    
    def __init__(self,
                 data_dir: str,
                 subject_ids: List[str],
                 window_size_sec: int = 30,
                 overlap: float = 0.5,
                 fs: int = 125,
                 mode: str = 'supervised',  # 'supervised', 'contrastive', 'both'
                 augmentation_strength: float = 0.8):
        """
        Initialize multi-task dataset.
        
        Args:
            data_dir: Directory containing BIDMC files
            subject_ids: List of subject IDs
            window_size_sec: Window size in seconds
            overlap: Overlap ratio
            fs: Sampling frequency
            mode: Dataset mode ('supervised', 'contrastive', 'both')
            augmentation_strength: Augmentation strength for contrastive learning
        """
        self.mode = mode
        self.augmentation_strength = augmentation_strength
        
        # Initialize base dataset
        self.supervised_dataset = BIDMCDataset(
            data_dir=data_dir,
            subject_ids=subject_ids,
            window_size_sec=window_size_sec,
            overlap=overlap,
            fs=fs,
            augment=(mode in ['supervised', 'both']),
            return_subject_info=True
        )
        
        if mode in ['contrastive', 'both']:
            self.contrastive_dataset = ContrastiveDataset(
                data_dir=data_dir,
                subject_ids=subject_ids,
                window_size_sec=window_size_sec,
                overlap=overlap,
                fs=fs,
                augmentation_strength=augmentation_strength
            )
    
    def set_mode(self, mode: str):
        """Change dataset mode."""
        assert mode in ['supervised', 'contrastive', 'both']
        self.mode = mode
    
    def __len__(self) -> int:
        if self.mode == 'supervised':
            return len(self.supervised_dataset)
        elif self.mode == 'contrastive':
            return len(self.contrastive_dataset)
        else:  # both
            return len(self.supervised_dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.mode == 'supervised':
            return self.supervised_dataset[idx]
        elif self.mode == 'contrastive':
            return self.contrastive_dataset[idx]
        else:  # both
            # Return both supervised and contrastive samples
            sup_sample = self.supervised_dataset[idx]
            
            # Create contrastive views from the same PPG signal
            ppg_signal = sup_sample['ppg'].numpy()
            augmentation = AugmentationTransforms(fs=125)
            
            view1 = augmentation.apply_augmentation(ppg_signal, self.augmentation_strength)
            view2 = augmentation.apply_augmentation(ppg_signal, self.augmentation_strength)
            
            sup_sample.update({
                'contrastive_view1': torch.tensor(view1, dtype=torch.float32),
                'contrastive_view2': torch.tensor(view2, dtype=torch.float32)
            })
            
            return sup_sample


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for handling variable-length sequences and metadata."""
    # Separate tensor and non-tensor data
    tensor_keys = ['ppg', 'resp_mod', 'resp_rate', 'age', 'view1', 'view2', 'original', 
                   'contrastive_view1', 'contrastive_view2']
    string_keys = ['subject_id', 'gender', 'location']
    int_keys = ['window_idx']
    
    collated = {}
    
    # Handle tensor data
    for key in tensor_keys:
        if key in batch[0]:
            collated[key] = torch.stack([item[key] for item in batch])
    
    # Handle string data
    for key in string_keys:
        if key in batch[0]:
            collated[key] = [item[key] for item in batch]
    
    # Handle integer data
    for key in int_keys:
        if key in batch[0]:
            collated[key] = torch.tensor([item[key] for item in batch])
    
    return collated


if __name__ == "__main__":
    # Test dataset classes
    data_dir = "/Users/eli/VscodeProjects/idea2/data"
    test_subjects = ['bidmc_01', 'bidmc_02', 'bidmc_03']
    
    print("Testing BIDMCDataset...")
    dataset = BIDMCDataset(
        data_dir=data_dir,
        subject_ids=test_subjects,
        window_size_sec=30,
        overlap=0.5,
        augment=True,
        return_subject_info=True
    )
    
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample keys: {list(sample.keys())}")
        print(f"PPG shape: {sample['ppg'].shape}")
        print(f"Respiratory rate: {sample['resp_rate'].item():.2f}")
        print(f"Subject ID: {sample['subject_id']}")
        if 'age' in sample:
            print(f"Age: {sample['age'].item()}")
        
        print(f"\nDataset size: {len(dataset)}")
    
    print("\nTesting ContrastiveDataset...")
    contrastive_dataset = ContrastiveDataset(
        data_dir=data_dir,
        subject_ids=test_subjects,
        window_size_sec=30,
        augmentation_strength=0.8
    )
    
    if len(contrastive_dataset) > 0:
        contrastive_sample = contrastive_dataset[0]
        print(f"Contrastive sample keys: {list(contrastive_sample.keys())}")
        print(f"View1 shape: {contrastive_sample['view1'].shape}")
        print(f"View2 shape: {contrastive_sample['view2'].shape}")
    
    print("\nTesting MultiTaskDataset...")
    multitask_dataset = MultiTaskDataset(
        data_dir=data_dir,
        subject_ids=test_subjects,
        mode='both',
        augmentation_strength=0.8
    )
    
    if len(multitask_dataset) > 0:
        multitask_sample = multitask_dataset[0]
        print(f"MultiTask sample keys: {list(multitask_sample.keys())}")
    
    # Test collate function
    from torch.utils.data import DataLoader
    
    if len(dataset) > 0:
        dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn, shuffle=False)
        batch = next(iter(dataloader))
        print(f"\nBatch keys: {list(batch.keys())}")
        print(f"Batch PPG shape: {batch['ppg'].shape}")
        print(f"Batch resp_rate shape: {batch['resp_rate'].shape}")
