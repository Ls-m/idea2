"""PyTorch Lightning DataModule for BIDMC dataset with cross-validation support."""

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
from utils.data_utils import get_subject_list, create_subject_info_dataframe, create_stratified_splits
from data.dataset import BIDMCDataset, ContrastiveDataset, MultiTaskDataset, collate_fn


class BIDMCDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for BIDMC dataset.
    Supports cross-validation with subject-wise splits and no data leakage.
    """
    
    def __init__(self,
                 data_dir: str,
                 window_size_sec: int = 30,
                 overlap: float = 0.5,
                 fs: int = 125,
                 batch_size: int = 32,
                 num_workers: int = 4,
                 n_folds: int = 5,
                 current_fold: int = 0,
                 test_ratio: float = 0.2,
                 val_ratio: float = 0.2,
                 random_seed: int = 42,
                 augment_train: bool = True,
                 augment_val: bool = False,
                 augment_test: bool = False,
                 mode: str = 'supervised',  # 'supervised', 'contrastive', 'both'
                 augmentation_strength: float = 0.8):
        """
        Initialize BIDMC DataModule.
        
        Args:
            data_dir: Directory containing BIDMC files
            window_size_sec: Window size in seconds
            overlap: Overlap ratio for windowing
            fs: Sampling frequency
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for dataloaders
            n_folds: Number of cross-validation folds
            current_fold: Current fold index (0 to n_folds-1)
            test_ratio: Ratio of subjects for testing
            val_ratio: Ratio of subjects for validation
            random_seed: Random seed for reproducible splits
            augment_train: Whether to augment training data
            augment_val: Whether to augment validation data
            augment_test: Whether to augment test data
            mode: Dataset mode ('supervised', 'contrastive', 'both')
            augmentation_strength: Strength for contrastive augmentation
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.data_dir = data_dir
        self.window_size_sec = window_size_sec
        self.overlap = overlap
        self.fs = fs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.n_folds = n_folds
        self.current_fold = current_fold
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self.random_seed = random_seed
        self.augment_train = augment_train
        self.augment_val = augment_val
        self.augment_test = augment_test
        self.mode = mode
        self.augmentation_strength = augmentation_strength
        
        # Initialize splits
        self.folds = None
        self.train_subjects = None
        self.val_subjects = None
        self.test_subjects = None
        
        # Datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
    def prepare_data(self):
        """
        Prepare data by creating subject-wise cross-validation splits.
        This ensures no data leakage between train/val/test sets.
        """
        print("Preparing BIDMC data splits...")
        
        # Get all subjects
        subjects = get_subject_list(self.data_dir)
        print(f"Found {len(subjects)} subjects")
        
        # Create subject info DataFrame
        subjects_df = create_subject_info_dataframe(self.data_dir)
        
        # Create stratified splits
        self.folds = create_stratified_splits(
            subjects_df,
            n_folds=self.n_folds,
            test_ratio=self.test_ratio,
            val_ratio=self.val_ratio,
            random_seed=self.random_seed
        )
        
        # Get current fold splits
        if self.current_fold < len(self.folds):
            current_split = self.folds[self.current_fold]
            self.train_subjects = current_split['train']
            self.val_subjects = current_split['val']
            self.test_subjects = current_split['test']
            
            print(f"Fold {self.current_fold + 1}/{self.n_folds}:")
            print(f"  Train subjects: {len(self.train_subjects)}")
            print(f"  Val subjects: {len(self.val_subjects)}")
            print(f"  Test subjects: {len(self.test_subjects)}")
            
            # Verify no overlap
            train_set = set(self.train_subjects)
            val_set = set(self.val_subjects)
            test_set = set(self.test_subjects)
            
            assert len(train_set & val_set) == 0, "Train and validation subjects overlap!"
            assert len(train_set & test_set) == 0, "Train and test subjects overlap!"
            assert len(val_set & test_set) == 0, "Validation and test subjects overlap!"
            
            print("✓ No data leakage detected in splits")
        else:
            raise ValueError(f"Invalid fold index: {self.current_fold} (max: {len(self.folds)-1})")
    
    def setup(self, stage: Optional[str] = None):
        """Setup datasets for train/val/test."""
        if self.folds is None:
            self.prepare_data()
        
        if stage == "fit" or stage is None:
            # Training dataset
            if self.mode == 'supervised':
                self.train_dataset = BIDMCDataset(
                    data_dir=self.data_dir,
                    subject_ids=self.train_subjects,
                    window_size_sec=self.window_size_sec,
                    overlap=self.overlap,
                    fs=self.fs,
                    augment=self.augment_train,
                    return_subject_info=True
                )
            elif self.mode == 'contrastive':
                self.train_dataset = ContrastiveDataset(
                    data_dir=self.data_dir,
                    subject_ids=self.train_subjects,
                    window_size_sec=self.window_size_sec,
                    overlap=self.overlap,
                    fs=self.fs,
                    augmentation_strength=self.augmentation_strength
                )
            else:  # 'both'
                self.train_dataset = MultiTaskDataset(
                    data_dir=self.data_dir,
                    subject_ids=self.train_subjects,
                    window_size_sec=self.window_size_sec,
                    overlap=self.overlap,
                    fs=self.fs,
                    mode='both',
                    augmentation_strength=self.augmentation_strength
                )
            
            # Validation dataset (always supervised for evaluation)
            self.val_dataset = BIDMCDataset(
                data_dir=self.data_dir,
                subject_ids=self.val_subjects,
                window_size_sec=self.window_size_sec,
                overlap=self.overlap,
                fs=self.fs,
                augment=self.augment_val,
                return_subject_info=True
            )
        
        if stage == "test" or stage is None:
            # Test dataset (always supervised for evaluation)
            self.test_dataset = BIDMCDataset(
                data_dir=self.data_dir,
                subject_ids=self.test_subjects,
                window_size_sec=self.window_size_sec,
                overlap=self.overlap,
                fs=self.fs,
                augment=self.augment_test,
                return_subject_info=True
            )
    
    def train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def test_dataloader(self) -> DataLoader:
        """Create test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def predict_dataloader(self) -> DataLoader:
        """Create prediction dataloader (same as test)."""
        return self.test_dataloader()
    
    def get_fold_info(self) -> Dict[str, List[str]]:
        """Get current fold information."""
        return {
            'train_subjects': self.train_subjects,
            'val_subjects': self.val_subjects,
            'test_subjects': self.test_subjects,
            'current_fold': self.current_fold,
            'total_folds': self.n_folds
        }
    
    def set_mode(self, mode: str):
        """Change dataset mode and recreate datasets."""
        assert mode in ['supervised', 'contrastive', 'both']
        self.mode = mode
        
        # Recreate training dataset with new mode
        if self.train_subjects is not None:
            self.setup('fit')
    
    def get_dataset_stats(self) -> Dict[str, int]:
        """Get dataset statistics."""
        stats = {}
        
        if self.train_dataset is not None:
            stats['train_samples'] = len(self.train_dataset)
        if self.val_dataset is not None:
            stats['val_samples'] = len(self.val_dataset)
        if self.test_dataset is not None:
            stats['test_samples'] = len(self.test_dataset)
        
        stats['train_subjects'] = len(self.train_subjects) if self.train_subjects else 0
        stats['val_subjects'] = len(self.val_subjects) if self.val_subjects else 0
        stats['test_subjects'] = len(self.test_subjects) if self.test_subjects else 0
        
        return stats


class CrossValidationDataModule:
    """
    Wrapper for running k-fold cross-validation with BIDMCDataModule.
    """
    
    def __init__(self, **datamodule_kwargs):
        """
        Initialize cross-validation wrapper.
        
        Args:
            **datamodule_kwargs: Arguments for BIDMCDataModule
        """
        self.datamodule_kwargs = datamodule_kwargs
        self.n_folds = datamodule_kwargs.get('n_folds', 5)
        self.fold_results = []
    
    def get_fold_datamodule(self, fold_idx: int) -> BIDMCDataModule:
        """Get datamodule for specific fold."""
        kwargs = self.datamodule_kwargs.copy()
        kwargs['current_fold'] = fold_idx
        print(kwargs)
        return BIDMCDataModule(**kwargs)
    
    def run_all_folds(self, train_func, **train_kwargs):
        """
        Run training for all folds.
        
        Args:
            train_func: Training function that takes (datamodule, fold_idx, **train_kwargs)
            **train_kwargs: Additional arguments for training function
        """
        self.fold_results = []
        
        for fold_idx in range(self.n_folds):
            print(f"\n{'='*50}")
            print(f"Starting Fold {fold_idx + 1}/{self.n_folds}")
            print(f"{'='*50}")
            
            # Get datamodule for this fold
            datamodule = self.get_fold_datamodule(fold_idx)
            print('in here and datamodule is, ',datamodule)
            
            # Run training
            fold_result = train_func(datamodule, fold_idx, **train_kwargs)
            self.fold_results.append(fold_result)
            
            print(f"Completed Fold {fold_idx + 1}/{self.n_folds}")
        
        return self.fold_results
    
    def get_average_results(self) -> Dict[str, float]:
        """Calculate average results across all folds."""
        if not self.fold_results:
            return {}
        
        # Aggregate results
        aggregated = {}
        for result in self.fold_results:
            if isinstance(result, dict):
                for key, value in result.items():
                    if isinstance(value, (int, float)):
                        if key not in aggregated:
                            aggregated[key] = []
                        aggregated[key].append(value)
        
        # Calculate averages and standard deviations
        avg_results = {}
        for key, values in aggregated.items():
            avg_results[f"{key}_mean"] = sum(values) / len(values)
            if len(values) > 1:
                variance = sum((x - avg_results[f"{key}_mean"]) ** 2 for x in values) / (len(values) - 1)
                avg_results[f"{key}_std"] = variance ** 0.5
            else:
                avg_results[f"{key}_std"] = 0.0
        
        return avg_results


if __name__ == "__main__":
    # Test DataModule
    data_dir = "/Users/eli/VscodeProjects/idea2/data"
    
    print("Testing BIDMCDataModule...")
    datamodule = BIDMCDataModule(
        data_dir=data_dir,
        window_size_sec=30,
        batch_size=4,
        n_folds=3,
        current_fold=0,
        num_workers=0,  # Use 0 for testing
        mode='supervised'
    )
    
    # Setup data
    datamodule.prepare_data()
    datamodule.setup('fit')
    
    # Get fold info
    fold_info = datamodule.get_fold_info()
    print(f"Fold info: {fold_info}")
    
    # Get dataset stats
    stats = datamodule.get_dataset_stats()
    print(f"Dataset stats: {stats}")
    
    # Test dataloaders
    if stats.get('train_samples', 0) > 0:
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()
        
        print(f"Train loader batches: {len(train_loader)}")
        print(f"Val loader batches: {len(val_loader)}")
        
        # Test a batch
        batch = next(iter(train_loader))
        print(f"Batch keys: {list(batch.keys())}")
        print(f"Batch PPG shape: {batch['ppg'].shape}")
        print(f"Batch resp_rate shape: {batch['resp_rate'].shape}")
    
    print("\nTesting CrossValidationDataModule...")
    cv_datamodule = CrossValidationDataModule(
        data_dir=data_dir,
        window_size_sec=30,
        batch_size=4,
        n_folds=3,
        num_workers=0,
        mode='supervised'
    )
    
    # Test getting fold datamodule
    fold_dm = cv_datamodule.get_fold_datamodule(0)
    print(f"Created fold datamodule: {type(fold_dm)}")
    
    print("DataModule tests completed!")
