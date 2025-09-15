"""Data utilities for BIDMC dataset preprocessing and subject information extraction."""

import os
import re
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path


def parse_subject_info(fix_file_path: str) -> Dict[str, str]:
    """
    Parse subject information from _Fix.txt files.
    
    Args:
        fix_file_path: Path to the _Fix.txt file
        
    Returns:
        Dictionary containing subject information
    """
    info = {}
    
    try:
        with open(fix_file_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            line = line.strip()
            if line.startswith('Age:'):
                info['age'] = int(line.split(':')[1].strip())
            elif line.startswith('Gender:'):
                info['gender'] = line.split(':')[1].strip()
            elif line.startswith('Location:'):
                info['location'] = line.split(':')[1].strip()
            elif line.startswith('Signals sampling frequency:'):
                info['signal_fs'] = int(line.split(':')[1].strip().split()[0])
            elif line.startswith('Numerics sampling frequency:'):
                info['numeric_fs'] = int(line.split(':')[1].strip().split()[0])
                
    except Exception as e:
        print(f"Error parsing {fix_file_path}: {e}")
        # Set default values
        info = {
            'age': 60,  # default age
            'gender': 'U',  # unknown
            'location': 'unknown',
            'signal_fs': 125,
            'numeric_fs': 1
        }
    
    return info


def get_subject_list(data_dir: str) -> List[str]:
    """
    Get list of all subjects from the data directory.
    
    Args:
        data_dir: Directory containing BIDMC files
        
    Returns:
        List of subject IDs (e.g., ['bidmc_01', 'bidmc_02', ...])
    """
    data_path = Path(data_dir)
    
    # Find all _Fix.txt files to get subject list
    fix_files = list(data_path.glob("bidmc_*_Fix.txt"))
    
    subjects = []
    for fix_file in fix_files:
        # Extract subject ID from filename
        match = re.match(r'bidmc_(\d+)_Fix\.txt', fix_file.name)
        if match:
            subject_id = f"bidmc_{match.group(1).zfill(2)}"
            subjects.append(subject_id)
    
    return sorted(subjects)


def validate_subject_files(data_dir: str, subject_id: str) -> bool:
    """
    Validate that all required files exist for a subject.
    
    Args:
        data_dir: Directory containing BIDMC files
        subject_id: Subject ID (e.g., 'bidmc_01')
        
    Returns:
        True if all files exist, False otherwise
    """
    data_path = Path(data_dir)
    
    required_files = [
        f"{subject_id}_Fix.txt",
        f"{subject_id}_Signals.csv",
        f"{subject_id}_Numerics.csv"
    ]
    
    for file_name in required_files:
        if not (data_path / file_name).exists():
            print(f"Missing file: {file_name}")
            return False
    
    return True


def create_subject_info_dataframe(data_dir: str) -> pd.DataFrame:
    """
    Create a DataFrame with all subject information.
    
    Args:
        data_dir: Directory containing BIDMC files
        
    Returns:
        DataFrame with subject information
    """
    subjects = get_subject_list(data_dir)
    subject_data = []
    
    for subject_id in subjects:
        if validate_subject_files(data_dir, subject_id):
            fix_file = Path(data_dir) / f"{subject_id}_Fix.txt"
            info = parse_subject_info(str(fix_file))
            info['subject_id'] = subject_id
            subject_data.append(info)
        else:
            print(f"Skipping {subject_id} due to missing files")
    
    df = pd.DataFrame(subject_data)
    return df


def create_stratified_splits(subjects_df: pd.DataFrame, 
                           n_folds: int = 5, 
                           test_ratio: float = 0.2,
                           val_ratio: float = 0.2,
                           random_seed: int = 42) -> List[Dict[str, List[str]]]:
    """
    Create stratified subject-wise splits for cross-validation.
    Stratification is based on age groups and gender.
    
    Args:
        subjects_df: DataFrame with subject information
        n_folds: Number of cross-validation folds
        test_ratio: Ratio of subjects for testing
        val_ratio: Ratio of subjects for validation
        random_seed: Random seed for reproducibility
        
    Returns:
        List of dictionaries, each containing train/val/test subject lists
    """
    np.random.seed(random_seed)
    
    # Create stratification groups based on age and gender
    subjects_df = subjects_df.copy()
    
    # Age groups: <40, 40-60, 60-80, >80
    subjects_df['age_group'] = pd.cut(subjects_df['age'], 
                                    bins=[0, 40, 60, 80, 100], 
                                    labels=['young', 'middle', 'senior', 'elderly'])
    
    # Create stratification key
    subjects_df['strat_key'] = subjects_df['age_group'].astype(str) + '_' + subjects_df['gender']
    
    # Get unique stratification groups
    strat_groups = subjects_df['strat_key'].unique()
    
    folds = []
    subjects = subjects_df['subject_id'].tolist()
    n_subjects = len(subjects)
    
    # Calculate split sizes
    n_test = max(1, int(n_subjects * test_ratio))
    n_val = max(1, int(n_subjects * val_ratio))
    
    for fold in range(n_folds):
        # Shuffle subjects for this fold
        fold_subjects = subjects.copy()
        np.random.shuffle(fold_subjects)
        
        # Create stratified splits
        test_subjects = []
        val_subjects = []
        train_subjects = []
        
        # For each stratification group, distribute subjects across splits
        for group in strat_groups:
            group_subjects = subjects_df[subjects_df['strat_key'] == group]['subject_id'].tolist()
            group_subjects = [s for s in group_subjects if s in fold_subjects]
            
            if len(group_subjects) == 0:
                continue
            
            np.random.shuffle(group_subjects)
            
            # Calculate group-specific split sizes
            group_n_test = max(1, int(len(group_subjects) * test_ratio))
            group_n_val = max(1, int(len(group_subjects) * val_ratio))
            
            # Ensure we don't exceed group size
            if group_n_test + group_n_val >= len(group_subjects):
                group_n_test = max(1, len(group_subjects) // 3)
                group_n_val = max(1, len(group_subjects) // 3)
            
            # Assign subjects to splits
            test_subjects.extend(group_subjects[:group_n_test])
            val_subjects.extend(group_subjects[group_n_test:group_n_test + group_n_val])
            train_subjects.extend(group_subjects[group_n_test + group_n_val:])
        
        # If we don't have enough subjects in test/val, redistribute
        all_assigned = set(test_subjects + val_subjects + train_subjects)
        remaining = [s for s in fold_subjects if s not in all_assigned]
        
        # Add remaining subjects to train
        train_subjects.extend(remaining)
        
        # Ensure minimum sizes
        if len(test_subjects) == 0 and len(train_subjects) > 0:
            test_subjects.append(train_subjects.pop())
        if len(val_subjects) == 0 and len(train_subjects) > 0:
            val_subjects.append(train_subjects.pop())
        
        fold_dict = {
            'train': train_subjects,
            'val': val_subjects,
            'test': test_subjects
        }
        
        folds.append(fold_dict)
        
        print(f"Fold {fold + 1}: Train={len(train_subjects)}, Val={len(val_subjects)}, Test={len(test_subjects)}")
    
    return folds


def check_data_leakage(folds: List[Dict[str, List[str]]]) -> bool:
    """
    Check for data leakage across folds.
    
    Args:
        folds: List of fold dictionaries
        
    Returns:
        True if no leakage detected, False otherwise
    """
    all_subjects = set()
    
    for i, fold in enumerate(folds):
        fold_subjects = set(fold['train'] + fold['val'] + fold['test'])
        
        # Check for duplicates within fold
        if len(fold_subjects) != len(fold['train']) + len(fold['val']) + len(fold['test']):
            print(f"Fold {i}: Subject appears in multiple splits within fold")
            return False
        
        # Check for overlap with other folds would be complex and not necessary
        # since we're doing cross-validation where subjects can appear in different folds
        
        all_subjects.update(fold_subjects)
    
    print(f"Data leakage check passed. Total unique subjects across all folds: {len(all_subjects)}")
    return True


if __name__ == "__main__":
    # Test the data utilities
    data_dir = "/Users/eli/VscodeProjects/idea2/data"
    
    print("Getting subject list...")
    subjects = get_subject_list(data_dir)
    print(f"Found {len(subjects)} subjects: {subjects[:5]}...")
    
    print("\nCreating subject info DataFrame...")
    subjects_df = create_subject_info_dataframe(data_dir)
    print(f"Subject info shape: {subjects_df.shape}")
    print("\nSubject info sample:")
    print(subjects_df.head())
    
    print("\nCreating stratified splits...")
    folds = create_stratified_splits(subjects_df, n_folds=5)
    
    print("\nChecking for data leakage...")
    check_data_leakage(folds)
