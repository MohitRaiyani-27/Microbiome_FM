"""
PyTorch Dataset classes
"""
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class GenomicDataset(Dataset):
    """Dataset for genomic data with categorical and numerical features."""
    
    def __init__(self, data, categorical_features, numerical_features, target_column):
        """
        Args:
            data: DataFrame with processed data
            categorical_features: List of categorical feature names
            numerical_features: List of numerical feature names
            target_column: Name of target column
        """
        self.data = data.reset_index(drop=True)
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.target_column = target_column
        
        # Prepare categorical data
        self.categorical_data = {}
        for feat in categorical_features:
            if feat in data.columns:
                self.categorical_data[feat] = torch.LongTensor(data[feat].values)
        
        # Prepare numerical data
        if numerical_features:
            num_cols = [f for f in numerical_features if f in data.columns]
            if num_cols:
                self.numerical_data = torch.FloatTensor(data[num_cols].values)
            else:
                self.numerical_data = None
        else:
            self.numerical_data = None
        
        # Prepare labels
        if target_column in data.columns:
            self.labels = torch.LongTensor(data[target_column].values)
        else:
            self.labels = None
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Get categorical features
        cat_data = {feat: self.categorical_data[feat][idx] for feat in self.categorical_data}
        
        # Get numerical features
        num_data = self.numerical_data[idx] if self.numerical_data is not None else None
        
        # Get label
        label = self.labels[idx] if self.labels is not None else -1
        
        return cat_data, num_data, label


class PretrainingDataset(Dataset):
    """Dataset for pretraining with masking."""
    
    def __init__(self, data, categorical_features, numerical_features, mask_prob=0.15):
        self.data = data.reset_index(drop=True)
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.mask_prob = mask_prob
        
        # Prepare data
        self.categorical_data = {}
        for feat in categorical_features:
            if feat in data.columns:
                self.categorical_data[feat] = torch.LongTensor(data[feat].values)
        
        if numerical_features:
            num_cols = [f for f in numerical_features if f in data.columns]
            if num_cols:
                self.numerical_data = torch.FloatTensor(data[num_cols].values)
            else:
                self.numerical_data = None
        else:
            self.numerical_data = None
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Get original data
        cat_data = {feat: self.categorical_data[feat][idx].clone() 
                    for feat in self.categorical_data}
        num_data = self.numerical_data[idx].clone() if self.numerical_data is not None else None
        
        # Create masks
        cat_mask = {feat: torch.rand(1).item() < self.mask_prob for feat in cat_data}
        
        if num_data is not None:
            num_mask = torch.rand(len(num_data)) < self.mask_prob
        else:
            num_mask = None
        
        return cat_data, num_data, cat_mask, num_mask
