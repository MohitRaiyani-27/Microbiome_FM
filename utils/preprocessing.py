"""
Data preprocessing pipeline
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.impute import SimpleImputer
import json
import sys
sys.path.append(str(Path(__file__).parent.parent))
import config


def clean_feature_name(name):
    """Clean feature name to be PyTorch-compatible."""
    # Replace problematic characters
    name = str(name)
    name = name.replace(':', '_')
    name = name.replace('.', '_')
    name = name.replace(' ', '_')
    name = name.replace('(', '_')
    name = name.replace(')', '_')
    name = name.replace('-', '_')
    name = name.replace('/', '_')
    name = name.replace('[', '_')
    name = name.replace(']', '_')
    name = name.replace('{', '_')
    name = name.replace('}', '_')
    name = name.replace('=', '_')
    name = name.replace('+', '_')
    name = name.replace('*', '_')
    name = name.replace('&', '_')
    name = name.replace('%', '_')
    name = name.replace('$', '_')
    name = name.replace('#', '_')
    name = name.replace('@', '_')
    name = name.replace('!', '_')
    name = name.replace(';', '_')
    name = name.replace(',', '_')
    # Remove multiple underscores
    while '__' in name:
        name = name.replace('__', '_')
    # Remove leading/trailing underscores
    name = name.strip('_')
    return name


class DataPreprocessor:
    """Complete data preprocessing pipeline."""
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = None
        self.categorical_features = []
        self.numerical_features = []
        self.all_features = []
        
    def load_data(self, data_path):
        """Load CSV data."""
        print(f"📂 Loading data from {data_path}...")
        df = pd.read_csv(data_path)
        
        # Clean column names
        df.columns = [clean_feature_name(col) for col in df.columns]
        
        print(f"✓ Loaded {len(df)} samples with {len(df.columns)} columns")
        return df
    
    def remove_high_missing(self, df):
        """Remove features with too many missing values."""
        missing_pct = df.isnull().sum() / len(df)
        to_drop = missing_pct[missing_pct > config.MISSING_THRESHOLD].index.tolist()
        
        if to_drop:
            print(f"✓ Dropping {len(to_drop)} features with >{config.MISSING_THRESHOLD*100:.0f}% missing")
            df = df.drop(columns=to_drop)
        
        return df
    
    def remove_specified_features(self, df):
        """Remove specified features."""
        to_drop = [f for f in config.FEATURES_TO_DROP if f in df.columns]
        if to_drop:
            print(f"✓ Dropping {len(to_drop)} specified features: {to_drop}")
            df = df.drop(columns=to_drop)
        return df
    
    def auto_detect_features(self, df):
        """Automatically detect categorical and numerical features."""
        # Exclude target column
        target_col = clean_feature_name(config.TARGET_COLUMN)
        feature_cols = [c for c in df.columns if c != target_col]
        
        categorical = []
        numerical = []
        
        for col in feature_cols:
            # Check if column is numeric
            if pd.api.types.is_numeric_dtype(df[col]):
                # If few unique values, treat as categorical
                if df[col].nunique() < 10:
                    categorical.append(col)
                else:
                    numerical.append(col)
            else:
                categorical.append(col)
        
        self.categorical_features = categorical
        self.numerical_features = numerical
        self.all_features = categorical + numerical
        
        print(f"✓ Detected {len(categorical)} categorical features")
        print(f"✓ Detected {len(numerical)} numerical features")
        
        return categorical, numerical
    
    def impute_missing(self, df):
        """Impute missing values."""
        df = df.copy()
        
        # Impute numerical features
        if self.numerical_features:
            num_cols = [c for c in self.numerical_features if c in df.columns]
            if num_cols:
                imputer = SimpleImputer(strategy='median')
                df[num_cols] = imputer.fit_transform(df[num_cols])
                print(f"✓ Imputed {len(num_cols)} numerical features (median)")
        
        # Impute categorical features
        if self.categorical_features:
            cat_cols = [c for c in self.categorical_features if c in df.columns]
            if cat_cols:
                for col in cat_cols:
                    df[col] = df[col].fillna('unknown')
                print(f"✓ Imputed {len(cat_cols)} categorical features (mode)")
        
        return df
    
    def encode_categorical(self, df):
        """Encode categorical features."""
        df = df.copy()
        vocab_sizes = {}
        
        # Clean target column name
        target_col = clean_feature_name(config.TARGET_COLUMN)
        
        # Encode target column
        if target_col in df.columns:
            le = LabelEncoder()
            df[target_col] = le.fit_transform(df[target_col].astype(str))
            self.label_encoders[target_col] = le
            vocab_sizes[target_col] = len(le.classes_)
            print(f"✓ Encoded target '{target_col}': {len(le.classes_)} classes")
        
        # Encode categorical features
        for col in self.categorical_features:
            if col in df.columns:
                df[col] = df[col].astype(str)
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                self.label_encoders[col] = le
                vocab_sizes[col] = len(le.classes_)
        
        print(f"✓ Encoded {len(self.categorical_features)} categorical features")
        return df, vocab_sizes
    
    def scale_numerical(self, df):
        """Scale numerical features."""
        df = df.copy()
        
        if self.numerical_features:
            num_cols = [c for c in self.numerical_features if c in df.columns]
            if num_cols:
                self.scaler = RobustScaler()
                df[num_cols] = self.scaler.fit_transform(df[num_cols])
                print(f"✓ Scaled {len(num_cols)} numerical features")
        
        return df
    
    def split_data(self, df):
        """Split data into train/val/test."""
        target_col = clean_feature_name(config.TARGET_COLUMN)
        
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found!")
        
        # Stratify on target if possible
        try:
            stratify = df[target_col]
            # Check if stratification is possible
            class_counts = stratify.value_counts()
            if class_counts.min() < 2:
                print("⚠️ Some classes have <2 samples. Using non-stratified split.")
                stratify = None
        except:
            stratify = None
        
        # First split: train+val vs test
        train_val, test = train_test_split(
            df, 
            test_size=config.TEST_RATIO,
            random_state=config.RANDOM_SEED,
            stratify=stratify
        )
        
        # Second split: train vs val
        val_size_adjusted = config.VAL_RATIO / (config.TRAIN_RATIO + config.VAL_RATIO)
        
        stratify_train_val = None
        if stratify is not None:
            try:
                stratify_train_val = train_val[target_col]
            except:
                pass
        
        train, val = train_test_split(
            train_val,
            test_size=val_size_adjusted,
            random_state=config.RANDOM_SEED,
            stratify=stratify_train_val
        )
        
        print(f"✓ Split data: Train={len(train)}, Val={len(val)}, Test={len(test)}")
        return train, val, test
    
    def run(self, data_path):
        """Run complete preprocessing pipeline."""
        print("\n" + "="*60)
        print("DATA PREPROCESSING PIPELINE")
        print("="*60 + "\n")
        
        # Load data
        df = self.load_data(data_path)
        
        # Remove high missing
        df = self.remove_high_missing(df)
        
        # Remove specified features
        df = self.remove_specified_features(df)
        
        # Auto-detect feature types
        self.auto_detect_features(df)
        
        # Impute missing
        df = self.impute_missing(df)
        
        # Encode categorical
        df, vocab_sizes = self.encode_categorical(df)
        
        # Scale numerical
        df = self.scale_numerical(df)
        
        # Split data
        train, val, test = self.split_data(df)
        
        # Save processed data
        train.to_csv(config.PROCESSED_DIR / 'train.csv', index=False)
        val.to_csv(config.PROCESSED_DIR / 'val.csv', index=False)
        test.to_csv(config.PROCESSED_DIR / 'test.csv', index=False)
        
        # Save metadata
        metadata = {
            'categorical_features': self.categorical_features,
            'numerical_features': self.numerical_features,
            'vocab_sizes': vocab_sizes,
            'target_column': clean_feature_name(config.TARGET_COLUMN),
            'n_features': len(self.all_features),
            'n_train': len(train),
            'n_val': len(val),
            'n_test': len(test)
        }
        
        with open(config.PROCESSED_DIR / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✅ Preprocessing complete!")
        print(f"📁 Saved to: {config.PROCESSED_DIR}")
        print("="*60 + "\n")
        
        return train, val, test, metadata


if __name__ == "__main__":
    config.set_seed()
    preprocessor = DataPreprocessor()
    train, val, test, metadata = preprocessor.run(config.DATA_FILE)
