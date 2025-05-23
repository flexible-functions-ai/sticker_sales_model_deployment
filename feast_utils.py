import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from fastai.tabular.all import add_datepart
from feast import FeatureStore
import os

class FeastFeatureProcessor:
    """
    Utility class to work with Feast feature store for sticker sales prediction.
    
    This class handles:
    1. Converting raw data to Feast-compatible format
    2. Creating date features using FastAI
    3. Encoding categorical variables
    4. Retrieving features for training and inference
    """
    
    def __init__(self, repo_path="feature_repo", data_path="data"):
        self.repo_path = repo_path
        self.data_path = Path(data_path)
        self.data_path.mkdir(exist_ok=True)
        
        # Initialize Feast feature store
        try:
            self.fs = FeatureStore(repo_path=repo_path)
            print(f"✅ Connected to Feast feature store at {repo_path}")
            print(f"📊 Online store type: {self.fs.config.online_store.type}")
            print(f"📁 Offline store type: {self.fs.config.offline_store.type}")
        except Exception as e:
            print(f"❌ Error connecting to Feast: {e}")
            print("💡 Make sure to run 'feast apply' first to initialize the feature store")
            raise
    
    def prepare_feature_data(self, df, is_training=True):
        """
        Prepare feature data from raw sticker sales data.
        
        This method:
        1. Creates unique sticker IDs for entity joining
        2. Extracts date features using FastAI
        3. Encodes categorical variables as integers
        4. Formats data for Feast consumption
        
        Args:
            df: Raw dataframe with columns: date, country, store, product, num_sold
            is_training: Whether this is training data (affects target handling)
        """
        print("🔄 Preparing feature data for Feast...")
        
        # Create a copy to avoid modifying original
        result_df = df.copy()
        
        # Create unique sticker_id for entity join
        # This combines country, store, and product into a unique identifier
        result_df['sticker_id'] = (
            result_df['country'].astype(str) + "_" + 
            result_df['store'].astype(str) + "_" + 
            result_df['product'].astype(str)
        )
        
        # Add date features using FastAI's add_datepart
        # This extracts Year, Month, Week, Day, etc. from the date column
        print("📅 Extracting date features...")
        result_df = add_datepart(result_df, 'date', drop=False)
        
        # Create categorical encodings
        # Convert text categories to integers that ML models can use
        print("🏷️ Encoding categorical features...")
        categorical_cols = ['country', 'store', 'product']
        for col in categorical_cols:
            if col in result_df.columns:
                # Create simple label encoding
                unique_vals = sorted(result_df[col].unique())
                mapping = {val: i for i, val in enumerate(unique_vals)}
                result_df[f'{col}_encoded'] = result_df[col].map(mapping)
                print(f"   {col}: {len(unique_vals)} unique values")
        
        # Add event timestamp for Feast (required field)
        result_df['event_timestamp'] = pd.to_datetime(result_df['date'])
        
        # Select features for Feast
        feature_columns = [
            'sticker_id', 'event_timestamp',
            # Date features (only include ones that exist)
            'Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',
            'Is_month_end', 'Is_month_start', 'Is_quarter_end', 
            'Is_quarter_start', 'Is_year_end', 'Is_year_start',
            # Categorical features
            'country_encoded', 'store_encoded', 'product_encoded'
        ]
        
        # Only include columns that actually exist in the dataframe
        available_columns = [col for col in feature_columns if col in result_df.columns]
        feast_df = result_df[available_columns].copy()
        
        # Convert boolean columns to int for Feast compatibility
        bool_columns = feast_df.select_dtypes(include=['bool']).columns
        for col in bool_columns:
            feast_df[col] = feast_df[col].astype(int)
        
        print(f"✅ Prepared {len(feast_df)} rows with {len(available_columns)} feature columns")
        print(f"📝 Available feature columns: {available_columns}")
        return feast_df
    
    def save_feature_data(self, feature_df):
        """
        Save feature data to parquet file for Feast to consume.
        
        This creates the parquet file that Feast's offline store will read from.
        """
        output_path = self.data_path / "feature_data.parquet"
        feature_df.to_parquet(output_path, index=False)
        print(f"💾 Saved feature data to {output_path}")
        return output_path
    
    def get_training_features(self, entity_df):
        """
        Get historical features for training using Feast.
        
        This queries Feast's offline store to get features for training.
        
        Args:
            entity_df: DataFrame with sticker_id and event_timestamp columns
        """
        print("🎯 Retrieving training features from Feast...")
        
        # Get available features dynamically
        available_features = []
        
        try:
            feature_views = self.fs.list_feature_views()
            print(f"📋 Available feature views: {[fv.name for fv in feature_views]}")
            
            for fv in feature_views:
                for feature in fv.features:
                    available_features.append(f"{fv.name}:{feature.name}")
        except Exception as e:
            print(f"⚠️ Error listing feature views: {e}")
            # Fallback to expected features
            available_features = [
                "date_features:Year",
                "date_features:Month", 
                "date_features:Week",
                "date_features:Day",
                "date_features:Dayofweek",
                "date_features:Dayofyear",
                "categorical_features:country_encoded",
                "categorical_features:store_encoded", 
                "categorical_features:product_encoded"
            ]
        
        print(f"🔍 Using features: {available_features}")
        
        # Get historical features from Feast
        training_df = self.fs.get_historical_features(
            entity_df=entity_df,
            features=available_features
        ).to_df()
        
        print(f"✅ Retrieved {len(training_df)} training samples with {len(training_df.columns)} features")
        return training_df
    
    def get_online_features(self, entity_rows):
        """
        Get online features for real-time inference from SQLite online store.
        
        This queries Feast's online store for fast inference.
        
        Args:
            entity_rows: List of dicts with entity keys, e.g. [{"sticker_id": "US_Store1_StickerA"}]
        """
        print(f"⚡ Retrieving online features for {len(entity_rows)} entities from SQLite store...")
        
        # Get available features dynamically
        available_features = []
        try:
            feature_views = self.fs.list_feature_views()
            for fv in feature_views:
                for feature in fv.features:
                    available_features.append(f"{fv.name}:{feature.name}")
        except Exception as e:
            print(f"⚠️ Error listing features: {e}")
            # Fallback
            available_features = [
                "date_features:Year",
                "date_features:Month",
                "categorical_features:country_encoded",
                "categorical_features:store_encoded",
                "categorical_features:product_encoded"
            ]
        
        print(f"🔍 Using online features: {available_features}")
        
        # Get online features from SQLite
        feature_vector = self.fs.get_online_features(
            entity_rows=entity_rows,
            features=available_features
        ).to_df()
        
        print(f"✅ Retrieved online features with shape: {feature_vector.shape}")
        return feature_vector
    
    def get_feature_info(self):
        """
        Get information about available features for debugging.
        """
        try:
            feature_views = self.fs.list_feature_views()
            info = {}
            for fv in feature_views:
                info[fv.name] = [f.name for f in fv.features]
            return info
        except Exception as e:
            print(f"Error getting feature info: {e}")
            return {}