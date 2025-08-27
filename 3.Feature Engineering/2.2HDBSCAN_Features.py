import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import logging
import joblib
from typing import Tuple, Dict

# Try to import advanced libraries, provide instructions if failed
try:
    import hdbscan
    import umap
    from sklearn.preprocessing import RobustScaler
    from sklearn.decomposition import PCA
    from sklearn.model_selection import KFold
    from sklearn.metrics import silhouette_score
except ImportError:
    print("Error: Missing required libraries. Please run: pip install hdbscan umap-learn scikit-learn")
    exit()

# --- Basic Settings ---
# Set font and ignore warnings
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SmartMoneyHDBSCAN:
    """
    A rigorous HDBSCAN entity clustering feature engineering module for Smart Money identification.
    Strictly follows the principle of dimensionality reduction first, then clustering, 
    and uses cross-validation to prevent data leakage.
    """
    
    def __init__(self, 
                 min_cluster_size: int = 15,
                 min_samples: int = 5,
                 n_splits: int = 5,
                 random_state: int = 42):
        """
        Initialize parameters
        Args:
            min_cluster_size: HDBSCAN parameter, minimum size of a cluster.
            min_samples: HDBSCAN parameter, neighborhood density of a point.
            n_splits: Number of folds for cross-validation.
            random_state: Random seed to ensure reproducibility.
        """
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.n_splits = n_splits
        self.random_state = random_state
        
        # Store models for each fold for debugging and inspection
        self.scalers_ = {}
        self.pca_models_ = {}
        self.umap_models_ = {}
        self.hdbscan_models_ = {}
        self.feature_names_ = []

    def engineer_wallet_features(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        """
        Core Step 1: Build wallet-level static behavioral features from raw transaction data.
        This is the key step to convert time-series data into static features required by HDBSCAN.
        """
        logger.info("Starting to build wallet-level static features...")
        
        trades_df['block_time'] = pd.to_datetime(trades_df['block_time'], errors='coerce')
        
        # Aggregate by wallet address
        # For performance, we use efficient Pandas aggregation methods
        agg_funcs = {
            'tx_hash': ['count'],
            'amount_usd': ['sum', 'mean', 'max', 'std'],
            'token_bought_symbol': ['nunique'],
            'token_sold_symbol': ['nunique'],
            'dex_name': ['nunique']
        }
        
        wallet_features = trades_df.groupby('wallet_address').agg(agg_funcs)
        wallet_features.columns = ['_'.join(col).strip() for col in wallet_features.columns.values]
        wallet_features = wallet_features.rename(columns={'tx_hash_count': 'total_trades'})
        
        # Calculate more complex features
        # Trading time span and active days
        time_stats = trades_df.groupby('wallet_address')['block_time'].agg(['min', 'max'])
        wallet_features['trading_span_days'] = (time_stats['max'] - time_stats['min']).dt.days + 1
        wallet_features['active_days'] = trades_df.groupby('wallet_address')['block_time'].apply(lambda x: x.dt.date.nunique())
        wallet_features['avg_trades_per_day'] = wallet_features['total_trades'] / wallet_features['active_days']
        
        # Trading volume coefficient of variation
        wallet_features['trade_size_cv'] = wallet_features['amount_usd_std'] / wallet_features['amount_usd_mean']
        
        # Fill all NaN values, which usually occur in std calculations (if only one transaction)
        wallet_features = wallet_features.fillna(0)
        
        self.feature_names_ = wallet_features.columns.tolist()
        logger.info(f"Successfully built {len(self.feature_names_)} features for {len(wallet_features)} wallets.")
        
        return wallet_features.reset_index()

    def _preprocess_and_cluster(self, X_train: np.ndarray, X_val: np.ndarray, fold_idx: int) -> Tuple:
        """
        Private method: Execute complete preprocessing -> dimensionality reduction -> clustering pipeline
        """
        # 1. Scaling (Scaler): Robust to outliers
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        self.scalers_[fold_idx] = scaler

        # 2. PCA dimensionality reduction: Keep 95% variance while decorrelating
        pca = PCA(n_components=0.95, random_state=self.random_state)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_val_pca = pca.transform(X_val_scaled)
        self.pca_models_[fold_idx] = pca
        logger.info(f"Fold-{fold_idx} PCA: Dimension reduced from {X_train_scaled.shape[1]} to {pca.n_components_}")

        # 3. UMAP dimensionality reduction: Further nonlinear reduction, preparing for HDBSCAN
        # UMAP n_components is usually small (e.g., 5-10) to capture macro structure
        umap_reducer = umap.UMAP(n_components=10, n_neighbors=30, min_dist=0.0,
                                 random_state=self.random_state)
        X_train_umap = umap_reducer.fit_transform(X_train_pca)
        # Validation set can only transform
        X_val_umap = umap_reducer.transform(X_val_pca)
        self.umap_models_[fold_idx] = umap_reducer

        # 4. HDBSCAN clustering
        clusterer = hdbscan.HDBSCAN(min_cluster_size=self.min_cluster_size,
                                    min_samples=self.min_samples,
                                    gen_min_span_tree=True,
                                    prediction_data=True,
                                    metric='euclidean')
        
        # Fit on training set
        clusterer.fit(X_train_umap)
        self.hdbscan_models_[fold_idx] = clusterer
        
        # Predict on validation set
        val_labels, val_strengths = hdbscan.approximate_predict(clusterer, X_val_umap)
        
        # Get training set labels and probabilities
        train_labels = clusterer.labels_
        train_probs = clusterer.probabilities_
        
        return train_labels, train_probs, val_labels, val_strengths

    def generate_hdbscan_features(self, wallet_features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Core Step 2: Generate HDBSCAN clustering features without data leakage using cross-validation.
        """
        logger.info(f"Starting to generate HDBSCAN features using {self.n_splits}-Fold cross-validation...")
        
        X = wallet_features_df[self.feature_names_].values
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        
        # Initialize arrays to store OOF (Out-of-Fold) results
        oof_labels = np.zeros(len(X))
        oof_probs = np.zeros(len(X))

        for fold_idx, (train_index, val_index) in enumerate(kf.split(X)):
            logger.info(f"--- Processing Fold {fold_idx+1}/{self.n_splits} ---")
            X_train, X_val = X[train_index], X[val_index]
            
            train_labels, train_probs, val_labels, val_probs = self._preprocess_and_cluster(X_train, X_val, fold_idx)
            
            # Store validation set results in OOF arrays
            oof_labels[val_index] = val_labels
            oof_probs[val_index] = val_probs

        # Add OOF results to DataFrame
        hdbscan_features = pd.DataFrame({
            'wallet_address': wallet_features_df['wallet_address'],
            'hdbscan_cluster_id': oof_labels.astype(int),
            'hdbscan_cluster_probability': oof_probs
        })
        
        # Derive other features from cluster_id
        hdbscan_features['hdbscan_is_noise'] = (hdbscan_features['hdbscan_cluster_id'] == -1).astype(int)
        
        # Calculate cluster sizes
        cluster_sizes = hdbscan_features['hdbscan_cluster_id'].value_counts().to_dict()
        hdbscan_features['hdbscan_cluster_size'] = hdbscan_features['hdbscan_cluster_id'].map(cluster_sizes)
        
        logger.info("HDBSCAN feature generation completed.")
        return hdbscan_features

def main():
    """
    Main execution function
    """
    logger.info("Starting Single Chain Smart Money Identification - HDBSCAN Feature Engineering Pipeline")

    # --- 1. Load Data ---
    # We only use transaction data to generate features required for this research
    try:
        trades_df = pd.read_csv('filtered_trades.csv')
        logger.info(f"Successfully loaded filtered_trades.csv with {len(trades_df)} transaction records.")
    except FileNotFoundError:
        logger.error("Error: 'filtered_trades.csv' not found. Please ensure the file is in the current directory.")
        return

    # --- 2. Instantiate and Run Pipeline ---
    hdbscan_processor = SmartMoneyHDBSCAN(min_cluster_size=20, min_samples=10, n_splits=5)
    
    # Step 1: Build wallet static features
    wallet_static_features = hdbscan_processor.engineer_wallet_features(trades_df)
    
    # Step 2: Generate HDBSCAN features using cross-validation
    hdbscan_cluster_features = hdbscan_processor.generate_hdbscan_features(wallet_static_features)
    
    # --- 3. Merge and Save ---
    # Merge original static features with newly generated clustering features
    final_features_df = pd.merge(wallet_static_features, hdbscan_cluster_features, on='wallet_address')

    # Save results for next steps (LSTM, GAT, XGBoost)
    wallet_static_features.to_csv('wallet_static_features.csv', index=False)
    hdbscan_cluster_features.to_csv('hdbscan_cluster_features.csv', index=False)
    final_features_df.to_csv('basic_hdbscan_for_xgb.csv', index=False)
    
    logger.info("All processes completed. Generated files:")
    logger.info("- wallet_static_features.csv: Basic static features at wallet level.")
    logger.info("- hdbscan_cluster_features.csv: Leakage-free HDBSCAN clustering features.")
    logger.info("- basic_hdbscan_for_xgb.csv: Complete merged features, ready for final models.")
    
    # --- 4. Visualize Results (Optional) ---
    logger.info("Generating clustering result visualization...")
    plt.figure(figsize=(12, 10))
    # We need to re-reduce the entire dataset for visualization purposes only, not for feature generation
    X_all = wallet_static_features[hdbscan_processor.feature_names_].values
    X_scaled = RobustScaler().fit_transform(X_all)
    X_pca = PCA(n_components=0.95).fit_transform(X_scaled)
    X_umap = umap.UMAP(n_components=2, random_state=42).fit_transform(X_pca) # Reduce to 2D for plotting

    plt.scatter(X_umap[:, 0], X_umap[:, 1], c=hdbscan_cluster_features['hdbscan_cluster_id'], s=10, cmap='Spectral')
    plt.title('HDBSCAN Clustering Results (UMAP 2D Visualization)', fontsize=16)
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.colorbar(label='Cluster ID')
    plt.savefig('hdbscan_visualization.png', dpi=300)
    logger.info("Visualization image saved as hdbscan_visualization.png")


if __name__ == "__main__":
    main()
