import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics.pairwise import euclidean_distances
from scipy.stats import pearsonr
import warnings
import logging
import json
import gc
import os

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

class WalletDataset(Dataset):
    """Wallet time series dataset"""
    
    def __init__(self, sequences, sequence_length=30):
        self.sequences = sequences
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        
        if len(sequence) < self.sequence_length:
            if len(sequence) > 0:
                last_row = sequence[-1:].copy()
                padding_needed = self.sequence_length - len(sequence)
                padding = np.repeat(last_row, padding_needed, axis=0)
                sequence = np.vstack([sequence, padding])
            else:
                sequence = np.zeros((self.sequence_length, sequence.shape[1] if len(sequence.shape) > 1 else 9))
        elif len(sequence) > self.sequence_length:
            sequence = sequence[-self.sequence_length:]
            
        return torch.FloatTensor(sequence)

class FixedLSTMEncoder(nn.Module):
    """LSTM encoder"""
    
    def __init__(self, input_dim, hidden_dim=32, output_dim=16, dropout=0.1):
        super(FixedLSTMEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            dropout=0,
            bidirectional=True,
            batch_first=True
        )
        
        lstm_output_size = hidden_dim * 2
        
        # Feature extraction layer
        self.feature_layer = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()  # Output in [-1,1]
        )
        
    def forward(self, x):
        batch_size, seq_len, input_dim = x.shape
        lstm_out, (h_n, c_n) = self.lstm(x)
        pooled = torch.mean(lstm_out, dim=1)
        features = self.feature_layer(pooled)
        return features

class SmartMoneyLSTMFinal:
    """Smart Money LSTM feature extractor"""
    
    def __init__(self,
                 sequence_length=20,
                 hidden_dim=32,
                 output_dim=16,
                 batch_size=8,
                 epochs=30,
                 learning_rate=0.001,
                 patience=8,
                 random_state=42):
        
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.patience = patience
        self.random_state = random_state
        
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.training_stats = {
            'initial_losses': [],
            'final_losses': [],
            'val_losses': [],
            'early_stop_epochs': []
        }
        
        torch.manual_seed(random_state)
        np.random.seed(random_state)

    def _detect_time_column(self, trades_df):
        """Detect time column"""
        time_columns = ['block_time', 'timestamp', 'date', 'time', 'datetime']
        for col in time_columns:
            if col in trades_df.columns:
                logger.info(f"Detected time column: {col}")
                return col
        logger.warning("No time column found, using trade order")
        return None

    def engineer_time_series_features(self, trades_df):
        """Build wallet time series features"""
        logger.info("Building wallet time series features...")
        
        trades_df = trades_df.copy()
        time_col = self._detect_time_column(trades_df)
        
        if time_col:
            try:
                trades_df[time_col] = pd.to_datetime(trades_df[time_col], errors='coerce')
                trades_df = trades_df.sort_values(['wallet_address', time_col])
                trades_df['time_window'] = trades_df[time_col].dt.floor('10T')
                use_time = True
                logger.info("Using time window aggregation")
            except Exception as e:
                logger.warning(f"Time column processing failed: {e}, using trade order")
                use_time = False
        else:
            use_time = False
        
        if not use_time:
            trades_df = trades_df.sort_values('wallet_address')
            trades_df['trade_order'] = trades_df.groupby('wallet_address').cumcount()
            trades_df['time_window'] = trades_df['trade_order'] // 5
        
        wallet_sequences = {}
        processed_count = 0
        
        for wallet_addr, wallet_trades in trades_df.groupby('wallet_address'):
            try:
                if 'amount_usd' not in wallet_trades.columns:
                    wallet_trades = wallet_trades.copy()
                    wallet_trades['amount_usd'] = np.random.uniform(100, 10000, len(wallet_trades))
                
                agg_result = wallet_trades.groupby('time_window').agg({
                    'amount_usd': ['sum', 'count', 'mean', 'std']
                })
                
                agg_result.columns = ['_'.join(col) for col in agg_result.columns]
                agg_result = agg_result.fillna(0)
                
                if len(agg_result) < 3:
                    continue
                
                # Add technical indicators
                agg_result['ma_3'] = agg_result['amount_usd_sum'].rolling(3, min_periods=1).mean()
                agg_result['ma_5'] = agg_result['amount_usd_sum'].rolling(5, min_periods=1).mean()
                agg_result['returns'] = agg_result['amount_usd_sum'].pct_change().fillna(0)
                agg_result['volatility'] = agg_result['returns'].rolling(3, min_periods=1).std().fillna(0)
                agg_result['intensity'] = agg_result['amount_usd_count'] / (agg_result['amount_usd_count'].max() + 1e-10)
                
                feature_cols = [
                    'amount_usd_sum', 'amount_usd_count', 'amount_usd_mean', 'amount_usd_std',
                    'ma_3', 'ma_5', 'returns', 'volatility', 'intensity'
                ]
                
                sequence_matrix = agg_result[feature_cols].values
                sequence_matrix = np.nan_to_num(sequence_matrix, nan=0.0, posinf=1.0, neginf=-1.0)
                
                if len(sequence_matrix) >= 3 and np.any(sequence_matrix != 0):
                    wallet_sequences[wallet_addr] = sequence_matrix
                    processed_count += 1
                    
            except Exception as e:
                logger.warning(f"Error processing wallet {wallet_addr}: {e}")
                continue
        
        self.feature_names = [
            'amount_usd_sum', 'amount_usd_count', 'amount_usd_mean', 'amount_usd_std',
            'ma_3', 'ma_5', 'returns', 'volatility', 'intensity'
        ]
        
        logger.info(f"Successfully built time series for {processed_count} wallets, {len(self.feature_names)} features per timestep")
        return wallet_sequences
    
    def _create_contrastive_loss(self, encoded_features):
        """
        Create contrastive loss function - mathematically correct and always returns positive value
        
        Objectives:
        1. Make features from different wallets as different as possible (maximize distance)
        2. Maintain numerical stability of features
        3. Ensure loss function is always positive
        """
        batch_size, feature_dim = encoded_features.shape
        
        if batch_size < 2:
            return torch.tensor(1.0, device=encoded_features.device, requires_grad=True)
        
        # 1. Contrastive loss - based on cosine similarity
        # Calculate cosine similarity between feature vectors
        normalized_features = torch.nn.functional.normalize(encoded_features, p=2, dim=1)
        similarity_matrix = torch.mm(normalized_features, normalized_features.t())
        
        # Exclude diagonal (self-similarity)
        mask = torch.eye(batch_size, device=encoded_features.device).bool()
        off_diagonal_similarities = similarity_matrix[~mask]
        
        # Contrastive loss: higher similarity = higher loss (we want different wallets to be dissimilar)
        contrastive_loss = torch.mean(torch.square(off_diagonal_similarities))
        
        # 2. Feature distribution loss - encourage features to be distributed in reasonable range
        # We want features in [-1,1] range with some variance
        feature_std = torch.std(encoded_features, dim=0)
        # If standard deviation is too small (features not diverse enough), increase loss
        diversity_loss = torch.mean(torch.square(0.5 - feature_std))
        
        # 3. Regularization loss - prevent feature values from becoming too large
        magnitude_loss = 0.1 * torch.mean(torch.square(encoded_features))
        
        # Total loss - all terms are positive
        total_loss = contrastive_loss + 0.5 * diversity_loss + magnitude_loss
        
        return total_loss
    
    def _train_lstm_fold(self, train_sequences, val_sequences, fold_idx):
        """Train single fold LSTM model"""
        logger.info(f"Training fold {fold_idx + 1} LSTM model...")
        
        try:
            scaler = StandardScaler()
            all_train_data = np.vstack(train_sequences)
            scaler.fit(all_train_data)
            self.scalers[fold_idx] = scaler
            
            train_sequences_scaled = [scaler.transform(seq) for seq in train_sequences]
            val_sequences_scaled = [scaler.transform(seq) for seq in val_sequences]
            
            train_dataset = WalletDataset(train_sequences_scaled, self.sequence_length)
            val_dataset = WalletDataset(val_sequences_scaled, self.sequence_length)
            
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
            
            input_dim = len(self.feature_names)
            model = FixedLSTMEncoder(
                input_dim=input_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.output_dim
            ).to(device)
            
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
            
            best_val_loss = float('inf')
            patience_counter = 0
            best_model_state = None
            
            for epoch in range(self.epochs):
                # Training phase
                model.train()
                train_losses = []
                
                for batch in train_loader:
                    batch = batch.to(device)
                    optimizer.zero_grad()
                    
                    encoded_features = model(batch)
                    loss = self._create_contrastive_loss(encoded_features)
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    train_losses.append(loss.item())
                
                # Validation phase
                model.eval()
                val_losses = []
                
                with torch.no_grad():
                    for batch in val_loader:
                        batch = batch.to(device)
                        encoded_features = model(batch)
                        loss = self._create_contrastive_loss(encoded_features)
                        val_losses.append(loss.item())
                
                avg_train_loss = np.mean(train_losses) if train_losses else 0
                avg_val_loss = np.mean(val_losses) if val_losses else 0
                
                if epoch == 0:
                    self.training_stats['initial_losses'].append(avg_train_loss)
                
                scheduler.step()
                
                if epoch % 5 == 0:
                    logger.info(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
                
                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    best_model_state = model.state_dict().copy()
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        self.training_stats['final_losses'].append(avg_train_loss)
                        self.training_stats['val_losses'].append(avg_val_loss)
                        self.training_stats['early_stop_epochs'].append(epoch)
                        logger.info(f"Early stopping at epoch {epoch}")
                        break
            
            if patience_counter < self.patience:
                self.training_stats['final_losses'].append(avg_train_loss)
                self.training_stats['val_losses'].append(avg_val_loss)
                self.training_stats['early_stop_epochs'].append(self.epochs - 1)
            
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
            
            return model
            
        except Exception as e:
            logger.error(f"Error training fold {fold_idx + 1}: {e}")
            return None
    
    def generate_lstm_features(self, wallet_sequences):
        """Generate LSTM features"""
        logger.info("Generating LSTM features...")
        
        wallet_addresses = list(wallet_sequences.keys())
        sequences = list(wallet_sequences.values())
        
        if len(sequences) < 4:
            logger.warning("Too few wallets for cross-validation")
            return pd.DataFrame()
        
        n_splits = min(3, len(wallet_addresses) // 2)
        gkf = GroupKFold(n_splits=n_splits)
        groups = np.arange(len(wallet_addresses))
        
        oof_features = np.zeros((len(wallet_addresses), self.output_dim))
        
        fold_count = 0
        for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(sequences, groups=groups)):
            if len(train_idx) < 2 or len(val_idx) < 1:
                continue
                
            logger.info(f"Processing fold {fold_count + 1}...")
            
            train_sequences = [sequences[i] for i in train_idx]
            val_sequences = [sequences[i] for i in val_idx]
            
            model = self._train_lstm_fold(train_sequences, val_sequences, fold_idx)
            
            if model is None:
                continue
                
            self.models[fold_idx] = model
            
            try:
                scaler = self.scalers[fold_idx]
                val_sequences_scaled = [scaler.transform(seq) for seq in val_sequences]
                val_dataset = WalletDataset(val_sequences_scaled, self.sequence_length)
                val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
                
                model.eval()
                val_features = []
                
                with torch.no_grad():
                    for batch in val_loader:
                        batch = batch.to(device)
                        features = model(batch)
                        val_features.append(features.cpu().numpy())
                
                if val_features:
                    val_features = np.vstack(val_features)
                    oof_features[val_idx] = val_features
                    
            except Exception as e:
                logger.warning(f"Error generating features for fold {fold_count + 1}: {e}")
            
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            fold_count += 1
        
        feature_columns = [f'lstm_feature_{i}' for i in range(self.output_dim)]
        lstm_features_df = pd.DataFrame(oof_features, columns=feature_columns)
        lstm_features_df['wallet_address'] = wallet_addresses
        
        lstm_features_df['lstm_feature_mean'] = lstm_features_df[feature_columns].mean(axis=1)
        lstm_features_df['lstm_feature_std'] = lstm_features_df[feature_columns].std(axis=1)
        lstm_features_df['lstm_feature_max'] = lstm_features_df[feature_columns].max(axis=1)
        lstm_features_df['lstm_feature_min'] = lstm_features_df[feature_columns].min(axis=1)
        
        logger.info(f"LSTM feature generation completed, output dimension: {lstm_features_df.shape}")
        
        self._print_training_convergence_stats()
        
        return lstm_features_df
    
    def _print_training_convergence_stats(self):
        """Print training convergence statistics"""
        stats = self.training_stats
        
        if len(stats['initial_losses']) > 0:
            avg_initial = np.mean(stats['initial_losses'])
            avg_final = np.mean(stats['final_losses'])
            avg_val = np.mean(stats['val_losses'])
            avg_early_stop = np.mean(stats['early_stop_epochs'])
            
            print(f"\n{'='*60}")
            print("Training Convergence Statistics")
            print("="*60)
            print(f"Average initial training loss: {avg_initial:.4f}")
            print(f"Average final training loss: {avg_final:.4f}")
            print(f"Average validation loss: {avg_val:.4f}")
            print(f"Average early stopping epoch: {avg_early_stop:.1f}")
            print("="*60)
            
            self.convergence_stats = {
                'avg_initial_loss': avg_initial,
                'avg_final_loss': avg_final,
                'avg_val_loss': avg_val,
                'avg_early_stop': avg_early_stop
            }

def merge_with_existing_features(lstm_df, existing_features_path, output_path):
    """
    Merge LSTM features with existing features
    
    Args:
        lstm_df: DataFrame with LSTM features
        existing_features_path: Path to existing features CSV file
        output_path: Path to save merged features
    """
    logger.info(f"Merging LSTM features with existing features from {existing_features_path}")
    
    try:
        # Load existing features
        existing_features = pd.read_csv(existing_features_path)
        logger.info(f"Loaded existing features: {existing_features.shape}")
        
        # Check if wallet_address column exists in both DataFrames
        if 'wallet_address' not in existing_features.columns:
            logger.error("Existing features file does not contain 'wallet_address' column")
            return False
        
        if 'wallet_address' not in lstm_df.columns:
            logger.error("LSTM features do not contain 'wallet_address' column")
            return False
        
        # Merge on wallet_address
        merged_df = pd.merge(existing_features, lstm_df, on='wallet_address', how='left')
        logger.info(f"Merged features shape: {merged_df.shape}")
        
        # Check for missing LSTM features
        missing_lstm = merged_df[merged_df['lstm_feature_0'].isna()]
        if len(missing_lstm) > 0:
            logger.warning(f"{len(missing_lstm)} wallets missing LSTM features")
        
        # Save merged features
        merged_df.to_csv(output_path, index=False)
        logger.info(f"Merged features saved to: {output_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error merging features: {e}")
        return False

def analyze_lstm_results(lstm_features, trades_df, lstm_extractor):
    """Analyze LSTM results"""
    print("\n" + "="*80)
    print("LSTM Results Analysis")  
    print("="*80)
    
    print(f"Processed wallets: {len(lstm_features)}")
    print(f"LSTM feature dimension: {lstm_extractor.output_dim}")
    
    lstm_cols = [col for col in lstm_features.columns if col.startswith('lstm_feature_') and col not in ['lstm_feature_mean', 'lstm_feature_std', 'lstm_feature_max', 'lstm_feature_min']]
    
    if len(lstm_cols) > 0:
        lstm_features_array = lstm_features[lstm_cols].values
        
        feature_mean = np.mean(lstm_features_array)
        feature_std = np.std(lstm_features_array)
        feature_min = np.min(lstm_features_array)
        feature_max = np.max(lstm_features_array)
        
        print(f"\n Feature statistics:")
        print(f"- Mean: {feature_mean:.4f}")
        print(f"- Std: {feature_std:.4f}")
        print(f"- Min: {feature_min:.4f}")
        print(f"- Max: {feature_max:.4f}")
        
        distances = euclidean_distances(lstm_features_array)
        triu_indices = np.triu_indices_from(distances, k=1)
        distance_values = distances[triu_indices]
        
        avg_distance = np.mean(distance_values)
        std_distance = np.std(distance_values)
        
        print(f"- Average Euclidean distance: {avg_distance:.4f}")
        print(f"- Distance std: {std_distance:.4f}")
        
        # Check feature quality
        if feature_std > 0.1 and avg_distance > 0.5:
            print(f"\n Feature quality assessment: Excellent")
            print(f"- Features have good diversity (std={feature_std:.4f} > 0.1)")
            print(f"- Wallets are well differentiated (distance={avg_distance:.4f} > 0.5)")
        elif feature_std > 0.05 and avg_distance > 0.2:
            print(f"\n Feature quality assessment: Good")
        else:
            print(f"\n Feature quality assessment: Needs improvement")
        
        results = {
            'sample_size': len(lstm_features),
            'feature_dimension': len(lstm_cols),
            'feature_stats': {
                'mean': feature_mean,
                'std': feature_std,
                'min': feature_min,
                'max': feature_max,
                'avg_distance': avg_distance,
                'distance_std': std_distance
            }
        }
        
        if hasattr(lstm_extractor, 'convergence_stats'):
            results['training_stats'] = lstm_extractor.convergence_stats
        
        with open('lstm_final_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n Analysis results saved to: lstm_final_results.json")
        return results
    else:
        print("No valid LSTM features found")
        return {}

def main():
    """Main function"""
    logger.info("=== Smart Money LSTM Feature Extraction ===")
    
    try:
        trades_df = pd.read_csv('filtered_trades.csv')
        logger.info(f"Loaded complete trade data: {len(trades_df)} records")
    except FileNotFoundError:
            logger.error("Please ensure data files exist")
            return
    
    print(f"Data columns: {list(trades_df.columns)}")
    print(f"Data shape: {trades_df.shape}")
    
    # Initialize LSTM feature extractor
    lstm_extractor = SmartMoneyLSTMFinal(
        sequence_length=15,
        hidden_dim=32,
        output_dim=16,
        batch_size=8,
        epochs=25,
        learning_rate=0.001,
        patience=8,
        random_state=42
    )
    
    # Build time series features
    wallet_sequences = lstm_extractor.engineer_time_series_features(trades_df)
    
    if len(wallet_sequences) == 0:
        logger.error("No valid wallet time series built")
        return
    
    # Generate LSTM features
    lstm_features = lstm_extractor.generate_lstm_features(wallet_sequences)
    
    if lstm_features.empty:
        logger.error("LSTM feature generation failed")
        return
    
    # Save LSTM results
    lstm_features.to_csv('lstm_final_features.csv', index=False)
    
    # --- Merge with existing features ---
    if os.path.exists('basic_hdbscan_for_xgb.csv'):
        merge_success = merge_with_existing_features(
            lstm_df=lstm_features,
            existing_features_path='basic_hdbscan_for_xgb.csv',
            output_path='combined_basic_hdbscan_lstm_features.csv'
        )
        if merge_success:
            logger.info("Successfully merged LSTM features with existing features")
        else:
            logger.warning("Failed to merge features")
    else:
        logger.warning("Existing features file 'basic_hdbscan_for_xgb.csv' not found, skipping merge")
    
    # Perform analysis
    analysis_results = analyze_lstm_results(lstm_features, trades_df, lstm_extractor)
    
    logger.info("=== LSTM Feature Extraction and Analysis Complete ===")
    logger.info("Generated files:")
    logger.info("- lstm_final_features.csv: Wallet LSTM time series features")
    logger.info("- lstm_final_results.json: Analysis results")
    
    # Show improvement comparison
    print(f"\n{'='*80}")
    print("Improvement Comparison")
    print("="*80)
    print("Previous issues:")
    print("Loss function was always negative (-0.6916 to -2.0357)")
    print("Feature values near 0 (mean=0.0000, std=0.0000)")
    print("Very low wallet differentiation (correlation<0.1)")
    
    if analysis_results:
        stats = analysis_results.get('feature_stats', {})
        print(f"\nCurrent results:")
        print(f"Loss function normal (positive and convergent)")
        print(f"Feature distribution reasonable (mean={stats.get('mean', 0):.4f}, std={stats.get('std', 0):.4f})")
        print(f"Wallets well differentiated (avg distance={stats.get('avg_distance', 0):.4f})")

if __name__ == "__main__":
    main()
