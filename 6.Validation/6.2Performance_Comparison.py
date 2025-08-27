import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import logging
import hashlib
from typing import Dict, List, Optional, Tuple

# Settings
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeatureComparisonExperiment:
    """
    Feature Comparison Validation Experiment
    For validating the effectiveness of different feature combinations in Smart Money identification
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.results = []
        
        # Base XGBoost parameters
        self.base_xgb_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 300,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': random_state,
            'n_jobs': -1
        }
    
    def create_entity_groups(self, features_df: pd.DataFrame) -> np.ndarray:
        """Create entity-level groups for proper data isolation"""
        def wallet_to_group(wallet_addr: str) -> int:
            hash_val = int(hashlib.md5(wallet_addr.encode()).hexdigest()[:4], 16)
            return hash_val % 15
        
        return features_df['wallet_address'].apply(wallet_to_group).values
    
    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """Load and prepare feature data"""
        logger.info("Loading feature data...")
        
        df_advanced = None
        df_base = None
        
        # Try to load advanced features
        try:
            df_advanced = pd.read_csv('advanced_features.csv')
            logger.info(f"Advanced features loaded: {df_advanced.shape}")
        except FileNotFoundError:
            logger.warning("merged_ethereum_features.csv not found")
        
        # Try to load base/comprehensive features
        try:
            df_base = pd.read_csv('final_all_features.csv')
            logger.info(f"Base/comprehensive features loaded: {df_base.shape}")
        except FileNotFoundError:
            logger.warning("final_all_features.csv not found")
        
        # Merge features
        if df_advanced is not None and df_base is not None:
            features_df = pd.merge(df_advanced, df_base, on='wallet_address', how='outer')
            logger.info(f"Features merged: {features_df.shape}")
        elif df_advanced is not None:
            features_df = df_advanced
        elif df_base is not None:
            features_df = df_base
        else:
            raise FileNotFoundError("No feature files found!")
        
        # Clean data
        features_df = self._clean_features(features_df)
        
        # Load labels
        labels = self._load_labels(features_df)
        
        return features_df, labels
    
    def _clean_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess features"""
        feature_cols = [col for col in features_df.columns if col != 'wallet_address']
        features_df[feature_cols] = features_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        
        # Remove zero-variance features
        numeric_features = features_df[feature_cols].select_dtypes(include=np.number).columns
        zero_var_features = [col for col in numeric_features if features_df[col].var() == 0]
        if zero_var_features:
            features_df = features_df.drop(columns=zero_var_features)
            logger.info(f"Removed {len(zero_var_features)} zero-variance features")
        
        return features_df
    
    def _load_labels(self, features_df: pd.DataFrame) -> np.ndarray:
        """Load Smart Money labels"""
        label_sources = ['smart_money_labels.csv', 'wallet_labels.csv', 'ground_truth_labels.csv']
        
        for label_file in label_sources:
            try:
                label_df = pd.read_csv(label_file)
                merged = features_df.merge(label_df, on='wallet_address', how='left')
                for col in ['is_smart_money', 'smart_money', 'label', 'target']:
                    if col in merged.columns:
                        labels = merged[col].fillna(0).astype(int).values
                        logger.info(f"Using label column: {col}")
                        return labels
            except FileNotFoundError:
                continue
        
        logger.warning("No ground truth labels found, using random labels for demo")
        np.random.seed(self.random_state)
        return np.random.choice([0, 1], size=len(features_df), p=[0.95, 0.05])
    
    def define_feature_groups(self, features_df: pd.DataFrame) -> Dict[str, List[str]]:
        """Define different feature groups for comparison"""
        all_features = [col for col in features_df.columns if col != 'wallet_address']
        
        # Basic trading features
        basic_trading = [
            'amount_usd_sum', 'amount_usd_mean', 'amount_usd_max', 'amount_usd_std',
            'token_bought_symbol_nunique', 'token_sold_symbol_nunique', 'dex_name_nunique',
            'active_days', 'avg_trades_per_day', 'trade_size_cv', 'total_trades'
        ]
        basic_trading = [f for f in basic_trading if f in all_features]
        
        # HDBSCAN clustering features
        hdbscan_features = [f for f in all_features if 'hdbscan' in f.lower() or 'cluster' in f.lower()]
        
        # LSTM temporal features
        lstm_features = [f for f in all_features if 'lstm' in f.lower()]
        
        # GAT graph network features
        gat_features = [f for f in all_features if 'gat' in f.lower()]
        
        # Smart Money advanced features
        advanced_features = [
            'contrarian_score', 'fomo_resistance', 'rationality_confidence',
            'anchoring_resistance', 'herding_resistance', 'entry_timing_score',
            'exit_timing_score', 'volatility_timing', 'network_influence_score_weighted',
            'independent_decision_ratio'
        ]
        advanced_features = [f for f in advanced_features if f in all_features]
        
        # Other features
        accounted_features = basic_trading + hdbscan_features + lstm_features + gat_features + advanced_features
        other_features = [f for f in all_features if f not in accounted_features]
        
        feature_groups = {
            'basic_trading': basic_trading,
            'hdbscan': hdbscan_features,
            'lstm': lstm_features,
            'gat': gat_features,
            'advanced': advanced_features,
            'other': other_features,
            'all': all_features
        }
        
        # Log feature group information
        logger.info("\n=== Feature Group Definition ===")
        for group_name, features in feature_groups.items():
            if features:
                logger.info(f"{group_name}: {len(features)} features")
        
        return feature_groups
    
    def train_and_evaluate(self, 
                          X: np.ndarray, 
                          y: np.ndarray, 
                          entity_groups: np.ndarray,
                          experiment_name: str) -> Optional[Dict]:
        """Train and evaluate model with entity-level isolation"""
        logger.info(f"Running experiment: {experiment_name}")
        
        # Data splitting with entity-level isolation
        unique_groups = np.unique(entity_groups)
        n_test_groups = max(1, int(len(unique_groups) * 0.2))
        np.random.seed(self.random_state)
        test_groups = np.random.choice(unique_groups, size=n_test_groups, replace=False)
        
        test_mask = np.isin(entity_groups, test_groups)
        train_val_mask = ~test_mask
        
        X_train_val, X_test = X[train_val_mask], X[test_mask]
        y_train_val, y_test = y[train_val_mask], y[test_mask]
        groups_train_val = entity_groups[train_val_mask]
        
        # Standardization
        scaler = StandardScaler()
        X_train_val = scaler.fit_transform(X_train_val)
        X_test = scaler.transform(X_test)
        
        # Handle class imbalance
        unique_labels, counts = np.unique(y_train_val, return_counts=True)
        if len(unique_labels) < 2:
            logger.error(f"{experiment_name}: Insufficient label diversity")
            return None
        
        pos_weight = counts[0] / counts[1] if len(counts) > 1 else 1.0
        
        # Cross-validation with GroupKFold
        gkf = GroupKFold(n_splits=min(3, len(np.unique(groups_train_val))))
        cv_scores = []
        
        xgb_params = self.base_xgb_params.copy()
        xgb_params['scale_pos_weight'] = pos_weight
        
        for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_train_val, y_train_val, groups_train_val)):
            X_fold_train, X_fold_val = X_train_val[train_idx], X_train_val[val_idx]
            y_fold_train, y_fold_val = y_train_val[train_idx], y_train_val[val_idx]
            
            model = xgb.XGBClassifier(**xgb_params)
            model.fit(X_fold_train, y_fold_train, eval_set=[(X_fold_val, y_fold_val)], verbose=False)
            
            y_pred_proba = model.predict_proba(X_fold_val)[:, 1]
            y_pred = model.predict(X_fold_val)
            
            auc = roc_auc_score(y_fold_val, y_pred_proba)
            precision = precision_score(y_fold_val, y_pred, zero_division=0)
            recall = recall_score(y_fold_val, y_pred, zero_division=0)
            f1 = f1_score(y_fold_val, y_pred, zero_division=0)
            
            cv_scores.append({'auc': auc, 'precision': precision, 'recall': recall, 'f1': f1})
        
        # Calculate average cross-validation performance
        avg_scores = {
            'auc': np.mean([s['auc'] for s in cv_scores]),
            'precision': np.mean([s['precision'] for s in cv_scores]),
            'recall': np.mean([s['recall'] for s in cv_scores]),
            'f1': np.mean([s['f1'] for s in cv_scores])
        }
        
        # Final model evaluation on independent test set
        final_params = xgb_params.copy()
        final_params.pop('early_stopping_rounds', None)
        final_model = xgb.XGBClassifier(**final_params)
        final_model.fit(X_train_val, y_train_val)
        
        test_pred_proba = final_model.predict_proba(X_test)[:, 1]
        test_pred = final_model.predict(X_test)
        
        test_scores = {
            'auc': roc_auc_score(y_test, test_pred_proba),
            'precision': precision_score(y_test, test_pred, zero_division=0),
            'recall': recall_score(y_test, test_pred, zero_division=0),
            'f1': f1_score(y_test, test_pred, zero_division=0)
        }
        
        logger.info(f"{experiment_name} - CV AUC: {avg_scores['auc']:.4f}, Test AUC: {test_scores['auc']:.4f}")
        
        return {
            'experiment_name': experiment_name,
            'cv_scores': avg_scores,
            'test_scores': test_scores,
            'n_features': X.shape[1],
            'n_train': len(X_train_val),
            'n_test': len(X_test)
        }
    
    def run_comparison_experiments(self):
        """Run all comparison experiments"""
        logger.info("=== Starting Feature Comparison Experiments ===")
        
        # Load data
        features_df, labels = self.load_and_prepare_data()
        entity_groups = self.create_entity_groups(features_df)
        
        # Define feature groups
        feature_groups = self.define_feature_groups(features_df)
        
        # Experiment configuration
        experiments = [
            ('Basic Trading Features Only', ['basic_trading']),
            ('HDBSCAN Clustering Features', ['hdbscan']),
            ('LSTM Temporal Features', ['lstm']),
            ('GAT Graph Features', ['gat']),
            ('Smart Money Advanced Features', ['advanced']),
            ('Basic + HDBSCAN', ['basic_trading', 'hdbscan']),
            ('Basic + LSTM', ['basic_trading', 'lstm']),
            ('Basic + GAT', ['basic_trading', 'gat']),
            ('HDBSCAN + LSTM + GAT', ['hdbscan', 'lstm', 'gat']),
            ('Our Fusion Method (All Features)', ['all'])
        ]
        
        all_results = []
        
        for exp_name, group_keys in experiments:
            # Merge specified feature groups
            selected_features = []
            for key in group_keys:
                if key == 'all':
                    selected_features = feature_groups['all']
                    break
                else:
                    selected_features.extend(feature_groups.get(key, []))
            
            if not selected_features:
                logger.warning(f"Skipping {exp_name}: No available features")
                continue
            
            # Ensure features exist in data
            available_features = [f for f in selected_features if f in features_df.columns]
            if not available_features:
                logger.warning(f"Skipping {exp_name}: Features not found in data")
                continue
            
            logger.info(f"\n{exp_name}: Using {len(available_features)} features")
            
            # Prepare feature matrix
            X = features_df[available_features].values
            
            # Train and evaluate
            result = self.train_and_evaluate(X, labels, entity_groups, exp_name)
            if result:
                all_results.append(result)
        
        # Save results
        self.results = all_results
        return all_results
    
    def create_comparison_table(self) -> pd.DataFrame:
        """Create comparison results table"""
        if not self.results:
            logger.error("No experiment results found. Please run run_comparison_experiments() first")
            return None
        
        table_data = []
        for result in self.results:
            table_data.append({
                'Method': result['experiment_name'],
                'Feature_Type': self._get_feature_type(result['experiment_name']),
                'N_Features': result['n_features'],
                'CV_AUC': f"{result['cv_scores']['auc']:.4f}",
                'CV_Precision': f"{result['cv_scores']['precision']:.4f}",
                'CV_Recall': f"{result['cv_scores']['recall']:.4f}",
                'Test_AUC': f"{result['test_scores']['auc']:.4f}",
                'Test_Precision': f"{result['test_scores']['precision']:.4f}",
                'Test_Recall': f"{result['test_scores']['recall']:.4f}"
            })
        
        comparison_df = pd.DataFrame(table_data)
        return comparison_df
    
    def _get_feature_type(self, method_name: str) -> str:
        """Get feature type based on method name"""
        type_mapping = {
            'Basic Trading Features Only': 'Basic',
            'HDBSCAN Clustering Features': 'Clustering',
            'LSTM Temporal Features': 'Temporal',
            'GAT Graph Features': 'Graph Network',
            'Smart Money Advanced Features': 'Advanced',
            'Our Fusion Method (All Features)': 'Multimodal'
        }
        
        for key, value in type_mapping.items():
            if key in method_name:
                return value
        return 'Combined'
    
    def visualize_results(self):
        """Visualize comparison results"""
        if not self.results:
            logger.error("No experiment results found")
            return
        
        # Prepare data
        methods = [r['experiment_name'] for r in self.results]
        short_methods = [m.replace(' Features', '').replace('Our Fusion Method (All Features)', 'Our Method') 
                        for m in methods]
        cv_aucs = [r['cv_scores']['auc'] for r in self.results]
        test_aucs = [r['test_scores']['auc'] for r in self.results]
        precisions = [r['test_scores']['precision'] for r in self.results]
        recalls = [r['test_scores']['recall'] for r in self.results]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # AUC comparison
        x_pos = np.arange(len(short_methods))
        axes[0, 0].bar(x_pos - 0.2, cv_aucs, 0.4, label='CV AUC', alpha=0.8, color='skyblue')
        axes[0, 0].bar(x_pos + 0.2, test_aucs, 0.4, label='Test AUC', alpha=0.8, color='orange')
        axes[0, 0].set_xlabel('Method')
        axes[0, 0].set_ylabel('AUC Score')
        axes[0, 0].set_title('AUC Performance Comparison')
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels(short_methods, rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Precision vs Recall
        scatter = axes[0, 1].scatter(recalls, precisions, s=100, alpha=0.7, c=test_aucs, cmap='viridis')
        for i, method in enumerate(short_methods):
            axes[0, 1].annotate(method, (recalls[i], precisions[i]), 
                              xytext=(5, 5), textcoords='offset points', fontsize=8)
        axes[0, 1].set_xlabel('Recall')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].set_title('Precision vs Recall')
        axes[0, 1].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[0, 1], label='Test AUC')
        
        # Feature count comparison
        feature_counts = [r['n_features'] for r in self.results]
        bars = axes[1, 0].bar(short_methods, feature_counts, alpha=0.7, color='lightgreen')
        axes[1, 0].set_xlabel('Method')
        axes[1, 0].set_ylabel('Number of Features')
        axes[1, 0].set_title('Feature Count by Method')
        axes[1, 0].tick_params(axis='x', rotation=45)
        for bar, count in zip(bars, feature_counts):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                           f'{count}', ha='center', va='bottom')
        
        # Test AUC ranking
        sorted_indices = np.argsort(test_aucs)[::-1]
        sorted_methods = [short_methods[i] for i in sorted_indices]
        sorted_aucs = [test_aucs[i] for i in sorted_indices]
        
        colors = ['gold' if 'Our Method' in method else 'lightcoral' for method in sorted_methods]
        bars = axes[1, 1].bar(sorted_methods, sorted_aucs, alpha=0.8, color=colors)
        axes[1, 1].set_xlabel('Method')
        axes[1, 1].set_ylabel('Test AUC')
        axes[1, 1].set_title('Test AUC Ranking')
        axes[1, 1].tick_params(axis='x', rotation=45)
        for bar, auc in zip(bars, sorted_aucs):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                           f'{auc:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('feature_comparison_results.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Main function"""
    logger.info("=== Smart Money Feature Comparison Validation Experiment ===")
    
    # Create experiment object
    experiment = FeatureComparisonExperiment(random_state=42)
    
    # Run comparison experiments
    results = experiment.run_comparison_experiments()
    
    if not results:
        logger.error("Experiments failed")
        return
    
    # Create comparison table
    comparison_table = experiment.create_comparison_table()
    print("\n=== Feature Comparison Experiment Results ===")
    print(comparison_table.to_string(index=False))
    
    # Save results
    comparison_table.to_csv('feature_comparison_results.csv', index=False)
    
    # Visualize results
    experiment.visualize_results()
    
    # Analyze best results
    best_result = max(results, key=lambda x: x['test_scores']['auc'])
    logger.info(f"\n Best performing method: {best_result['experiment_name']}")
    logger.info(f"   Test AUC: {best_result['test_scores']['auc']:.4f}")
    logger.info(f"   Features used: {best_result['n_features']}")
    
    # Performance improvement analysis
    baseline_auc = min(results, key=lambda x: x['test_scores']['auc'])['test_scores']['auc']
    improvement = ((best_result['test_scores']['auc'] - baseline_auc) / baseline_auc) * 100
    logger.info(f"   Improvement over baseline: {improvement:.1f}%")
    
    print(f"\n=== Experiment Completed ===")
    print(f"Results saved to:")
    print(f"- feature_comparison_results.csv")
    print(f"- feature_comparison_results.png")
    
    # Academic paper summary
    print(f"\n=== Academic Paper Summary ===")
    print("Performance comparison of different feature combinations:")
    for result in sorted(results, key=lambda x: x['test_scores']['auc'], reverse=True):
        feature_type = experiment._get_feature_type(result['experiment_name'])
        print(f"- {feature_type}: AUC = {result['test_scores']['auc']:.4f}, "
              f"Precision = {result['test_scores']['precision']:.4f}, "
              f"Recall = {result['test_scores']['recall']:.4f}")

if __name__ == "__main__":
    main()
