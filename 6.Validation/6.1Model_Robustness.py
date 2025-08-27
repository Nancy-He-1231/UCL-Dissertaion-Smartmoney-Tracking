import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import logging
import joblib
from typing import Dict, List, Optional
import hashlib

# --- Setup ---
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SmartMoneyClassifier:
    """
    Smart Money Final Identification Classifier 

    Core Design Principles:
    1. Entity-level Isolation - Uses GroupKFold to prevent data leakage
    2. Multi-Feature Fusion - Integrates advanced, GAT, LSTM, HDBSCAN and basic trading features
    3. Interpretability - Provides feature importance analysis
    """

    def __init__(self,
                 xgb_params: Dict = None,
                 cv_folds: int = 5,
                 random_state: int = 42):

        self.cv_folds = cv_folds
        self.random_state = random_state

        # XGBoost default parameters
        if xgb_params is None:
            self.xgb_params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 500,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'early_stopping_rounds': 50,
                'random_state': random_state,
                'n_jobs': -1
            }
        else:
            self.xgb_params = xgb_params

        # Store models and preprocessors
        self.xgb_model = None
        self.feature_scaler = None
        self.feature_importance = None
        self.feature_groups = {}

    def create_entity_groups(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Creates stable entity-level groups based on hash of wallet address
        """
        def wallet_to_group(wallet_addr: str) -> int:
            hash_val = int(hashlib.md5(wallet_addr.encode()).hexdigest()[:4], 16)
            return hash_val % 20

        entity_groups = features_df['wallet_address'].apply(wallet_to_group).values
        return entity_groups

    def load_and_prepare_features(self,
                                  advanced_features_path: str = 'advanced_features.csv',
                                  base_features_path: str = 'final_all_features.csv') -> Optional[pd.DataFrame]:
        """
        Loads and merges features from multiple files
        """
        df_advanced = None
        df_base = None

        try:
            df_advanced = pd.read_csv(advanced_features_path)
        except FileNotFoundError:
            pass  # Silent handling

        try:
            df_base = pd.read_csv(base_features_path)
        except FileNotFoundError:
            pass  # Silent handling

        if df_advanced is not None and df_base is not None:
            features_df = pd.merge(df_advanced, df_base, on='wallet_address', how='outer')
        elif df_advanced is not None:
            features_df = df_advanced
        elif df_base is not None:
            features_df = df_base
        else:
            logger.error("Error: All specified feature files were not found!")
            return None

        features_df = self._clean_features(features_df)
        self.feature_groups = self._categorize_features(features_df)
        
        return features_df

    def _categorize_features(self, features_df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Groups features - includes basic trading features
        """
        feature_groups = {
            'basic_trading': [], 'hdbscan': [], 'lstm': [], 'gat': [],
            'behavioral_bias': [], 'timing_intelligence': [], 'social_network': [], 'other_advanced': []
        }
        
        behavioral_bias_features = ['contrarian_score', 'fomo_resistance', 'rationality_confidence', 'anchoring_resistance', 'herding_resistance']
        timing_intelligence_features = ['entry_timing_score', 'exit_timing_score', 'volatility_timing']
        social_network_features = ['network_influence_score_weighted', 'independent_decision_ratio']
        
        basic_trading_features = [
            'amount_usd_sum', 'amount_usd_mean', 'amount_usd_max', 'amount_usd_std',
            'token_bought_symbol_nunique', 'token_sold_symbol_nunique', 'dex_name_nunique',
            'active_days', 'avg_trades_per_day', 'trade_size_cv'
        ]

        for col in features_df.columns:
            if col == 'wallet_address': continue
            elif col in basic_trading_features: feature_groups['basic_trading'].append(col)
            elif 'hdbscan' in col or 'cluster' in col: feature_groups['hdbscan'].append(col)
            elif 'lstm' in col: feature_groups['lstm'].append(col)
            elif 'gat' in col: feature_groups['gat'].append(col)
            elif col in behavioral_bias_features: feature_groups['behavioral_bias'].append(col)
            elif col in timing_intelligence_features: feature_groups['timing_intelligence'].append(col)
            elif col in social_network_features: feature_groups['social_network'].append(col)
            else: feature_groups['other_advanced'].append(col)

        return feature_groups

    def _clean_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Feature cleaning and preprocessing - retains all types of features
        """
        feature_cols = [col for col in features_df.columns if col != 'wallet_address']
        
        features_df[feature_cols] = features_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)

        numeric_features = features_df[feature_cols].select_dtypes(include=np.number).columns.tolist()
        zero_var_features = [col for col in numeric_features if features_df[col].var() == 0]
        if zero_var_features:
            features_df = features_df.drop(columns=zero_var_features)
            numeric_features = [f for f in numeric_features if f not in zero_var_features]

        if len(numeric_features) > 1:
            corr_matrix = features_df[numeric_features].corr().abs()
            upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            high_corr_features = [col for col in upper_triangle.columns if any(upper_triangle[col] > 0.95)]
            if high_corr_features:
                features_df = features_df.drop(columns=high_corr_features)

        return features_df

    def load_smart_money_labels(self, features_df: pd.DataFrame) -> np.ndarray:
        """Loads true Smart Money labels"""
        label_sources = ['smart_money_labels.csv', 'wallet_labels.csv', 'ground_truth_labels.csv']
        
        for label_file in label_sources:
            try:
                label_df = pd.read_csv(label_file)
                if 'wallet_address' in label_df.columns:
                    merged = features_df.merge(label_df, on='wallet_address', how='left')
                    for col in ['is_smart_money', 'smart_money', 'label', 'target']:
                        if col in merged.columns:
                            labels = merged[col].fillna(0).astype(int).values
                            return labels
            except FileNotFoundError:
                continue
        
        logger.warning("Ground truth label file not found, using random demo labels!")
        np.random.seed(self.random_state)
        return np.random.choice([0, 1], size=len(features_df), p=[0.95, 0.05])

    def train_xgboost_model_with_proper_isolation(self,
                                                features_df: pd.DataFrame,
                                                labels: np.ndarray) -> Optional[Dict]:
        """
        XGBoost training with strict entity-level isolation
        """
        numeric_cols = features_df.select_dtypes(include=np.number).columns.tolist()
        if not numeric_cols:
            logger.error("No numerical features available for training.")
            return None
        
        X, y, entity_groups = features_df[numeric_cols].values, labels, self.create_entity_groups(features_df)

        unique_labels, counts = np.unique(y, return_counts=True)
        if len(unique_labels) < 2:
            logger.error("Insufficient label data, need both positive and negative samples.")
            return None

        unique_groups = np.unique(entity_groups)
        n_test_groups = max(1, int(len(unique_groups) * 0.2))
        np.random.seed(self.random_state)
        test_groups = np.random.choice(unique_groups, size=n_test_groups, replace=False)
        
        test_mask = np.isin(entity_groups, test_groups)
        train_val_mask = ~test_mask

        X_train_val_raw, X_test_raw = X[train_val_mask], X[test_mask]
        y_train_val, y_test = y[train_val_mask], y[test_mask]
        groups_train_val = entity_groups[train_val_mask]
        
        scaler = StandardScaler()
        X_train_val = scaler.fit_transform(X_train_val_raw)
        X_test = scaler.transform(X_test_raw)
        self.feature_scaler = scaler
        
        pos_weight = counts[0] / counts[1]
        cv_xgb_params = self.xgb_params.copy()
        cv_xgb_params['scale_pos_weight'] = pos_weight

        gkf = GroupKFold(n_splits=min(self.cv_folds, len(np.unique(groups_train_val))))
        
        cv_scores = []
        for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_train_val, y_train_val, groups_train_val), 1):
            X_fold_train, X_fold_val = X_train_val[train_idx], X_train_val[val_idx]
            y_fold_train, y_fold_val = y_train_val[train_idx], y_train_val[val_idx]
            model = xgb.XGBClassifier(**cv_xgb_params)
            model.fit(X_fold_train, y_fold_train, eval_set=[(X_fold_val, y_fold_val)], verbose=False)
            y_pred_proba = model.predict_proba(X_fold_val)[:, 1]
            auc_score = roc_auc_score(y_fold_val, y_pred_proba)
            cv_scores.append(auc_score)
        
        final_xgb_params = cv_xgb_params.copy()
        final_xgb_params.pop('early_stopping_rounds', None)
        self.xgb_model = xgb.XGBClassifier(**final_xgb_params)
        self.xgb_model.fit(X_train_val, y_train_val)
        
        test_pred_proba = self.xgb_model.predict_proba(X_test)[:, 1]
        test_pred_labels = self.xgb_model.predict(X_test)
        
        return {
            'test_auc': roc_auc_score(y_test, test_pred_proba),
            'precision': precision_score(y_test, test_pred_labels, zero_division=0),
            'recall': recall_score(y_test, test_pred_labels, zero_division=0),
            'f1_score': f1_score(y_test, test_pred_labels, zero_division=0)
        }


def main():
    """
    [Modified] Main function - adds robustness testing
    """
    logger.info("=== Smart Money XGBoost Identification Model (Robustness Test Version) ===")
    
    # --- [Important Modification] ---
    # Define multiple random seeds for repeated experiments
    seeds = [42, 123, 2024, 888, 101]
    all_results = []

    # 1. Load data once
    # Assuming the classifier has no randomness in loading data, load once for efficiency
    # If the loading process also has randomness, it needs to be moved into the loop
    temp_classifier = SmartMoneyClassifier(random_state=seeds[0])
    features_df = temp_classifier.load_and_prepare_features()
    if features_df is None:
        logger.error("Data loading failed, terminating program.")
        return
    
    logger.info(f"Data loading completed, total wallets: {len(features_df)}")
    logger.info(f"Will perform {len(seeds)} independent experiments using the following random seeds: {seeds}")

    # 2. Run experiments in a loop
    for seed in seeds:
        print(f"\n--- Running experiment (Random Seed: {seed}) ---")
        
        # Create a new classifier instance for each experiment
        classifier = SmartMoneyClassifier(cv_folds=5, random_state=seed)
        
        # Load labels (if labels are dynamically generated, need to reload each time to ensure randomness is controlled)
        smart_money_labels = classifier.load_smart_money_labels(features_df)
        
        # Train and evaluate model
        training_results = classifier.train_xgboost_model_with_proper_isolation(features_df, smart_money_labels)
        
        if training_results:
            all_results.append(training_results)
            logger.info(f"Experiment completed (Seed: {seed}) - Test AUC: {training_results['test_auc']:.4f}")
        else:
            logger.error(f"Experiment failed (Seed: {seed})")

    # 3. Summarize and report final results
    if not all_results:
        logger.error("All experiments failed, cannot generate final report.")
        return
        
    # Convert results list to DataFrame for easier calculation
    results_df = pd.DataFrame(all_results)
    
    # Calculate mean and standard deviation
    mean_scores = results_df.mean()
    std_scores = results_df.std()

    print("\n" + "="*60)
    print(" " * 20 + "Robustness Test Final Results")
    print("="*60)
    print(f"  Total independent experiments conducted: {len(all_results)}")
    print(f"  Random seeds used: {seeds}")
    
    print("\n--- Average Performance Metrics (Mean ± Std Dev) ---")
    
    # Create a nice report table
    report_data = {
        "Metric": ["Test AUC", "Precision", "Recall", "F1-Score"],
        "Score": [
            f"{mean_scores['test_auc']:.4f} ± {std_scores['test_auc']:.4f}",
            f"{mean_scores['precision']:.4f} ± {std_scores['precision']:.4f}",
            f"{mean_scores['recall']:.4f} ± {std_scores['recall']:.4f}",
            f"{mean_scores['f1_score']:.4f} ± {std_scores['f1_score']:.4f}"
        ]
    }
    report_df = pd.DataFrame(report_data)
    print(report_df.to_string(index=False))

    print("\n--- Conclusion ---")
    if std_scores['test_auc'] < 0.05:
        print("Model performance is stable: Small standard deviation across multiple experiments indicates")
        print("  the model's performance is not coincidental, results are not affected by random data splits,")
        print("  and the model has strong robustness.")
    else:
        print("Model performance fluctuates significantly: Large standard deviation across experiments")
        print("  suggests checking data splitting strategy or model stability, results may be sensitive")
        print("  to specific data splits.")
    print("="*60)


if __name__ == "__main__":
    main()
