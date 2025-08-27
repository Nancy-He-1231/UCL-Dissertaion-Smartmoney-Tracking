import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from statsmodels.stats.contingency_tables import mcnemar
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
    Final Classifier for Smart Money Identification (V3 - English Version)

    Core Design Principles:
    1. Entity-level Isolation: Uses GroupKFold to prevent data leakage.
    2. Multi-Feature Fusion: Integrates advanced, GAT, LSTM, HDBSCAN, and basic trading features.
    3. Interpretability: Provides feature importance analysis.
    4. Comprehensive Evaluation: Outputs a confusion matrix and McNemar's test for statistical significance.
    """

    def __init__(self,
                 xgb_params: Dict = None,
                 cv_folds: int = 5,
                 random_state: int = 42):

        self.cv_folds = cv_folds
        self.random_state = random_state

        if xgb_params is None:
            self.xgb_params = {
                'objective': 'binary:logistic', 'eval_metric': 'auc', 'max_depth': 6,
                'learning_rate': 0.1, 'n_estimators': 500, 'subsample': 0.8,
                'colsample_bytree': 0.8, 'early_stopping_rounds': 50,
                'random_state': random_state, 'n_jobs': -1
            }
        else:
            self.xgb_params = xgb_params

        self.xgb_model = None
        self.feature_scaler = None
        self.feature_importance = None
        self.feature_groups = {}

    def create_entity_groups(self, features_df: pd.DataFrame) -> np.ndarray:
        """Creates stable, entity-level groups based on a hash of the wallet address."""
        def wallet_to_group(wallet_addr: str) -> int:
            hash_val = int(hashlib.md5(wallet_addr.encode()).hexdigest()[:4], 16)
            return hash_val % 20
        entity_groups = features_df['wallet_address'].apply(wallet_to_group).values
        logger.info(f"Created {len(np.unique(entity_groups))} entity groups.")
        return entity_groups

    def load_and_prepare_features(self,
                                  advanced_features_path: str = 'advanced_features.csv',
                                  base_features_path: str = 'final_all_features.csv') -> Optional[pd.DataFrame]:
        """Loads and merges features from multiple source files."""
        logger.info("Starting to load and merge feature data...")
        df_advanced, df_base = None, None
        try:
            df_advanced = pd.read_csv(advanced_features_path)
            logger.info(f"✓ Successfully loaded advanced features: {advanced_features_path} - Shape: {df_advanced.shape}")
        except FileNotFoundError:
            logger.warning(f"⚠ Advanced feature file not found: {advanced_features_path}")
        try:
            df_base = pd.read_csv(base_features_path)
            logger.info(f"✓ Successfully loaded base & comprehensive features: {base_features_path} - Shape: {df_base.shape}")
        except FileNotFoundError:
            logger.warning(f"⚠ Base & comprehensive feature file not found: {base_features_path}")

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
        self._display_final_features(features_df)
        logger.info("Feature preparation complete.")
        return features_df

    def _categorize_features(self, features_df: pd.DataFrame) -> Dict[str, List[str]]:
        """Categorizes features into logical groups."""
        feature_groups = {
            'basic_trading': [], 'hdbscan': [], 'lstm': [], 'gat': [], 'behavioral_bias': [],
            'timing_intelligence': [], 'social_network': [], 'other_advanced': []
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
        """Cleans and preprocesses the feature DataFrame."""
        logger.info("Cleaning feature data...")
        feature_cols = [col for col in features_df.columns if col != 'wallet_address']
        features_df[feature_cols] = features_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        numeric_features = features_df[feature_cols].select_dtypes(include=np.number).columns.tolist()
        zero_var_features = [col for col in numeric_features if features_df[col].var() == 0]
        if zero_var_features:
            features_df = features_df.drop(columns=zero_var_features)
            logger.info(f"✓ Removed {len(zero_var_features)} zero-variance features.")
            numeric_features = [f for f in numeric_features if f not in zero_var_features]
        if len(numeric_features) > 1:
            corr_matrix = features_df[numeric_features].corr().abs()
            upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            high_corr_features = [col for col in upper_triangle.columns if any(upper_triangle[col] > 0.95)]
            if high_corr_features:
                features_df = features_df.drop(columns=high_corr_features)
                logger.info(f"✓ Removed {len(high_corr_features)} highly correlated features.")
        logger.info(f"✓ Feature cleaning complete. Final feature count: {len(features_df.columns) - 1}")
        return features_df

    def _display_final_features(self, features_df: pd.DataFrame):
        """Displays a summary of the final feature groups."""
        logger.info("\n=== Final Feature Groups Retained ===")
        category_map = {
            'basic_trading': 'Basic Trading Features', 'hdbscan': 'HDBSCAN Clustering Features', 'lstm': 'LSTM Time Series Features',
            'gat': 'GAT Graph Network Features', 'behavioral_bias': 'Behavioral Bias Control Features', 'timing_intelligence': 'Timing Intelligence Features',
            'social_network': 'Social Network Intelligence Features', 'other_advanced': 'Other Advanced/Derived Features'
        }
        total_features = 0
        for key, name in category_map.items():
            count = len([f for f in self.feature_groups.get(key, []) if f in features_df.columns])
            if count > 0:
                logger.info(f"  {name}: {count}")
                total_features += count
        logger.info(f"\nTotal: {total_features} features will be used for Smart Money identification.")

    def load_smart_money_labels(self, features_df: pd.DataFrame) -> np.ndarray:
        """Loads ground truth Smart Money labels."""
        logger.info("Loading ground truth Smart Money labels...")
        label_sources = ['smart_money_labels.csv', 'wallet_labels.csv', 'ground_truth_labels.csv']
        for label_file in label_sources:
            try:
                label_df = pd.read_csv(label_file)
                logger.info(f"✓ Found label file: {label_file}")
                if 'wallet_address' in label_df.columns:
                    merged = features_df.merge(label_df, on='wallet_address', how='left')
                    for col in ['is_smart_money', 'smart_money', 'label', 'target']:
                        if col in merged.columns:
                            labels = merged[col].fillna(0).astype(int).values
                            logger.info(f"✓ Using label column: '{col}'")
                            n_smart = labels.sum()
                            logger.info(f"Label statistics: {n_smart}/{len(features_df)} ({n_smart/len(features_df)*100:.1f}%) are Smart Money.")
                            return labels
            except FileNotFoundError: continue
        logger.warning("⚠ Ground truth label file not found. Using random demo labels for execution!")
        np.random.seed(self.random_state)
        return np.random.choice([0, 1], size=len(features_df), p=[0.95, 0.05])

    def train_xgboost_model_with_proper_isolation(self,
                                                features_df: pd.DataFrame,
                                                labels: np.ndarray) -> Optional[Dict]:
        """Trains the XGBoost model with strict entity-level isolation and returns a comprehensive evaluation."""
        logger.info("Starting entity-level isolated XGBoost training...")
        
        numeric_cols = features_df.select_dtypes(include=np.number).columns.tolist()
        if not numeric_cols:
            logger.error("No numerical features available for training."); return None
        
        X, y, entity_groups = features_df[numeric_cols].values, labels, self.create_entity_groups(features_df)
        unique_labels, counts = np.unique(y, return_counts=True)
        if len(unique_labels) < 2:
            logger.error("Label data must contain both positive and negative classes."); return None

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
        
        logger.info(f"Data split: Train/Validation set size: {len(X_train_val)} | Test set size: {len(X_test)}")

        pos_weight = counts[0] / counts[1] if counts[1] > 0 else 1
        cv_xgb_params = self.xgb_params.copy()
        cv_xgb_params['scale_pos_weight'] = pos_weight

        gkf = GroupKFold(n_splits=min(self.cv_folds, len(np.unique(groups_train_val))))
        cv_scores = []
        for _, (train_idx, val_idx) in enumerate(gkf.split(X_train_val, y_train_val, groups_train_val), 1):
            X_fold_train, X_fold_val = X_train_val[train_idx], X_train_val[val_idx]
            y_fold_train, y_fold_val = y_train_val[train_idx], y_train_val[val_idx]
            model = xgb.XGBClassifier(**cv_xgb_params)
            model.fit(X_fold_train, y_fold_train, eval_set=[(X_fold_val, y_fold_val)], verbose=False)
            cv_scores.append(roc_auc_score(y_fold_val, model.predict_proba(X_fold_val)[:, 1]))
        
        final_xgb_params = cv_xgb_params.copy()
        final_xgb_params.pop('early_stopping_rounds', None)
        self.xgb_model = xgb.XGBClassifier(**final_xgb_params)
        self.xgb_model.fit(X_train_val, y_train_val)
        
        self.feature_importance = dict(zip(numeric_cols, self.xgb_model.feature_importances_))
        
        test_pred_proba = self.xgb_model.predict_proba(X_test)[:, 1]
        test_pred_labels = self.xgb_model.predict(X_test)
        
        cm = confusion_matrix(y_test, test_pred_labels)
        mcnemar_result = mcnemar(cm, exact=False)
        
        results = {
            'mean_auc': np.mean(cv_scores), 'std_auc': np.std(cv_scores),
            'test_auc': roc_auc_score(y_test, test_pred_proba),
            'precision': precision_score(y_test, test_pred_labels),
            'recall': recall_score(y_test, test_pred_labels),
            'f1_score': f1_score(y_test, test_pred_labels),
            'classification_report': classification_report(y_test, test_pred_labels, target_names=['Normal Wallet', 'Smart Money']),
            'confusion_matrix': cm,
            'mcnemar_statistic': mcnemar_result.statistic,
            'mcnemar_pvalue': mcnemar_result.pvalue,
            'test_indices': test_mask
        }
        
        logger.info(f"\n=== Cross-validation results: AUC = {results['mean_auc']:.4f} ± {results['std_auc']:.4f}")
        logger.info(f"=== Independent test set results: AUC = {results['test_auc']:.4f}")
        return results

    def analyze_feature_importance(self, top_n: int = 20) -> Optional[pd.DataFrame]:
        """Analyzes and returns the top N most important features."""
        if self.feature_importance is None:
            logger.warning("Feature importances not calculated. Please train the model first."); return None
        importance_df = pd.DataFrame(self.feature_importance.items(), columns=['Feature', 'Importance'])
        return importance_df.sort_values('Importance', ascending=False).head(top_n)

    def generate_report(self, features_df: pd.DataFrame, labels: np.ndarray, training_results: Dict) -> pd.DataFrame:
        """Generates a final report with predictions and probabilities."""
        logger.info("Generating Smart Money identification report...")
        numeric_cols = features_df.select_dtypes(include=np.number).columns
        X_scaled = self.feature_scaler.transform(features_df[numeric_cols].values)
        test_mask = training_results['test_indices']
        proba = np.full(len(features_df), np.nan)
        pred = np.full(len(features_df), -1, dtype=int)
        proba[~test_mask] = self.xgb_model.predict_proba(X_scaled[~test_mask])[:, 1]
        pred[~test_mask] = self.xgb_model.predict(X_scaled[~test_mask])
        report_df = pd.DataFrame({
            'wallet_address': features_df['wallet_address'], 'ground_truth_label': labels,
            'smart_money_probability': proba, 'smart_money_prediction': pred,
            'data_split': np.where(test_mask, 'test', 'train_val')
        })
        return report_df.sort_values('smart_money_probability', ascending=False, na_position='last').reset_index(drop=True)

    def visualize_results(self, importance_df: pd.DataFrame, cm: np.ndarray):
        """Visualizes the analysis results, including feature importance and confusion matrix."""
        logger.info("Generating result visualizations...")
        
        plt.figure(figsize=(12, 8))
        if importance_df is not None:
            sns.barplot(x='Importance', y='Feature', data=importance_df.head(20), palette='viridis')
            plt.title('Top 20 Feature Importances', fontsize=16)
            plt.xlabel('Importance Score', fontsize=12)
            plt.ylabel('Feature', fontsize=12)
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300)
        plt.show()

        self.visualize_confusion_matrix(cm, class_names=['Normal Wallet', 'Smart Money'])

    def visualize_confusion_matrix(self, cm: np.ndarray, class_names: List[str]):
        """Creates and saves a publication-quality confusion matrix plot."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names,
                    annot_kws={"size": 14})
        plt.title('Confusion Matrix (on Independent Test Set)', fontsize=16)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        logger.info("✓ Confusion matrix plot saved to confusion_matrix.png")
        plt.show()

    def save_model(self, filepath: str):
        """Saves the complete classifier (model, scaler, etc.) to a file."""
        model_data = {
            'xgb_model': self.xgb_model, 'feature_scaler': self.feature_scaler,
            'feature_importance': self.feature_importance, 'feature_groups': self.feature_groups
        }
        joblib.dump(model_data, filepath)
        logger.info(f"Classifier saved to {filepath}")

def main():
    """Main execution function."""
    logger.info("=== Smart Money XGBoost Classifier (V3 - Comprehensive Evaluation) ===")
    
    classifier = SmartMoneyClassifier(cv_folds=5, random_state=42)
    
    features_df = classifier.load_and_prepare_features()
    if features_df is None: return

    smart_money_labels = classifier.load_smart_money_labels(features_df)
    
    training_results = classifier.train_xgboost_model_with_proper_isolation(features_df, smart_money_labels)
    if training_results is None:
        logger.error("Model training failed."); return

    importance_df = classifier.analyze_feature_importance(top_n=20)
    report_df = classifier.generate_report(features_df, smart_money_labels, training_results)
    
    classifier.visualize_results(importance_df, training_results['confusion_matrix'])
    
    report_df.to_csv('smart_money_identification_report.csv', index=False)
    if importance_df is not None:
        importance_df.to_csv('feature_importance_analysis.csv', index=False)
    classifier.save_model('smart_money_classifier.pkl')
    
    logger.info("=== Smart Money Identification Pipeline Finished ===")
    
    print("\n" + "="*60)
    print(" " * 15 + "Final Model Performance (on Independent Test Set)")
    print("="*60)
    print(f"  Total Wallets Analyzed: {len(features_df)}")
    print(f"  Cross-Validation AUC: {training_results['mean_auc']:.4f} ± {training_results['std_auc']:.4f}")
    
    print("\n--- Classification Performance Metrics ---")
    metrics_df = pd.DataFrame({
        'Metric': ['AUC', 'Precision', 'Recall', 'F1-Score'],
        'Score': [f"{v:.4f}" for v in [training_results['test_auc'], training_results['precision'], training_results['recall'], training_results['f1_score']]]
    })
    print(metrics_df.to_string(index=False))
    
    print("\n--- Detailed Classification Report ---")
    print(training_results['classification_report'])
    
    print("\n--- Statistical Significance (McNemar's Test) ---")
    print(f"  Chi-squared statistic: {training_results['mcnemar_statistic']:.4f}")
    print(f"  p-value: {training_results['mcnemar_pvalue']:.4f}")
    if training_results['mcnemar_pvalue'] < 0.05:
        print("  Conclusion: p < 0.05, the model has a statistically significant difference between\n              the number of false positive and false negative errors.")
    else:
        print("  Conclusion: p >= 0.05, there is no statistically significant difference between\n              the number of false positive and false negative errors.")
    
    print("="*60)
    
    print("\nTop 5 Predicted Smart Money Wallets (from Train/Validation Set):")
    top_5 = report_df[report_df['data_split'] == 'train_val'].head(5)
    for _, row in top_5.iterrows():
        print(f"  Address: {row['wallet_address'][:15]}... | Probability: {row['smart_money_probability']:.4f}")

if __name__ == "__main__":
    main()
