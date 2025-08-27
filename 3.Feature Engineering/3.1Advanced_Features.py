import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. BEHAVIORAL FEATURES CALCULATOR
# ==============================================================================

class BehavioralFeaturesCalculator:
    """
    Calculates 5 core behavioral bias features based on academic formulas.
    This class is designed to be managed by the main AdvancedFeatureEngineer.
    """

    def __init__(self, trades_df, prices_df, returns_df, network_df, activity_df):
        print("Initializing BehavioralFeaturesCalculator...")
        self.trades_df = trades_df
        self.prices_df = prices_df
        self.returns_df = returns_df
        self.network_df = network_df
        self.activity_df = activity_df

    def calculate_contrarian_score(self, wallet_address: str) -> float:
        """
        Calculates the contrarian investment score.
        Formula: Contrarian_Score_i = (1/(n-1)) * Σ(-1) * ρ(R_t-1, A_t)
        Theory: Contrarian investors buy after price drops and sell after price rises.
        """
        wallet_trades = self.trades_df[self.trades_df['wallet_address'] == wallet_address].copy()
        if len(wallet_trades) < 2:
            return 0.0

        contrarian_scores = []
        for token in wallet_trades['token_bought_address'].dropna().unique():
            token_trades = wallet_trades[
                (wallet_trades['token_bought_address'] == token) |
                (wallet_trades['token_sold_address'] == token)
            ].copy().sort_values('block_time')
            if len(token_trades) < 2:
                continue

            token_prices = self.prices_df[self.prices_df['contract_address'] == token].copy().sort_values('minute')
            if len(token_prices) < 2:
                continue

            for i in range(1, len(token_trades)):
                current_trade = token_trades.iloc[i]
                current_time = current_trade['block_time']
                current_action = current_trade['trade_action']
                if pd.isna(current_action):
                    continue

                recent_prices = token_prices[token_prices['minute'] <= current_time].tail(10)
                if len(recent_prices) < 2:
                    continue

                price_returns = recent_prices['price_return'].dropna()
                if not price_returns.empty:
                    r_t_minus_1 = price_returns.mean()
                    a_t = current_action
                    if abs(r_t_minus_1) > 1e-8 and abs(a_t) > 0:
                        rho = (r_t_minus_1 * a_t) / (abs(r_t_minus_1) * abs(a_t))
                        contrarian_single_score = -1 * rho
                        contrarian_scores.append(contrarian_single_score)
        
        if not contrarian_scores:
            return 0.0

        n = len(contrarian_scores) + 1
        if n <= 1:
            return 0.0

        final_score = (1 / (n - 1)) * sum(contrarian_scores)
        normalized_score = (final_score + 1) / 2  # Map from [-1, 1] to [0, 1]
        return max(0.0, min(1.0, normalized_score))

    def calculate_fomo_resistance(self, wallet_address: str) -> float:
        """
        Calculates FOMO (Fear Of Missing Out) resistance score.
        Formula: FOMO_Resistance = 0.25*MR + 0.20*TimD + 0.25*FC + 0.15*SR + 0.15*CS
        """
        wallet_trades = self.trades_df[self.trades_df['wallet_address'] == wallet_address].copy()
        if wallet_trades.empty:
            return 0.0

        momentum_resistance = self._calculate_momentum_resistance(wallet_trades)
        timing_discipline = self._calculate_timing_discipline(wallet_trades)
        frequency_control = self._calculate_frequency_control(wallet_trades)
        size_rationality = self._calculate_size_rationality(wallet_trades)
        cost_sensitivity = self._calculate_cost_sensitivity(wallet_trades)

        fomo_resistance = (
            0.25 * momentum_resistance +
            0.20 * timing_discipline +
            0.25 * frequency_control +
            0.15 * size_rationality +
            0.15 * cost_sensitivity
        )
        return max(0.0, min(1.0, fomo_resistance))

    def _calculate_momentum_resistance(self, wallet_trades: pd.DataFrame) -> float:
        """MR = 1 - |Correlation(trade_direction, price_momentum)|"""
        if len(wallet_trades) < 3:
            return 0.0
        momentum_scores, actions = [], []
        for _, trade in wallet_trades.iterrows():
            token = trade.get('token_bought_address') or trade.get('token_sold_address')
            if pd.isna(token) or pd.isna(trade['trade_action']):
                continue
            
            price_window_end = trade['block_time'] - pd.Timedelta(minutes=1)
            price_window_start = trade['block_time'] - pd.Timedelta(hours=1)
            token_prices = self.prices_df[
                (self.prices_df['contract_address'] == token) &
                (self.prices_df['minute'] >= price_window_start) &
                (self.prices_df['minute'] <= price_window_end)
            ]
            if len(token_prices) >= 10:
                prices = token_prices['price_usd'].values
                short_ma = np.mean(prices[-3:])
                long_ma = np.mean(prices[-10:])
                if long_ma > 0:
                    momentum = (short_ma - long_ma) / long_ma
                    momentum_scores.append(momentum)
                    actions.append(trade['trade_action'])
        
        if len(momentum_scores) < 3:
            return 0.0
        
        # Remove outliers using IQR method
        momentum_scores = np.array(momentum_scores)
        q1, q3 = np.percentile(momentum_scores, [25, 75])
        iqr = q3 - q1
        mask = (momentum_scores >= q1 - 1.5 * iqr) & (momentum_scores <= q3 + 1.5 * iqr)
        
        if np.sum(mask) < 2:
            return 0.0

        correlation = np.corrcoef(np.array(actions)[mask], momentum_scores[mask])[0, 1]
        return 1 - abs(correlation) if not np.isnan(correlation) else 0.0

    def _calculate_timing_discipline(self, wallet_trades: pd.DataFrame) -> float:
        """Timing Discipline = 1 - HHI(trading_hour_concentration)"""
        if 'hour_of_day' not in wallet_trades.columns:
            wallet_trades['hour_of_day'] = wallet_trades['block_time'].dt.hour
        hour_counts = wallet_trades['hour_of_day'].value_counts()
        total_trades = len(wallet_trades)
        if total_trades <= 1:
            return 1.0
        hhi = sum((count / total_trades) ** 2 for count in hour_counts)
        return 1 - hhi

    def _calculate_frequency_control(self, wallet_trades: pd.DataFrame) -> float:
        """Calculates control over trading frequency."""
        time_span = wallet_trades['block_time'].max() - wallet_trades['block_time'].min()
        total_days = time_span.total_seconds() / (24 * 3600) if len(wallet_trades) >= 2 else 1
        if total_days <= 0: return 0.0
        freq = len(wallet_trades) / total_days
        optimal_freq, max_freq = 2, 50 # Optimal trades/day, max trades/day
        return 1.0 if freq <= optimal_freq else max(0.0, 1 - (freq - optimal_freq) / max_freq)

    def _calculate_size_rationality(self, wallet_trades: pd.DataFrame) -> float:
        """Size Rationality = 1 / (1 + CV(trade_size_usd))"""
        trade_sizes = wallet_trades['amount_usd'].dropna()
        if len(trade_sizes) < 2:
            return 0.0
        mean_size = trade_sizes.mean()
        if mean_size == 0: return 0.0
        cv = trade_sizes.std() / mean_size
        return 1 / (1 + cv)

    def _calculate_cost_sensitivity(self, wallet_trades: pd.DataFrame) -> float:
        """Cost Sensitivity = 1 / (1 + avg_cost_ratio * 100)"""
        if 'tx_cost_eth' not in wallet_trades.columns or 'amount_usd' not in wallet_trades.columns:
            return 0.0
        cost_ratios = (wallet_trades['tx_cost_eth'] / wallet_trades['amount_usd']).dropna()
        if cost_ratios.empty:
            return 0.0
        return 1 / (1 + cost_ratios.mean() * 100)

    def calculate_rationality_confidence(self, wallet_address: str) -> float:
        """
        Rationality Confidence = 1 - Overconfidence Score.
        Overconfidence = Rank(Frequency) * (1 - Rank(Returns)).
        """
        wallet_activity = self.activity_df[self.activity_df['wallet_address'] == wallet_address]
        wallet_returns = self.returns_df[self.returns_df['wallet_address'] == wallet_address]
        if wallet_activity.empty or wallet_returns.empty:
            return 0.5

        f_i = wallet_activity['avg_trades_per_day'].iloc[0]
        r_i = wallet_returns['avg_return'].mean()
        if pd.isna(f_i) or pd.isna(r_i):
            return 0.5
        
        all_frequencies = self.activity_df['avg_trades_per_day'].dropna()
        all_returns = self.returns_df['avg_return'].dropna()

        rank_f_i = stats.percentileofscore(all_frequencies, f_i) / 100.0
        rank_r_i = stats.percentileofscore(all_returns, r_i) / 100.0
        
        overconfidence_score = rank_f_i * (1 - rank_r_i)
        rationality_confidence = 1 - overconfidence_score
        return max(0.0, min(1.0, rationality_confidence))

    def calculate_anchoring_resistance(self, wallet_address: str) -> float:
        """
        Anchoring Resistance measures the tendency to hold losing assets for too long.
        Resistance = 1 - Normalized_Anchoring_Bias.
        """
        wallet_trades = self.trades_df[self.trades_df['wallet_address'] == wallet_address].copy()
        if wallet_trades.empty:
            return 0.5

        bias_scores = []
        for token in wallet_trades['token_bought_address'].dropna().unique():
            token_trades = wallet_trades[
                (wallet_trades['token_bought_address'] == token) |
                (wallet_trades['token_sold_address'] == token)
            ].sort_values('block_time')
            if len(token_trades) < 2: continue

            buys = token_trades[token_trades['token_bought_address'] == token]
            sells = token_trades[token_trades['token_sold_address'] == token]

            # Simple FIFO matching
            for _, buy in buys.iterrows():
                matched_sell = sells[sells['block_time'] > buy['block_time']].iloc[0:1]
                if not matched_sell.empty:
                    sell = matched_sell.iloc[0]
                    p_buy, p_sell = buy['price_bought'], sell['price_sold']
                    if pd.isna(p_buy) or pd.isna(p_sell) or p_buy <= 0: continue
                    
                    holding_hours = (sell['block_time'] - buy['block_time']).total_seconds() / 3600
                    time_threshold_hours = 24

                    is_loss = 1 if p_sell < p_buy else 0
                    long_holding = 1 if holding_hours > time_threshold_hours else 0
                    loss_ratio = (p_buy - p_sell) / p_buy

                    bias_score = loss_ratio * is_loss * long_holding
                    if bias_score > 0:
                        bias_scores.append(bias_score)

        if not bias_scores:
            return 1.0  # High resistance if no anchoring bias is detected
        
        avg_bias = np.mean(bias_scores)
        normalized_bias = 1 / (1 + np.exp(-avg_bias * 5))
        return 1 - normalized_bias

    def calculate_herding_resistance(self, wallet_address: str) -> float:
        """
        Herding Resistance = 1 - Herding Tendency.
        Herding is measured by the similarity of a user's actions to the market consensus.
        """
        wallet_trades = self.trades_df[self.trades_df['wallet_address'] == wallet_address].copy()
        if wallet_trades.empty:
            return 0.0

        herding_scores = []
        for _, trade in wallet_trades.iterrows():
            trade_time = trade['block_time']
            trade_action = trade['trade_action']
            if pd.isna(trade_action): continue

            window_start = trade_time - pd.Timedelta(minutes=30)
            window_end = trade_time + pd.Timedelta(minutes=30)
            
            concurrent_trades = self.trades_df[
                (self.trades_df['block_time'] >= window_start) &
                (self.trades_df['block_time'] <= window_end) &
                (self.trades_df['trade_action'].notna())
            ]
            if len(concurrent_trades) < 5: continue

            mean_action = concurrent_trades['trade_action'].mean()
            # Herding tendency: how close is the user to the mean action?
            # 1 is perfect alignment, 0 is perfect opposition.
            herding_tendency = 1 - abs(trade_action - mean_action) / 2
            herding_scores.append(herding_tendency)

        if not herding_scores:
             # Fallback to network data if available
            wallet_network = self.network_df[self.network_df['wallet_address'] == wallet_address]
            if not wallet_network.empty and 'avg_synchronicity_score' in wallet_network.columns:
                sync_score = wallet_network['avg_synchronicity_score'].iloc[0]
                return 1 - sync_score if pd.notna(sync_score) else 0.5
            return 0.5 # Default value

        avg_herding_tendency = np.mean(herding_scores)
        herding_resistance = 1 - avg_herding_tendency
        return max(0.0, min(1.0, herding_resistance))

    def calculate_all_features(self, wallet_address: str) -> Dict[str, float]:
        """Calculates all 5 behavioral features for a single wallet."""
        return {
            'contrarian_score': self.calculate_contrarian_score(wallet_address),
            'fomo_resistance': self.calculate_fomo_resistance(wallet_address),
            'rationality_confidence': self.calculate_rationality_confidence(wallet_address),
            'anchoring_resistance': self.calculate_anchoring_resistance(wallet_address),
            'herding_resistance': self.calculate_herding_resistance(wallet_address)
        }

# ==============================================================================
# 2. TIMING INTELLIGENCE CALCULATOR
# ==============================================================================

class TimingIntelligenceCalculator:
    """
    Calculates 3 core timing intelligence features.
    """

    def __init__(self, trades_df, prices_df):
        print("Initializing TimingIntelligenceCalculator...")
        self.trades_df = trades_df
        self.prices_df = prices_df

    def calculate_entry_timing_score(self, wallet_address: str) -> float:
        """
        Measures the quality of entry timing based on short-term post-buy performance.
        """
        buy_trades = self.trades_df[
            (self.trades_df['wallet_address'] == wallet_address) &
            (self.trades_df['trade_action'] == 1)
        ].copy()
        if buy_trades.empty:
            return 0.0

        scores = []
        for _, trade in buy_trades.iterrows():
            token = trade['token_bought_address']
            entry_time, entry_price = trade['block_time'], trade['price_bought']
            if pd.isna(entry_price) or entry_price <= 0: continue
            
            token_prices = self.prices_df[self.prices_df['contract_address'] == token]
            if len(token_prices) < 10: continue

            score = self._calculate_single_entry_timing(token_prices, entry_time, entry_price)
            if score is not None:
                scores.append(score)

        if not scores: return 0.0
        # Map score from roughly [-1, 1] to [0, 1]
        return max(0.0, min(1.0, (np.mean(scores) + 1) / 2))

    def _calculate_single_entry_timing(self, token_prices: pd.DataFrame, entry_time, entry_price) -> Optional[float]:
        """Calculates timing score for a single entry."""
        window_scores = []
        for minutes in [15, 60, 240]: # 15m, 1h, 4h windows
            future_prices = token_prices[
                (token_prices['minute'] > entry_time) &
                (token_prices['minute'] <= entry_time + pd.Timedelta(minutes=minutes))
            ]
            if len(future_prices) < 3: continue

            actual_return = (future_prices['price_usd'].mean() - entry_price) / entry_price
            
            historical_data = token_prices[
                (token_prices['minute'] <= entry_time) &
                (token_prices['minute'] >= entry_time - pd.Timedelta(hours=24))
            ]
            if len(historical_data) > 10:
                historical_returns = historical_data['price_return'].dropna()
                if len(historical_returns) > 5:
                    expected_return = historical_returns.mean()
                    volatility = historical_returns.std()
                    if volatility > 0:
                        # Sharpe-like ratio for timing advantage
                        timing_advantage = (actual_return - expected_return) / volatility
                        window_scores.append(timing_advantage)
        
        if not window_scores: return None
        weights = [0.5, 0.3, 0.2][:len(window_scores)] # Short-term focus
        return np.average(window_scores, weights=weights)

    def calculate_exit_timing_score(self, wallet_address: str) -> float:
        """
        Measures the quality of exit timing, i.e., selling near local price peaks.
        """
        sell_trades = self.trades_df[
            (self.trades_df['wallet_address'] == wallet_address) &
            (self.trades_df['trade_action'] == -1)
        ].copy()
        if sell_trades.empty:
            return 0.0
            
        scores = []
        for _, trade in sell_trades.iterrows():
            token = trade['token_sold_address']
            exit_time, exit_price = trade['block_time'], trade['price_sold']
            if pd.isna(exit_price) or exit_price <= 0: continue
            
            token_prices = self.prices_df[self.prices_df['contract_address'] == token]
            if len(token_prices) < 10: continue

            score = self._calculate_single_exit_timing(token_prices, exit_time, exit_price)
            if score is not None:
                scores.append(score)

        if not scores: return 0.0
        return max(0.0, min(1.0, np.mean(scores)))

    def _calculate_single_exit_timing(self, token_prices: pd.DataFrame, exit_time, exit_price) -> Optional[float]:
        """Calculates timing score for a single exit."""
        past_prices = token_prices[
            (token_prices['minute'] >= exit_time - pd.Timedelta(hours=4)) &
            (token_prices['minute'] <= exit_time)
        ]
        future_prices = token_prices[
            (token_prices['minute'] > exit_time) &
            (token_prices['minute'] <= exit_time + pd.Timedelta(hours=4))
        ]
        if len(past_prices) < 10 or len(future_prices) < 10:
            return None

        # Score 1: Position within the recent price range (0-1)
        past_max, past_min = past_prices['price_usd'].max(), past_prices['price_usd'].min()
        score1 = (exit_price - past_min) / (past_max - past_min) if (past_max - past_min) > 0 else 0.5
        
        # Score 2: Avoided loss (price decline after exit)
        price_decline = (exit_price - future_prices['price_usd'].mean()) / exit_price
        score2 = max(0, min(1, price_decline * 5)) # Scaled score

        return (score1 + score2) / 2

    def calculate_volatility_timing(self, wallet_address: str) -> float:
        """
        Measures the ability to buy in low volatility and sell in high volatility periods.
        """
        wallet_trades = self.trades_df[self.trades_df['wallet_address'] == wallet_address].copy()
        if wallet_trades.empty:
            return 0.0
        
        scores = []
        for _, trade in wallet_trades.iterrows():
            token = trade.get('token_bought_address') or trade.get('token_sold_address')
            trade_action, trade_time = trade['trade_action'], trade['block_time']
            if pd.isna(token): continue
            
            token_prices = self.prices_df[self.prices_df['contract_address'] == token]
            if len(token_prices) < 100: continue

            score = self._calculate_volatility_timing_score(token_prices, trade_time, trade_action)
            if score is not None:
                scores.append(score)

        if not scores: return 0.0
        return max(0.0, min(1.0, np.mean(scores)))

    def _calculate_volatility_timing_score(self, token_prices, trade_time, trade_action) -> Optional[float]:
        """Calculates volatility timing score for a single trade."""
        trade_data = token_prices[token_prices['minute'] <= trade_time]
        if len(trade_data) < 50: return None

        current_vol = trade_data['volatility_1h'].iloc[-1]
        if pd.isna(current_vol): return None
        
        historical_vol = trade_data['volatility_1h'].dropna()
        if len(historical_vol) < 20: return None
            
        vol_percentile = stats.percentileofscore(historical_vol, current_vol) / 100.0
        
        if trade_action == 1: # Buy
            return 1 - vol_percentile # Good to buy at low volatility
        else: # Sell
            return vol_percentile # Good to sell at high volatility

    def calculate_all_features(self, wallet_address: str) -> Dict[str, float]:
        """Calculates all 3 timing features for a single wallet."""
        return {
            'entry_timing_score': self.calculate_entry_timing_score(wallet_address),
            'exit_timing_score': self.calculate_exit_timing_score(wallet_address),
            'volatility_timing': self.calculate_volatility_timing(wallet_address)
        }

# ==============================================================================
# 3. SOCIAL NETWORK EXTRACTOR
# ==============================================================================

class SocialNetworkExtractor:
    """
    Extracts and calculates social intelligence features from pre-computed network analysis data.
    """

    def __init__(self, network_df, trades_df):
        print("Initializing SocialNetworkExtractor...")
        self.network_df = network_df
        self.trades_df = trades_df
        self._validate_data()
        
    def _validate_data(self):
        """Validates that required columns exist in the network data."""
        required = ['wallet_address', 'network_influence_score', 'independent_decision_ratio']
        if not all(col in self.network_df.columns for col in required):
            print(f"Warning: Network data is missing one of {required}. Some features may be 0.")

    def calculate_network_influence_score_weighted(self, wallet_address: str) -> float:
        """
        Calculates a volume-weighted network influence score.
        Score = base_influence * log(1 + volume) / log(1 + max_volume)
        """
        wallet_data = self.network_df[self.network_df['wallet_address'] == wallet_address]
        if wallet_data.empty: return 0.0
        
        base_influence = wallet_data['network_influence_score'].iloc[0]
        if pd.isna(base_influence): return 0.0
            
        # Calculate volume weight
        wallet_trades = self.trades_df[self.trades_df['wallet_address'] == wallet_address]
        if wallet_trades.empty: return base_influence # Return base score if no trades
            
        total_volume_usd = wallet_trades['amount_usd'].sum()
        max_volume_usd = self.trades_df.groupby('wallet_address')['amount_usd'].sum().max()
        
        if max_volume_usd <= 0: return base_influence
            
        volume_weight = np.log(1 + total_volume_usd) / np.log(1 + max_volume_usd)
        return base_influence * volume_weight

    def get_independent_decision_ratio(self, wallet_address: str) -> float:
        """Extracts the independent decision ratio."""
        wallet_data = self.network_df[self.network_df['wallet_address'] == wallet_address]
        if wallet_data.empty: return 0.0
        ratio = wallet_data['independent_decision_ratio'].iloc[0]
        return ratio if pd.notna(ratio) else 0.0

    def calculate_all_features(self, wallet_address: str) -> Dict[str, float]:
        """Extracts all social network features for a single wallet."""
        return {
            'network_influence_score_weighted': self.calculate_network_influence_score_weighted(wallet_address),
            'independent_decision_ratio': self.get_independent_decision_ratio(wallet_address)
        }

# ==============================================================================
# 4. MASTER FEATURE ENGINEER
# ==============================================================================

class AdvancedFeatureEngineer:
    """
    Master class to orchestrate the calculation of behavioral, timing,
    and social network features.
    """
    def __init__(self, data_files: Dict[str, str]):
        self.files = data_files
        self._load_and_preprocess_data()
        
        # Initialize individual feature calculators with preprocessed data
        self.behavioral_calc = BehavioralFeaturesCalculator(
            self.trades_df, self.prices_df, self.returns_df, self.network_df, self.activity_df
        )
        self.timing_calc = TimingIntelligenceCalculator(self.trades_df, self.prices_df)
        self.social_calc = SocialNetworkExtractor(self.network_df, self.trades_df)

    def _load_and_preprocess_data(self):
        """Loads and preprocesses all necessary data files once."""
        print("\n--- Starting Data Loading and Preprocessing ---")
        
        # Load data
        self.trades_df = pd.read_csv(self.files['trades'])
        self.prices_df = pd.read_csv(self.files['prices'])
        self.returns_df = pd.read_csv(self.files['returns'])
        self.network_df = pd.read_csv(self.files['network'])
        self.activity_df = pd.read_csv(self.files['activity'])
        print("All data files loaded successfully.")

        # Preprocess Trades Data
        self.trades_df['block_time'] = pd.to_datetime(self.trades_df['block_time'])
        self.trades_df.drop_duplicates(subset=['tx_hash', 'wallet_address'], inplace=True)
        self.trades_df['trade_action'] = self.trades_df['trade_direction'].map(
            {'buy_stable': 1, 'sell_stable': -1, 'buy_eth': 1, 'sell_eth': -1}
        )
        for col in ['amount_usd', 'price_bought', 'price_sold', 'tx_cost_eth']:
             if col in self.trades_df.columns:
                self.trades_df[col] = pd.to_numeric(self.trades_df[col], errors='coerce')
        self.trades_df.sort_values(['wallet_address', 'block_time'], inplace=True)
        
        # =================== CORRECTED SECTION START ===================
        # Preprocess Prices Data
        
        # I have set this to 'block_time' based on your confirmation.
        actual_timestamp_column = 'block_time' 

        if actual_timestamp_column not in self.prices_df.columns:
            print(f"Error: The specified timestamp column '{actual_timestamp_column}' was not found in prices_df.")
            print(f"Available columns are: {self.prices_df.columns.tolist()}")
            raise KeyError(f"Column '{actual_timestamp_column}' not found.")

        # Rename the column to 'minute' so the rest of the script works without further changes.
        self.prices_df.rename(columns={actual_timestamp_column: 'minute'}, inplace=True)
        
        self.prices_df['minute'] = pd.to_datetime(self.prices_df['minute'])
        self.prices_df.sort_values(['contract_address', 'minute'], inplace=True)
        self.prices_df['price_return'] = self.prices_df.groupby('contract_address')['price_usd'].pct_change()
        self.prices_df['volatility_1h'] = self.prices_df.groupby('contract_address')['price_return'].rolling(window=60, min_periods=5).std().reset_index(0, drop=True)
        # =================== CORRECTED SECTION END =====================
        
        print("--- Data Preprocessing Complete ---\n")

    def calculate_all_features_for_wallet(self, wallet_address: str) -> Dict:
        """
        Calculates all advanced features for a single wallet and merges them.
        """
        # This print statement can be noisy for batch processing, so we comment it out.
        # print(f"\nCalculating all features for wallet: {wallet_address[:15]}...")
        
        behavioral_features = self.behavioral_calc.calculate_all_features(wallet_address)
        timing_features = self.timing_calc.calculate_all_features(wallet_address)
        social_features = self.social_calc.calculate_all_features(wallet_address)
        
        # Merge all features
        all_features = {'wallet_address': wallet_address}
        all_features.update(behavioral_features)
        all_features.update(timing_features)
        all_features.update(social_features)
        
        return all_features

    def batch_calculate_all_features(self, wallet_addresses: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Runs batch calculation for all feature sets and merges the results.
        """
        if wallet_addresses is None:
            # Use intersection of wallets present in all key data sources
            trade_wallets = set(self.trades_df['wallet_address'].unique())
            return_wallets = set(self.returns_df['wallet_address'].unique())
            network_wallets = set(self.network_df['wallet_address'].unique())
            wallet_addresses = list(trade_wallets.intersection(return_wallets).intersection(network_wallets))
        
        print(f"\n--- Starting batch feature calculation for {len(wallet_addresses)} wallets ---")
        
        all_results = []
        total = len(wallet_addresses)
        for i, wallet in enumerate(wallet_addresses):
            if (i + 1) % 50 == 0:
                print(f"Processing progress: {i+1}/{total} wallets...")
            try:
                features = self.calculate_all_features_for_wallet(wallet)
                all_results.append(features)
            except Exception as e:
                print(f"Error calculating features for {wallet}: {e}")

        final_df = pd.DataFrame(all_results)
        print("\n--- Batch feature calculation complete! ---")
        return final_df


# ==============================================================================
# 5. MAIN EXECUTION BLOCK
# ==============================================================================

def main():
    """
    Main function to demonstrate the use of the AdvancedFeatureEngineer.
    """
    print("======================================================")
    print(" Advanced Feature Engineering Pipeline for Smart Money")
    print("======================================================")

    # Define the paths to your data files
    data_files = {
        'trades': 'filtered_trades.csv',
        'prices': 'price_usd.csv',
        'returns': 'returnrate.csv',
        'network': 'networkanalysis.csv',
        'activity': 'walletactivity.csv'
    }

    # Initialize the master feature engineer
    try:
        engineer = AdvancedFeatureEngineer(data_files)
    except FileNotFoundError as e:
        print(f"\nError: Data file not found. Please ensure all required CSVs are in the same directory.")
        print(f"Missing file: {e.filename}")
        return

    # --- Batch Calculation for All Wallets ---
    # The script will automatically find all common wallets across the datasets.
    all_features_df = engineer.batch_calculate_all_features()

    if not all_features_df.empty:
        # --- Save and Analyze Results ---
        
        # Define the list of columns you want to keep
        # Based on your image, but using the actual column names from the script.
        final_columns_to_keep = [
            'wallet_address',
            'unique_tokens',
            'avg_trade_size',
            'contrarian_score',
            'fomo_resistance',
            'rationality_confidence',
            'anchoring_resistance',
            'herding_resistance',
            'network_influence_score_weighted',
            'independent_decision_ratio',
            'entry_timing_score',
            'exit_timing_score',
            'volatility_timing'
        ]

        # Check for missing columns before filtering
        existing_columns = [col for col in final_columns_to_keep if col in all_features_df.columns]
        if len(existing_columns) != len(final_columns_to_keep):
            missing = set(final_columns_to_keep) - set(existing_columns)
            print(f"\nWarning: The following requested columns were not found and will be skipped: {missing}")

        # Filter the DataFrame to keep only the specified columns
        filtered_df = all_features_df[existing_columns]

        # Change the output filename
        output_filename = 'advanced_features.csv'
        filtered_df.to_csv(output_filename, index=False)
        print(f"\n✅ Filtered features saved to '{output_filename}'")
        
        print("\n--- Final Feature DataFrame Statistics (Filtered) ---")
        pd.set_option('display.width', 100)
        print(filtered_df.describe().round(3))
    else:
        print("\nNo features were calculated. Please check the input data and logs.")


if __name__ == "__main__":
    main()
