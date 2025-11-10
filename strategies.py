import pandas as pd
import ta
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class MACDStrategy:
    def __init__(self, fast=12, slow=26, signal=9):
        self.fast = fast
        self.slow = slow
        self.signal = signal

    def generate_signals(self, data):
        macd = ta.trend.MACD(data['close'],
                            window_fast=self.fast,
                            window_slow=self.slow,
                            window_sign=self.signal)
        data['macd'] = macd.macd()
        data['signal'] = macd.macd_signal()

        signals = [0] * len(data)
        for i in range(1, len(data)):
            if data['macd'].iloc[i] > data['signal'].iloc[i] and \
               data['macd'].iloc[i-1] <= data['signal'].iloc[i-1]:
                signals[i] = 1  # Buy
            elif data['macd'].iloc[i] < data['signal'].iloc[i] and \
                 data['macd'].iloc[i-1] >= data['signal'].iloc[i-1]:
                signals[i] = -1  # Sell

        return signals


class RSIStrategy:
    def __init__(self, period=14, oversold=30, overbought=70):
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def generate_signals(self, data):
        data['rsi'] = ta.momentum.RSIIndicator(data['close'],
                                              window=self.period).rsi()

        signals = [0] * len(data)
        for i in range(1, len(data)):
            if data['rsi'].iloc[i] < self.oversold:
                signals[i] = 1  # Buy
            elif data['rsi'].iloc[i] > self.overbought:
                signals[i] = -1  # Sell

        return signals


class CombinedMACDRSI:
    def __init__(self):
        self.macd = MACDStrategy(fast=3, slow=10, signal=16)
        self.rsi = RSIStrategy(period=14, oversold=20, overbought=80)

    def generate_signals(self, data):
        macd_signals = self.macd.generate_signals(data.copy())
        rsi_signals = self.rsi.generate_signals(data.copy())

        combined = [0] * len(data)
        for i in range(len(data)):
            if macd_signals[i] == 1 and rsi_signals[i] == 1:
                combined[i] = 1
            elif macd_signals[i] == -1 and rsi_signals[i] == -1:
                combined[i] = -1

        return combined


class SimpleDeepQLearning:
    def __init__(self, lookback=20, learning_rate=0.01):
        self.lookback = lookback
        self.learning_rate = learning_rate
        self.q_table = {}

    def _discretize_state(self, returns, volatility):
        return (round(returns, 2), round(volatility, 3))

    def generate_signals(self, data):
        if len(data) < self.lookback:
            return [0] * len(data)

        signals = [0] * len(data)
        closes = data['close'].values

        returns = np.diff(closes) / closes[:-1] * 100
        returns = np.insert(returns, 0, 0)

        volatility = np.zeros(len(closes))
        for i in range(self.lookback, len(closes)):
            volatility[i] = np.std(returns[i-self.lookback:i])

        for i in range(self.lookback, len(closes)):
            state = self._discretize_state(returns[i], volatility[i])

            if state not in self.q_table:
                self.q_table[state] = [0, 0, -1]

            q_values = self.q_table[state]
            best_action = np.argmax(q_values)

            if best_action == 1:
                signals[i] = 1
            elif best_action == 2:
                signals[i] = -1

            if i + 1 < len(closes):
                future_return = (closes[i+1] - closes[i]) / closes[i]
                reward = future_return * 100
                old_value = q_values[best_action]
                new_value = old_value + self.learning_rate * reward
                q_values[best_action] = new_value

        return signals


class EnhancedMLStrategy:
    def __init__(self, lookback=30):
        self.lookback = lookback
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def _calculate_features(self, data):
        closes = data['close'].values
        features = []

        for i in range(len(closes)):
            if i < self.lookback:
                features.append([0, 0, 0])
                continue

            momentum = (closes[i] - closes[i-self.lookback]) / closes[i-self.lookback]
            returns = np.diff(closes[i-self.lookback:i]) / closes[i-self.lookback:i-1]
            volatility = np.std(returns)
            trend = np.mean(returns)

            features.append([momentum, volatility, trend])

        return np.array(features)

    def generate_signals(self, data):
        features = self._calculate_features(data)
        signals = [0] * len(data)

        for i in range(self.lookback, len(data)):
            momentum, volatility, trend = features[i]

            if momentum > 0 and trend > 0 and volatility < 0.05:
                signals[i] = 1
            elif momentum < -0.02 and trend < 0:
                signals[i] = -1
            elif momentum > 0.03 and volatility > 0.1:
                signals[i] = -1

        return signals


class MovingAverageCrossover:
    def __init__(self, fast=20, slow=50, use_ema=False):
        self.fast = fast
        self.slow = slow
        self.use_ema = use_ema

    def generate_signals(self, data):
        if self.use_ema:
            data['ma_fast'] = data['close'].ewm(span=self.fast, adjust=False).mean()
            data['ma_slow'] = data['close'].ewm(span=self.slow, adjust=False).mean()
        else:
            data['ma_fast'] = data['close'].rolling(window=self.fast).mean()
            data['ma_slow'] = data['close'].rolling(window=self.slow).mean()

        signals = [0] * len(data)
        for i in range(1, len(data)):
            if pd.notna(data['ma_fast'].iloc[i]) and pd.notna(data['ma_slow'].iloc[i]):
                if data['ma_fast'].iloc[i] > data['ma_slow'].iloc[i] and \
                   data['ma_fast'].iloc[i-1] <= data['ma_slow'].iloc[i-1]:
                    signals[i] = 1 
                elif data['ma_fast'].iloc[i] < data['ma_slow'].iloc[i] and \
                     data['ma_fast'].iloc[i-1] >= data['ma_slow'].iloc[i-1]:
                    signals[i] = -1 

        return signals


class MomentumStrategy:
    def __init__(self, lookback=14, threshold=0.02):
        self.lookback = lookback
        self.threshold = threshold

    def generate_signals(self, data):
        closes = data['close'].values
        signals = [0] * len(data)

        for i in range(self.lookback, len(data)):
            roc = (closes[i] - closes[i-self.lookback]) / closes[i-self.lookback]

            if roc > self.threshold:
                signals[i] = 1
            elif roc < -self.threshold:
                signals[i] = -1

        return signals


class PairsTradingStrategy:
    def __init__(self, lookback=30, entry_threshold=2.0, exit_threshold=0.5):
        self.lookback = lookback
        self.entry_threshold = entry_threshold 
        self.exit_threshold = exit_threshold

    def generate_signals(self, data):
        data['ma'] = data['close'].rolling(window=self.lookback).mean()
        data['std'] = data['close'].rolling(window=self.lookback).std()

        data['zscore'] = (data['close'] - data['ma']) / data['std']

        signals = [0] * len(data)
        position = 0

        for i in range(self.lookback, len(data)):
            if pd.notna(data['zscore'].iloc[i]):
                zscore = data['zscore'].iloc[i]

                if zscore > self.entry_threshold and position != -1:
                    signals[i] = -1
                    position = -1
                elif zscore < -self.entry_threshold and position != 1:
                    signals[i] = 1
                    position = 1
                elif abs(zscore) < self.exit_threshold:
                    if position == 1:
                        signals[i] = -1
                        position = 0
                    elif position == -1:
                        signals[i] = 1
                        position = 0

        return signals


class CarryStrategy:
    def __init__(self, lookback=20, volatility_threshold=0.02):
        self.lookback = lookback
        self.volatility_threshold = volatility_threshold

    def generate_signals(self, data):
        closes = data['close'].values
        signals = [0] * len(data)

        for i in range(self.lookback, len(data)):
            returns = np.diff(closes[i-self.lookback:i]) / closes[i-self.lookback:i-1]
            avg_return = np.mean(returns)
            volatility = np.std(returns)

            if avg_return > 0 and volatility < self.volatility_threshold:
                signals[i] = 1
            elif avg_return < 0 or volatility > self.volatility_threshold * 2:
                signals[i] = -1

        return signals


class BasisTradingStrategy:
    def __init__(self, lookback=14, threshold=0.01):
        self.lookback = lookback
        self.threshold = threshold

    def generate_signals(self, data):
        data['basis'] = (data['high'] - data['low']) / data['close']
        data['basis_ma'] = data['basis'].rolling(window=self.lookback).mean()

        signals = [0] * len(data)

        for i in range(self.lookback, len(data)):
            if pd.notna(data['basis'].iloc[i]) and pd.notna(data['basis_ma'].iloc[i]):
                # Basis wider than average - contango (sell signal)
                if data['basis'].iloc[i] > data['basis_ma'].iloc[i] * (1 + self.threshold):
                    signals[i] = -1
                # Basis narrower than average - backwardation (buy signal)
                elif data['basis'].iloc[i] < data['basis_ma'].iloc[i] * (1 - self.threshold):
                    signals[i] = 1

        return signals


class PutCallParityArbitrage:
    def __init__(self, lookback=10, volatility_threshold=0.03):
        self.lookback = lookback
        self.volatility_threshold = volatility_threshold

    def generate_signals(self, data):
        closes = data['close'].values
        signals = [0] * len(data)

        for i in range(self.lookback, len(data)):
            returns = np.diff(closes[i-self.lookback:i]) / closes[i-self.lookback:i-1]
            realized_vol = np.std(returns)

            if i >= self.lookback * 2:
                hist_returns = np.diff(closes[i-self.lookback*2:i-self.lookback]) / \
                              closes[i-self.lookback*2:i-self.lookback-1]
                historical_vol = np.std(hist_returns)

                if realized_vol < historical_vol * 0.7:
                    signals[i] = 1
                elif realized_vol > historical_vol * 1.3:
                    signals[i] = -1

        return signals


class SentimentStrategy:
    def __init__(self, lookback=20, volume_threshold=1.5):
        self.lookback = lookback
        self.volume_threshold = volume_threshold

    def generate_signals(self, data):
        data['volume_ma'] = data['volume'].rolling(window=self.lookback).mean()

        # Calculate price momentum
        data['price_change'] = data['close'].pct_change(self.lookback)

        signals = [0] * len(data)

        for i in range(self.lookback, len(data)):
            if pd.notna(data['volume_ma'].iloc[i]) and pd.notna(data['price_change'].iloc[i]):
                volume_ratio = data['volume'].iloc[i] / data['volume_ma'].iloc[i]
                price_change = data['price_change'].iloc[i]

                # Bullish sentiment: high volume + price increase
                if volume_ratio > self.volume_threshold and price_change > 0.02:
                    signals[i] = 1
                # Bearish sentiment: high volume + price decrease
                elif volume_ratio > self.volume_threshold and price_change < -0.02:
                    signals[i] = -1

        return signals
class BollingerBandsStrategy:
    def __init__(self, window=20, num_std=2):
        self.window = window
        self.num_std = num_std
    
    def generate_signals(self, data):
        # Calculate Bollinger Bands
        data['bb_middle'] = data['close'].rolling(window=self.window).mean()
        data['bb_std'] = data['close'].rolling(window=self.window).std()
        data['bb_upper'] = data['bb_middle'] + (self.num_std * data['bb_std'])
        data['bb_lower'] = data['bb_middle'] - (self.num_std * data['bb_std'])
        
        signals = [0] * len(data)
        
        for i in range(1, len(data)):
            if pd.notna(data['bb_upper'].iloc[i]) and pd.notna(data['bb_lower'].iloc[i]):
                # Buy when price touches lower band
                if data['close'].iloc[i] <= data['bb_lower'].iloc[i]:
                    signals[i] = 1
                # Sell when price touches upper band
                elif data['close'].iloc[i] >= data['bb_upper'].iloc[i]:
                    signals[i] = -1
        
        return signals
import pandas as pd
import numpy as np
import ta
from datetime import datetime, timedelta

class InsiderMomentumEnsemble:
    """
    Ensemble strategy combining insider trading signals with momentum
    Targets: >= 20% return, positive alpha
    Data sources: OpenInsider, SEC Form 4, Finviz, QuiverQuant
    """
    
    def __init__(self, 
                 min_insiders=3,
                 min_purchase_value=500000,
                 rsi_period=14,
                 rsi_range=(30, 40),
                 momentum_lookback=14,
                 drawdown_range=(0.10, 0.20),
                 insider_window=60):
        """
        Parameters:
        -----------
        min_insiders : int
            Minimum number of insider buyers for cluster signal
        min_purchase_value : float
            Minimum aggregate insider purchase value ($)
        rsi_period : int
            RSI calculation period
        rsi_range : tuple
            (low, high) RSI range for oversold condition
        momentum_lookback : int
            Lookback period for momentum calculation
        drawdown_range : tuple
            (min, max) acceptable drawdown from 52-week high
        insider_window : int
            Days to look back for insider purchases
        """
        self.min_insiders = min_insiders
        self.min_purchase_value = min_purchase_value
        self.rsi_period = rsi_period
        self.rsi_range = rsi_range
        self.momentum_lookback = momentum_lookback
        self.drawdown_range = drawdown_range
        self.insider_window = insider_window
        
        # Weights for ensemble voting
        self.weights = {
            'insider': 0.40,
            'technical': 0.35,
            'momentum': 0.25
        }
    
    def generate_signals(self, data, insider_data=None):
        """
        Generate trading signals based on ensemble logic
        
        Parameters:
        -----------
        data : pd.DataFrame
            OHLCV price data with columns: timestamp, open, high, low, close, volume
        insider_data : pd.DataFrame, optional
            Insider trading data from OpenInsider/SEC Form 4
            Columns: date, insider_name, relationship, shares, price, value, transaction_type
        
        Returns:
        --------
        list : Trading signals (1=buy, -1=sell, 0=hold)
        """
        if len(data) < max(self.rsi_period, self.momentum_lookback, 252):
            return [0] * len(data)
        
        # Calculate component signals
        insider_signals = self._generate_insider_signals(data, insider_data)
        technical_signals = self._generate_technical_signals(data)
        momentum_signals = self._generate_momentum_signals(data)
        
        # Ensemble voting with weights
        signals = [0] * len(data)
        for i in range(len(data)):
            weighted_score = (
                insider_signals[i] * self.weights['insider'] +
                technical_signals[i] * self.weights['technical'] +
                momentum_signals[i] * self.weights['momentum']
            )
            
            # Buy signal: weighted score > 0.5
            if weighted_score > 0.5:
                signals[i] = 1
            # Sell signal: weighted score < -0.3 (tighter exit)
            elif weighted_score < -0.3:
                signals[i] = -1
            else:
                signals[i] = 0
        
        return signals
    
    def _generate_insider_signals(self, data, insider_data):
        """
        Generate signals from insider trading activity
        Data from: OpenInsider, SEC Form 4, QuiverQuant
        """
        signals = [0] * len(data)
        
        if insider_data is None or len(insider_data) == 0:
            return signals
        
        # Filter for purchases only (transaction_type == 'P' or 'A')
        purchases = insider_data[
            insider_data['transaction_type'].isin(['P', 'A'])
        ].copy()
        
        for i in range(len(data)):
            current_date = data['timestamp'].iloc[i]
            window_start = current_date - timedelta(days=self.insider_window)
            
            # Get recent insider purchases within window
            recent_purchases = purchases[
                (purchases['date'] >= window_start) &
                (purchases['date'] <= current_date)
            ]
            
            if len(recent_purchases) == 0:
                continue
            
            # Count unique insiders
            unique_insiders = recent_purchases['insider_name'].nunique()
            total_value = recent_purchases['value'].sum()
            
            # Strong buy signal: cluster buying + high value
            if (unique_insiders >= self.min_insiders and 
                total_value >= self.min_purchase_value):
                signals[i] = 1
            # Moderate signal: meets one condition
            elif (unique_insiders >= self.min_insiders or 
                  total_value >= self.min_purchase_value * 0.7):
                signals[i] = 0.5
        
        return signals
    
    def _generate_technical_signals(self, data):
        """
        Generate technical confirmation signals
        Uses RSI and drawdown from 52-week high
        Data enriched from: Finviz screener data
        """
        signals = [0] * len(data)
        
        # Calculate RSI
        rsi = ta.momentum.RSIIndicator(
            data['close'], 
            window=self.rsi_period
        ).rsi()
        
        # Calculate 52-week high and drawdown
        closes = data['close'].values
        rolling_max = np.zeros(len(closes))
        
        for i in range(252, len(closes)):
            rolling_max[i] = np.max(closes[max(0, i-252):i])
        
        for i in range(252, len(data)):
            if rolling_max[i] == 0:
                continue
            
            current_price = closes[i]
            drawdown = (rolling_max[i] - current_price) / rolling_max[i]
            
            # Buy signal: RSI oversold + acceptable drawdown
            if (self.rsi_range[0] <= rsi.iloc[i] <= self.rsi_range[1] and
                self.drawdown_range[0] <= drawdown <= self.drawdown_range[1]):
                signals[i] = 1
            # Sell signal: RSI overbought
            elif rsi.iloc[i] > 70:
                signals[i] = -1
        
        return signals
    
    def _generate_momentum_signals(self, data):
        """
        Generate momentum reversal signals
        Adapted from existing MomentumStrategy
        """
        signals = [0] * len(data)
        closes = data['close'].values
        
        for i in range(self.momentum_lookback, len(data)):
            # Rate of change
            roc = (closes[i] - closes[i-self.momentum_lookback]) / closes[i-self.momentum_lookback]
            
            # Check for momentum reversal
            if i >= self.momentum_lookback + 1:
                prev_roc = (closes[i-1] - closes[i-1-self.momentum_lookback]) / closes[i-1-self.momentum_lookback]
                
                # Bullish reversal: negative to positive momentum
                if prev_roc < -0.02 and roc > 0:
                    signals[i] = 1
                # Bearish signal: strong negative momentum
                elif roc < -0.05:
                    signals[i] = -1
                # Hold positive momentum
                elif roc > 0.02:
                    signals[i] = 0.5
        
        return signals
    
    def fetch_insider_data(self, symbol, lookback_days=90):
        """
        Fetch insider trading data from multiple sources
        Sources: OpenInsider, SEC API, QuiverQuant
        
        Note: This is a template - implement actual API calls
        """
        # Template for data fetching
        # In production, integrate with:
        # 1. OpenInsider scraper (openinsider.com)
        # 2. SEC-API.io for Form 4 filings
        # 3. QuiverQuant API for aggregated signals
        # 4. Finviz for supplementary screening
        
        insider_data = pd.DataFrame()
        
        # Example structure (implement actual API calls):
        # insider_data = fetch_openinsider(symbol, lookback_days)
        # sec_data = fetch_sec_form4(symbol, lookback_days)
        # quiver_data = fetch_quiverquant(symbol, lookback_days)
        # insider_data = pd.concat([insider_data, sec_data, quiver_data])
        
        return insider_data


class MomentumCarryEnsemble:
    """
    Alternative ensemble: Momentum + Carry strategies
    Both showed strong performance in your backtests
    Targets: >= 20% return, positive alpha
    """
    
    def __init__(self, 
                 momentum_lookback=14,
                 momentum_threshold=0.02,
                 carry_lookback=20,
                 volatility_threshold=0.02):
        self.momentum_lookback = momentum_lookback
        self.momentum_threshold = momentum_threshold
        self.carry_lookback = carry_lookback
        self.volatility_threshold = volatility_threshold
        
        # Equal weighting for simplicity
        self.momentum_weight = 0.5
        self.carry_weight = 0.5
    
    def generate_signals(self, data):
        """
        Combine momentum and carry strategies
        Both strategies must agree for entry
        """
        momentum_signals = self._momentum_signals(data)
        carry_signals = self._carry_signals(data)
        
        signals = [0] * len(data)
        for i in range(len(data)):
            # Require agreement for buy signals
            if momentum_signals[i] == 1 and carry_signals[i] == 1:
                signals[i] = 1
            # Require either for sell signals (risk management)
            elif momentum_signals[i] == -1 or carry_signals[i] == -1:
                signals[i] = -1
        
        return signals
    
    def _momentum_signals(self, data):
        """Momentum strategy component"""
        signals = [0] * len(data)
        closes = data['close'].values
        
        for i in range(self.momentum_lookback, len(data)):
            roc = (closes[i] - closes[i-self.momentum_lookback]) / closes[i-self.momentum_lookback]
            
            if roc > self.momentum_threshold:
                signals[i] = 1
            elif roc < -self.momentum_threshold:
                signals[i] = -1
        
        return signals
    
    def _carry_signals(self, data):
        """Carry strategy component"""
        signals = [0] * len(data)
        closes = data['close'].values
        
        for i in range(self.carry_lookback, len(data)):
            returns = np.diff(closes[i-self.carry_lookback:i]) / closes[i-self.carry_lookback:i-1]
            avg_return = np.mean(returns)
            volatility = np.std(returns)
            
            # Positive carry with low volatility
            if avg_return > 0 and volatility < self.volatility_threshold:
                signals[i] = 1
            # Negative carry or high volatility
            elif avg_return < 0 or volatility > self.volatility_threshold * 2:
                signals[i] = -1
        
        return signals
