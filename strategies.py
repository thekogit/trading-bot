import pandas as pd
import ta 

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
        """Combine MACD and RSI for stronger signals"""
        macd_signals = self.macd.generate_signals(data.copy())
        rsi_signals = self.rsi.generate_signals(data.copy())
        combined = [0] * len(data)
        for i in range(len(data)):
            if macd_signals[i] == 1 and rsi_signals[i] == 1:
                combined[i] = 1
            elif macd_signals[i] == -1 and rsi_signals[i] == -1:
                combined[i] = -1
        
        return combined
