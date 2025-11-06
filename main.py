import yfinance as yf
import pandas as pd
import inspect
import json
from datetime import datetime
import strategies

class DataFetcher:
    def __init__(self):
        pass
    def fetch_ohlcv(self, symbol, period='1y', interval='1h'):
        """Fetch historical OHLCV data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            
            if df.empty:
                return None
            
            df.columns = df.columns.str.lower()
            df = df.reset_index()
            df.columns = df.columns.str.lower()
            
            if 'date' in df.columns:
                df = df.rename(columns={'date': 'timestamp'})
            elif 'datetime' in df.columns:
                df = df.rename(columns={'datetime': 'timestamp'})
            
            return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        except Exception as e:
            print(f"  Error fetching data: {e}")
            return None

class Backtester:
    def __init__(self, data, strategy, initial_capital=10000):
        self.data = data
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position = 0
        self.trades = []
    
    def run(self):
        signals = self.strategy.generate_signals(self.data.copy())
        
        for i in range(len(self.data)):
            if signals[i] == 1 and self.position == 0:
                self.position = self.capital / self.data['close'].iloc[i]
                self.capital = 0
                self.trades.append(('BUY', self.data['timestamp'].iloc[i], self.data['close'].iloc[i]))
            elif signals[i] == -1 and self.position > 0:
                self.capital = self.position * self.data['close'].iloc[i]
                self.position = 0
                self.trades.append(('SELL', self.data['timestamp'].iloc[i], self.data['close'].iloc[i]))
        
        return self.calculate_performance()
    
    def calculate_performance(self):
        final_value = self.capital + (self.position * self.data['close'].iloc[-1])
        total_return = ((final_value / self.initial_capital) - 1) * 100
        
        portfolio_values = [self.initial_capital]
        running_capital = self.initial_capital
        running_position = 0
        
        signals = self.strategy.generate_signals(self.data.copy())
        for i in range(len(self.data)):
            if signals[i] == 1 and running_position == 0:
                running_position = running_capital / self.data['close'].iloc[i]
                running_capital = 0
            elif signals[i] == -1 and running_position > 0:
                running_capital = running_position * self.data['close'].iloc[i]
                running_position = 0
            portfolio_values.append(running_capital + running_position * self.data['close'].iloc[i])
        
        peak = portfolio_values[0]
        max_dd = 0
        for value in portfolio_values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak * 100
            if dd > max_dd:
                max_dd = dd
        
        return {
            'strategy': self.strategy.__class__.__name__,
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'return_pct': total_return,
            'max_drawdown': max_dd,
            'num_trades': len(self.trades),
        }

def load_config(config_file='assets.json'):
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"Error: {config_file} not found!")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing {config_file}: {e}")
        return None

def get_all_strategies():
    strategy_classes = []
    
    for name, obj in inspect.getmembers(strategies, inspect.isclass):
        if obj.__module__ == 'strategies':
            strategy_classes.append(obj)
    
    return strategy_classes

def test_all_strategies(data, asset_info, interval_info, initial_capital=10000):
    strategy_classes = get_all_strategies()
    results = []
    
    print(f"  Testing {len(strategy_classes)} strategies on {interval_info['interval']} timeframe...")
    
    for strategy_class in strategy_classes:
        try:
            strategy = strategy_class()
            backtester = Backtester(data, strategy, initial_capital)
            result = backtester.run()
            result['asset_symbol'] = asset_info['symbol']
            result['asset_name'] = asset_info['name']
            result['interval'] = interval_info['interval']
            result['period'] = interval_info['period']
            results.append(result)
            
            print(f"    ✓ {result['strategy']:20} → {result['return_pct']:7.2f}% return, "
                  f"{result['max_drawdown']:6.2f}% drawdown, {result['num_trades']:3} trades")
            
        except Exception as e:
            print(f"    ✗ Error: {strategy_class.__name__}: {e}")
    
    return results

def generate_summary(all_results):
    print(f"\n{'='*90}")
    print(f"COMPREHENSIVE SUMMARY")
    print(f"{'='*90}\n")
    print("Best Strategy Per Asset & Interval:")
    print("-" * 90)
    
    assets = {}
    for result in all_results:
        key = (result['asset_symbol'], result['interval'])
        if key not in assets:
            assets[key] = []
        assets[key].append(result)
    
    for (asset, interval), results in sorted(assets.items()):
        best = max(results, key=lambda x: x['return_pct'])
        asset_name = best['asset_name']
        print(f"{asset_name:15} ({asset:10}) @ {interval:5} → {best['strategy']:20} "
              f"{best['return_pct']:7.2f}% ({best['num_trades']:3} trades)")
    print(f"\n{'='*90}")
    print("Top 10 Overall Performances (Asset + Interval + Strategy):")
    print("-" * 90)
    sorted_results = sorted(all_results, key=lambda x: x['return_pct'], reverse=True)[:10]
    for i, result in enumerate(sorted_results, 1):
        print(f"{i:2}. {result['asset_name']:15} @ {result['interval']:5} + {result['strategy']:20} → "
              f"{result['return_pct']:7.2f}% return, {result['max_drawdown']:6.2f}% drawdown")
    print(f"\n{'='*90}")
    print("Strategy Performance Across All Assets & Intervals (Average):")
    print("-" * 90)
    strategy_stats = {}
    for result in all_results:
        strat = result['strategy']
        if strat not in strategy_stats:
            strategy_stats[strat] = {'returns': [], 'drawdowns': [], 'trades': []}
        strategy_stats[strat]['returns'].append(result['return_pct'])
        strategy_stats[strat]['drawdowns'].append(result['max_drawdown'])
        strategy_stats[strat]['trades'].append(result['num_trades'])
    
    for strategy, stats in sorted(strategy_stats.items(), 
                                   key=lambda x: sum(x[1]['returns'])/len(x[1]['returns']), 
                                   reverse=True):
        avg_return = sum(stats['returns']) / len(stats['returns'])
        avg_drawdown = sum(stats['drawdowns']) / len(stats['drawdowns'])
        avg_trades = sum(stats['trades']) / len(stats['trades'])
        num_tests = len(stats['returns'])
        print(f"{strategy:25} → Avg Return: {avg_return:7.2f}%, "
              f"Avg Drawdown: {avg_drawdown:6.2f}%, Avg Trades: {avg_trades:5.1f} ({num_tests} tests)")
    print(f"\n{'='*90}")
    print("Timeframe Performance (Average Across All Assets & Strategies):")
    print("-" * 90)
    interval_stats = {}
    for result in all_results:
        interval = result['interval']
        if interval not in interval_stats:
            interval_stats[interval] = {'returns': [], 'drawdowns': []}
        interval_stats[interval]['returns'].append(result['return_pct'])
        interval_stats[interval]['drawdowns'].append(result['max_drawdown'])
    
    for interval, stats in sorted(interval_stats.items(), 
                                   key=lambda x: sum(x[1]['returns'])/len(x[1]['returns']), 
                                   reverse=True):
        avg_return = sum(stats['returns']) / len(stats['returns'])
        avg_drawdown = sum(stats['drawdowns']) / len(stats['drawdowns'])
        num_tests = len(stats['returns'])
        print(f"{interval:10} → Avg Return: {avg_return:7.2f}%, "
              f"Avg Drawdown: {avg_drawdown:6.2f}% ({num_tests} tests)")

if __name__ == '__main__':
    config = load_config('assets.json')
    if not config:
        exit(1)
    
    assets = config['assets']
    intervals = config['settings']['intervals']
    initial_capital = config['settings']['initial_capital']
    
    fetcher = DataFetcher()
    all_results = []
    
    print(f"\n{'='*90}")
    print(f"STARTING MULTI-INTERVAL BACKTEST SESSION")
    print(f"{'='*90}")
    print(f"Assets: {len(assets)} | Intervals: {len(intervals)} | Initial Capital: ${initial_capital:,}")
    print(f"Total tests per strategy: {len(assets) * len(intervals)}")
    print(f"{'='*90}")
    
    for asset in assets:
        print(f"\n{'='*90}")
        print(f"Processing: {asset['name']} ({asset['symbol']})")
        print(f"{'='*90}")
        
        for interval_config in intervals:
            interval = interval_config['interval']
            period = interval_config['period']
            
            print(f"\n  Fetching {interval} data (period: {period})...")
            
            try:
                data = fetcher.fetch_ohlcv(asset['symbol'], period=period, interval=interval)
                
                if data is None or data.empty:
                    print(f"  ⚠ Warning: No data retrieved for {asset['symbol']} @ {interval}")
                    continue
                
                print(f"  ✓ Retrieved {len(data)} data points")
                
                results = test_all_strategies(data, asset, interval_config, initial_capital)
                all_results.extend(results)
                
            except Exception as e:
                print(f"  ✗ Error processing {asset['symbol']} @ {interval}: {e}")
                continue
    
    if all_results:
        generate_summary(all_results)
        print(f"\n{'='*90}")
        print(f"BACKTEST COMPLETE - Tested {len(all_results)} combinations")
        print(f"{'='*90}\n")
    else:
        print("\n⚠ No results to display!")
