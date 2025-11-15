import yfinance as yf
import pandas as pd
import numpy as np
import inspect
import json
from datetime import datetime
import strategies
import warnings
warnings.filterwarnings('ignore')
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import time
import pickle
from pathlib import Path
import multiprocessing
from report_generator import ReportGenerator

print_lock = Lock()

def safe_print(*args, **kwargs):
    with print_lock:
        print(*args, **kwargs)

class DataCache:
    def __init__(self, cache_dir='./cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.memory_cache = {}
        self.cache_lock = Lock()
        self.download_lock = Lock()
        self.last_download_time = {}
        self.min_request_interval = 0.25

    def get_cache_path(self, symbol, period, interval):
        """Generate cache file path"""
        filename = f"{symbol}_{period}_{interval}.pkl"
        return self.cache_dir / filename

    def load_from_disk(self, symbol, period, interval):
        """Load from disk cache"""
        cache_path = self.get_cache_path(symbol, period, interval)

        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                    safe_print(f"Cache: {symbol} {interval}")
                    return data
            except:
                pass
        return None

    def save_to_disk(self, symbol, period, interval, data):
        cache_path = self.get_cache_path(symbol, period, interval)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
        except:
            pass

    def fetch_with_rate_limit(self, symbol, period, interval):
        disk_data = self.load_from_disk(symbol, period, interval)
        if disk_data is not None:
            with self.cache_lock:
                self.memory_cache[f"{symbol}_{period}_{interval}"] = disk_data.copy()
            return disk_data.copy()

        cache_key = f"{symbol}_{period}_{interval}"
        with self.cache_lock:
            if cache_key in self.memory_cache:
                safe_print(f"Memory: {symbol} {interval}")
                return self.memory_cache[cache_key].copy()

        with self.download_lock:
            with self.cache_lock:
                if cache_key in self.memory_cache:
                    return self.memory_cache[cache_key].copy()

            last_time = self.last_download_time.get('last_download', 0)
            elapsed = time.time() - last_time
            if elapsed < self.min_request_interval:
                time.sleep(self.min_request_interval - elapsed)

            safe_print(f"Download: {symbol} {interval}")

            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(
                    period=period,
                    interval=interval,
                    prepost=False,
                    auto_adjust=True,
                    timeout=10
                )

                if df.empty or len(df) < 30:
                    safe_print(f"No data: {symbol} {interval}")
                    return None

                df.reset_index(inplace=True)
                df.columns = df.columns.str.lower()

                if 'date' in df.columns:
                    df.rename(columns={'date': 'timestamp'}, inplace=True)
                elif 'datetime' in df.columns:
                    df.rename(columns={'datetime': 'timestamp'}, inplace=True)

                result = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()

                with self.cache_lock:
                    self.memory_cache[cache_key] = result.copy()
                self.save_to_disk(symbol, period, interval, result)

                self.last_download_time['last_download'] = time.time()
                safe_print(f"Downloaded: {symbol} {interval} ({len(result)} bars)")

                return result

            except Exception as e:
                safe_print(f"Error: {symbol} {interval}: {str(e)[:40]}")
                return None

    def prefetch(self, assets, intervals_config):
        safe_print("\n🔄 Pre-fetching all data (rate-limited)...")
        safe_print("   First run downloads, future runs use cache\n")

        total = len(assets) * len(intervals_config)
        count = 0
        downloaded = 0
        cached = 0

        for asset in assets:
            symbol = asset['symbol']

            for interval_config in intervals_config:
                count += 1
                interval = interval_config['interval']
                period  = interval_config['period']

                prev_ts = self.last_download_time.get('last_download', 0)

                data = self.fetch_with_rate_limit(symbol, period, interval)
                if data is None:
                    pass
                else:
                    new_ts = self.last_download_time.get('last_download', 0)
                    if new_ts != prev_ts:
                        downloaded += 1
                    else:
                        cached += 1

                if count % 10 == 0:
                    safe_print(f"   Progress: {count}/{total} ({cached} cached, {downloaded} downloaded)")

        safe_print(f"\n✅ Pre-fetch complete!")
        safe_print(f"   Total: {total} | Cached: {cached} | Downloaded: {downloaded}\n")


class Backtester:

    __slots__ = ['data', 'strategy', 'interval', 'initial_capital', 'commission', 'slippage',
                 'position_size_pct', 'capital', 'position', 'entry_price', 
                 'trades', 'signals', 'risk_free_annual']

    def __init__(self, data, strategy, interval='1d', initial_capital=10000, 
                 commission=0.001, slippage=0.0005, position_size_pct=0.95, risk_free_annual=0.04):

        self.data = data
        self.strategy = strategy
        self.interval = interval
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.position_size_pct = position_size_pct
        self.capital = initial_capital
        self.position = 0
        self.entry_price = 0
        self.trades = []
        self.signals = []
        self.risk_free_annual = risk_free_annual

    def run_fast(self):
        try:
            signals = np.nan_to_num(np.asarray(self.strategy.generate_signals(self.data)), nan=0.0)
            signals = np.sign(signals)
        except Exception:
            return None
        
        signals = np.roll(signals, 1)
        signals[0] = 0

        warmup = getattr(self.strategy, 'warmup_bars', 0)
        if warmup > 0:
            signals[:warmup] = 0
        
        self.signals = signals

        closes = self.data['close'].values
        n = len(closes)
        portfolio_values = np.zeros(n)

        for i in range(n):
            close_price = closes[i]
            signal = self.signals[i]

            portfolio_values[i] = self.capital + (self.position * close_price)

            if signal == 1 and self.position == 0:
                shares_to_buy = (self.capital * self.position_size_pct) / close_price
                if shares_to_buy > 0:
                    execution_price = close_price * (1 + self.slippage)
                    cost = shares_to_buy * execution_price * (1 + self.commission)
                    if cost <= self.capital:
                        self.trades.append({'i': i, 'side': 'buy', 'price': execution_price, 'size': shares_to_buy})
                        self.position = shares_to_buy
                        self.capital -= cost
                        self.entry_price = execution_price

            elif signal == -1 and self.position > 0:
                execution_price = close_price * (1 - self.slippage)
                proceeds = self.position * execution_price * (1 - self.commission)
                self.trades.append({'i': i, 'side': 'sell', 'price': execution_price, 'size': self.position})
                self.capital += proceeds
                self.position = 0

        if self.position > 0:
            execution_price = closes[-1] * (1 - self.slippage)
            proceeds = self.position * execution_price * (1 - self.commission)
            self.trades.append({'i': n-1, 'side': 'sell', 'price': execution_price, 'size': self.position})
            self.capital += proceeds
            self.position = 0

        return portfolio_values

    @staticmethod
    def bars_per_year(interval, market = 'equity'):
        if market == 'crypto':
            days_per_year = 365
            hours_per_day = 24
            mins_per_day = 24*60
        else:
            days_per_year = 252
            hours_per_day = 6.5
            mins_per_day = 390

        if interval.endswith('d'):
            return days_per_year

        elif interval.endswith('h'):
            hours = int(interval[:-1])
            return int((hours_per_day / hours) * days_per_year)

        elif interval.endswith('m'):
            mins = int(interval[:-1])
            return int((mins_per_day / mins) * days_per_year)

        return days_per_year


    def calculate_performance_fast(self, portfolio_values):
        if portfolio_values is None or len(portfolio_values) == 0:
            return None

        final_value = portfolio_values[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital

        cummax = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - cummax) / cummax
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0

        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        returns = returns[np.isfinite(returns)]

        bpyr = Backtester.bars_per_year(self.interval)

        rf_bar = (1 + self.risk_free_annual)**(1 / bpyr) - 1
        excess = returns - rf_bar
        
        if np.std(excess) > 0:
            sharpe = np.mean(excess) / np.std(excess) * np.sqrt(bpyr)
        else:
            sharpe = 0.0

        buy_hold_return = (self.data['close'].iloc[-1] - self.data['close'].iloc[0]) / self.data['close'].iloc[0]
        alpha = total_return - buy_hold_return

        num_trades = sum(1 for t in self.trades if t.get('side') == 'sell')

        return {
            'total_return': total_return,
            'final_value': final_value,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'alpha': alpha,
            'num_trades': num_trades
        }

def get_all_strategies():
    strategy_classes = []
    for name in dir(strategies):
        obj = getattr(strategies, name)
        if inspect.isclass(obj) and hasattr(obj, 'generate_signals'):
            try:
                instance = obj()
                strategy_classes.append(instance)
            except:
                pass
    return strategy_classes

def test_combination_cached(args):
    asset, interval_config, strategy, cache = args

    symbol = asset['symbol']
    interval = interval_config['interval']
    period = interval_config['period']
    strategy_name = strategy.__class__.__name__

    try:
        data = cache.fetch_with_rate_limit(symbol, period, interval)

        if data is None or len(data) < 30:
            return None

        backtester = Backtester(data, strategy, interval=interval)
        portfolio_values = backtester.run_fast()
        metrics = backtester.calculate_performance_fast(portfolio_values)

        if metrics:
            return_pct = metrics['total_return'] * 100

            if return_pct > 20:
                status = "🟢"
            elif return_pct > 0:
                status = "🟡"
            else:
                status = "🔴"

            safe_print(f"  {status} {symbol:10s} | {interval:4s} | {strategy_name:20s} | R: {return_pct:7.2f}% | A: {metrics['alpha']*100:7.2f}%")

            return {
                'symbol': symbol,
                'asset_name': asset['name'],
                'asset_type': asset['type'],
                'interval': interval,
                'strategy_name': strategy_name,
                'metrics': metrics
            }

        return None

    except Exception as e:
        return None

def run_cached_backtest(assets_file='assets.json',
                              test_all_intervals=True,
                              max_workers=None):

    if max_workers is None:
        max_workers = multiprocessing.cpu_count()

    with open(assets_file, 'r') as f:
        config = json.load(f)

    assets = config['assets']
    intervals_config = config['settings']['intervals']

    if not test_all_intervals:
        intervals_config = [intervals_config[-1]]

    all_strategies = get_all_strategies()
    cache = DataCache()

    safe_print("\n" + "="*70)
    safe_print("CACHED BACKTESTER v3")
    safe_print("="*70)
    safe_print(f"Assets: {len(assets)}")
    safe_print(f"Intervals: {len(intervals_config)}")
    safe_print(f"Strategies: {len(all_strategies)}")
    safe_print(f"Threads: {max_workers}")
    safe_print(f"Cache: ./cache/")
    safe_print("="*70)

    cache.prefetch(assets, intervals_config)

    tasks = []
    for asset in assets:
        for interval_config in intervals_config:
            for strategy in all_strategies:
                tasks.append((asset, interval_config, strategy, cache))

    results = []
    completed = 0
    total = len(tasks)

    safe_print(f"⚡ Testing {total} combinations on cached data...\n")
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(test_combination_cached, task): task for task in tasks}

        for future in as_completed(futures):
            completed += 1
            result = future.result()

            if result:
                results.append(result)

            if completed % 100 == 0:
                elapsed = time.time() - start_time
                rate = completed / elapsed
                remaining = (total - completed) / rate if rate > 0 else 0
                safe_print(f"\n📊 [{completed}/{total}] Rate: {rate:.1f}/s | ETA: {remaining:.0f}s\n")

    elapsed_time = time.time() - start_time

    safe_print("\n" + "="*70)
    safe_print("BACKTEST COMPLETE!")
    safe_print("="*70)
    safe_print(f"Time: {elapsed_time:.1f}s")
    safe_print(f"Tests: {len(results)}/{total}")
    safe_print(f"Rate: {total/elapsed_time:.1f} tests/sec")
    safe_print(f"Cache: ./cache/")
    safe_print("="*70)

    return results

if __name__ == "__main__":

    results = run_cached_backtest(
        assets_file='assets.json',
        test_all_intervals=True,
        max_workers=None
    )
    if results:
        report_gen = ReportGenerator(output_dir='./reports')
        report_gen.generate(results, output_file='backtest_report')
        print("\n DONE!")