import pandas as pd
from datetime import datetime
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np

class ReportGenerator:
    """Generates HTML and CSV reports from backtest results with interactive charts"""
    
    def __init__(self, output_dir='./reports'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def generate(self, results, output_file='backtest_report'):
        """Generate both HTML and CSV reports with charts"""
        all_results = []
        
        for result in results:
            metrics = result['metrics']
            all_results.append({
                'Symbol': result['symbol'],
                'Asset': result['asset_name'],
                'Type': result['asset_type'],
                'Interval': result['interval'],
                'Strategy': result['strategy_name'],
                'Return (%)': metrics['total_return'] * 100,
                'Alpha (%)': metrics['alpha'] * 100,
                'Sharpe': metrics['sharpe_ratio'],
                'Max DD (%)': metrics['max_drawdown'] * 100
            })
        
        df = pd.DataFrame(all_results)
        df_sorted = df.sort_values('Return (%)', ascending=False)
        
        # Generate charts
        chart1_html = self._create_avg_return_vs_alpha_chart(df)
        chart2_html = self._create_timeframe_return_vs_alpha_chart(df)
        chart3_html = self._create_profitability_bar_chart(df)
        chart4_html = self._create_strategy_performance_over_time_charts(results)
        
        # Generate reports with charts
        self._generate_html(df_sorted, output_file, chart1_html, chart2_html, chart3_html, chart4_html)
        self._generate_csv(df_sorted, output_file)
        
        return df_sorted
    
    def _create_avg_return_vs_alpha_chart(self, df):
        """Create scatter plot: Average Return vs Alpha across all timeframes"""
        strategy_avg = df.groupby('Strategy').agg({
            'Return (%)': 'mean',
            'Alpha (%)': 'mean'
        }).reset_index()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=strategy_avg['Return (%)'],
            y=strategy_avg['Alpha (%)'],
            mode='markers+text',
            name='Strategies',
            text=strategy_avg['Strategy'],
            textposition='top center',
            marker=dict(
                size=12,
                color=strategy_avg['Return (%)'],
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Return (%)")
            ),
            hovertemplate='<b>%{text}</b><br>Return: %{x:.2f}%<br>Alpha: %{y:.2f}%<extra></extra>'
        ))
        
        fig.update_layout(
            title='Strategy Performance: Average Return vs Alpha',
            xaxis_title='Average Return (%)',
            yaxis_title='Average Alpha (%)',
            hovermode='closest',
            height=600
        )
        
        return fig.to_html(full_html=False, include_plotlyjs='cdn')
    
    def _create_timeframe_return_vs_alpha_chart(self, df):
        """Create grouped bar chart showing Return vs Alpha by timeframe"""
        interval_avg = df.groupby('Interval').agg({
            'Return (%)': 'mean',
            'Alpha (%)': 'mean'
        }).reset_index()
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Average Return',
            x=interval_avg['Interval'],
            y=interval_avg['Return (%)'],
            marker_color='lightblue'
        ))
        
        fig.add_trace(go.Bar(
            name='Average Alpha',
            x=interval_avg['Interval'],
            y=interval_avg['Alpha (%)'],
            marker_color='lightgray'
        ))
        
        fig.update_layout(
            title='Performance by Timeframe: Return vs Alpha',
            xaxis_title='Timeframe',
            yaxis_title='Percentage (%)',
            barmode='group',
            height=500
        )
        
        return fig.to_html(full_html=False, include_plotlyjs='cdn')
    
    def _create_profitability_bar_chart(self, df):
        """Create bar chart comparing strategies by return and alpha"""
        strategy_grouped = df.groupby('Strategy').agg({
            'Return (%)': 'mean',
            'Alpha (%)': 'mean'
        }).reset_index().sort_values('Return (%)', ascending=False)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Average Return',
            x=strategy_grouped['Strategy'],
            y=strategy_grouped['Return (%)'],
            marker_color='green',
            opacity=0.7
        ))
        
        fig.add_trace(go.Bar(
            name='Average Alpha',
            x=strategy_grouped['Strategy'],
            y=strategy_grouped['Alpha (%)'],
            marker_color='gray',
            opacity=0.5
        ))
        
        fig.update_layout(
            title='Strategy Comparison: Return and Alpha',
            xaxis_title='Strategy',
            yaxis_title='Percentage (%)',
            barmode='overlay',
            height=500
        )
        
        return fig.to_html(full_html=False, include_plotlyjs='cdn')
    
    def _create_strategy_performance_over_time_charts(self, results):
        """Create time series charts for each strategy showing cumulative returns"""
        strategies = list(set([r['strategy_name'] for r in results]))
        
        html_parts = []
        
        for strategy in strategies:
            strategy_results = [r for r in results if r['strategy_name'] == strategy]
            
            fig = go.Figure()
            
            for result in strategy_results:
                if 'equity_curve' in result and result['equity_curve'] is not None:
                    equity = result['equity_curve']
                    label = f"{result['asset_name']} ({result['interval']})"
                    
                    fig.add_trace(go.Scatter(
                        x=list(range(len(equity))),
                        y=equity,
                        mode='lines',
                        name=label,
                        hovertemplate=f'{label}<br>Time: %{{x}}<br>Value: %{{y:.2f}}<extra></extra>'
                    ))
            
            fig.update_layout(
                title=f'{strategy} - Performance Over Time',
                xaxis_title='Time Period',
                yaxis_title='Portfolio Value',
                height=500,
                hovermode='x unified'
            )
            
            html_parts.append(fig.to_html(full_html=False, include_plotlyjs='cdn'))
        
        return '<div class="strategy-charts">' + ''.join(html_parts) + '</div>'
    
    def _generate_html(self, df, output_file, chart1_html, chart2_html, chart3_html, chart4_html):
        """Generate HTML report with embedded charts"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        html_content = f'''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Backtest Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
            border-radius: 8px;
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 40px;
            margin-bottom: 20px;
        }}
        .timestamp {{
            color: #888;
            font-size: 14px;
            margin-bottom: 20px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 14px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .positive {{
            color: #4CAF50;
            font-weight: bold;
        }}
        .negative {{
            color: #f44336;
            font-weight: bold;
        }}
        .chart-container {{
            margin: 30px 0;
            padding: 20px;
            background-color: #fafafa;
            border-radius: 5px;
        }}
        .strategy-charts {{
            margin-top: 30px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Backtest Performance Report</h1>
        <p class="timestamp">Generated: {timestamp}</p>
        
        <h2>Summary Statistics</h2>
        <table>
            <thead>
                <tr>
                    <th>Symbol</th>
                    <th>Asset</th>
                    <th>Type</th>
                    <th>Interval</th>
                    <th>Strategy</th>
                    <th>Return (%)</th>
                    <th>Alpha (%)</th>
                    <th>Sharpe</th>
                    <th>Max DD (%)</th>
                </tr>
            </thead>
            <tbody>
'''
        
        # Add table rows
        for _, row in df.iterrows():
            return_class = 'positive' if row['Return (%)'] > 0 else 'negative'
            alpha_class = 'positive' if row['Alpha (%)'] > 0 else 'negative'
            
            html_content += f'''
                <tr>
                    <td>{row['Symbol']}</td>
                    <td>{row['Asset']}</td>
                    <td>{row['Type']}</td>
                    <td>{row['Interval']}</td>
                    <td>{row['Strategy']}</td>
                    <td class="{return_class}">{row['Return (%)']:.2f}</td>
                    <td class="{alpha_class}">{row['Alpha (%)']:.2f}</td>
                    <td>{row['Sharpe']:.2f}</td>
                    <td class="negative">{row['Max DD (%)']:.2f}</td>
                </tr>
'''
        
        html_content += f'''
            </tbody>
        </table>
        
        <h2>Performance Analysis</h2>
        
        <div class="chart-container">
            {chart1_html}
        </div>
        
        <div class="chart-container">
            {chart2_html}
        </div>
        
        <div class="chart-container">
            {chart3_html}
        </div>
        
        <h2>Strategy Performance Over Time</h2>
        <div class="chart-container">
            {chart4_html}
        </div>
    </div>
</body>
</html>
'''
        
        # Write HTML file
        output_path = self.output_dir / f'{output_file}.html'
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"HTML report generated: {output_path}")
    
    def _generate_csv(self, df, output_file):
        """Generate CSV report"""
        output_path = self.output_dir / f'{output_file}.csv'
        df.to_csv(output_path, index=False)
        print(f"CSV report generated: {output_path}")
    