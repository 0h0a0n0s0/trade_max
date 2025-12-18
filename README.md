# TradeMax - Rank 77 Strategy

A professional grid trading bot optimized for BTC/TWD, featuring "Rank 77" parameters that balance profitability in bull markets with hard-stop safety in bear markets.

## 📁 Project Structure

- **core/**: The heart of the bot (Backtester & Optimizer).
- **configs/**: Strategy configurations (includes `config_rank77.yaml`).
- **data/**: Historical market data (CSV).
- **tools/**: Helper scripts for batch validation and log analysis.
- **docs/**: Detailed documentation and guides.

## 🚀 Quick Start

### 1. Run Backtest (Rank 77)

To test the strategy against 2024 data:

```bash
python core/backtester.py --csv data/btctwd_1m_2024.csv --config configs/config_rank77.yaml
```

### 2. Optimization (Optional)

To train new parameters:

```bash
python core/optimizer.py --csv data/btctwd_1m_2024.csv --config configs/config_rank77.yaml --n-trials 100
```

## 📚 Documentation

- [Strategy Explanation](docs/guides/strategy_explanation.md)
- [Running Guide](docs/guides/running_guide.md)
