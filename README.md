# ESN + GA Quant Project

## Overview

This project aims to build a quantitative trading system using:

* Financial time series data
* Technical indicators
* Genetic Algorithm (GA)
* Echo State Network (ESN)

## Project Structure

```
data/
  raw/
  processed/
src/
  data/
  indicators/
  ga/
  esn/
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

## Data

* Source: Yahoo Finance
* Assets: SPY, QQQ, GLD
* Features:

  * OHLCV
  * pct_return
  * log_return

## Pipeline

1. Download data
2. Preprocess data
3. Generate indicators
4. Optimize with GA
5. Train ESN

## TODO

* [ ] Add technical indicators
* [ ] Implement GA
* [ ] Implement ESN
