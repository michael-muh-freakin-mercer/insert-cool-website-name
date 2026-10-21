# Bolo-v1: The Baseline House Bot & Moving-Average Benchmark Gladiator

> *"Nobody enters an arena without an opponent to punch."*

`bolo_v1.py` is the official baseline reference trading bot for **WARBOT** (2027 Autonomous Capital Combat). It establishes the public moving-average performance tape that all community entrants must outperform to qualify or win.

---

## 1. Core Mechanics & Strategy

Bolo-v1 implements a disciplined **Dual Exponential Moving Average (EMA) Crossover** strategy.

### Indicator Formula
The Exponential Moving Average (EMA) assigns exponentially decreasing weights to historical prices:
$$\alpha = \frac{2}{N + 1}$$
$$\text{EMA}_t = \text{Price}_t \times \alpha + \text{EMA}_{t-1} \times (1 - \alpha)$$

- **Default Parameters:**
  - Fast EMA: **9 periods** (`--fast-ema 9`)
  - Slow EMA: **21 periods** (`--slow-ema 21`)
  - Bar Timeframe: **5-minute bars** (`--timeframe 5Min`)
  - Universe: **SPY, QQQ** (configurable via `--symbols`)

### Signal Generation
On every bar completion, Bolo evaluates the relation between Fast and Slow EMAs:
1. **Bullish Crossover (`BUY`):**
   $$\text{EMA}_{9}[t-1] \le \text{EMA}_{21}[t-1] \quad \text{and} \quad \text{EMA}_{9}[t] > \text{EMA}_{21}[t]$$
   - Triggers position entry if currently flat.
   - Computes maximum allowable position size under the 5% equity rule.
2. **Bearish Crossover (`SELL`):**
   $$\text{EMA}_{9}[t-1] \ge \text{EMA}_{21}[t-1] \quad \text{and} \quad \text{EMA}_{9}[t] < \text{EMA}_{21}[t]$$
   - Triggers complete exit to cash if currently long.
3. **Uptrend / Downtrend Continuation (`HOLD`):**
   - If already in position during an uptrend, holds.
   - If flat during a downtrend, stays in cash.

---

## 2. WARBOT Rulebook Enforcement

Bolo-v1 strictly enforces all four WARBOT tournament rules inside its `RiskManager` and order validation pipeline:

| Tournament Rule | Limit / Constraint | Implementation Mechanism |
|---|---|---|
| **Order Tagging** | Mandatory `[WARBOT]` tag | Every `client_order_id` is stamped `[WARBOT]:BOLO:{symbol}:{side}:{timestamp}`. Orders missing the tag are rejected before network transmission. |
| **Position Sizing** | Max **5.0%** equity per symbol | `calculate_position_size()` caps target exposure at `portfolio_equity * 0.05`. If existing holdings meet or exceed the cap, 0 additional shares are allowed. |
| **Circuit Breaker** | Max **15.0%** intraday drawdown | Tracks high-water peak equity. If $(Peak - Current) / Peak \ge 0.15$, trips the circuit breaker, permanently halts trading for the session, and cancels all open orders. |
| **Zero Overnight Leverage** | Gross exposure $\le 1.0\times$ equity | Checks gross market value of positions against equity. Blocks new orders if exposure exceeds 100% of equity. |

### Scoring Formula
WARBOT scoring penalizes cowardice and reward risk-adjusted returns:
$$\text{Score} = \begin{cases} \text{Return}, & \text{if } \text{Return} \le 0 \\ \frac{\text{Return}}{\max(\text{Max Drawdown}, 5\%)}, & \text{if } \text{Return} > 0 \end{cases}$$
The 5% floor prevents zero-risk bots with tiny returns from dividing by zero to game the leaderboard.

---

## 3. Architecture & Zero-Dependency Design

- **Pure Python Standard Library:** Uses standard `urllib.request`, `json`, `dataclasses`, and `math`. Requires **zero `pip` dependencies** and works out of the box on Python 3.8 through 3.14+.
- **Injectable HTTP Transport:** Decouples network communication from strategy logic, enabling 100% deterministic test coverage with mock payloads.
- **Fail-Safe Fallbacks:** When executed with `--dry-run` without API credentials, Bolo automatically generates realistic synthetic market bars to verify order sizing and signal generation.

---

## 4. Usage & Command Reference

### Backtest Mode
Run historical simulation on 500 synthetic bars (or provide `--bars-file` with custom CSV OHLCV data):
```bash
python3 bolo_v1.py --backtest
```
Sample output:
```text
=======================================================================
               WARBOT BENCHMARK GLADIATOR: BOLO-V1 REPORT              
=======================================================================
 Strategy                 : Dual EMA Crossover (9 / 21)
 Initial Equity           : $100,000.00
 Final Equity             : $100,636.24
 Net Return               : +0.64%
 Peak Drawdown            : 0.04%
 Drawdown Floor (Rule 7)  : 5.00%
 WARBOT Score             : 0.1272 [Return / max(DD, 5%)]
-----------------------------------------------------------------------
 Total Trades Completed   : 4
 Winning Trades           : 4
 Losing Trades            : 0
 Win Rate                 : 100.0%
 Circuit Breaker Tripped  : NO (CLEAR)
=======================================================================
```

### Dry-Run Mode
Inspect live paper signals and simulated order execution without sending orders:
```bash
# Single inspection cycle
python3 bolo_v1.py --dry-run --once

# Continuous 60-second polling dry-run
python3 bolo_v1.py --dry-run --symbols SPY,QQQ,AAPL --interval 60
```

### Paper Trading Loop
Execute live paper trades on Alpaca:
```bash
export APCA_API_KEY_ID="your_paper_key_id"
export APCA_API_SECRET_KEY="your_paper_secret_key"

python3 bolo_v1.py --symbols SPY,QQQ --timeframe 5Min --interval 60
```

### CLI Flag Reference
- `--dry-run`: Simulate order execution and risk checks without network trades.
- `--backtest`: Execute strategy backtesting simulation.
- `--bars-file <path>`: Path to custom CSV file (`timestamp,open,high,low,close,volume`).
- `--symbols <list>`: Comma-separated list of symbols (default: `SPY,QQQ`).
- `--fast-ema <int>`: Fast EMA period (default: `9`).
- `--slow-ema <int>`: Slow EMA period (default: `21`).
- `--timeframe <str>`: Bar timeframe (default: `5Min`).
- `--max-pos-pct <float>`: Maximum position size as fraction of equity (default: `0.05`).
- `--max-drawdown <float>`: Circuit breaker halt threshold (default: `0.15`).
- `--once`: Run a single cycle and exit.
- `--interval <int>`: Polling loop interval in seconds (default: `60`).
- `--live`: Target Alpaca Live API instead of Paper API.

---

## 5. Verification & Testing

The test suite in `test_bolo_v1.py` validates all core functionality using unit tests:
```bash
python3 -m unittest -v test_bolo_v1.py
```
Covers:
- Exponential Moving Average mathematical precision and edge cases.
- Bullish and bearish crossover signal transitions.
- Strict 5% position sizing cap and existing holding deductions.
- 15% drawdown circuit breaker trip, order cancellation, and persistent lock.
- Zero overnight leverage compliance.
- Client order ID formatting and `[WARBOT]` tag enforcement.
- Alpaca REST client HTTP mock handling.
- Backtest engine metrics and WARBOT score calculation.
