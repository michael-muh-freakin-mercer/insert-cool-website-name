#!/usr/bin/env python3
"""
Bolo-v1: The Baseline House Bot & Moving-Average Benchmark Gladiator.
Official reference trading bot for WARBOT (2027 Autonomous Capital Combat).

Strategy:
  Dual Exponential Moving Average (EMA) Crossover (default: 9-EMA / 21-EMA on 5-minute bars).
  - Fast EMA crossing above Slow EMA -> BULLISH SIGNAL (Enter / Maintain Long).
  - Fast EMA crossing below Slow EMA -> BEARISH SIGNAL (Exit Long to Cash).

WARBOT Tournament Rules Strictly Enforced:
  1. Order Tagging: Every submitted order MUST carry `[WARBOT]` in client_order_id.
  2. Position Sizing: Maximum 5% of portfolio equity per symbol.
  3. Max Drawdown Circuit Breaker: Immediate halt if equity drops 15% from intraday high.
  4. Zero Overnight Leverage: Gross exposure capped at 1.0x equity; no margin borrowing.

Dependencies:
  Pure Python standard library (urllib, json, math, datetime, dataclasses).
  Zero third-party package dependencies required.

Usage:
  python3 bolo_v1.py --backtest                     # Run historical backtest simulation
  python3 bolo_v1.py --dry-run                      # Live paper evaluation without sending orders
  python3 bolo_v1.py --dry-run --once               # Single-cycle dry-run inspection
  python3 bolo_v1.py --paper                        # Live paper trading loop (requires APCA keys)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants & Tournament Rules
# ---------------------------------------------------------------------------

DEFAULT_FAST_EMA: int = 9
DEFAULT_SLOW_EMA: int = 21
DEFAULT_TIMEFRAME: str = "5Min"
DEFAULT_SYMBOLS: List[str] = ["SPY", "QQQ"]

# WARBOT Core Rules
MAX_POSITION_PCT: float = 0.05    # Rule 3b: 5% max equity per symbol
MAX_DRAWDOWN_HALT_PCT: float = 0.15  # Rule 3c: 15% max intraday drawdown halt
WARBOT_ORDER_TAG: str = "[WARBOT]"  # Rule 3a: Mandatory order tag
DD_COWARDICE_FLOOR: float = 0.05  # 5% floor for scoring formula

ALPACA_PAPER_URL: str = "https://paper-api.alpaca.markets"
ALPACA_LIVE_URL: str = "https://api.alpaca.markets"
ALPACA_DATA_URL: str = "https://data.alpaca.markets"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
logger = logging.getLogger("BOLO-V1")


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class Bar:
    """Represents a single OHLCV candlestick bar."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class AccountSnapshot:
    """Portfolio equity and status snapshot."""
    equity: float
    cash: float
    buying_power: float
    currency: str = "USD"
    status: str = "ACTIVE"


@dataclass
class Position:
    """Current asset holding."""
    symbol: str
    qty: float
    market_value: float
    current_price: float
    unrealized_pl: float


@dataclass
class OrderRequest:
    """Outbound order request."""
    symbol: str
    qty: int
    side: str  # "buy" | "sell"
    type: str = "market"
    time_in_force: str = "day"
    client_order_id: str = ""

    def validate_warbot_rules(self) -> None:
        """Validate order complies with WARBOT rulebook."""
        if WARBOT_ORDER_TAG not in self.client_order_id:
            raise ValueError(
                f"Order rejected: client_order_id '{self.client_order_id}' "
                f"missing mandatory '{WARBOT_ORDER_TAG}' tag"
            )
        if self.qty <= 0:
            raise ValueError(f"Order rejected: quantity must be positive, got {self.qty}")
        if self.side not in ("buy", "sell"):
            raise ValueError(f"Order rejected: side must be 'buy' or 'sell', got {self.side}")


@dataclass
class SignalResult:
    """Strategy evaluation outcome."""
    symbol: str
    timestamp: datetime
    close_price: float
    fast_ema: float
    slow_ema: float
    action: str  # "BUY", "SELL", "HOLD"
    reason: str


# ---------------------------------------------------------------------------
# Indicator Calculations
# ---------------------------------------------------------------------------

def calculate_ema_series(prices: List[float], period: int) -> List[float]:
    """
    Compute Exponential Moving Average (EMA) series for given prices.

    Uses standard multiplier: alpha = 2.0 / (period + 1).
    Initial seed is simple average of the first `period` prices.
    """
    if len(prices) < period or period <= 0:
        return [float("nan")] * len(prices)

    ema_series: List[float] = [float("nan")] * (period - 1)
    sma_seed = sum(prices[:period]) / period
    ema_series.append(sma_seed)

    multiplier = 2.0 / (period + 1.0)
    current_ema = sma_seed

    for price in prices[period:]:
        current_ema = (price - current_ema) * multiplier + current_ema
        ema_series.append(current_ema)

    return ema_series


def calculate_ema(prices: List[float], period: int) -> float:
    """Calculate the latest EMA value from a price list."""
    series = calculate_ema_series(prices, period)
    return series[-1] if series else float("nan")


# ---------------------------------------------------------------------------
# Risk Manager (WARBOT Rules Enforcer)
# ---------------------------------------------------------------------------

class RiskManager:
    """
    Enforces all WARBOT league rules:
      - 5% maximum position size per symbol.
      - 15% maximum drawdown circuit breaker (halts trading).
      - Zero overnight leverage.
      - Mandatory order tagging with [WARBOT].
    """

    def __init__(
        self,
        max_position_pct: float = MAX_POSITION_PCT,
        max_drawdown_pct: float = MAX_DRAWDOWN_HALT_PCT,
        order_tag: str = WARBOT_ORDER_TAG,
    ) -> None:
        self.max_position_pct = max_position_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.order_tag = order_tag
        self.peak_equity: float = 0.0
        self.circuit_breaker_tripped: bool = False
        self.halt_reason: Optional[str] = None

    def update_equity(self, current_equity: float) -> Tuple[bool, float]:
        """
        Record current equity and evaluate the 15% drawdown circuit breaker.
        Returns: (is_halted, current_drawdown_fraction).
        """
        if current_equity <= 0:
            return self.circuit_breaker_tripped, 1.0

        if self.peak_equity <= 0:
            self.peak_equity = current_equity

        if current_equity > self.peak_equity:
            self.peak_equity = current_equity

        drawdown = (self.peak_equity - current_equity) / self.peak_equity

        if drawdown >= self.max_drawdown_pct and not self.circuit_breaker_tripped:
            self.circuit_breaker_tripped = True
            self.halt_reason = (
                f"CIRCUIT BREAKER TRIPPED: Drawdown {drawdown * 100:.2f}% "
                f"exceeds WARBOT limit of {self.max_drawdown_pct * 100:.1f}%. "
                f"Peak Equity: ${self.peak_equity:,.2f}, Current: ${current_equity:,.2f}."
            )
            logger.critical(self.halt_reason)

        return self.circuit_breaker_tripped, drawdown

    def calculate_position_size(
        self,
        symbol: str,
        current_price: float,
        portfolio_equity: float,
        existing_position_qty: float = 0.0,
    ) -> int:
        """
        Calculate allowed buy quantity strictly capped at 5% of portfolio equity.
        Returns integer shares.
        """
        if self.circuit_breaker_tripped:
            logger.warning("RiskManager: Trading halted due to circuit breaker. Zero shares allowed.")
            return 0

        if current_price <= 0 or portfolio_equity <= 0:
            return 0

        max_allowed_dollars = portfolio_equity * self.max_position_pct
        current_holding_dollars = existing_position_qty * current_price
        available_budget_dollars = max(0.0, max_allowed_dollars - current_holding_dollars)

        if available_budget_dollars < current_price:
            logger.info(
                f"RiskManager: Symbol {symbol} position (${current_holding_dollars:.2f}) "
                f"already near or at 5% cap (${max_allowed_dollars:.2f}). No shares added."
            )
            return 0

        shares = int(available_budget_dollars // current_price)
        return shares

    def verify_zero_overnight_leverage(
        self,
        total_market_value: float,
        portfolio_equity: float,
    ) -> bool:
        """
        Ensure gross exposure <= 1.0x portfolio equity (zero overnight leverage).
        """
        if portfolio_equity <= 0:
            return False
        leverage = total_market_value / portfolio_equity
        return leverage <= 1.0001  # Allow minimal floating point margin

    def generate_client_order_id(self, symbol: str, side: str) -> str:
        """Generate a compliant client_order_id prefixed with [WARBOT]."""
        timestamp = int(time.time() * 1000)
        # Limit length to 128 characters per Alpaca requirements
        return f"{self.order_tag}:BOLO:{symbol}:{side.upper()}:{timestamp}"


# ---------------------------------------------------------------------------
# Alpaca Raw REST Client (Stdlib Only)
# ---------------------------------------------------------------------------

HttpTransport = Callable[[str, str, Dict[str, str], Optional[bytes]], Any]


def default_http_transport(
    method: str,
    url: str,
    headers: Dict[str, str],
    body: Optional[bytes] = None,
) -> Any:
    """Standard library HTTP transport using urllib.request."""
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
            if not data:
                return None
            return json.loads(data.decode("utf-8"))
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode("utf-8", errors="replace")
        logger.error(f"HTTP {exc.code} for {method} {url}: {err_body}")
        raise RuntimeError(f"Alpaca API error {exc.code}: {err_body}") from exc


class AlpacaClient:
    """
    Pure Python standard-library Alpaca Trading & Market Data API client.
    Compatible with paper and live environments.
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        paper: bool = True,
        transport: HttpTransport = default_http_transport,
    ) -> None:
        self.api_key = api_key
        self.api_secret = api_secret
        self.paper = paper
        self.trading_base = ALPACA_PAPER_URL if paper else ALPACA_LIVE_URL
        self.data_base = ALPACA_DATA_URL
        self.transport = transport

    @property
    def headers(self) -> Dict[str, str]:
        return {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
            "Content-Type": "application/json",
            "User-Agent": "WARBOT-Bolo-v1/1.0",
        }

    def get_account(self) -> AccountSnapshot:
        """Fetch account snapshot."""
        data = self.transport(
            "GET",
            f"{self.trading_base}/v2/account",
            self.headers,
            None,
        )
        return AccountSnapshot(
            equity=float(data.get("equity", 0.0)),
            cash=float(data.get("cash", 0.0)),
            buying_power=float(data.get("buying_power", 0.0)),
            currency=data.get("currency", "USD"),
            status=data.get("status", "UNKNOWN"),
        )

    def get_positions(self) -> List[Position]:
        """Fetch all current open positions."""
        data = self.transport(
            "GET",
            f"{self.trading_base}/v2/positions",
            self.headers,
            None,
        ) or []
        positions: List[Position] = []
        for pos in data:
            positions.append(
                Position(
                    symbol=pos["symbol"],
                    qty=float(pos["qty"]),
                    market_value=float(pos["market_value"]),
                    current_price=float(pos["current_price"]),
                    unrealized_pl=float(pos.get("unrealized_pl", 0.0)),
                )
            )
        return positions

    def get_bars(
        self,
        symbol: str,
        timeframe: str = "5Min",
        limit: int = 100,
    ) -> List[Bar]:
        """
        Fetch historical bars for a symbol from Alpaca Market Data v2.
        """
        query = urllib.parse.urlencode({
            "timeframe": timeframe,
            "limit": limit,
            "feed": "iex",
            "adjustment": "raw",
        })
        url = f"{self.data_base}/v2/stocks/{symbol}/bars?{query}"
        data = self.transport("GET", url, self.headers, None) or {}
        raw_bars = data.get("bars") or []

        bars: List[Bar] = []
        for b in raw_bars:
            # Parse ISO timestamp
            ts_str = b["t"]
            if ts_str.endswith("Z"):
                ts_str = ts_str[:-1] + "+00:00"
            ts = datetime.fromisoformat(ts_str)
            bars.append(
                Bar(
                    timestamp=ts,
                    open=float(b["o"]),
                    high=float(b["h"]),
                    low=float(b["l"]),
                    close=float(b["c"]),
                    volume=float(b["v"]),
                )
            )
        return bars

    def submit_order(self, order: OrderRequest) -> Dict[str, Any]:
        """Submit a new order. Enforces WARBOT rules before network transmission."""
        order.validate_warbot_rules()
        payload = json.dumps({
            "symbol": order.symbol,
            "qty": str(order.qty),
            "side": order.side,
            "type": order.type,
            "time_in_force": order.time_in_force,
            "client_order_id": order.client_order_id,
        }).encode("utf-8")

        return self.transport(
            "POST",
            f"{self.trading_base}/v2/orders",
            self.headers,
            payload,
        )

    def cancel_all_orders(self) -> Any:
        """Cancel all open orders (used during circuit breaker trip)."""
        return self.transport(
            "DELETE",
            f"{self.trading_base}/v2/orders",
            self.headers,
            None,
        )


# ---------------------------------------------------------------------------
# Dual Moving Average Strategy Engine
# ---------------------------------------------------------------------------

class DualEMACrossoverStrategy:
    """
    Dual Exponential Moving Average (EMA) Crossover Strategy.
    """

    def __init__(
        self,
        fast_period: int = DEFAULT_FAST_EMA,
        slow_period: int = DEFAULT_SLOW_EMA,
    ) -> None:
        if fast_period >= slow_period:
            raise ValueError(f"Fast EMA ({fast_period}) must be strictly less than Slow EMA ({slow_period})")
        self.fast_period = fast_period
        self.slow_period = slow_period

    def evaluate(self, symbol: str, bars: List[Bar]) -> SignalResult:
        """
        Evaluate bars and detect crossover signals.
        Needs at least slow_period + 2 bars for stable signal verification.
        """
        if len(bars) < self.slow_period + 2:
            return SignalResult(
                symbol=symbol,
                timestamp=bars[-1].timestamp if bars else datetime.now(timezone.utc),
                close_price=bars[-1].close if bars else 0.0,
                fast_ema=float("nan"),
                slow_ema=float("nan"),
                action="HOLD",
                reason=f"Insufficient bars ({len(bars)}) for slow period {self.slow_period}",
            )

        closes = [b.close for b in bars]
        fast_emas = calculate_ema_series(closes, self.fast_period)
        slow_emas = calculate_ema_series(closes, self.slow_period)

        current_fast = fast_emas[-1]
        current_slow = slow_emas[-1]
        prev_fast = fast_emas[-2]
        prev_slow = slow_emas[-2]
        latest_bar = bars[-1]

        # Bullish Crossover: Fast crosses above Slow
        if prev_fast <= prev_slow and current_fast > current_slow:
            action = "BUY"
            reason = (
                f"Bullish crossover: Fast EMA ({current_fast:.2f}) crossed above "
                f"Slow EMA ({current_slow:.2f})"
            )
        # Bearish Crossover: Fast crosses below Slow
        elif prev_fast >= prev_slow and current_fast < current_slow:
            action = "SELL"
            reason = (
                f"Bearish crossover: Fast EMA ({current_fast:.2f}) crossed below "
                f"Slow EMA ({current_slow:.2f})"
            )
        elif current_fast > current_slow:
            action = "HOLD"
            reason = (
                f"Fast EMA ({current_fast:.2f}) > Slow EMA ({current_slow:.2f}) [Uptrend intact]"
            )
        else:
            action = "HOLD"
            reason = (
                f"Fast EMA ({current_fast:.2f}) <= Slow EMA ({current_slow:.2f}) [Downtrend / Flat]"
            )

        return SignalResult(
            symbol=symbol,
            timestamp=latest_bar.timestamp,
            close_price=latest_bar.close,
            fast_ema=current_fast,
            slow_ema=current_slow,
            action=action,
            reason=reason,
        )


# ---------------------------------------------------------------------------
# Bolo Bot Arena Gladiator Runner
# ---------------------------------------------------------------------------

class BoloBot:
    """
    Main orchestration class for Bolo-v1:
      - Interacts with Alpaca paper/live account.
      - Maintains symbol state.
      - Executes Dual EMA Crossover strategy.
      - Enforces WARBOT rules via RiskManager.
      - Supports live paper trading and dry-run simulation.
    """

    def __init__(
        self,
        client: Optional[AlpacaClient] = None,
        risk_manager: Optional[RiskManager] = None,
        strategy: Optional[DualEMACrossoverStrategy] = None,
        symbols: Optional[List[str]] = None,
        timeframe: str = DEFAULT_TIMEFRAME,
        dry_run: bool = False,
    ) -> None:
        self.client = client
        self.risk_manager = risk_manager or RiskManager()
        self.strategy = strategy or DualEMACrossoverStrategy()
        self.symbols = symbols or DEFAULT_SYMBOLS
        self.timeframe = timeframe
        self.dry_run = dry_run

    def run_cycle(self) -> List[SignalResult]:
        """Execute one complete evaluation and trading cycle across universe."""
        results: List[SignalResult] = []

        if not self.dry_run and self.client is None:
            raise ValueError("Alpaca client must be provided when not running in dry-run mode")

        # 1. Fetch account & evaluate circuit breaker
        if self.client:
            account = self.client.get_account()
            positions = {p.symbol: p for p in self.client.get_positions()}
        else:
            # Mock initial state for pure dry run when no client credentials provided
            account = AccountSnapshot(equity=100000.0, cash=100000.0, buying_power=100000.0)
            positions = {}

        is_halted, dd = self.risk_manager.update_equity(account.equity)
        logger.info(
            f"Account: Equity=${account.equity:,.2f} | Peak=${self.risk_manager.peak_equity:,.2f} | "
            f"DD={dd * 100:.2f}% | Halted={is_halted}"
        )

        if is_halted:
            logger.critical(
                f"WARBOT CIRCUIT BREAKER ACTIVE: Trading halted! ({self.risk_manager.halt_reason})"
            )
            if self.client and not self.dry_run:
                logger.warning("Canceling all open orders due to circuit breaker trip...")
                self.client.cancel_all_orders()
            return results

        # 2. Check overnight leverage rule
        total_market_value = sum(p.market_value for p in positions.values())
        if not self.risk_manager.verify_zero_overnight_leverage(total_market_value, account.equity):
            logger.warning(
                f"LEVERAGE VIOLATION: Gross exposure ${total_market_value:,.2f} exceeds "
                f"equity ${account.equity:,.2f}. Halting new entries."
            )
            return results

        # 3. Evaluate each symbol in universe
        for symbol in self.symbols:
            try:
                if self.client:
                    bars = self.client.get_bars(symbol, timeframe=self.timeframe, limit=100)
                else:
                    seed = sum(ord(c) for c in symbol)
                    start_price = 500.0 if symbol == "SPY" else (420.0 if symbol == "QQQ" else 200.0)
                    bars = generate_synthetic_bars(symbol, num_bars=100, start_price=start_price, seed=seed)

                if not bars:
                    logger.warning(f"No bars returned for {symbol}, skipping.")
                    continue

                sig = self.strategy.evaluate(symbol, bars)
                results.append(sig)
                logger.info(
                    f"[{symbol}] Price=${sig.close_price:.2f} | "
                    f"9-EMA={sig.fast_ema:.2f} | 21-EMA={sig.slow_ema:.2f} | "
                    f"Signal={sig.action} ({sig.reason})"
                )

                pos = positions.get(symbol)
                current_qty = pos.qty if pos else 0.0

                if sig.action == "BUY":
                    if current_qty > 0:
                        logger.info(f"[{symbol}] Already holding {current_qty} shares. No add.")
                        continue

                    shares_to_buy = self.risk_manager.calculate_position_size(
                        symbol=symbol,
                        current_price=sig.close_price,
                        portfolio_equity=account.equity,
                        existing_position_qty=current_qty,
                    )

                    if shares_to_buy <= 0:
                        logger.info(f"[{symbol}] Allowed shares to buy is 0 under 5% position rule.")
                        continue

                    order_id = self.risk_manager.generate_client_order_id(symbol, "BUY")
                    order = OrderRequest(
                        symbol=symbol,
                        qty=shares_to_buy,
                        side="buy",
                        type="market",
                        time_in_force="day",
                        client_order_id=order_id,
                    )

                    if self.dry_run:
                        logger.info(
                            f"[DRY-RUN] Would submit: BUY {shares_to_buy} {symbol} "
                            f"@ ~${sig.close_price:.2f} (Est: ${shares_to_buy * sig.close_price:,.2f}, "
                            f"{(shares_to_buy * sig.close_price / account.equity) * 100:.2f}% of equity) "
                            f"Tag='{order_id}'"
                        )
                    elif self.client:
                        logger.info(f"Submitting LIVE/PAPER Order: {order}")
                        resp = self.client.submit_order(order)
                        logger.info(f"Order submitted successfully: ID={resp.get('id')}")

                elif sig.action == "SELL":
                    if current_qty <= 0:
                        logger.info(f"[{symbol}] Bearish signal, but no open position to exit.")
                        continue

                    shares_to_sell = int(current_qty)
                    order_id = self.risk_manager.generate_client_order_id(symbol, "SELL")
                    order = OrderRequest(
                        symbol=symbol,
                        qty=shares_to_sell,
                        side="sell",
                        type="market",
                        time_in_force="day",
                        client_order_id=order_id,
                    )

                    if self.dry_run:
                        logger.info(
                            f"[DRY-RUN] Would submit: SELL {shares_to_sell} {symbol} "
                            f"to close position. Tag='{order_id}'"
                        )
                    elif self.client:
                        logger.info(f"Submitting LIVE/PAPER Order to EXIT: {order}")
                        resp = self.client.submit_order(order)
                        logger.info(f"Exit order submitted successfully: ID={resp.get('id')}")

            except Exception as exc:
                logger.error(f"Error processing {symbol}: {exc}", exc_info=True)

        return results


# ---------------------------------------------------------------------------
# Backtesting & Simulation Engine
# ---------------------------------------------------------------------------

@dataclass
class BacktestMetrics:
    """WARBOT standard tournament performance metrics."""
    initial_equity: float
    final_equity: float
    net_return_pct: float
    max_drawdown_pct: float
    warbot_score: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate_pct: float
    circuit_breaker_tripped: bool


def generate_synthetic_bars(
    symbol: str,
    num_bars: int = 500,
    start_price: float = 400.0,
    seed: int = 42,
) -> List[Bar]:
    """Generate realistic OHLCV bars with cyclical waves for deterministic testing."""
    bars: List[Bar] = []
    base_time = datetime(2027, 3, 8, 9, 30, tzinfo=timezone.utc)
    price = start_price

    # Realistic basis points per 5-minute bar (10-30 bps) to prevent runaway exponential explosion
    for i in range(num_bars):
        trend = 0.0012 * math.sin(i / 18.0) + 0.0006 * math.cos(i / 40.0)
        noise = 0.0008 * math.sin(i * 1.618 + seed)
        pct_change = trend + noise

        open_p = price
        close_p = max(1.0, open_p * (1.0 + pct_change))
        high_p = max(open_p, close_p) * 1.0015
        low_p = min(open_p, close_p) * 0.9985
        vol = 25000.0 + 10000.0 * abs(pct_change * 1000)

        bars.append(
            Bar(
                timestamp=base_time + timedelta(minutes=5 * i),
                open=round(open_p, 2),
                high=round(high_p, 2),
                low=round(low_p, 2),
                close=round(close_p, 2),
                volume=round(vol, 0),
            )
        )
        price = close_p

    return bars


def run_backtest(
    bars: List[Bar],
    initial_equity: float = 100000.0,
    fast_period: int = DEFAULT_FAST_EMA,
    slow_period: int = DEFAULT_SLOW_EMA,
    max_position_pct: float = MAX_POSITION_PCT,
    max_drawdown_pct: float = MAX_DRAWDOWN_HALT_PCT,
) -> Tuple[BacktestMetrics, List[Dict[str, Any]]]:
    """
    Run backtest simulation enforcing WARBOT rules:
      - 5% maximum position size per symbol
      - 15% intraday drawdown circuit breaker
      - Official WARBOT Score calculation: Return / max(MDD, 5%)
    """
    strategy = DualEMACrossoverStrategy(fast_period, slow_period)
    risk_manager = RiskManager(max_position_pct, max_drawdown_pct)

    equity = initial_equity
    cash = initial_equity
    position_shares = 0
    trades: List[Dict[str, Any]] = []
    entry_price = 0.0
    equity_history: List[float] = [equity]

    winning_trades = 0
    losing_trades = 0

    for idx in range(slow_period + 2, len(bars)):
        sub_bars = bars[:idx]
        current_bar = bars[idx - 1]
        price = current_bar.close

        # Current portfolio equity
        portfolio_equity = cash + (position_shares * price)
        equity_history.append(portfolio_equity)

        # Check circuit breaker
        is_halted, _ = risk_manager.update_equity(portfolio_equity)
        if is_halted:
            # Force liquidate on circuit breaker halt
            if position_shares > 0:
                pnl = position_shares * (price - entry_price)
                cash += position_shares * price
                trades.append({
                    "time": current_bar.timestamp.isoformat(),
                    "type": "CIRCUIT_BREAKER_HALT_LIQUIDATE",
                    "price": price,
                    "shares": position_shares,
                    "pnl": pnl,
                })
                position_shares = 0
                if pnl > 0:
                    winning_trades += 1
                else:
                    losing_trades += 1
            break

        signal = strategy.evaluate("BACKTEST", sub_bars)

        # Execute signals
        if signal.action == "BUY" and position_shares == 0:
            shares = risk_manager.calculate_position_size(
                "BACKTEST",
                price,
                portfolio_equity,
                existing_position_qty=0,
            )
            if shares > 0 and cash >= shares * price:
                cost = shares * price
                cash -= cost
                position_shares = shares
                entry_price = price
                trades.append({
                    "time": current_bar.timestamp.isoformat(),
                    "type": "BUY",
                    "price": price,
                    "shares": shares,
                    "cost": cost,
                    "order_tag": risk_manager.generate_client_order_id("BACKTEST", "BUY"),
                })

        elif signal.action == "SELL" and position_shares > 0:
            proceeds = position_shares * price
            pnl = proceeds - (position_shares * entry_price)
            cash += proceeds
            trades.append({
                "time": current_bar.timestamp.isoformat(),
                "type": "SELL",
                "price": price,
                "shares": position_shares,
                "pnl": pnl,
                "order_tag": risk_manager.generate_client_order_id("BACKTEST", "SELL"),
            })
            position_shares = 0
            if pnl > 0:
                winning_trades += 1
            else:
                losing_trades += 1

    # Close any open position at the end of the simulation
    if position_shares > 0:
        last_price = bars[-1].close
        pnl = position_shares * (last_price - entry_price)
        cash += position_shares * last_price
        trades.append({
            "time": bars[-1].timestamp.isoformat(),
            "type": "CLOSE_FINAL",
            "price": last_price,
            "shares": position_shares,
            "pnl": pnl,
        })
        position_shares = 0
        if pnl > 0:
            winning_trades += 1
        else:
            losing_trades += 1

    final_equity = cash
    net_return_pct = ((final_equity - initial_equity) / initial_equity) * 100.0

    # Max Drawdown calculation
    peak = equity_history[0]
    mdd = 0.0
    for eq in equity_history:
        if eq > peak:
            peak = eq
        elif peak > 0:
            dd = (peak - eq) / peak
            if dd > mdd:
                mdd = dd

    max_dd_pct = mdd * 100.0

    # WARBOT Scoring Formula:
    # Score = Return / max(Drawdown, 5%)
    # If Return <= 0, Score = Return
    ret_fraction = (final_equity - initial_equity) / initial_equity
    if ret_fraction <= 0:
        warbot_score = ret_fraction
    else:
        warbot_score = ret_fraction / max(mdd, DD_COWARDICE_FLOOR)

    total_trades = winning_trades + losing_trades
    win_rate = (winning_trades / total_trades * 100.0) if total_trades > 0 else 0.0

    metrics = BacktestMetrics(
        initial_equity=initial_equity,
        final_equity=final_equity,
        net_return_pct=net_return_pct,
        max_drawdown_pct=max_dd_pct,
        warbot_score=warbot_score,
        total_trades=total_trades,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        win_rate_pct=win_rate,
        circuit_breaker_tripped=risk_manager.circuit_breaker_tripped,
    )

    return metrics, trades


def print_backtest_report(metrics: BacktestMetrics, trades: List[Dict[str, Any]]) -> None:
    """Print standard ASCII report for WARBOT backtest."""
    card = [
        "=======================================================================",
        "               WARBOT BENCHMARK GLADIATOR: BOLO-V1 REPORT              ",
        "=======================================================================",
        f" Strategy                 : Dual EMA Crossover ({DEFAULT_FAST_EMA} / {DEFAULT_SLOW_EMA})",
        f" Initial Equity           : ${metrics.initial_equity:,.2f}",
        f" Final Equity             : ${metrics.final_equity:,.2f}",
        f" Net Return               : {metrics.net_return_pct:+.2f}%",
        f" Peak Drawdown            : {metrics.max_drawdown_pct:.2f}%",
        f" Drawdown Floor (Rule 7)  : 5.00%",
        f" WARBOT Score             : {metrics.warbot_score:.4f} [Return / max(DD, 5%)]",
        "-----------------------------------------------------------------------",
        f" Total Trades Completed   : {metrics.total_trades}",
        f" Winning Trades           : {metrics.winning_trades}",
        f" Losing Trades            : {metrics.losing_trades}",
        f" Win Rate                 : {metrics.win_rate_pct:.1f}%",
        f" Circuit Breaker Tripped  : {'YES (HALTED)' if metrics.circuit_breaker_tripped else 'NO (CLEAR)'}",
        "=======================================================================",
    ]
    print("\n" + "\n".join(card) + "\n")


# ---------------------------------------------------------------------------
# CLI Argument Parsing & Entry Point
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bolo-v1: WARBOT Dual Moving Average Benchmark House Bot",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate execution and risk rules without submitting live orders",
    )
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Run strategy backtester on synthetic or CSV bars",
    )
    parser.add_argument(
        "--bars-file",
        type=str,
        default=None,
        help="Path to CSV file with OHLCV bars for backtesting",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=",".join(DEFAULT_SYMBOLS),
        help="Comma-separated ticker symbols to trade",
    )
    parser.add_argument(
        "--fast-ema",
        type=int,
        default=DEFAULT_FAST_EMA,
        help="Fast Exponential Moving Average period",
    )
    parser.add_argument(
        "--slow-ema",
        type=int,
        default=DEFAULT_SLOW_EMA,
        help="Slow Exponential Moving Average period",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default=DEFAULT_TIMEFRAME,
        help="Bar timeframe (e.g. 1Min, 5Min, 15Min, 1Hour)",
    )
    parser.add_argument(
        "--max-pos-pct",
        type=float,
        default=MAX_POSITION_PCT,
        help="Maximum position size as fraction of portfolio equity (0.05 = 5%%)",
    )
    parser.add_argument(
        "--max-drawdown",
        type=float,
        default=MAX_DRAWDOWN_HALT_PCT,
        help="Maximum drawdown before circuit breaker halt (0.15 = 15%%)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single evaluation cycle and exit immediately",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Polling interval in seconds between cycles in continuous loop",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use Alpaca Live API instead of Paper Trading API (CAUTION)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("APCA_API_KEY_ID", ""),
        help="Alpaca API Key ID (or via APCA_API_KEY_ID env var)",
    )
    parser.add_argument(
        "--api-secret",
        type=str,
        default=os.getenv("APCA_API_SECRET_KEY", ""),
        help="Alpaca API Secret Key (or via APCA_API_SECRET_KEY env var)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    # Mode 1: Backtesting Mode
    if args.backtest:
        logger.info("Initializing Bolo-v1 in BACKTEST mode...")
        if args.bars_file:
            logger.info(f"Loading bars from {args.bars_file}...")
            # CSV loader: timestamp,open,high,low,close,volume
            bars: List[Bar] = []
            with open(args.bars_file, "r", encoding="utf-8") as f:
                header = f.readline()
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) >= 6:
                        bars.append(
                            Bar(
                                timestamp=datetime.fromisoformat(parts[0]),
                                open=float(parts[1]),
                                high=float(parts[2]),
                                low=float(parts[3]),
                                close=float(parts[4]),
                                volume=float(parts[5]),
                            )
                        )
        else:
            logger.info("Generating synthetic benchmark simulation bars (500 bars, 5-minute)...")
            bars = generate_synthetic_bars("BENCHMARK", num_bars=500, start_price=400.0)

        metrics, trades = run_backtest(
            bars=bars,
            initial_equity=100000.0,
            fast_period=args.fast_ema,
            slow_period=args.slow_ema,
            max_position_pct=args.max_pos_pct,
            max_drawdown_pct=args.max_drawdown,
        )
        print_backtest_report(metrics, trades)
        return 0

    # Mode 2: Live or Dry-Run Mode
    logger.info("Starting Bolo-v1: The Benchmark House Bot...")
    logger.info(f"Strategy: Dual EMA ({args.fast_ema} / {args.slow_ema}) on {args.timeframe} bars")
    logger.info(f"Universe: {symbols}")
    logger.info(f"Rules: Max Position={args.max_pos_pct * 100:.1f}%, Halt Drawdown={args.max_drawdown * 100:.1f}%")

    client: Optional[AlpacaClient] = None
    if args.api_key and args.api_secret:
        client = AlpacaClient(
            api_key=args.api_key,
            api_secret=args.api_secret,
            paper=not args.live,
        )
    elif not args.dry_run:
        logger.error(
            "Alpaca API credentials missing! Provide --api-key and --api-secret "
            "or set APCA_API_KEY_ID and APCA_API_SECRET_KEY, or run with --dry-run."
        )
        return 1

    risk_manager = RiskManager(
        max_position_pct=args.max_pos_pct,
        max_drawdown_pct=args.max_drawdown,
        order_tag=WARBOT_ORDER_TAG,
    )

    strategy = DualEMACrossoverStrategy(
        fast_period=args.fast_ema,
        slow_period=args.slow_ema,
    )

    bot = BoloBot(
        client=client,
        risk_manager=risk_manager,
        strategy=strategy,
        symbols=symbols,
        timeframe=args.timeframe,
        dry_run=args.dry_run,
    )

    if args.once:
        logger.info("Executing single cycle (--once)...")
        bot.run_cycle()
        return 0

    logger.info(f"Entering continuous cycle loop (interval={args.interval}s, press Ctrl+C to stop)...")
    try:
        while True:
            bot.run_cycle()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        logger.info("Bolo-v1 stopped by user.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
