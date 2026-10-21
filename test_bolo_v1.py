#!/usr/bin/env python3
"""
Unit tests for bolo_v1.py (WARBOT Baseline House Bot).
Tests all tournament rules, mathematical indicators, risk management limits,
and execution modes using fakes and mocks (zero external network requests).
"""

import json
from datetime import datetime, timedelta, timezone
import unittest

from bolo_v1 import (
    AccountSnapshot,
    AlpacaClient,
    Bar,
    BoloBot,
    DualEMACrossoverStrategy,
    OrderRequest,
    Position,
    RiskManager,
    calculate_ema,
    calculate_ema_series,
    generate_synthetic_bars,
    run_backtest,
    WARBOT_ORDER_TAG,
    MAX_POSITION_PCT,
    MAX_DRAWDOWN_HALT_PCT,
    DD_COWARDICE_FLOOR,
)


class TestEMACalculations(unittest.TestCase):
    """Verify EMA indicator formulas."""

    def test_ema_series_length_and_seed(self):
        prices = [10.0, 11.0, 12.0, 13.0, 14.0]
        # Period 3: seed at index 2 is SMA of [10, 11, 12] = 11.0
        # multiplier = 2 / (3 + 1) = 0.5
        # Index 3: (13 - 11) * 0.5 + 11 = 12.0
        # Index 4: (14 - 12) * 0.5 + 12 = 13.0
        emas = calculate_ema_series(prices, period=3)
        self.assertEqual(len(emas), 5)
        self.assertTrue(emas[0] != emas[0])  # NaN
        self.assertTrue(emas[1] != emas[1])  # NaN
        self.assertAlmostEqual(emas[2], 11.0)
        self.assertAlmostEqual(emas[3], 12.0)
        self.assertAlmostEqual(emas[4], 13.0)

    def test_calculate_ema_latest(self):
        prices = [10.0, 11.0, 12.0, 13.0, 14.0]
        latest = calculate_ema(prices, period=3)
        self.assertAlmostEqual(latest, 13.0)

    def test_insufficient_prices(self):
        prices = [10.0, 11.0]
        emas = calculate_ema_series(prices, period=5)
        self.assertEqual(len(emas), 2)
        self.assertTrue(all(emas[i] != emas[i] for i in range(len(emas))))


class TestDualEMACrossoverStrategy(unittest.TestCase):
    """Verify Crossover Strategy logic and signals."""

    def setUp(self):
        self.strategy = DualEMACrossoverStrategy(fast_period=3, slow_period=5)
        self.base_time = datetime(2027, 3, 8, 9, 30, tzinfo=timezone.utc)

    def _make_bars(self, closes: list[float]) -> list[Bar]:
        bars = []
        for i, c in enumerate(closes):
            bars.append(
                Bar(
                    timestamp=self.base_time + timedelta(minutes=5 * i),
                    open=c,
                    high=c * 1.01,
                    low=c * 0.99,
                    close=c,
                    volume=1000.0,
                )
            )
        return bars

    def test_insufficient_bars_gives_hold(self):
        bars = self._make_bars([100.0, 101.0, 102.0])
        sig = self.strategy.evaluate("TEST", bars)
        self.assertEqual(sig.action, "HOLD")
        self.assertIn("Insufficient bars", sig.reason)

    def test_bullish_crossover_triggers_buy(self):
        # Establish downtrend, then sharp upward reversal crossing fast above slow
        closes = [100.0, 98.0, 96.0, 94.0, 92.0, 90.0, 95.0, 105.0]
        bars = self._make_bars(closes)
        sig = self.strategy.evaluate("TEST", bars)
        self.assertEqual(sig.action, "BUY")
        self.assertIn("Bullish crossover", sig.reason)

    def test_bearish_crossover_triggers_sell(self):
        # Establish uptrend, then sharp drop crossing fast below slow
        closes = [90.0, 92.0, 94.0, 96.0, 98.0, 100.0, 94.0, 85.0]
        bars = self._make_bars(closes)
        sig = self.strategy.evaluate("TEST", bars)
        self.assertEqual(sig.action, "SELL")
        self.assertIn("Bearish crossover", sig.reason)

    def test_invalid_ema_periods_raise_error(self):
        with self.assertRaises(ValueError):
            DualEMACrossoverStrategy(fast_period=21, slow_period=9)


class TestWARBOTRiskManager(unittest.TestCase):
    """Verify strict adherence to WARBOT Tournament Rules."""

    def setUp(self):
        self.rm = RiskManager(
            max_position_pct=0.05,
            max_drawdown_pct=0.15,
            order_tag=WARBOT_ORDER_TAG,
        )

    def test_order_tagging_format(self):
        order_id = self.rm.generate_client_order_id("SPY", "BUY")
        self.assertTrue(order_id.startswith("[WARBOT]"))
        self.assertIn("BOLO", order_id)
        self.assertIn("SPY", order_id)
        self.assertIn("BUY", order_id)
        self.assertLess(len(order_id), 128)

    def test_order_tag_validation_rejection(self):
        # Valid order
        valid_order = OrderRequest(
            symbol="SPY",
            qty=10,
            side="buy",
            client_order_id="[WARBOT]:BOLO:SPY:BUY:12345",
        )
        valid_order.validate_warbot_rules()

        # Missing [WARBOT] tag
        untagged_order = OrderRequest(
            symbol="SPY",
            qty=10,
            side="buy",
            client_order_id="UNTAGGED_ORDER_12345",
        )
        with self.assertRaises(ValueError) as ctx:
            untagged_order.validate_warbot_rules()
        self.assertIn("missing mandatory '[WARBOT]' tag", str(ctx.exception))

        # Invalid zero qty
        zero_qty_order = OrderRequest(
            symbol="SPY",
            qty=0,
            side="buy",
            client_order_id="[WARBOT]:BOLO:SPY:BUY:12345",
        )
        with self.assertRaises(ValueError):
            zero_qty_order.validate_warbot_rules()

    def test_position_sizing_rule_5_percent(self):
        # Portfolio: $100,000 equity. Max 5% = $5,000.
        equity = 100000.0
        price = 100.0
        # 5% of 100k = $5,000 -> 50 shares of $100
        shares = self.rm.calculate_position_size("SPY", price, equity, existing_position_qty=0)
        self.assertEqual(shares, 50)
        self.assertLessEqual(shares * price, equity * 0.05)

        # Existing position of 30 shares ($3,000). Remaining budget = $2,000 -> 20 shares
        shares_add = self.rm.calculate_position_size("SPY", price, equity, existing_position_qty=30)
        self.assertEqual(shares_add, 20)
        self.assertLessEqual((30 + shares_add) * price, equity * 0.05)

        # Already at or exceeding 5% cap (50 shares = $5,000). Allowed = 0
        shares_full = self.rm.calculate_position_size("SPY", price, equity, existing_position_qty=50)
        self.assertEqual(shares_full, 0)

    def test_max_drawdown_circuit_breaker_15_percent(self):
        # Starting equity $100,000
        halted, dd = self.rm.update_equity(100000.0)
        self.assertFalse(halted)
        self.assertAlmostEqual(dd, 0.0)

        # High watermark climbs to $110,000
        halted, dd = self.rm.update_equity(110000.0)
        self.assertFalse(halted)
        self.assertEqual(self.rm.peak_equity, 110000.0)

        # Equity drops to $95,000 (drawdown = (110k - 95k)/110k = 13.6% < 15%)
        halted, dd = self.rm.update_equity(95000.0)
        self.assertFalse(halted)
        self.assertAlmostEqual(dd, 15000.0 / 110000.0)

        # Equity drops to $93,000 (drawdown = 17k / 110k = 15.45% >= 15%) -> TRIP!
        halted, dd = self.rm.update_equity(93000.0)
        self.assertTrue(halted)
        self.assertTrue(self.rm.circuit_breaker_tripped)
        self.assertIn("CIRCUIT BREAKER TRIPPED", self.rm.halt_reason)

        # Once tripped, position size must return 0
        shares = self.rm.calculate_position_size("SPY", 100.0, 93000.0)
        self.assertEqual(shares, 0)

        # Circuit breaker stays tripped even if equity bounces back up
        halted, dd = self.rm.update_equity(120000.0)
        self.assertTrue(halted)
        self.assertTrue(self.rm.circuit_breaker_tripped)

    def test_zero_overnight_leverage(self):
        equity = 100000.0
        # 100k exposure on 100k equity = 1.0x -> Pass
        self.assertTrue(self.rm.verify_zero_overnight_leverage(100000.0, equity))
        # 80k exposure on 100k equity = 0.8x -> Pass
        self.assertTrue(self.rm.verify_zero_overnight_leverage(80000.0, equity))
        # 105k exposure on 100k equity = 1.05x -> Violates zero leverage
        self.assertFalse(self.rm.verify_zero_overnight_leverage(105000.0, equity))


class TestMockAlpacaClient(unittest.TestCase):
    """Verify raw REST Alpaca client logic with transport mock."""

    def test_client_get_account(self):
        def fake_transport(method, url, headers, body):
            self.assertEqual(method, "GET")
            self.assertIn("/v2/account", url)
            self.assertEqual(headers["APCA-API-KEY-ID"], "TEST_KEY")
            return {
                "equity": "100000.00",
                "cash": "100000.00",
                "buying_power": "100000.00",
                "currency": "USD",
                "status": "ACTIVE",
            }

        client = AlpacaClient("TEST_KEY", "TEST_SECRET", paper=True, transport=fake_transport)
        acc = client.get_account()
        self.assertEqual(acc.equity, 100000.0)
        self.assertEqual(acc.cash, 100000.0)
        self.assertEqual(acc.status, "ACTIVE")

    def test_client_submit_order_enforces_tags(self):
        calls = []

        def fake_transport(method, url, headers, body):
            calls.append((method, url, json.loads(body.decode("utf-8"))))
            return {"id": "order-12345", "status": "accepted"}

        client = AlpacaClient("TEST_KEY", "TEST_SECRET", paper=True, transport=fake_transport)

        # Untagged order fails before sending
        bad_order = OrderRequest(symbol="SPY", qty=10, side="buy", client_order_id="NO_TAG")
        with self.assertRaises(ValueError):
            client.submit_order(bad_order)
        self.assertEqual(len(calls), 0)

        # Tagged order succeeds
        good_order = OrderRequest(
            symbol="SPY",
            qty=10,
            side="buy",
            client_order_id="[WARBOT]:BOLO:SPY:BUY:1001",
        )
        resp = client.submit_order(good_order)
        self.assertEqual(resp["id"], "order-12345")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][2]["client_order_id"], "[WARBOT]:BOLO:SPY:BUY:1001")


class TestBacktestEngine(unittest.TestCase):
    """Verify Backtest engine and WARBOT Scoring formula."""

    def test_synthetic_bars_generation(self):
        bars = generate_synthetic_bars("TEST", num_bars=100, start_price=500.0)
        self.assertEqual(len(bars), 100)
        self.assertTrue(all(b.high >= b.low for b in bars))
        self.assertTrue(all(b.volume > 0 for b in bars))

    def test_run_backtest_scoring(self):
        bars = generate_synthetic_bars("TEST", num_bars=300, start_price=400.0)
        metrics, trades = run_backtest(
            bars=bars,
            initial_equity=100000.0,
            fast_period=9,
            slow_period=21,
        )
        self.assertEqual(metrics.initial_equity, 100000.0)
        self.assertGreaterEqual(metrics.total_trades, 0)
        # Verify WARBOT score formula: Return / max(MDD, 5%)
        ret_frac = (metrics.final_equity - metrics.initial_equity) / metrics.initial_equity
        mdd_frac = metrics.max_drawdown_pct / 100.0
        expected_score = ret_frac if ret_frac <= 0 else ret_frac / max(mdd_frac, DD_COWARDICE_FLOOR)
        self.assertAlmostEqual(metrics.warbot_score, expected_score, places=4)

        # All executed trades must have [WARBOT] order tag
        for t in trades:
            if "order_tag" in t:
                self.assertIn(WARBOT_ORDER_TAG, t["order_tag"])


class TestBoloBotDryRun(unittest.TestCase):
    """Verify BoloBot dry-run mode."""

    def test_dry_run_cycle_without_credentials(self):
        bot = BoloBot(
            client=None,
            symbols=["SPY", "QQQ"],
            dry_run=True,
        )
        signals = bot.run_cycle()
        self.assertEqual(len(signals), 2)
        symbols_found = {s.symbol for s in signals}
        self.assertEqual(symbols_found, {"SPY", "QQQ"})


if __name__ == "__main__":
    unittest.main()
