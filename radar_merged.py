import tkinter as tk
from tkinter import ttk
import pandas as pd
import numpy as np
import threading
import time
import json
import os
import logging
import queue
from datetime import datetime, timezone
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple

# --- Optional Libraries ---
try:
    import requests
    REQUESTS_OK = True
except ImportError:
    REQUESTS_OK = False
    raise RuntimeError("pip install requests")

try:
    import websocket
    WEBSOCKET_OK = True
except ImportError:
    raise RuntimeError("pip install websocket-client")

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import GradientBoostingRegressor
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

try:
    import xgboost as xgb
    XGBOOST_OK = True
except ImportError:
    XGBOOST_OK = False
    logging.warning("XGBoost not installed. Run: pip install xgboost")

try:
    import winsound
    def alert_sound():
        winsound.Beep(1200, 300)
        winsound.Beep(1500, 300)
except ImportError:
    def alert_sound():
        print('\x07')

def _calc_macd(closes, fast=12, slow=26, signal=9):
    if len(closes) < slow + signal:
        return np.zeros(len(closes)), np.zeros(len(closes)), np.zeros(len(closes))
    ema_fast = pd.Series(closes).ewm(span=fast, adjust=False).mean().values
    ema_slow = pd.Series(closes).ewm(span=slow, adjust=False).mean().values
    macd_line = ema_fast - ema_slow
    signal_line = pd.Series(macd_line).ewm(span=signal, adjust=False).mean().values
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def _calc_vwap(highs, lows, closes, volumes):
    if len(closes) < 2:
        return np.zeros(len(closes))
    tp = (highs + lows + closes) / 3.0
    cum_vol = np.cumsum(volumes)
    cum_tp_vol = np.cumsum(tp * volumes)
    vwap = np.where(cum_vol > 0, cum_tp_vol / cum_vol, closes)
    return vwap

def _calc_stoch_rsi(closes, period=14, stoch_period=14, k_period=3, d_period=3):
    if len(closes) < period + stoch_period + k_period + d_period:
        return np.full(len(closes), 50.0), np.full(len(closes), 50.0), np.full(len(closes), 50.0)
    deltas = np.diff(closes, prepend=closes[0])
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    alpha = 1.0 / period
    avg_g = np.zeros(len(closes)); avg_l = np.zeros(len(closes))
    avg_g[0] = gains[0]; avg_l[0] = losses[0]
    for i in range(1, len(closes)):
        avg_g[i] = alpha * gains[i] + (1 - alpha) * avg_g[i-1]
        avg_l[i] = alpha * losses[i] + (1 - alpha) * avg_l[i-1]
    rs = np.where(avg_l > 1e-9, avg_g / avg_l, 100.0)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi_min = pd.Series(rsi).rolling(window=stoch_period, min_periods=1).min().values
    rsi_max = pd.Series(rsi).rolling(window=stoch_period, min_periods=1).max().values
    denom = rsi_max - rsi_min
    stoch = np.where(denom > 1e-9, (rsi - rsi_min) / denom, 0.5) * 100.0
    k = pd.Series(stoch).rolling(window=k_period, min_periods=1).mean().values
    d = pd.Series(k).rolling(window=d_period, min_periods=1).mean().values
    j = 3 * k - 2 * d
    return k, d, j

LOG_FILE = "radar_v21.log"
TF_MS = {
    "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000,
    "30m": 1_800_000, "1h": 3_600_000, "2h": 7_200_000,
    "4h": 14_400_000, "6h": 21_600_000, "8h": 28_800_000,
    "12h": 43_200_000, "1d": 86_400_000,
}

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()
    ]
)

# --- Config ---
PRICE_HISTORY_MAX = 200
UI_REFRESH_MS = 100
SIGNAL_HISTORY_MAX = 50
VERSION = "21.0-ULTRA"
API_DELAY_SEC = 0.08

# --- AI Signal Report (v21) ---
AI_SIGNAL_THRESHOLD_PCT = {
    "5m":  0.20,
    "15m": 0.35,
    "1h":  0.60,
    "4h":  1.00,
    "1d":  1.80,
}
AI_TP_SAFETY_MARGIN     = 0.985
AI_SL_PCT               = {
    "5m":  0.40,
    "15m": 0.60,
    "1h":  1.00,
    "4h":  1.50,
    "1d":  2.50,
}
AI_MIN_COMMISSION_PCT   = 0.08

# --- Rate Limit Protection ---
RATE_LIMIT_COOLDOWN_SEC = 60
MIN_VOLUME_USDT = 5_000_000

# --- Reinforcement Learning ---
RL_WIN_THRESHOLD   = 0.65
RL_LOSS_THRESHOLD  = 0.45
RL_CHECK_INTERVAL  = 300
SAVED_SIGNALS_FILE = "saved_signals.json"

# --- AI Prediction (v21 ENHANCED) ---
AI_PRED_LOOKBACK    = 300
AI_LOOKAHEAD_SHIFT  = {
    "1m": 3, "3m": 3, "5m": 3, "15m": 4, "1h": 6, "4h": 8, "1d": 10
}

def _get_lookahead_shift(tf: str) -> int:
    return AI_LOOKAHEAD_SHIFT.get(tf, 3)
AI_1H_CANDLES       = 30

TF_HISTORY_MAP = {
    "5m":  300,
    "15m": 350,
    "1h":  250,
    "4h":  400,
    "1d":  500,
}

# --- Session Filters (UTC) ---
SESSION_ALLOWED = [(7, 11), (12.5, 20.0)]
SESSION_KILL = (21.0, 23.0)

# --- Meme Coin Filter ---
MEME_COINS = {"HMSTR","DOGE","SHIB","PEPE","FLOKI","BONK","WIF","BOME","BRETT","POPCAT","MOG","SLERF","MEW","TURBO","MYRO","PONKE","WEN","SNAP","BABYDOGE","ELON","SAMO","DOBO","KISHU"}
MAX_LEVERAGE_MEME = 5
MAX_LEVERAGE_STANDARD = 20
MEME_MIN_MTF = 5
STANDARD_MIN_MTF = 3

# --- Funding / OI Thresholds ---
FUNDING_WARNING = -0.01
FUNDING_DANGER = -0.05
OI_DROP_PCT = 5.0

# ==========================================
# v21 ULTRA: WEIGHT CONFIGURATION
# ==========================================
# Price Action + Volume + Market Structure = HIGHEST
# EMA + Trend = MEDIUM
# RSI + Stochastic = LOW
WEIGHT_CONFIG = {
    "market_structure": 3.0,   # HH/HL, LH/LL, Breakouts, Pullbacks
    "volume": 2.5,           # Volume confirmation, delta
    "price_action": 2.5,     # Candle patterns, wicks, engulfing
    "ema_trend": 1.5,        # EMA alignment
    "adx_trend": 1.5,        # ADX trend strength
    "rsi": 0.8,              # RSI (reduced weight)
    "stochastic": 0.6,       # Stochastic (reduced weight)
    "macd": 1.0,             # MACD (medium, but ignored if conflicting)
    "supertrend": 1.2,       # SuperTrend
    "patterns": 2.0,         # Chart patterns (flags, wedges, etc.)
    "ob_imbalance": 0.5,     # Order Book (auxiliary only)
}

# Minimum agreement threshold for signals
MIN_CONFLUENCE_THRESHOLD = 7  # out of weighted max
MIN_RR_RATIO = 2.0

# ==========================================

# --- Data Structures ---
@dataclass
class Candle:
    t: int
    o: float
    h: float
    l: float
    c: float
    v: float

@dataclass
class AnalysisResult:
    direction: str = "WAIT"
    strength: str = "NEUTRAL"
    confidence: float = 0.0
    confluence: int = 0
    reason: str = ""
    score: float = 0.0
    entry_low: float = 0.0
    entry_high: float = 0.0
    tp: float = 0.0
    sl: float = 0.0
    rr: float = 0.0
    ema20: float = 0.0
    ema50: float = 0.0
    ema200: float = 0.0
    rsi: float = 50.0
    adx: float = 0.0
    di_plus: float = 0.0
    di_minus: float = 0.0
    atr: float = 0.0
    bb_upper: float = 0.0
    bb_lower: float = 0.0
    bb_mid: float = 0.0
    bb_pct: float = 0.5
    bb_bandwidth: float = 0.0
    bb_pct_b: float = 0.5
    bb_squeeze: bool = False
    vol_ratio: float = 0.0
    vol_delta: float = 0.0
    ob_imbalance: float = 1.0
    sar: float = 0.0
    supertrend: float = 0.0
    supertrend_dir: str = "NONE"
    kdj_k: float = 50.0
    kdj_d: float = 50.0
    kdj_j: float = 50.0
    m_pattern: bool = False
    w_pattern: bool = False
    double_top: bool = False
    double_bottom: bool = False
    rising_wedge: bool = False
    falling_wedge: bool = False
    flag_bull: bool = False
    flag_bear: bool = False
    pennant: bool = False
    rectangle: bool = False
    head_shoulders: bool = False
    inv_head_shoulders: bool = False
    pattern_label: str = ""
    price_action_bias: str = "NEUTRAL"
    swing_low: float = 0.0
    swing_high: float = 0.0
    fib_levels: Dict = field(default_factory=dict)
    candle_strength: float = 0.0
    candle_open_price: float = 0.0
    buy_pressure: float = 0.0
    sell_pressure: float = 0.0
    rsi_divergence: str = "NONE"
    rsi_micro_divergence: str = "NONE"
    market_regime: str = "UNKNOWN"
    smart_sl: float = 0.0
    smart_tp: float = 0.0
    position_size_pct: float = 1.0
    win_rate_est: float = 0.0
    smart_entry_low: float = 0.0
    smart_entry_high: float = 0.0
    smart_tp1: float = 0.0
    smart_tp2: float = 0.0
    smart_tp3: float = 0.0
    entry_method: str = ""
    swing_structure: str = "UNKNOWN"
    swing_top2_highs: List = field(default_factory=list)
    swing_top2_lows: List = field(default_factory=list)
    swing_struct_detail: str = ""
    swing_struct_confirmed: bool = False
    price_prediction: Dict = field(default_factory=dict)
    market_structure: Dict = field(default_factory=dict)
    time_to_target: int = 0
    trailing_sl: float = 0.0
    session_valid: bool = True
    session_msg: str = ""
    # v21: New fields for conflict detection
    indicator_conflict: bool = False
    trend_aligned: bool = False
    volume_confirmed: bool = False
    structure_aligned: bool = False

class NarrativeResult:
    def __init__(self):
        self.headline = ""
        self.market_story = ""
        self.key_zones = ""
        self.scenarios = ""
        self.comparison = ""
        self.risk_assessment = ""
        self.action_plan = ""
        self.sentiment = "NEUTRAL"
        self.urgency = "LOW"

class NarrativeEngine:
    def __init__(self):
        self._last_cp = 0.0

    def generate_narrative(self, analysis, cp, prev_cp, sr_levels, symbol, timeframe, comparison_data=None):
        result = NarrativeResult()
        change_pct = ((cp - prev_cp) / prev_cp * 100) if prev_cp > 0 else 0

        direction = analysis.direction
        strength = analysis.strength
        if direction == "BUY":
            if strength == "VERY STRONG":
                result.headline = f"[EXPLOSIVE] {symbol} EXPLOSIVE BULLISH SETUP on {timeframe}"
            elif strength == "STRONG":
                result.headline = f"[BULL] {symbol} Strong Buy Signal on {timeframe}"
            else:
                result.headline = f"[LOW] {symbol} Moderate Bullish Bias on {timeframe}"
        elif direction == "SELL":
            if strength == "VERY STRONG":
                result.headline = f"[CRITICAL] {symbol} CRITICAL BEARISH ALERT on {timeframe}"
            elif strength == "STRONG":
                result.headline = f"[BEAR] {symbol} Strong Sell Signal on {timeframe}"
            else:
                result.headline = f"[HIGH] {symbol} Moderate Bearish Bias on {timeframe}"
        else:
            if abs(change_pct) > 2:
                result.headline = f"[VOLATILE] {symbol} High Volatility on {timeframe} - Wait for Direction"
            else:
                result.headline = f"[WAIT] {symbol} Consolidating on {timeframe} - Patience Required"

        result.market_story = ""

        zones_text = []
        supports = [l for l in sr_levels if l['type'] == 'S']
        resistances = [l for l in sr_levels if l['type'] == 'R']
        supports.sort(key=lambda x: abs(x['price'] - cp))
        resistances.sort(key=lambda x: abs(x['price'] - cp))

        if supports:
            s = supports[0]
            dist = abs(s['price'] - cp) / cp * 100
            emoji = "[LOW]" if s['strength'] == 'STRONG' else "[MODERATE]" if s['strength'] == 'MODERATE' else "[WEAK]"
            zones_text.append(f"{emoji} SUPPORT: {s['price']:.4f} ({s['touches']} touches, {dist:.1f}% away)")

        if resistances:
            r = resistances[0]
            dist = abs(r['price'] - cp) / cp * 100
            emoji = "[HIGH]" if r['strength'] == 'STRONG' else "[MED]" if r['strength'] == 'MODERATE' else "[WEAK]"
            zones_text.append(f"{emoji} RESISTANCE: {r['price']:.4f} ({r['touches']} touches, {dist:.1f}% away)")

        strong_zones = [l for l in sr_levels if l['strength'] == 'STRONG' and l not in supports[:1] and l not in resistances[:1]]
        for z in strong_zones[:2]:
            dist = abs(z['price'] - cp) / cp * 100
            emoji = "[LOW]" if z['type'] == 'S' else "[HIGH]"
            zones_text.append(f"{emoji} {z['type']}: {z['price']:.4f} (strong, {dist:.1f}% away)")

        if not zones_text:
            zones_text.append("No significant S/R zones detected - trade with ATR-based levels.")

        result.key_zones = "\n".join(zones_text)

        scenarios = []
        resistances_sorted = sorted([l['price'] for l in sr_levels if l['type'] == 'R' and l['price'] > cp])
        if resistances_sorted:
            r1 = resistances_sorted[0]
            r2 = resistances_sorted[1] if len(resistances_sorted) > 1 else r1 * 1.05
            scenarios.append(f"[BULL] BULL CASE: Break above {r1:.4f} → Target {r2:.4f} ({(r2/cp-1)*100:.1f}%)")
        else:
            tp = cp + analysis.atr * 3
            scenarios.append(f"[BULL] BULL CASE: Continuation → Target {tp:.4f} ({(tp/cp-1)*100:.1f}%)")

        supports_sorted = sorted([l['price'] for l in sr_levels if l['type'] == 'S' and l['price'] < cp], reverse=True)
        if supports_sorted:
            s1 = supports_sorted[0]
            s2 = supports_sorted[1] if len(supports_sorted) > 1 else s1 * 0.95
            scenarios.append(f"[BEAR] BEAR CASE: Break below {s1:.4f} → Target {s2:.4f} ({(s2/cp-1)*100:.1f}%)")
        else:
            sl = cp - analysis.atr * 2
            scenarios.append(f"[BEAR] BEAR CASE: Reversal → Target {sl:.4f} ({(sl/cp-1)*100:.1f}%)")

        if direction == "WAIT":
            _wait_low  = supports_sorted[0] if supports_sorted else cp * 0.98
            _wait_high = resistances_sorted[0] if resistances_sorted else cp * 1.02
            scenarios.append(f"[WAIT] WAIT: Range between {_wait_low:.4f} and {_wait_high:.4f}")

        result.scenarios = "\n".join(scenarios)

        risks = []
        atr_pct = (analysis.atr / cp) * 100 if cp > 0 else 0
        if atr_pct > 2:
            risks.append(f"[HIGH] HIGH VOLATILITY: ATR at {atr_pct:.1f}% - wide stops required")
        elif atr_pct > 1:
            risks.append(f"[MODERATE] MODERATE VOLATILITY: ATR at {atr_pct:.1f}% - standard risk management")
        else:
            risks.append(f"[LOW] LOW VOLATILITY: ATR at {atr_pct:.1f}% - tight stops possible")

        if analysis.rising_wedge or analysis.double_top:
            risks.append("[RISK] REVERSAL PATTERN: Bullish positions carry elevated risk")
        elif analysis.falling_wedge or analysis.double_bottom:
            risks.append("[RISK] REVERSAL PATTERN: Bearish positions carry elevated risk")

        if analysis.confluence < MIN_CONFLUENCE_THRESHOLD:
            risks.append(f"[RISK] LOW CONFLUENCE: Only {analysis.confluence}/{MIN_CONFLUENCE_THRESHOLD} indicators agree - weak signal")
        elif analysis.confluence >= 8:
            risks.append(f"[OK] HIGH CONFLUENCE: {analysis.confluence}/10 indicators align - strong signal")

        if not analysis.session_valid:
            risks.append(f"[SESSION] {analysis.session_msg}")

        # v21: Add conflict warning
        if analysis.indicator_conflict:
            risks.append("[CONFLICT] INDICATOR CONFLICT DETECTED - Signal suppressed")

        if not risks:
            risks.append("[OK] Standard risk environment - follow your trading plan.")

        result.risk_assessment = "\n".join(risks)

        plan = []
        if direction == "BUY":
            plan.append(f"[ZONES] ENTRY: {analysis.entry_low:.4f} - {analysis.entry_high:.4f}")
            plan.append(f"[SL] STOP LOSS: {analysis.smart_sl if analysis.smart_sl > 0 else analysis.sl:.4f}")
            plan.append(f"[ZONES] TAKE PROFIT: {analysis.smart_tp if analysis.smart_tp > 0 else analysis.tp:.4f}")
            plan.append(f"[SCENARIOS] R:R RATIO: 1:{analysis.rr:.1f}")
            plan.append(f"[SIZE] POSITION SIZE: {analysis.position_size_pct:.2f}% of balance")
            plan.append(f"[WR] ESTIMATED WIN RATE: {analysis.win_rate_est:.0f}%")
            if analysis.time_to_target > 0:
                plan.append(f"[TIME] Est. Time to Target: ~{analysis.time_to_target} candles")
            if analysis.rr >= MIN_RR_RATIO:
                plan.append("[OK] FAVORABLE R:R - Setup meets minimum criteria")
            else:
                plan.append("[RISK] POOR R:R - Consider waiting for better entry")
        elif direction == "SELL":
            plan.append(f"[ZONES] ENTRY: {analysis.entry_low:.4f} - {analysis.entry_high:.4f}")
            plan.append(f"[SL] STOP LOSS: {analysis.smart_sl if analysis.smart_sl > 0 else analysis.sl:.4f}")
            plan.append(f"[ZONES] TAKE PROFIT: {analysis.smart_tp if analysis.smart_tp > 0 else analysis.tp:.4f}")
            plan.append(f"[SCENARIOS] R:R RATIO: 1:{analysis.rr:.1f}")
            plan.append(f"[SIZE] POSITION SIZE: {analysis.position_size_pct:.2f}% of balance")
            plan.append(f"[WR] ESTIMATED WIN RATE: {analysis.win_rate_est:.0f}%")
            if analysis.time_to_target > 0:
                plan.append(f"[TIME] Est. Time to Target: ~{analysis.time_to_target} candles")
            if analysis.rr >= MIN_RR_RATIO:
                plan.append("[OK] FAVORABLE R:R - Setup meets minimum criteria")
            else:
                plan.append("[RISK] POOR R:R - Consider waiting for better entry")
        else:
            plan.append("[WAIT] NO TRADE - Conditions don't meet entry criteria")
            plan.append(f"[TIP] Wait for: Break above {analysis.entry_high:.4f} or below {analysis.entry_low:.4f}")
            plan.append(f"[SCENARIOS] Current score: {analysis.score} (need ±{int(35*WEIGHT_CONFIG['market_structure'])} for signal)")

        result.action_plan = "\n".join(plan)
        result.sentiment = direction
        urgency_score = 0
        if analysis.confidence >= 80: urgency_score += 3
        elif analysis.confidence >= 60: urgency_score += 2
        elif analysis.confidence >= 40: urgency_score += 1
        if abs(change_pct) > 2: urgency_score += 2
        elif abs(change_pct) > 1: urgency_score += 1
        if analysis.vol_ratio > 150: urgency_score += 1
        if analysis.confluence >= 8: urgency_score += 1
        if urgency_score >= 5: result.urgency = "HIGH"
        elif urgency_score >= 3: result.urgency = "MEDIUM"
        else: result.urgency = "LOW"

        return result

class ScrollableFrame(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.configure(bg="#0b0e11")
        self.canvas = tk.Canvas(self, bg="#0b0e11", highlightthickness=0)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar = tk.Scrollbar(self, orient="vertical", command=self.canvas.yview,
                                       bg="#2b3139", troughcolor="#0b0e11",
                                       activebackground="#f0b90b", width=10)
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.inner = tk.Frame(self.canvas, bg="#0b0e11")
        self.inner_id = self.canvas.create_window((0, 0), window=self.inner, anchor="nw", width=360)
        self.inner.bind("<Configure>", self._on_inner_configure)
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfig(self.inner_id, width=e.width))
        self._bind_scroll(self.canvas)
        self._bind_scroll(self.inner)

    def _on_inner_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        self._bind_scroll_recursive(self.inner)

    def _bind_scroll(self, widget):
        widget.bind("<MouseWheel>", self._on_mousewheel, add="+")
        widget.bind("<Button-4>",   self._on_mousewheel, add="+")
        widget.bind("<Button-5>",   self._on_mousewheel, add="+")
        widget.bind("<ButtonPress-1>",   self._on_touch_start, add="+")
        widget.bind("<B1-Motion>",       self._on_touch_move,  add="+")
        widget.bind("<ButtonRelease-1>", self._on_touch_end,   add="+")

    def _bind_scroll_recursive(self, widget):
        self._bind_scroll(widget)
        for child in widget.winfo_children():
            self._bind_scroll_recursive(child)

    def _on_mousewheel(self, event):
        if event.num == 4 or event.delta > 0:
            self.canvas.yview_scroll(-3, "units")
        elif event.num == 5 or event.delta < 0:
            self.canvas.yview_scroll(3, "units")

    def _on_touch_start(self, event):
        self._touch_y = event.y_root
        self._touch_scrolling = False

    def _on_touch_move(self, event):
        if not hasattr(self, '_touch_y') or self._touch_y is None:
            return
        dy = self._touch_y - event.y_root
        if abs(dy) > 5:
            self._touch_scrolling = True
            self.canvas.yview_scroll(int(dy / 8), "units")
            self._touch_y = event.y_root

    def _on_touch_end(self, event):
        self._touch_y = None
        self._touch_scrolling = False

class BinanceRadarPro:
    def __init__(self, root):
        self.root = root
        self.root.title(f"PRO AI RADAR v{VERSION} -- ULTRA [STRUCTURE-FIRST ENGINE]")
        self.root.geometry("400x700")
        self.root.configure(bg="#0b0e11")
        self.root.resizable(False, False)
        self.running = True

        self.symbol = "BTCUSDT"
        self.interval = "1h"
        self.price = 0.0
        self.prev_price = 0.0
        self.mark_price = 0.0
        self.display_price = 0.0
        self.candle_open_price = 0.0
        self.price_anim_speed = 1.0
        self.price_history = deque(maxlen=PRICE_HISTORY_MAX)
        self.last_update = "--:--:--"
        self.error_msg = ""
        self.color = "#aeb4bc"
        self.ml_status = "ML: ON" if SKLEARN_OK else "ML: OFF"
        self.ws_status = "WS: OFF"

        self.current_analysis = AnalysisResult()
        self.analysis_lock = threading.Lock()

        self.candle_deque = deque(maxlen=600)
        self.df_lock = threading.Lock()
        self.new_data_event = threading.Event()

        self.ui_queue = queue.Queue()

        self.ws_mark = None
        self.ws_kline = None
        self.ws_connected = False
        self.ws_last_ping = 0
        self._ws_lock = threading.Lock()
        self._last_price_update = 0

        self.signal_history = deque(maxlen=SIGNAL_HISTORY_MAX)
        self.last_signal_direction = "WAIT"
        self.signal_stats = {"buy": 0, "sell": 0, "wait": 0}
        self._hist_pool = []

        self._bt_signals   = []
        self._bt_wins      = 0
        self._bt_losses    = 0
        self._bt_open      = None
        self._bt_lock      = threading.Lock()

        # v21: Hierarchical TF system
        self._higher_tf_trend = "NEUTRAL"  # 1H trend direction
        self._higher_tf_confidence = 0.0
        self._mid_tf_confirm = "NEUTRAL"  # 15M confirmation
        self._mtf_master = "WAIT"
        self.mtf_agree = 0
        self._mtf_cached_results = {}
        self._mtf_coin_data = {}
        self._mtf_data_lock = threading.Lock()
        self._coin_lookup = {}
        self._current_coin_base = "BTC"

        self.intel_sr_levels = []
        self.intel_trend500 = "WAIT"
        self.intel_trend_str = ""
        self.intel_ema_stack = "--"
        self.intel_structure = "--"
        self.intel_slope = "--"
        self.intel_liq_buy = 0.0
        self.intel_liq_sell = 0.0
        self.intel_liq_dom = "--"
        self.intel_liq_zones = "--"
        self.intel_liq_sweep = "--"
        self.intel_liq_1h = {}
        self.intel_liq_4h = {}
        self.intel_news = []
        self.intel_news_sentiment = "--"
        self.intel_last_refresh = 0
        self.intel_lock = threading.Lock()

        self._initial_coin_order = []
        self._mtf_coin_scores = {}
        self._vol_24h_usdt = 0.0
        self._rl_conf_boost = 1.0

        self.funding_rate = 0.0
        self.funding_next = 0
        self.oi_value = 0.0
        self.oi_change_pct = 0.0
        self.oi_prev = 0.0
        self._last_funding_update = 0
        self._last_oi_update = 0

        self._last_api_call = 0
        self._api_lock = threading.Lock()
        self._rate_limit_until = 0.0

        self._mtf_last_fetch_time = 0
        self._mtf_fetch_interval = 300
        self._mtf_is_fetching = False

        self._last_analysis_time = 0.0
        self._analysis_throttle_map = {
            "5m": 0, "15m": 0, "1h": 300, "4h": 300, "1d": 300,
        }

        self._last_ob_update = 0
        self._ob_imbalance = 1.0

        self._winrate_predictions = []
        self._winrate_lock = threading.Lock()
        self._winrate_wins = 0
        self._winrate_losses = 0

        # v21: Enhanced prediction cache
        self._ai_pred_cache = {}
        self._ai_pred_cache_key = None
        self._ai_pred_last_candle_t = 0
        self._prediction_errors = deque(maxlen=50)
        self._last_prediction = None
        self._force_retrain_counter = 0

        # v21: Structure-first cache
        self._structure_cache = {}
        self._structure_cache_time = 0

        # v21: Conflict tracking
        self._last_conflict_reason = ""
        self._conflict_count = 0

        self.build_ui()
        self.root.after(300, self.start_bg)
        self.root.after(500, self._resize_coin_tv)
        self.refresh_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _get_history_limit(self, tf=None) -> int:
        tf = tf or self.interval
        limit = TF_HISTORY_MAP.get(tf, 300)
        return limit + 50

    def _api_get(self, url, timeout=8):
        with self._api_lock:
            elapsed = time.time() - self._last_api_call
            if elapsed < API_DELAY_SEC:
                time.sleep(API_DELAY_SEC - elapsed)

            if getattr(self, '_rate_limit_until', 0) > time.time():
                remaining = int(self._rate_limit_until - time.time())
                raise RuntimeError(f"RATE LIMIT COOLDOWN: {remaining}s remaining")
            try:
                r = requests.get(url, timeout=timeout)
                self._last_api_call = time.time()

                if r.status_code == 429:
                    self._rate_limit_until = time.time() + RATE_LIMIT_COOLDOWN_SEC
                    logging.warning("Binance 429 Rate Limit! Cooling down for %ds", RATE_LIMIT_COOLDOWN_SEC)
                    _cd = RATE_LIMIT_COOLDOWN_SEC
                    self.ui_queue.put(lambda cd=_cd: self.status_lbl.config(
                        text=f"⚠ RATE LIMIT! Cooldown {cd}s", fg="#ff4d4d"))
                    raise RuntimeError("HTTP 429 Rate Limited by Binance")
                return r
            except RuntimeError:
                raise
            except Exception as e:
                self._last_api_call = time.time()
                raise e

    def _fetch_funding(self):
        try:
            if not self.symbol:
                return
            url = f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={self.symbol}&limit=1"
            r = self._api_get(url, timeout=4)
            if r.status_code == 200:
                data = r.json()
                if data and len(data) > 0:
                    self.funding_rate = float(data[0].get("fundingRate", 0))
                    self.funding_next = int(data[0].get("fundingTime", 0)) // 1000
                    self._last_funding_update = time.time()
                    logging.info("Funding rate updated: %.4f%%", self.funding_rate * 100)
        except Exception as e:
            logging.warning("Funding fetch error: %s", str(e)[:40])

    def _fetch_oi(self):
        try:
            if not self.symbol:
                return
            url = f"https://fapi.binance.com/fapi/v1/openInterest?symbol={self.symbol}"
            r = self._api_get(url, timeout=4)
            if r.status_code == 200:
                data = r.json()
                oi = float(data.get("openInterest", 0))
                if self.oi_value > 0 and oi > 0:
                    self.oi_change_pct = ((oi - self.oi_value) / self.oi_value) * 100
                else:
                    self.oi_change_pct = 0.0
                self.oi_prev = self.oi_value
                self.oi_value = oi
                self._last_oi_update = time.time()
                logging.info("OI updated: %.2fM (%.1f%%)", oi / 1e6, self.oi_change_pct)
        except Exception as e:
            logging.warning("OI fetch error: %s", str(e)[:40])

    def _fetch_orderbook_imbalance(self):
        try:
            if not self.symbol:
                return 1.0
            url = f"https://fapi.binance.com/fapi/v1/depth?symbol={self.symbol}&limit=50"
            r = self._api_get(url, timeout=3)
            if r.status_code == 200:
                data = r.json()
                bids = data.get("bids", [])
                asks = data.get("asks", [])
                if not bids or not asks:
                    return 1.0
                bid_vol = sum(float(b[1]) for b in bids[:10])
                ask_vol = sum(float(a[1]) for a in asks[:10])
                if ask_vol > 0:
                    ratio = bid_vol / ask_vol
                    self._ob_imbalance = ratio
                    self._last_ob_update = time.time()
                    return ratio
            return 1.0
        except Exception:
            return 1.0

    def _check_session_filter(self):
        # v21: Session filter disabled per user request - 24/7 trading
        return True, ""

    def _is_meme_coin(self):
        base = self.symbol.replace("USDT", "").upper() if self.symbol else ""
        if base in MEME_COINS:
            return True
        for meme in MEME_COINS:
            if meme in base:
                return True
        return False

    def _get_max_leverage(self):
        return MAX_LEVERAGE_MEME if self._is_meme_coin() else MAX_LEVERAGE_STANDARD

    def _get_min_mtf(self):
        return MEME_MIN_MTF if self._is_meme_coin() else STANDARD_MIN_MTF

    def _check_funding_risk(self):
        if self.funding_rate <= FUNDING_DANGER:
            return False, f"FUNDING DANGER: {self.funding_rate*100:.4f}% -- Short squeeze likely!"
        if self.funding_rate <= FUNDING_WARNING:
            return False, f"FUNDING WARNING: {self.funding_rate*100:.4f}% -- Crowd is short"
        if self.funding_rate >= 0.05:
            return False, f"FUNDING WARNING: {self.funding_rate*100:.4f}% -- Crowd is long (long squeeze risk)"
        return True, ""

    def _check_oi_risk(self, price_direction="BUY"):
        if self.oi_change_pct < -OI_DROP_PCT and price_direction == "BUY":
            return False, f"OI DROP: {self.oi_change_pct:.1f}% -- Fake pump, liquidity leaving"
        if self.oi_change_pct < -OI_DROP_PCT and price_direction == "SELL":
            return True, ""
        return True, ""

    def _check_ob_risk(self, direction="BUY"):
        # v21: OB is AUXILIARY ONLY - low weight, never blocking alone
        ratio = self._ob_imbalance
        if direction == "BUY":
            if ratio >= 2.0:
                return True, f"OB STRONG BUY: {ratio:.2f} (Heavy bid dominance)"
            return True, f"OB OK: {ratio:.2f}"
        elif direction == "SELL":
            if ratio <= 0.5:
                return True, f"OB STRONG SELL: {ratio:.2f} (Heavy ask dominance)"
            return True, f"OB OK: {ratio:.2f}"
        return True, ""

    def start_bg(self):
        threading.Thread(target=self._ui_queue_consumer, daemon=True).start()
        threading.Thread(target=self.load_symbols_thread, daemon=True).start()
        self.root.after(1000, self._load_initial_data)
        threading.Thread(target=self.ws_manager_loop, daemon=True).start()
        threading.Thread(target=self.analysis_loop, daemon=True).start()
        threading.Thread(target=self.intel_auto_refresh_loop, daemon=True).start()
        threading.Thread(target=self.price_fallback_loop, daemon=True).start()
        threading.Thread(target=self._mtf_auto_loop, daemon=True).start()
        threading.Thread(target=self._coin_rescan_loop, daemon=True).start()
        threading.Thread(target=self._funding_oi_loop, daemon=True).start()
        threading.Thread(target=self._orderbook_loop, daemon=True).start()
        threading.Thread(target=self._reinforcement_learning_loop, daemon=True).start()
        threading.Thread(target=self._winrate_checker_loop, daemon=True).start()
        # v21: Hierarchical TF monitoring
        threading.Thread(target=self._hierarchical_tf_loop, daemon=True).start()

    def _load_initial_data(self):
        try:
            if self.symbol:
                self.load_historical_klines()
                logging.info("Initial data loaded for %s", self.symbol)
        except Exception as e:
            logging.warning("Initial data load failed: %s", str(e)[:60])

    def _funding_oi_loop(self):
        time.sleep(5)
        while self.running:
            self._fetch_funding()
            self._fetch_oi()
            time.sleep(20)

    def _orderbook_loop(self):
        time.sleep(5)
        while self.running:
            self._fetch_orderbook_imbalance()
            time.sleep(10)

    def _ui_queue_consumer(self):
        """
        In GUI mode: schedules tasks via root.after().
        In headless mode: executes UI tasks directly but swallows
        any AttributeError from missing widgets silently.
        """
        headless = isinstance(self.root, FakeRoot)
        while self.running:
            try:
                task = self.ui_queue.get(timeout=0.1)
                if callable(task):
                    if headless:
                        try:
                            task()
                        except AttributeError:
                            pass   # missing widget — ignore silently
                        except Exception:
                            pass
                    else:
                        self.root.after(0, task)
            except queue.Empty:
                continue
            except Exception as e:
                logging.warning("UI queue error: %s", e)

    def _coin_rescan_loop(self):
        time.sleep(5)
        while self.running:
            try:
                if hasattr(self, '_mtf_coin_data') and self._mtf_coin_data:
                    self._refresh_coin_display_from_mtf()
            except Exception as e:
                logging.warning("Coin display refresh error: %s", str(e)[:60])
            time.sleep(3)

    def _hierarchical_tf_loop(self):
        """v21: Hierarchical TF monitoring - 1H trend, 15M confirm, 5M entry"""
        time.sleep(10)
        while self.running:
            try:
                if not self.symbol:
                    time.sleep(30)
                    continue

                # Fetch 1H trend
                try:
                    limit_1h = TF_HISTORY_MAP.get("1h", 250)
                    url_1h = f"https://fapi.binance.com/fapi/v1/klines?symbol={self.symbol}&interval=1h&limit={limit_1h}"
                    r_1h = self._api_get(url_1h, timeout=5)
                    if r_1h.status_code == 200:
                        klines_1h = r_1h.json()
                        if klines_1h and len(klines_1h) >= 50:
                            df_1h = self._klines_to_df(klines_1h)
                            cp_1h = float(df_1h["c"].iloc[-1])
                            result_1h = self._deep_analyze(df_1h, cp_1h)
                            self._higher_tf_trend = result_1h.direction
                            self._higher_tf_confidence = result_1h.confidence
                            logging.info("[HIERARCHY] 1H Trend: %s (conf: %.1f%%)", 
                                       result_1h.direction, result_1h.confidence)
                except Exception as e:
                    logging.warning("1H trend fetch error: %s", str(e)[:60])

                time.sleep(15)

                # Fetch 15M confirmation (only if current TF is 5M or 15M)
                if self.interval in ("5m", "15m", "3m", "1m"):
                    try:
                        limit_15m = TF_HISTORY_MAP.get("15m", 350)
                        url_15m = f"https://fapi.binance.com/fapi/v1/klines?symbol={self.symbol}&interval=15m&limit={limit_15m}"
                        r_15m = self._api_get(url_15m, timeout=5)
                        if r_15m.status_code == 200:
                            klines_15m = r_15m.json()
                            if klines_15m and len(klines_15m) >= 50:
                                df_15m = self._klines_to_df(klines_15m)
                                cp_15m = float(df_15m["c"].iloc[-1])
                                result_15m = self._deep_analyze(df_15m, cp_15m)
                                self._mid_tf_confirm = result_15m.direction
                                logging.info("[HIERARCHY] 15M Confirm: %s (conf: %.1f%%)", 
                                           result_15m.direction, result_15m.confidence)
                    except Exception as e:
                        logging.warning("15M confirm fetch error: %s", str(e)[:60])

                time.sleep(45)
            except Exception as e:
                logging.warning("Hierarchical TF loop error: %s", str(e)[:60])
                time.sleep(30)

    def _refresh_coin_display_from_mtf(self):
        if not getattr(self, '_initial_coin_order', []):
            return
        coin_data = []
        with self._mtf_data_lock:
            mtf_snapshot = dict(self._mtf_coin_data)
        for base in getattr(self, '_initial_coin_order', []):
            if base in mtf_snapshot:
                d = mtf_snapshot[base]
                coin_data.append((base, (d['agree'], abs(d['conf'])), d['agree'], d['direction'], d['conf'], d.get('score', 0.0)))
            else:
                coin_data.append((base, (-1, 0), 0, "WAIT", 0.0, 0.0))

        coin_data.sort(key=lambda x: abs(x[5]), reverse=True)
        self.ui_queue.put(lambda c=coin_data: self._update_coin_list_with_mtf(c))

    def _update_coin_list_with_mtf(self, coin_data):
        try:
            if not hasattr(self, 'coin_box'):
                return
            display_values = []
            self._coin_lookup = {}
            if not coin_data:
                for base in getattr(self, '_initial_coin_order', []):
                    display = f"{base:6s}  WAIT   --"
                    display_values.append(display)
                    self._coin_lookup[display] = base
            else:
                for item in coin_data:
                    base = item[0]
                    agree = item[2]
                    direction = item[3]
                    conf = item[4]
                    if direction == "WAIT" or agree == 0:
                        display = f"{base:6s}  WAIT   --"
                    else:
                        conf_val = abs(conf)
                        conf_str = f"{conf_val:.0f}%" if conf_val > 0 else "--"
                        display = f"{base:6s}  {direction:4s} {conf_str:>4s}"
                    display_values.append(display)
                    self._coin_lookup[display] = base
            self.coin_box["values"] = display_values
            current_base = getattr(self, '_current_coin_base', None)
            if current_base:
                for disp, base in self._coin_lookup.items():
                    if base == current_base:
                        self.coin_box.set(disp)
                        break
            elif display_values:
                self.coin_box.set(display_values[0])

            if hasattr(self, '_coin_tv'):
                self._coin_tv.delete(*self._coin_tv.get_children())
                for item in coin_data[:50]:
                    base      = item[0]
                    agree     = item[2]
                    direction = item[3]
                    conf      = item[4]
                    score     = item[5] if len(item) > 5 else 0.0
                    conf_str  = f"{abs(conf):.1f}%" if conf != 0 else "--"
                    score_str = f"{abs(score):.1f}" if score != 0 else "--"
                    tag = "buy" if direction == "BUY" else "sell" if direction == "SELL" else "wait"
                    self._coin_tv.insert("", "end",
                                         values=(base, direction, conf_str, score_str),
                                         tags=(tag,))
        except Exception as e:
            logging.warning("MTF list update error: %s", str(e)[:60])

    def _scan_and_sort_by_mtf_fast(self, coin_list, initial_coins):
        self._bulk_mtf_scan([base for base, _, _, _ in coin_list])

    def _bulk_mtf_scan(self, coin_bases):
        if not coin_bases:
            return
        tfs = ["5m", "15m", "1h", "4h", "1d"]
        results = {}
        total = len(coin_bases)

        quick_scores = {}
        for base in coin_bases:
            if not self.running:
                break
            sym = base + "USDT"
            try:
                limit = TF_HISTORY_MAP.get("1h", 250)
                url = f"https://fapi.binance.com/fapi/v1/klines?symbol={sym}&interval=1h&limit={limit}"
                try:
                    r = self._api_get(url, timeout=5)
                except RuntimeError:
                    quick_scores[base] = 0.0
                    continue
                if r.status_code != 200:
                    quick_scores[base] = 0.0
                    continue
                klines = r.json()
                if not klines or len(klines) < 50:
                    quick_scores[base] = 0.0
                    continue
                df = self._klines_to_df(klines)
                cp = float(df["c"].iloc[-1])
                result = self._deep_analyze(df, cp)
                quick_scores[base] = abs(result.confidence) if result.direction != "WAIT" else 0.0
            except Exception:
                quick_scores[base] = 0.0
            time.sleep(API_DELAY_SEC)

        top10 = sorted(quick_scores, key=lambda b: quick_scores[b], reverse=True)[:10]
        remaining = [b for b in coin_bases if b not in top10]

        for idx, base in enumerate(top10):
            if not self.running:
                break
            sym = base + "USDT"
            tf_results = {}
            for tf in tfs:
                try:
                    limit = TF_HISTORY_MAP.get(tf, 150)
                    url = f"https://fapi.binance.com/fapi/v1/klines?symbol={sym}&interval={tf}&limit={limit}"
                    try:
                        r = self._api_get(url, timeout=5)
                    except RuntimeError:
                        tf_results[tf] = ("WAIT", 0)
                        continue
                    if r.status_code != 200:
                        tf_results[tf] = ("WAIT", 0)
                        continue
                    klines = r.json()
                    if not klines or len(klines) < 50:
                        tf_results[tf] = ("WAIT", 0)
                        continue
                    df = self._klines_to_df(klines)
                    cp = float(df["c"].iloc[-1])
                    result = self._deep_analyze(df, cp)
                    tf_results[tf] = (result.direction, result.confidence)
                except Exception:
                    tf_results[tf] = ("WAIT", 0)
                time.sleep(API_DELAY_SEC)

            buy_count  = sum(1 for v in tf_results.values() if v[0] == "BUY")
            sell_count = sum(1 for v in tf_results.values() if v[0] == "SELL")
            if buy_count >= sell_count and buy_count > 0:
                master_dir = "BUY";  agree = buy_count
                max_conf = max((v[1] for v in tf_results.values() if v[0] == "BUY"), default=0)
            elif sell_count > buy_count and sell_count > 0:
                master_dir = "SELL"; agree = sell_count
                max_conf = max((v[1] for v in tf_results.values() if v[0] == "SELL"), default=0)
            else:
                master_dir = "WAIT"; agree = 0; max_conf = 0.0
            score = max_conf if master_dir == "BUY" else -max_conf if master_dir == "SELL" else 0.0
            results[base] = {'score': score, 'agree': agree, 'direction': master_dir, 'conf': max_conf}

            with self._mtf_data_lock:
                self._mtf_coin_data.update(results)

            progress_pct = int((idx + 1) / len(top10) * 100)
            self.ui_queue.put(lambda p=progress_pct, c=idx+1, t=len(top10): self.status_lbl.config(
                text=f"MTF Deep Scan: {c}/{t} coins ({p}%)", fg="#f0b90b"))

        for base in remaining:
            if base not in results:
                q = quick_scores.get(base, 0.0)
                results[base] = {'score': q, 'agree': 1 if q > 0 else 0,
                                 'direction': "BUY" if q > 30 else "WAIT", 'conf': q}

        with self._mtf_data_lock:
            self._mtf_coin_data.update(results)

        with self._mtf_data_lock:
            mtf_snapshot = dict(self._mtf_coin_data)

        all_data = []
        for base in getattr(self, '_initial_coin_order', []):
            if base in mtf_snapshot:
                d = mtf_snapshot[base]
                all_data.append((base, (d['agree'], abs(d['conf'])), d['agree'], d['direction'], d['conf'], d['score']))
            else:
                all_data.append((base, (-1, 0), 0, "WAIT", 0.0, 0.0))
        all_data.sort(key=lambda x: abs(x[5]), reverse=True)
        self.ui_queue.put(lambda c=all_data: self._update_coin_list_with_mtf(c))
        self.ui_queue.put(lambda: self.status_lbl.config(
            text=f"MTF AI ranked {len(results)} coins (top-10 deep scan)", fg="#02c076"))

    def _reinforcement_learning_loop(self):
        time.sleep(30)
        while self.running:
            try:
                wins = 0
                losses = 0
                if not os.path.exists(SAVED_SIGNALS_FILE):
                    time.sleep(RL_CHECK_INTERVAL)
                    continue
                with open(SAVED_SIGNALS_FILE, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        result = entry.get("result", "")
                        if result == "WIN":
                            wins += 1
                        elif result == "LOSS":
                            losses += 1
                    except Exception:
                        continue
                total = wins + losses
                if total >= 10:
                    win_rate = wins / total
                    if win_rate > RL_WIN_THRESHOLD:
                        new_boost = min(self._rl_conf_boost * 0.95, 1.2)
                        if abs(new_boost - self._rl_conf_boost) > 0.01:
                            self._rl_conf_boost = new_boost
                            logging.info("RL: Win Rate %.1f%% > 65%% → Sensitivity DOWN (boost=%.2f)", win_rate*100, new_boost)
                            self.ui_queue.put(lambda b=new_boost: self.status_lbl.config(
                                text=f"RL: WinRate {win_rate*100:.0f}% ↑ → Sensitivity relaxed (x{b:.2f})", fg="#02c076"))
                    elif win_rate < RL_LOSS_THRESHOLD:
                        new_boost = max(self._rl_conf_boost * 1.05, 0.7)
                        if abs(new_boost - self._rl_conf_boost) > 0.01:
                            self._rl_conf_boost = new_boost
                            logging.info("RL: Win Rate %.1f%% < 45%% → Sensitivity UP (boost=%.2f)", win_rate*100, new_boost)
                            self.ui_queue.put(lambda b=new_boost: self.status_lbl.config(
                                text=f"RL: WinRate {win_rate*100:.0f}% ↓ → Sensitivity tightened (x{b:.2f})", fg="#f0b90b"))
                    else:
                        logging.debug("RL: Win Rate %.1f%% (normal range, no change)", win_rate * 100)
            except Exception as e:
                logging.warning("RL loop error: %s", str(e)[:80])
            time.sleep(RL_CHECK_INTERVAL)

    def price_fallback_loop(self):
        while self.running:
            time.sleep(2)
            try:
                if time.time() - self._last_price_update > 4 and self.symbol:
                    url = f"https://fapi.binance.com/fapi/v1/premiumIndex?symbol={self.symbol}"
                    try:
                        r = self._api_get(url, timeout=2)
                    except RuntimeError:
                        continue
                    if r.status_code == 200:
                        p = float(r.json().get("markPrice", 0))
                        if p > 0:
                            self.prev_price = self.mark_price if self.mark_price > 0 else p
                            self.mark_price = p
                            self.price = p
                            self._last_price_update = time.time()
                            logging.info("Fallback REST price: %s", p)
            except Exception:
                pass

    def intel_auto_refresh_loop(self):
        time.sleep(5)
        self._intel_thread()
        while self.running:
            time.sleep(5)
            if self.running:
                self._intel_thread_fast()

    def ws_manager_loop(self):
        reconnect_delay = 2
        while self.running:
            try:
                if not self.ws_connected:
                    self.ws_status = "WS: CONNECTING"
                    self.ui_queue.put(lambda: self.ws_status_lbl.config(
                        text="WS: CONNECTING", fg="#f0b90b") if hasattr(self, 'ws_status_lbl') else None)
                    self.connect_websockets()
                    reconnect_delay = 2
                stable_cycles = 0
                while self.running and self.ws_connected:
                    time.sleep(5)
                    stable_cycles += 1
                    if stable_cycles % 6 == 0:
                        self.ws_ping()

                    if (time.time() - self._last_price_update > 45
                            and self.price > 0
                            and self._last_price_update > 0):
                        logging.warning("Price stale >45s -- reconnecting WS")
                        self.ws_connected = False
                        break
            except Exception as e:
                self.set_error("WS Mgr: " + str(e)[:60])
                self.ws_status = "WS: RECONNECTING"
                logging.warning("WS manager error, retry in %ds: %s", reconnect_delay, str(e)[:60])
                time.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 1.5, 30)

    def ws_ping(self):
        try:
            mark_ok = False
            kline_ok = False

            with self._ws_lock:
                if self.ws_mark is not None and hasattr(self.ws_mark, 'sock'):
                    try:
                        mark_ok = self.ws_mark.sock is not None
                    except Exception:
                        mark_ok = False

                if self.ws_kline is not None and hasattr(self.ws_kline, 'sock'):
                    try:
                        kline_ok = self.ws_kline.sock is not None
                    except Exception:
                        kline_ok = False

            price_fresh = (time.time() - self._last_price_update) < 35
            if (mark_ok or kline_ok) and price_fresh:
                self.ws_last_ping = time.time()
            elif not price_fresh and self._last_price_update > 0:
                logging.warning("ws_ping: price stale → marking disconnected")
                self.ws_connected = False
        except Exception as e:
            logging.warning("ws_ping error: %s", str(e)[:60])
            self.ws_connected = False

    def connect_websockets(self):
        try:
            with self._ws_lock:
                self.ws_connected = False
                self._close_ws()

            time.sleep(0.5)
            self.connect_mark_ws()
            time.sleep(0.3)
            self.connect_kline_ws()
            time.sleep(0.5)
            with self._ws_lock:
                self.ws_connected = True
                self.ws_status = "WS: LIVE"
                self.ws_last_ping = time.time()
        except Exception as e:
            self.ws_connected = False
            self.ws_status = "WS: ERROR"
            self.set_error(f"WS Connect: {str(e)[:60]}")
            raise

    def _close_ws(self):
        ws_mark = self.ws_mark
        ws_kline = self.ws_kline
        self.ws_mark = None
        self.ws_kline = None
        if ws_mark is not None:
            try:
                ws_mark.close()
            except Exception:
                pass
        if ws_kline is not None:
            try:
                ws_kline.close()
            except Exception:
                pass

    def connect_mark_ws(self):
        sym = self.symbol.lower()
        url = f"wss://fstream.binance.com/ws/{sym}@markPrice@1s"
        self.ws_mark = websocket.WebSocketApp(
            url,
            on_open=lambda ws: None,
            on_message=self.on_mark_message,
            on_error=lambda ws, e: self.set_error(f"Mark: {e}"),
            on_close=lambda ws, c, m: self.on_ws_close()
        )
        threading.Thread(
            target=lambda: self.ws_mark.run_forever(ping_interval=20, ping_timeout=10, reconnect=3),
            daemon=True
        ).start()

    def on_ws_close(self):
        self.ws_connected = False
        self.ws_status = "WS: RECONNECTING"
        logging.info("WebSocket closed -- ws_manager_loop will handle reconnect")

    def on_mark_message(self, ws, message):
        try:
            with self._ws_lock:
                if ws is not self.ws_mark and self.ws_mark is not None:
                    return
            data = json.loads(message)
            p = 0.0
            if "p" in data:
                p = float(data["p"])
            elif "markPrice" in data:
                p = float(data["markPrice"])
            else:
                return
            if p > 0:
                self.prev_price = self.mark_price if self.mark_price > 0 else p
                self.mark_price = p
                self.price = p
                self._last_price_update = time.time()
                if not self.price_history or abs(p - self.price_history[-1]) > 0.0001:
                    self.price_history.append(p)
                em = self.error_msg.lower()
                if "price" in em or "mark" in em or "stale" in em:
                    self.error_msg = ""
                logging.debug("Mark price updated: %f", p)
        except Exception as e:
            self.set_error("Mark msg: " + str(e)[:60])

    def connect_kline_ws(self):
        sym = self.symbol.lower()
        url = f"wss://fstream.binance.com/ws/{sym}@kline_{self.interval}"
        self.ws_kline = websocket.WebSocketApp(
            url,
            on_open=lambda ws: self.load_historical_klines(),
            on_message=self.on_kline_message,
            on_error=lambda ws, e: self.set_error(f"Kline: {e}"),
            on_close=lambda ws, c, m: self.on_ws_close()
        )
        threading.Thread(
            target=lambda: self.ws_kline.run_forever(ping_interval=20, ping_timeout=10, reconnect=3),
            daemon=True
        ).start()

    def on_kline_message(self, ws, message):
        try:
            with self._ws_lock:
                if ws is not self.ws_kline and self.ws_kline is not None:
                    return
            data = json.loads(message)
            k = data.get("k", {})
            if not k:
                return
            candle = Candle(
                t=k["t"],
                o=float(k["o"]),
                h=float(k["h"]),
                l=float(k["l"]),
                c=float(k["c"]),
                v=float(k["v"])
            )
            is_closed = k.get("x", False)
            with self.df_lock:
                if len(self.candle_deque) > 0 and self.candle_deque[-1].t == candle.t:
                    self.candle_deque[-1] = candle
                else:
                    self.candle_deque.append(candle)
            if candle.o > 0:
                self.candle_open_price = candle.o
            kline_price = candle.c
            if kline_price > 0:
                self.price = kline_price
                self.mark_price = kline_price
                self._last_price_update = time.time()
                if not self.price_history or abs(kline_price - self.price_history[-1]) > 0.0001:
                    self.price_history.append(kline_price)
            # v21: Only trigger analysis on CLOSED candles for accuracy
            if is_closed:
                self.new_data_event.set()
                logging.debug("Kline CLOSED: %s @ %f", self.symbol, kline_price)
        except Exception as e:
            self.set_error("Kline msg: " + str(e)[:60])

    def _klines_to_df(self, klines) -> pd.DataFrame:
        df = pd.DataFrame(klines).iloc[:, :6]
        df.columns = ["t", "o", "h", "l", "c", "v"]
        df[["o", "h", "l", "c", "v"]] = df[["o", "h", "l", "c", "v"]].astype(float)
        return df

    def _deque_to_df(self) -> pd.DataFrame:
        with self.df_lock:
            if len(self.candle_deque) == 0:
                return pd.DataFrame()
            data = [
                {"t": c.t, "o": c.o, "h": c.h, "l": c.l, "c": c.c, "v": c.v}
                for c in self.candle_deque
            ]
            return pd.DataFrame(data)

    def load_historical_klines(self):
        # Try futures first, then spot as fallback — with retries
        ENDPOINTS = [
            "https://fapi.binance.com/fapi/v1/klines",
            "https://api.binance.com/api/v3/klines",
        ]
        limit = self._get_history_limit()
        logging.info("Loading %d candles for %s @ %s", limit, self.symbol, self.interval)

        klines = None
        for attempt in range(5):
            for base_url in ENDPOINTS:
                try:
                    url = f"{base_url}?symbol={self.symbol}&interval={self.interval}&limit={limit}"
                    r = self._api_get(url, timeout=15)
                    data = r.json()
                    if isinstance(data, list) and len(data) >= 50:
                        klines = data
                        break
                    elif isinstance(data, list) and len(data) > 0:
                        logging.warning(
                            "Only %d candles from %s (attempt %d/5), retrying...",
                            len(data), base_url.split("/")[2], attempt + 1
                        )
                    else:
                        logging.warning(
                            "Bad response from %s (attempt %d/5): %s",
                            base_url.split("/")[2], attempt + 1, str(data)[:80]
                        )
                except Exception as e:
                    logging.warning("Klines fetch error (attempt %d/5): %s", attempt + 1, str(e)[:60])
            if klines:
                break
            wait = min(5 * (attempt + 1), 30)
            logging.info("Waiting %ds before retry...", wait)
            time.sleep(wait)

        if not klines:
            logging.error("Failed to load candles for %s after all retries", self.symbol)
            return

        logging.info("Loaded %d candles for %s @ %s", len(klines), self.symbol, self.interval)
        try:
            df = self._klines_to_df(klines)
            with self.df_lock:
                self.candle_deque.clear()
                for _, row in df.iterrows():
                    self.candle_deque.append(Candle(
                        t=row["t"], o=row["o"], h=row["h"],
                        l=row["l"], c=row["c"], v=row["v"]
                    ))
            last_open = float(df.iloc[-1]["o"])
            if last_open > 0:
                self.candle_open_price = last_open
            last_close = float(df.iloc[-1]["c"])
            if last_close > 0:
                self.price = last_close
                self.mark_price = last_close
                self.display_price = last_close
                self._last_price_update = time.time()
            self.new_data_event.set()
            threading.Thread(target=self._immediate_analysis, daemon=True).start()
        except Exception as e:
            self.set_error("REST parse: " + str(e)[:60])

    def _immediate_analysis(self):
        try:
            time.sleep(0.5)
            df = self._deque_to_df()
            if len(df) < 50:
                return
            cp = self.mark_price if self.mark_price > 0 else df["c"].iloc[-1]
            if cp <= 0:
                return
            result = self._deep_analyze(df, cp)
            with self.analysis_lock:
                self.current_analysis = result
            self.last_update = datetime.now().strftime("%H:%M:%S")
        except Exception as e:
            self.set_error("Immediate analysis: " + str(e)[:60])

    def load_symbols_thread(self):
        try:
            url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
            r = self._api_get(url, timeout=10)
            info = r.json()
            all_syms = [
                s for s in info["symbols"]
                if s.get("quoteAsset") == "USDT"
                and s.get("contractType") == "PERPETUAL"
                and s.get("status") == "TRADING"
            ]
            ticker_url = "https://fapi.binance.com/fapi/v1/ticker/24hr"
            tr = self._api_get(ticker_url, timeout=10)
            tickers = {t["symbol"]: t for t in tr.json()}
            sym_data = []
            for s in all_syms:
                sym = s["symbol"]
                base = s["baseAsset"]
                t = tickers.get(sym, {})
                change = float(t.get("priceChangePercent", 0.0))
                sym_data.append((base, sym, change))
            sym_data.sort(key=lambda x: x[2], reverse=True)
            self._initial_coin_order = [x[0] for x in sym_data]

            current_ticker = tickers.get(self.symbol, {})
            try:
                self._vol_24h_usdt = float(current_ticker.get("quoteVolume", 0.0))
            except Exception:
                self._vol_24h_usdt = 0.0
            self.ui_queue.put(lambda: self._set_coins(self._initial_coin_order))
            logging.info("Loaded %d coins from Binance Futures", len(sym_data))
            top_active = sorted(sym_data, key=lambda x: abs(x[2]), reverse=True)[:100]
            threading.Thread(target=self._scan_and_sort_by_mtf,
                             args=(top_active, self._initial_coin_order),
                             daemon=True).start()
        except Exception as e:
            logging.error("Failed to load symbols: %s", str(e)[:100])
            self.ui_queue.put(lambda: self._set_coins([], str(e)))

    def _scan_and_sort_by_mtf(self, coin_list, initial_coins):
        self.ui_queue.put(lambda: self.status_lbl.config(
            text="AI Scanning top coins by MTF...", fg="#f0b90b"))
        all_data = []
        for idx, base in enumerate(initial_coins):
            all_data.append((base, (-1, 0), 0, "WAIT", 0.0))
        self.ui_queue.put(lambda c=all_data: self._update_coin_list_with_mtf(c))
        top_bases = [base for base, _, _ in coin_list]
        self._bulk_mtf_scan(top_bases)
        remaining_bases = [c for c in initial_coins if c not in top_bases]
        if remaining_bases:
            self.ui_queue.put(lambda: self.status_lbl.config(
                text=f"Background MTF scan for {len(remaining_bases)} coins...", fg="#f0b90b"))
            threading.Thread(
                target=self._bulk_mtf_scan,
                args=(remaining_bases,),
                daemon=True
            ).start()
        else:
            self.ui_queue.put(lambda: self.status_lbl.config(
                text=f"AI Sorted {len(top_bases)} coins by MTF (5-TF)", fg="#02c076"))

    def _set_coins(self, coins, error=None):
        if error:
            self.status_lbl.config(text="Connection Error", fg="#ff4d4d")
            self.error_msg = str(error)[:60]
        else:
            self._initial_coin_order = coins
            self.status_lbl.config(text=f"Connected | {len(coins)} coins loaded", fg="#02c076")
            self._update_narrative_comparison_box(coins)

    def _update_narrative_comparison_box(self, coins):
        try:
            if hasattr(self, '_nar_comp_box') and coins:
                values = ["None"] + [c for c in coins[:50] if c != self.symbol.replace("USDT", "")]
                self._nar_comp_box["values"] = values
                current = self._nar_comp_box.get()
                if current not in values:
                    self._nar_comp_box.set("None")
                    self._comparison_symbol = None
        except Exception:
            pass

    def set_error(self, msg):
        self.error_msg = msg[:100]
        logging.warning("Error: %s", msg)

    def analysis_loop(self):
        self._last_analysis_time = 0.0
        while self.running:
            self.new_data_event.wait(timeout=5)
            self.new_data_event.clear()
            if not self.running:
                break
            try:
                throttle_sec = self._analysis_throttle_map.get(self.interval, 0)
                if throttle_sec > 0:
                    elapsed = time.time() - self._last_analysis_time
                    if elapsed < throttle_sec:
                        logging.debug("Analysis throttled for %s: %.0fs/%.0fs elapsed",
                                      self.interval, elapsed, throttle_sec)
                        continue
                df = self._deque_to_df()
                min_required = max(50, TF_HISTORY_MAP.get(self.interval, 300) - 100)
                if len(df) < min_required:
                    logging.warning("Analysis: %d candles (need %d for %s), waiting...", 
                                  len(df), min_required, self.interval)
                    if len(df) == 0:
                        self.load_historical_klines()
                    continue
                logging.debug("Analyzing %d candles for %s @ %s", len(df), self.symbol, self.interval)
                cp = self.price if self.price > 0 else self.mark_price if self.mark_price > 0 else df["c"].iloc[-1]
                if cp <= 0:
                    continue
                if self.price == 0 and cp > 0:
                    self.price = cp
                result = self._deep_analyze(df, cp)
                with self.analysis_lock:
                    self.current_analysis = result
                self.last_update = datetime.now().strftime("%H:%M:%S")
                self._last_analysis_time = time.time()
                if "kline" in self.error_msg.lower() or "rest" in self.error_msg.lower():
                    self.error_msg = ""
                self._process_signal(result, cp)
                self._narrative_cache = {}
                logging.info("Analysis complete: %s | Confidence: %.1f%% | Score: %.1f | Structure: %s", 
                           result.direction, result.confidence, result.score, result.swing_structure)
            except Exception as e:
                self.set_error(str(e)[:100])

    def _process_signal(self, result: AnalysisResult, cp: float):
        new_dir = result.direction
        if new_dir != self.last_signal_direction and new_dir in ("BUY", "SELL"):
            self._last_narrative_update = 0
        with self._bt_lock:
            if self._bt_open is not None:
                o = self._bt_open
                if o.get("trailing_active") and o["dir"] == "BUY" and cp > o["trailing_trigger"]:
                    new_sl = max(o["sl"], cp - o["trailing_dist"])
                    if new_sl > o["sl"]:
                        o["sl"] = new_sl
                        logging.info("Trailing SL raised to %f", new_sl)
                elif o.get("trailing_active") and o["dir"] == "SELL" and cp < o["trailing_trigger"]:
                    new_sl = min(o["sl"], cp + o["trailing_dist"])
                    if new_sl < o["sl"]:
                        o["sl"] = new_sl
                        logging.info("Trailing SL lowered to %f", new_sl)
                hit_tp = (o["dir"] == "BUY"  and cp >= o["tp"]) or (o["dir"] == "SELL" and cp <= o["tp"])
                hit_sl = (o["dir"] == "BUY"  and cp <= o["sl"]) or (o["dir"] == "SELL" and cp >= o["sl"])
                if hit_tp:
                    self._bt_wins += 1
                    self._bt_signals.append({**o, "result": "WIN",  "exit": round(cp, 6)})
                    self._save_closed_trade({**o, "result": "WIN", "exit": round(cp, 6)})
                    self._bt_open = None
                    logging.info("Backtest WIN: %s @ %f", o["dir"], cp)
                elif hit_sl:
                    self._bt_losses += 1
                    self._bt_signals.append({**o, "result": "LOSS", "exit": round(cp, 6)})
                    self._save_closed_trade({**o, "result": "LOSS", "exit": round(cp, 6)})
                    self._bt_open = None
                    logging.info("Backtest LOSS: %s @ %f", o["dir"], cp)
        if new_dir != self.last_signal_direction and new_dir in ("BUY", "SELL"):
            self.last_signal_direction = new_dir
            entry = {
                "time":       datetime.now().strftime("%H:%M:%S"),
                "symbol":     self.symbol,
                "tf":         self.interval,
                "direction":  new_dir,
                "strength":   result.strength,
                "confidence": round(result.confidence, 1),
                "price":      round(cp, 6),
                "sl":         round(result.smart_sl if result.smart_sl > 0 else result.sl, 6),
                "tp":         round(result.smart_tp  if result.smart_tp  > 0 else result.tp,  6),
                "rr":         round(result.rr, 1),
                "pos_size":   round(result.position_size_pct, 2),
                "win_est":    round(result.win_rate_est, 1),
            }
            self.signal_history.appendleft(entry)
            self.signal_stats[new_dir.lower()] += 1
            atr_val = max(result.atr, cp * 0.002)
            with self._bt_lock:
                self._bt_open = {
                    "dir":   new_dir,
                    "entry": round(cp, 6),
                    "tp":    entry["tp"],
                    "sl":    entry["sl"],
                    "time":  entry["time"],
                    "trailing_active": True,
                    "trailing_dist": atr_val * 1.5,
                    "trailing_trigger": cp + atr_val * 2.0 if new_dir == "BUY" else cp - atr_val * 2.0,
                    "tp1_hit": False,
                }

            tf_seconds = {"5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}
            candle_sec = tf_seconds.get(self.interval, 3600)
            shift_eval = _get_lookahead_shift(self.interval)
            maturity_time = time.time() + candle_sec * shift_eval
            pred_price = result.price_prediction.get("pred_price", 0.0) if result.price_prediction else 0.0
            with self._winrate_lock:
                self._winrate_predictions.append({
                    "direction":    new_dir,
                    "start_price":  round(cp, 6),
                    "pred_price":   round(pred_price, 6),
                    "maturity":     maturity_time,
                    "evaluated":    False,
                    "result":       None,
                    "symbol":       self.symbol,
                    "time":         entry["time"],
                })

            logging.info("New signal: %s %s @ %f (conf: %.1f%%)", new_dir, result.strength, cp, result.confidence)
            self.root.after(0, self.flash_alert)
        elif new_dir == "WAIT":
            self.last_signal_direction = "WAIT"

    def _save_closed_trade(self, trade_data: dict):
        try:
            entry = {
                "time":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "symbol": self.symbol,
                "tf":     self.interval,
                "dir":    trade_data.get("dir", "UNKNOWN"),
                "entry":  trade_data.get("entry", 0),
                "exit":   trade_data.get("exit", 0),
                "tp":     trade_data.get("tp", 0),
                "sl":     trade_data.get("sl", 0),
                "result": trade_data.get("result", "UNKNOWN"),
            }
            with open(SAVED_SIGNALS_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logging.warning("_save_closed_trade error: %s", str(e)[:60])

    def _generate_ai_signal_report(self, cp: float, result: 'AnalysisResult') -> str:
        try:
            pred = result.price_prediction
            if not pred:
                return "-- No prediction data --"
            if pred.get("pred_price", 0.0) <= 0:
                return f"-- Prediction price invalid (pred_price={pred.get('pred_price', 'N/A')}) --"

            ai_price   = float(pred["pred_price"])
            pred_high  = float(pred.get("pred_high", 0.0))
            pred_low   = float(pred.get("pred_low",  0.0))
            bb_entry   = float(pred.get("bb_entry",  0.0))
            bb_exit    = float(pred.get("bb_exit",   0.0))
            bb_upper   = float(pred.get("bb_upper",  0.0))
            bb_lower   = float(pred.get("bb_lower",  0.0))
            bb_mid     = float(pred.get("bb_mid",    0.0))
            xgb_pct    = float(pred.get("xgb_pct",  0.0))
            adx_regime = str(pred.get("adx_regime", "UNKNOWN"))
            adx_val    = float(pred.get("adx_val",  0.0))
            rsi_zone   = str(pred.get("rsi_zone",   "NEUTRAL"))
            pred_r2    = float(pred.get("pred_r2",  0.0))
            tf         = self.interval
            symbol     = self.symbol

            if cp <= 0:
                return ""

            expected_change = (ai_price - cp) / cp * 100
            abs_change      = abs(expected_change)
            tf_threshold    = AI_SIGNAL_THRESHOLD_PCT.get(tf, 0.60)
            min_change      = tf_threshold + AI_MIN_COMMISSION_PCT
            sl_pct          = AI_SL_PCT.get(tf, 1.00)
            sep             = "-" * 36

            if abs_change < min_change:
                lines = [
                    sep,
                    f" {symbol}  [{tf.upper()}]  v21-ULTRA",
                    sep,
                    f" Direction   : HOLD",
                    f" Current     : {cp:,.4f}",
                    f" AI Price    : {ai_price:,.4f}",
                    f" Change      : {expected_change:+.3f}%",
                    f" Min Needed  : {min_change:.2f}%",
                    sep,
                    f" AI High     : {pred_high:,.4f}",
                    f" AI Low      : {pred_low:,.4f}",
                    sep,
                    f" BB Upper    : {bb_upper:,.4f}",
                    f" BB Mid      : {bb_mid:,.4f}",
                    f" BB Lower    : {bb_lower:,.4f}",
                    sep,
                    f" ADX Regime  : {adx_regime}  (ADX={adx_val:.1f})",
                    f" RSI Zone    : {rsi_zone}",
                    f" XGB Pred    : {xgb_pct:+.3f}%  |  R²={pred_r2:.2f}",
                    sep,
                    " * Change below threshold - no trade",
                ]
                return "\n".join(lines)

            if expected_change > 0:
                direction  = "BUY  (LONG)"
                entry      = cp
                raw_tp     = ai_price * AI_TP_SAFETY_MARGIN
                min_tp     = entry * (1 + AI_MIN_COMMISSION_PCT / 100)
                tp         = round(max(raw_tp, min_tp), 6)
                sl         = round(entry * (1 - sl_pct / 100), 6)
                reward_pct = (tp - entry) / entry * 100
                risk_pct   = (entry - sl) / entry * 100
                if adx_regime == "RANGING":
                    bb_note    = f" BB Zone     : {bb_lower:,.4f} → {bb_mid:,.4f}  (support zones)" if bb_lower > 0 else ""
                    bb_exit_note = f" BB Exit     : {bb_exit:,.4f}  (BB Mid / Take Profit)" if bb_exit > 0 else ""
                else:
                    bb_note    = f" BB Zone     : {bb_lower:,.4f} → {bb_upper:,.4f}  (uptrend channels)" if bb_lower > 0 else ""
                    bb_exit_note = f" BB Exit     : {bb_exit:,.4f}  (BB Upper / Target)" if bb_exit > 0 else ""
            else:
                direction  = "SELL  (SHORT)"
                entry      = cp
                raw_tp     = ai_price * AI_TP_SAFETY_MARGIN
                max_tp     = entry * (1 - AI_MIN_COMMISSION_PCT / 100)
                tp         = round(min(raw_tp, max_tp), 6)
                sl         = round(entry * (1 + sl_pct / 100), 6)
                reward_pct = (entry - tp) / entry * 100
                risk_pct   = (sl - entry) / entry * 100
                if adx_regime == "RANGING":
                    bb_note    = f" BB Zone     : {bb_mid:,.4f} → {bb_upper:,.4f}  (resistance zones)" if bb_upper > 0 else ""
                    bb_exit_note = f" BB Exit     : {bb_exit:,.4f}  (BB Mid / Take Profit)" if bb_exit > 0 else ""
                else:
                    bb_note    = f" BB Zone     : {bb_lower:,.4f} → {bb_mid:,.4f}  (downtrend channels)" if bb_lower > 0 else ""
                    bb_exit_note = f" BB Exit     : {bb_exit:,.4f}  (BB Lower / Target)" if bb_exit > 0 else ""

            rr_ratio = reward_pct / risk_pct if risk_pct > 0 else 0.0
            quality  = "STRONG" if abs_change >= min_change * 2 else "MODERATE" if rr_ratio >= MIN_RR_RATIO else "WEAK"

            regime_warn = ""
            if adx_regime == "RANGING" and rr_ratio < MIN_RR_RATIO:
                regime_warn = " [!] RANGING MARKET — reduce risk"
            elif rsi_zone in ("OVERBOUGHT_EXTREME",) and expected_change > 0:
                regime_warn = " [!] RSI OVERBOUGHT — reversal risk"
            elif rsi_zone in ("OVERSOLD_EXTREME",) and expected_change < 0:
                regime_warn = " [!] RSI OVERSOLD — reversal risk"

            lines = [
                sep,
                f" {symbol}  [{tf.upper()}]  v21-ULTRA",
                sep,
                f" Direction   : {direction}",
                f" Quality     : {quality}{regime_warn}",
                f" 3C Forecast : {pred.get('pred_class', 'N/A')}",
                sep,
                f" ENTRY       : {entry:,.4f}",
                f" TP          : {tp:,.4f}  ({reward_pct:+.2f}%)",
                f" SL          : {sl:,.4f}  (-{risk_pct:.2f}%)",
                f" R:R         : 1 : {rr_ratio:.2f}",
                sep,
                f" AI Price    : {ai_price:,.4f}  ({expected_change:+.3f}%)",
                f" AI High     : {pred_high:,.4f}",
                f" AI Low      : {pred_low:,.4f}",
                sep,
            ]
            if bb_note:    lines.append(bb_note)
            if bb_exit_note: lines.append(bb_exit_note)
            lines += [
                f" BB Upper    : {bb_upper:,.4f}",
                f" BB Mid      : {bb_mid:,.4f}",
                f" BB Lower    : {bb_lower:,.4f}",
                sep,
                f" ADX Regime  : {adx_regime}  (ADX={adx_val:.1f})",
                f" RSI Zone    : {rsi_zone}",
                f" XGB Pred    : {xgb_pct:+.3f}%  |  R²={pred_r2:.2f}",
                f" Threshold   : {abs_change:.3f}% / {min_change:.2f}%",
                sep,
            ]

            try:
                df_snap = self._deque_to_df()
                if len(df_snap) >= 3:
                    cpats = self._detect_candle_patterns(df_snap)
                    pat_flags = []
                    if cpats.get("hammer"):        pat_flags.append("Hammer ▲")
                    if cpats.get("shooting_star"): pat_flags.append("Shooting Star ▼")
                    if cpats.get("bull_engulf"):   pat_flags.append("Bull Engulf ▲")
                    if cpats.get("bear_engulf"):   pat_flags.append("Bear Engulf ▼")
                    candle_line = " | ".join(pat_flags) if pat_flags else "--"
                    lines.append(f" Candle Pats : {candle_line}")
                    lines.append(sep)
            except Exception:
                pass
            report = "\n".join(lines)
            logging.info("AI SIGNAL v21-ULTRA: %s %s Entry=%.4f TP=%.4f SL=%.4f RR=1:%.2f ADX=%s RSI=%s XGB=%.3f%%",
                         symbol, direction, entry, tp, sl, rr_ratio, adx_regime, rsi_zone, xgb_pct)
            return report

        except Exception as e:
            logging.warning("_generate_ai_signal_report v21 error: %s", str(e)[:80])
            return ""

    def _compute_confidence_rate(self, r: 'AnalysisResult', direction: str) -> float:
        if direction not in ("BUY", "SELL"):
            return 15.0

        # v21: Reduced RSI weight, increased structure weight
        rsi_score = 0.0
        if direction == "BUY":
            if 45 <= r.rsi <= 65:     rsi_score = 20.0 * WEIGHT_CONFIG["rsi"]
            elif r.rsi > 65:          rsi_score = 10.0 * WEIGHT_CONFIG["rsi"]
            elif 35 <= r.rsi < 45:    rsi_score = 15.0 * WEIGHT_CONFIG["rsi"]
            else:                     rsi_score = 5.0 * WEIGHT_CONFIG["rsi"]
        else:
            if 35 <= r.rsi <= 55:     rsi_score = 20.0 * WEIGHT_CONFIG["rsi"]
            elif r.rsi < 35:          rsi_score = 10.0 * WEIGHT_CONFIG["rsi"]
            elif 55 < r.rsi <= 65:    rsi_score = 15.0 * WEIGHT_CONFIG["rsi"]
            else:                     rsi_score = 5.0 * WEIGHT_CONFIG["rsi"]

        if r.rsi_divergence == "BULLISH" and direction == "BUY":   rsi_score = min(rsi_score + 5, 20)
        elif r.rsi_divergence == "BEARISH" and direction == "SELL": rsi_score = min(rsi_score + 5, 20)

        bb_score = 0.0
        bb_pct = r.bb_pct_b if r.bb_pct_b != 0.5 else r.bb_pct
        bb_bw  = r.bb_bandwidth
        if bb_bw > 0.005:
            if direction == "BUY":
                if bb_pct <= 0.25:   bb_score = 20.0 * WEIGHT_CONFIG["price_action"]
                elif bb_pct <= 0.50: bb_score = 15.0 * WEIGHT_CONFIG["price_action"]
                elif bb_pct <= 0.75: bb_score = 8.0 * WEIGHT_CONFIG["price_action"]
                else:                bb_score = 3.0 * WEIGHT_CONFIG["price_action"]
            else:
                if bb_pct >= 0.75:   bb_score = 20.0 * WEIGHT_CONFIG["price_action"]
                elif bb_pct >= 0.50: bb_score = 15.0 * WEIGHT_CONFIG["price_action"]
                elif bb_pct >= 0.25: bb_score = 8.0 * WEIGHT_CONFIG["price_action"]
                else:                bb_score = 3.0 * WEIGHT_CONFIG["price_action"]
        else:
            bb_score = 5.0 * WEIGHT_CONFIG["price_action"]

        ema_score = 0.0
        if direction == "BUY":
            if r.ema20 > r.ema50 > r.ema200:  ema_score += 25.0 * WEIGHT_CONFIG["ema_trend"]
            elif r.ema20 > r.ema50:            ema_score += 15.0 * WEIGHT_CONFIG["ema_trend"]
            if r.ema20 > r.ema200:             ema_score += 10.0 * WEIGHT_CONFIG["ema_trend"]
            if r.supertrend_dir == "BUY":      ema_score += 8.0 * WEIGHT_CONFIG["supertrend"]
        else:
            if r.ema20 < r.ema50 < r.ema200:  ema_score += 25.0 * WEIGHT_CONFIG["ema_trend"]
            elif r.ema20 < r.ema50:            ema_score += 15.0 * WEIGHT_CONFIG["ema_trend"]
            if r.ema20 < r.ema200:             ema_score += 10.0 * WEIGHT_CONFIG["ema_trend"]
            if r.supertrend_dir == "SELL":     ema_score += 8.0 * WEIGHT_CONFIG["supertrend"]
        ema_score = min(ema_score, 35.0)

        raw = rsi_score + bb_score + ema_score
        boosted = raw * getattr(self, '_rl_conf_boost', 1.0)
        return round(min(max(boosted, 5.0), 100.0), 1)

    def analyze_market_structure(self, df: pd.DataFrame, lookback: int = 100) -> dict:
        """
        v21 ULTRA: Market Structure Analysis - THE FOUNDATION
        تحليل هيكل السوق لتحديد الاتجاه العام والقيعان/القمم التاريخية القريبة.
        """
        result = {
            "trend": "NEUTRAL",
            "swing_highs": [],
            "swing_lows": [],
            "near_swing_high": False,
            "near_swing_low": False,
            "last_swing_high": 0.0,
            "last_swing_low": 0.0,
            "prev_swing_high": 0.0,
            "prev_swing_low": 0.0,
            "hh_count": 0,
            "hl_count": 0,
            "lh_count": 0,
            "ll_count": 0,
        }
        if len(df) < lookback:
            lookback = len(df)
        if lookback < 20:
            return result

        recent = df.tail(lookback)
        highs = recent["h"].values
        lows = recent["l"].values
        closes = recent["c"].values
        n = len(highs)

        # تحديد Swing Highs و Swing Lows (نافذة 3 شموع)
        window = 3
        swing_highs = []
        swing_lows = []
        for i in range(window, n - window):
            if highs[i] == max(highs[i - window:i + window + 1]):
                swing_highs.append((i, float(highs[i])))
            if lows[i] == min(lows[i - window:i + window + 1]):
                swing_lows.append((i, float(lows[i])))

        result["swing_highs"] = swing_highs
        result["swing_lows"] = swing_lows

        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return result

        # آخر قمتين وقيعان
        last_sh = swing_highs[-1][1]
        prev_sh = swing_highs[-2][1]
        last_sl = swing_lows[-1][1]
        prev_sl = swing_lows[-2][1]

        result["last_swing_high"] = last_sh
        result["last_swing_low"] = last_sl
        result["prev_swing_high"] = prev_sh
        result["prev_swing_low"] = prev_sl

        # تصنيف الاتجاه
        hh = last_sh > prev_sh   # Higher High
        hl = last_sl > prev_sl   # Higher Low
        lh = last_sh < prev_sh   # Lower High
        ll = last_sl < prev_sl   # Lower Low

        result["hh_count"] = sum(1 for i in range(1, min(len(swing_highs), 6)) if swing_highs[-i][1] > swing_highs[-(i+1)][1])
        result["hl_count"] = sum(1 for i in range(1, min(len(swing_lows), 6)) if swing_lows[-i][1] > swing_lows[-(i+1)][1])
        result["lh_count"] = sum(1 for i in range(1, min(len(swing_highs), 6)) if swing_highs[-i][1] < swing_highs[-(i+1)][1])
        result["ll_count"] = sum(1 for i in range(1, min(len(swing_lows), 6)) if swing_lows[-i][1] < swing_lows[-(i+1)][1])

        if hh and hl:
            result["trend"] = "BULLISH_TREND"
        elif lh and ll:
            result["trend"] = "BEARISH_TREND"
        elif hh and ll:
            result["trend"] = "DISTRIBUTION"
        elif lh and hl:
            result["trend"] = "ACCUMULATION"

        # هل السعر الحالي قريب من قمة أو قاع تاريخي؟
        cp = float(closes[-1])
        atr = max(cp * 0.005, (highs[-1] - lows[-1]) if len(highs) > 0 else cp * 0.01)

        # قرب من قمة تاريخية (آخر 5 قمم)
        recent_highs = sorted([sh[1] for sh in swing_highs[-5:]], reverse=True)
        if recent_highs and abs(cp - recent_highs[0]) / cp < 0.015:
            result["near_swing_high"] = True

        # قرب من قاع تاريخي (آخر 5 قيعان)
        recent_lows = sorted([sl[1] for sl in swing_lows[-5:]])
        if recent_lows and abs(cp - recent_lows[0]) / cp < 0.015:
            result["near_swing_low"] = True

        return result

    def _deep_analyze(self, df: pd.DataFrame, cp: float) -> AnalysisResult:
        if len(df) < 50 or cp <= 0:
            logging.warning("_deep_analyze: insufficient data (len=%d, cp=%f)", len(df), cp)
            return AnalysisResult()

        for col in ["o", "h", "l", "c", "v"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(np.float64)
        cp = float(cp)

        df_ind = self.compute_indicators(df.copy())
        last = df_ind.iloc[-1]

        result = AnalysisResult()
        result.ema20 = round(float(last["ema20"]), 6) if pd.notna(last.get("ema20")) else round(float(cp), 6)
        result.ema50 = round(float(last["ema50"]), 6) if pd.notna(last.get("ema50")) else round(float(cp), 6)
        result.ema200 = round(float(last["ema200"]), 6) if pd.notna(last.get("ema200")) else round(float(cp), 6)
        result.rsi = round(float(last["rsi"]), 2) if pd.notna(last.get("rsi")) else 50.0
        result.adx = round(float(last["adx"]), 2) if pd.notna(last.get("adx")) else 0.0
        result.di_plus = round(float(last["di_plus"]), 2) if pd.notna(last.get("di_plus")) else 0.0
        result.di_minus = round(float(last["di_minus"]), 2) if pd.notna(last.get("di_minus")) else 0.0
        result.atr = round(float(last["atr"]), 6) if pd.notna(last.get("atr")) else round(cp * 0.002, 6)
        result.bb_upper = round(float(last["bb_upper"]), 6) if pd.notna(last.get("bb_upper")) else round(cp * 1.02, 6)
        result.bb_lower = round(float(last["bb_lower"]), 6) if pd.notna(last.get("bb_lower")) else round(cp * 0.98, 6)
        result.bb_mid = round(float(last["bb_mid"]), 6) if pd.notna(last.get("bb_mid")) else round(cp, 6)
        result.bb_pct = round(float(last["bb_pct"]), 4) if pd.notna(last.get("bb_pct")) else 0.5
        result.bb_bandwidth = round(float(last["bb_bandwidth"]), 4) if pd.notna(last.get("bb_bandwidth")) else 0.0
        result.bb_pct_b = round(float(last["bb_pct_b"]), 4) if pd.notna(last.get("bb_pct_b")) else 0.5
        result.bb_squeeze = bool(last["bb_squeeze"]) if pd.notna(last.get("bb_squeeze")) else False
        result.vol_ratio = round(float(last.get("vol_ratio", 100.0)), 1) if pd.notna(last.get("vol_ratio")) else 100.0
        result.vol_delta = round(float(last.get("vol_delta", 0)), 2) if pd.notna(last.get("vol_delta")) else 0.0
        result.sar = round(float(last["sar"]), 6) if pd.notna(last.get("sar")) else round(cp, 6)
        result.supertrend = round(float(last["supertrend"]), 6) if pd.notna(last.get("supertrend")) else round(cp, 6)
        result.supertrend_dir = str(last["supertrend_dir"]) if pd.notna(last.get("supertrend_dir")) else "NONE"
        result.kdj_k = round(float(last["kdj_k"]), 2) if pd.notna(last.get("kdj_k")) else 50.0
        result.kdj_d = round(float(last["kdj_d"]), 2) if pd.notna(last.get("kdj_d")) else 50.0
        result.kdj_j = round(float(last["kdj_j"]), 2) if pd.notna(last.get("kdj_j")) else 50.0
        result.ob_imbalance = self._ob_imbalance

        candle_body = abs(last["c"] - last["o"])
        candle_range = max(last["h"] - last["l"], 0.000001)
        result.candle_strength = round((candle_body / candle_range) * 100, 1)
        result.candle_open_price = float(last["o"]) if last["o"] > 0 else 0.0
        if last["c"] > last["o"]:
            result.buy_pressure = result.candle_strength
        else:
            result.sell_pressure = result.candle_strength

        result.m_pattern, _ = self.detect_m_pattern(df_ind)
        result.w_pattern, _ = self.detect_w_pattern(df_ind)
        result.double_top, _dt = self.detect_double_top(df_ind)
        result.double_bottom, _db = self.detect_double_bottom(df_ind)
        result.rising_wedge, _rw = self.detect_rising_wedge(df_ind)
        result.falling_wedge, _fw = self.detect_falling_wedge(df_ind)
        result.flag_bull, _fb = self.detect_flag_pennant(df_ind, "bull")
        result.flag_bear, _fbe = self.detect_flag_pennant(df_ind, "bear")
        result.pennant, _pn = self.detect_flag_pennant(df_ind, "pennant")
        result.rectangle, _rc = self.detect_rectangle(df_ind)
        result.head_shoulders, _hs = self.detect_head_and_shoulders(df_ind)
        result.inv_head_shoulders, _ihs = self.detect_head_and_shoulders(df_ind, inverse=True)

        pat_parts = []
        if result.double_top: pat_parts.append("DOUBLE TOP v")
        if result.double_bottom: pat_parts.append("DOUBLE BOTTOM ^")
        if result.m_pattern: pat_parts.append("M-PATTERN v")
        if result.w_pattern: pat_parts.append("W-PATTERN ^")
        if result.rising_wedge: pat_parts.append("RISING WEDGE v")
        if result.falling_wedge: pat_parts.append("FALLING WEDGE ^")
        if result.flag_bull: pat_parts.append("BULL FLAG ^")
        if result.flag_bear: pat_parts.append("BEAR FLAG v")
        if result.pennant: pat_parts.append("PENNANT !")
        if result.rectangle: pat_parts.append("RECTANGLE ~")
        if result.head_shoulders: pat_parts.append("H&S v")
        if result.inv_head_shoulders: pat_parts.append("INV H&S ^")
        result.pattern_label = " | ".join(pat_parts) if pat_parts else "No Pattern"
        result.price_action_bias = self.compute_price_action_bias(df_ind, cp)
        result.market_regime = self._detect_market_regime(df_ind)
        result.rsi_divergence = self._detect_rsi_divergence(df_ind, self.interval)
        result.rsi_micro_divergence = self._detect_micro_divergence(df_ind, self.interval)
        result.swing_low = last["swing_low_20"] if pd.notna(last.get("swing_low_20")) else last["swing_low_10"]
        result.swing_high = last["swing_high_20"] if pd.notna(last.get("swing_high_20")) else last["swing_high_10"]
        result.fib_levels = self.compute_fibonacci(result.swing_low, result.swing_high)

        swing_struct = self._detect_swing_structure(df_ind)
        result.swing_structure      = swing_struct["structure"]
        result.swing_top2_highs     = swing_struct["top2_highs"]
        result.swing_top2_lows      = swing_struct["top2_lows"]
        result.swing_struct_detail  = swing_struct["detail"]
        result.swing_struct_confirmed = swing_struct["confirmed"]

        # v21: Market Structure Analysis - THE FOUNDATION
        market_struct = self.analyze_market_structure(df_ind, lookback=100)
        result.market_structure = market_struct

        # v21: Store structure alignment status
        is_bullish_trend = market_struct.get("trend") == "BULLISH_TREND"
        is_bearish_trend = market_struct.get("trend") == "BEARISH_TREND"
        is_bounce_from_low = market_struct.get("near_swing_low", False)
        is_near_swing_high = market_struct.get("near_swing_high", False)

        # v21: Volume confirmation check
        result.volume_confirmed = result.vol_ratio >= 100

        # v21: Structure alignment check
        result.structure_aligned = False
        if is_bullish_trend and result.ema20 > result.ema50:
            result.structure_aligned = True
        elif is_bearish_trend and result.ema20 < result.ema50:
            result.structure_aligned = True

        direction, strength, entry_low, entry_high, target, stop, confidence, reason, confluence =             self.compute_directional_forecast(cp, result, swing_struct, market_struct)

        result.direction  = direction
        result.strength   = strength
        result.entry_low  = entry_low
        result.entry_high = entry_high
        result.tp         = target
        result.sl         = stop
        result.confidence = confidence
        result.reason     = reason
        result.confluence = confluence

        if direction in ("BUY", "SELL"):
            conf_engine = self._compute_confidence_rate(result, direction)
            result.confidence = round(min(0.60 * confidence + 0.40 * conf_engine, 97.0), 1)

        # v21: Momentum decay check (reduced penalty if structure is strong)
        if result.direction == "BUY":
            try:
                closes_arr = df_ind["c"].values
                n_c = len(closes_arr)
                if n_c >= 6:
                    roc_now  = (closes_arr[-1] - closes_arr[-4]) / closes_arr[-4] if closes_arr[-4] > 0 else 0.0
                    roc_prev = (closes_arr[-4] - closes_arr[-7]) / closes_arr[-7] if n_c >= 7 and closes_arr[-7] > 0 else roc_now
                    if roc_now > 0 and roc_prev > 0 and roc_now < roc_prev * 0.50:
                        # v21: Only reduce confidence, don't block if structure is strong
                        if not result.structure_aligned:
                            result.confidence = round(max(result.confidence * 0.80, 15.0), 1)
                            result.reason += f" | RULE5-WARN: MomDecay {roc_prev*100:.2f}%→{roc_now*100:.2f}% (upward slowing)"
                            logging.debug("RULE5: Momentum decay detected — BUY confidence reduced to %.1f", result.confidence)
                            if roc_now < roc_prev * 0.25 and result.rsi >= 65:
                                result.direction = "WAIT"
                                result.strength  = "NEUTRAL"
                                result.reason   += " | RULE5-BLOCK: Severe MomDecay at RSI peak — no buy"
                                logging.info("RULE5: BUY blocked — severe momentum decay at RSI %.1f", result.rsi)
            except Exception:
                pass

        result.score = sum([
            20 if result.ema20 > result.ema50 else -20,
            15 if result.rsi >= 65 else -15 if result.rsi <= 35 else 0,
            10 if cp > result.ema200 else -10,
            15 if result.adx >= 25 and result.di_plus > result.di_minus else -15 if result.adx >= 25 and result.di_minus > result.di_plus else 0,
            10 if result.direction == "BUY" else -10 if result.direction == "SELL" else 0,
            8 if result.supertrend_dir == "BUY" else -8 if result.supertrend_dir == "SELL" else 0,
            5 if result.kdj_j > 80 and result.kdj_k > result.kdj_d else -5 if result.kdj_j < 20 and result.kdj_k < result.kdj_d else 0,
        ])

        atr_val = max(result.atr, cp * 0.002)
        with self.intel_lock:
            sr = list(self.intel_sr_levels)
        if direction == "BUY":
            supports = [lvl["price"] for lvl in sr if lvl["type"] == "S" and lvl["price"] < cp]
            if supports:
                nearest_support = max(supports)
                result.smart_sl = round(nearest_support - atr_val * 0.3, 6)
            else:
                result.smart_sl = round(cp - atr_val * 1.8, 6)
            resistances = [lvl["price"] for lvl in sr if lvl["type"] == "R" and lvl["price"] > cp]
            result.smart_tp = round(min(resistances), 6) if resistances else round(cp + atr_val * 3.5, 6)
        elif direction == "SELL":
            resistances = [lvl["price"] for lvl in sr if lvl["type"] == "R" and lvl["price"] > cp]
            if resistances:
                nearest_res = min(resistances)
                result.smart_sl = round(nearest_res + atr_val * 0.3, 6)
            else:
                result.smart_sl = round(cp + atr_val * 1.8, 6)
            supports = [lvl["price"] for lvl in sr if lvl["type"] == "S" and lvl["price"] < cp]
            result.smart_tp = round(max(supports), 6) if supports else round(cp - atr_val * 3.5, 6)
        else:
            result.smart_sl = result.sl
            result.smart_tp = result.tp

        smart_levels = self._compute_smart_entry_exit(cp, result, direction)
        if smart_levels["smart_entry_low"] > 0:
            result.smart_entry_low  = smart_levels["smart_entry_low"]
            result.smart_entry_high = smart_levels["smart_entry_high"]
        else:
            result.smart_entry_low  = result.entry_low
            result.smart_entry_high = result.entry_high
        result.smart_tp1     = smart_levels["smart_tp1"]
        result.smart_tp2     = smart_levels["smart_tp2"]
        result.smart_tp3     = smart_levels["smart_tp3"]
        result.entry_method  = smart_levels["entry_method"]
        if smart_levels["smart_sl"] > 0 and direction in ("BUY", "SELL"):
            result.smart_sl = smart_levels["smart_sl"]

        risk_pct = 0.01
        sl_distance = abs(cp - result.smart_sl) if result.smart_sl > 0 else atr_val * 1.8
        if sl_distance > 0 and cp > 0:
            raw_pos = (risk_pct / (sl_distance / cp)) * 100
            conf_multiplier = 0.7 + (confidence / 100) * 0.6
            result.position_size_pct = round(min(raw_pos * conf_multiplier, 5.0), 2)
        else:
            result.position_size_pct = 1.0

        if confluence >= 10:
            result.win_rate_est = round(min(50 + (confluence - 10) * 1.5 + (confidence / 100) * 10, 65), 1)
        elif confluence >= 6:
            result.win_rate_est = round(42 + (confluence - 6) * 1.5 + (confidence / 100) * 8, 1)
        else:
            result.win_rate_est = round(35 + confluence * 1.2, 1)

        risk = abs(cp - stop)
        reward = abs(target - cp)
        result.rr = round(reward / risk, 1) if risk > 0 else 0.0

        try:
            current_candle_t = int(df_ind["t"].iloc[-1]) if "t" in df_ind.columns else 0
            cache_key = (current_candle_t, result.direction)
            if cache_key != getattr(self, '_ai_pred_cache_key', None) or not self._ai_pred_cache:
                self._ai_pred_cache = self._compute_price_prediction(
                    df_ind, cp, result.atr, forecast_direction=result.direction)
                self._ai_pred_cache_key = cache_key
                self._ai_pred_last_candle_t = current_candle_t
            result.price_prediction = self._ai_pred_cache
        except Exception:
            result.price_prediction = {}

        try:
            ai_report_text = self._generate_ai_signal_report(cp, result)
            result.ai_report_text = ai_report_text
        except Exception as e:
            logging.warning("AI signal report error: %s", str(e)[:60])
            result.ai_report_text = ""

        try:
            result.time_to_target = self._estimate_time_to_target(df_ind, cp, target, result.atr)
        except Exception:
            result.time_to_target = 0

        # v21: Bearish candle pattern block (only if not strong structure)
        if result.direction == "BUY":
            try:
                _cpats_15m = self._detect_candle_patterns(df_ind)
                has_shooting_star = bool(_cpats_15m.get("shooting_star", 0))
                has_bear_engulf   = bool(_cpats_15m.get("bear_engulf", 0))
                _o = df_ind["o"].values; _h = df_ind["h"].values
                _l = df_ind["l"].values; _c = df_ind["c"].values
                _i = len(_c) - 1
                _body   = abs(_c[_i] - _o[_i])
                _range  = max(_h[_i] - _l[_i], 1e-9)
                _lo_wick = min(_c[_i], _o[_i]) - _l[_i]
                has_hanging_man = (
                    _lo_wick >= 2 * _body
                    and (_h[_i] - max(_c[_i], _o[_i])) <= 0.1 * _range
                    and _body / _range >= 0.1
                    and cp > result.ema20
                )
                # v21: Only block if structure is not strongly bullish
                if (has_shooting_star or has_bear_engulf or has_hanging_man) and not result.structure_aligned:
                    pat_name = ("Shooting Star" if has_shooting_star
                                else "Bearish Engulfing" if has_bear_engulf
                                else "Hanging Man")
                    result.direction = "WAIT"
                    result.strength  = "NEUTRAL"
                    result.reason   += f" | RULE7-BLOCK: {pat_name} (bearish reversal candle, weak structure)"
                    logging.info("RULE7: BUY blocked — bearish candle pattern: %s", pat_name)
                    self.ui_queue.put(lambda pn=pat_name: self.status_lbl.config(
                        text=f"Candle Block: {pn} — BUY cancelled", fg="#ff4d4d"))
            except Exception:
                pass

        # v21: Bearish divergence block (only if structure weak)
        if result.direction == "BUY" and result.rsi_divergence == "BEARISH" and not result.structure_aligned:
            result.direction = "WAIT"
            result.strength  = "NEUTRAL"
            result.reason   += " | RULE6-BLOCK: Bearish RSI Divergence (price HH, RSI LH — fake rally, no structure)"
            logging.info("RULE6: BUY blocked — Bearish RSI Divergence detected, weak structure")
            self.ui_queue.put(lambda: self.status_lbl.config(
                text="Bearish Divergence: BUY blocked", fg="#ff4d4d"))

        # v21: STRICT Trend-First Rules
        # BUY only if: Price > EMA200, HH/HL exists, Volume > average
        # SELL only if: Price < EMA200, LH/LL exists, clear support break
        if result.direction == "BUY":
            # RULE1: Price MUST be above EMA200 for BUY
            if cp <= result.ema200:
                result.direction = "WAIT"
                result.strength = "NEUTRAL"
                result.reason += " | RULE1-BLOCK: Price <= EMA200 (no bull trend foundation)"
                logging.info("RULE1: BUY blocked — Price %.4f <= EMA200 %.4f", cp, result.ema200)
            # RULE2: Must have HH/HL structure
            elif not is_bullish_trend and not is_bounce_from_low:
                result.direction = "WAIT"
                result.strength = "NEUTRAL"
                result.reason += " | RULE2-BLOCK: No HH/HL structure detected"
                logging.info("RULE2: BUY blocked — No bullish market structure")
            # RULE3: Volume must confirm
            elif not result.volume_confirmed:
                result.direction = "WAIT"
                result.strength = "NEUTRAL"
                result.reason += f" | RULE3-BLOCK: Volume {result.vol_ratio:.0f}% < 100% (no confirmation)"
                logging.info("RULE3: BUY blocked — Volume too low %.0f%%", result.vol_ratio)

        elif result.direction == "SELL":
            # RULE1: Price MUST be below EMA200 for SELL
            if cp >= result.ema200:
                result.direction = "WAIT"
                result.strength = "NEUTRAL"
                result.reason += " | RULE1-BLOCK: Price >= EMA200 (no bear trend foundation)"
                logging.info("RULE1: SELL blocked — Price %.4f >= EMA200 %.4f", cp, result.ema200)
            # RULE2: Must have LH/LL structure
            elif not is_bearish_trend:
                result.direction = "WAIT"
                result.strength = "NEUTRAL"
                result.reason += " | RULE2-BLOCK: No LH/LL structure detected"
                logging.info("RULE2: SELL blocked — No bearish market structure")
            # RULE3: Volume must confirm
            elif not result.volume_confirmed:
                result.direction = "WAIT"
                result.strength = "NEUTRAL"
                result.reason += f" | RULE3-BLOCK: Volume {result.vol_ratio:.0f}% < 100% (no confirmation)"
                logging.info("RULE3: SELL blocked — Volume too low %.0f%%", result.vol_ratio)

        # v21: Range-bound market block
        if result.market_regime == "RANGING" and result.bb_squeeze:
            atr_pct = (result.atr / cp) * 100 if cp > 0 else 999
            if atr_pct < 0.5:
                result.direction = "WAIT"
                result.strength = "NEUTRAL"
                result.reason += f" | RANGE-BLOCK: ATR {atr_pct:.2f}% (range-bound, no trade)"
                logging.info("RANGE: Blocked — ATR too low %.2f%%", atr_pct)

        # v21: R:R Filter - STRICT
        if result.direction in ("BUY", "SELL") and result.rr < MIN_RR_RATIO:
            result.direction = "WAIT"
            result.strength = "NEUTRAL"
            result.reason += f" | RR-BLOCK: R:R 1:{result.rr:.1f} < 1:{MIN_RR_RATIO} (poor reward)"
            logging.info("RR-BLOCK: Blocked — R:R 1:%.1f < 1:%.1f", result.rr, MIN_RR_RATIO)

        # Session filter disabled — trading 24/7
        result.session_valid = True
        result.session_msg = ""

        # v21: Hierarchical TF Filter
        if result.direction in ("BUY", "SELL"):
            # If current TF is 5M or 15M, check higher TFs
            if self.interval in ("5m", "15m", "3m", "1m"):
                higher_trend = self._higher_tf_trend
                if result.direction == "BUY" and higher_trend == "SELL":
                    result.direction = "WAIT"
                    result.strength = "NEUTRAL"
                    result.reason += " | HIERARCHY-BLOCK: 1H trend is BEARISH vs BUY signal"
                    logging.info("HIERARCHY: BUY blocked — 1H trend is BEARISH")
                elif result.direction == "SELL" and higher_trend == "BUY":
                    result.direction = "WAIT"
                    result.strength = "NEUTRAL"
                    result.reason += " | HIERARCHY-BLOCK: 1H trend is BULLISH vs SELL signal"
                    logging.info("HIERARCHY: SELL blocked — 1H trend is BULLISH")
                # Check 15M confirmation for 5M entries
                elif self.interval == "5m" and self._mid_tf_confirm != result.direction and self._mid_tf_confirm != "WAIT":
                    result.direction = "WAIT"
                    result.strength = "NEUTRAL"
                    result.reason += f" | HIERARCHY-BLOCK: 15M is {self._mid_tf_confirm} vs {result.direction}"
                    logging.info("HIERARCHY: 5M blocked — 15M disagrees")

        # v21: Indicator Conflict Detection (RSI vs MACD)
        if result.direction in ("BUY", "SELL"):
            macd_bull = result.ema20 > result.ema50  # Simplified MACD proxy
            rsi_bull = result.rsi > 50
            if macd_bull != rsi_bull and result.adx < 25:
                # Strong conflict in weak trend
                result.indicator_conflict = True
                result.direction = "WAIT"
                result.strength = "NEUTRAL"
                result.reason += " | CONFLICT-BLOCK: RSI vs Trend conflict in weak market"
                logging.info("CONFLICT: Blocked — RSI=%.1f vs Trend direction mismatch, ADX=%.1f", result.rsi, result.adx)
                self._conflict_count += 1
                self._last_conflict_reason = "RSI_TREND_CONFLICT"

        # v21: Final quality gate - minimum 70% weighted agreement
        if result.direction in ("BUY", "SELL"):
            weighted_agreement = result.confluence / 10.0  # Normalized
            if weighted_agreement < 0.70:
                result.direction = "WAIT"
                result.strength = "NEUTRAL"
                result.reason += f" | QUALITY-BLOCK: Agreement {weighted_agreement*100:.0f}% < 70%"
                logging.info("QUALITY: Blocked — Agreement only %.0f%%", weighted_agreement*100)

        return result

    def _scalping_filter(self, cp: float, r: AnalysisResult, tf: str) -> Tuple[bool, str]:
        if tf not in ("5m", "15m"):
            return True, ""
        reasons = []
        is_valid = True
        atr_pct = (r.atr / cp) * 100 if cp > 0 else 0
        if atr_pct > 3.0:
            is_valid = False; reasons.append(f"ATR too high ({atr_pct:.2f}%)")
        elif atr_pct < 0.02:
            is_valid = False; reasons.append(f"ATR too low ({atr_pct:.2f}%)")
        if r.vol_ratio < 60:
            is_valid = False; reasons.append(f"Volume too low ({r.vol_ratio:.0f}%)")
        ema_spread = abs(r.ema20 - r.ema50) / cp * 100 if cp > 0 else 0
        if ema_spread < 0.01:
            is_valid = False; reasons.append(f"EMA spread too tight ({ema_spread:.3f}%)")
        if r.rr < MIN_RR_RATIO:
            is_valid = False; reasons.append(f"R:R too low (1:{r.rr:.1f} < 1:{MIN_RR_RATIO})")
        msg = " | ".join(reasons) if reasons else "Scalping filter passed"
        return is_valid, msg

    def detect_m_pattern(self, df, tolerance=0.005):
        highs = df["h"].values
        n = len(highs)
        if n < 15:
            return False, None
        for i in range(3, n - 8):
            peak1 = highs[i]
            if highs[i] <= highs[i-1] or highs[i] <= highs[i+1]:
                continue
            for j in range(i + 4, min(i + 18, n - 3)):
                peak2 = highs[j]
                if highs[j] <= highs[j-1] or highs[j] <= highs[j+1]:
                    continue
                if abs(peak1 - peak2) / max(peak1, 1e-9) < tolerance:
                    valley = min(highs[i+1:j])
                    if peak1 > valley and peak2 > valley:
                        if j + 2 < n:
                            if highs[j+1] < valley or highs[j+2] < valley or df["c"].iloc[j+1] < valley:
                                return True, {"peak1": round(float(peak1), 6), "peak2": round(float(peak2), 6), "neckline": round(float(valley), 6), "peak1_idx": int(i), "peak2_idx": int(j)}
        return False, None

    def detect_w_pattern(self, df, tolerance=0.005):
        lows = df["l"].values
        highs = df["h"].values
        n = len(lows)
        if n < 15:
            return False, None
        for i in range(3, n - 8):
            bottom1 = lows[i]
            if lows[i] >= lows[i-1] or lows[i] >= lows[i+1]:
                continue
            for j in range(i + 4, min(i + 18, n - 3)):
                bottom2 = lows[j]
                if lows[j] >= lows[j-1] or lows[j] >= lows[j+1]:
                    continue
                if abs(bottom1 - bottom2) / max(bottom1, 1e-9) < tolerance:
                    peak = max(highs[i+1:j]) if j > i+1 else highs[i+1]
                    if bottom1 < peak and bottom2 < peak:
                        if j + 2 < n:
                            if lows[j+1] > peak or lows[j+2] > peak or df["c"].iloc[j+1] > peak:
                                return True, {"bottom1": round(float(bottom1), 6), "bottom2": round(float(bottom2), 6), "neckline": round(float(peak), 6), "bottom1_idx": int(i), "bottom2_idx": int(j)}
        return False, None

    def detect_double_top(self, df, tolerance=0.008):
        highs = df["h"].values
        closes = df["c"].values
        n = len(highs)
        if n < 20:
            return False, None
        for i in range(5, n - 10):
            if highs[i] <= highs[i-1] or highs[i] <= highs[i+1]:
                continue
            for j in range(i + 6, min(i + 25, n - 3)):
                if highs[j] <= highs[j-1] or highs[j] <= highs[j+1]:
                    continue
                if abs(highs[i] - highs[j]) / max(highs[i], 1e-9) < tolerance:
                    neckline = min(closes[i:j])
                    if j + 2 < n and closes[j+1] < neckline:
                        return True, {"peak1": round(float(highs[i]), 6), "peak2": round(float(highs[j]), 6), "neckline": round(float(neckline), 6), "target": round(float(neckline - (highs[i] - neckline)), 6)}
        return False, None

    def detect_double_bottom(self, df, tolerance=0.008):
        lows = df["l"].values
        closes = df["c"].values
        n = len(lows)
        if n < 20:
            return False, None
        for i in range(5, n - 10):
            if lows[i] >= lows[i-1] or lows[i] >= lows[i+1]:
                continue
            for j in range(i + 6, min(i + 25, n - 3)):
                if lows[j] >= lows[j-1] or lows[j] >= lows[j+1]:
                    continue
                if abs(lows[i] - lows[j]) / max(lows[i], 1e-9) < tolerance:
                    neckline = max(closes[i:j])
                    if j + 2 < n and closes[j+1] > neckline:
                        return True, {"bottom1": round(float(lows[i]), 6), "bottom2": round(float(lows[j]), 6), "neckline": round(float(neckline), 6), "target": round(float(neckline + (neckline - lows[i])), 6)}
        return False, None

    def detect_rising_wedge(self, df, lookback=30):
        if len(df) < lookback:
            return False, None
        recent = df.tail(lookback)
        highs = recent["h"].values
        lows = recent["l"].values
        closes = recent["c"].values
        n = len(highs)
        x = np.arange(n)
        try:
            high_slope = np.polyfit(x, highs, 1)[0]
            low_slope = np.polyfit(x, lows, 1)[0]
            if high_slope > 0 and low_slope > 0 and low_slope > high_slope * 1.1:
                low_trend_end = lows[0] + low_slope * (n - 1)
                if closes[-1] < low_trend_end:
                    return True, {"high_slope": round(float(high_slope), 6), "low_slope": round(float(low_slope), 6), "bias": "BEARISH"}
        except Exception:
            pass
        return False, None

    def detect_falling_wedge(self, df, lookback=30):
        if len(df) < lookback:
            return False, None
        recent = df.tail(lookback)
        highs = recent["h"].values
        lows = recent["l"].values
        closes = recent["c"].values
        n = len(highs)
        x = np.arange(n)
        try:
            high_slope = np.polyfit(x, highs, 1)[0]
            low_slope = np.polyfit(x, lows, 1)[0]
            if high_slope < 0 and low_slope < 0 and high_slope < low_slope * 1.1:
                high_trend_end = highs[0] + high_slope * (n - 1)
                if closes[-1] > high_trend_end:
                    return True, {"high_slope": round(float(high_slope), 6), "low_slope": round(float(low_slope), 6), "bias": "BULLISH"}
        except Exception:
            pass
        return False, None

    def detect_flag_pennant(self, df, mode="bull", lookback=25):
        if len(df) < lookback + 10:
            return False, None
        recent = df.tail(lookback)
        highs = recent["h"].values
        lows = recent["l"].values
        closes = recent["c"].values
        n = len(highs)
        if n < 15:
            return False, None
        pole_start = df.tail(lookback + 10).head(10)
        pole_move = abs(pole_start["c"].iloc[-1] - pole_start["c"].iloc[0]) / pole_start["c"].iloc[0] * 100 if pole_start["c"].iloc[0] > 0 else 0
        if pole_move < 2.0:
            return False, None
        consolidation = recent.tail(n - 5)
        cons_highs = consolidation["h"].values
        cons_lows = consolidation["l"].values
        if len(cons_highs) < 5:
            return False, None
        h_slope = np.polyfit(np.arange(len(cons_highs)), cons_highs, 1)[0] if len(cons_highs) > 1 else 0
        l_slope = np.polyfit(np.arange(len(cons_lows)), cons_lows, 1)[0] if len(cons_lows) > 1 else 0
        if mode == "bull":
            if h_slope < 0 and l_slope > 0 and abs(h_slope) > abs(l_slope) * 0.3:
                return True, {"type": "bull_flag", "pole_move": round(pole_move, 2)}
            if abs(h_slope) < 0.0001 and abs(l_slope) < 0.0001 and pole_move > 3.0:
                return True, {"type": "bull_pennant", "pole_move": round(pole_move, 2)}
        elif mode == "bear":
            if h_slope < 0 and l_slope > 0 and abs(l_slope) > abs(h_slope) * 0.3:
                return True, {"type": "bear_flag", "pole_move": round(pole_move, 2)}
            if abs(h_slope) < 0.0001 and abs(l_slope) < 0.0001 and pole_move > 3.0:
                return True, {"type": "bear_pennant", "pole_move": round(pole_move, 2)}
        elif mode == "pennant":
            if abs(h_slope) < 0.0001 and abs(l_slope) < 0.0001 and pole_move > 3.0:
                return True, {"type": "pennant", "pole_move": round(pole_move, 2)}
        return False, None

    def detect_rectangle(self, df, lookback=30, tolerance=0.015):
        if len(df) < lookback:
            return False, None
        recent = df.tail(lookback)
        highs = recent["h"].values
        lows = recent["l"].values
        top = np.percentile(highs, 90)
        bottom = np.percentile(lows, 10)
        height = top - bottom
        if height / recent["c"].iloc[-1] < tolerance:
            touches_top = sum(1 for h in highs if abs(h - top) / top < 0.005)
            touches_bottom = sum(1 for l in lows if abs(l - bottom) / bottom < 0.005)
            if touches_top >= 2 and touches_bottom >= 2:
                return True, {"top": round(float(top), 6), "bottom": round(float(bottom), 6), "touches_top": touches_top, "touches_bottom": touches_bottom}
        return False, None

    def detect_head_and_shoulders(self, df, inverse=False, lookback=50):
        highs = df["h"].values if not inverse else df["l"].values
        lows = df["l"].values if not inverse else df["h"].values
        n = len(highs)
        if n < lookback:
            return False, None
        extrema = []
        for i in range(2, n - 2):
            if not inverse:
                if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                    extrema.append((i, highs[i], "peak"))
            else:
                if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                    extrema.append((i, lows[i], "trough"))
        if len(extrema) < 3:
            return False, None
        e1, e2, e3 = extrema[-3], extrema[-2], extrema[-1]
        if not inverse:
            if e2[1] > e1[1] and e2[1] > e3[1]:
                shoulder_diff = abs(e1[1] - e3[1]) / max(e1[1], 1e-9)
                if shoulder_diff < 0.03:
                    neckline = max(lows[e1[0]:e3[0]])
                    return True, {"left_shoulder": round(float(e1[1]), 6), "head": round(float(e2[1]), 6), "right_shoulder": round(float(e3[1]), 6), "neckline": round(float(neckline), 6)}
        else:
            if e2[1] < e1[1] and e2[1] < e3[1]:
                shoulder_diff = abs(e1[1] - e3[1]) / max(e1[1], 1e-9)
                if shoulder_diff < 0.03:
                    neckline = min(highs[e1[0]:e3[0]])
                    return True, {"left_shoulder": round(float(e1[1]), 6), "head": round(float(e2[1]), 6), "right_shoulder": round(float(e3[1]), 6), "neckline": round(float(neckline), 6)}
        return False, None

    def _detect_candle_patterns(self, df: pd.DataFrame) -> dict:
        result = {"hammer": 0, "shooting_star": 0, "bull_engulf": 0, "bear_engulf": 0}
        if len(df) < 3:
            return result
        o  = df["o"].values
        h  = df["h"].values
        l  = df["l"].values
        c  = df["c"].values
        n  = len(c)
        i  = n - 1

        body      = abs(c[i] - o[i])
        total_rng = max(h[i] - l[i], 1e-9)
        upper_wick = h[i] - max(c[i], o[i])
        lower_wick = min(c[i], o[i]) - l[i]

        if (lower_wick >= 2 * body and upper_wick <= 0.1 * total_rng
                and body / total_rng >= 0.1):
            result["hammer"] = 1

        if (upper_wick >= 2 * body and lower_wick <= 0.1 * total_rng
                and body / total_rng >= 0.1):
            result["shooting_star"] = 1

        if i >= 1:
            prev_body = abs(c[i-1] - o[i-1])
            cur_body  = abs(c[i] - o[i])
            if (c[i-1] < o[i-1]
                    and c[i] > o[i]
                    and c[i] > o[i-1]
                    and o[i] < c[i-1]
                    and cur_body > prev_body * 0.8):
                result["bull_engulf"] = 1

            if (c[i-1] > o[i-1]
                    and c[i] < o[i]
                    and c[i] < o[i-1]
                    and o[i] > c[i-1]
                    and cur_body > prev_body * 0.8):
                result["bear_engulf"] = 1

        return result

    def compute_price_action_bias(self, df, cp):
        if len(df) < 10:
            return "NEUTRAL"
        closes = df["c"].values[-10:]
        highs = df["h"].values[-10:]
        lows = df["l"].values[-10:]
        hh = sum(1 for i in range(1, len(highs)) if highs[i] > highs[i-1])
        hl = sum(1 for i in range(1, len(lows)) if lows[i] > lows[i-1])
        lh = sum(1 for i in range(1, len(highs)) if highs[i] < highs[i-1])
        ll = sum(1 for i in range(1, len(lows)) if lows[i] < lows[i-1])
        bull_score = hh + hl
        bear_score = lh + ll
        if bull_score > bear_score + 3:
            return "UPWARD"
        elif bear_score > bull_score + 3:
            return "DOWNWARD"
        return "NEUTRAL"

    def _detect_swing_structure(self, df: pd.DataFrame) -> dict:
        if len(df) < 20:
            return {"structure": "UNKNOWN", "score_bonus": 0, "detail": "Not enough data",
                    "top2_highs": [], "top2_lows": [], "confirmed": False}
        highs = df["h"].values
        lows  = df["l"].values
        n     = len(highs)
        window = 3
        swing_highs = []
        swing_lows  = []
        for i in range(window, n - window):
            if highs[i] == max(highs[i - window: i + window + 1]):
                swing_highs.append((i, highs[i]))
            if lows[i] == min(lows[i - window: i + window + 1]):
                swing_lows.append((i, lows[i]))
        recent_highs = sorted(swing_highs, key=lambda x: x[0], reverse=True)[:6]
        recent_lows  = sorted(swing_lows,  key=lambda x: x[0], reverse=True)[:6]
        top2_highs = sorted(recent_highs, key=lambda x: x[1], reverse=True)[:2]
        top2_highs = sorted(top2_highs, key=lambda x: x[0], reverse=True)
        top2_lows  = sorted(recent_lows, key=lambda x: x[1])[:2]
        top2_lows  = sorted(top2_lows,  key=lambda x: x[0], reverse=True)
        if len(top2_highs) < 2 or len(top2_lows) < 2:
            return {"structure": "UNKNOWN", "score_bonus": 0, "detail": "Insufficient swings",
                    "top2_highs": [p for _, p in top2_highs], "top2_lows": [p for _, p in top2_lows], "confirmed": False}
        h1, h2 = top2_highs[0][1], top2_highs[1][1]
        l1, l2 = top2_lows[0][1], top2_lows[1][1]
        hh = h1 > h2; lh = h1 < h2; hl = l1 > l2; ll = l1 < l2
        if hh and hl:
            structure = "BULLISH"; score_bonus = 25; detail = f"HH({h1:.4f}>{h2:.4f}) + HL({l1:.4f}>{l2:.4f}) → BULLISH"; confirmed = True
        elif lh and ll:
            structure = "BEARISH"; score_bonus = -25; detail = f"LH({h1:.4f}<{h2:.4f}) + LL({l1:.4f}<{l2:.4f}) → BEARISH"; confirmed = True
        elif hh and ll:
            structure = "DISTRIBUTION"; score_bonus = -8; detail = f"HH({h1:.4f}) + LL({l1:.4f}) → Distribution"; confirmed = False
        elif lh and hl:
            structure = "ACCUMULATION"; score_bonus = 8; detail = f"LH({h1:.4f}) + HL({l1:.4f}) → Accumulation"; confirmed = False
        else:
            structure = "NEUTRAL"; score_bonus = 0; detail = "No clear structure"; confirmed = False
        return {"structure": structure, "score_bonus": score_bonus, "detail": detail,
                "top2_highs": [h1, h2], "top2_lows": [l1, l2], "hh": hh, "hl": hl, "lh": lh, "ll": ll, "confirmed": confirmed}

    def compute_indicators(self, df):
        if len(df) < 50:
            logging.warning("compute_indicators: insufficient data (%d rows)", len(df))
        delta = df["c"].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/14, min_periods=14).mean()
        avg_loss = loss.ewm(alpha=1/14, min_periods=14).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df["rsi"] = 100 - (100 / (1 + rs))
        df["rsi"] = df["rsi"].fillna(50.0)
        df["pc"] = df["c"].shift(1)
        tr = pd.concat([df["h"] - df["l"], (df["h"] - df["pc"]).abs(), (df["l"] - df["pc"]).abs()], axis=1).max(axis=1)
        df["atr"] = tr.ewm(alpha=1/14, min_periods=14).mean()
        df["vol_avg20"] = df["v"].rolling(20).mean()
        df["vol_ratio"] = (df["v"] / df["vol_avg20"].replace(0, np.nan)) * 100
        df["vol_ratio"] = df["vol_ratio"].fillna(100.0)
        df["vol_delta"] = (df["v"] * (df["c"] - df["o"]) / (df["h"] - df["l"]).replace(0, np.nan)).fillna(0)
        df["roc"] = df["c"].pct_change(5)
        df["ema20"] = df["c"].ewm(span=20, adjust=False).mean()
        df["ema50"] = df["c"].ewm(span=50, adjust=False).mean()
        df["ema200"] = df["c"].ewm(span=200, adjust=False).mean()
        for ema_col in ["ema20", "ema50", "ema200"]:
            df[ema_col] = df[ema_col].fillna(df["c"])
        df["swing_low_10"] = df["l"].rolling(window=10, min_periods=5).min()
        df["swing_high_10"] = df["h"].rolling(window=10, min_periods=5).max()
        df["swing_low_20"] = df["l"].rolling(window=20, min_periods=10).min()
        df["swing_high_20"] = df["h"].rolling(window=20, min_periods=10).max()
        bb_mid = df["c"].rolling(20).mean()
        bb_std = df["c"].rolling(20).std()
        df["bb_upper"] = bb_mid + 2 * bb_std
        df["bb_lower"] = bb_mid - 2 * bb_std
        df["bb_mid"] = bb_mid
        df["bb_pct"] = (df["c"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)
        df["bb_bandwidth"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"].replace(0, np.nan)
        df["bb_pct_b"] = (df["c"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)
        df["bb_squeeze"] = (df["bb_bandwidth"] < df["bb_bandwidth"].rolling(20).mean() * 0.6).fillna(False)
        df = self._compute_adx(df)
        df = self._compute_sar(df)
        df = self._compute_supertrend(df)
        df = self._compute_kdj(df)
        return df

    def _compute_adx(self, df, period=14):
        if len(df) < period + 1:
            df["adx"] = np.nan; df["di_plus"] = np.nan; df["di_minus"] = np.nan
            return df
        high = df["h"]; low = df["l"]; close = df["c"]
        plus_dm = high.diff(); minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0; minus_dm[minus_dm < 0] = 0

        pdm = plus_dm.values.copy()
        ndm = minus_dm.values.copy()
        for i in range(len(pdm)):
            if pdm[i] > ndm[i]:
                ndm[i] = 0
            elif ndm[i] > pdm[i]:
                pdm[i] = 0
            else:
                pdm[i] = 0
                ndm[i] = 0
        plus_dm = pd.Series(pdm, index=plus_dm.index)
        minus_dm = pd.Series(ndm, index=minus_dm.index)

        tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/period, min_periods=period).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=1/period, min_periods=period).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(alpha=1/period, min_periods=period).mean() / atr)
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)) * 100
        adx = dx.ewm(alpha=1/period, min_periods=period).mean()
        df["di_plus"] = plus_di; df["di_minus"] = minus_di; df["adx"] = adx
        return df

    def _compute_sar(self, df, af=0.02, max_af=0.2):
        if len(df) < 5:
            df["sar"] = df["c"]
            return df
        high = df["h"].values; low = df["l"].values; close = df["c"].values
        n = len(high)
        sar = np.zeros(n)
        ep = np.zeros(n)
        af_vals = np.zeros(n)
        trend = np.zeros(n)
        trend[0] = 1 if close[0] > close[1] else -1 if n > 1 else 1
        sar[0] = low[0] if trend[0] == 1 else high[0]
        ep[0] = high[0] if trend[0] == 1 else low[0]
        af_vals[0] = af
        for i in range(1, n):
            sar[i] = sar[i-1] + af_vals[i-1] * (ep[i-1] - sar[i-1])
            if trend[i-1] == 1:
                if low[i] < sar[i]:
                    trend[i] = -1
                    sar[i] = ep[i-1]
                    ep[i] = low[i]
                    af_vals[i] = af
                else:
                    trend[i] = 1
                    ep[i] = max(high[i], ep[i-1])
                    af_vals[i] = min(af_vals[i-1] + af, max_af) if high[i] > ep[i-1] else af_vals[i-1]
            else:
                if high[i] > sar[i]:
                    trend[i] = 1
                    sar[i] = ep[i-1]
                    ep[i] = high[i]
                    af_vals[i] = af
                else:
                    trend[i] = -1
                    ep[i] = min(low[i], ep[i-1])
                    af_vals[i] = min(af_vals[i-1] + af, max_af) if low[i] < ep[i-1] else af_vals[i-1]
        df["sar"] = sar
        return df

    def _compute_supertrend(self, df, period=10, multiplier=3):
        if len(df) < period + 1:
            df["supertrend"] = df["c"]
            df["supertrend_dir"] = "NONE"
            return df
        hl2 = (df["h"] + df["l"]) / 2
        atr = df["atr"] if "atr" in df.columns else self._compute_adx(df.copy(), period)["atr"]
        upperband = hl2 + multiplier * atr
        lowerband = hl2 - multiplier * atr
        supertrend = np.zeros(len(df))
        direction = np.zeros(len(df))
        final_upperband = upperband.values.copy()
        final_lowerband = lowerband.values.copy()
        for i in range(1, len(df)):
            if df["c"].iloc[i-1] > final_upperband[i-1]:
                final_upperband[i] = max(upperband.iloc[i], final_upperband[i-1])
            else:
                final_upperband[i] = upperband.iloc[i]
            if df["c"].iloc[i-1] < final_lowerband[i-1]:
                final_lowerband[i] = min(lowerband.iloc[i], final_lowerband[i-1])
            else:
                final_lowerband[i] = lowerband.iloc[i]
        for i in range(len(df)):
            if df["c"].iloc[i] > final_upperband[i]:
                supertrend[i] = final_lowerband[i]
                direction[i] = 1
            elif df["c"].iloc[i] < final_lowerband[i]:
                supertrend[i] = final_upperband[i]
                direction[i] = -1
            else:
                supertrend[i] = supertrend[i-1] if i > 0 else final_lowerband[i]
                direction[i] = direction[i-1] if i > 0 else 1
        df["supertrend"] = supertrend
        df["supertrend_dir"] = ["BUY" if d == 1 else "SELL" if d == -1 else "NONE" for d in direction]
        return df

    def _compute_kdj(self, df, n=9, m1=3, m2=3):
        if len(df) < n + m1 + m2:
            df["kdj_k"] = 50.0; df["kdj_d"] = 50.0; df["kdj_j"] = 50.0
            return df
        low_n = df["l"].rolling(window=n).min()
        high_n = df["h"].rolling(window=n).max()
        rsv = (df["c"] - low_n) / (high_n - low_n).replace(0, np.nan) * 100
        rsv = rsv.fillna(50)
        k = rsv.ewm(com=m1-1, adjust=False).mean()
        d = k.ewm(com=m2-1, adjust=False).mean()
        j = 3 * k - 2 * d
        df["kdj_k"] = k.fillna(50)
        df["kdj_d"] = d.fillna(50)
        df["kdj_j"] = j.fillna(50)
        return df

    def compute_fibonacci(self, swing_low, swing_high):
        diff = swing_high - swing_low
        if diff <= 0:
            return {}
        return {
            "0.0": swing_low, "23.6": swing_low + diff * 0.236, "38.2": swing_low + diff * 0.382,
            "50.0": swing_low + diff * 0.5, "61.8": swing_low + diff * 0.618,
            "78.6": swing_low + diff * 0.786, "100.0": swing_high,
            "127.2": swing_high + diff * 0.272, "161.8": swing_high + diff * 0.618,
            "261.8": swing_high + diff * 1.618
        }

    def _compute_smart_entry_exit(self, cp: float, r: 'AnalysisResult', direction: str) -> dict:
        atr = max(r.atr, cp * 0.002)
        result = {"smart_entry_low": 0.0, "smart_entry_high": 0.0, "smart_tp1": 0.0, "smart_tp2": 0.0, "smart_tp3": 0.0, "smart_sl": 0.0, "entry_method": "ATR-Fallback"}
        fib = r.fib_levels
        has_fib = bool(fib) and "61.8" in fib and "38.2" in fib
        if direction == "BUY":
            if has_fib:
                fib_618 = fib["61.8"]; fib_382 = fib["38.2"]; fib_236 = fib["23.6"]
                if fib_382 < cp and fib_618 < cp:
                    result["smart_entry_low"] = round(fib_618, 6); result["smart_entry_high"] = round(fib_382, 6); result["entry_method"] = "Fib 38.2-61.8% Pullback"
                elif fib_236 < cp:
                    result["smart_entry_low"] = round(fib_236 - atr * 0.2, 6); result["smart_entry_high"] = round(fib_236 + atr * 0.1, 6); result["entry_method"] = "Fib 23.6% Shallow Pullback"
            ema_candidates = []
            if r.ema20 > 0 and abs(cp - r.ema20) / cp < 0.015: ema_candidates.append(("EMA20", r.ema20))
            if r.ema50 > 0 and abs(cp - r.ema50) / cp < 0.025: ema_candidates.append(("EMA50", r.ema50))
            if r.ema200 > 0 and abs(cp - r.ema200) / cp < 0.04: ema_candidates.append(("EMA200", r.ema200))
            if ema_candidates:
                best_ema_name, best_ema = min(ema_candidates, key=lambda x: abs(cp - x[1]))
                result["smart_entry_low"] = round(best_ema - atr * 0.3, 6); result["smart_entry_high"] = round(best_ema + atr * 0.15, 6); result["entry_method"] = f"{best_ema_name} Pullback Zone"
            if result["smart_entry_low"] == 0.0 and r.bb_lower > 0:
                result["smart_entry_low"] = round(r.bb_lower, 6); result["smart_entry_high"] = round(r.bb_mid, 6); result["entry_method"] = "BB Lower Band Bounce"
            if result["smart_entry_low"] == 0.0:
                result["smart_entry_low"] = round(cp - atr * 0.4, 6); result["smart_entry_high"] = round(cp + atr * 0.1, 6); result["entry_method"] = "ATR-Based Entry"
            if has_fib and fib.get("127.2", 0) > cp: result["smart_tp1"] = round(fib["127.2"], 6)
            else: result["smart_tp1"] = round(cp + atr * 2.0, 6)
            if has_fib and fib.get("161.8", 0) > cp: result["smart_tp2"] = round(fib["161.8"], 6)
            elif r.bb_upper > cp: result["smart_tp2"] = round(r.bb_upper, 6)
            else: result["smart_tp2"] = round(cp + atr * 3.5, 6)
            if has_fib and fib.get("261.8", 0) > cp: result["smart_tp3"] = round(fib["261.8"], 6)
            else: result["smart_tp3"] = round(cp + atr * 5.5, 6)
            if has_fib and fib["0.0"] > 0: result["smart_sl"] = round(fib["0.0"] - atr * 0.5, 6)
            else: result["smart_sl"] = round(result["smart_entry_low"] - atr * 0.8, 6)
        elif direction == "SELL":
            if has_fib:
                fib_382 = fib["38.2"]; fib_618 = fib["61.8"]
                if fib_382 > cp and fib_618 > cp:
                    result["smart_entry_high"] = round(fib_618, 6); result["smart_entry_low"] = round(fib_382, 6); result["entry_method"] = "Fib 38.2-61.8% Rejection"
                elif fib.get("23.6", 0) > cp:
                    result["smart_entry_high"] = round(fib["23.6"] + atr * 0.2, 6); result["smart_entry_low"] = round(fib["23.6"] - atr * 0.1, 6); result["entry_method"] = "Fib 23.6% Rejection"
            ema_candidates = []
            if r.ema20 > 0 and abs(cp - r.ema20) / cp < 0.015: ema_candidates.append(("EMA20", r.ema20))
            if r.ema50 > 0 and abs(cp - r.ema50) / cp < 0.025: ema_candidates.append(("EMA50", r.ema50))
            if r.ema200 > 0 and abs(cp - r.ema200) / cp < 0.04: ema_candidates.append(("EMA200", r.ema200))
            if ema_candidates:
                best_ema_name, best_ema = min(ema_candidates, key=lambda x: abs(cp - x[1]))
                result["smart_entry_high"] = round(best_ema + atr * 0.3, 6); result["smart_entry_low"] = round(best_ema - atr * 0.15, 6); result["entry_method"] = f"{best_ema_name} Rejection Zone"
            if result["smart_entry_high"] == 0.0 and r.bb_upper > 0:
                result["smart_entry_high"] = round(r.bb_upper, 6); result["smart_entry_low"] = round(r.bb_mid, 6); result["entry_method"] = "BB Upper Band Rejection"
            if result["smart_entry_high"] == 0.0:
                result["smart_entry_high"] = round(cp + atr * 0.4, 6); result["smart_entry_low"] = round(cp - atr * 0.1, 6); result["entry_method"] = "ATR-Based Entry"
            fib_0 = fib.get("0.0", 0); fib_236 = fib.get("23.6", 0); fib_382 = fib.get("38.2", 0)
            if has_fib and fib_236 > 0 and fib_236 < cp: result["smart_tp1"] = round(fib_236, 6)
            else: result["smart_tp1"] = round(cp - atr * 2.0, 6)
            if has_fib and fib_382 > 0 and fib_382 < cp: result["smart_tp2"] = round(fib_382, 6)
            elif r.bb_lower > 0 and r.bb_lower < cp: result["smart_tp2"] = round(r.bb_lower, 6)
            else: result["smart_tp2"] = round(cp - atr * 3.5, 6)
            if has_fib and fib_0 > 0 and fib_0 < cp: result["smart_tp3"] = round(fib_0, 6)
            else: result["smart_tp3"] = round(cp - atr * 5.5, 6)
            if has_fib and fib["100.0"] > 0: result["smart_sl"] = round(fib["100.0"] + atr * 0.5, 6)
            else: result["smart_sl"] = round(result["smart_entry_high"] + atr * 0.8, 6)
        if direction == "BUY" and r.rsi > 70:
            result["smart_entry_low"] = round(result["smart_entry_low"] - atr * 0.3, 6); result["entry_method"] += " [RSI OB: Deeper Entry]"
        elif direction == "SELL" and r.rsi < 30:
            result["smart_entry_high"] = round(result["smart_entry_high"] + atr * 0.3, 6); result["entry_method"] += " [RSI OS: Higher Short Entry]"
        return result

    def compute_directional_forecast(self, cp: float, r: AnalysisResult, swing_struct: dict = None, market_struct: dict = None) -> Tuple:
        """
        v21 ULTRA: Structure-First Directional Forecast
        أوزان معدلة:
        - Price Action + Volume + Market Structure = أعلى وزن
        - EMA + Trend = وزن متوسط
        - RSI + Stochastic = وزن ضعيف
        """
        score = 0
        reasons = []
        confluence = 0

        regime = r.market_regime
        adx_val = r.adx
        is_ranging = (regime == "RANGING" or adx_val < 20)
        is_trending = (regime == "TRENDING" and adx_val >= 25)
        is_scalping = self.interval in ("5m", "15m")

        # ==========================================
        # 1. MARKET STRUCTURE (HIGHEST WEIGHT: 3.0)
        # ==========================================
        struct_score = 0
        if market_struct:
            trend = market_struct.get("trend", "NEUTRAL")
            hh_count = market_struct.get("hh_count", 0)
            hl_count = market_struct.get("hl_count", 0)
            lh_count = market_struct.get("lh_count", 0)
            ll_count = market_struct.get("ll_count", 0)

            if trend == "BULLISH_TREND":
                struct_score += int(30 * WEIGHT_CONFIG["market_structure"])
                reasons.append(f"STRUCT: BULLISH_TREND (HH={hh_count}, HL={hl_count})")
                confluence += 1
            elif trend == "BEARISH_TREND":
                struct_score -= int(30 * WEIGHT_CONFIG["market_structure"])
                reasons.append(f"STRUCT: BEARISH_TREND (LH={lh_count}, LL={ll_count})")
                confluence += 1
            elif trend == "DISTRIBUTION":
                struct_score -= int(15 * WEIGHT_CONFIG["market_structure"])
                reasons.append("STRUCT: DISTRIBUTION (caution)")
            elif trend == "ACCUMULATION":
                struct_score += int(15 * WEIGHT_CONFIG["market_structure"])
                reasons.append("STRUCT: ACCUMULATION (watch)")
            else:
                reasons.append("STRUCT: NEUTRAL (no clear trend)")

        if swing_struct and swing_struct.get("structure") != "UNKNOWN":
            sb = swing_struct["score_bonus"]
            if is_scalping: 
                sb = int(sb * 0.6)
            struct_score += sb
            if swing_struct.get("confirmed"): 
                confluence += 1
            conf_tag = "✓" if swing_struct.get("confirmed") else "~"
            reasons.append(f"Swing: {swing_struct['structure']} {conf_tag}")

        score += struct_score

        # ==========================================
        # 2. PRICE ACTION (HIGH WEIGHT: 2.5)
        # ==========================================
        pa_score = 0
        if r.price_action_bias == "UPWARD":
            pa_score += int(12 * WEIGHT_CONFIG["price_action"])
            reasons.append("PA: HH+HL Upward")
            confluence += 1
        elif r.price_action_bias == "DOWNWARD":
            pa_score -= int(12 * WEIGHT_CONFIG["price_action"])
            reasons.append("PA: LH+LL Downward")
            confluence += 1

        if r.candle_strength > 65:
            if r.buy_pressure > r.sell_pressure:
                pa_score += int(10 * WEIGHT_CONFIG["price_action"])
                reasons.append("PA: Strong Bull Candle")
                confluence += 1
            else:
                pa_score -= int(10 * WEIGHT_CONFIG["price_action"])
                reasons.append("PA: Strong Bear Candle")
                confluence += 1

        # Candle patterns
        cpats = self._detect_candle_patterns(self._deque_to_df()) if hasattr(self, '_deque_to_df') else {}
        if cpats.get("hammer") and cp > r.ema20:
            pa_score += int(8 * WEIGHT_CONFIG["price_action"])
            reasons.append("PA: Hammer @ support")
        if cpats.get("shooting_star") and cp < r.ema20:
            pa_score -= int(8 * WEIGHT_CONFIG["price_action"])
            reasons.append("PA: Shooting Star @ resist")
        if cpats.get("bull_engulf"):
            pa_score += int(12 * WEIGHT_CONFIG["price_action"])
            reasons.append("PA: Bull Engulfing")
            confluence += 1
        if cpats.get("bear_engulf"):
            pa_score -= int(12 * WEIGHT_CONFIG["price_action"])
            reasons.append("PA: Bear Engulfing")
            confluence += 1

        score += pa_score

        # ==========================================
        # 3. VOLUME (HIGH WEIGHT: 2.5)
        # ==========================================
        vol_score = 0
        vol_agree = False
        if r.vol_ratio > 150:
            if score > 0: 
                vol_score += int(12 * WEIGHT_CONFIG["volume"])
                reasons.append(f"VOL: {r.vol_ratio:.0f}% Strong (Confirms Bull)")
                vol_agree = True
            else: 
                vol_score -= int(12 * WEIGHT_CONFIG["volume"])
                reasons.append(f"VOL: {r.vol_ratio:.0f}% Strong (Confirms Bear)")
                vol_agree = True
        elif r.vol_ratio > 120: 
            vol_score += int(5 * WEIGHT_CONFIG["volume"])
            reasons.append(f"VOL: {r.vol_ratio:.0f}% Elevated")
        elif r.vol_ratio < 80: 
            vol_score = int(vol_score * 0.85)
            reasons.append(f"VOL: {r.vol_ratio:.0f}% Low (Weak)")

        if r.vol_delta > 0 and score > 0: 
            vol_score += int(5 * WEIGHT_CONFIG["volume"])
            reasons.append("VOL: Positive Delta")
        elif r.vol_delta < 0 and score < 0: 
            vol_score -= int(5 * WEIGHT_CONFIG["volume"])
            reasons.append("VOL: Negative Delta")

        if vol_agree: 
            confluence += 1
        score += vol_score

        # ==========================================
        # 4. PATTERNS (HIGH WEIGHT: 2.0)
        # ==========================================
        pat_score = 0
        pat_agree = False
        if r.double_bottom: 
            pat_score += int(15 * WEIGHT_CONFIG["patterns"])
            reasons.append("PAT: DOUBLE BOTTOM (Bull Rev)")
            pat_agree = True
        if r.double_top: 
            pat_score -= int(15 * WEIGHT_CONFIG["patterns"])
            reasons.append("PAT: DOUBLE TOP (Bear Rev)")
            pat_agree = True
        if r.falling_wedge: 
            pat_score += int(10 * WEIGHT_CONFIG["patterns"])
            reasons.append("PAT: FALLING WEDGE (Bull Rev)")
            pat_agree = True
        if r.rising_wedge: 
            pat_score -= int(10 * WEIGHT_CONFIG["patterns"])
            reasons.append("PAT: RISING WEDGE (Bear Rev)")
            pat_agree = True
        if r.flag_bull: 
            pat_score += int(8 * WEIGHT_CONFIG["patterns"])
            reasons.append("PAT: BULL FLAG (Cont)")
            pat_agree = True
        if r.flag_bear: 
            pat_score -= int(8 * WEIGHT_CONFIG["patterns"])
            reasons.append("PAT: BEAR FLAG (Cont)")
            pat_agree = True
        if r.pennant: 
            pat_score += int(5 * WEIGHT_CONFIG["patterns"])
            reasons.append("PAT: PENNANT (Breakout)")
            pat_agree = True
        if r.rectangle: 
            reasons.append("PAT: RECTANGLE (Wait)")
        if r.head_shoulders: 
            pat_score -= int(12 * WEIGHT_CONFIG["patterns"])
            reasons.append("PAT: H&S (Bear Rev)")
            pat_agree = True
        if r.inv_head_shoulders: 
            pat_score += int(12 * WEIGHT_CONFIG["patterns"])
            reasons.append("PAT: INV H&S (Bull Rev)")
            pat_agree = True
        if r.w_pattern: 
            pat_score += int(8 * WEIGHT_CONFIG["patterns"])
            reasons.append("PAT: W-Pattern (Bull)")
            pat_agree = True
        if r.m_pattern: 
            pat_score -= int(8 * WEIGHT_CONFIG["patterns"])
            reasons.append("PAT: M-Pattern (Bear)")
            pat_agree = True
        if pat_agree: 
            confluence += 1
        score += pat_score

        # ==========================================
        # 5. EMA TREND (MEDIUM WEIGHT: 1.5)
        # ==========================================
        ema_score = 0
        ema_agree = False
        ema200_valid = r.ema200 > 0 and abs(r.ema200 - cp) / cp < 0.5
        ema200_weight = int(12 * WEIGHT_CONFIG["ema_trend"])
        if ema200_valid:
            if cp > r.ema200: 
                ema_score += ema200_weight
                reasons.append("EMA: Price > EMA200 (Bull Trend)")
            else: 
                ema_score -= ema200_weight
                reasons.append("EMA: Price < EMA200 (Bear Trend)")
        ema_cross_weight = int(10 * WEIGHT_CONFIG["ema_trend"])
        if r.ema20 > r.ema50: 
            ema_score += ema_cross_weight
            reasons.append("EMA: EMA20 > EMA50 (Short Bull)")
            ema_agree = True
        else: 
            ema_score -= ema_cross_weight
            reasons.append("EMA: EMA20 < EMA50 (Short Bear)")
            ema_agree = True
        if cp > r.ema20: 
            ema_score += int(5 * WEIGHT_CONFIG["ema_trend"])
            reasons.append("EMA: Price > EMA20")
        else: 
            ema_score -= int(5 * WEIGHT_CONFIG["ema_trend"])
            reasons.append("EMA: Price < EMA20")
        if ema_agree: 
            confluence += 1
        score += ema_score

        # ==========================================
        # 6. ADX TREND (MEDIUM WEIGHT: 1.5)
        # ==========================================
        adx_score = 0
        adx_agree = False
        if adx_val >= 25:
            if r.di_plus > r.di_minus: 
                adx_score += int(10 * WEIGHT_CONFIG["adx_trend"])
                reasons.append(f"ADX: {adx_val:.0f} +DI > -DI (Strong Bull)")
                adx_agree = True
            elif r.di_minus > r.di_plus: 
                adx_score -= int(10 * WEIGHT_CONFIG["adx_trend"])
                reasons.append(f"ADX: {adx_val:.0f} -DI > +DI (Strong Bear)")
                adx_agree = True
            else: 
                adx_score += int(3 * WEIGHT_CONFIG["adx_trend"])
                reasons.append(f"ADX: {adx_val:.0f} (Trending)")
        elif adx_val >= 18: 
            adx_score += int(3 * WEIGHT_CONFIG["adx_trend"])
            reasons.append(f"ADX: {adx_val:.0f} (Moderate)")
        else: 
            adx_score = int(adx_score * 0.7)
            reasons.append(f"ADX: {adx_val:.0f} < 20 (Weak/Ranging)")
        if adx_agree: 
            confluence += 1
        score += adx_score

        # ==========================================
        # 7. SUPERTREND (MEDIUM WEIGHT: 1.2)
        # ==========================================
        if r.supertrend_dir == "BUY": 
            score += int(8 * WEIGHT_CONFIG["supertrend"])
            reasons.append("ST: BUY")
            confluence += 1
        elif r.supertrend_dir == "SELL": 
            score -= int(8 * WEIGHT_CONFIG["supertrend"])
            reasons.append("ST: SELL")
            confluence += 1

        if r.sar < cp and cp > r.ema20: 
            score += int(3 * WEIGHT_CONFIG["supertrend"])
            reasons.append("SAR: Below price")
        elif r.sar > cp and cp < r.ema20: 
            score -= int(3 * WEIGHT_CONFIG["supertrend"])
            reasons.append("SAR: Above price")

        # ==========================================
        # 8. MACD (MEDIUM WEIGHT: 1.0) - IGNORE IF CONFLICT
        # ==========================================
        macd_score = 0
        # We use EMA cross as MACD proxy since we have it
        if r.ema20 > r.ema50 and r.rsi > 50:
            macd_score += int(8 * WEIGHT_CONFIG["macd"])
            reasons.append("MACD: Bullish alignment")
        elif r.ema20 < r.ema50 and r.rsi < 50:
            macd_score -= int(8 * WEIGHT_CONFIG["macd"])
            reasons.append("MACD: Bearish alignment")
        else:
            # Conflict detected - reduce MACD weight
            macd_score = int(macd_score * 0.3)
            reasons.append("MACD: Weak/Conflict (reduced weight)")
        score += macd_score

        # ==========================================
        # 9. RSI (LOW WEIGHT: 0.8)
        # ==========================================
        rsi_score = 0
        rsi_agree = False
        rsi_mult = 1.4 if is_scalping else 1.0
        if r.rsi >= 75: 
            rsi_score -= int(8 * WEIGHT_CONFIG["rsi"] * rsi_mult)
            reasons.append(f"RSI: {r.rsi:.0f} Overbought")
            rsi_agree = True
        elif r.rsi >= 65: 
            rsi_score += int(12 * WEIGHT_CONFIG["rsi"] * rsi_mult)
            reasons.append(f"RSI: {r.rsi:.0f} Bullish Momentum")
            rsi_agree = True
        elif r.rsi <= 25: 
            rsi_score += int(8 * WEIGHT_CONFIG["rsi"] * rsi_mult)
            reasons.append(f"RSI: {r.rsi:.0f} Extremely Oversold")
            rsi_agree = True
        elif r.rsi <= 35: 
            rsi_score -= int(12 * WEIGHT_CONFIG["rsi"] * rsi_mult)
            reasons.append(f"RSI: {r.rsi:.0f} Bearish Momentum")
            rsi_agree = True
        elif r.rsi > 55: 
            rsi_score += int(6 * WEIGHT_CONFIG["rsi"] * rsi_mult)
            reasons.append(f"RSI: {r.rsi:.0f} Mild Bull")
            rsi_agree = True
        elif r.rsi < 45: 
            rsi_score -= int(6 * WEIGHT_CONFIG["rsi"] * rsi_mult)
            reasons.append(f"RSI: {r.rsi:.0f} Mild Bear")
            rsi_agree = True
        if r.rsi_divergence == "BULLISH": 
            rsi_score += int(15 * WEIGHT_CONFIG["rsi"])
            reasons.append("RSI: Bull Divergence")
            rsi_agree = True
        elif r.rsi_divergence == "BEARISH": 
            rsi_score -= int(15 * WEIGHT_CONFIG["rsi"])
            reasons.append("RSI: Bear Divergence")
            rsi_agree = True
        micro_weight = int(15 * WEIGHT_CONFIG["rsi"])
        if is_scalping: 
            micro_weight = int(micro_weight * 1.5)
        if r.rsi_micro_divergence == "BULLISH_MICRO": 
            rsi_score += micro_weight
            reasons.append("RSI: Micro Bull Divergence (Scalping)")
            rsi_agree = True
        elif r.rsi_micro_divergence == "BEARISH_MICRO": 
            rsi_score -= micro_weight
            reasons.append("RSI: Micro Bear Divergence (Scalping)")
            rsi_agree = True
        if rsi_agree: 
            confluence += 1
        score += rsi_score

        # ==========================================
        # 10. STOCHASTIC (LOW WEIGHT: 0.6)
        # ==========================================
        kdj_score = 0
        if r.kdj_j > 80 and r.kdj_k < r.kdj_j: 
            kdj_score -= int(5 * WEIGHT_CONFIG["stochastic"])
            reasons.append("KDJ: Overbought")
        elif r.kdj_j < 20 and r.kdj_k > r.kdj_j: 
            kdj_score += int(5 * WEIGHT_CONFIG["stochastic"])
            reasons.append("KDJ: Oversold")
        elif r.kdj_k > r.kdj_d and r.kdj_j > r.kdj_k: 
            kdj_score += int(3 * WEIGHT_CONFIG["stochastic"])
            reasons.append("KDJ: Bull Cross")
        elif r.kdj_k < r.kdj_d and r.kdj_j < r.kdj_k: 
            kdj_score -= int(3 * WEIGHT_CONFIG["stochastic"])
            reasons.append("KDJ: Bear Cross")
        score += kdj_score

        # ==========================================
        # 11. ORDER BOOK (AUXILIARY: 0.5) - NEVER BLOCKING
        # ==========================================
        ob_score = 0
        if r.ob_imbalance >= 2.0 and score > 0:
            ob_score += int(3 * WEIGHT_CONFIG["ob_imbalance"])
            reasons.append("OB: Strong bid dominance")
        elif r.ob_imbalance <= 0.5 and score < 0:
            ob_score -= int(3 * WEIGHT_CONFIG["ob_imbalance"])
            reasons.append("OB: Strong ask dominance")
        score += ob_score

        # ==========================================
        # 12. BB POSITION (Part of Price Action)
        # ==========================================
        bb_agree = False
        if is_ranging:
            if r.bb_lower > 0 and r.bb_upper > 0:
                if cp <= r.bb_lower: 
                    score += int(9 * WEIGHT_CONFIG["price_action"])
                    reasons.append("BB: Lower Touch (Ranging Buy)")
                    bb_agree = True
                elif cp >= r.bb_upper: 
                    score -= int(9 * WEIGHT_CONFIG["price_action"])
                    reasons.append("BB: Upper Touch (Ranging Sell)")
                    bb_agree = True
                elif r.bb_pct < 0.2: 
                    score += int(5 * WEIGHT_CONFIG["price_action"])
                    reasons.append("BB: Near Lower (Ranging)")
                    bb_agree = True
                elif r.bb_pct > 0.8: 
                    score -= int(5 * WEIGHT_CONFIG["price_action"])
                    reasons.append("BB: Near Upper (Ranging)")
                    bb_agree = True
            if r.bb_bandwidth < 0.008: 
                score = int(score * 0.5)
                reasons.append("BB: Tight Squeeze (Choppy)")
        else:
            if r.bb_lower > 0 and cp <= r.bb_lower and r.ema20 > r.ema50: 
                score += int(4 * WEIGHT_CONFIG["price_action"])
                reasons.append("BB: Lower + Uptrend (Pullback)")
                bb_agree = True
            elif r.bb_upper > 0 and cp >= r.bb_upper and r.ema20 < r.ema50: 
                score -= int(4 * WEIGHT_CONFIG["price_action"])
                reasons.append("BB: Upper + Downtrend (Pullback)")
                bb_agree = True
        if bb_agree: 
            confluence += 1

        # ==========================================
        # TRENDING BONUS: Full EMA Stack
        # ==========================================
        if is_trending:
            ema20_dist_pct = abs(cp - r.ema20) / r.ema20 * 100 if r.ema20 > 0 else 0.0
            ema_dist_threshold = {"5m": 1.5, "15m": 2.0, "1h": 2.5, "4h": 3.5, "1d": 5.0}.get(self.interval, 2.5)
            price_too_far = ema20_dist_pct > ema_dist_threshold
            if r.ema20 > r.ema50 > r.ema200 and cp > r.ema20:
                if price_too_far:
                    reasons.append(f"EMA: Full Bull Stack BUT price {ema20_dist_pct:.1f}% from EMA20 (no bonus)")
                else:
                    score += int(10 * WEIGHT_CONFIG["ema_trend"])
                    reasons.append("EMA: Full Bull Stack + Price > EMA20")
                    confluence += 1
            elif r.ema20 < r.ema50 < r.ema200 and cp < r.ema20: 
                score -= int(10 * WEIGHT_CONFIG["ema_trend"])
                reasons.append("EMA: Full Bear Stack + Price < EMA20")
                confluence += 1
        else:
            if r.ema20 > r.ema50: 
                score = int(score * 0.9)
                reasons.append("EMA: Ranging — reduced weight")

        # ==========================================
        # CONFLUENCE CHECK
        # ==========================================
        if confluence < 3 and abs(score) >= 50:
            reasons.append(f"Low Confluence ({confluence})")
            score = int(score * 0.65)

        # ==========================================
        # DIRECTION & THRESHOLD DETERMINATION
        # ==========================================
        atr_pre = max(r.atr, cp * 0.002) if r.atr > 0 else cp * 0.002
        entry_low  = cp - atr_pre * 0.5
        entry_high = cp + atr_pre * 0.5
        tp         = cp + atr_pre * 2
        sl         = cp - atr_pre * 1.5
        direction  = "WAIT"
        strength   = "NEUTRAL"
        confidence = 0.0

        # v21: STRICT Scalping Filter
        if self.interval in ("1m", "3m", "5m", "15m"):
            scalping_valid, scalping_msg = self._scalping_filter(cp, r, self.interval)
            if not scalping_valid:
                direction = "WAIT"
                strength = "NEUTRAL"
                reasons.append("SCALPING FILTER: " + scalping_msg)
                confluence_ratio = min(confluence / 8.0, 1.0)
                score_component = min(abs(score) * 0.6, 75.0)
                confl_bonus = confluence_ratio * 20.0
                raw_conf = score_component + confl_bonus
                confidence = round(min(max(raw_conf * 0.7, 10.0), 60.0), 1)
                return direction, strength, round(entry_low, 6), round(entry_high, 6), round(tp, 6), round(sl, 6), confidence, " | ".join(reasons), confluence

        # v21: Adaptive threshold based on trend strength
        if is_ranging: 
            threshold_mod = 1.3
        elif is_trending: 
            threshold_mod = 0.9
        else: 
            threshold_mod = 1.0
        mod = threshold_mod

        if score >= 80 * mod: 
            direction = "BUY"; strength = "VERY STRONG"
        elif score >= 60 * mod: 
            direction = "BUY"; strength = "STRONG"
        elif score >= 35 * mod: 
            direction = "BUY"; strength = "MODERATE"
        elif score <= -80 * mod: 
            direction = "SELL"; strength = "VERY STRONG"
        elif score <= -60 * mod: 
            direction = "SELL"; strength = "STRONG"
        elif score <= -35 * mod: 
            direction = "SELL"; strength = "MODERATE"
        else: 
            direction = "WAIT"; strength = "NEUTRAL"

        # v21: ATR-based levels
        atr = max(r.atr, cp * 0.002) if r.atr > 0 else cp * 0.002
        if direction == "BUY":
            sl = cp - atr * 1.8; tp = cp + atr * 4
            entry_low = cp - atr * 0.3; entry_high = cp + atr * 0.1
        elif direction == "SELL":
            sl = cp + atr * 1.8; tp = cp - atr * 4
            entry_low = cp - atr * 0.1; entry_high = cp + atr * 0.3
        else:
            sl = cp - atr * 1.5; tp = cp + atr * 2
            entry_low = cp - atr * 0.5; entry_high = cp + atr * 0.5

        # v21: Confidence calculation
        confluence_ratio = min(confluence / MIN_CONFLUENCE_THRESHOLD, 1.0)
        score_component = min(abs(score) * 0.6, 75.0)
        confl_bonus = confluence_ratio * 20.0
        raw_confidence = score_component + confl_bonus
        if confluence < 3 and abs(score) > 40: 
            raw_confidence *= 0.70
        confidence = round(min(max(raw_confidence, 10.0), 95.0), 1)

        # v21: Swing structure bonus/penalty
        if swing_struct and swing_struct.get("structure") != "UNKNOWN":
            struct = swing_struct["structure"]
            confirmed = swing_struct.get("confirmed", False)
            struct_aligns = ((direction == "BUY" and struct == "BULLISH") or (direction == "SELL" and struct == "BEARISH"))
            struct_conflicts = ((direction == "BUY" and struct == "BEARISH") or (direction == "SELL" and struct == "BULLISH"))
            if struct_aligns and confirmed: 
                confidence = round(min(confidence + 12.0, 97.0), 1)
                reasons.append(f"[+12%] Swing {struct} confirms {direction}")
            elif struct_aligns and not confirmed: 
                confidence = round(min(confidence + 5.0, 97.0), 1)
                reasons.append(f"[+5%] Swing {struct} partial")
            elif struct_conflicts and confirmed:
                confidence = round(max(confidence - 15.0, 10.0), 1)
                reasons.append(f"[-15%] Swing {struct} conflicts {direction}")
                if direction in ("BUY", "SELL") and confidence < 35.0: 
                    direction = "WAIT"; strength = "NEUTRAL"
                    reasons.append("[BLOCKED] Swing conflict")

        # v21: Meme coin + Funding + OI + OB checks (OB is auxiliary only)
        is_meme = self._is_meme_coin()
        funding_safe, funding_msg = self._check_funding_risk()
        oi_safe, oi_msg = self._check_oi_risk(direction)
        ob_safe, ob_msg = self._check_ob_risk(direction)

        if is_meme and direction in ("BUY", "SELL"):
            if confidence < 75: 
                direction = "WAIT"; strength = "NEUTRAL"; reasons.append("MEME COIN: Confidence < 75% -- Blocked")
            if not funding_safe: 
                direction = "WAIT"; strength = "NEUTRAL"; reasons.append(funding_msg)
            if not oi_safe: 
                direction = "WAIT"; strength = "NEUTRAL"; reasons.append(oi_msg)
            if not ob_safe: 
                direction = "WAIT"; strength = "NEUTRAL"; reasons.append(ob_msg)
            reasons.append(f"MAX LEVERAGE: {MAX_LEVERAGE_MEME}x")
        else:
            if not funding_safe: 
                reasons.append(funding_msg)
            if not oi_safe: 
                reasons.append(oi_msg)
            # v21: OB is auxiliary - only warn, never block alone
            if not ob_safe:
                reasons.append(ob_msg + " [AUX]")

        # v21: Volume liquidity check
        try:
            vol_24h_usdt = getattr(self, '_vol_24h_usdt', 0.0)
            if vol_24h_usdt > 0 and vol_24h_usdt < MIN_VOLUME_USDT and direction in ("BUY", "SELL"):
                direction = "WAIT"
                strength = "NEUTRAL"
                reasons.append(f"LOW LIQUIDITY: Vol {vol_24h_usdt/1e6:.1f}M < {MIN_VOLUME_USDT/1e6:.0f}M USDT")
        except Exception:
            pass

        # v21: Recent volume trend check
        try:
            if direction in ("BUY", "SELL"):
                with self.df_lock:
                    recent_vols = [float(c.v) for c in list(self.candle_deque)[-5:]]
                if len(recent_vols) >= 3:
                    vol_avg = sum(recent_vols) / len(recent_vols)
                    last_vol = recent_vols[-1]
                    vol_rising = recent_vols[-1] >= recent_vols[-2] >= recent_vols[-3] * 0.9
                    vol_above_avg = last_vol >= vol_avg * 0.85
                    if not vol_rising and not vol_above_avg:
                        reasons.append(f"VOL WEAK: volume declining/below avg ({last_vol/vol_avg*100:.0f}% of avg)")
                        score = int(score * 0.80)
        except Exception:
            pass

        # v21: BB Squeeze / Sideways block
        try:
            if direction in ("BUY", "SELL"):
                bb_squeeze_active = r.bb_squeeze
                atr_pct_now = (r.atr / cp) * 100 if cp > 0 else 999
                if bb_squeeze_active and atr_pct_now < 0.15:
                    direction = "WAIT"
                    strength = "NEUTRAL"
                    reasons.append(f"SIDEWAYS BLOCK: BB Squeeze + ATR {atr_pct_now:.3f}% (dead market)")
                elif bb_squeeze_active and atr_pct_now < 0.3:
                    reasons.append(f"SIDEWAYS WARN: BB Squeeze detected (ATR {atr_pct_now:.3f}%)")
                    score = int(score * 0.85)
        except Exception:
            pass

        if direction == "WAIT":
            confidence = max(15.0, confidence * 0.7)

        short_reasons = []
        for r_txt in reasons:
            r_txt = r_txt.strip()
            if len(r_txt) > 28: 
                r_txt = r_txt[:26] + ".."
            short_reasons.append(r_txt)
        reason = " | ".join(short_reasons[:5]) if short_reasons else "No signal"
        return direction, strength, round(entry_low, 6), round(entry_high, 6), round(tp, 6), round(sl, 6), round(confidence, 1), reason, confluence

    _xgb_model_cache: dict = {}
    _xgb_train_candle: dict = {}
    _xgb_model_cache_trending: dict = {}
    _xgb_train_candle_trending: dict = {}
    _xgb_model_cache_ranging: dict = {}
    _xgb_train_candle_ranging: dict = {}
    _xgb_model_cache_breakout: dict = {}
    _xgb_train_candle_breakout: dict = {}
    XGB_RETRAIN_CANDLES = 20

    def _build_xgb_features(self, closes, highs, lows, volumes,
                             rsi_arr, adx_arr, bb_pct_arr, bb_bw_arr) -> np.ndarray:
        n = len(closes)
        ema20 = np.zeros(n, dtype=np.float64)
        alpha_ema = 2.0 / (20 + 1)
        ema20[0] = closes[0]
        for i in range(1, n):
            ema20[i] = alpha_ema * closes[i] + (1 - alpha_ema) * ema20[i-1]

        macd_line, macd_signal, macd_hist = _calc_macd(closes)
        vwap_vals = _calc_vwap(highs, lows, closes, volumes)
        stoch_k, stoch_d, stoch_j = _calc_stoch_rsi(closes)

        # v21: Sequence features (simulating LSTM/GRU memory)
        feats = []
        for i in range(n):
            c = closes[i]
            ret1  = (c / closes[i-1] - 1.0) if i >= 1 and closes[i-1] > 1e-9 else 0.0
            ret3  = (c / closes[i-3] - 1.0) if i >= 3 and closes[i-3] > 1e-9 else ret1
            ret5  = (c / closes[i-5] - 1.0) if i >= 5 and closes[i-5] > 1e-9 else ret1
            ret10 = (c / closes[i-10] - 1.0) if i >= 10 and closes[i-10] > 1e-9 else ret5
            log1  = float(np.log(c / closes[i-1])) if i >= 1 and closes[i-1] > 1e-9 else 0.0
            rsi_v = float(rsi_arr[i]) if i < len(rsi_arr) else 50.0
            rsi_n = rsi_v / 100.0
            rdlt  = float((rsi_arr[i] - rsi_arr[i-1]) / 100.0) if i >= 1 and i < len(rsi_arr) else 0.0
            adx_v = float(adx_arr[i]) if i < len(adx_arr) else 20.0
            adx_n = min(adx_v / 60.0, 1.0)
            bbp  = float(bb_pct_arr[i]) if i < len(bb_pct_arr) else 0.5
            bbbw = float(bb_bw_arr[i])  if i < len(bb_bw_arr)  else 0.02
            vol_a  = float(np.mean(volumes[max(0, i-20):i+1])) if i > 0 else float(volumes[i])
            vratio = float(volumes[i]) / vol_a if vol_a > 1e-9 else 1.0
            hlrat  = float((highs[i] - lows[i]) / c) if c > 1e-9 else 0.0
            ema_d  = (c - ema20[i]) / c if c > 1e-9 else 0.0
            macd_d = float(macd_hist[i]) / c if c > 1e-9 else 0.0
            macd_s = 1.0 if macd_line[i] > macd_signal[i] else 0.0
            vwap_d = (c - vwap_vals[i]) / c if c > 1e-9 else 0.0
            stoch_k_n = float(stoch_k[i]) / 100.0 if i < len(stoch_k) else 0.5
            stoch_d_n = float(stoch_d[i]) / 100.0 if i < len(stoch_d) else 0.5
            stoch_j_n = float(stoch_j[i]) / 100.0 if i < len(stoch_j) else 0.5

            # v21: Sequence momentum features (LSTM-like)
            mom3 = (c - closes[max(0, i-3)]) / c if c > 1e-9 else 0.0
            mom5 = (c - closes[max(0, i-5)]) / c if c > 1e-9 else 0.0
            mom10 = (c - closes[max(0, i-10)]) / c if c > 1e-9 else 0.0
            vol_trend = float(np.mean(volumes[max(0, i-5):i+1])) / vol_a if vol_a > 1e-9 else 1.0

            feats.append([
                ret1, ret3, ret5, ret10, log1, rsi_n, rdlt, adx_n, bbp, bbbw, 
                vratio, hlrat, ema_d, macd_d, macd_s, vwap_d, 
                stoch_k_n, stoch_d_n, stoch_j_n,
                mom3, mom5, mom10, vol_trend  # v21: Enhanced sequence features
            ])
        arr = np.array(feats, dtype=np.float64)
        arr = np.clip(arr, -0.5, 0.5)
        return arr

    def _train_xgb_model(self, closes, highs, lows, volumes,
                         rsi_arr, adx_arr, bb_pct_arr, bb_bw_arr,
                         shift=None, regime=None):
        if not XGBOOST_OK:
            return None, 0.02
        n = len(closes)
        min_train = 60
        if shift is None:
            shift = _get_lookahead_shift(getattr(self, 'interval', '1h'))
        if n < min_train + shift:
            return None, 0.02
        try:
            X = self._build_xgb_features(closes, highs, lows, volumes,
                                         rsi_arr, adx_arr, bb_pct_arr, bb_bw_arr)
            y = np.array([
                (closes[i + shift] - closes[i]) / closes[i]
                if closes[i] > 1e-9 else 0.0
                for i in range(n - shift)
            ], dtype=np.float64)
            X_train = X[:n - shift]

            valid_mask = (
                np.isfinite(X_train).all(axis=1) &
                np.isfinite(y)
            )
            X_train = X_train[valid_mask]
            y_clean = y[valid_mask]

            if len(X_train) < min_train:
                logging.warning("XGB train: insufficient clean samples (%d)", len(X_train))
                return None, 0.02

            # v21: Dynamic regularization based on data size and regime
            reg_lambda = 1.5 if len(X_train) < 150 else 1.0
            reg_alpha = 0.5 if len(X_train) < 150 else 0.3

            # v21: Regime-aware hyperparameters
            if regime == "RANGING":
                max_depth = 2
                learning_rate = 0.03
            elif regime == "TRENDING":
                max_depth = 4
                learning_rate = 0.06
            else:
                max_depth = 3
                learning_rate = 0.05

            model = xgb.XGBRegressor(
                n_estimators=120,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=0.75,
                colsample_bytree=0.75,
                min_child_weight=4,
                reg_lambda=reg_lambda,
                reg_alpha=reg_alpha,
                random_state=42,
                verbosity=0,
                n_jobs=1,
            )
            model.fit(X_train, y_clean)

            # Calculate residual std for confidence intervals
            try:
                preds = model.predict(X_train)
                residuals = y_clean - preds
                res_std = float(np.std(residuals))
                if not np.isfinite(res_std) or res_std <= 0:
                    res_std = 0.02
            except Exception:
                res_std = 0.02

            return model, res_std
        except Exception as e:
            logging.warning("XGB train error: %s", str(e)[:80])
            return None, 0.02

    def _compute_price_prediction(self, df: pd.DataFrame, cp: float, atr: float, forecast_direction: str = "WAIT") -> dict:
        MIN_LOOKBACK = 20
        shift = _get_lookahead_shift(getattr(self, 'interval', '1h'))
        result = {
            "pred_price": 0.0, "pred_high": 0.0, "pred_low": 0.0,
            "pred_pct": 0.0, "pred_candles": shift,
            "pred_1c": 0.0, "pred_1c_pct": 0.0,
            "pred_3c": 0.0, "pred_3c_pct": 0.0,
            "pred_5c": 0.0, "pred_5c_pct": 0.0,
            "confidence_95": 0.0, "residual_std": 0.0,
            "pred_label": "N/A", "pred_color": "#888",
            "pred_method": "None", "pred_live_price": 0.0, "pred_r2": 0.0,
            "slope": 0.0, "slope_dir": "NEUTRAL",
            "bb_entry": 0.0, "bb_exit": 0.0,
            "bb_upper": 0.0, "bb_lower": 0.0, "bb_mid": 0.0,
            "xgb_pct": 0.0, "adx_regime": "UNKNOWN",
            "adx_val": 0.0, "rsi_zone": "NEUTRAL",
        }
        logging.info("[AI-PRED] Starting prediction for %s @ %s, dir=%s, cp=%.4f", self.symbol, self.interval, forecast_direction, cp)
        try:
            if len(df) < MIN_LOOKBACK:
                return result

            live_price = self.price if self.price > 0 else self.mark_price if self.mark_price > 0 else cp
            if live_price <= 0:
                logging.warning("[AI-PRED] live_price <= 0, aborting prediction")
                return result
            logging.info("[AI-PRED] live_price=%.4f, shift=%d, lookback=%d", live_price, shift, max(MIN_LOOKBACK, AI_PRED_LOOKBACK))

            result["pred_live_price"] = round(float(live_price), 6)
            atr_safe = max(float(atr), float(live_price) * 0.001)

            lookback = max(MIN_LOOKBACK, AI_PRED_LOOKBACK)
            with self.df_lock:
                raw_candles = list(self.candle_deque)[-lookback:]
                raw_closes  = [float(c.c) for c in raw_candles]
                raw_highs   = [float(c.h) for c in raw_candles]
                raw_lows    = [float(c.l) for c in raw_candles]
                raw_volumes = [float(c.v) for c in raw_candles]

            if len(raw_closes) < MIN_LOOKBACK:
                logging.warning("[AI-PRED] candle_deque insufficient (%d), falling back to df", len(raw_closes))
                raw_closes  = [float(v) for v in df["c"].values[-lookback:]]
                raw_highs   = [float(v) for v in df["h"].values[-lookback:]]
                raw_lows    = [float(v) for v in df["l"].values[-lookback:]]
                raw_volumes = [float(v) for v in df["v"].values[-lookback:]]
            logging.info("[AI-PRED] Data loaded: n=%d closes, range=%.4f-%.4f", len(raw_closes), min(raw_closes), max(raw_closes))

            closes  = np.array(raw_closes,  dtype=np.float64)
            highs   = np.array(raw_highs,   dtype=np.float64)
            lows    = np.array(raw_lows,    dtype=np.float64)
            volumes = np.array(raw_volumes, dtype=np.float64)
            n = len(closes)

            deltas  = np.diff(closes, prepend=closes[0])
            gains   = np.where(deltas > 0, deltas, 0.0)
            losses  = np.where(deltas < 0, -deltas, 0.0)

            alpha = 1.0 / 14.0
            avg_g = np.zeros(n); avg_l = np.zeros(n)
            avg_g[0] = gains[0]; avg_l[0] = losses[0]
            for i in range(1, n):
                avg_g[i] = alpha * gains[i]  + (1 - alpha) * avg_g[i-1]
                avg_l[i] = alpha * losses[i] + (1 - alpha) * avg_l[i-1]
            rs_arr  = np.where(avg_l > 1e-9, avg_g / avg_l, 100.0)
            rsi_arr = 100.0 - (100.0 / (1.0 + rs_arr))
            rsi_raw = float(rsi_arr[-1])

            if rsi_raw >= 75:
                rsi_zone = "OVERBOUGHT_EXTREME"
            elif rsi_raw >= 65:
                rsi_zone = "OVERBOUGHT"
            elif rsi_raw <= 25:
                rsi_zone = "OVERSOLD_EXTREME"
            elif rsi_raw <= 35:
                rsi_zone = "OVERSOLD"
            elif rsi_raw >= 50:
                rsi_zone = "BULLISH_NEUTRAL"
            else:
                rsi_zone = "BEARISH_NEUTRAL"
            result["rsi_zone"] = rsi_zone

            def _calc_adx_np(h, l, c, period=14):
                n_ = len(c)
                if n_ < period + 1:
                    return np.full(n_, 20.0), np.full(n_, 10.0), np.full(n_, 10.0)
                tr_ = np.maximum(h[1:]-l[1:], np.maximum(np.abs(h[1:]-c[:-1]), np.abs(l[1:]-c[:-1])))
                pdm = np.where((h[1:]-h[:-1]) > (l[:-1]-l[1:]), np.maximum(h[1:]-h[:-1], 0), 0.0)
                ndm = np.where((l[:-1]-l[1:]) > (h[1:]-h[:-1]), np.maximum(l[:-1]-l[1:], 0), 0.0)
                a = 1.0 / period
                atr_ = np.zeros(n_-1); pdi_ = np.zeros(n_-1); ndi_ = np.zeros(n_-1)
                atr_[0]=tr_[0]; pdi_[0]=pdm[0]; ndi_[0]=ndm[0]
                for i in range(1, n_-1):
                    atr_[i] = a*tr_[i]  + (1-a)*atr_[i-1]
                    pdi_[i] = a*pdm[i]  + (1-a)*pdi_[i-1]
                    ndi_[i] = a*ndm[i]  + (1-a)*ndi_[i-1]
                eps = 1e-9
                pdi_n = 100 * pdi_ / (atr_ + eps)
                ndi_n = 100 * ndi_ / (atr_ + eps)
                dx    = 100 * np.abs(pdi_n - ndi_n) / (pdi_n + ndi_n + eps)
                adx_  = np.zeros(n_-1); adx_[0] = dx[0]
                for i in range(1, n_-1):
                    adx_[i] = a*dx[i] + (1-a)*adx_[i-1]
                adx_full  = np.concatenate([[adx_[0]],  adx_])
                pdi_full  = np.concatenate([[pdi_n[0]], pdi_n])
                ndi_full  = np.concatenate([[ndi_n[0]], ndi_n])
                return adx_full[:n], pdi_full[:n], ndi_full[:n]

            adx_arr14, pdi14, ndi14 = _calc_adx_np(highs, lows, closes, 14)
            adx_arr28, _,    _     = _calc_adx_np(highs, lows, closes, 28)
            adx_val    = float(adx_arr14[-1])
            adx_val28  = float(adx_arr28[-1])
            di_plus    = float(pdi14[-1])
            di_minus   = float(ndi14[-1])

            if adx_val >= 28 and adx_val28 >= 22:
                adx_regime = "TRENDING"
            elif adx_val >= 20 and adx_val28 >= 18:
                adx_regime = "MIXED"
            elif adx_val < 18 or adx_val28 < 15:
                adx_regime = "RANGING"
            else:
                adx_regime = "MIXED"
            result["adx_regime"] = adx_regime
            result["adx_val"]    = round(adx_val, 2)

            # Compute enhanced indicators for bias & XGB
            macd_line, macd_signal, macd_hist = _calc_macd(closes)
            vwap_vals = _calc_vwap(highs, lows, closes, volumes)
            stoch_k, stoch_d, stoch_j = _calc_stoch_rsi(closes)

            bb_period = 20
            if n >= bb_period:
                bb_sma = np.array([np.mean(closes[max(0,i-bb_period+1):i+1]) for i in range(n)])
                bb_std = np.array([np.std(closes[max(0,i-bb_period+1):i+1], ddof=0) for i in range(n)])
            else:
                bb_sma = np.full(n, float(live_price))
                bb_std = np.full(n, float(live_price) * 0.01)

            bb_upper_arr = bb_sma + 2.0 * bb_std
            bb_lower_arr = bb_sma - 2.0 * bb_std
            bb_bw_arr    = (bb_upper_arr - bb_lower_arr) / (bb_sma + 1e-9)
            bb_pct_arr   = (closes - bb_lower_arr) / (bb_upper_arr - bb_lower_arr + 1e-9)

            bb_upper_cur = float(bb_upper_arr[-1])
            bb_lower_cur = float(bb_lower_arr[-1])
            bb_mid_cur   = float(bb_sma[-1])
            bb_bw_cur    = float(bb_bw_arr[-1])
            bb_pct_cur   = float(bb_pct_arr[-1])

            result["bb_upper"] = round(bb_upper_cur, 6)
            result["bb_lower"] = round(bb_lower_cur, 6)
            result["bb_mid"]   = round(bb_mid_cur, 6)

            if forecast_direction == "BUY":
                if adx_regime == "RANGING":
                    if float(live_price) <= bb_lower_cur * 1.005:
                        result["bb_entry"] = round(float(live_price), 6)
                    else:
                        result["bb_entry"] = round(bb_lower_cur, 6)
                    result["bb_exit"] = round(bb_mid_cur, 6)
                else:
                    result["bb_entry"] = round(float(live_price), 6)
                    result["bb_exit"]  = round(bb_upper_cur, 6)
            elif forecast_direction == "SELL":
                if adx_regime == "RANGING":
                    if float(live_price) >= bb_upper_cur * 0.995:
                        result["bb_entry"] = round(float(live_price), 6)
                    else:
                        result["bb_entry"] = round(bb_upper_cur, 6)
                    result["bb_exit"] = round(bb_mid_cur, 6)
                else:
                    result["bb_entry"] = round(float(live_price), 6)
                    result["bb_exit"]  = round(bb_lower_cur, 6)
            else:
                result["bb_entry"] = round(float(live_price), 6)
                result["bb_exit"]  = round(bb_upper_cur if float(live_price) < bb_mid_cur else bb_lower_cur, 6)

            tr_arr   = np.maximum(highs[1:] - lows[1:],
                       np.maximum(np.abs(highs[1:] - closes[:-1]),
                                  np.abs(lows[1:]  - closes[:-1])))
            atr_live = float(np.mean(tr_arr[-14:])) if len(tr_arr) >= 14 else atr_safe

            vol_avg      = float(np.mean(volumes)) if len(volumes) > 0 else 1.0
            vol_last3    = float(np.mean(volumes[-3:])) if len(volumes) >= 3 else vol_avg
            vol_momentum = vol_last3 / vol_avg if vol_avg > 1e-9 else 1.0

            x_all = np.arange(n, dtype=np.float64).reshape(-1, 1)
            sw    = np.exp(np.linspace(0.0, 6.0, n))

            if SKLEARN_OK and n >= MIN_LOOKBACK:
                model_slope = LinearRegression()
                model_slope.fit(x_all, closes, sample_weight=sw)
                slope_per_candle = float(model_slope.coef_[0])
                r2_slope = float(model_slope.score(x_all, closes, sample_weight=sw))
                future_idx = n + shift - 1
                pred_lr = float(model_slope.predict(np.array([[future_idx]]))[0])
            else:
                wt = sw / sw.sum(); xf = x_all.flatten()
                mx = np.dot(wt, xf); my = np.dot(wt, closes)
                cov = np.dot(wt, (xf-mx)*(closes-my))
                var = np.dot(wt, (xf-mx)**2)
                slope_per_candle = cov / var if var > 1e-12 else 0.0
                pred_lr  = float(closes[-1]) + slope_per_candle * shift
                r2_slope = 0.0

            slope_pct = (slope_per_candle / float(live_price)) * 100
            slope_dir = "BULLISH" if slope_pct > 0.01 else "BEARISH" if slope_pct < -0.01 else "NEUTRAL"
            result["slope"]    = round(slope_per_candle, 8)
            result["slope_dir"]= slope_dir
            result["pred_r2"]  = round(r2_slope, 3)

            xgb_pct  = 0.0
            xgb_conf = 0.0
            xgb_used = False
            residual_std = 0.02  # Default fallback

            df_closes  = df["c"].values.astype(np.float64)
            df_highs   = df["h"].values.astype(np.float64)
            df_lows    = df["l"].values.astype(np.float64)
            df_volumes = df["v"].values.astype(np.float64)
            df_n       = len(df_closes)

            if XGBOOST_OK and df_n >= 40:
                df_deltas = np.diff(df_closes, prepend=df_closes[0])
                df_gains  = np.where(df_deltas > 0, df_deltas, 0.0)
                df_losses = np.where(df_deltas < 0, -df_deltas, 0.0)
                df_avg_g  = np.zeros(df_n); df_avg_l = np.zeros(df_n)
                df_avg_g[0]=df_gains[0]; df_avg_l[0]=df_losses[0]
                for i in range(1, df_n):
                    df_avg_g[i] = alpha*df_gains[i]  + (1-alpha)*df_avg_g[i-1]
                    df_avg_l[i] = alpha*df_losses[i] + (1-alpha)*df_avg_l[i-1]
                df_rsi = 100.0 - (100.0/(1.0 + np.where(df_avg_l>1e-9, df_avg_g/df_avg_l, 100.0)))

                df_adx_arr, _, _ = _calc_adx_np(df_highs, df_lows, df_closes, 14)

                df_bb_sma = np.array([np.mean(df_closes[max(0,i-20):i+1]) for i in range(df_n)])
                df_bb_std = np.array([np.std(df_closes[max(0,i-20):i+1], ddof=0) for i in range(df_n)])
                df_bbupp  = df_bb_sma + 2*df_bb_std
                df_bblow  = df_bb_sma - 2*df_bb_std
                df_bb_bw  = (df_bbupp - df_bblow) / (df_bb_sma + 1e-9)
                df_bb_pct = (df_closes - df_bblow) / (df_bbupp - df_bblow + 1e-9)

                candle_pats = self._detect_candle_patterns(df)
                _hammer       = float(candle_pats.get("hammer", 0))
                _shooting     = float(candle_pats.get("shooting_star", 0))
                _bull_engulf  = float(candle_pats.get("bull_engulf", 0))
                _bear_engulf  = float(candle_pats.get("bear_engulf", 0))

                cache_key = (getattr(self, 'symbol', ''), getattr(self, 'interval', ''))
                last_t    = int(df["t"].iloc[-1]) if "t" in df.columns else 0
                prev_t    = BinanceRadarPro._xgb_train_candle.get(cache_key, -1)
                candles_since = abs(last_t - prev_t) // TF_MS.get(getattr(self, 'interval', '1h'), 3_600_000) if prev_t > 0 else 9999

                # Real-time retrain trigger: if recent prediction error > 1%
                force_retrain = False
                if self._last_prediction is not None:
                    last_pred = self._last_prediction
                    if last_pred.get("evaluated", False) and last_pred.get("pred_price", 0) > 0:
                        actual = float(live_price)
                        pred_err = abs(actual - last_pred["pred_price"]) / last_pred["pred_price"]
                        if pred_err > 0.01:
                            force_retrain = True
                            self._force_retrain_counter += 1
                            logging.info("Force retrain: prediction error %.2f%% > 1%%", pred_err*100)

                # Select regime-specific cache
                if adx_regime == "TRENDING":
                    cache = BinanceRadarPro._xgb_model_cache_trending
                    train_candle_cache = BinanceRadarPro._xgb_train_candle_trending
                elif adx_regime == "RANGING":
                    cache = BinanceRadarPro._xgb_model_cache_ranging
                    train_candle_cache = BinanceRadarPro._xgb_train_candle_ranging
                elif adx_regime == "BREAKOUT":
                    cache = BinanceRadarPro._xgb_model_cache_breakout
                    train_candle_cache = BinanceRadarPro._xgb_train_candle_breakout
                else:
                    cache = BinanceRadarPro._xgb_model_cache
                    train_candle_cache = BinanceRadarPro._xgb_train_candle

                if cache_key not in cache or candles_since >= BinanceRadarPro.XGB_RETRAIN_CANDLES or force_retrain:
                    logging.info("[AI-PRED] Training XGB model for %s %s regime=%s", cache_key[0], cache_key[1], adx_regime)
                    model_xgb, res_std = self._train_xgb_model(
                        df_closes, df_highs, df_lows, df_volumes,
                        df_rsi, df_adx_arr, df_bb_pct, df_bb_bw,
                        shift=shift, regime=adx_regime)
                    logging.info("[AI-PRED] XGB train result: model=%s, res_std=%.4f", "OK" if model_xgb is not None else "FAIL", res_std)
                    if model_xgb is not None:
                        cache[cache_key] = model_xgb
                        train_candle_cache[cache_key] = last_t
                        BinanceRadarPro._xgb_model_cache[cache_key + "_std"] = res_std
                        logging.info("XGB retrained: %s %s regime=%s shift=%d (n=%d)", cache_key[0], cache_key[1], adx_regime, shift, df_n)

                if cache_key in cache:
                    model_xgb = cache[cache_key]
                    residual_std = BinanceRadarPro._xgb_model_cache.get(cache_key + "_std", 0.02)
                    feat_last = self._build_xgb_features(
                        closes[-30:], highs[-30:], lows[-30:], volumes[-30:],
                        rsi_arr[-30:], adx_arr14[-30:], bb_pct_arr[-30:], bb_bw_arr[-30:]
                    )
                    if len(feat_last) > 0:
                        xgb_pct_raw = float(model_xgb.predict(feat_last[[-1]])[0])
                        xgb_pct = float(np.clip(xgb_pct_raw, -0.15, 0.15))
                        xgb_used = True
                        result["xgb_pct"] = round(xgb_pct * 100, 3)

            if xgb_used:
                xgb_pred_price = float(live_price) * (1.0 + xgb_pct)
                # v21: Dynamic weighting: trust XGB more in trending, LR more in ranging
                xgb_weight = 0.55 + (0.15 if adx_regime == "TRENDING" else 0.0) - (0.10 if adx_regime == "RANGING" else 0.0)
                xgb_weight = max(0.35, min(0.75, xgb_weight))
                lr_weight = 1.0 - xgb_weight
                pred_raw = xgb_weight * xgb_pred_price + lr_weight * pred_lr
                method_tag = f"XGB({xgb_weight:.0%})+LR({lr_weight:.0%}) ADX={adx_val:.0f} RSI={rsi_raw:.0f}"
            else:
                pred_raw = pred_lr
                method_tag = f"LR-WSlope ADX={adx_val:.0f} RSI={rsi_raw:.0f}"

            # Calculate momentum continuation for small timeframes
            momentum_continuation = 0.0
            if n >= 6:
                recent_bodies = [closes[i] - closes[i-1] for i in range(-5, 0)]
                positive_bodies = sum(1 for b in recent_bodies if b > 0)
                negative_bodies = sum(1 for b in recent_bodies if b < 0)
                if positive_bodies >= 4:
                    momentum_continuation = +1.0  # Strong upward continuation
                elif negative_bodies >= 4:
                    momentum_continuation = -1.0  # Strong downward continuation

            # Small timeframe multiplier: reduce RSI bias when momentum is strong
            tf_rsi_mult = 0.35 if self.interval in ("1m", "3m", "5m", "15m") else 1.0

            rsi_bias = 0.0
            if rsi_raw >= 75:
                # If strong momentum continuation on small TF, don't punish overbought
                if momentum_continuation > 0 and tf_rsi_mult < 1.0:
                    rsi_bias = -atr_live * 0.10 * tf_rsi_mult
                else:
                    rsi_bias = -atr_live * 0.45 * tf_rsi_mult
            elif rsi_raw >= 65:
                if momentum_continuation > 0 and tf_rsi_mult < 1.0:
                    rsi_bias = -atr_live * 0.05 * tf_rsi_mult
                else:
                    rsi_bias = -atr_live * 0.20 * tf_rsi_mult
            elif rsi_raw <= 25:
                if momentum_continuation < 0 and tf_rsi_mult < 1.0:
                    rsi_bias = +atr_live * 0.10 * tf_rsi_mult
                else:
                    rsi_bias = +atr_live * 0.45 * tf_rsi_mult
            elif rsi_raw <= 35:
                if momentum_continuation < 0 and tf_rsi_mult < 1.0:
                    rsi_bias = +atr_live * 0.05 * tf_rsi_mult
                else:
                    rsi_bias = +atr_live * 0.20 * tf_rsi_mult

            rsi_delta5 = float(rsi_arr[-1] - rsi_arr[-5]) if n >= 5 else 0.0
            price_delta5 = (closes[-1] - closes[-5]) / closes[-5] * 100 if n >= 5 and closes[-5] > 0 else 0.0
            if rsi_delta5 > 3 and price_delta5 < 0:
                rsi_bias += atr_live * 0.15
            elif rsi_delta5 < -3 and price_delta5 > 0:
                rsi_bias -= atr_live * 0.15

            adx_bias_multiplier = 1.0
            if adx_regime == "TRENDING":
                adx_bias_multiplier = 1.20
            elif adx_regime == "RANGING":
                adx_bias_multiplier = 0.55
            else:
                adx_bias_multiplier = 0.85

            vol_bias = 0.0
            if vol_momentum > 1.5 and slope_dir == "BULLISH":
                vol_bias = +atr_live * 0.20
            elif vol_momentum > 1.5 and slope_dir == "BEARISH":
                vol_bias = -atr_live * 0.20

            atm = atr_live * shift * 0.30
            enhanced_bias = self._build_enhanced_bias(
                df, closes, highs, lows, volumes, rsi_arr, adx_arr14, macd_line, macd_signal, macd_hist,
                vwap_vals, stoch_k, stoch_d, stoch_j, forecast_direction, float(live_price), atr_live
            )
            total_bias = (rsi_bias + vol_bias + enhanced_bias) * adx_bias_multiplier

            # Multi-horizon prediction function
            def _predict_for_horizon(horizon_shift):
                if SKLEARN_OK and n >= MIN_LOOKBACK:
                    future_idx = n + horizon_shift - 1
                    pred_lr_h = float(model_slope.predict(np.array([[future_idx]]))[0])
                else:
                    pred_lr_h = float(closes[-1]) + slope_per_candle * horizon_shift
                if xgb_used:
                    xgb_pct_h = xgb_pct * (horizon_shift / max(shift, 1))
                    xgb_pred_h = float(live_price) * (1.0 + xgb_pct_h)
                    pred_h = xgb_weight * xgb_pred_h + lr_weight * pred_lr_h
                else:
                    pred_h = pred_lr_h
                bias_h = total_bias * (horizon_shift / max(shift, 1))
                if forecast_direction == "BUY":
                    # Small TF: if strong momentum, bias should be less restrictive
                    is_small_tf = self.interval in ("1m", "3m", "5m", "15m")
                    min_pred = float(live_price) + atr_live * 0.5 * horizon_shift
                    if is_small_tf and momentum_continuation > 0:
                        min_pred = float(live_price) + atr_live * 0.8 * horizon_shift  # Higher floor
                    pred_h = max(pred_h + bias_h, min_pred)
                    if rsi_zone == "OVERBOUGHT_EXTREME":
                        # Small TF: allow higher overbought extension
                        max_ext = float(live_price) + atr_live * (2.5 if is_small_tf else 1.5)
                        pred_h = min(pred_h, max_ext)
                elif forecast_direction == "SELL":
                    is_small_tf = self.interval in ("1m", "3m", "5m", "15m")
                    max_pred = float(live_price) - atr_live * 0.5 * horizon_shift
                    if is_small_tf and momentum_continuation < 0:
                        max_pred = float(live_price) - atr_live * 0.8 * horizon_shift  # Lower ceiling
                    pred_h = min(pred_h + bias_h, max_pred)
                    if rsi_zone == "OVERSOLD_EXTREME":
                        max_ext = float(live_price) - atr_live * (2.5 if is_small_tf else 1.5)
                        pred_h = max(pred_h, max_ext)
                else:
                    pred_h = float(live_price) + (pred_h - float(live_price)) * 0.40 + bias_h * 0.50
                # EMA bias per horizon
                ema20_v  = float(df["ema20"].iloc[-1])  if "ema20"  in df.columns else float(live_price)
                ema50_v  = float(df["ema50"].iloc[-1])  if "ema50"  in df.columns else float(live_price)
                ema200_v = float(df["ema200"].iloc[-1]) if "ema200" in df.columns else float(live_price)
                eb = 0.0
                if float(live_price) > ema200_v > 0: eb += 0.00015 * horizon_shift
                else: eb -= 0.00015 * horizon_shift
                if ema20_v > ema50_v: eb += 0.00008 * horizon_shift
                else: eb -= 0.00008 * horizon_shift
                pred_h = pred_h * (1.0 + eb)
                return pred_h

            pred_1c = _predict_for_horizon(1)
            pred_3c = _predict_for_horizon(min(3, shift))
            pred_5c = _predict_for_horizon(min(5, shift + 2))
            pred_main = _predict_for_horizon(shift)

            # Confidence interval (95%)
            recent_errors = list(self._prediction_errors)[-20:] if len(self._prediction_errors) > 0 else []
            if len(recent_errors) > 5:
                error_std = float(np.std(recent_errors))
                confidence_95 = 1.96 * error_std * float(live_price)
            else:
                confidence_95 = 1.96 * residual_std * float(live_price)
            result["residual_std"] = round(residual_std, 4)
            result["confidence_95"] = round(confidence_95, 2)

            # Track prediction for real-time retrain
            self._last_prediction = {
                "pred_price": round(pred_main, 6),
                "pred_time": time.time(),
                "evaluated": False,
            }

            pred = pred_main
            method_tag += f" [↑{forecast_direction}]" if forecast_direction in ("BUY", "SELL") else " [~WAIT]"
            logging.info("[AI-PRED] pred_main=%.4f (change=%.2f%%), method=%s", pred_main, (pred_main-live_price)/live_price*100, method_tag)

            r2_tightness = max(0.40, min(r2_slope, 1.0))
            tf_band_scale = {"5m": 1.0, "15m": 1.2, "1h": 1.5, "4h": 2.0, "1d": 3.0}
            tf_scale = tf_band_scale.get(getattr(self, 'interval', '1h'), 1.5)
            # Regime-aware band multiplier
            regime_mult = 1.0
            if adx_regime == "RANGING":
                regime_mult = 0.55  # Tighter bands in ranging
            elif adx_regime == "TRENDING":
                regime_mult = 1.25  # Wider bands in trending
            else:
                regime_mult = 0.85
            atr_band_multiplier = (1.0 + shift * 0.12) * tf_scale * (2.0 - r2_tightness) * regime_mult
            if vol_momentum > 1.5:
                atr_band_multiplier *= 1.15
            if adx_regime == "RANGING":
                atr_band_multiplier *= 0.65
            atr_band = atr_live * atr_band_multiplier

            def _clamp_prediction(pred_h, horizon):
                band = atr_band * (horizon / max(shift, 1))
                raw_high = pred_h + band
                raw_low  = max(pred_h - band, 0.0)
                pred_high_final = min(raw_high, bb_upper_cur * 1.005)
                pred_low_final  = max(raw_low,  bb_lower_cur * 0.995)
                with self.intel_lock:
                    sr_levels_now = list(self.intel_sr_levels)
                if sr_levels_now:
                    res_above = [l["price"] for l in sr_levels_now if l["type"]=="R" and l["price"] > float(live_price)]
                    sup_below = [l["price"] for l in sr_levels_now if l["type"]=="S" and l["price"] < float(live_price)]
                    if res_above:
                        nearest_res = min(res_above)
                        if pred_high_final > nearest_res: pred_high_final = nearest_res
                    if sup_below:
                        nearest_sup = max(sup_below)
                        if pred_low_final < nearest_sup: pred_low_final = nearest_sup
                return pred_high_final, pred_low_final

            ph, pl = _clamp_prediction(pred_main, shift)
            price_change_pct = (pred_main - float(live_price)) / float(live_price) * 100
            result["pred_price"]   = round(pred_main, 6)
            result["pred_high"]    = round(ph, 6)
            result["pred_low"]     = round(pl, 6)
            result["pred_pct"]     = round(price_change_pct, 2)
            result["pred_candles"] = shift
            result["pred_method"]  = method_tag

            # Multi-horizon results
            ph1, pl1 = _clamp_prediction(pred_1c, 1)
            result["pred_1c"] = round(pred_1c, 6)
            result["pred_1c_pct"] = round((pred_1c - float(live_price)) / float(live_price) * 100, 2)
            ph3, pl3 = _clamp_prediction(pred_3c, min(3, shift))
            result["pred_3c"] = round(pred_3c, 6)
            result["pred_3c_pct"] = round((pred_3c - float(live_price)) / float(live_price) * 100, 2)
            ph5, pl5 = _clamp_prediction(pred_5c, min(5, shift + 2))
            result["pred_5c"] = round(pred_5c, 6)
            result["pred_5c_pct"] = round((pred_5c - float(live_price)) / float(live_price) * 100, 2)

            try:
                roc3_now  = float((closes[-1] - closes[-4]) / closes[-4] * 100) if n >= 4 and closes[-4] > 0 else 0.0
                roc3_prev = float((closes[-4] - closes[-7]) / closes[-7] * 100) if n >= 7 and closes[-7] > 0 else 0.0
                momentum_decelerating = (roc3_now > 0 and roc3_prev > 0 and roc3_now < roc3_prev * 0.75)

                if forecast_direction == "BUY":
                    # Small TF: require BOTH extreme RSI AND momentum deceleration for reversal
                    is_small_tf = self.interval in ("1m", "3m", "5m", "15m")
                    if (rsi_raw >= 75 and (momentum_decelerating or bb_pct_cur >= 0.95)) or                         (not is_small_tf and rsi_raw >= 70 and bb_pct_cur >= 0.90):
                        pred_class = "REVERSE_DOWN"
                        pred_class_color = "#cf304a"
                    elif momentum_decelerating and rsi_raw >= 60:
                        pred_class = "SIDEWAYS"
                        pred_class_color = "#f0b90b"
                    else:
                        pred_class = "CONTINUE_UP"
                        pred_class_color = "#02c076"
                elif forecast_direction == "SELL":
                    is_small_tf = self.interval in ("1m", "3m", "5m", "15m")
                    if (rsi_raw <= 25 and (momentum_decelerating or bb_pct_cur <= 0.05)) or                         (not is_small_tf and rsi_raw <= 30 and bb_pct_cur <= 0.10):
                        pred_class = "REVERSE_UP"
                        pred_class_color = "#02c076"
                    elif momentum_decelerating and rsi_raw <= 40:
                        pred_class = "SIDEWAYS"
                        pred_class_color = "#f0b90b"
                    else:
                        pred_class = "CONTINUE_DOWN"
                        pred_class_color = "#cf304a"
                else:
                    pred_class = "SIDEWAYS"
                    pred_class_color = "#f0b90b"
                result["pred_class"]       = pred_class
                result["pred_class_color"] = pred_class_color
                logging.debug("RULE4 Classification: %s (RSI=%.1f, BB%%=%.2f, MomDecay=%s)",
                              pred_class, rsi_raw, bb_pct_cur, momentum_decelerating)
            except Exception:
                result["pred_class"]       = "UNKNOWN"
                result["pred_class_color"] = "#888"

            if forecast_direction == "BUY":
                result["pred_label"] = "BULLISH"; result["pred_color"] = "#02c076"
            elif forecast_direction == "SELL":
                result["pred_label"] = "BEARISH"; result["pred_color"] = "#cf304a"
            else:
                th = {"5m":0.06,"15m":0.10,"1h":0.18,"4h":0.22,"1d":0.28}.get(getattr(self,'interval','1h'),0.18)
                if result["pred_pct"] > th:    result["pred_label"]="BULLISH"; result["pred_color"]="#02c076"
                elif result["pred_pct"] < -th: result["pred_label"]="BEARISH"; result["pred_color"]="#cf304a"
                else:                        result["pred_label"]="NEUTRAL";  result["pred_color"]="#f0b90b"

            # Ultimate fallback: if prediction is 0 or unrealistic, use EMA-based directional estimate
            if result["pred_price"] <= 0 or abs(result["pred_pct"]) < 0.01:
                ema20_v = float(df["ema20"].iloc[-1]) if "ema20" in df.columns else live_price
                ema50_v = float(df["ema50"].iloc[-1]) if "ema50" in df.columns else live_price
                direction_sign = 1.0 if ema20_v > ema50_v else -1.0 if ema20_v < ema50_v else 0.0
                fallback_pred = live_price * (1.0 + direction_sign * max(abs(result["pred_pct"]), 0.5) / 100)
                result["pred_price"] = round(fallback_pred, 6)
                result["pred_pct"] = round(direction_sign * 0.5, 2)
                result["pred_method"] += " [EMA-FALLBACK]"
                logging.info("[AI-PRED] Applied EMA fallback: pred=%.4f", fallback_pred)

            logging.info("[AI-PRED] FINAL: price=%.4f pct=%.2f%% label=%s method=%s", result["pred_price"], result["pred_pct"], result["pred_label"], result["pred_method"])

        except Exception as e:
            logging.error("[AI-PRED] CRITICAL ERROR: %s", str(e)[:200])
            logging.error("[AI-PRED] Traceback: %s", __import__('traceback').format_exc()[:500])
            # Even on total failure, provide minimal directional prediction
            try:
                ema20_v = float(df["ema20"].iloc[-1]) if "ema20" in df.columns else cp
                ema50_v = float(df["ema50"].iloc[-1]) if "ema50" in df.columns else cp
                direction_sign = 1.0 if ema20_v > ema50_v else -1.0
                fallback_pred = cp * (1.0 + direction_sign * 0.3 / 100)
                result["pred_price"] = round(fallback_pred, 6)
                result["pred_pct"] = round(direction_sign * 0.3, 2)
                result["pred_label"] = "BULLISH" if direction_sign > 0 else "BEARISH"
                result["pred_color"] = "#02c076" if direction_sign > 0 else "#cf304a"
                result["pred_method"] = "EMA-EMERGENCY-FALLBACK"
                result["pred_high"] = round(fallback_pred * 1.005, 6)
                result["pred_low"] = round(fallback_pred * 0.995, 6)
                result["pred_live_price"] = round(cp, 6)
                logging.info("[AI-PRED] Emergency fallback applied: %.4f", fallback_pred)
            except Exception as e2:
                logging.error("[AI-PRED] Even emergency fallback failed: %s", str(e2)[:80])
        return result

    def _build_enhanced_bias(self, df, closes, highs, lows, volumes, rsi_arr, adx_arr,
                              macd_line, macd_signal, macd_hist,
                              vwap_vals, stoch_k, stoch_d, stoch_j,
                              direction, live_price, atr_live) -> float:
        """Compute enhanced directional bias from MACD, VWAP, StochRSI, and patterns."""
        bias = 0.0
        i = len(closes) - 1
        if i < 5:
            return 0.0

        # MACD divergence bias
        macd_hist_now = float(macd_hist[i]) if i < len(macd_hist) else 0.0
        macd_hist_prev = float(macd_hist[i-1]) if i-1 >= 0 and i-1 < len(macd_hist) else 0.0
        macd_hist_prev2 = float(macd_hist[i-2]) if i-2 >= 0 and i-2 < len(macd_hist) else 0.0
        macd_rising = macd_hist_now > macd_hist_prev > macd_hist_prev2
        macd_falling = macd_hist_now < macd_hist_prev < macd_hist_prev2

        # Small TF momentum override: if price is making strong continuation, reduce MACD counter-bias
        is_small_tf = self.interval in ("1m", "3m", "5m", "15m")
        momentum_override = False
        if is_small_tf and len(closes) >= 6:
            recent_bodies = [closes[j] - closes[j-1] for j in range(-5, 0)]
            pos_bodies = sum(1 for b in recent_bodies if b > 0)
            neg_bodies = sum(1 for b in recent_bodies if b < 0)
            if direction == "BUY" and pos_bodies >= 4:
                momentum_override = True  # Strong uptrend, don't let MACD falling kill the prediction
            elif direction == "SELL" and neg_bodies >= 4:
                momentum_override = True  # Strong downtrend, don't let MACD rising kill the prediction

        if direction == "BUY" and macd_rising:
            bias += atr_live * 0.25
        elif direction == "SELL" and macd_falling:
            bias -= atr_live * 0.25
        elif direction == "BUY" and macd_falling:
            if not momentum_override:
                bias -= atr_live * 0.15
            else:
                bias -= atr_live * 0.05  # Reduced penalty on small TF strong momentum
        elif direction == "SELL" and macd_rising:
            if not momentum_override:
                bias += atr_live * 0.15
            else:
                bias += atr_live * 0.05  # Reduced penalty on small TF strong momentum

        # MACD zero-cross bias
        macd_line_now = float(macd_line[i]) if i < len(macd_line) else 0.0
        macd_sig_now = float(macd_signal[i]) if i < len(macd_signal) else 0.0
        macd_line_prev = float(macd_line[i-1]) if i-1 >= 0 and i-1 < len(macd_line) else 0.0
        macd_sig_prev = float(macd_signal[i-1]) if i-1 >= 0 and i-1 < len(macd_signal) else 0.0
        if macd_line_prev < macd_sig_prev and macd_line_now > macd_sig_now:
            bias += atr_live * 0.30  # Bullish cross
        elif macd_line_prev > macd_sig_prev and macd_line_now < macd_sig_now:
            bias -= atr_live * 0.30  # Bearish cross

        # VWAP deviation bias
        vwap_now = float(vwap_vals[i]) if i < len(vwap_vals) else live_price
        vwap_dev = (live_price - vwap_now) / live_price if live_price > 0 else 0.0
        if direction == "BUY" and vwap_dev < -0.005:
            bias += atr_live * 0.20  # Price below VWAP, mean reversion up expected
        elif direction == "SELL" and vwap_dev > 0.005:
            bias -= atr_live * 0.20  # Price above VWAP, mean reversion down expected
        elif direction == "BUY" and vwap_dev > 0.01:
            bias -= atr_live * 0.10  # Overextended above VWAP
        elif direction == "SELL" and vwap_dev < -0.01:
            bias += atr_live * 0.10  # Overextended below VWAP

        # StochRSI extreme bias
        sk_now = float(stoch_k[i]) if i < len(stoch_k) else 50.0
        sd_now = float(stoch_d[i]) if i < len(stoch_d) else 50.0
        if sk_now < 20 and sd_now < 20:
            if direction == "BUY":
                bias += atr_live * 0.20
            elif direction == "SELL":
                bias -= atr_live * 0.10  # Counter-trend short into extreme oversold = reduced confidence
        elif sk_now > 80 and sd_now > 80:
            if direction == "SELL":
                bias -= atr_live * 0.20
            elif direction == "BUY":
                bias += atr_live * 0.10  # Counter-trend buy into extreme overbought = reduced confidence

        # StochRSI cross bias
        sk_prev = float(stoch_k[i-1]) if i-1 >= 0 and i-1 < len(stoch_k) else sk_now
        sd_prev = float(stoch_d[i-1]) if i-1 >= 0 and i-1 < len(stoch_d) else sd_now
        if sk_prev < sd_prev and sk_now > sd_now and sk_now < 50:
            bias += atr_live * 0.15  # Bull cross in lower half
        elif sk_prev > sd_prev and sk_now < sd_now and sk_now > 50:
            bias -= atr_live * 0.15  # Bear cross in upper half

        # Volume confirmation bias
        vol_sma20 = float(np.mean(volumes[max(0, i-19):i+1])) if i > 0 else float(volumes[i])
        vol_now = float(volumes[i])
        if vol_now > vol_sma20 * 1.5:
            if direction == "BUY":
                bias += atr_live * 0.10
            elif direction == "SELL":
                bias -= atr_live * 0.10
        elif vol_now < vol_sma20 * 0.5:
            # Low volume = reduce all directional bias
            bias *= 0.7

        # ADX trend strength confirmation
        adx_now = float(adx_arr[i]) if i < len(adx_arr) else 20.0
        if adx_now > 30:
            if direction == "BUY":
                bias += atr_live * 0.10
            elif direction == "SELL":
                bias -= atr_live * 0.10
        elif adx_now < 15:
            # Weak trend = suppress directional bias
            bias *= 0.6

        # Pattern confirmation bias (if patterns detected in current df)
        try:
            cpats = self._detect_candle_patterns(df)
            if direction == "BUY":
                if cpats.get("hammer"): bias += atr_live * 0.15
                if cpats.get("bull_engulf"): bias += atr_live * 0.20
                if cpats.get("shooting_star"): bias -= atr_live * 0.20
                if cpats.get("bear_engulf"): bias -= atr_live * 0.20
            elif direction == "SELL":
                if cpats.get("shooting_star"): bias -= atr_live * 0.15
                if cpats.get("bear_engulf"): bias -= atr_live * 0.20
                if cpats.get("hammer"): bias += atr_live * 0.20
                if cpats.get("bull_engulf"): bias += atr_live * 0.20
        except Exception:
            pass

        # Clamp bias to reasonable range
        max_bias = atr_live * 2.0
        return float(np.clip(bias, -max_bias, max_bias))

    def _get_1h_trend_filter(self) -> str:
        try:
            limit = AI_1H_CANDLES + 5
            if self.interval == "1h":
                df_local = self._deque_to_df()
                if len(df_local) < 20:
                    return "NEUTRAL"
                closes = np.array([float(v) for v in df_local["c"].values[-limit:]], dtype=np.float64)
            else:
                url = (f"https://fapi.binance.com/fapi/v1/klines"
                       f"?symbol={self.symbol}&interval=1h&limit={limit}")
                try:
                    r = self._api_get(url, timeout=5)
                except RuntimeError:
                    return "NEUTRAL"
                if r.status_code != 200:
                    return "NEUTRAL"
                klines = r.json()
                if not klines or len(klines) < 20:
                    return "NEUTRAL"
                closes = np.array([float(k[4]) for k in klines], dtype=np.float64)

            n = len(closes)
            x = np.arange(n, dtype=np.float64).reshape(-1, 1)
            if SKLEARN_OK:
                sw = np.exp(np.linspace(0.0, 5.0, n))
                m = LinearRegression()
                m.fit(x, closes, sample_weight=sw)
                slope = float(m.coef_[0])
            else:
                xf = x.flatten()
                slope = float(np.polyfit(xf, closes, 1)[0])

            slope_pct = (slope / float(closes[-1])) * 100 if closes[-1] > 0 else 0.0
            if slope_pct > 0.02:
                return "BULLISH"
            elif slope_pct < -0.02:
                return "BEARISH"
            else:
                return "NEUTRAL"
        except Exception as e:
            logging.warning("MTF 1h filter error: %s", str(e)[:60])
            return "NEUTRAL"

    def _estimate_time_to_target(self, df: pd.DataFrame, cp: float, target: float, atr: float) -> int:
        if cp == 0 or target == 0 or atr <= 0:
            return 0
        distance = abs(target - cp)
        candles_est = int(distance / (atr * 0.5))
        bounds = {"5m": (1, 20), "15m": (1, 15), "1h": (1, 12), "4h": (1, 8), "1d": (1, 5)}
        low, high = bounds.get(self.interval, (1, 10))
        return max(low, min(candles_est, high))

    def _detect_rsi_divergence(self, df, tf=None):
        if tf is None: tf = getattr(self, 'interval', '1h')
        lookback_map = {"5m": 60, "15m": 70, "1h": 50, "4h": 80, "1d": 150}
        lookback = lookback_map.get(tf, 50)
        if len(df) < lookback + 5:
            return "NONE"
        closes = df["c"].values
        rsi_vals = df["rsi"].values
        for i in range(-lookback, -2):
            if closes[i] < closes[i-1] and closes[i-1] > closes[i-2]:
                for j in range(i-3, i-lookback, -1):
                    if closes[j] < closes[j-1] and closes[j-1] > closes[j-2]:
                        if closes[i] < closes[j] and rsi_vals[i] > rsi_vals[j]:
                            return "BULLISH"
                        break
            if closes[i] > closes[i-1] and closes[i-1] < closes[i-2]:
                for j in range(i-3, i-lookback, -1):
                    if closes[j] > closes[j-1] and closes[j-1] < closes[j-2]:
                        if closes[i] > closes[j] and rsi_vals[i] < rsi_vals[j]:
                            return "BEARISH"
                        break
        return "NONE"

    def _detect_micro_divergence(self, df, tf=None):
        if tf is None: tf = getattr(self, 'interval', '5m')
        if tf not in ("5m", "15m"): return "NONE"
        if len(df) < 15: return "NONE"
        closes = df["c"].values; rsi_vals = df["rsi"].values
        micro_lookback = {"5m": 15, "15m": 12}.get(tf, 12)
        tol = 1.003
        for i in range(-micro_lookback, -2):
            if closes[i] <= closes[i-1] * tol and closes[i-1] >= closes[i-2] / tol:
                if rsi_vals[i] > rsi_vals[i-1] + 1.5 and rsi_vals[i] < 50:
                    return "BULLISH_MICRO"
            if closes[i] >= closes[i-1] / tol and closes[i-1] <= closes[i-2] * tol:
                if rsi_vals[i] < rsi_vals[i-1] - 1.5 and rsi_vals[i] > 50:
                    return "BEARISH_MICRO"
        return "NONE"

    def _detect_market_regime(self, df):
        if len(df) < 50: return "UNKNOWN"
        adx_val = df["adx"].iloc[-1] if "adx" in df.columns else 0
        bb_bw = df["bb_bandwidth"].iloc[-1] if "bb_bandwidth" in df.columns else 0
        if pd.isna(adx_val): return "UNKNOWN"
        if adx_val > 25 and bb_bw > 0.02: return "TRENDING"
        elif adx_val < 20 and bb_bw < 0.015: return "RANGING"
        else: return "MIXED"

    def _winrate_checker_loop(self):
        time.sleep(15)
        while self.running:
            try:
                now = time.time()
                cp = self.mark_price if self.mark_price > 0 else self.price
                if cp > 0:
                    with self._winrate_lock:
                        for snap in self._winrate_predictions:
                            if snap["evaluated"]:
                                continue
                            if now >= snap["maturity"]:
                                start = snap["start_price"]
                                direction = snap["direction"]
                                if direction == "BUY":
                                    snap["result"] = "WIN" if cp > start else "LOSS"
                                else:
                                    snap["result"] = "WIN" if cp < start else "LOSS"
                                snap["final_price"] = round(cp, 6)
                                snap["evaluated"] = True
                                if snap["result"] == "WIN":
                                    self._winrate_wins += 1
                                else:
                                    self._winrate_losses += 1
                        total = self._winrate_wins + self._winrate_losses
                        if total > 0:
                            wr = self._winrate_wins / total * 100
                            self.ui_queue.put(lambda w=wr, t=total: self.status_lbl.config(
                                text=f"AI Win Rate: {w:.1f}% ({t} evaluated)",
                                fg="#02c076" if w >= 55 else "#f0b90b" if w >= 45 else "#cf304a"))
            except Exception as e:
                logging.warning("WinRate checker error: %s", str(e)[:60])
            time.sleep(30)

    def show_winrate_report(self):
        from tkinter import messagebox
        with self._winrate_lock:
            wins   = self._winrate_wins
            losses = self._winrate_losses
            preds  = list(self._winrate_predictions)
        total     = wins + losses
        pending   = sum(1 for p in preds if not p["evaluated"])
        win_rate  = (wins / total * 100) if total > 0 else 0.0
        lines = [
            "╔══════════════════════════════╗",
            "     📊 AI WIN RATE TRACKER    ",
            "╚══════════════════════════════╝",
            "",
            f"  Total Evaluated : {total}",
            f"  ✔ Correct (Wins) : {wins}",
            f"  ❌ Incorrect (Losses): {losses}",
            f"  ⏳ Pending      : {pending}",
            "",
            f"  🎯 Win Rate     : {win_rate:.1f}%",
            "",
        ]
        if total == 0:
            lines.append("  No predictions evaluated yet.")
            lines.append("  Win/loss results appear after")
            lines.append(f"  {_get_lookahead_shift(getattr(self, 'interval', '1h'))} candle maturity period.")
        elif win_rate >= 60:
            lines.append("  ✅ Model performing WELL")
        elif win_rate >= 50:
            lines.append("  ⚠ Model performing AVERAGE")
        else:
            lines.append("  ❌ Model underperforming")
        report = "\n".join(lines)
        messagebox.showinfo("📊 AI WIN RATE", report)
        summary = f"AI Win Rate: {win_rate:.1f}% | {wins}W / {losses}L / {pending} pending"
        self.status_lbl.config(
            text=summary,
            fg="#02c076" if win_rate >= 55 else "#f0b90b" if win_rate >= 45 else "#cf304a"
        )

    def _resize_coin_tv(self):
        try:
            if not hasattr(self, '_coin_tv'):
                return
            self._coin_tv.update_idletasks()
            total = self._coin_tv.winfo_width()
            if total < 50:
                self.root.after(300, self._resize_coin_tv)
                return
            sb_width = 18
            usable = max(total - sb_width, 100)
            self._coin_tv.column("symbol",     width=int(usable * 0.28))
            self._coin_tv.column("direction",  width=int(usable * 0.22))
            self._coin_tv.column("confidence", width=int(usable * 0.28))
            self._coin_tv.column("score",      width=int(usable * 0.22))
        except Exception as e:
            logging.warning("_resize_coin_tv: %s", str(e)[:60])

    def _card(self, parent, title, title_color="#aeb4bc", pady=5):
        outer = tk.Frame(parent, bg="#1a1d24", bd=0)
        outer.pack(fill="x", padx=8, pady=pady)
        tk.Frame(outer, bg=title_color, height=2).pack(fill="x")
        inner = tk.Frame(outer, bg="#1a1d24")
        inner.pack(fill="x", padx=10, pady=(6, 8))
        tk.Label(inner, text=title, fg=title_color, bg="#1a1d24",
                 font=("Arial", 8, "bold")).pack(anchor="w")
        return inner

    def build_ui(self):
        BG = "#0b0e11"
        CARD = "#1a1d24"
        self.root.configure(bg=BG)

        hdr = tk.Frame(self.root, bg=BG)
        hdr.pack(fill="x", padx=8, pady=(8, 0))
        tk.Label(hdr, text="PRO AI RADAR", fg="#f0b90b", bg=BG,
                 font=("Arial", 15, "bold")).pack(side="left")
        self.ws_live_lbl = tk.Label(hdr, text="* LIVE", fg="#02c076", bg=BG,
                                    font=("Arial", 8, "bold"))
        self.ws_live_lbl.pack(side="right", pady=2)

        self.status_lbl = tk.Label(self.root, text="Connecting...", fg="#f0b90b",
                                   bg=BG, font=("Arial", 8))
        self.status_lbl.pack()

        sel = tk.Frame(self.root, bg=BG)
        sel.pack(fill="x", padx=8, pady=(4, 2))

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Dark.TCombobox", fieldbackground=CARD, background=CARD,
                        foreground="#ffffff", selectbackground=CARD,
                        selectforeground="#f0b90b", arrowcolor="#f0b90b",
                        bordercolor="#2b3139", relief="flat")
        style.map("Dark.TCombobox",
                  fieldbackground=[("readonly", CARD)],
                  selectbackground=[("readonly", CARD)],
                  foreground=[("readonly", "#ffffff")])

        self.coin_box = ttk.Combobox(sel, font=("Arial", 10, "bold"), state="readonly",
                                     width=28, style="Dark.TCombobox")
        self.coin_box.set("BTC")
        self.coin_box.pack(side="left", padx=(0, 6))
        self.coin_box.bind("<<ComboboxSelected>>", self.change_coin)

        self.tf_box = ttk.Combobox(sel, values=["5m", "15m", "1h", "4h", "1d"],
                                   font=("Arial", 11), state="readonly", width=6,
                                   style="Dark.TCombobox")
        self.tf_box.set("1h")
        self.tf_box.pack(side="left")
        self.tf_box.bind("<<ComboboxSelected>>", self.change_tf)

        tab_bar = tk.Frame(self.root, bg=BG)
        tab_bar.pack(fill="x", padx=8, pady=(6, 0))
        tab_bar2 = tk.Frame(self.root, bg=BG)
        tab_bar2.pack(fill="x", padx=8, pady=(2, 0))

        self._tab_frames = {}
        self._tab_btns = {}
        self._active_tab = tk.StringVar(value="MARKET")

        scroll_host = tk.Frame(self.root, bg=BG)
        scroll_host.pack(fill="both", expand=True)

        for tab_name in ("MARKET", "ANALYSIS", "INTEL", "LIQ+NEWS", "AI SIGNAL", "MTF", "BACKTEST"):
            sf = ScrollableFrame(scroll_host, bg=BG)
            sf.pack(fill="both", expand=True)
            sf.pack_forget()
            self._tab_frames[tab_name] = sf
        _nar_frame = tk.Frame(scroll_host, bg="#0b0e11")
        _nar_frame.pack_forget()
        self._tab_frames["NARRATIVE"] = type("FakeTab", (), {"inner": _nar_frame, "pack": _nar_frame.pack, "pack_forget": _nar_frame.pack_forget})()
        self._nar_full_frame = _nar_frame

        def _switch_tab(name):
            for n, f in self._tab_frames.items():
                if n == name and n != "NARRATIVE":
                    f.pack(fill="both", expand=True)
                else:
                    f.pack_forget()
            if name == "NARRATIVE":
                self._nar_full_frame.pack(fill="both", expand=True)
            else:
                self._nar_full_frame.pack_forget()
            self._active_tab.set(name)
            for n, b in self._tab_btns.items():
                if n == name: b.config(bg="#f0b90b", fg="#000000")
                else: b.config(bg="#2b3139", fg="#aeb4bc")
            if name == "NARRATIVE":
                self._narrative_cache = {}
                self.root.after(100, self._render_narrative)

        for tab_name in ("MARKET", "ANALYSIS", "INTEL", "LIQ+NEWS", "AI SIGNAL", "MTF"):
            btn = tk.Button(tab_bar, text=tab_name, font=("Arial", 8, "bold"),
                            bg="#2b3139", fg="#aeb4bc", relief="flat", bd=0,
                            padx=6, pady=5, cursor="hand2",
                            command=lambda n=tab_name: _switch_tab(n))
            btn.pack(side="left", padx=(0, 3))
            self._tab_btns[tab_name] = btn

        btn_nar = tk.Button(tab_bar2, text="NARRATIVE",
                            font=("Arial", 8, "bold"),
                            bg="#2b3139", fg="#aeb4bc", relief="flat", bd=0,
                            padx=10, pady=4, cursor="hand2",
                            command=lambda: _switch_tab("NARRATIVE"))
        btn_nar.pack(side="left", fill="x", expand=True, padx=(0, 3))
        self._tab_btns["NARRATIVE"] = btn_nar

        btn_bt = tk.Button(tab_bar2, text="[SCENARIOS] BACKTEST / LIVE TRACKER",
                           font=("Arial", 8, "bold"),
                           bg="#2b3139", fg="#aeb4bc", relief="flat", bd=0,
                           padx=10, pady=4, cursor="hand2",
                           command=lambda: _switch_tab("BACKTEST"))
        btn_bt.pack(side="left", fill="x", expand=True)
        self._tab_btns["BACKTEST"] = btn_bt

        # -- TAB 1: MARKET --
        c1 = self._tab_frames["MARKET"].inner
        pc = tk.Frame(c1, bg="#000000", bd=0)
        pc.pack(fill="x", padx=8, pady=(8, 4))
        tk.Label(pc, text="MARK PRICE (LIVE)", fg="#555", bg="#000000",
                 font=("Arial", 8)).pack(pady=(6, 0))
        self.price_lbl = tk.Label(pc, text="0.000000 $", fg="#ffffff", bg="#000000",
                                  font=("Arial", 22, "bold"))
        self.price_lbl.pack()
        self.change_lbl = tk.Label(pc, text="+0.00%", fg="#02c076", bg="#000000",
                                   font=("Arial", 11))
        self.change_lbl.pack()
        self.candle_count_lbl = tk.Label(pc, text="Candles: --", fg="#555", bg="#000000",
                                          font=("Arial", 8))
        self.candle_count_lbl.pack(pady=(2, 0))

        fund_oi_frame = tk.Frame(pc, bg="#000000")
        fund_oi_frame.pack(fill="x", pady=(4, 0))
        self.funding_lbl = tk.Label(fund_oi_frame, text="Funding: --", fg="#888", bg="#000000",
                                    font=("Arial", 8))
        self.funding_lbl.pack(side="left", expand=True)
        self.oi_lbl = tk.Label(fund_oi_frame, text="OI: --", fg="#888", bg="#000000",
                               font=("Arial", 8))
        self.oi_lbl.pack(side="right", expand=True)

        self.meme_warn_lbl = tk.Label(pc, text="", fg="#ff4d4d", bg="#000000",
                                      font=("Arial", 9, "bold"))
        self.meme_warn_lbl.pack(pady=(2, 0))

        self.trend_lbl = tk.Label(c1, text="WAIT", bg=BG,
                                  font=("Arial", 24, "bold"), fg="#888")
        self.trend_lbl.pack(pady=(4, 0))
        self.time_lbl = tk.Label(c1, text="Updated: --:--:--", fg="#555", bg=BG,
                                 font=("Arial", 8))
        self.time_lbl.pack(pady=(0, 4))

        meter_card = tk.Frame(c1, bg="#0d1117", bd=0)
        meter_card.pack(fill="x", padx=8, pady=(0, 6))
        tk.Frame(meter_card, bg="#f0b90b", height=2).pack(fill="x")
        meter_inner = tk.Frame(meter_card, bg="#0d1117")
        meter_inner.pack(fill="x", padx=8, pady=(4, 6))
        tk.Label(meter_inner, text="MARKET DIRECTION METER", fg="#f0b90b",
                 bg="#0d1117", font=("Arial", 7, "bold")).pack(anchor="w")
        meter_top = tk.Frame(meter_inner, bg="#0d1117")
        meter_top.pack(fill="x", pady=(4, 2))
        self._meter_arrow_lbl = tk.Label(meter_top, text="━", fg="#888",
                                          bg="#0d1117", font=("Arial", 32, "bold"))
        self._meter_arrow_lbl.pack(side="left", padx=(0, 8))
        meter_right = tk.Frame(meter_top, bg="#0d1117")
        meter_right.pack(side="left", fill="x", expand=True)
        self._meter_dir_lbl = tk.Label(meter_right, text="WAIT", fg="#888",
                                        bg="#0d1117", font=("Arial", 13, "bold"))
        self._meter_dir_lbl.pack(anchor="w")
        self._meter_strength_lbl = tk.Label(meter_right, text="Strength: --  |  Score: 0",
                                             fg="#555", bg="#0d1117", font=("Arial", 8))
        self._meter_strength_lbl.pack(anchor="w")
        self._meter_conf_lbl = tk.Label(meter_right, text="Confidence: 0%",
                                         fg="#888", bg="#0d1117", font=("Arial", 9, "bold"))
        self._meter_conf_lbl.pack(anchor="w")
        pct_row = tk.Frame(meter_inner, bg="#0d1117")
        pct_row.pack(fill="x", pady=(2, 2))
        self._meter_buy_pct_lbl = tk.Label(pct_row, text="BUY 50%", fg="#02c076",
                                            bg="#0d1117", font=("Arial", 8, "bold"))
        self._meter_buy_pct_lbl.pack(side="left")
        self._meter_sell_pct_lbl = tk.Label(pct_row, text="SELL 50%", fg="#cf304a",
                                             bg="#0d1117", font=("Arial", 8, "bold"))
        self._meter_sell_pct_lbl.pack(side="right")
        bar_outer = tk.Frame(meter_inner, bg="#2b3139", height=14)
        bar_outer.pack(fill="x", pady=(0, 2))
        bar_outer.pack_propagate(False)
        self._meter_buy_bar = tk.Frame(bar_outer, bg="#02c076", height=14)
        self._meter_buy_bar.place(x=0, y=0, relheight=1.0, relwidth=0.5)
        self._meter_sell_bar = tk.Frame(bar_outer, bg="#cf304a", height=14)
        self._meter_sell_bar.place(relx=0.5, y=0, relheight=1.0, relwidth=0.5)
        self._meter_bar_outer = bar_outer

        pred_card = self._card(c1, "🔮 AI PRICE PREDICTION (Live)", "#9b59b6")
        pred_row1 = tk.Frame(pred_card, bg="#1a1d24")
        pred_row1.pack(fill="x", pady=(4, 0))
        pred_left = tk.Frame(pred_row1, bg="#0d1117", bd=0)
        pred_left.pack(side="left", expand=True, fill="x", padx=(0, 3))
        tk.Label(pred_left, text="PREDICTED", fg="#555", bg="#0d1117",
                 font=("Arial", 7)).pack(pady=(4, 0))
        self._market_pred_price_lbl = tk.Label(pred_left, text="--", fg="#9b59b6",
                                                bg="#0d1117", font=("Consolas", 12, "bold"))
        self._market_pred_price_lbl.pack(pady=(0, 4))
        pred_mid = tk.Frame(pred_row1, bg="#0d1117", bd=0)
        pred_mid.pack(side="left", expand=True, fill="x", padx=(0, 3))
        tk.Label(pred_mid, text="CHANGE", fg="#555", bg="#0d1117",
                 font=("Arial", 7)).pack(pady=(4, 0))
        self._market_pred_pct_lbl = tk.Label(pred_mid, text="--", fg="#888",
                                              bg="#0d1117", font=("Arial", 11, "bold"))
        self._market_pred_pct_lbl.pack(pady=(0, 4))
        pred_right = tk.Frame(pred_row1, bg="#0d1117", bd=0)
        pred_right.pack(side="left", expand=True, fill="x")
        tk.Label(pred_right, text="DIRECTION", fg="#555", bg="#0d1117",
                 font=("Arial", 7)).pack(pady=(4, 0))
        self._market_pred_dir_lbl = tk.Label(pred_right, text="--", fg="#888",
                                              bg="#0d1117", font=("Arial", 10, "bold"))
        self._market_pred_dir_lbl.pack(pady=(0, 4))
        pred_row2 = tk.Frame(pred_card, bg="#1a1d24")
        pred_row2.pack(fill="x", pady=(2, 2))
        self._market_pred_range_lbl = tk.Label(pred_row2, text="Range: --", fg="#666",
                                                bg="#1a1d24", font=("Consolas", 8), anchor="w")
        self._market_pred_range_lbl.pack(side="left")
        self._market_pred_method_lbl = tk.Label(pred_row2, text="", fg="#555",
                                                 bg="#1a1d24", font=("Arial", 7), anchor="e")
        self._market_pred_method_lbl.pack(side="right")
        pred_row3 = tk.Frame(pred_card, bg="#1a1d24")
        pred_row3.pack(fill="x", pady=(0, 2))
        self._market_pred_live_lbl = tk.Label(pred_row3, text="Live: --", fg="#444",
                                               bg="#1a1d24", font=("Consolas", 7))
        self._market_pred_live_lbl.pack(side="left")
        self._market_pred_candles_lbl = tk.Label(pred_row3, text="", fg="#444",
                                                  bg="#1a1d24", font=("Arial", 7), anchor="e")
        self._market_pred_candles_lbl.pack(side="right")

        mp = self._card(c1, "MARKET PULSE", "#00d2ff")
        pulse_row = tk.Frame(mp, bg="#1a1d24")
        pulse_row.pack(fill="x", pady=(4, 0))
        self.rsi_lbl = tk.Label(pulse_row, text="RSI\n--", fg="#ffffff", bg="#1a1d24", font=("Arial", 9, "bold"), justify="center")
        self.rsi_lbl.pack(side="left", expand=True)
        adx_col = tk.Frame(pulse_row, bg="#1a1d24")
        adx_col.pack(side="left", expand=True)
        tk.Label(adx_col, text="ADX", fg="#aeb4bc", bg="#1a1d24",
                 font=("Arial", 8)).pack()
        self.macd_lbl = tk.Label(adx_col, text="--", fg="#ffffff", bg="#1a1d24",
                                 font=("Arial", 10, "bold"))
        self.macd_lbl.pack()
        self.vol_lbl = tk.Label(pulse_row, text="VOL\n--", fg="#ffffff", bg="#1a1d24", font=("Arial", 9, "bold"), justify="center")
        self.vol_lbl.pack(side="left", expand=True)

        fc = self._card(c1, "FORECAST (Structure+Volume+PA > EMA+Trend > RSI)", "#02c076")
        fc_row = tk.Frame(fc, bg="#1a1d24")
        fc_row.pack(fill="x", pady=(4, 0))
        self.fore_dir_lbl = tk.Label(fc_row, text="WAIT", fg="#888888", bg="#1a1d24",
                                     font=("Arial", 14, "bold"))
        self.fore_dir_lbl.pack(side="left")
        self.fore_conf_lbl = tk.Label(fc_row, text="0%", fg="#f0b90b", bg="#1a1d24",
                                      font=("Arial", 13, "bold"),
                                      width=5, anchor="e",
                                      relief="groove", bd=1)
        self.fore_conf_lbl.pack(side="right")

        fc2 = tk.Frame(fc, bg="#1a1d24")
        fc2.pack(fill="x", pady=(6, 0))
        self.fore_entry_lbl = tk.Label(fc2, text="Entry: --", fg="#00d2ff", bg="#1a1d24",
                                       font=("Arial", 9))
        self.fore_entry_lbl.grid(row=0, column=0, sticky="w", pady=1)
        self.fore_tp_lbl = tk.Label(fc2, text="TP: --", fg="#02c076", bg="#1a1d24",
                                    font=("Arial", 9))
        self.fore_tp_lbl.grid(row=0, column=1, sticky="w", padx=10, pady=1)
        self.fore_sl_lbl = tk.Label(fc2, text="SL: --", fg="#ff4d4d", bg="#1a1d24",
                                    font=("Arial", 9))
        self.fore_sl_lbl.grid(row=1, column=0, sticky="w", pady=1)
        self.fore_rr_lbl = tk.Label(fc2, text="R:R: --", fg="#f0b90b", bg="#1a1d24",
                                    font=("Arial", 9))
        self.fore_rr_lbl.grid(row=1, column=1, sticky="w", padx=10, pady=1)
        fc2.columnconfigure(0, weight=1); fc2.columnconfigure(1, weight=1)

        self.timer_lbl = tk.Label(fc, text="[T] --:--", fg="#f0b90b", bg="#1a1d24",
                                  font=("Arial", 9))
        self.timer_lbl.pack(anchor="w", pady=(4, 0))

        reason_tbl_frame = tk.Frame(fc, bg="#1a1d24")
        reason_tbl_frame.pack(fill="x", pady=(4, 0))
        for col in range(3):
            reason_tbl_frame.columnconfigure(col, weight=1, uniform="r")
        self._reason_cells = []
        for col in range(3):
            cell = tk.Frame(reason_tbl_frame, bg="#0d1117", bd=0)
            cell.grid(row=0, column=col, sticky="nsew", padx=1, pady=0)
            lbl = tk.Label(cell, text="--", fg="#888", bg="#0d1117",
                           font=("Arial", 7), wraplength=108,
                           justify="center", pady=4)
            lbl.pack(fill="both", expand=True)
            self._reason_cells.append(lbl)
        self.fore_reason_lbl = self._reason_cells[0]

        self.ws_status_lbl = tk.Label(self.root, text=self.ws_status, fg="#888",
                                      bg=BG, font=("Arial", 7))
        self.ws_status_lbl.pack()
        self.ml_lbl = tk.Label(self.root, text=self.ml_status, fg="#888",
                               bg=BG, font=("Arial", 7))
        self.ml_lbl.pack()

        hist_hdr = self._card(c1, "SIGNAL HISTORY", "#f0b90b")
        stats_row = tk.Frame(hist_hdr, bg="#1a1d24")
        stats_row.pack(fill="x", pady=(4, 0))
        self._hist_buy_lbl = tk.Label(stats_row, text="BUY: 0", fg="#02c076",
                                      bg="#1a1d24", font=("Arial", 9, "bold"))
        self._hist_buy_lbl.pack(side="left", padx=(0, 10))
        self._hist_sell_lbl = tk.Label(stats_row, text="SELL: 0", fg="#cf304a",
                                       bg="#1a1d24", font=("Arial", 9, "bold"))
        self._hist_sell_lbl.pack(side="left")
        tk.Button(hist_hdr, text="Clear", font=("Arial", 8),
                  bg="#2b3139", fg="#aeb4bc", relief="flat",
                  command=self._clear_signal_history,
                  cursor="hand2").pack(side="right", pady=2)

        self._hist_container = tk.Frame(c1, bg="#0b0e11")
        self._hist_container.pack(fill="x", padx=8, pady=(4, 0))
        self._hist_empty_lbl = tk.Label(
            self._hist_container,
            text="No signals yet -- waiting...",
            fg="#555", bg="#1a1d24",
            font=("Arial", 9),
            justify="center",
            pady=6
        )
        self._hist_empty_lbl.pack(fill="x")
        self._hist_rendered_count = -1
        self._hist_pool_rows = []
        for _ in range(SIGNAL_HISTORY_MAX):
            row = tk.Frame(self._hist_container, bg="#0b0e11", bd=0)
            bar = tk.Frame(row, bg="#02c076", width=3)
            bar.pack(side="left", fill="y")
            content = tk.Frame(row, bg="#0b0e11")
            content.pack(side="left", fill="x", expand=True, padx=(6, 6), pady=4)
            time_lbl = tk.Label(content, text="", fg="#555", bg="#0b0e11", font=("Arial", 7), anchor="w")
            time_lbl.pack(fill="x")
            line2 = tk.Frame(content, bg="#0b0e11")
            line2.pack(fill="x")
            dir_lbl = tk.Label(line2, text="", fg="#02c076", bg="#0b0e11", font=("Arial", 10, "bold"))
            dir_lbl.pack(side="left")
            str_lbl = tk.Label(line2, text="", fg="#aeb4bc", bg="#0b0e11", font=("Arial", 8))
            str_lbl.pack(side="left")
            conf_lbl = tk.Label(line2, text="", fg="#f0b90b", bg="#0b0e11", font=("Arial", 9, "bold"))
            conf_lbl.pack(side="right")
            line3 = tk.Frame(content, bg="#0b0e11")
            line3.pack(fill="x")
            price_lbl = tk.Label(line3, text="", fg="#888", bg="#0b0e11", font=("Consolas", 7, "bold"), anchor="w")
            price_lbl.pack(side="left")
            sl_lbl = tk.Label(line3, text="", fg="#cf304a", bg="#0b0e11", font=("Consolas", 7), anchor="w")
            sl_lbl.pack(side="left", padx=(6, 0))
            tp_lbl = tk.Label(line3, text="", fg="#02c076", bg="#0b0e11", font=("Consolas", 7), anchor="w")
            tp_lbl.pack(side="left", padx=(6, 0))
            rr_lbl = tk.Label(line3, text="", fg="#f0b90b", bg="#0b0e11", font=("Consolas", 7, "bold"), anchor="e")
            rr_lbl.pack(side="right")
            row.pack_forget()
            self._hist_pool_rows.append({
                'frame': row, 'bar': bar, 'content': content, 'time': time_lbl,
                'line2': line2, 'dir': dir_lbl, 'strength': str_lbl, 'conf': conf_lbl,
                'line3': line3, 'price': price_lbl, 'sl': sl_lbl, 'tp': tp_lbl, 'rr': rr_lbl
            })

        # -- TAB 2: ANALYSIS --
        c2 = self._tab_frames["ANALYSIS"].inner
        ac = self._card(c2, "TREND (EMA)", "#00d2ff")
        self.jaw_lbl = tk.Label(ac, text="EMA20: --", fg="#2962FF", bg="#1a1d24",
                                font=("Consolas", 10, "bold"))
        self.jaw_lbl.pack(anchor="w")
        self.lips_lbl = tk.Label(ac, text="EMA50: --", fg="#00C853", bg="#1a1d24",
                                 font=("Consolas", 10, "bold"))
        self.lips_lbl.pack(anchor="w")
        self.teeth_lbl = tk.Label(ac, text="EMA200: --", fg="#FF5252", bg="#1a1d24",
                                  font=("Consolas", 10, "bold"))
        self.teeth_lbl.pack(anchor="w")
        self.alli_state_lbl = tk.Label(ac, text="ADX: --", fg="#888", bg="#1a1d24",
                                       font=("Arial", 8))
        self.alli_state_lbl.pack(anchor="w", pady=(2, 0))
        self.regime_lbl = tk.Label(ac, text="Regime: --", fg="#f0b90b", bg="#1a1d24",
                                   font=("Arial", 9, "bold"))
        self.regime_lbl.pack(anchor="w", pady=(4, 0))
        self.bb_status_lbl = tk.Label(ac, text="BB: OFF (Trending)", fg="#555", bg="#1a1d24",
                                      font=("Arial", 8))
        self.bb_status_lbl.pack(anchor="w")

        nic = self._card(c2, "ENHANCED INDICATORS (SAR / SuperTrend / KDJ / Vol)", "#9b59b6")
        self._sar_lbl = tk.Label(nic, text="SAR: --", fg="#e67e22", bg="#1a1d24",
                                 font=("Consolas", 9, "bold"))
        self._sar_lbl.pack(anchor="w")
        self._supertrend_lbl = tk.Label(nic, text="SuperTrend: --", fg="#e67e22", bg="#1a1d24",
                                        font=("Consolas", 9, "bold"))
        self._supertrend_lbl.pack(anchor="w")
        self._kdj_lbl = tk.Label(nic, text="KDJ(K/D/J): --", fg="#e67e22", bg="#1a1d24",
                                 font=("Consolas", 9, "bold"))
        self._kdj_lbl.pack(anchor="w")
        self._vol_delta_lbl = tk.Label(nic, text="Vol Delta: --", fg="#e67e22", bg="#1a1d24",
                                       font=("Consolas", 9, "bold"))
        self._vol_delta_lbl.pack(anchor="w")
        self._bb_squeeze_lbl = tk.Label(nic, text="BB Squeeze: --", fg="#e67e22", bg="#1a1d24",
                                        font=("Consolas", 9, "bold"))
        self._bb_squeeze_lbl.pack(anchor="w")

        ptc = self._card(c2, "PATTERNS  (M/W/Double/Top/Bottom/Wedges/Flags/Pennants/H&S)", "#FFD600")
        pat_row = tk.Frame(ptc, bg="#1a1d24")
        pat_row.pack(fill="x", pady=(4, 0))
        m_box = tk.Frame(pat_row, bg="#2b3139", bd=0)
        m_box.pack(side="left", expand=True, fill="x", padx=(0, 3))
        self.m_pat_lbl = tk.Label(m_box, text="M:\n--", fg="#ffffff", bg="#2b3139", font=("Arial", 10, "bold"), justify="center", pady=6)
        self.m_pat_lbl.pack(expand=True)
        w_box = tk.Frame(pat_row, bg="#2b3139", bd=0)
        w_box.pack(side="left", expand=True, fill="x")
        self.w_pat_lbl = tk.Label(w_box, text="W:\n--", fg="#ffffff", bg="#2b3139", font=("Arial", 10, "bold"), justify="center", pady=6)
        self.w_pat_lbl.pack(expand=True)

        pat_row2 = tk.Frame(ptc, bg="#1a1d24")
        pat_row2.pack(fill="x", pady=(3, 0))
        dt_box = tk.Frame(pat_row2, bg="#2b3139", bd=0)
        dt_box.pack(side="left", expand=True, fill="x", padx=(0, 3))
        self.double_top_lbl = tk.Label(dt_box, text="DBL TOP\n--", fg="#ffffff", bg="#2b3139", font=("Arial", 9, "bold"), justify="center", pady=5)
        self.double_top_lbl.pack(expand=True)
        db_box = tk.Frame(pat_row2, bg="#2b3139", bd=0)
        db_box.pack(side="left", expand=True, fill="x")
        self.double_bot_lbl = tk.Label(db_box, text="DBL BOT\n--", fg="#ffffff", bg="#2b3139", font=("Arial", 9, "bold"), justify="center", pady=5)
        self.double_bot_lbl.pack(expand=True)

        pat_row3 = tk.Frame(ptc, bg="#1a1d24")
        pat_row3.pack(fill="x", pady=(3, 0))
        rw_box = tk.Frame(pat_row3, bg="#2b3139", bd=0)
        rw_box.pack(side="left", expand=True, fill="x", padx=(0, 3))
        self.rising_wedge_lbl = tk.Label(rw_box, text="RISE WEDGE\n--", fg="#ffffff", bg="#2b3139", font=("Arial", 8, "bold"), justify="center", pady=5)
        self.rising_wedge_lbl.pack(expand=True)
        fw_box = tk.Frame(pat_row3, bg="#2b3139", bd=0)
        fw_box.pack(side="left", expand=True, fill="x")
        self.fall_wedge_lbl = tk.Label(fw_box, text="FALL WEDGE\n--", fg="#ffffff", bg="#2b3139", font=("Arial", 8, "bold"), justify="center", pady=5)
        self.fall_wedge_lbl.pack(expand=True)

        pat_row4 = tk.Frame(ptc, bg="#1a1d24")
        pat_row4.pack(fill="x", pady=(3, 0))
        bf_box = tk.Frame(pat_row4, bg="#2b3139", bd=0)
        bf_box.pack(side="left", expand=True, fill="x", padx=(0, 3))
        self.flag_bull_lbl = tk.Label(bf_box, text="BULL FLAG\n--", fg="#ffffff", bg="#2b3139", font=("Arial", 8, "bold"), justify="center", pady=5)
        self.flag_bull_lbl.pack(expand=True)
        be_box = tk.Frame(pat_row4, bg="#2b3139", bd=0)
        be_box.pack(side="left", expand=True, fill="x", padx=(0, 3))
        self.flag_bear_lbl = tk.Label(be_box, text="BEAR FLAG\n--", fg="#ffffff", bg="#2b3139", font=("Arial", 8, "bold"), justify="center", pady=5)
        self.flag_bear_lbl.pack(expand=True)
        pn_box = tk.Frame(pat_row4, bg="#2b3139", bd=0)
        pn_box.pack(side="left", expand=True, fill="x")
        self.pennant_lbl = tk.Label(pn_box, text="PENNANT\n--", fg="#ffffff", bg="#2b3139", font=("Arial", 8, "bold"), justify="center", pady=5)
        self.pennant_lbl.pack(expand=True)

        pat_row5 = tk.Frame(ptc, bg="#1a1d24")
        pat_row5.pack(fill="x", pady=(3, 0))
        rc_box = tk.Frame(pat_row5, bg="#2b3139", bd=0)
        rc_box.pack(side="left", expand=True, fill="x", padx=(0, 3))
        self.rectangle_lbl = tk.Label(rc_box, text="RECTANGLE\n--", fg="#ffffff", bg="#2b3139", font=("Arial", 8, "bold"), justify="center", pady=5)
        self.rectangle_lbl.pack(expand=True)
        hs_box = tk.Frame(pat_row5, bg="#2b3139", bd=0)
        hs_box.pack(side="left", expand=True, fill="x", padx=(0, 3))
        self.hs_lbl = tk.Label(hs_box, text="H&S\n--", fg="#ffffff", bg="#2b3139", font=("Arial", 8, "bold"), justify="center", pady=5)
        self.hs_lbl.pack(expand=True)
        ihs_box = tk.Frame(pat_row5, bg="#2b3139", bd=0)
        ihs_box.pack(side="left", expand=True, fill="x")
        self.ihs_lbl = tk.Label(ihs_box, text="INV H&S\n--", fg="#ffffff", bg="#2b3139", font=("Arial", 8, "bold"), justify="center", pady=5)
        self.ihs_lbl.pack(expand=True)

        self.pat_bias_lbl = tk.Label(ptc, text="Price Action: --", fg="#888", bg="#1a1d24", font=("Arial", 9, "bold"))
        self.pat_bias_lbl.pack(anchor="w", pady=(4, 0))

        rc = self._card(c2, "RISK", "#ff4d4d")
        risk_row = tk.Frame(rc, bg="#1a1d24")
        risk_row.pack(fill="x", pady=(4, 0))
        self.rr_lbl = tk.Label(risk_row, text="R:R: --", fg="#00d2ff", bg="#1a1d24",
                               font=("Consolas", 10, "bold"))
        self.rr_lbl.pack(side="left")
        self.tp_lbl = tk.Label(risk_row, text="TP: --", fg="#02c076", bg="#1a1d24",
                               font=("Consolas", 10, "bold"))
        self.tp_lbl.pack(side="right")
        risk_row2 = tk.Frame(rc, bg="#1a1d24")
        risk_row2.pack(fill="x", pady=(2, 0))
        self.sl_lbl = tk.Label(risk_row2, text="SL: --", fg="#ff4d4d", bg="#1a1d24",
                               font=("Consolas", 10, "bold"))
        self.sl_lbl.pack(side="left")
        self.dist_lbl = tk.Label(risk_row2, text="Dist: --", fg="#f0b90b", bg="#1a1d24",
                                 font=("Consolas", 9))
        self.dist_lbl.pack(side="right")

        lvc = self._card(c2, "LEVELS", "#f0b90b")
        self.swing_lbl = tk.Label(lvc, text="Swing: --", fg="#ffffff", bg="#1a1d24",
                                  font=("Consolas", 9))
        self.swing_lbl.pack(anchor="w")
        self.fib_lbl = tk.Label(lvc, text="Fib: --", fg="#888", bg="#1a1d24",
                                font=("Consolas", 8), wraplength=340, justify="left")
        self.fib_lbl.pack(anchor="w")

        btn_frm = tk.Frame(c2, bg=BG)
        btn_frm.pack(fill="x", padx=8, pady=4)
        self.save_btn = tk.Button(btn_frm, text="Save Signal", command=self.save_signal,
                                  bg="#2b3139", fg="white", font=("Arial", 11),
                                  activebackground="#f0b90b")
        self.save_btn.pack(side="left", padx=6)
        self.alert_btn = tk.Button(btn_frm, text="Test Alert", command=self.test_alert,
                                   bg="#2b3139", fg="white", font=("Arial", 11),
                                   activebackground="#f0b90b")
        self.alert_btn.pack(side="left", padx=6)

        self.err_lbl = tk.Label(self.root, text="", fg="#ff4d4d", bg=BG,
                                font=("Arial", 8), wraplength=360)
        self.err_lbl.pack()

        # -- TAB 3: INTEL --
        c3 = self._tab_frames["INTEL"].inner
        tr_card = self._card(c3, "TREND  (500 Candles)", "#f0b90b")
        tr_row = tk.Frame(tr_card, bg="#1a1d24")
        tr_row.pack(fill="x", pady=(4, 0))
        self._trend500_lbl = tk.Label(tr_row, text="--", fg="#888", bg="#1a1d24",
                                      font=("Arial", 14, "bold"))
        self._trend500_lbl.pack(side="left")
        self._trend500_str_lbl = tk.Label(tr_row, text="", fg="#888", bg="#1a1d24",
                                          font=("Arial", 9))
        self._trend500_str_lbl.pack(side="left", padx=(8, 0))
        tr_detail = tk.Frame(tr_card, bg="#1a1d24")
        tr_detail.pack(fill="x", pady=(6, 0))
        self._ema_trend_lbl = tk.Label(tr_detail, text="EMA Stack: --", fg="#aeb4bc",
                                       bg="#1a1d24", font=("Consolas", 9),
                                       wraplength=340, justify="left")
        self._ema_trend_lbl.pack(anchor="w")
        self._hh_hl_lbl = tk.Label(tr_detail, text="Structure: --", fg="#aeb4bc",
                                   bg="#1a1d24", font=("Consolas", 9),
                                   wraplength=340, justify="left")
        self._hh_hl_lbl.pack(anchor="w")
        self._trend_angle_lbl = tk.Label(tr_detail, text="Slope: --", fg="#aeb4bc",
                                         bg="#1a1d24", font=("Consolas", 9))
        self._trend_angle_lbl.pack(anchor="w")

        sr_card = self._card(c3, "SUPPORT & RESISTANCE  (Real)", "#00d2ff")
        sr_note = tk.Label(sr_card, text="500 candles -- Volume confirmed",
                           fg="#555", bg="#1a1d24", font=("Arial", 7))
        sr_note.pack(anchor="w", pady=(0, 4))
        self._sr_container = tk.Frame(sr_card, bg="#1a1d24")
        self._sr_container.pack(fill="x")
        self._sr_labels = []

        intel_refresh_btn = tk.Button(c3, text="  Refresh Intel",
                                      command=self.refresh_intel,
                                      bg="#f0b90b", fg="#000000",
                                      font=("Arial", 10, "bold"),
                                      relief="flat", bd=0, pady=8, cursor="hand2",
                                      activebackground="#d4a017")
        intel_refresh_btn.pack(fill="x", padx=8, pady=(8, 4))
        self._intel_status_lbl = tk.Label(c3, text="Auto-updates every 5 min",
                                          fg="#555", bg=BG, font=("Arial", 8))
        self._intel_status_lbl.pack(pady=(0, 8))

        # -- TAB 4: LIQ+NEWS --
        c4 = self._tab_frames["LIQ+NEWS"].inner
        liq_card = self._card(c4, "LIQUIDITY  (1H / 4H)", "#ff4d4d")
        liq_row = tk.Frame(liq_card, bg="#1a1d24")
        liq_row.pack(fill="x", pady=(4, 0))
        self._liq_buy_lbl = tk.Label(liq_row, text="Buy Liq: --", fg="#02c076",
                                      bg="#1a1d24", font=("Consolas", 10, "bold"))
        self._liq_buy_lbl.pack(side="left", expand=True)
        self._liq_sell_lbl = tk.Label(liq_row, text="Sell Liq: --", fg="#cf304a",
                                       bg="#1a1d24", font=("Consolas", 10, "bold"))
        self._liq_sell_lbl.pack(side="left", expand=True)
        self._liq_dom_lbl = tk.Label(liq_card, text="Dominance: --", fg="#f0b90b",
                                      bg="#1a1d24", font=("Arial", 9, "bold"))
        self._liq_dom_lbl.pack(anchor="w", pady=(4, 0))
        self._liq_zones_lbl = tk.Label(liq_card, text="Zones: --", fg="#888",
                                        bg="#1a1d24", font=("Consolas", 8), wraplength=340)
        self._liq_zones_lbl.pack(anchor="w")
        self._liq_sweep_lbl = tk.Label(liq_card, text="Sweep: --", fg="#888",
                                        bg="#1a1d24", font=("Arial", 9, "bold"))
        self._liq_sweep_lbl.pack(anchor="w", pady=(4, 0))

        news_card = self._card(c4, "NEWS  (CryptoPanic)", "#00d2ff")
        self._news_container = tk.Frame(news_card, bg="#1a1d24")
        self._news_container.pack(fill="x")
        self._news_empty_lbl = tk.Label(self._news_container,
                                         text="No news loaded yet -- tap Refresh",
                                         fg="#555", bg="#1a1d24", font=("Arial", 9),
                                         justify="center", pady=6)
        self._news_empty_lbl.pack(fill="x")
        self._news_pool = []
        for _ in range(6):
            row = tk.Frame(self._news_container, bg="#0b0e11", bd=0)
            bar = tk.Frame(row, bg="#00d2ff", width=3)
            bar.pack(side="left", fill="y")
            content = tk.Frame(row, bg="#0b0e11")
            content.pack(side="left", fill="x", expand=True, padx=(6, 6), pady=4)
            title_lbl = tk.Label(content, text="", fg="#ffffff", bg="#0b0e11",
                                 font=("Arial", 9, "bold"), wraplength=310, justify="left")
            title_lbl.pack(anchor="w")
            meta_lbl = tk.Label(content, text="", fg="#555", bg="#0b0e11",
                                font=("Arial", 7))
            meta_lbl.pack(anchor="w")
            row.pack_forget()
            self._news_pool.append({'frame': row, 'bar': bar, 'title': title_lbl, 'meta': meta_lbl})
        self._news_sentiment_lbl = tk.Label(news_card, text="Sentiment: --",
                                            fg="#888", bg="#1a1d24", font=("Arial", 9, "bold"))
        self._news_sentiment_lbl.pack(anchor="w", pady=(4, 0))
        news_btn = tk.Button(c4, text="  Refresh News",
                             command=self.refresh_news,
                             bg="#00d2ff", fg="#000000",
                             font=("Arial", 10, "bold"),
                             relief="flat", bd=0, pady=8, cursor="hand2",
                             activebackground="#00a0cc")
        news_btn.pack(fill="x", padx=8, pady=(8, 4))

        # -- TAB 5: AI SIGNAL --
        c5 = self._tab_frames["AI SIGNAL"].inner
        sig_card = self._card(c5, "AI TRADING SIGNAL", "#02c076")
        sig_dir_row = tk.Frame(sig_card, bg="#1a1d24")
        sig_dir_row.pack(fill="x", pady=(4, 0))
        self._ai_dir_lbl = tk.Label(sig_dir_row, text="WAIT", fg="#888",
                                      bg="#1a1d24", font=("Arial", 18, "bold"))
        self._ai_dir_lbl.pack(side="left")
        self._ai_conf_lbl = tk.Label(sig_dir_row, text="0%", fg="#f0b90b",
                                       bg="#1a1d24", font=("Arial", 14, "bold"))
        self._ai_conf_lbl.pack(side="right")
        self._ai_strength_lbl = tk.Label(sig_card, text="Strength: --",
                                           fg="#888", bg="#1a1d24", font=("Arial", 9))
        self._ai_strength_lbl.pack(anchor="w", pady=(2, 0))
        self._ai_entry_lbl = tk.Label(sig_card, text="Entry: --",
                                       fg="#00d2ff", bg="#1a1d24", font=("Consolas", 10, "bold"))
        self._ai_entry_lbl.pack(anchor="w", pady=(4, 0))
        self._ai_sl_lbl = tk.Label(sig_card, text="SL: --",
                                     fg="#ff4d4d", bg="#1a1d24", font=("Consolas", 10, "bold"))
        self._ai_sl_lbl.pack(anchor="w")
        self._ai_tp_lbl = tk.Label(sig_card, text="TP: --",
                                     fg="#02c076", bg="#1a1d24", font=("Consolas", 10, "bold"))
        self._ai_tp_lbl.pack(anchor="w")
        self._ai_rr_lbl = tk.Label(sig_card, text="R:R: --",
                                     fg="#f0b90b", bg="#1a1d24", font=("Consolas", 10, "bold"))
        self._ai_rr_lbl.pack(anchor="w")
        self._ai_pos_size_lbl = tk.Label(sig_card, text="Position: --",
                                          fg="#9b59b6", bg="#1a1d24", font=("Consolas", 9, "bold"))
        self._ai_pos_size_lbl.pack(anchor="w", pady=(2, 0))
        self._ai_win_rate_lbl = tk.Label(sig_card, text="Win Rate Est: --",
                                           fg="#9b59b6", bg="#1a1d24", font=("Consolas", 9, "bold"))
        self._ai_win_rate_lbl.pack(anchor="w")
        self._ai_reason_lbl = tk.Label(sig_card, text="Reason: --",
                                        fg="#888", bg="#1a1d24", font=("Arial", 8),
                                        wraplength=340, justify="left")
        self._ai_reason_lbl.pack(anchor="w", pady=(4, 0))
        self._ai_time_to_target_lbl = tk.Label(sig_card, text="Est. Time: --",
                                                fg="#555", bg="#1a1d24", font=("Arial", 8))
        self._ai_time_to_target_lbl.pack(anchor="w")
        self._ai_session_lbl = tk.Label(sig_card, text="Session: OK",
                                         fg="#02c076", bg="#1a1d24", font=("Arial", 8, "bold"))
        self._ai_session_lbl.pack(anchor="w", pady=(2, 0))
        # v21: Add structure alignment indicator
        self._ai_structure_lbl = tk.Label(sig_card, text="Structure: --",
                                           fg="#888", bg="#1a1d24", font=("Arial", 8, "bold"))
        self._ai_structure_lbl.pack(anchor="w")
        self._ai_conflict_lbl = tk.Label(sig_card, text="Conflicts: None",
                                          fg="#888", bg="#1a1d24", font=("Arial", 8))
        self._ai_conflict_lbl.pack(anchor="w")

        # -- TAB 6: MTF --
        c6 = self._tab_frames["MTF"].inner
        mtf_hdr = self._card(c6, "MULTI-TIMEFRAME CONFLUENCE", "#f0b90b")
        self._mtf_master_lbl = tk.Label(mtf_hdr, text="MASTER: WAIT", fg="#888",
                                        bg="#1a1d24", font=("Arial", 14, "bold"))
        self._mtf_master_lbl.pack(anchor="w", pady=(4, 0))
        self._mtf_agree_lbl = tk.Label(mtf_hdr, text="Agree: 0/5",
                                       fg="#888", bg="#1a1d24", font=("Arial", 9))
        self._mtf_agree_lbl.pack(anchor="w")
        # v21: Add hierarchy display
        self._mtf_hierarchy_lbl = tk.Label(mtf_hdr, text="Hierarchy: 1H=-- | 15M=--",
                                            fg="#888", bg="#1a1d24", font=("Consolas", 8))
        self._mtf_hierarchy_lbl.pack(anchor="w", pady=(2, 0))
        self._mtf_table_container = tk.Frame(mtf_hdr, bg="#1a1d24")
        self._mtf_table_container.pack(fill="x", pady=(6, 0))
        self._mtf_rows = {}
        for tf in ("5m", "15m", "1h", "4h", "1d"):
            row = tk.Frame(self._mtf_table_container, bg="#0d1117")
            row.pack(fill="x", pady=(0, 2))
            bar = tk.Frame(row, bg="#888", width=4)
            bar.pack(side="left", fill="y")
            tf_lbl = tk.Label(row, text=tf, fg="#555", bg="#0d1117",
                              font=("Arial", 9, "bold"), width=5, anchor="w")
            tf_lbl.pack(side="left", padx=(6, 8))
            dir_lbl = tk.Label(row, text="WAIT", fg="#888", bg="#0d1117",
                               font=("Arial", 9, "bold"), width=6, anchor="w")
            dir_lbl.pack(side="left")
            conf_lbl = tk.Label(row, text="0%", fg="#888", bg="#0d1117",
                                font=("Arial", 9), width=6, anchor="e")
            conf_lbl.pack(side="right", padx=(0, 6))
            self._mtf_rows[tf] = {'frame': row, 'bar': bar, 'dir': dir_lbl, 'conf': conf_lbl}
        mtf_btn = tk.Button(c6, text="  Run Full MTF Scan",
                            command=self.run_mtf_scan,
                            bg="#f0b90b", fg="#000000",
                            font=("Arial", 10, "bold"),
                            relief="flat", bd=0, pady=8, cursor="hand2",
                            activebackground="#d4a017")
        mtf_btn.pack(fill="x", padx=8, pady=(8, 4))
        self._mtf_status_lbl = tk.Label(c6, text="Auto-scanning top coins...",
                                          fg="#555", bg=BG, font=("Arial", 8))
        self._mtf_status_lbl.pack(pady=(0, 8))

        coin_tv_card = self._card(c6, "COIN TRACKER  (AI Ranked)", "#00d2ff")
        style.configure("CoinTree.Treeview",
                        background="#0d1117", foreground="#aeb4bc",
                        fieldbackground="#0d1117", rowheight=24,
                        font=("Consolas", 9))
        style.configure("CoinTree.Treeview.Heading",
                        background="#1a1d24", foreground="#f0b90b",
                        font=("Arial", 8, "bold"))
        style.map("CoinTree.Treeview",
                  background=[("selected", "#2b3139")],
                  foreground=[("selected", "#f0b90b")])
        tv_frame = tk.Frame(coin_tv_card, bg="#0d1117")
        tv_frame.pack(fill="both", expand=True, pady=(4, 0))
        tv_sb = ttk.Scrollbar(tv_frame, orient="vertical")
        tv_sb.pack(side="right", fill="y")
        self._coin_tv = ttk.Treeview(
            tv_frame,
            columns=("symbol", "direction", "confidence", "score"),
            show="headings",
            height=14,
            style="CoinTree.Treeview",
            yscrollcommand=tv_sb.set,
        )
        tv_sb.config(command=self._coin_tv.yview)

        self._coin_tv.heading("symbol",     text="Symbol",       anchor="w")
        self._coin_tv.heading("direction",  text="Signal",       anchor="center")
        self._coin_tv.heading("confidence", text="Conf %",       anchor="center")
        self._coin_tv.heading("score",      text="Score",        anchor="center")

        self._coin_tv.column("symbol",     width=90,  minwidth=70,  anchor="w",      stretch=True)
        self._coin_tv.column("direction",  width=70,  minwidth=55,  anchor="center", stretch=True)
        self._coin_tv.column("confidence", width=80,  minwidth=65,  anchor="center", stretch=True)
        self._coin_tv.column("score",      width=65,  minwidth=50,  anchor="center", stretch=True)
        self._coin_tv.pack(side="left", fill="both", expand=True)

        self._coin_tv.tag_configure("buy",  foreground="#02c076")
        self._coin_tv.tag_configure("sell", foreground="#cf304a")
        self._coin_tv.tag_configure("wait", foreground="#888888")

        # -- TAB 7: NARRATIVE --
        c7 = self._nar_full_frame
        self._nar_story_lbl = tk.Text(
            c7,
            fg="#e0e0e0", bg="#0b0e11",
            font=("Courier", 9),
            wrap="none",
            relief="flat", bd=0,
            state="disabled",
            cursor="arrow",
            padx=10,
            pady=8,
            selectbackground="#0b0e11",
            selectforeground="#e0e0e0",
            inactiveselectbackground="#0b0e11",
        )
        self._nar_story_lbl.bindtags((str(self._nar_story_lbl), str(c7), "all"))
        self._nar_story_lbl.bind("<Button-1>", lambda e: "break")
        self._nar_story_lbl.bind("<B1-Motion>", lambda e: "break")
        self._nar_story_lbl.bind("<Double-Button-1>", lambda e: "break")
        self._nar_story_lbl.bind("<Triple-Button-1>", lambda e: "break")
        self._nar_story_lbl.pack(fill="both", expand=True)

        self._nar_headline_lbl  = tk.Label(c7, bg="#0b0e11")
        self._nar_urgency_lbl   = tk.Label(c7, bg="#0b0e11")
        self._nar_zones_lbl     = tk.Label(c7, bg="#0b0e11")
        self._nar_scenarios_lbl = tk.Label(c7, bg="#0b0e11")
        self._nar_risk_lbl      = tk.Label(c7, bg="#0b0e11")
        self._nar_plan_lbl      = tk.Label(c7, bg="#0b0e11")
        self._nar_comp_lbl      = tk.Label(c7, bg="#0b0e11")
        self._nar_comp_box      = ttk.Combobox(c7, values=["None"], state="readonly")
        self._comparison_symbol = None
        self._narrative_cache   = {}
        self._last_narrative_update = 0

        # -- TAB 8: BACKTEST --
        c8 = self._tab_frames["BACKTEST"].inner
        bt_hdr = self._card(c8, "LIVE BACKTEST TRACKER", "#f0b90b")
        bt_stats = tk.Frame(bt_hdr, bg="#1a1d24")
        bt_stats.pack(fill="x", pady=(4, 0))
        self._bt_wins_lbl = tk.Label(bt_stats, text="Wins: 0", fg="#02c076",
                                       bg="#1a1d24", font=("Arial", 10, "bold"))
        self._bt_wins_lbl.pack(side="left", expand=True)
        self._bt_losses_lbl = tk.Label(bt_stats, text="Losses: 0", fg="#cf304a",
                                         bg="#1a1d24", font=("Arial", 10, "bold"))
        self._bt_losses_lbl.pack(side="left", expand=True)
        self._bt_pnl_lbl = tk.Label(bt_stats, text="P&L: 0%", fg="#888",
                                      bg="#1a1d24", font=("Arial", 10, "bold"))
        self._bt_pnl_lbl.pack(side="left", expand=True)
        self._bt_open_lbl = tk.Label(bt_hdr, text="Open: None", fg="#f0b90b",
                                      bg="#1a1d24", font=("Arial", 9, "bold"))
        self._bt_open_lbl.pack(anchor="w", pady=(4, 0))
        self._bt_container = tk.Frame(c8, bg="#0b0e11")
        self._bt_container.pack(fill="x", padx=8, pady=(4, 0))
        self._bt_empty_lbl = tk.Label(self._bt_container,
                                       text="No backtest trades yet -- waiting for signals...",
                                       fg="#555", bg="#1a1d24", font=("Arial", 9),
                                       justify="center", pady=6)
        self._bt_empty_lbl.pack(fill="x")
        self._bt_pool_rows = []
        for _ in range(SIGNAL_HISTORY_MAX):
            row = tk.Frame(self._bt_container, bg="#0b0e11", bd=0)
            bar = tk.Frame(row, bg="#02c076", width=3)
            bar.pack(side="left", fill="y")
            content = tk.Frame(row, bg="#0b0e11")
            content.pack(side="left", fill="x", expand=True, padx=(6, 6), pady=4)
            time_lbl = tk.Label(content, text="", fg="#555", bg="#0b0e11", font=("Arial", 7), anchor="w")
            time_lbl.pack(fill="x")
            line2 = tk.Frame(content, bg="#0b0e11")
            line2.pack(fill="x")
            dir_lbl = tk.Label(line2, text="", fg="#02c076", bg="#0b0e11", font=("Arial", 10, "bold"))
            dir_lbl.pack(side="left")
            res_lbl = tk.Label(line2, text="", fg="#888", bg="#0b0e11", font=("Arial", 9, "bold"))
            res_lbl.pack(side="right")
            line3 = tk.Frame(content, bg="#0b0e11")
            line3.pack(fill="x")
            entry_lbl = tk.Label(line3, text="", fg="#888", bg="#0b0e11", font=("Consolas", 7, "bold"), anchor="w")
            entry_lbl.pack(side="left")
            exit_lbl = tk.Label(line3, text="", fg="#888", bg="#0b0e11", font=("Consolas", 7), anchor="w")
            exit_lbl.pack(side="left", padx=(6, 0))
            rr_lbl = tk.Label(line3, text="", fg="#f0b90b", bg="#0b0e11", font=("Consolas", 7, "bold"), anchor="e")
            rr_lbl.pack(side="right")
            row.pack_forget()
            self._bt_pool_rows.append({
                'frame': row, 'bar': bar, 'time': time_lbl,
                'dir': dir_lbl, 'res': res_lbl,
                'entry': entry_lbl, 'exit': exit_lbl, 'rr': rr_lbl
            })
        bt_btn = tk.Button(c8, text="  Reset Backtest",
                           command=self._reset_backtest,
                           bg="#cf304a", fg="#ffffff",
                           font=("Arial", 10, "bold"),
                           relief="flat", bd=0, pady=8, cursor="hand2",
                           activebackground="#a02030")
        bt_btn.pack(fill="x", padx=8, pady=(8, 2))

        btn_8 = tk.Button(c8, text="📊 AI WIN RATE",
                          command=self.show_winrate_report,
                          bg="#9b59b6", fg="#ffffff",
                          font=("Arial", 10, "bold"),
                          relief="flat", bd=0, pady=8, cursor="hand2",
                          activebackground="#7d3c98")
        btn_8.pack(fill="x", padx=8, pady=(0, 4))
        self.btn_8 = btn_8

    def refresh_ui(self):
        if not self.running:
            return
        try:
            self.update_market_ui()
            self.update_analysis_ui()
            self.update_intel_ui()
            self.update_liq_news_ui()
            self.update_ai_signal_ui()
            self.update_mtf_ui()
            self.update_signal_history()
            self.update_backtest_ui()
            if self._active_tab.get() == "NARRATIVE":
                self._render_narrative()
            if self.error_msg:
                self.err_lbl.config(text=self.error_msg[:80])
            else:
                self.err_lbl.config(text="")
            self.ws_status_lbl.config(text=self.ws_status)
            self.ml_lbl.config(text=self.ml_status)
        except Exception as e:
            logging.warning("UI refresh error: %s", str(e)[:80])
        self.root.after(UI_REFRESH_MS, self.refresh_ui)

    def update_market_ui(self):
        cp = self.mark_price if self.mark_price > 0 else self.price
        if cp <= 0:
            return

        self.display_price = cp
        dp = self.display_price
        self.price_lbl.config(text=f"{dp:,.6f} $")
        if self.candle_open_price > 0:
            change = (dp - self.candle_open_price) / self.candle_open_price * 100
            col = "#02c076" if change >= 0 else "#cf304a"
            self.change_lbl.config(text=f"{change:+.2f}%", fg=col)
        else:
            self.change_lbl.config(text="+0.00%", fg="#888")
        with self.df_lock:
            candle_count = len(self.candle_deque)
        self.candle_count_lbl.config(text=f"Candles: {candle_count}")
        if self._last_funding_update > 0:
            fund_text = f"Funding: {self.funding_rate*100:.4f}%"
            fund_col = "#ff4d4d" if self.funding_rate <= FUNDING_WARNING else "#02c076" if self.funding_rate >= 0.01 else "#f0b90b"
            self.funding_lbl.config(text=fund_text, fg=fund_col)
        if self._last_oi_update > 0:
            oi_text = f"OI: {self.oi_value/1e6:.2f}M ({self.oi_change_pct:+.1f}%)"
            oi_col = "#02c076" if self.oi_change_pct > 0 else "#cf304a" if self.oi_change_pct < 0 else "#888"
            self.oi_lbl.config(text=oi_text, fg=oi_col)
        is_meme = self._is_meme_coin()
        if is_meme:
            self.meme_warn_lbl.config(text=f"⚠ MEME COIN -- Max {MAX_LEVERAGE_MEME}x Leverage", fg="#ff4d4d")
        else:
            self.meme_warn_lbl.config(text="")
        with self.analysis_lock:
            r = self.current_analysis
        if r.direction == "BUY":
            self.trend_lbl.config(text="BUY", fg="#02c076")
        elif r.direction == "SELL":
            self.trend_lbl.config(text="SELL", fg="#cf304a")
        else:
            self.trend_lbl.config(text="WAIT", fg="#888")
        self.time_lbl.config(text=f"Updated: {self.last_update}")
        if r.direction == "BUY":
            self._meter_dir_lbl.config(text="BUY", fg="#02c076")
            self._meter_arrow_lbl.config(text="▲", fg="#02c076")
        elif r.direction == "SELL":
            self._meter_dir_lbl.config(text="SELL", fg="#cf304a")
            self._meter_arrow_lbl.config(text="▼", fg="#cf304a")
        else:
            self._meter_dir_lbl.config(text="WAIT", fg="#888")
            self._meter_arrow_lbl.config(text="━", fg="#888")
        self._meter_strength_lbl.config(text=f"Strength: {r.strength}  |  Score: {r.score}")
        self._meter_conf_lbl.config(text=f"Confidence: {r.confidence}%", fg="#f0b90b" if r.confidence >= 60 else "#888")
        buy_pct = min(max(50 + r.score * 0.5, 5), 95)
        sell_pct = 100 - buy_pct
        self._meter_buy_pct_lbl.config(text=f"BUY {buy_pct:.0f}%")
        self._meter_sell_pct_lbl.config(text=f"SELL {sell_pct:.0f}%")
        self._meter_buy_bar.place_configure(relwidth=buy_pct/100)
        self._meter_sell_bar.place_configure(relx=buy_pct/100, relwidth=sell_pct/100)
        pred = r.price_prediction
        if pred and pred.get("pred_price", 0) > 0:
            self._market_pred_price_lbl.config(text=f"{pred['pred_price']:,.6f}", fg=pred.get("pred_color", "#9b59b6"))
            self._market_pred_pct_lbl.config(text=f"{pred['pred_pct']:+.2f}%", fg=pred.get("pred_color", "#9b59b6"))
            self._market_pred_dir_lbl.config(text=pred.get("pred_label", "--"), fg=pred.get("pred_color", "#9b59b6"))
            range_text = f"High: {pred.get('pred_high', 0):,.6f}  |  Low: {pred.get('pred_low', 0):,.6f}"
            self._market_pred_range_lbl.config(text=range_text)
            self._market_pred_method_lbl.config(text=pred.get("pred_method", ""))
            self._market_pred_live_lbl.config(text=f"Live: {pred.get('pred_live_price', 0):,.6f}")
            self._market_pred_candles_lbl.config(text=f"{pred.get('pred_candles', 0)} candles | R²={pred.get('pred_r2', 0):.2f}")
        else:
            self._market_pred_price_lbl.config(text="--", fg="#9b59b6")
            self._market_pred_pct_lbl.config(text="--", fg="#888")
            self._market_pred_dir_lbl.config(text="--", fg="#888")
            self._market_pred_range_lbl.config(text="Range: --")
            self._market_pred_method_lbl.config(text="")
            self._market_pred_live_lbl.config(text="Live: --")
            self._market_pred_candles_lbl.config(text="")
        rsi_val = r.rsi
        rsi_color = "#02c076" if rsi_val >= 55 else "#cf304a" if rsi_val <= 45 else "#f0b90b"
        self.rsi_lbl.config(text=f"RSI\n{rsi_val:.0f}", fg=rsi_color)
        adx_val = r.adx
        macd_color = "#02c076" if adx_val >= 25 else "#f0b90b" if adx_val >= 18 else "#888"
        self.macd_lbl.config(text=f"{adx_val:.0f}", fg=macd_color)
        vol_color = "#02c076" if r.vol_ratio >= 120 else "#f0b90b" if r.vol_ratio >= 80 else "#888"
        self.vol_lbl.config(text=f"VOL\n{r.vol_ratio:.0f}%", fg=vol_color)
        if r.direction == "BUY":
            self.fore_dir_lbl.config(text="BUY", fg="#02c076")
        elif r.direction == "SELL":
            self.fore_dir_lbl.config(text="SELL", fg="#cf304a")
        else:
            self.fore_dir_lbl.config(text="WAIT", fg="#888")
        self.fore_conf_lbl.config(text=f"{r.confidence}%", fg="#f0b90b")
        self.fore_entry_lbl.config(text=f"Entry: {r.entry_low:.4f} - {r.entry_high:.4f}")
        tp_val = r.smart_tp if r.smart_tp > 0 else r.tp
        sl_val = r.smart_sl if r.smart_sl > 0 else r.sl
        self.fore_tp_lbl.config(text=f"TP: {tp_val:.4f}")
        self.fore_sl_lbl.config(text=f"SL: {sl_val:.4f}")
        self.fore_rr_lbl.config(text=f"R:R: 1:{r.rr:.1f}")
        if self.funding_next > 0:
            now_ts = int(time.time())
            secs = self.funding_next - now_ts
            if secs > 0:
                h, rem = divmod(secs, 3600)
                m, s = divmod(rem, 60)
                self.timer_lbl.config(text=f"[T] {h:02d}:{m:02d}:{s:02d} until funding", fg="#f0b90b")
            else:
                self.timer_lbl.config(text="[T] Funding soon", fg="#ff4d4d")
        else:
            self.timer_lbl.config(text="[T] --:--", fg="#f0b90b")
        reasons = r.reason.split(" | ") if r.reason else []
        for i in range(3):
            if i < len(reasons):
                self._reason_cells[i].config(text=reasons[i], fg="#aeb4bc")
            else:
                self._reason_cells[i].config(text="--", fg="#555")

    def update_analysis_ui(self):
        with self.analysis_lock:
            r = self.current_analysis
        cp = self.mark_price if self.mark_price > 0 else self.price
        if cp <= 0:
            return
        self.jaw_lbl.config(text=f"EMA20: {r.ema20:,.6f}", fg="#2962FF")
        self.lips_lbl.config(text=f"EMA50: {r.ema50:,.6f}", fg="#00C853")
        self.teeth_lbl.config(text=f"EMA200: {r.ema200:,.6f}", fg="#FF5252")
        adx_val = r.adx
        self.alli_state_lbl.config(text=f"ADX: {adx_val:.0f}  |  +DI: {r.di_plus:.0f}  |  -DI: {r.di_minus:.0f}",
                                   fg="#f0b90b" if adx_val >= 25 else "#888")
        self.regime_lbl.config(text=f"Regime: {r.market_regime}")
        if r.market_regime == "RANGING":
            self.bb_status_lbl.config(text="BB: ON (Ranging)", fg="#02c076")
        else:
            self.bb_status_lbl.config(text="BB: OFF (Trending)", fg="#555")
        self._sar_lbl.config(text=f"SAR: {r.sar:,.6f} {'▲' if r.sar < cp else '▼'}",
                             fg="#02c076" if r.sar < cp and cp > r.ema20 else "#cf304a" if r.sar > cp and cp < r.ema20 else "#e67e22")
        self._supertrend_lbl.config(text=f"SuperTrend: {r.supertrend:,.6f} [{r.supertrend_dir}]",
                                    fg="#02c076" if r.supertrend_dir == "BUY" else "#cf304a" if r.supertrend_dir == "SELL" else "#e67e22")
        self._kdj_lbl.config(text=f"KDJ(K/D/J): {r.kdj_k:.1f} / {r.kdj_d:.1f} / {r.kdj_j:.1f}",
                             fg="#02c076" if r.kdj_k > r.kdj_d and r.kdj_j > 50 else "#cf304a" if r.kdj_k < r.kdj_d and r.kdj_j < 50 else "#e67e22")
        self._vol_delta_lbl.config(text=f"Vol Delta: {r.vol_delta:+.2f}",
                                   fg="#02c076" if r.vol_delta > 0 else "#cf304a" if r.vol_delta < 0 else "#e67e22")
        self._bb_squeeze_lbl.config(text=f"BB Squeeze: {'YES' if r.bb_squeeze else 'NO'} (BW: {r.bb_bandwidth:.4f})",
                                    fg="#f0b90b" if r.bb_squeeze else "#e67e22")
        def _pat_color(val): return "#02c076" if val else "#cf304a"
        self.m_pat_lbl.config(text=f"M:\n{'YES' if r.m_pattern else 'NO'}", fg=_pat_color(r.m_pattern))
        self.w_pat_lbl.config(text=f"W:\n{'YES' if r.w_pattern else 'NO'}", fg=_pat_color(r.w_pattern))
        self.double_top_lbl.config(text=f"DBL TOP\n{'YES' if r.double_top else 'NO'}", fg=_pat_color(r.double_top))
        self.double_bot_lbl.config(text=f"DBL BOT\n{'YES' if r.double_bottom else 'NO'}", fg=_pat_color(r.double_bottom))
        self.rising_wedge_lbl.config(text=f"RISE WEDGE\n{'YES' if r.rising_wedge else 'NO'}", fg=_pat_color(r.rising_wedge))
        self.fall_wedge_lbl.config(text=f"FALL WEDGE\n{'YES' if r.falling_wedge else 'NO'}", fg=_pat_color(r.falling_wedge))
        self.flag_bull_lbl.config(text=f"BULL FLAG\n{'YES' if r.flag_bull else 'NO'}", fg=_pat_color(r.flag_bull))
        self.flag_bear_lbl.config(text=f"BEAR FLAG\n{'YES' if r.flag_bear else 'NO'}", fg=_pat_color(r.flag_bear))
        self.pennant_lbl.config(text=f"PENNANT\n{'YES' if r.pennant else 'NO'}", fg=_pat_color(r.pennant))
        self.rectangle_lbl.config(text=f"RECTANGLE\n{'YES' if r.rectangle else 'NO'}", fg=_pat_color(r.rectangle))
        self.hs_lbl.config(text=f"H&S\n{'YES' if r.head_shoulders else 'NO'}", fg=_pat_color(r.head_shoulders))
        self.ihs_lbl.config(text=f"INV H&S\n{'YES' if r.inv_head_shoulders else 'NO'}", fg=_pat_color(r.inv_head_shoulders))
        pa_color = "#02c076" if r.price_action_bias == "UPWARD" else "#cf304a" if r.price_action_bias == "DOWNWARD" else "#888"
        self.pat_bias_lbl.config(text=f"Price Action: {r.price_action_bias} | {r.pattern_label}", fg=pa_color)
        tp_val = r.smart_tp if r.smart_tp > 0 else r.tp
        sl_val = r.smart_sl if r.smart_sl > 0 else r.sl
        self.rr_lbl.config(text=f"R:R: 1:{r.rr:.1f}")
        self.tp_lbl.config(text=f"TP: {tp_val:.4f}")
        self.sl_lbl.config(text=f"SL: {sl_val:.4f}")
        dist = abs(tp_val - cp) if tp_val > 0 else 0
        self.dist_lbl.config(text=f"Dist: {dist:.4f}")
        sl_text = f"Swing Low: {r.swing_low:.4f}  |  Swing High: {r.swing_high:.4f}"
        self.swing_lbl.config(text=sl_text)
        fib_text = " | ".join([f"{k}:{v:.4f}" for k, v in r.fib_levels.items()]) if r.fib_levels else "--"
        self.fib_lbl.config(text=f"Fib: {fib_text}")

    def update_intel_ui(self):
        with self.intel_lock:
            trend = self.intel_trend500
            trend_str = self.intel_trend_str
            ema_stack = self.intel_ema_stack
            struct = self.intel_structure
            slope = self.intel_slope
            sr_levels = list(self.intel_sr_levels)
        if trend == "BUY":
            self._trend500_lbl.config(text="BUY", fg="#02c076")
        elif trend == "SELL":
            self._trend500_lbl.config(text="SELL", fg="#cf304a")
        else:
            self._trend500_lbl.config(text="WAIT", fg="#888")
        self._trend500_str_lbl.config(text=trend_str, fg="#f0b90b")
        self._ema_trend_lbl.config(text=ema_stack)
        self._hh_hl_lbl.config(text=struct)
        self._trend_angle_lbl.config(text=slope)
        for lbl in getattr(self, '_sr_labels', []):
            lbl.destroy()
        self._sr_labels = []
        if sr_levels:
            for lvl in sr_levels[:8]:
                t = lvl.get("type", "?")
                s = lvl.get("strength", "WEAK")
                p = lvl.get("price", 0)
                touches = lvl.get("touches", 0)
                color = "#02c076" if t == "S" and s == "STRONG" else "#cf304a" if t == "R" and s == "STRONG" else "#f0b90b" if s == "MODERATE" else "#888"
                txt = f"{t}: {p:,.6f}  ({s}, {touches} touches)"
                lbl = tk.Label(self._sr_container, text=txt, fg=color, bg="#1a1d24",
                               font=("Consolas", 9), anchor="w")
                lbl.pack(anchor="w")
                self._sr_labels.append(lbl)
        else:
            lbl = tk.Label(self._sr_container, text="No S/R data -- tap Refresh",
                           fg="#555", bg="#1a1d24", font=("Arial", 9))
            lbl.pack(anchor="w")
            self._sr_labels.append(lbl)

    def update_liq_news_ui(self):
        with self.intel_lock:
            buy_liq = self.intel_liq_buy
            sell_liq = self.intel_liq_sell
            dom = self.intel_liq_dom
            zones = self.intel_liq_zones
            sweep = self.intel_liq_sweep
            news = list(self.intel_news)
            news_sent = self.intel_news_sentiment
        self._liq_buy_lbl.config(text=f"Buy Liq: {buy_liq:,.0f}")
        self._liq_sell_lbl.config(text=f"Sell Liq: {sell_liq:,.0f}")
        self._liq_dom_lbl.config(text=f"Dominance: {dom}")
        self._liq_zones_lbl.config(text=f"Zones: {zones}")
        self._liq_sweep_lbl.config(text=f"Sweep: {sweep}")
        if news:
            self._news_empty_lbl.pack_forget()
            for i, row_info in enumerate(self._news_pool):
                if i < len(news):
                    n = news[i]
                    row_info['frame'].pack(fill="x", pady=(0, 2))
                    row_info['title'].config(text=n.get("title", "")[:60])
                    row_info['meta'].config(text=f"{n.get('source', '')} | {n.get('time', '')}")
                    sent = n.get("sentiment", "neutral")
                    bar_color = "#02c076" if sent == "positive" else "#cf304a" if sent == "negative" else "#00d2ff"
                    row_info['bar'].config(bg=bar_color)
                else:
                    row_info['frame'].pack_forget()
        else:
            for row_info in self._news_pool:
                row_info['frame'].pack_forget()
            self._news_empty_lbl.pack(fill="x")
        self._news_sentiment_lbl.config(text=f"Sentiment: {news_sent}")

    def update_ai_signal_ui(self):
        with self.analysis_lock:
            r = self.current_analysis
        cp = self.mark_price if self.mark_price > 0 else self.price
        if cp <= 0:
            return
        if r.direction == "BUY":
            self._ai_dir_lbl.config(text="BUY", fg="#02c076")
        elif r.direction == "SELL":
            self._ai_dir_lbl.config(text="SELL", fg="#cf304a")
        else:
            self._ai_dir_lbl.config(text="WAIT", fg="#888")
        self._ai_conf_lbl.config(text=f"{r.confidence}%", fg="#f0b90b")
        self._ai_strength_lbl.config(text=f"Strength: {r.strength}")
        self._ai_entry_lbl.config(text=f"Entry: {r.smart_entry_low:.4f} - {r.smart_entry_high:.4f} [{r.entry_method}]")
        sl_val = r.smart_sl if r.smart_sl > 0 else r.sl
        tp_val = r.smart_tp if r.smart_tp > 0 else r.tp
        self._ai_sl_lbl.config(text=f"SL: {sl_val:.4f}")
        self._ai_tp_lbl.config(text=f"TP: {tp_val:.4f}")
        self._ai_rr_lbl.config(text=f"R:R: 1:{r.rr:.1f}")
        self._ai_pos_size_lbl.config(text=f"Position: {r.position_size_pct:.2f}% of balance (Max {self._get_max_leverage()}x)")
        self._ai_win_rate_lbl.config(text=f"Win Rate Est: {r.win_rate_est:.0f}%")
        self._ai_reason_lbl.config(text=f"Reason: {r.reason}")
        if r.time_to_target > 0:
            self._ai_time_to_target_lbl.config(text=f"Est. Time to Target: ~{r.time_to_target} candles", fg="#f0b90b")
        else:
            self._ai_time_to_target_lbl.config(text="Est. Time: --", fg="#555")
        if r.session_valid:
            self._ai_session_lbl.config(text="Session: OK", fg="#02c076")
        else:
            self._ai_session_lbl.config(text=f"Session: BLOCKED -- {r.session_msg}", fg="#ff4d4d")
        # v21: Structure alignment display
        struct_text = f"Structure: {r.swing_structure}"
        if r.structure_aligned:
            struct_text += " ✓ ALIGNED"
            struct_color = "#02c076"
        else:
            struct_text += " ✗ NOT ALIGNED"
            struct_color = "#cf304a"
        self._ai_structure_lbl.config(text=struct_text, fg=struct_color)
        # v21: Conflict display
        if r.indicator_conflict:
            self._ai_conflict_lbl.config(text=f"Conflicts: INDICATOR CONFLICT DETECTED", fg="#ff4d4d")
        else:
            self._ai_conflict_lbl.config(text="Conflicts: None", fg="#888")

    def update_mtf_ui(self):
        master = self.mtf_master
        agree = self.mtf_agree
        if master == "BUY":
            self._mtf_master_lbl.config(text="MASTER: BUY", fg="#02c076")
        elif master == "SELL":
            self._mtf_master_lbl.config(text="MASTER: SELL", fg="#cf304a")
        else:
            self._mtf_master_lbl.config(text="MASTER: WAIT", fg="#888")
        self._mtf_agree_lbl.config(text=f"Agree: {agree}/5")
        # v21: Hierarchy display
        hierarchy_text = f"Hierarchy: 1H={self._higher_tf_trend} | 15M={self._mid_tf_confirm}"
        self._mtf_hierarchy_lbl.config(text=hierarchy_text, fg="#f0b90b")
        with self._mtf_data_lock:
            cached = dict(self._mtf_cached_results)
        for tf in ("5m", "15m", "1h", "4h", "1d"):
            row = self._mtf_rows[tf]
            res = cached.get(tf, ("WAIT", 0.0))
            direction, conf = res
            if direction == "BUY":
                row['dir'].config(text="BUY", fg="#02c076")
                row['bar'].config(bg="#02c076")
            elif direction == "SELL":
                row['dir'].config(text="SELL", fg="#cf304a")
                row['bar'].config(bg="#cf304a")
            else:
                row['dir'].config(text="WAIT", fg="#888")
                row['bar'].config(bg="#888")
            row['conf'].config(text=f"{conf:.0f}%")

    def update_signal_history(self):
        sigs = list(self.signal_history)
        if not sigs:
            self._hist_empty_lbl.pack(fill="x")
            for row_info in self._hist_pool_rows:
                row_info['frame'].pack_forget()
            return
        self._hist_empty_lbl.pack_forget()
        for i, row_info in enumerate(self._hist_pool_rows):
            if i < len(sigs):
                s = sigs[i]
                row_info['frame'].pack(fill="x", pady=(0, 2))
                color = "#02c076" if s["direction"] == "BUY" else "#cf304a"
                row_info['bar'].config(bg=color)
                row_info['time'].config(text=f"{s['time']}  |  {s['symbol']}  |  {s['tf']}")
                row_info['dir'].config(text=s["direction"], fg=color)
                row_info['strength'].config(text=f"  {s['strength']}")
                row_info['conf'].config(text=f"{s['confidence']}%")
                row_info['price'].config(text=f"@{s['price']:.4f}")
                row_info['sl'].config(text=f"SL:{s['sl']:.4f}")
                row_info['tp'].config(text=f"TP:{s['tp']:.4f}")
                row_info['rr'].config(text=f"R:R 1:{s['rr']}")
            else:
                row_info['frame'].pack_forget()
        self._hist_buy_lbl.config(text=f"BUY: {self.signal_stats['buy']}")
        self._hist_sell_lbl.config(text=f"SELL: {self.signal_stats['sell']}")

    def _clear_signal_history(self):
        self.signal_history.clear()
        self.signal_stats = {"buy": 0, "sell": 0, "wait": 0}
        self.last_signal_direction = "WAIT"
        with self._bt_lock:
            self._bt_signals = []
            self._bt_wins = 0
            self._bt_losses = 0
            self._bt_open = None

    def update_backtest_ui(self):
        with self._bt_lock:
            wins = self._bt_wins
            losses = self._bt_losses
            open_trade = self._bt_open
            signals = list(self._bt_signals)
        total = wins + losses
        pnl = 0.0
        if total > 0:
            win_rate = wins / total * 100
            pnl = (wins * 2.0) - (losses * 1.0)
        else:
            win_rate = 0.0
        self._bt_wins_lbl.config(text=f"Wins: {wins}")
        self._bt_losses_lbl.config(text=f"Losses: {losses}")
        pnl_color = "#02c076" if pnl > 0 else "#cf304a" if pnl < 0 else "#888"
        self._bt_pnl_lbl.config(text=f"P&L: {pnl:+.1f}R", fg=pnl_color)
        if open_trade:
            self._bt_open_lbl.config(text=f"Open: {open_trade['dir']} @ {open_trade['entry']:.4f} | SL:{open_trade['sl']:.4f} | TP:{open_trade['tp']:.4f}")
        else:
            self._bt_open_lbl.config(text="Open: None")
        if not signals:
            self._bt_empty_lbl.pack(fill="x")
            for row_info in self._bt_pool_rows:
                row_info['frame'].pack_forget()
            return
        self._bt_empty_lbl.pack_forget()
        for i, row_info in enumerate(self._bt_pool_rows):
            if i < len(signals):
                s = signals[-(i+1)]
                row_info['frame'].pack(fill="x", pady=(0, 2))
                color = "#02c076" if s["result"] == "WIN" else "#cf304a"
                row_info['bar'].config(bg=color)
                row_info['time'].config(text=f"{s['time']}  |  {s.get('symbol', self.symbol)}")
                row_info['dir'].config(text=s["dir"], fg=color)
                row_info['res'].config(text=s["result"], fg=color)
                row_info['entry'].config(text=f"Entry:{s['entry']:.4f}")
                row_info['exit'].config(text=f"Exit:{s['exit']:.4f}")
                rr = s.get("rr", "--")
                row_info['rr'].config(text=f"R:R 1:{rr}")
            else:
                row_info['frame'].pack_forget()

    def _reset_backtest(self):
        with self._bt_lock:
            self._bt_signals = []
            self._bt_wins = 0
            self._bt_losses = 0
            self._bt_open = None

    def _render_narrative(self):
        try:
            if self._active_tab.get() != "NARRATIVE":
                return
            now = time.time()
            if now - getattr(self, '_last_narrative_update', 0) < 2:
                return
            self._last_narrative_update = now
            with self.analysis_lock:
                r = self.current_analysis
            cp = self.mark_price if self.mark_price > 0 else self.price
            report = self._generate_ai_signal_report(cp, r)
            txt = report if report else "-- Waiting... --"
            self._nar_story_lbl.config(state="normal")
            self._nar_story_lbl.delete("1.0", "end")
            self._nar_story_lbl.insert("end", txt)
            self._nar_story_lbl.config(state="disabled")
            self._nar_story_lbl.yview_moveto(0.0)
        except Exception as e:
            logging.warning("Narrative render error: %s", str(e)[:80])

    def _on_narrative_comparison_change(self, event=None):
        val = self._nar_comp_box.get()
        if val == "None":
            self._comparison_symbol = None
            self._nar_comp_lbl.config(text="")
        else:
            self._comparison_symbol = val + "USDT"
            self._render_narrative_comparison(self._comparison_symbol)

    def _refresh_narrative_comparison(self):
        val = self._nar_comp_box.get()
        if val != "None":
            self._comparison_symbol = val + "USDT"
            self._render_narrative_comparison(self._comparison_symbol)

    def _render_narrative_comparison(self, comp_sym):
        try:
            if not comp_sym:
                return
            base = comp_sym.replace("USDT", "")
            url = f"https://fapi.binance.com/fapi/v1/klines?symbol={comp_sym}&interval={self.interval}&limit=100"
            try:
                r = self._api_get(url, timeout=5)
            except RuntimeError:
                self._nar_comp_lbl.config(text=f"Rate limit — skip {base}")
                return
            if r.status_code != 200:
                self._nar_comp_lbl.config(text=f"Failed to load {base}")
                return
            klines = r.json()
            if not klines or len(klines) < 50:
                self._nar_comp_lbl.config(text=f"Insufficient data for {base}")
                return
            df = self._klines_to_df(klines)
            cp = float(df["c"].iloc[-1])
            result = self._deep_analyze(df, cp)
            with self.analysis_lock:
                current = self.current_analysis
            diff = cp - (self.mark_price if self.mark_price > 0 else self.price)
            diff_pct = (diff / (self.mark_price if self.mark_price > 0 else 1)) * 100
            comp_text = f"\n{base}: {result.direction} ({result.strength}) @ {cp:.4f} | Conf: {result.confidence}%\n"
            comp_text += f"Diff: {diff:+.4f} ({diff_pct:+.2f}%) vs {self.symbol.replace('USDT','')}\n"
            if result.direction == current.direction:
                comp_text += f"✓ Both align {result.direction}"
            else:
                comp_text += f"⚠ Divergence: {self.symbol.replace('USDT','')}={current.direction}, {base}={result.direction}"
            self._nar_comp_lbl.config(text=comp_text)
        except Exception as e:
            self._nar_comp_lbl.config(text=f"Comparison error: {str(e)[:40]}")

    def refresh_intel(self):
        threading.Thread(target=self._intel_thread, daemon=True).start()

    def _intel_thread(self):
        try:
            sym = self.symbol
            if not sym:
                return
            self.ui_queue.put(lambda: self._intel_status_lbl.config(
                text="Fetching 500-candle intelligence...", fg="#f0b90b"))
            url = f"https://fapi.binance.com/fapi/v1/klines?symbol={sym}&interval={self.interval}&limit=500"
            try:
                r = self._api_get(url, timeout=15)
            except RuntimeError:
                self.ui_queue.put(lambda: self._intel_status_lbl.config(
                    text="Intel: Rate limited", fg="#cf304a"))
                return
            klines = r.json()
            if not klines or len(klines) < 100:
                self.ui_queue.put(lambda: self._intel_status_lbl.config(
                    text="Intel: Insufficient data", fg="#cf304a"))
                return
            df = self._klines_to_df(klines)
            cp = float(df["c"].iloc[-1])
            df_ind = self.compute_indicators(df)
            last = df_ind.iloc[-1]
            ema20 = float(last["ema20"]) if pd.notna(last.get("ema20")) else cp
            ema50 = float(last["ema50"]) if pd.notna(last.get("ema50")) else cp
            ema200 = float(last["ema200"]) if pd.notna(last.get("ema200")) else cp
            rsi = float(last["rsi"]) if pd.notna(last.get("rsi")) else 50.0
            adx_val = float(last["adx"]) if pd.notna(last.get("adx")) else 0.0
            di_plus = float(last["di_plus"]) if pd.notna(last.get("di_plus")) else 0.0
            di_minus = float(last["di_minus"]) if pd.notna(last.get("di_minus")) else 0.0
            atr = float(last["atr"]) if pd.notna(last.get("atr")) else cp * 0.002
            vol_ratio = float(last.get("vol_ratio", 100.0)) if pd.notna(last.get("vol_ratio")) else 100.0
            trend = "BUY" if ema20 > ema50 and cp > ema200 else "SELL" if ema20 < ema50 and cp < ema200 else "WAIT"
            trend_str = f"RSI {rsi:.0f} | ADX {adx_val:.0f} | VOL {vol_ratio:.0f}%"
            ema_stack = f"EMA20:{ema20:.4f} > EMA50:{ema50:.4f} > EMA200:{ema200:.4f}" if ema20 > ema50 > ema200 else                         f"EMA20:{ema20:.4f} < EMA50:{ema50:.4f} < EMA200:{ema200:.4f}" if ema20 < ema50 < ema200 else                         f"EMA20:{ema20:.4f} / EMA50:{ema50:.4f} / EMA200:{ema200:.4f}"
            swing_struct = self._detect_swing_structure(df_ind)
            struct_text = swing_struct["detail"]
            x = np.arange(len(df))
            y = df["c"].values
            slope = np.polyfit(x, y, 1)[0]
            slope_text = f"Slope: {slope:.6f} ({'Rising' if slope > 0 else 'Falling'})"
            sr_levels = self._compute_sr_levels(df_ind, cp, lookback=500)
            liq_1h = self._compute_liquidity_levels(df_ind, cp, "1h")
            liq_4h = self._compute_liquidity_levels(df_ind, cp, "4h")
            buy_liq = sum(l["amount"] for l in liq_1h.get("clusters", []) if l["side"] == "buy")
            sell_liq = sum(l["amount"] for l in liq_1h.get("clusters", []) if l["side"] == "sell")
            dom = "BUY DOMINANT" if buy_liq > sell_liq * 1.5 else "SELL DOMINANT" if sell_liq > buy_liq * 1.5 else "BALANCED"
            zones = " | ".join([f"{l['side'].upper()} {l['price']:.4f} ({l['amount']/1000:.0f}K)" for l in liq_1h.get("clusters", [])[:3]])
            sweep = "SWEEP DETECTED" if liq_1h.get("sweep", False) else "No sweep"
            with self.intel_lock:
                self.intel_trend500 = trend
                self.intel_trend_str = trend_str
                self.intel_ema_stack = ema_stack
                self.intel_structure = struct_text
                self.intel_slope = slope_text
                self.intel_sr_levels = sr_levels
                self.intel_liq_buy = buy_liq
                self.intel_liq_sell = sell_liq
                self.intel_liq_dom = dom
                self.intel_liq_zones = zones
                self.intel_liq_sweep = sweep
                self.intel_liq_1h = liq_1h
                self.intel_liq_4h = liq_4h
                self.intel_last_refresh = time.time()
            self.ui_queue.put(lambda: self._intel_status_lbl.config(
                text=f"Intel refreshed ({len(klines)} candles)", fg="#02c076"))
        except Exception as e:
            logging.warning("Intel thread error: %s", str(e)[:80])
            self.ui_queue.put(lambda: self._intel_status_lbl.config(
                text=f"Intel error: {str(e)[:40]}", fg="#cf304a"))

    def _intel_thread_fast(self):
        try:
            if time.time() - getattr(self, 'intel_last_refresh', 0) < 300:
                return
            self._intel_thread()
        except Exception:
            pass

    def _compute_sr_levels(self, df, cp, lookback=500, cluster_pct=0.008):
        if len(df) < 50:
            return []
        recent = df.tail(min(lookback, len(df)))
        highs = recent["h"].values
        lows = recent["l"].values
        volumes = recent["v"].values
        levels = []
        for i in range(2, len(highs) - 2):
            if highs[i] >= highs[i-1] and highs[i] >= highs[i-2] and highs[i] >= highs[i+1] and highs[i] >= highs[i+2]:
                levels.append(("R", highs[i], volumes[i]))
            if lows[i] <= lows[i-1] and lows[i] <= lows[i-2] and lows[i] <= lows[i+1] and lows[i] <= lows[i+2]:
                levels.append(("S", lows[i], volumes[i]))
        clusters = []
        for t, p, v in levels:
            found = False
            for c in clusters:
                if abs(p - c["price"]) / max(c["price"], 1e-9) < cluster_pct:
                    c["touches"] += 1
                    c["volume"] += v
                    c["types"].add(t)
                    found = True
                    break
            if not found:
                clusters.append({"price": p, "touches": 1, "volume": v, "types": {t}})
        for c in clusters:
            c["strength"] = "STRONG" if c["touches"] >= 3 and c["volume"] > np.percentile(volumes, 75) else                             "MODERATE" if c["touches"] >= 2 else "WEAK"
            c["type"] = "S" if "S" in c["types"] and "R" not in c["types"] else                         "R" if "R" in c["types"] and "S" not in c["types"] else "BOTH"
        clusters.sort(key=lambda x: (x["strength"] == "STRONG", x["touches"], x["volume"]), reverse=True)
        return clusters[:15]

    def _compute_liquidity_levels(self, df, cp, tf="1h"):
        if len(df) < 50:
            return {"clusters": [], "sweep": False}
        recent = df.tail(100)
        highs = recent["h"].values
        lows = recent["l"].values
        volumes = recent["v"].values
        clusters = []
        for i in range(2, len(highs) - 2):
            if highs[i] >= highs[i-1] and highs[i] >= highs[i+1]:
                clusters.append({"price": highs[i], "amount": volumes[i] * highs[i], "side": "sell"})
            if lows[i] <= lows[i-1] and lows[i] <= lows[i+1]:
                clusters.append({"price": lows[i], "amount": volumes[i] * lows[i], "side": "buy"})
        sweep = False
        if len(highs) > 10:
            recent_high = max(highs[-10:-1])
            recent_low = min(lows[-10:-1])
            if cp > recent_high * 1.005 and highs[-1] < recent_high:
                sweep = True
            elif cp < recent_low * 0.995 and lows[-1] > recent_low:
                sweep = True
        return {"clusters": sorted(clusters, key=lambda x: x["amount"], reverse=True)[:5], "sweep": sweep}

    def refresh_news(self):
        threading.Thread(target=self._news_thread, daemon=True).start()

    def _news_thread(self):
        try:
            self.ui_queue.put(lambda: self._news_sentiment_lbl.config(
                text="Fetching news...", fg="#f0b90b"))
            url = "https://cryptopanic.com/api/v1/posts/?auth_token=demo&public=true&filter=important"
            r = requests.get(url, timeout=8)
            if r.status_code != 200:
                self.ui_queue.put(lambda: self._news_sentiment_lbl.config(
                    text="News API unavailable", fg="#cf304a"))
                return
            data = r.json()
            results = data.get("results", [])
            news_items = []
            for item in results[:6]:
                title = item.get("title", "")
                source = item.get("source", {}).get("title", "Unknown")
                published = item.get("published_at", "")[:16]
                votes = item.get("votes", {})
                positive = votes.get("positive", 0)
                negative = votes.get("negative", 0)
                sentiment = "positive" if positive > negative else "negative" if negative > positive else "neutral"
                news_items.append({"title": title, "source": source, "time": published, "sentiment": sentiment})
            sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
            for n in news_items:
                sentiment_counts[n["sentiment"]] += 1
            overall = "Bullish" if sentiment_counts["positive"] > sentiment_counts["negative"] else                       "Bearish" if sentiment_counts["negative"] > sentiment_counts["positive"] else "Neutral"
            with self.intel_lock:
                self.intel_news = news_items
                self.intel_news_sentiment = overall
            self.ui_queue.put(lambda: self._news_sentiment_lbl.config(
                text=f"Sentiment: {overall}", fg="#02c076" if overall == "Bullish" else "#cf304a" if overall == "Bearish" else "#888"))
        except Exception as e:
            logging.warning("News thread error: %s", str(e)[:80])
            self.ui_queue.put(lambda: self._news_sentiment_lbl.config(
                text=f"News error: {str(e)[:40]}", fg="#cf304a"))

    def _mtf_auto_loop(self):
        time.sleep(10)
        while self.running:
            try:
                if time.time() - self._mtf_last_fetch_time > self._mtf_fetch_interval and not self._mtf_is_fetching:
                    self.run_mtf_scan()
            except Exception:
                pass
            time.sleep(30)

    def run_mtf_scan(self):
        if self._mtf_is_fetching:
            return
        self._mtf_is_fetching = True
        threading.Thread(target=self._mtf_scan_thread, daemon=True).start()

    def _mtf_scan_thread(self):
        try:
            self.ui_queue.put(lambda: self._mtf_status_lbl.config(
                text="Scanning 5 timeframes...", fg="#f0b90b"))
            sym = self.symbol
            tfs = ["5m", "15m", "1h", "4h", "1d"]
            results = {}
            for tf in tfs:
                try:
                    limit = TF_HISTORY_MAP.get(tf, 150)
                    url = f"https://fapi.binance.com/fapi/v1/klines?symbol={sym}&interval={tf}&limit={limit}"
                    try:
                        r = self._api_get(url, timeout=8)
                    except RuntimeError:
                        results[tf] = ("WAIT", 0.0)
                        continue
                    if r.status_code != 200:
                        results[tf] = ("WAIT", 0.0)
                        continue
                    klines = r.json()
                    if not klines or len(klines) < 50:
                        results[tf] = ("WAIT", 0.0)
                        continue
                    df = self._klines_to_df(klines)
                    cp = float(df["c"].iloc[-1])
                    result = self._deep_analyze(df, cp)
                    results[tf] = (result.direction, result.confidence)
                except Exception:
                    results[tf] = ("WAIT", 0.0)
                time.sleep(0.2)
            buy_count = sum(1 for v in results.values() if v[0] == "BUY")
            sell_count = sum(1 for v in results.values() if v[0] == "SELL")
            if buy_count >= sell_count and buy_count > 0:
                master_dir = "BUY"; agree = buy_count
                max_conf = max((v[1] for v in results.values() if v[0] == "BUY"), default=0)
            elif sell_count > buy_count and sell_count > 0:
                master_dir = "SELL"; agree = sell_count
                max_conf = max((v[1] for v in results.values() if v[0] == "SELL"), default=0)
            else:
                master_dir = "WAIT"; agree = 0; max_conf = 0.0
            with self._mtf_data_lock:
                self._mtf_cached_results = results
            self.mtf_master = master_dir
            self.mtf_agree = agree
            self._mtf_last_fetch_time = time.time()
            self.ui_queue.put(lambda: self._mtf_status_lbl.config(
                text=f"MTF: {master_dir} ({agree}/5 TFs agree)",
                fg="#02c076" if master_dir == "BUY" else "#cf304a" if master_dir == "SELL" else "#888"))
        except Exception as e:
            logging.warning("MTF scan error: %s", str(e)[:80])
            self.ui_queue.put(lambda: self._mtf_status_lbl.config(
                text=f"MTF error: {str(e)[:40]}", fg="#cf304a"))
        finally:
            self._mtf_is_fetching = False

    def change_coin(self, event=None):
        val = self.coin_box.get()
        if val in self._coin_lookup:
            base = self._coin_lookup[val]
        else:
            base = val.split()[0].strip()
        self._current_coin_base = base
        new_sym = base + "USDT"
        if new_sym == self.symbol:
            return
        self.symbol = new_sym
        self.price = 0.0
        self.prev_price = 0.0
        self.mark_price = 0.0
        self.display_price = 0.0
        self.candle_open_price = 0.0
        self.price_history.clear()
        self.last_update = "--:--:--"
        self.error_msg = ""
        self.funding_rate = 0.0
        self.funding_next = 0
        self.oi_value = 0.0
        self.oi_change_pct = 0.0
        self._last_funding_update = 0
        self._last_oi_update = 0
        self._last_ob_update = 0
        self._ob_imbalance = 1.0
        with self.df_lock:
            self.candle_deque.clear()
        with self.analysis_lock:
            self.current_analysis = AnalysisResult()
        self.last_signal_direction = "WAIT"
        self.signal_history.clear()
        self.signal_stats = {"buy": 0, "sell": 0, "wait": 0}
        with self._bt_lock:
            self._bt_signals = []
            self._bt_wins = 0
            self._bt_losses = 0
            self._bt_open = None
        self.intel_sr_levels = []
        self.intel_trend500 = "WAIT"
        self.intel_trend_str = ""
        self.intel_ema_stack = "--"
        self.intel_structure = "--"
        self.intel_slope = "--"
        self.intel_liq_buy = 0.0
        self.intel_liq_sell = 0.0
        self.intel_liq_dom = "--"
        self.intel_liq_zones = "--"
        self.intel_liq_sweep = "--"
        self.intel_news = []
        self.intel_news_sentiment = "--"
        self._mtf_cached_results = {}
        self.mtf_master = "WAIT"
        self.mtf_agree = 0
        self._mtf_last_fetch_time = 0
        self._last_narrative_update = 0
        self._narrative_cache = {}
        self._comparison_symbol = None
        self.ws_connected = False
        with self._ws_lock:
            self._close_ws()
        self.connect_websockets()
        self.load_historical_klines()
        self.status_lbl.config(text=f"Switched to {base}", fg="#02c076")
        threading.Thread(target=self._intel_thread, daemon=True).start()
        threading.Thread(target=self._news_thread, daemon=True).start()
        self.run_mtf_scan()

    def change_tf(self, event=None):
        new_tf = self.tf_box.get()
        if new_tf == self.interval:
            return
        self.interval = new_tf
        with self.df_lock:
            self.candle_deque.clear()
        with self.analysis_lock:
            self.current_analysis = AnalysisResult()
        self.last_signal_direction = "WAIT"
        self._last_analysis_time = 0.0
        self._mtf_cached_results = {}
        self.mtf_master = "WAIT"
        self.mtf_agree = 0
        self._mtf_last_fetch_time = 0
        with self._ws_lock:
            self._close_ws()
        self.connect_websockets()
        self.load_historical_klines()
        self.status_lbl.config(text=f"TF changed to {new_tf}", fg="#02c076")
        threading.Thread(target=self._intel_thread, daemon=True).start()
        self.run_mtf_scan()

    def save_signal(self):
        with self.analysis_lock:
            r = self.current_analysis
        cp = self.mark_price if self.mark_price > 0 else self.price
        if cp <= 0 or r.direction == "WAIT":
            self.status_lbl.config(text="No signal to save", fg="#cf304a")
            return
        entry = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": self.symbol,
            "tf": self.interval,
            "direction": r.direction,
            "strength": r.strength,
            "confidence": round(r.confidence, 1),
            "price": round(cp, 6),
            "sl": round(r.smart_sl if r.smart_sl > 0 else r.sl, 6),
            "tp": round(r.smart_tp if r.smart_tp > 0 else r.tp, 6),
            "rr": round(r.rr, 1),
            "pos_size": round(r.position_size_pct, 2),
            "win_est": round(r.win_rate_est, 1),
            "entry_method": r.entry_method,
            "session_valid": r.session_valid,
        }
        try:
            with open("saved_signals.json", "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
            self.status_lbl.config(text="Signal saved!", fg="#02c076")
        except Exception as e:
            self.status_lbl.config(text=f"Save failed: {e}", fg="#cf304a")

    def test_alert(self):
        alert_sound()
        self.flash_alert()
        self.status_lbl.config(text="Alert test triggered!", fg="#02c076")

    def flash_alert(self):
        try:
            orig_bg = self.root.cget("bg")
            self.root.configure(bg="#f0b90b")
            self.root.after(150, lambda: self.root.configure(bg=orig_bg))
            self.root.after(300, lambda: self.root.configure(bg="#f0b90b"))
            self.root.after(450, lambda: self.root.configure(bg=orig_bg))
        except Exception:
            pass

    def on_close(self):
        self.running = False
        with self._ws_lock:
            self._close_ws()
        self.root.destroy()

    def set_error(self, msg):
        self.error_msg = msg[:100]
        logging.warning("Error: %s", msg)



# ════════════════════════════════════════════════════════════════════
#  HEADLESS / TELEGRAM BOT SUPPORT
#  When running without a display (or explicitly in headless mode),
#  FakeRoot replaces tk.Tk() so all GUI calls become no-ops.
# ════════════════════════════════════════════════════════════════════

class _FakeMock:
    """Absorbs any attribute access / call silently."""
    def __getattr__(self, name): return self
    def __call__(self, *a, **kw): return self
    def pack(self, *a, **kw): pass
    def config(self, *a, **kw): pass
    def configure(self, *a, **kw): pass
    def grid(self, *a, **kw): pass
    def place(self, *a, **kw): pass
    def winfo_width(self): return 400
    def update_idletasks(self): pass
    def column(self, *a, **kw): pass
    def heading(self, *a, **kw): pass
    def insert(self, *a, **kw): pass
    def delete(self, *a, **kw): pass
    def get(self, *a, **kw): return ""
    def set(self, *a, **kw): pass
    def get_children(self): return []
    def selection(self): return []
    def yview(self, *a, **kw): pass
    def xview(self, *a, **kw): pass
    def bind(self, *a, **kw): pass
    def tag_configure(self, *a, **kw): pass
    def __iter__(self): return iter([])
    def __bool__(self): return True


class FakeRoot:
    """
    Drop-in replacement for tk.Tk() — allows BinanceRadarPro to run
    as a background service with no display.
    """
    def __init__(self):
        self._bg = "#0b0e11"
        self._running = True

    def title(self, t): pass
    def geometry(self, g): pass
    def configure(self, **kw):
        if "bg" in kw:
            self._bg = kw["bg"]
    def resizable(self, *a): pass
    def protocol(self, name, cb): pass

    def after(self, ms, func=None, *args):
        """
        Schedule `func` to run after `ms` milliseconds.
        Uses a daemon Timer so it fires even without mainloop.
        """
        if func is not None:
            t = threading.Timer(ms / 1000.0, func, args=args)
            t.daemon = True
            t.start()

    def cget(self, key):
        if key == "bg":
            return self._bg
        return ""

    def destroy(self):
        self._running = False

    def mainloop(self):
        """Block the calling thread until destroy() is called."""
        while self._running:
            time.sleep(0.5)

    # GUI widget factories — all return silent mocks
    def __getattr__(self, name):
        return _FakeMock()


# ─── Patch BinanceRadarPro to work headlessly ──────────────────────

_original_build_ui = BinanceRadarPro.build_ui
_original_refresh_ui = BinanceRadarPro.refresh_ui


def _safe_build_ui(self):
    """Build the real UI only when a real Tk root is present."""
    if isinstance(self.root, FakeRoot):
        mock = _FakeMock()
        _ALL_WIDGET_ATTRS = (
            "status_lbl", "ws_status_lbl", "ml_lbl", "err_lbl", "ws_live_lbl",
            "bb_status_lbl",
            "price_lbl", "dir_lbl", "conf_lbl", "strength_lbl",
            "entry_lbl", "sl_lbl", "tp_lbl", "rr_lbl",
            "pos_lbl", "win_lbl", "struct_lbl", "trend_lbl",
            "vol_lbl", "conflict_lbl", "session_lbl", "reason_lbl",
            "fore_dir_lbl", "fore_conf_lbl", "fore_entry_lbl",
            "fore_sl_lbl", "fore_tp_lbl", "fore_rr_lbl",
            "_meter_dir_lbl", "_meter_conf_lbl", "_meter_strength_lbl",
            "_ai_dir_lbl", "_ai_conf_lbl", "_ai_strength_lbl",
            "_ai_entry_lbl", "_ai_sl_lbl", "_ai_tp_lbl",
            "_ai_rr_lbl", "_ai_session_lbl", "_ai_conflict_lbl",
            "_ai_reason_lbl",
            "_market_pred_dir_lbl", "_market_pred_price_lbl",
            "_ema_trend_lbl", "_supertrend_lbl",
            "_intel_status_lbl", "_news_sentiment_lbl", "_mtf_status_lbl",
            "_coin_tv", "_tab_frame",
        )
        for attr in _ALL_WIDGET_ATTRS:
            setattr(self, attr, mock)
        logging.info("[HEADLESS] build_ui skipped — running in background mode")
    else:
        _original_build_ui(self)


def _safe_refresh_ui(self):
    """Refresh the UI only when a real Tk root is present."""
    if isinstance(self.root, FakeRoot):
        return
    try:
        _original_refresh_ui(self)
    except Exception as e:
        logging.debug("refresh_ui error: %s", e)


BinanceRadarPro.build_ui   = _safe_build_ui
BinanceRadarPro.refresh_ui = _safe_refresh_ui


def _safe_flash_alert(self):
    if isinstance(self.root, FakeRoot):
        return
    try:
        orig_bg = self.root.cget("bg")
        self.root.configure(bg="#f0b90b")
        self.root.after(150, lambda: self.root.configure(bg=orig_bg))
        self.root.after(300, lambda: self.root.configure(bg="#f0b90b"))
        self.root.after(450, lambda: self.root.configure(bg=orig_bg))
    except Exception:
        pass


BinanceRadarPro.flash_alert = _safe_flash_alert


# ════════════════════════════════════════════════════════════════════
#  TELEGRAM BOT  (embedded — no separate file needed)
# ════════════════════════════════════════════════════════════════════

import json as _json_tg
import requests as _requests_tg

TG_TOKEN   = "8719430579:AAEbcBu7yZ9cr_LtaKLR8XsgJ6kIabg_J0o"
TG_BASE    = f"https://api.telegram.org/bot{TG_TOKEN}"

# ── Subscribers ──────────────────────────────────────────────────────
_TG_SUBSCRIBERS: set = set()
_TG_SUBS_LOCK = threading.Lock()
_tg_bot_offset = 0
_tg_radar_ref  = None  # set by start_telegram_bot()


def _tg_get_radar():
    return _tg_radar_ref


# ── Low-level API helpers ─────────────────────────────────────────────

def _tg_send(chat_id: int, text: str, reply_markup=None, parse_mode="HTML"):
    payload = {"chat_id": chat_id, "text": text, "parse_mode": parse_mode}
    if reply_markup:
        payload["reply_markup"] = _json_tg.dumps(reply_markup)
    try:
        r = _requests_tg.post(f"{TG_BASE}/sendMessage", json=payload, timeout=10)
        return r.json()
    except Exception as e:
        logging.warning("tg_send error: %s", e)
        return {}


def _tg_edit(chat_id: int, message_id: int, text: str, reply_markup=None):
    payload = {
        "chat_id": chat_id, "message_id": message_id,
        "text": text, "parse_mode": "HTML",
    }
    if reply_markup:
        payload["reply_markup"] = _json_tg.dumps(reply_markup)
    try:
        r = _requests_tg.post(f"{TG_BASE}/editMessageText", json=payload, timeout=10)
        return r.json()
    except Exception as e:
        logging.warning("tg_edit error: %s", e)
        return {}


def _tg_answer(callback_id: str, text: str = ""):
    try:
        _requests_tg.post(f"{TG_BASE}/answerCallbackQuery", json={
            "callback_query_id": callback_id,
            "text": text, "show_alert": False,
        }, timeout=5)
    except Exception:
        pass


def _tg_get_updates(offset: int = 0):
    try:
        r = _requests_tg.get(f"{TG_BASE}/getUpdates", params={
            "offset": offset, "timeout": 30,
            "allowed_updates": ["message", "callback_query"],
        }, timeout=35)
        return r.json().get("result", [])
    except Exception as e:
        logging.warning("get_updates error: %s", e)
        return []


# ── Keyboards ─────────────────────────────────────────────────────────

_TG_MAIN_KB = {
    "inline_keyboard": [
        [{"text": "📊 السيگنال الحالي", "callback_data": "signal"},
         {"text": "📈 MTF كامل",        "callback_data": "mtf"}],
        [{"text": "🔍 Intel / SR Levels","callback_data": "intel"},
         {"text": "📰 الأخبار",          "callback_data": "news"}],
        [{"text": "📂 سجل السيگنالات",   "callback_data": "history"},
         {"text": "🏆 Win Rate AI",      "callback_data": "winrate"}],
        [{"text": "⚙️ تغيير العملة",     "callback_data": "change_coin"},
         {"text": "⏱ تغيير الإطار",     "callback_data": "change_tf"}],
        [{"text": "💾 حفظ السيگنال",     "callback_data": "save_signal"},
         {"text": "🔄 Refresh",          "callback_data": "refresh"}],
        [{"text": "🔔 تفعيل الإشعارات", "callback_data": "subscribe"},
         {"text": "🔕 إلغاء الإشعارات", "callback_data": "unsubscribe"}],
        [{"text": "📉 Backtest",         "callback_data": "backtest"},
         {"text": "🗑 مسح السجل",       "callback_data": "clear_history"}],
    ]
}

_TG_COIN_KB = {
    "inline_keyboard": [
        [{"text": "BTC",  "callback_data": "coin_BTC"},
         {"text": "ETH",  "callback_data": "coin_ETH"},
         {"text": "BNB",  "callback_data": "coin_BNB"}],
        [{"text": "SOL",  "callback_data": "coin_SOL"},
         {"text": "XRP",  "callback_data": "coin_XRP"},
         {"text": "DOGE", "callback_data": "coin_DOGE"}],
        [{"text": "ADA",  "callback_data": "coin_ADA"},
         {"text": "AVAX", "callback_data": "coin_AVAX"},
         {"text": "DOT",  "callback_data": "coin_DOT"}],
        [{"text": "LINK", "callback_data": "coin_LINK"},
         {"text": "LTC",  "callback_data": "coin_LTC"},
         {"text": "NEAR", "callback_data": "coin_NEAR"}],
        [{"text": "« رجوع", "callback_data": "main_menu"}],
    ]
}

_TG_TF_KB = {
    "inline_keyboard": [
        [{"text": "5m",  "callback_data": "tf_5m"},
         {"text": "15m", "callback_data": "tf_15m"},
         {"text": "1h",  "callback_data": "tf_1h"}],
        [{"text": "4h",  "callback_data": "tf_4h"},
         {"text": "1d",  "callback_data": "tf_1d"}],
        [{"text": "« رجوع", "callback_data": "main_menu"}],
    ]
}


# ── Message formatters ────────────────────────────────────────────────

def _tg_dir_emoji(d: str) -> str:
    return {"BUY": "🟢 BUY", "SELL": "🔴 SELL"}.get(d, "⏸ WAIT")


def _tg_format_signal(radar: "BinanceRadarPro") -> str:
    with radar.analysis_lock:
        r = radar.current_analysis
    cp = radar.mark_price if radar.mark_price > 0 else radar.price
    if cp <= 0:
        return "⚠️ لا يوجد سعر بعد، انتظر لحظة..."

    sl_val = r.smart_sl if r.smart_sl > 0 else r.sl
    tp_val = r.smart_tp if r.smart_tp > 0 else r.tp
    conf_bar = "█" * int(r.confidence / 10) + "░" * (10 - int(r.confidence / 10))

    lines = [
        f"<b>🎯 BinanceRadarPro v{VERSION} — {radar.symbol} [{radar.interval}]</b>",
        "",
        f"💰 السعر: <b>{cp:.4f}</b>",
        f"📡 الاتجاه: <b>{_tg_dir_emoji(r.direction)}</b>",
        f"💪 القوة: <b>{r.strength}</b>",
        "",
        f"📊 الثقة: <b>{r.confidence:.1f}%</b>",
        f"  {conf_bar}",
        "",
        f"🎯 Entry: <b>{r.smart_entry_low:.4f} — {r.smart_entry_high:.4f}</b>",
        f"  [{r.entry_method}]",
        f"🛑 SL: <b>{sl_val:.4f}</b>",
        f"✅ TP: <b>{tp_val:.4f}</b>",
    ]
    if r.smart_tp1 > 0:
        lines.append(f"  TP1: {r.smart_tp1:.4f}  TP2: {r.smart_tp2:.4f}  TP3: {r.smart_tp3:.4f}")
    lines += [
        f"⚖️ R:R  1:{r.rr:.1f}",
        f"📦 Position: <b>{r.position_size_pct:.2f}%</b>",
        f"🏆 Win Rate Est: <b>{r.win_rate_est:.0f}%</b>",
        "",
        f"🏗 Structure: {'✅' if r.structure_aligned else '❌'} {r.swing_structure}",
        f"📈 Trend: {'✅' if r.trend_aligned else '❌'}  Vol: {'✅' if r.volume_confirmed else '❌'}",
        f"⚡ {'⚠️ CONFLICT' if r.indicator_conflict else '✅ No Conflict'}",
        "",
        f"🕐 {datetime.now().strftime('%H:%M:%S')}",
    ]
    if r.reason:
        lines.insert(-1, f"💬 <i>{r.reason[:200]}</i>")
    return "\n".join(lines)


def _tg_format_mtf(radar: "BinanceRadarPro") -> str:
    with radar._mtf_data_lock:
        cached = dict(radar._mtf_cached_results)
    master = getattr(radar, "_mtf_master", "WAIT")
    lines = [
        f"<b>📈 MTF Analysis — {radar.symbol}</b>",
        "",
        f"🔑 MASTER: <b>{_tg_dir_emoji(master)}</b>",
        f"✅ Agree: <b>{radar.mtf_agree}/5</b>",
        "",
    ]
    for tf in ("5m", "15m", "1h", "4h", "1d"):
        res = cached.get(tf, ("WAIT", 0.0))
        direction, conf = res
        icon = {"5m": "⚡", "15m": "🕐", "1h": "⏰", "4h": "📅", "1d": "🗓"}.get(tf, "")
        d_icon = {"BUY": "🟢", "SELL": "🔴"}.get(direction, "⏸")
        lines.append(f"  {icon} {tf:<5} {d_icon} {direction:<5} {conf:.0f}%")
    lines.append(f"\n🕐 {datetime.now().strftime('%H:%M:%S')}")
    return "\n".join(lines)


def _tg_format_intel(radar: "BinanceRadarPro") -> str:
    cp = radar.mark_price if radar.mark_price > 0 else radar.price
    sr_text = "\n".join(
        f"  {lvl:.4f}{' ← السعر' if abs(lvl - cp) / (cp + 1e-9) < 0.005 else ''}"
        for lvl in sorted(radar.intel_sr_levels)[-8:]
    ) or "  لا يوجد بيانات"
    liq_icon = "🟢" if "BUY" in str(radar.intel_liq_dom).upper() else (
               "🔴" if "SELL" in str(radar.intel_liq_dom).upper() else "⚪")
    lines = [
        f"<b>🔍 Market Intel — {radar.symbol} [{radar.interval}]</b>",
        "",
        f"📊 Trend(500): <b>{radar.intel_trend500}</b>  Str: {radar.intel_trend_str}",
        f"📏 EMA Stack: {radar.intel_ema_stack}  Struct: {radar.intel_structure}",
        "",
        f"💧 Liquidity:",
        f"  Buy: {radar.intel_liq_buy:.2f}  Sell: {radar.intel_liq_sell:.2f}",
        f"  {liq_icon} Dom: <b>{radar.intel_liq_dom}</b>",
        f"  Zones: {radar.intel_liq_zones}  Sweep: {radar.intel_liq_sweep}",
        "",
        f"📍 S/R Levels:",
        sr_text,
        "",
        f"🕐 {datetime.now().strftime('%H:%M:%S')}",
    ]
    return "\n".join(lines)


def _tg_format_news(radar: "BinanceRadarPro") -> str:
    lines = [f"<b>📰 الأخبار — {radar.symbol}</b>", "",
             f"المزاج: <b>{radar.intel_news_sentiment}</b>", ""]
    if radar.intel_news:
        for item in radar.intel_news[:8]:
            if isinstance(item, dict):
                title = item.get("title", str(item))[:120]
                pub   = item.get("pubDate", "")[:16]
            else:
                title, pub = str(item)[:120], ""
            lines.append(f"• {title}{f'  <i>{pub}</i>' if pub else ''}")
    else:
        lines.append("لا تتوفر أخبار حالياً")
    lines.append(f"\n🕐 {datetime.now().strftime('%H:%M:%S')}")
    return "\n".join(lines)


def _tg_format_history(radar: "BinanceRadarPro") -> str:
    sigs  = list(radar.signal_history)
    stats = radar.signal_stats
    lines = [
        f"<b>📂 سجل السيگنالات — {radar.symbol}</b>",
        "",
        f"🟢 BUY: {stats['buy']}   🔴 SELL: {stats['sell']}   ⏸ WAIT: {stats['wait']}",
        "",
    ]
    if not sigs:
        lines.append("لا توجد سيگنالات بعد")
    else:
        for s in sigs[:10]:
            d_icon = "🟢" if s["direction"] == "BUY" else "🔴"
            lines.append(
                f"{d_icon} <b>{s['direction']}</b> {s['symbol']} [{s['tf']}] @ {s['price']:.4f}\n"
                f"   SL:{s['sl']:.4f}  TP:{s['tp']:.4f}  R:R 1:{s['rr']}  Conf:{s['confidence']}%\n"
                f"   <i>{s['time']}</i>\n"
            )
    return "\n".join(lines)


def _tg_format_backtest(radar: "BinanceRadarPro") -> str:
    with radar._bt_lock:
        wins   = radar._bt_wins
        losses = radar._bt_losses
        open_t = radar._bt_open
        sigs   = list(radar._bt_signals[-10:])
    total = wins + losses
    wr = (wins / total * 100) if total > 0 else 0
    lines = [
        f"<b>📉 Backtest Live — {radar.symbol}</b>",
        "",
        f"✅ Wins: {wins}  ❌ Losses: {losses}",
        f"📊 Win Rate: <b>{wr:.1f}%</b>",
        "",
    ]
    if open_t:
        cp = radar.mark_price if radar.mark_price > 0 else radar.price
        pnl_icon = "🟢" if (
            (open_t["dir"] == "BUY"  and cp > open_t["entry"]) or
            (open_t["dir"] == "SELL" and cp < open_t["entry"])
        ) else "🔴"
        lines += [f"📂 مفتوحة: {pnl_icon} {open_t['dir']} @ {open_t['entry']:.4f}  CP: {cp:.4f}", ""]
    if sigs:
        lines.append("آخر الصفقات:")
        for s in reversed(sigs):
            r_icon = "✅" if s.get("result") == "WIN" else "❌"
            lines.append(f"  {r_icon} {s['dir']} {s['entry']:.4f}→{s.get('exit','?')} [{s.get('result','?')}]")
    lines.append(f"\n🕐 {datetime.now().strftime('%H:%M:%S')}")
    return "\n".join(lines)


def _tg_format_winrate(radar: "BinanceRadarPro") -> str:
    with radar._winrate_lock:
        preds  = list(radar._winrate_predictions)
        wins   = radar._winrate_wins
        losses = radar._winrate_losses
    total_ev = wins + losses
    wr = (wins / total_ev * 100) if total_ev > 0 else 0
    evaluated = [p for p in preds if p.get("evaluated")]
    pending   = [p for p in preds if not p.get("evaluated")]
    lines = [
        f"<b>🏆 AI Win Rate — {radar.symbol}</b>",
        "",
        f"✅ Wins: {wins}  ❌ Losses: {losses}  ⏳ Pending: {len(pending)}",
        f"📊 Win Rate: <b>{wr:.1f}%</b>",
        "",
        "آخر التقييمات:",
    ]
    for p in evaluated[-5:]:
        r_icon = "✅" if p.get("result") == "correct" else "❌"
        lines.append(f"  {r_icon} {p['direction']} {p['symbol']} @ {p['start_price']:.4f}  <i>{p['time']}</i>")
    lines.append(f"\n🕐 {datetime.now().strftime('%H:%M:%S')}")
    return "\n".join(lines)


# ── Command / callback handlers ────────────────────────────────────────

def _tg_handle_start(chat_id: int):
    radar = _tg_get_radar()
    if radar:
        cp = radar.mark_price if radar.mark_price > 0 else radar.price
        status = f"\n\n💰 {radar.symbol} [{radar.interval}] @ {cp:.4f}"
    else:
        status = "\n\n⚠️ الرادار لا يزال يُحمّل..."
    _tg_send(chat_id,
        f"<b>🚀 BinanceRadarPro v{VERSION} ULTRA</b>\n"
        f"AI-Powered Crypto Signals Bot{status}\n\n"
        f"اختر أمراً من القائمة:",
        reply_markup=_TG_MAIN_KB,
    )


def _tg_handle_callback(chat_id: int, message_id: int, cb_id: str, data: str):
    radar = _tg_get_radar()
    _tg_answer(cb_id, "⏳ جاري المعالجة...")

    if data == "main_menu":
        _tg_edit(chat_id, message_id, "🎛 <b>القائمة الرئيسية</b>\nاختر أمراً:", reply_markup=_TG_MAIN_KB)
        return

    if not radar:
        _tg_edit(chat_id, message_id, "⚠️ الرادار لا يزال يُحمّل، حاول مجدداً بعد لحظة.", reply_markup=_TG_MAIN_KB)
        return

    _back_kb = {"inline_keyboard": [[{"text": "« رجوع", "callback_data": "main_menu"}]]}

    if data == "signal":
        _tg_edit(chat_id, message_id, _tg_format_signal(radar),
                 reply_markup={"inline_keyboard": [
                     [{"text": "🔄 تحديث", "callback_data": "signal"},
                      {"text": "💾 حفظ",   "callback_data": "save_signal"}],
                     [{"text": "« رجوع", "callback_data": "main_menu"}],
                 ]})

    elif data == "mtf":
        _tg_edit(chat_id, message_id, _tg_format_mtf(radar),
                 reply_markup={"inline_keyboard": [
                     [{"text": "🔄 Rescan MTF", "callback_data": "run_mtf"}],
                     [{"text": "« رجوع", "callback_data": "main_menu"}],
                 ]})

    elif data == "run_mtf":
        _tg_answer(cb_id, "⏳ جاري فحص MTF...")
        try:
            threading.Thread(target=radar.run_mtf_scan, daemon=True).start()
            time.sleep(2)
        except Exception:
            pass
        _tg_edit(chat_id, message_id, _tg_format_mtf(radar),
                 reply_markup={"inline_keyboard": [
                     [{"text": "🔄 Rescan MTF", "callback_data": "run_mtf"}],
                     [{"text": "« رجوع", "callback_data": "main_menu"}],
                 ]})

    elif data == "intel":
        _tg_edit(chat_id, message_id, _tg_format_intel(radar),
                 reply_markup={"inline_keyboard": [
                     [{"text": "🔄 Refresh Intel", "callback_data": "refresh_intel"}],
                     [{"text": "« رجوع", "callback_data": "main_menu"}],
                 ]})

    elif data == "refresh_intel":
        _tg_answer(cb_id, "⏳ جاري تحديث Intel...")
        try:
            threading.Thread(target=radar.refresh_intel, daemon=True).start()
            time.sleep(3)
        except Exception:
            pass
        _tg_edit(chat_id, message_id, _tg_format_intel(radar),
                 reply_markup={"inline_keyboard": [
                     [{"text": "🔄 Refresh Intel", "callback_data": "refresh_intel"}],
                     [{"text": "« رجوع", "callback_data": "main_menu"}],
                 ]})

    elif data == "news":
        _tg_edit(chat_id, message_id, _tg_format_news(radar),
                 reply_markup={"inline_keyboard": [
                     [{"text": "🔄 Refresh News", "callback_data": "refresh_news"}],
                     [{"text": "« رجوع", "callback_data": "main_menu"}],
                 ]})

    elif data == "refresh_news":
        _tg_answer(cb_id, "⏳ جاري تحديث الأخبار...")
        try:
            threading.Thread(target=radar.refresh_news, daemon=True).start()
            time.sleep(3)
        except Exception:
            pass
        _tg_edit(chat_id, message_id, _tg_format_news(radar),
                 reply_markup={"inline_keyboard": [
                     [{"text": "🔄 Refresh News", "callback_data": "refresh_news"}],
                     [{"text": "« رجوع", "callback_data": "main_menu"}],
                 ]})

    elif data == "history":
        _tg_edit(chat_id, message_id, _tg_format_history(radar),
                 reply_markup={"inline_keyboard": [
                     [{"text": "🗑 مسح السجل", "callback_data": "clear_history"}],
                     [{"text": "« رجوع", "callback_data": "main_menu"}],
                 ]})

    elif data == "clear_history":
        radar._clear_signal_history()
        _tg_edit(chat_id, message_id, "✅ تم مسح سجل السيگنالات.", reply_markup=_back_kb)

    elif data == "winrate":
        _tg_edit(chat_id, message_id, _tg_format_winrate(radar),
                 reply_markup={"inline_keyboard": [
                     [{"text": "🔄 تحديث", "callback_data": "winrate"}],
                     [{"text": "« رجوع", "callback_data": "main_menu"}],
                 ]})

    elif data == "change_coin":
        _tg_edit(chat_id, message_id,
                 f"⚙️ <b>اختر العملة</b>\n(الحالية: {radar.symbol})",
                 reply_markup=_TG_COIN_KB)

    elif data.startswith("coin_"):
        coin    = data[5:]
        new_sym = coin + "USDT"
        if new_sym != radar.symbol:
            _tg_answer(cb_id, f"⏳ جاري التبديل إلى {coin}...")
            radar.symbol = new_sym
            radar._current_coin_base = coin
            radar.price = radar.mark_price = 0.0
            radar.candle_deque.clear()
            with radar.analysis_lock:
                radar.current_analysis = AnalysisResult()
            radar.last_signal_direction = "WAIT"
            radar.signal_history.clear()
            radar.signal_stats = {"buy": 0, "sell": 0, "wait": 0}
            with radar._bt_lock:
                radar._bt_signals = []; radar._bt_wins = 0
                radar._bt_losses  = 0; radar._bt_open  = None
            radar._mtf_cached_results = {}
            radar.mtf_agree = 0
            radar._mtf_last_fetch_time = 0
            try:
                with radar._ws_lock:
                    radar._close_ws()
                radar.connect_websockets()
                threading.Thread(target=radar.load_historical_klines, daemon=True).start()
                threading.Thread(target=radar._intel_thread, daemon=True).start()
                threading.Thread(target=radar._news_thread,  daemon=True).start()
            except Exception as e:
                logging.warning("Coin switch WS error: %s", e)
        _tg_edit(chat_id, message_id,
                 f"✅ تم التبديل إلى <b>{coin}USDT</b>\n\nيتم الآن تحميل البيانات...",
                 reply_markup={"inline_keyboard": [
                     [{"text": "📊 السيگنال الحالي", "callback_data": "signal"}],
                     [{"text": "« رجوع", "callback_data": "main_menu"}],
                 ]})

    elif data == "change_tf":
        _tg_edit(chat_id, message_id,
                 f"⏱ <b>اختر الإطار الزمني</b>\n(الحالي: {radar.interval})",
                 reply_markup=_TG_TF_KB)

    elif data.startswith("tf_"):
        new_tf = data[3:]
        if new_tf != radar.interval:
            _tg_answer(cb_id, f"⏳ جاري التبديل إلى {new_tf}...")
            radar.interval = new_tf
            radar.candle_deque.clear()
            with radar.analysis_lock:
                radar.current_analysis = AnalysisResult()
            radar.last_signal_direction = "WAIT"
            radar._last_analysis_time = 0.0
            radar._mtf_cached_results = {}
            radar.mtf_agree = 0
            radar._mtf_last_fetch_time = 0
            try:
                with radar._ws_lock:
                    radar._close_ws()
                radar.connect_websockets()
                threading.Thread(target=radar.load_historical_klines, daemon=True).start()
                threading.Thread(target=radar._intel_thread, daemon=True).start()
            except Exception as e:
                logging.warning("TF switch error: %s", e)
        _tg_edit(chat_id, message_id,
                 f"✅ تم التبديل إلى <b>{new_tf}</b>",
                 reply_markup={"inline_keyboard": [
                     [{"text": "📊 السيگنال الحالي", "callback_data": "signal"}],
                     [{"text": "« رجوع", "callback_data": "main_menu"}],
                 ]})

    elif data == "save_signal":
        with radar.analysis_lock:
            r = radar.current_analysis
        cp = radar.mark_price if radar.mark_price > 0 else radar.price
        if cp <= 0 or r.direction == "WAIT":
            _tg_answer(cb_id, "⚠️ لا يوجد سيگنال لحفظه الآن")
            return
        entry = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": radar.symbol, "tf": radar.interval,
            "direction": r.direction, "strength": r.strength,
            "confidence": round(r.confidence, 1), "price": round(cp, 6),
            "sl": round(r.smart_sl if r.smart_sl > 0 else r.sl, 6),
            "tp": round(r.smart_tp if r.smart_tp > 0 else r.tp, 6),
            "rr": round(r.rr, 1), "pos_size": round(r.position_size_pct, 2),
            "win_est": round(r.win_rate_est, 1), "entry_method": r.entry_method,
        }
        try:
            with open("saved_signals_tg.json", "a", encoding="utf-8") as f:
                f.write(_json_tg.dumps(entry) + "\n")
            msg = (f"💾 <b>تم حفظ السيگنال!</b>\n\n"
                   f"{_tg_dir_emoji(r.direction)} {radar.symbol} [{radar.interval}]\n"
                   f"@ {cp:.4f}  Conf: {r.confidence:.1f}%\n"
                   f"SL: {entry['sl']}  TP: {entry['tp']}")
        except Exception as e:
            msg = f"❌ خطأ في الحفظ: {e}"
        _tg_edit(chat_id, message_id, msg, reply_markup=_back_kb)

    elif data == "refresh":
        cp = radar.mark_price if radar.mark_price > 0 else radar.price
        _tg_edit(chat_id, message_id,
                 f"🔄 <b>تم التحديث</b>\n{radar.symbol} [{radar.interval}] @ {cp:.4f}\n"
                 f"{datetime.now().strftime('%H:%M:%S')}",
                 reply_markup=_TG_MAIN_KB)

    elif data == "backtest":
        _tg_edit(chat_id, message_id, _tg_format_backtest(radar),
                 reply_markup={"inline_keyboard": [
                     [{"text": "🔄 تحديث", "callback_data": "backtest"},
                      {"text": "🗑 Reset",  "callback_data": "reset_bt"}],
                     [{"text": "« رجوع", "callback_data": "main_menu"}],
                 ]})

    elif data == "reset_bt":
        radar._reset_backtest()
        _tg_edit(chat_id, message_id, "✅ تم إعادة تعيين Backtest.", reply_markup=_back_kb)

    elif data == "subscribe":
        with _TG_SUBS_LOCK:
            _TG_SUBSCRIBERS.add(chat_id)
        _tg_answer(cb_id, "✅ تم تفعيل الإشعارات!")
        _tg_edit(chat_id, message_id,
                 "🔔 <b>تم تفعيل الإشعارات!</b>\nستصلك السيگنالات الجديدة تلقائياً.",
                 reply_markup=_TG_MAIN_KB)

    elif data == "unsubscribe":
        with _TG_SUBS_LOCK:
            _TG_SUBSCRIBERS.discard(chat_id)
        _tg_answer(cb_id, "🔕 تم إلغاء الإشعارات")
        _tg_edit(chat_id, message_id, "🔕 <b>تم إلغاء الإشعارات</b>", reply_markup=_TG_MAIN_KB)

    else:
        _tg_answer(cb_id, "غير معروف")


# ── Polling loop ───────────────────────────────────────────────────────

def _tg_polling_loop():
    global _tg_bot_offset
    logging.info("🤖 Telegram polling بدأ...")
    while True:
        try:
            updates = _tg_get_updates(_tg_bot_offset)
            for upd in updates:
                _tg_bot_offset = upd["update_id"] + 1
                if "message" in upd:
                    msg  = upd["message"]
                    cid  = msg["chat"]["id"]
                    _tg_handle_start(cid)
                elif "callback_query" in upd:
                    cq   = upd["callback_query"]
                    cid  = cq["message"]["chat"]["id"]
                    mid  = cq["message"]["message_id"]
                    _tg_handle_callback(cid, mid, cq["id"], cq.get("data", ""))
        except KeyboardInterrupt:
            logging.info("إيقاف Telegram polling...")
            break
        except Exception as e:
            logging.error("polling error: %s", e)
            time.sleep(3)


# ── Auto signal notifier ───────────────────────────────────────────────

def _tg_notifier_loop():
    last_sig_time = ""
    while True:
        try:
            radar = _tg_get_radar()
            if radar and radar.signal_history:
                latest   = radar.signal_history[0]
                sig_time = latest.get("time", "")
                if sig_time != last_sig_time and latest.get("direction") in ("BUY", "SELL"):
                    last_sig_time = sig_time
                    with _TG_SUBS_LOCK:
                        subs = set(_TG_SUBSCRIBERS)
                    if subs:
                        alert_msg = f"🚨 <b>سيگنال جديد!</b>\n\n{_tg_format_signal(radar)}"
                        for cid in subs:
                            _tg_send(cid, alert_msg, reply_markup=_TG_MAIN_KB)
        except Exception as e:
            logging.warning("notifier error: %s", e)
        time.sleep(5)


# ── Public launcher ────────────────────────────────────────────────────

def start_telegram_bot(radar_instance: "BinanceRadarPro"):
    """
    Launch the Telegram bot in the background.
    Call this *after* BinanceRadarPro has been created and started.
    """
    global _tg_radar_ref
    _tg_radar_ref = radar_instance
    threading.Thread(target=_tg_notifier_loop, daemon=True, name="tg-notifier").start()
    threading.Thread(target=_tg_polling_loop,  daemon=True, name="tg-polling").start()
    logging.info("✅ Telegram bot threads launched (polling + notifier)")


# ════════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
#  --gui        : open the Tkinter window (default when display is available)
#  --headless   : background mode, Telegram bot only
#  (auto-detect): tries GUI, falls back to headless if no display
# ════════════════════════════════════════════════════════════════════

def _run_headless():
    """Run radar engine + Telegram bot with no GUI."""
    logging.info("=" * 55)
    logging.info(f"🚀 BinanceRadarPro v{VERSION} — HEADLESS / TELEGRAM MODE")
    logging.info("=" * 55)

    root   = FakeRoot()
    radar  = BinanceRadarPro(root)
    logging.info("✅ رادار جاهز: %s [%s]", radar.symbol, radar.interval)

    # Give the radar a few seconds to load initial data
    time.sleep(5)

    # Start Telegram bot
    start_telegram_bot(radar)
    logging.info("✅ البوت جاهز! ابدأ المحادثة بـ /start")

    # Keep alive — block the main thread
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        logging.info("إيقاف...")
        radar.running = False


def _run_gui():
    """Run the full Tkinter GUI + Telegram bot in background."""
    logging.info("=" * 55)
    logging.info(f"🚀 BinanceRadarPro v{VERSION} — GUI + TELEGRAM MODE")
    logging.info("=" * 55)

    root  = tk.Tk()
    radar = BinanceRadarPro(root)

    # Start Telegram bot in background after 5 s
    def _delayed_tg():
        time.sleep(5)
        start_telegram_bot(radar)
    threading.Thread(target=_delayed_tg, daemon=True, name="tg-init").start()

    root.mainloop()


if __name__ == "__main__":
    import sys as _sys

    _mode = None
    if "--headless" in _sys.argv:
        _mode = "headless"
    elif "--gui" in _sys.argv:
        _mode = "gui"
    else:
        # Auto-detect: if no DISPLAY env var and we're not on Windows, go headless
        import os as _os
        _on_windows = _sys.platform.startswith("win")
        _has_display = bool(_os.environ.get("DISPLAY") or _os.environ.get("WAYLAND_DISPLAY") or _on_windows)
        _mode = "gui" if _has_display else "headless"

    if _mode == "headless":
        _run_headless()
    else:
        try:
            _run_gui()
        except Exception as _gui_err:
            logging.warning("GUI failed (%s) — switching to headless mode", _gui_err)
            _run_headless()
