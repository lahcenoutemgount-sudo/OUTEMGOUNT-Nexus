import json
import sqlite3
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf

# =========================================================
# OUTEMGOUNT MA NEXUS - PROFESSIONAL EDITION
# =========================================================

st.set_page_config(
    page_title="OUTEMGOUNT MA NEXUS",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DB_PATH = "nexus_trades.db"

# ------------------------- DESIGN -------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: Inter, sans-serif; }
.stApp { background: #080e16; color: #e7edf5; }
.block-container { padding-top: 4.2rem; padding-bottom: 2rem; max-width: 1600px; }
[data-testid="stSidebar"], [data-testid="collapsedControl"] { display: none; }
h1,h2,h3 { color: #f0c75e !important; letter-spacing: .2px; }
[data-testid="stMetric"] {
    background: linear-gradient(145deg,#121e2b,#0e1722);
    border: 1px solid #25364a; border-radius: 13px; padding: 13px 15px;
}
[data-testid="stMetricLabel"] { color: #9aadc1 !important; }
[data-testid="stMetricValue"] { color: #f4f6fa !important; }
.nexus-header {
    background: linear-gradient(110deg,#111d2b,#182638 60%,#111923);
    border: 1px solid #34465a; border-radius: 18px;
    padding: 22px 24px; margin-bottom: 12px;
}
.nexus-kicker { color:#9bacc0; font-size:12px; letter-spacing:2px; font-weight:700; }
.nexus-title { color:#f0c75e; font-size:30px; font-weight:800; margin:3px 0; }
.nexus-sub { color:#aebdce; font-size:13px; }
.trade-card {
    background: linear-gradient(145deg,#142131,#101923);
    border:1px solid #2a3d52; border-radius:15px; padding:16px;
    margin: 5px 0 12px 0; box-shadow: 0 8px 24px rgba(0,0,0,.18);
}
.trade-label { color:#a9bfd3; font-size:11px; letter-spacing:1.2px; font-weight:700; }
.trade-number { color:#65c7f2; font-size:21px; font-weight:800; margin-top:3px; }
.buy { color:#35c69a !important; font-weight:800; }
.sell { color:#ff6b72 !important; font-weight:800; }
.wait { color:#f0c75e !important; font-weight:800; }
.small-note { color:#91a2b5; font-size:12px; }
hr { border-color:#26384b !important; }
.stButton>button { border-radius:10px; font-weight:700; }
footer { visibility: hidden; }
.nexus-header { text-align: center; }
.nexus-title { font-size: clamp(20px, 5.4vw, 32px) !important; white-space: nowrap; }
.live-dot { display:inline-block; width:8px; height:8px; border-radius:50%;
    background:#35c69a; box-shadow:0 0 8px #35c69a; margin-right:7px; }
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] { background:#0f1a26; border-radius:10px 10px 0 0; padding:8px 14px; }
.stTabs [aria-selected="true"] { background:#182638; color:#f0c75e; }
div[data-baseweb="select"] > div { background:#0f1a26; border-color:#2a3d52; border-radius:10px; }
@media (max-width: 640px) {
    .block-container { padding-left:.8rem; padding-right:.8rem; padding-top:4.6rem; }
    .nexus-header { padding:16px 16px; border-radius:16px; }
    .nexus-kicker { font-size:10px; letter-spacing:1.4px; }
    [data-testid="stMetricValue"] { font-size:1.5rem !important; }
    .trade-number { font-size:18px !important; }
}
</style>
""", unsafe_allow_html=True)

# ------------------------- HELPERS -------------------------

def fmt(v, digits=4):
    """Safe number formatting (handles None / NaN)."""
    if v is None or pd.isna(v):
        return "—"
    return f"{float(v):,.{digits}f}"

# ------------------------- DATABASE -------------------------

def db_connect():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    with db_connect() as con:
        con.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            asset TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            side TEXT NOT NULL,
            entry REAL NOT NULL,
            stop_loss REAL,
            tp1 REAL,
            tp2 REAL,
            tp3 REAL,
            status TEXT NOT NULL DEFAULT 'Planned',
            exit_price REAL,
            notes TEXT
        )
        """)
        con.commit()

def add_trade(asset, timeframe, side, entry, sl, tp1, tp2, tp3):
    with db_connect() as con:
        con.execute("""
            INSERT INTO trades
            (created_at,asset,timeframe,side,entry,stop_loss,tp1,tp2,tp3,status)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """, (
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
            asset, timeframe, side, float(entry),
            float(sl) if sl is not None else None,
            float(tp1) if tp1 is not None else None,
            float(tp2) if tp2 is not None else None,
            float(tp3) if tp3 is not None else None,
            "Planned"
        ))
        con.commit()

def get_trades():
    with db_connect() as con:
        return pd.read_sql_query("SELECT * FROM trades ORDER BY id DESC", con)

def update_trade_status(trade_id, status, exit_price=None, notes=None):
    with db_connect() as con:
        con.execute("""
            UPDATE trades SET status=?, exit_price=?, notes=? WHERE id=?
        """, (status, exit_price, notes, int(trade_id)))
        con.commit()

init_db()

# ------------------------- DATA -------------------------

ASSETS = {
    "XAUUSD · Gold": "GC=F",
    "EURUSD": "EURUSD=X",
    "BTCUSD": "BTC-USD",
    "Crude Oil": "CL=F",
    "NASDAQ 100": "^NDX",
}
TF_SETTINGS = {
    "1m": ("1m", "5d"),
    "5m": ("5m", "5d"),
    "15m": ("15m", "5d"),
    "30m": ("30m", "1mo"),
    "1h": ("1h", "1mo"),
    "4h": ("1h", "3mo"),
    "1d": ("1d", "1y"),
    "1w": ("1wk", "5y"),
}

@st.cache_data(ttl=60)
def fetch_data(symbol, timeframe):
    interval, period = TF_SETTINGS[timeframe]
    df = yf.download(
        symbol, period=period, interval=interval,
        auto_adjust=False, progress=False, threads=False
    )
    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.loc[:, ~df.columns.duplicated()]
    required = ["Open", "High", "Low", "Close"]
    if not all(c in df.columns for c in required):
        return pd.DataFrame()

    if "Volume" not in df.columns:
        df["Volume"] = 0

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=required)

    if timeframe == "4h":
        df = df.resample("4h").agg({
            "Open": "first", "High": "max", "Low": "min",
            "Close": "last", "Volume": "sum"
        }).dropna(subset=required)

    return df

def add_indicators(df):
    d = df.copy()
    d["MA5"] = d["Close"].rolling(5).mean()
    d["MA13"] = d["Close"].rolling(13).mean()
    d["MA20"] = d["Close"].rolling(20).mean()
    d["MA50"] = d["Close"].rolling(50).mean()

    delta = d["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    d["RSI"] = (100 - 100 / (1 + rs)).fillna(50)

    prev = d["Close"].shift(1)
    tr = pd.concat([
        d["High"] - d["Low"],
        (d["High"] - prev).abs(),
        (d["Low"] - prev).abs()
    ], axis=1).max(axis=1)
    d["ATR"] = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean()

    return d.dropna(subset=["MA5", "MA13", "MA20", "MA50", "ATR"])

def analyze(d):
    last, prev = d.iloc[-1], d.iloc[-2]
    cross_up = prev["MA5"] <= prev["MA13"] and last["MA5"] > last["MA13"]
    cross_down = prev["MA5"] >= prev["MA13"] and last["MA5"] < last["MA13"]

    if cross_up:
        signal = "BUY"
    elif cross_down:
        signal = "SELL"
    elif last["MA5"] > last["MA13"] and last["Close"] > last["MA20"]:
        signal = "BULLISH"
    elif last["MA5"] < last["MA13"] and last["Close"] < last["MA20"]:
        signal = "BEARISH"
    else:
        signal = "NEUTRAL"

    if last["MA5"] > last["MA13"] > last["MA20"] and last["Close"] > last["MA50"]:
        trend = "STRONG UPTREND"
    elif last["MA5"] < last["MA13"] < last["MA20"] and last["Close"] < last["MA50"]:
        trend = "STRONG DOWNTREND"
    elif last["MA5"] > last["MA13"]:
        trend = "BULLISH"
    elif last["MA5"] < last["MA13"]:
        trend = "BEARISH"
    else:
        trend = "SIDEWAYS"

    return last, signal, trend

# ------------------------- CHART -------------------------

def render_chart(d, side="WAIT", t0=None,
                 entry=None, sl=None, tp1=None, tp2=None, tp3=None):
    chart_df = d.tail(300)
    candles, volumes, times = [], [], []

    for ts, row in chart_df.iterrows():
        t = int(pd.Timestamp(ts).timestamp())
        times.append(t)
        candles.append({
            "time": t, "open": float(row["Open"]),
            "high": float(row["High"]), "low": float(row["Low"]),
            "close": float(row["Close"])
        })
        vol = 0 if pd.isna(row["Volume"]) else float(row["Volume"])
        volumes.append({
            "time": t, "value": vol,
            "color": "rgba(53,198,154,.42)" if row["Close"] >= row["Open"]
                     else "rgba(255,107,114,.42)"
        })

    step = int(np.median(np.diff(times))) if len(times) > 1 else 900
    t_end = times[-1] + step * 6
    t_start = times[0]
    marker = None
    if t0 is not None and side != "WAIT":
        t_start = max(int(pd.Timestamp(t0).timestamp()), times[0])
        marker = {
            "time": t_start,
            "position": "belowBar" if side == "LONG" else "aboveBar",
            "color": "#b6f23a",
            "shape": "arrowUp" if side == "LONG" else "arrowDown",
        }

    lines = [
        {
            "title": label, "color": color,
            "data": [{"time": t_start, "value": float(v)},
                     {"time": t_end, "value": float(v)}],
        }
        for v, color, label in [
            (entry, "#b6f23a", "ENTRY"),
            (sl, "#ff6b72", "SL"),
            (tp1, "#35c69a", "TP1"),
            (tp2, "#35c69a", "TP2"),
            (tp3, "#35c69a", "TP3"),
        ] if v is not None and pd.notna(v)
    ]

    payload = {"candles": candles, "volumes": volumes,
               "lines": lines, "marker": marker}
    chart_json = json.dumps(payload, allow_nan=False)
    html = """
    <div id="nexus-chart" style="width:100%;height:420px;"></div>
    <script src="https://unpkg.com/lightweight-charts@5.0.0/dist/lightweight-charts.standalone.production.js"></script>
    <script>
    const d = """ + chart_json + """;
    const el = document.getElementById('nexus-chart');
    if (typeof LightweightCharts === 'undefined') {
      el.innerHTML = '<div style="color:#ffb4b4;padding:20px">Chart library could not load. Check internet connection.</div>';
    } else {
      const chart = LightweightCharts.createChart(el, {
        width: el.clientWidth, height: 420,
        layout: { background: { color: '#0d1722' }, textColor: '#c5d0dc' },
        grid: { vertLines: { color: '#1b2938' }, horzLines: { color: '#1b2938' } },
        crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
        rightPriceScale: { borderColor: '#34465a' },
        timeScale: { borderColor: '#34465a', timeVisible: true, secondsVisible: false },
        handleScroll: true, handleScale: true
      });

      const cs = chart.addSeries(LightweightCharts.CandlestickSeries, {
        upColor: '#35c69a', downColor: '#ff6b72',
        borderUpColor: '#35c69a', borderDownColor: '#ff6b72',
        wickUpColor: '#35c69a', wickDownColor: '#ff6b72'
      });
      cs.setData(d.candles);

      const vs = chart.addSeries(LightweightCharts.HistogramSeries, {
        priceFormat: { type: 'volume' }, priceScaleId: ''
      });
      vs.setData(d.volumes);
      vs.priceScale().applyOptions({ scaleMargins: { top: 0.86, bottom: 0 } });

      d.lines.forEach(l => {
        const s = chart.addSeries(LightweightCharts.LineSeries, {
          color: l.color, lineWidth: 1, lineStyle: 0, title: l.title,
          priceLineVisible: false, lastValueVisible: true,
          crosshairMarkerVisible: false
        });
        s.setData(l.data);
      });

      if (d.marker && LightweightCharts.createSeriesMarkers) {
        LightweightCharts.createSeriesMarkers(cs, [d.marker]);
      }

      chart.timeScale().fitContent();
      new ResizeObserver(entries => {
        if (entries.length) chart.applyOptions({ width: entries[0].contentRect.width });
      }).observe(el);
    }
    </script>
    """
    components.html(html, height=440, scrolling=False)

# ------------------------- HEADER -------------------------

st.markdown("""
<div class="nexus-header">
  <div class="nexus-kicker">MARKET INTELLIGENCE · TECHNICAL ANALYSIS</div>
  <div class="nexus-title">OUTEMGOUNT MA NEXUS</div>
  <div class="nexus-sub"><span class="live-dot"></span>Professional trading workspace · Live market data</div>
</div>
""", unsafe_allow_html=True)

# ------------------------- MARKET SELECTOR (under the title) -------------------------

sel1, sel2, sel3 = st.columns([3, 2, 1.4])
with sel1:
    asset_name = st.selectbox("Market", list(ASSETS.keys()))
with sel2:
    timeframe = st.selectbox("Timeframe", list(TF_SETTINGS.keys()), index=list(TF_SETTINGS.keys()).index("15m"))
with sel3:
    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    if st.button("↻ Refresh", use_container_width=True):
        fetch_data.clear()
        st.rerun()

symbol = ASSETS[asset_name]

with st.spinner("Loading market data..."):
    raw = fetch_data(symbol, timeframe)

if raw.empty or len(raw) < 60:
    st.error("Market data is unavailable or insufficient for this selection. Try another timeframe.")
    st.stop()

data = add_indicators(raw)
if len(data) < 2:
    st.error("Not enough valid candles to calculate indicators.")
    st.stop()

last, signal, trend = analyze(data)
price = float(last["Close"])
atr = float(last["ATR"])
rsi = float(last["RSI"])
prev_close = float(data["Close"].iloc[-2])
chg_pct = (price / prev_close - 1) * 100 if prev_close else 0.0

# Trade levels (ATR multiples, measured from the MA5/MA13 crossover candle)
SL_MULT = 1.2
TP_MULTS = (1.0, 1.8, 2.6)


def find_signal_bar(d, direction):
    """Latest candle (last 300) where MA5 crossed MA13 in the given direction."""
    w = d.tail(300)
    if direction == "LONG":
        m = (w["MA5"] > w["MA13"]) & (w["MA5"].shift(1) <= w["MA13"].shift(1))
    else:
        m = (w["MA5"] < w["MA13"]) & (w["MA5"].shift(1) >= w["MA13"].shift(1))
    hits = w.index[m.fillna(False)]
    return hits[-1] if len(hits) else w.index[-1]


def evaluate_trade(d, pos, direction, stop, targets):
    """Scan candles after the signal candle.
    Returns "SL" if the stop is hit before TP1, else the highest TP reached (0-3)."""
    hit = 0
    for i in range(pos + 1, len(d)):
        hi, lo = float(d["High"].iloc[i]), float(d["Low"].iloc[i])
        if direction == "LONG":
            if hit == 0 and lo <= stop:
                return "SL"
            for k, tp in enumerate(targets, 1):
                if k > hit and hi >= tp:
                    hit = k
        else:
            if hit == 0 and hi >= stop:
                return "SL"
            for k, tp in enumerate(targets, 1):
                if k > hit and lo <= tp:
                    hit = k
    return hit


if signal in ("BUY", "BULLISH"):
    side = "LONG"
elif signal in ("SELL", "BEARISH"):
    side = "SHORT"
else:
    side = "WAIT"

signal_time, candles_ago = None, None
stop_distance = atr * SL_MULT
if side != "WAIT":
    signal_time = find_signal_bar(data, side)
    pos = data.index.get_loc(signal_time)
    sig_close = float(data["Close"].iloc[pos])
    sig_atr = float(data["ATR"].iloc[pos])
    candles_ago = len(data) - 1 - pos
    dirn = 1 if side == "LONG" else -1
    entry = sig_close
    stop_distance = sig_atr * SL_MULT
    sl = entry - dirn * stop_distance
    tp1, tp2, tp3 = [entry + dirn * sig_atr * m for m in TP_MULTS]
    outcome = evaluate_trade(data, pos, side, sl, (tp1, tp2, tp3))
else:
    entry = sl = tp1 = tp2 = tp3 = None
    outcome = None

st.markdown("---")

# ------------------------- MARKET SNAPSHOT -------------------------

m1, m2, m3 = st.columns(3)
m1.metric("LAST PRICE", f"{price:,.4f}", delta=f"{chg_pct:+.2f}% last candle")
m2.metric("MARKET TREND", trend)
m3.metric("CURRENT BIAS", signal)

st.markdown("---")

# ------------------------- FEATURED TRADE SETUP -------------------------

st.markdown("<h2 style='white-space:nowrap;font-size:clamp(18px,5vw,26px);margin:0 0 6px 0'>🎯 Trade Plan</h2>", unsafe_allow_html=True)

signal_class = "buy" if signal in ("BUY", "BULLISH") else (
    "sell" if signal in ("SELL", "BEARISH") else "wait"
)
st.markdown(
    f"<div class='trade-card'>"
    f"<div class='trade-label'>LIVE TECHNICAL SETUP · {asset_name} · {timeframe}</div>"
    f"<div class='{signal_class}' style='font-size:30px;margin:5px 0'>{signal}</div>"
    f"<div class='small-note'>Market direction: {trend} &nbsp; · &nbsp; "
    f"Setup status: {'Directional setup' if side != 'WAIT' else 'Waiting for confirmation'}</div>"
    f"</div>",
    unsafe_allow_html=True
)

if side != "WAIT":
    trade_cols = st.columns(5)
    trade_values = [
        ("DIRECTION", side, "#f0c75e"),
        ("ENTRY PRICE", fmt(entry), "#b6f23a"),
        ("STOP LOSS", fmt(sl), "#ff747c"),
        ("TAKE PROFIT 1", fmt(tp1), "#46d6a5"),
        ("TAKE PROFIT 2", fmt(tp2), "#46d6a5"),
    ]
    for col, (label, value, color) in zip(trade_cols, trade_values):
        with col:
            st.markdown(
                f"<div class='trade-card' style='min-height:100px'>"
                f"<div class='trade-label'>{label}</div>"
                f"<div class='trade-number' style='color:{color};font-size:20px'>{value}</div>"
                f"</div>",
                unsafe_allow_html=True
            )
    st.markdown(
        f"<div class='trade-card' style='border-color:#315b50'>"
        f"<span class='trade-label'>TAKE PROFIT 3</span><br>"
        f"<span style='color:#46d6a5;font-size:21px;font-weight:800'>{fmt(tp3)}</span>"
        f"<span class='small-note'> &nbsp; · &nbsp; Stop distance: {stop_distance:.4f} &nbsp; · &nbsp; R/R 1:{TP_MULTS[0]/SL_MULT:.1f} / 1:{TP_MULTS[1]/SL_MULT:.1f} / 1:{TP_MULTS[2]/SL_MULT:.1f} &nbsp; · &nbsp; Signal candle: {candles_ago} candle(s) ago</span>"
        f"</div>",
        unsafe_allow_html=True
    )
    ALERTS = {
        1: ("TP1 HIT ✓", "#46d6a5", "احجز الأرباح وأمّن الصفقة",
            "مهم جداً: حجز جزء من الأرباح عند TP1 وتأمين الصفقة بتحريك وقف الخسارة إلى نقطة الدخول، وفق خطة التداول.",
            "إعادة الدخول المحتملة",
            "إذا رجع السعر إلى منطقة ENTRY، راجع شروط الدخول من جديد قبل اتخاذ القرار."),
        2: ("TP2 HIT ✓", "#46d6a5", "الهدف الثاني تحقق",
            "واصل تأمين الصفقة: أبقِ وقف الخسارة عند الدخول أو أعلى، واحجز جزءاً آخر من الأرباح حسب خطتك.",
            "إعادة الدخول المحتملة",
            "إذا رجع السعر إلى منطقة ENTRY، راجع شروط الدخول من جديد قبل اتخاذ القرار."),
        3: ("TP3 HIT ✓", "#46d6a5", "الهدف الأخير تحقق",
            "أغلق ما تبقّى من الصفقة وسجّل النتيجة في دفتر الصفقات.",
            "إعادة الدخول المحتملة",
            "انتظر إشارة جديدة وراجع شروط الدخول قبل اتخاذ أي قرار."),
        "SL": ("SL HIT ✗", "#ff747c", "تم ضرب وقف الخسارة",
               "الصفقة انتهت بخسارة محسوبة وفق الخطة. لا تعوّضها بصفقة عاطفية.",
               "إعادة الدخول المحتملة",
               "انتظر إشارة جديدة وراجع شروط الدخول من جديد قبل اتخاذ القرار."),
    }
    if outcome in ALERTS:
        a_title, a_color, a_head, a_body, a_sub, a_sub_body = ALERTS[outcome]
        st.markdown(
            f"<div class='trade-card' style='border-color:{a_color};direction:rtl;text-align:right'>"
            f"<div style='color:{a_color};font-size:24px;font-weight:800;direction:ltr;text-align:left'>{a_title}</div>"
            f"<div style='color:#f0c75e;font-size:17px;font-weight:800;margin:8px 0 4px'>{a_head}</div>"
            f"<div style='color:#d5dfea;font-size:14px;line-height:1.7'>{a_body}</div>"
            f"<div style='color:#f0c75e;font-size:15px;font-weight:800;margin:12px 0 4px'>{a_sub}</div>"
            f"<div style='color:#d5dfea;font-size:14px;line-height:1.7'>{a_sub_body}</div>"
            f"<div class='small-note' style='margin-top:10px'>التنبيه يظهر حسب تحقق شروط السعر، وليس بمجرد ظهور الإشارة.</div>"
            f"</div>",
            unsafe_allow_html=True
        )
    else:
        st.caption("لم يتحقق أي هدف بعد · التنبيه يظهر حسب تحقق شروط السعر، وليس بمجرد ظهور الإشارة.")
    if st.button("＋ Save setup to trade journal", type="primary", use_container_width=True):
        add_trade(asset_name, timeframe, side, entry, sl, tp1, tp2, tp3)
        st.success("Setup saved to your journal as Planned.")
        st.rerun()
else:
    st.info("No directional setup right now. Wait for confirmation before considering a trade.")

st.markdown("---")

# ------------------------- TABS -------------------------

tab_overview, tab_journal, tab_history = st.tabs([
    "📊 Market chart", "📂 Trade journal", "📈 Performance"
])

with tab_overview:
    st.subheader("Professional market chart")
    render_chart(data, side=side, t0=signal_time,
                 entry=entry, sl=sl, tp1=tp1, tp2=tp2, tp3=tp3)

    st.markdown("---")
    st.markdown("#### Oscillators & volatility")
    low1, low2, low3 = st.columns(3)
    low1.metric("RSI · 14", f"{rsi:.1f}")
    low2.metric("ATR · 14", f"{atr:.4f}")
    low3.metric(f"STOP DISTANCE · {SL_MULT}x ATR", f"{stop_distance:.4f}")
    st.progress(
        min(max(rsi / 100, 0.0), 1.0),
        text=f"RSI 14 · {rsi:.1f} · " +
             ("Overbought zone" if rsi >= 70 else "Oversold zone" if rsi <= 30 else "Neutral zone")
    )
    st.caption("ATR-based levels are analytical estimates, not guaranteed execution prices.")
    st.caption(
        f"Last candle: {data.index[-1]} · "
        f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

with tab_journal:
    st.subheader("Trade journal")
    trades = get_trades()

    if trades.empty:
        st.info("Your journal is empty. Save a setup from the Trade Plan above.")
    else:
        open_trades = trades[trades["status"].isin(["Planned", "Open"])]
        st.markdown(f"**Active / planned setups: {len(open_trades)}**")

        for _, trade in open_trades.iterrows():
            trade_id = int(trade["id"])
            side_color = "#35c69a" if trade["side"] == "LONG" else "#ff6b72"
            st.markdown(
                f"<div class='trade-card'>"
                f"<div class='trade-label'>#{trade_id} · {trade['asset']} · {trade['timeframe']}</div>"
                f"<div style='font-size:20px;font-weight:800;color:{side_color}'>"
                f"{trade['side']} · {trade['status']}</div>"
                f"<div class='small-note'>Entry: {fmt(trade['entry'])} "
                f" | SL: {fmt(trade['stop_loss'])} "
                f" | TP1: {fmt(trade['tp1'])} "
                f" | TP2: {fmt(trade['tp2'])} "
                f" | TP3: {fmt(trade['tp3'])}</div>"
                f"</div>",
                unsafe_allow_html=True
            )

        if not open_trades.empty:
            selected_id = st.selectbox(
                "Select setup to update",
                open_trades["id"].astype(int).tolist(),
                format_func=lambda x: f"Trade #{x} · {open_trades.loc[open_trades['id']==x, 'asset'].iloc[0]}"
            )
            chosen = open_trades[open_trades["id"] == selected_id].iloc[0]
            statuses = ["Planned", "Open", "Closed", "Cancelled"]
            status_choice = st.selectbox(
                "Update status", statuses,
                index=statuses.index(chosen["status"])
            )
            exit_value = None
            if status_choice == "Closed":
                exit_value = st.number_input(
                    "Exit price",
                    value=float(chosen["entry"]),
                    format="%.5f"
                )
            note = st.text_input(
                "Trade note (optional)",
                value=chosen["notes"] if pd.notna(chosen["notes"]) else ""
            )

            if st.button("Save journal update", type="primary"):
                update_trade_status(selected_id, status_choice, exit_value, note)
                st.success("Journal updated.")
                st.rerun()

with tab_history:
    st.subheader("Trade history & statistics")
    trades = get_trades()

    if trades.empty:
        st.info("No journal records yet.")
    else:
        total = len(trades)
        closed = trades[trades["status"] == "Closed"].copy()
        open_count = int(trades["status"].isin(["Planned", "Open"]).sum())

        wins = 0
        losses = 0
        for _, trd in closed.iterrows():
            if pd.notna(trd["exit_price"]):
                if trd["side"] == "LONG":
                    won = trd["exit_price"] > trd["entry"]
                else:
                    won = trd["exit_price"] < trd["entry"]
                if won:
                    wins += 1
                else:
                    losses += 1

        win_rate = (wins / (wins + losses) * 100) if (wins + losses) else 0

        p1, p2, p3, p4 = st.columns(4)
        p1.metric("TOTAL RECORDS", total)
        p2.metric("PLANNED / OPEN", open_count)
        p3.metric("CLOSED WINS", wins)
        p4.metric("WIN RATE", f"{win_rate:.1f}%")

        display = trades.copy()
        display["created_at"] = pd.to_datetime(display["created_at"], errors="coerce")
        st.dataframe(
            display[[
                "id", "created_at", "asset", "timeframe", "side",
                "entry", "stop_loss", "tp1", "tp2", "tp3",
                "status", "exit_price", "notes"
            ]],
            use_container_width=True,
            hide_index=True
        )

        csv = trades.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇ Download trade history CSV",
            data=csv,
            file_name="outemgount_nexus_trade_history.csv",
            mime="text/csv"
        )

# ------------------------- FOOTER -------------------------

st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#8294a8;font-size:12px'>"
    "OUTEMGOUNT MA NEXUS · PROFESSIONAL EDITION<br>"
    "Data source: Yahoo Finance · Technical analysis only · Not a broker · No trade execution"
    "</div>",
    unsafe_allow_html=True
)
