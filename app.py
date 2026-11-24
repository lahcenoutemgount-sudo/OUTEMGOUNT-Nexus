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
    initial_sidebar_state="expanded",
)

DB_PATH = "nexus_trades.db"

# ------------------------- DESIGN -------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: Inter, sans-serif; }
.stApp { background: #080e16; color: #e7edf5; }
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1600px; }
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
    padding: 22px 24px; margin-bottom: 15px;
}
.nexus-kicker { color:#9bacc0; font-size:12px; letter-spacing:2px; font-weight:700; }
.nexus-title { color:#f0c75e; font-size:30px; font-weight:800; margin:3px 0; }
.nexus-sub { color:#aebdce; font-size:13px; }
.trade-card {
    background: linear-gradient(145deg,#142131,#101923);
    border:1px solid #2a3d52; border-radius:15px; padding:16px;
    margin: 5px 0 12px 0;
}
.trade-label { color:#91a5ba; font-size:11px; letter-spacing:1.2px; font-weight:700; }
.trade-number { color:#65c7f2; font-size:21px; font-weight:800; margin-top:3px; }
.trade-card .trade-label { color:#a9bfd3; }
.trade-card { box-shadow: 0 8px 24px rgba(0,0,0,.18); }
.trade-card .buy { color:#46d6a5 !important; }
.trade-card .sell { color:#ff747c !important; }

.buy { color:#35c69a !important; font-weight:800; }
.sell { color:#ff6b72 !important; font-weight:800; }
.wait { color:#f0c75e !important; font-weight:800; }
.small-note { color:#91a2b5; font-size:12px; }
hr { border-color:#26384b !important; }
.stButton>button { border-radius:10px; font-weight:700; }
</style>
""", unsafe_allow_html=True)

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
    "15m": ("15m", "5d"),
    "1h": ("1h", "1mo"),
    "4h": ("1h", "3mo"),
    "1d": ("1d", "1y"),
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

    d["Kijun"] = (
        d["High"].rolling(26).max() + d["Low"].rolling(26).min()
    ) / 2

    return d.dropna(subset=["MA5", "MA13", "MA20", "MA50", "ATR", "Kijun"])

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

def render_chart(d, show_kijun=True,
                 entry=None, sl=None, tp1=None, tp2=None, tp3=None):
    chart_df = d.tail(300)
    candles, volumes = [], []
    lines = {"kijun": []}

    for ts, row in chart_df.iterrows():
        t = int(pd.Timestamp(ts).timestamp())
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
        if pd.notna(row["Kijun"]):
            lines["kijun"].append({"time": t, "value": float(row["Kijun"])})

    payload = {
        "candles": candles, "volumes": volumes, **lines,
        "visible": {"kijun": show_kijun},
        "levels": [
            {"price": float(v), "color": c, "title": label, "style": style}
            for v, c, label, style in [
                (entry, "#f0c75e", "ENTRY", 0),
                (sl, "#ff6b72", "STOP LOSS", 2),
                (tp1, "#35c69a", "TP1", 2),
                (tp2, "#35c69a", "TP2", 2),
                (tp3, "#35c69a", "TP3", 2),
            ] if v is not None and pd.notna(v)
        ]
    }

    chart_json = json.dumps(payload, allow_nan=False)
    html = """
    <div id="nexus-chart" style="width:100%;height:590px;"></div>
    <script src="https://unpkg.com/lightweight-charts/dist/lightweight-charts.standalone.production.js"></script>
    <script>
    const d = """ + chart_json + """;
    const el = document.getElementById('nexus-chart');
    if (typeof LightweightCharts === 'undefined') {
      el.innerHTML = '<div style="color:#ffb4b4;padding:20px">Chart library could not load. Check internet connection.</div>';
    } else {
      const chart = LightweightCharts.createChart(el, {
        width: el.clientWidth, height: 590,
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

      function addLine(key, color, title) {
        const s = chart.addSeries(LightweightCharts.LineSeries, {
          color, lineWidth: 2, title, visible: d.visible[key],
          priceLineVisible: false, lastValueVisible: true
        });
        s.setData(d[key]);
      }
      addLine('kijun', '#ff9966', 'Kijun Sen');

      const vs = chart.addSeries(LightweightCharts.HistogramSeries, {
        priceFormat: { type: 'volume' }, priceScaleId: ''
      });
      vs.setData(d.volumes);
      vs.priceScale().applyOptions({ scaleMargins: { top: 0.84, bottom: 0 } });

      d.levels.forEach(x => cs.createPriceLine({
        price: x.price, color: x.color, lineWidth: 2,
        lineStyle: x.style, axisLabelVisible: true, title: x.title
      }));

      chart.timeScale().fitContent();
      new ResizeObserver(entries => {
        if (entries.length) chart.applyOptions({ width: entries[0].contentRect.width });
      }).observe(el);
    }
    </script>
    """
    components.html(html, height=610, scrolling=False)

# ------------------------- HEADER -------------------------

st.markdown("""
<div class="nexus-header">
  <div class="nexus-kicker">MARKET INTELLIGENCE · TECHNICAL ANALYSIS</div>
  <div class="nexus-title">OUTEMGOUNT MA NEXUS</div>
  <div class="nexus-sub">Professional trading workspace</div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## ⚙️ Workspace")
    asset_name = st.selectbox("Market", list(ASSETS.keys()))
    timeframe = st.selectbox("Timeframe", list(TF_SETTINGS.keys()), index=0)
    if st.button("↻ Refresh market data", use_container_width=True):
        fetch_data.clear()
        st.rerun()
    st.markdown("---")
    st.caption("Data source: Yahoo Finance")
    st.caption("No broker execution or live order connection.")

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
stop_distance = atr * 1.5

if signal in ("BUY", "BULLISH"):
    side = "LONG"
    entry = price
    sl = price - stop_distance
    tp1, tp2, tp3 = price + stop_distance*1.5, price + stop_distance*2, price + stop_distance*3
elif signal in ("SELL", "BEARISH"):
    side = "SHORT"
    entry = price
    sl = price + stop_distance
    tp1, tp2, tp3 = price - stop_distance*1.5, price - stop_distance*2, price - stop_distance*3
else:
    side = "WAIT"
    entry = sl = tp1 = tp2 = tp3 = None

# ------------------------- MARKET SNAPSHOT -------------------------

m1, m2, m3 = st.columns(3)
m1.metric("LAST PRICE", f"{price:,.4f}")
m2.metric("MARKET TREND", trend)
m3.metric("CURRENT BIAS", signal)

st.markdown("---")

# ------------------------- FEATURED TRADE SETUP -------------------------

st.markdown("## 🎯 Featured trade setup")

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
        ("ENTRY PRICE", f"{entry:,.4f}", "#65c7f2"),
        ("STOP LOSS", f"{sl:,.4f}", "#ff747c"),
        ("TAKE PROFIT 1", f"{tp1:,.4f}", "#46d6a5"),
        ("TAKE PROFIT 2", f"{tp2:,.4f}", "#46d6a5"),
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
        f"<span style='color:#46d6a5;font-size:21px;font-weight:800'>{tp3:,.4f}</span>"
        f"<span class='small-note'> &nbsp; · &nbsp; ATR stop distance: {stop_distance:.4f}</span>"
        f"</div>",
        unsafe_allow_html=True
    )
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
    vk = st.checkbox("Show Kijun Sen", True, key="vk")
    render_chart(
        data, vk,
        entry=entry, sl=sl, tp1=tp1, tp2=tp2, tp3=tp3
    )

    st.markdown("#### Ichimoku reference")
    st.metric("Kijun Sen", f"{last['Kijun']:.4f}")

    # RSI and ATR intentionally placed BELOW the trade setup and chart.
    st.markdown("---")
    st.markdown("#### Oscillators & volatility")
    low1, low2, low3 = st.columns(3)
    low1.metric("RSI · 14", f"{rsi:.1f}")
    low2.metric("ATR · 14", f"{atr:.4f}")
    low3.metric("ATR STOP DISTANCE · 1.5x", f"{stop_distance:.4f}")
    st.progress(
        min(max(rsi / 100, 0.0), 1.0),
        text=f"RSI 14 · {rsi:.1f} · " +
             ("Overbought zone" if rsi >= 70 else "Oversold zone" if rsi <= 30 else "Neutral zone")
    )
    st.caption(
        "ATR-based levels are analytical estimates, not guaranteed execution prices."
    )
    st.caption(
        f"Last candle: {data.index[-1]} · "
        f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

with tab_journal:
    st.subheader("Trade journal")
    trades = get_trades()

    if trades.empty:
        st.info("Your journal is empty. Save a setup from the Signal setup tab.")
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
                f"<div class='small-note'>Entry: {trade['entry']:.4f} "
                f" | SL: {trade['stop_loss']:.4f} "
                f" | TP1: {trade['tp1']:.4f} "
                f" | TP2: {trade['tp2']:.4f} "
                f" | TP3: {trade['tp3']:.4f}</div>"
                f"</div>",
                unsafe_allow_html=True
            )

        selected_id = st.selectbox(
            "Select setup to update",
            open_trades["id"].astype(int).tolist(),
            format_func=lambda x: f"Trade #{x} · {open_trades.loc[open_trades['id']==x, 'asset'].iloc[0]}"
        )
        chosen = open_trades[open_trades["id"] == selected_id].iloc[0]
        status_choice = st.selectbox(
            "Update status",
            ["Planned", "Open", "Closed", "Cancelled"],
            index=["Planned", "Open", "Closed", "Cancelled"].index(chosen["status"])
        )
        exit_value = None
        if status_choice == "Closed":
            exit_value = st.number_input(
                "Exit price",
                value=float(chosen["entry"]),
                format="%.5f"
            )
        note = st.text_input("Trade note (optional)", value=chosen["notes"] or "")

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
        if not closed.empty:
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
    "Technical analysis only · Not a broker · No trade execution"
    "</div>",
    unsafe_allow_html=True
)
