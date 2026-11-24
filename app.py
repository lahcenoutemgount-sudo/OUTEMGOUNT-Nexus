
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
import pandas as pd
import numpy as np
import json

# =========================================================
# 1. PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="OUTEMGOUNT MA NEXUS",
    page_icon="📈",
    layout="wide"
)

# =========================================================
# 2. DESIGN
# =========================================================

st.markdown("""
<style>
.stApp {
    background-color: #0b1118;
    color: #e6edf3;
}
.block-container {
    padding-top: 1.5rem;
}
h1, h2, h3 {
    color: #f5c542 !important;
}
div[data-testid="stMetric"] {
    background: #151f2b;
    border: 1px solid #293849;
    padding: 12px;
    border-radius: 10px;
}
.signal-buy {
    color: #26a69a;
    font-size: 25px;
    font-weight: bold;
}
.signal-sell {
    color: #ef5350;
    font-size: 25px;
    font-weight: bold;
}
.signal-neutral {
    color: #f5c542;
    font-size: 25px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# 3. HEADER
# =========================================================

st.markdown(
    "<h1 style='text-align:center;'>OUTEMGOUNT MA NEXUS</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;color:#9baec1;'>"
    "PROFESSIONAL TRADING DASHBOARD</p>",
    unsafe_allow_html=True
)

st.divider()

# =========================================================
# 4. MARKET SELECTION
# =========================================================

ASSETS = {
    "XAUUSD - Gold": "GC=F",
    "EURUSD": "EURUSD=X",
    "BTCUSD": "BTC-USD",
    "Crude Oil": "CL=F",
    "NASDAQ 100": "^NDX"
}

col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    selected_asset = st.selectbox("Select Market", list(ASSETS.keys()))

with col2:
    timeframe = st.selectbox(
        "Timeframe",
        ["15m", "1h", "4h", "1d"]
    )

with col3:
    refresh = st.button("🔄 Refresh Data", use_container_width=True)

ticker = ASSETS[selected_asset]

# =========================================================
# 5. DOWNLOAD DATA
# =========================================================

@st.cache_data(ttl=60)
def download_data(symbol, tf):

    settings = {
        "15m": ("15m", "5d"),
        "1h": ("1h", "1mo"),
        "4h": ("1h", "3mo"),
        "1d": ("1d", "1y")
    }

    interval, period = settings[tf]

    df = yf.download(
        symbol,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False
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

    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df = df.dropna(subset=required)

    if tf == "4h":
        df = df.resample("4h").agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum"
        }).dropna(subset=required)

    return df


if refresh:
    download_data.clear()

with st.spinner("Loading market data..."):
    data = download_data(ticker, timeframe)

if data.empty or len(data) < 55:
    st.error("Market data unavailable. Try another timeframe or refresh.")
    st.stop()

# =========================================================
# 6. INDICATORS
# =========================================================

data = data.copy()

data["MA5"] = data["Close"].rolling(5).mean()
data["MA13"] = data["Close"].rolling(13).mean()
data["MA20"] = data["Close"].rolling(20).mean()
data["MA50"] = data["Close"].rolling(50).mean()

# RSI 14
delta = data["Close"].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.ewm(
    alpha=1/14, min_periods=14, adjust=False
).mean()

avg_loss = loss.ewm(
    alpha=1/14, min_periods=14, adjust=False
).mean()

rs = avg_gain / avg_loss.replace(0, np.nan)
data["RSI"] = (100 - (100 / (1 + rs))).fillna(50)

# ATR 14
previous_close = data["Close"].shift(1)

tr = pd.concat([
    data["High"] - data["Low"],
    (data["High"] - previous_close).abs(),
    (data["Low"] - previous_close).abs()
], axis=1).max(axis=1)

data["ATR"] = tr.ewm(
    alpha=1/14, min_periods=14, adjust=False
).mean()

# Kijun 26
data["Kijun"] = (
    data["High"].rolling(26).max()
    + data["Low"].rolling(26).min()
) / 2

data = data.dropna(
    subset=["MA5", "MA13", "MA20", "MA50", "ATR", "Kijun"]
)

if data.empty:
    st.error("Not enough data for indicators.")
    st.stop()

# =========================================================
# 7. MARKET ANALYSIS
# =========================================================

last = data.iloc[-1]
previous = data.iloc[-2]

price = float(last["Close"])
atr = float(last["ATR"])
rsi = float(last["RSI"])

bullish_cross = (
    previous["MA5"] <= previous["MA13"]
    and last["MA5"] > last["MA13"]
)

bearish_cross = (
    previous["MA5"] >= previous["MA13"]
    and last["MA5"] < last["MA13"]
)

if bullish_cross:
    signal = "BUY"
elif bearish_cross:
    signal = "SELL"
elif last["MA5"] > last["MA13"] and price > last["MA20"]:
    signal = "BULLISH"
elif last["MA5"] < last["MA13"] and price < last["MA20"]:
    signal = "BEARISH"
else:
    signal = "NEUTRAL"

if (
    last["MA5"] > last["MA13"]
    and last["MA13"] > last["MA20"]
    and price > last["MA50"]
):
    trend = "STRONG UPTREND"
elif (
    last["MA5"] < last["MA13"]
    and last["MA13"] < last["MA20"]
    and price < last["MA50"]
):
    trend = "STRONG DOWNTREND"
elif last["MA5"] > last["MA13"]:
    trend = "BULLISH"
elif last["MA5"] < last["MA13"]:
    trend = "BEARISH"
else:
    trend = "SIDEWAYS"

# =========================================================
# 8. RISK MANAGEMENT
# =========================================================

atr_multiplier = 1.5
risk_distance = atr * atr_multiplier

if signal in ["BUY", "BULLISH"]:
    direction = "LONG"
    entry = price
    stop_loss = entry - risk_distance
    tp1 = entry + risk_distance * 1.5
    tp2 = entry + risk_distance * 2
    tp3 = entry + risk_distance * 3

elif signal in ["SELL", "BEARISH"]:
    direction = "SHORT"
    entry = price
    stop_loss = entry + risk_distance
    tp1 = entry - risk_distance * 1.5
    tp2 = entry - risk_distance * 2
    tp3 = entry - risk_distance * 3

else:
    direction = "WAIT"
    entry = price
    stop_loss = None
    tp1 = None
    tp2 = None
    tp3 = None

# =========================================================
# 9. MARKET OVERVIEW
# =========================================================

st.subheader("📊 MARKET OVERVIEW")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Market Price", f"{price:,.4f}")
c2.metric("Trend", trend)
c3.metric("RSI 14", f"{rsi:.2f}")
c4.metric("ATR 14", f"{atr:.4f}")

st.divider()

# =========================================================
# 10. SIGNAL
# =========================================================

st.subheader("🎯 TRADING SIGNAL")

s1, s2, s3 = st.columns(3)

with s1:
    if signal in ["BUY", "BULLISH"]:
        st.markdown(
            f"<div class='signal-buy'>{signal}</div>",
            unsafe_allow_html=True
        )
    elif signal in ["SELL", "BEARISH"]:
        st.markdown(
            f"<div class='signal-sell'>{signal}</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div class='signal-neutral'>NEUTRAL</div>",
            unsafe_allow_html=True
        )

with s2:
    st.metric("Direction", direction)

with s3:
    st.metric("Kijun", f"{last['Kijun']:,.4f}")

st.divider()

# =========================================================
# 11. RISK MANAGER
# =========================================================

st.subheader("🛡️ PHOENIX RISK MANAGER")

r1, r2, r3 = st.columns(3)

r1.metric("ATR 14", f"{atr:.4f}")
r2.metric("Stop Distance", f"{risk_distance:.4f}")
r3.metric("ATR Multiplier", f"{atr_multiplier:.1f}x")

if direction != "WAIT":

    a, b, c = st.columns(3)

    a.metric("ENTRY", f"{entry:.4f}")
    b.metric("STOP LOSS", f"{stop_loss:.4f}")
    c.metric("TAKE PROFIT 1", f"{tp1:.4f}")

    a, b = st.columns(2)

    a.metric("TAKE PROFIT 2", f"{tp2:.4f}")
    b.metric("TAKE PROFIT 3", f"{tp3:.4f}")

else:
    st.info("No directional setup. Wait for confirmation.")

st.caption(
    "ATR-based levels are analytical estimates, not guaranteed execution prices."
)

st.divider()

# =========================================================
# 12. BROKER STYLE CHART
# =========================================================

st.subheader("📈 PROFESSIONAL BROKER CHART")

v1, v2, v3, v4, v5 = st.columns(5)

with v1:
    show_ma5 = st.checkbox("MA 5", True)
with v2:
    show_ma13 = st.checkbox("MA 13", True)
with v3:
    show_ma20 = st.checkbox("MA 20", True)
with v4:
    show_ma50 = st.checkbox("MA 50", True)
with v5:
    show_kijun = st.checkbox("Kijun", True)

chart_data = data.tail(300).copy()

candles = []
volumes = []
ma5 = []
ma13 = []
ma20 = []
ma50 = []
kijun = []

for timestamp, row in chart_data.iterrows():

    t = int(timestamp.timestamp())

    candles.append({
        "time": t,
        "open": float(row["Open"]),
        "high": float(row["High"]),
        "low": float(row["Low"]),
        "close": float(row["Close"])
    })

    volume_value = row["Volume"]
    if pd.isna(volume_value):
        volume_value = 0

    volumes.append({
        "time": t,
        "value": float(volume_value),
        "color": (
            "rgba(38,166,154,0.45)"
            if row["Close"] >= row["Open"]
            else "rgba(239,83,80,0.45)"
        )
    })

    for column, target in [
        ("MA5", ma5),
        ("MA13", ma13),
        ("MA20", ma20),
        ("MA50", ma50),
        ("Kijun", kijun)
    ]:
        value = row[column]
        if not pd.isna(value):
            target.append({
                "time": t,
                "value": float(value)
            })

chart_payload = {
    "candles": candles,
    "volumes": volumes,
    "ma5": ma5,
    "ma13": ma13,
    "ma20": ma20,
    "ma50": ma50,
    "kijun": kijun,
    "show_ma5": show_ma5,
    "show_ma13": show_ma13,
    "show_ma20": show_ma20,
    "show_ma50": show_ma50,
    "show_kijun": show_kijun,
    "entry": entry if direction != "WAIT" else None,
    "sl": stop_loss,
    "tp1": tp1,
    "tp2": tp2,
    "tp3": tp3
}

chart_json = json.dumps(chart_payload, allow_nan=False)

html = """
<div id="broker-chart" style="width:100%;height:620px;"></div>

<script src="https://unpkg.com/lightweight-charts/dist/lightweight-charts.standalone.production.js"></script>

<script>
const data = """ + chart_json + """;

const container = document.getElementById('broker-chart');

const chart = LightweightCharts.createChart(container, {
    width: container.clientWidth,
    height: 620,

    layout: {
        background: { color: '#101820' },
        textColor: '#d1d4dc'
    },

    grid: {
        vertLines: { color: '#202b38' },
        horzLines: { color: '#202b38' }
    },

    crosshair: {
        mode: LightweightCharts.CrosshairMode.Normal
    },

    rightPriceScale: {
        borderColor: '#485465'
    },

    timeScale: {
        borderColor: '#485465',
        timeVisible: true,
        secondsVisible: false
    },

    handleScroll: true,
    handleScale: true
});

// CANDLESTICKS
const candleSeries = chart.addSeries(
    LightweightCharts.CandlestickSeries,
    {
        upColor: '#26a69a',
        downColor: '#ef5350',
        borderUpColor: '#26a69a',
        borderDownColor: '#ef5350',
        wickUpColor: '#26a69a',
        wickDownColor: '#ef5350'
    }
);

candleSeries.setData(data.candles);

// MOVING AVERAGES
function addMA(values, color, title, visible) {

    const line = chart.addSeries(
        LightweightCharts.LineSeries,
        {
            color: color,
            lineWidth: 2,
            title: title,
            priceLineVisible: false,
            lastValueVisible: true,
            visible: visible
        }
    );

    line.setData(values);
}

addMA(data.ma5, '#f5c542', 'MA 5', data.show_ma5);
addMA(data.ma13, '#42a5f5', 'MA 13', data.show_ma13);
addMA(data.ma20, '#ab47bc', 'MA 20', data.show_ma20);
addMA(data.ma50, '#eeeeee', 'MA 50', data.show_ma50);
addMA(data.kijun, '#ff8a65', 'Kijun', data.show_kijun);

// VOLUME
const volumeSeries = chart.addSeries(
    LightweightCharts.HistogramSeries,
    {
        priceFormat: { type: 'volume' },
        priceScaleId: ''
    }
);

volumeSeries.setData(data.volumes);

volumeSeries.priceScale().applyOptions({
    scaleMargins: {
        top: 0.82,
        bottom: 0
    }
});

// ENTRY / SL / TP LINES
function addLevel(value, color, title, style) {

    if (value !== null && value !== undefined) {

        candleSeries.createPriceLine({
            price: value,
            color: color,
            lineWidth: 2,
            lineStyle: style,
            axisLabelVisible: true,
            title: title
        });
    }
}

addLevel(data.entry, '#f5c542', 'ENTRY', 0);
addLevel(data.sl, '#ef5350', 'STOP LOSS', 2);
addLevel(data.tp1, '#26a69a', 'TP 1', 2);
addLevel(data.tp2, '#26a69a', 'TP 2', 2);
addLevel(data.tp3, '#26a69a', 'TP 3', 2);

// FIT CHART
chart.timeScale().fitContent();

// RESPONSIVE WIDTH
const resizeObserver = new ResizeObserver(entries => {
    if (entries.length > 0) {
        chart.applyOptions({
            width: entries[0].contentRect.width
        });
    }
});

resizeObserver.observe(container);
</script>
"""

components.html(
    html,
    height=640,
    scrolling=False
)

st.caption(
    "Chart powered by TradingView Lightweight Charts. "
    "Market data supplied by Yahoo Finance."
)

st.divider()

# =========================================================
# 13. RSI PANEL
# =========================================================

st.subheader("📉 RSI ANALYSIS")

rsi_status = (
    "OVERBOUGHT" if rsi >= 70
    else "OVERSOLD" if rsi <= 30
    else "NEUTRAL"
)

rc1, rc2 = st.columns(2)

rc1.metric("RSI 14", f"{rsi:.2f}")
rc2.metric("RSI Condition", rsi_status)

st.progress(
    min(max(rsi / 100, 0.0), 1.0),
    text=f"RSI: {rsi:.2f}"
)

st.caption(
    "RSI above 70 may indicate overbought conditions; "
    "below 30 may indicate oversold conditions."
)

st.divider()

# =========================================================
# 14. MARKET INFORMATION
# =========================================================

st.subheader("ℹ️ MARKET INFORMATION")

info1, info2 = st.columns(2)

with info1:
    st.write("**Asset:**", selected_asset)
    st.write("**Timeframe:**", timeframe)
    st.write("**Data Source:** Yahoo Finance")

with info2:
    st.write("**Last Candle:**", str(data.index[-1]))
    st.write("**Last Update:**", str(pd.Timestamp.now()))

st.caption(
    "OUTEMGOUNT MA NEXUS | Analytical dashboard only. "
    "Signals and risk levels are estimates, not financial advice."
)
