
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="OUTEMGOUNT MA Nexus",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# 2. PROFESSIONAL DESIGN
# ==========================================
st.markdown("""
<style>
.stApp {
    background-color: #0E1117;
}
h1, h2, h3 {
    color: #FFD700 !important;
}
div[data-testid="stMetricValue"] {
    color: #FFD700;
    font-size: 24px;
    font-weight: bold;
}
div[data-testid="stMetricLabel"] {
    color: #A0AEC0;
}
.stDataFrame {
    border: 1px solid #2D3748;
    border-radius: 10px;
}
div.stButton > button {
    background: linear-gradient(90deg,#B8860B,#FFD700);
    color: black;
    font-weight: bold;
    border-radius: 8px;
    border: none;
}
hr {
    border-color: #2D3748;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. HEADER
# ==========================================
st.title("📊 OUTEMGOUNT MA Nexus")
st.caption("PROFESSIONAL TRADING ANALYSIS | PHOENIX EDITION")

col1, col2, col3 = st.columns([2, 2, 1])

symbol_map = {
    "XAUUSD (Gold)": "GC=F",
    "EURUSD": "EURUSD=X",
    "BTCUSD (Bitcoin)": "BTC-USD",
    "Oil (WTI)": "CL=F",
    "Nasdaq 100": "^IXIC"
}

with col1:
    symbol = st.selectbox(
        "Select Asset",
        list(symbol_map.keys())
    )

with col2:
    timeframe = st.selectbox(
        "Select Timeframe",
        ["15m", "1h", "4h", "1d"],
        index=1
    )

with col3:
    st.write("")
    st.write("")
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# ==========================================
# 4. INDICATOR SETTINGS
# ==========================================
with st.expander("⚙️ Chart Settings", expanded=False):
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        show_ma5 = st.checkbox("MA 5", True)
        show_ma13 = st.checkbox("MA 13", True)

    with c2:
        show_ma20 = st.checkbox("MA 20", True)
        show_ma50 = st.checkbox("MA 50", True)

    with c3:
        show_kijun = st.checkbox("Kijun", True)
        show_signals = st.checkbox("BUY / SELL Signals", True)

    with c4:
        show_rsi = st.checkbox("RSI Panel", True)
        show_atr = st.checkbox("ATR Information", True)

# ==========================================
# 5. DATA DOWNLOAD
# ==========================================
@st.cache_data(ttl=60)
def get_data(sym, tf):

    if tf == "15m":
        period = "5d"
        interval = "15m"

    elif tf == "1h":
        period = "3mo"
        interval = "1h"

    elif tf == "4h":
        period = "6mo"
        interval = "1h"

    else:
        period = "2y"
        interval = "1d"

    df = yf.download(
        sym,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False
    )

    if df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        for level in range(df.columns.nlevels):
            values = df.columns.get_level_values(level)
            if "Open" in values and "Close" in values:
                df.columns = values
                break

    required = ["Open", "High", "Low", "Close"]

    if not all(col in df.columns for col in required):
        return pd.DataFrame()

    if tf == "4h":
        df = df.resample("4h").agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum"
        } if "Volume" in df.columns else {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last"
        })

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.dropna(subset=required)


# ==========================================
# 6. CALCULATE INDICATORS
# ==========================================
def calculate_indicators(df):

    df = df.copy()

    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA13"] = df["Close"].rolling(13).mean()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()

    # RSI 14
    delta = df["Close"].diff()

    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()

    rs = gain / loss.replace(0, np.nan)

    df["RSI"] = 100 - (100 / (1 + rs))

    df.loc[(loss == 0) & (gain > 0), "RSI"] = 100
    df.loc[(loss == 0) & (gain == 0), "RSI"] = 50

    # ATR 14
    high_low = df["High"] - df["Low"]

    high_close = abs(
        df["High"] - df["Close"].shift(1)
    )

    low_close = abs(
        df["Low"] - df["Close"].shift(1)
    )

    df["TR"] = pd.concat(
        [high_low, high_close, low_close],
        axis=1
    ).max(axis=1)

    df["ATR"] = df["TR"].rolling(14).mean()

    # KIJUN
    df["Kijun_High"] = df["High"].rolling(26).max()
    df["Kijun_Low"] = df["Low"].rolling(26).min()

    df["Kijun"] = (
        df["Kijun_High"] + df["Kijun_Low"]
    ) / 2

    # MA CROSSOVER
    df["Crossover"] = np.where(
        df["MA5"] > df["MA13"], 1, -1
    )

    df["Signal"] = df["Crossover"].diff()

    return df


# ==========================================
# 7. MAIN ANALYSIS
# ==========================================
try:
    actual_symbol = symbol_map[symbol]
    data = get_data(actual_symbol, timeframe)

    if data.empty or len(data) < 55:
        st.error(
            "Not enough market data. Please try another timeframe."
        )
        st.stop()

    data = calculate_indicators(data)

    # Last CLOSED candle
    last_closed = -2

    current_price = float(
        data["Close"].iloc[last_closed]
    )

    current_atr = float(
        data["ATR"].iloc[last_closed]
    )

    current_kijun = float(
        data["Kijun"].iloc[last_closed]
    )

    current_ma20 = float(
        data["MA20"].iloc[last_closed]
    )

    current_rsi = float(
        data["RSI"].iloc[last_closed]
    )

    if not np.isfinite(current_atr) or current_atr <= 0:
        st.error("ATR data is not available.")
        st.stop()

    latest_signal = data["Signal"].iloc[last_closed]

    # ======================================
    # 8. MARKET TREND
    # ======================================
    st.subheader("🌍 Market Trend | اتجاه السوق")

    if current_price > current_kijun and current_price > current_ma20:
        trend = "UPTREND 🚀"
        st.success("UPTREND | اتجاه صاعد")

    elif current_price < current_kijun and current_price < current_ma20:
        trend = "DOWNTREND 📉"
        st.error("DOWNTREND | اتجاه هابط")

    else:
        trend = "RANGE ↔️"
        st.warning("RANGE | اتجاه عرضي")

    # ======================================
    # 9. TRADING SIGNALS
    # ======================================
    st.subheader("🎯 Trading Signals")

    if latest_signal == 2:
        signal_name = "BUY"
        st.success(f"🟢 BUY SIGNAL | إشارة شراء - {timeframe}")

    elif latest_signal == -2:
        signal_name = "SELL"
        st.error(f"🔴 SELL SIGNAL | إشارة بيع - {timeframe}")

    else:
        signal_name = "WAIT"
        st.info("⏳ WAIT | لا توجد إشارة جديدة")

    # ======================================
    # 10. ENTRY / SL / TP
    # ======================================
    st.subheader("💰 Trade Management")

    risk = current_atr * 1.5
    entry = current_price

    if signal_name == "BUY":

        sl = entry - risk
        tp1 = entry + risk * 1.5
        tp2 = entry + risk * 3

    elif signal_name == "SELL":

        sl = entry + risk
        tp1 = entry - risk * 1.5
        tp2 = entry - risk * 3

    else:
        sl = np.nan
        tp1 = np.nan
        tp2 = np.nan

    if signal_name in ["BUY", "SELL"]:

        a, b, c, d = st.columns(4)

        a.metric("ENTRY", f"{entry:.2f}")
        b.metric("STOP LOSS", f"{sl:.2f}")
        c.metric("TARGET 1", f"{tp1:.2f}")
        d.metric("TARGET 2", f"{tp2:.2f}")

    else:
        st.info("No active trade levels.")

    # ======================================
    # 11. MARKET METRICS
    # ======================================
    st.subheader("📈 Market Dashboard")

    a, b, c, d = st.columns(4)

    a.metric(
        "Market Price",
        f"{current_price:.2f}"
    )

    b.metric(
        "RSI 14",
        f"{current_rsi:.1f}"
    )

    c.metric(
        "ATR 14",
        f"{current_atr:.4f}"
    )

    d.metric(
        "Trend",
        trend
    )

    # RSI STATUS
    if current_rsi >= 70:
        st.warning("RSI: Overbought | تشبع شرائي")

    elif current_rsi <= 30:
        st.warning("RSI: Oversold | تشبع بيعي")

    else:
        st.info("RSI: Neutral Zone | منطقة محايدة")

    # ======================================
    # 12. PROFESSIONAL CANDLESTICK CHART
    # ======================================
    st.subheader("🕯️ Professional Price Chart")

    chart_data = data.tail(100)

    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=chart_data.index,
        open=chart_data["Open"],
        high=chart_data["High"],
        low=chart_data["Low"],
        close=chart_data["Close"],
        name="Price",
        increasing_line_color="#00FF00",
        decreasing_line_color="#FF0000"
    ))

    # MOVING AVERAGES
    ma_settings = [
        ("MA5", show_ma5, "#FFD700"),
        ("MA13", show_ma13, "#00FFFF"),
        ("MA20", show_ma20, "#FF8C00"),
        ("MA50", show_ma50, "#FF00FF"),
        ("Kijun", show_kijun, "#FFFFFF")
    ]

    for name, visible, color in ma_settings:

        if visible:
            fig.add_trace(go.Scatter(
                x=chart_data.index,
                y=chart_data[name],
                name=name,
                mode="lines",
                line=dict(
                    color=color,
                    width=1.5
                )
            ))

    # BUY / SELL MARKERS
    if show_signals:

        buy_points = chart_data[
            chart_data["Signal"] == 2
        ]

        sell_points = chart_data[
            chart_data["Signal"] == -2
        ]

        fig.add_trace(go.Scatter(
            x=buy_points.index,
            y=buy_points["Low"],
            mode="markers+text",
            name="BUY",
            text=["BUY"] * len(buy_points),
            textposition="bottom center",
            marker=dict(
                symbol="triangle-up",
                size=12,
                color="#00FF00"
            )
        ))

        fig.add_trace(go.Scatter(
            x=sell_points.index,
            y=sell_points["High"],
            mode="markers+text",
            name="SELL",
            text=["SELL"] * len(sell_points),
            textposition="top center",
            marker=dict(
                symbol="triangle-down",
                size=12,
                color="#FF0000"
            )
        ))

    # ENTRY / SL / TP LINES
    if signal_name in ["BUY", "SELL"]:

        for value, label, color in [
            (entry, "ENTRY", "#FFD700"),
            (sl, "STOP LOSS", "#FF0000"),
            (tp1, "TP1", "#00FF00"),
            (tp2, "TP2", "#00FFFF")
        ]:

            fig.add_hline(
                y=value,
                line_dash="dash",
                line_color=color,
                annotation_text=label,
                annotation_position="top right"
            )

    fig.update_layout(
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        height=550,
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ======================================
    # 13. RSI CHART
    # ======================================
    if show_rsi:

        st.subheader("📊 RSI Analysis")

        rsi_fig = go.Figure()

        rsi_fig.add_trace(go.Scatter(
            x=chart_data.index,
            y=chart_data["RSI"],
            name="RSI 14",
            line=dict(
                color="#FFD700",
                width=2
            )
        ))

        rsi_fig.add_hline(
            y=70,
            line_dash="dash",
            line_color="red"
        )

        rsi_fig.add_hline(
            y=30,
            line_dash="dash",
            line_color="green"
        )

        rsi_fig.add_hline(
            y=50,
            line_dash="dot",
            line_color="gray"
        )

        rsi_fig.update_layout(
            template="plotly_dark",
            height=220,
            yaxis=dict(
                range=[0, 100]
            ),
            paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117"
        )

        st.plotly_chart(
            rsi_fig,
            use_container_width=True
        )

    # ======================================
    # 14. ATR INFORMATION
    # ======================================
    if show_atr:

        st.subheader("🛡️ Risk Management")

        st.write(
            f"ATR 14: {current_atr:.4f}"
        )

        st.write(
            f"Suggested ATR Stop Distance: {risk:.4f}"
        )

        st.caption(
            "ATR-based levels are analytical estimates, "
            "not guaranteed execution prices."
        )

    # ======================================
    # 15. DATA INFORMATION
    # ======================================
    st.divider()

    st.caption(
        f"Asset: {symbol} | "
        f"Timeframe: {timeframe} | "
        f"Last candle: {data.index[last_closed]} | "
        f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    st.caption(
        "OUTEMGOUNT MA Nexus | Professional Trading Dashboard"
    )

except Exception as e:

    st.error(
        "An unexpected error occurred while loading market data."
    )

    st.exception(e)

    st.info(
        "Please refresh the application or try another asset."
    )
