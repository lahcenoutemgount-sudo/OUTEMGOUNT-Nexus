
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import pytz

st.set_page_config(page_title="OUTEMGOUNT MA Nexus", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    .stApp { background-color: #0E1117; }
    h1, h2, h3 { color: #FFD700 !important; }
    div[data-testid="stMetricValue"] { color: #FFD700 !important; font-size: 24px; font-weight: bold; }
    div[data-testid="stMetricLabel"] { color: #A0AEC0 !important; font-size: 14px; }
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .stDataFrame { border: 1px solid #2D3748; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

st.title("📊 OUTEMGOUNT MA Nexus")
st.caption("Professional Trading Signals & Analysis")

col_sym, col_tf = st.columns(2)

with col_sym:
    symbol = st.selectbox("Select Asset", ["XAUUSD (Gold)", "EURUSD", "BTCUSD (Bitcoin)", "Oil (WTI)", "Nasdaq 100"], index=0)
with col_tf:
    timeframe = st.selectbox("Select Timeframe", ["15m", "1h", "4h", "1d"], index=1)

symbol_map = {"XAUUSD (Gold)": "GC=F", "EURUSD": "EURUSD=X", "BTCUSD (Bitcoin)": "BTC-USD", "Oil (WTI)": "CL=F", "Nasdaq 100": "^IXIC"}
actual_symbol = symbol_map[symbol]

@st.cache_data(ttl=60) 
def get_data(sym, tf):
    if tf in ["15m", "30m"]: period = "5d" 
    elif tf == "1h": period = "1mo"
    else: period = "3mo"
    return yf.download(sym, period=period, interval=tf, ignore_tz=True, auto_adjust=False, progress=False)

data = get_data(actual_symbol, timeframe)

if not data.empty:
    data = data.dropna()
    if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
    
    data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
    data['High'] = pd.to_numeric(data['High'], errors='coerce')
    data['Low'] = pd.to_numeric(data['Low'], errors='coerce')
    data['Open'] = pd.to_numeric(data['Open'], errors='coerce')

    data['MA5'] = data['Close'].rolling(window=5).mean()
    data['MA13'] = data['Close'].rolling(window=13).mean()
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()

    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))

    data['High-Low'] = data['High'] - data['Low']
    data['High-PrevClose'] = np.abs(data['High'] - data['Close'].shift(1))
    data['Low-PrevClose'] = np.abs(data['Low'] - data['Close'].shift(1))
    data['TR'] = data[['High-Low', 'High-PrevClose', 'Low-PrevClose']].max(axis=1)
    data['ATR'] = data['TR'].rolling(window=14).mean()

    data['Kijun_High'] = data['High'].rolling(window=26).max()
    data['Kijun_Low'] = data['Low'].rolling(window=26).min()
    data['Kijun'] = (data['Kijun_High'] + data['Kijun_Low']) / 2

    data['Crossover'] = np.where(data['MA5'] > data['MA13'], 1, -1)
    data['Signal'] = data['Crossover'].diff()

    last_closed = -2 
    current_price = float(data['Close'].iloc[last_closed])
    current_atr = float(data['ATR'].iloc[last_closed])
    current_kijun = float(data['Kijun'].iloc[last_closed])
    current_ma20 = float(data['MA20'].iloc[last_closed])
    
    risk = current_atr * 1.5
    entry_price = current_price
    sl_long = entry_price - risk
    sl_short = entry_price + risk
    tp1_long = entry_price + (risk * 1.5)
    tp2_long = entry_price + (risk * 3.0)
    tp1_short = entry_price - (risk * 1.5)
    tp2_short = entry_price - (risk * 3.0)

    latest_signal = data['Signal'].iloc[last_closed]

    if latest_signal == 2: st.success(f"🟢 BUY Signal | إشارة شراء - {timeframe}")
    elif latest_signal == -2: st.error(f"🔴 SELL Signal | إشارة بيع - {timeframe}")
    else: st.info(f"⏳ No Signal | لا توجد إشارة - {timeframe}")

    st.subheader("Market Trend | اتجاه السوق")
    if current_price > current_kijun and current_price > current_ma20: st.success("🚀 UPTREND | اتجاه صاعد")
    elif current_price < current_kijun and current_price < current_ma20: st.error("📉 DOWNTREND | اتجاه هابط")
    else: st.warning("↔️ RANGE | اتجاه عرضي")

    st.subheader("Trade Levels | مستويات الصفقة")
    col_e, col_sl, col_tp1, col_tp2 = st.columns(4)
    
    if latest_signal == 2 or latest_signal == -2:
        with col_e: st.metric(label="Entry | الدخول", value=f"{entry_price:.2f}")
        if latest_signal == 2:
            with col_sl: st.metric(label="Stop Loss", value=f"{sl_long:.2f}")
            with col_tp1: st.metric(label="Target 1", value=f"{tp1_long:.2f}")
            with col_tp2: st.metric(label="Target 2", value=f"{tp2_long:.2f}")
        else:
            with col_sl: st.metric(label="Stop Loss", value=f"{sl_short:.2f}")
            with col_tp1: st.metric(label="Target 1", value=f"{tp1_short:.2f}")
            with col_tp2: st.metric(label="Target 2", value=f"{tp2_short:.2f}")
    else:
        st.info("No active trade levels | لا توجد مستويات صفقة حالياً")

    col1, col2, col3 = st.columns(3)
    with col1: st.metric(label="Current Price", value=f"{current_price:.2f}")
    with col2: st.metric(label="RSI (14)", value=f"{float(data['RSI'].iloc[last_closed]):.1f}")
    with col3: st.metric(label="ATR (14)", value=f"{current_atr:.4f}")

    fig = go.Figure(data=[go.Candlestick(x=data.index.tail(50), open=data['Open'].tail(50), high=data['High'].tail(50), low=data['Low'].tail(50), close=data['Close'].tail(50), increasing_line_color='#00FF00', decreasing_line_color='#FF0000')])
    fig.update_layout(template='plotly_dark', xaxis_rangeslider_visible=False, height=400, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor='#0E1117', plot_bgcolor='#0E1117')
    st.plotly_chart(fig, use_container_width=True)
else:
    st.error("Could not fetch data")
