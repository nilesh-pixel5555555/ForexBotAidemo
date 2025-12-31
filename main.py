# main.py - FOREX AI BOT V2.5 (Bulletproof Edition)

import os
import sys
import logging
import asyncio # Added for the fix
from datetime import datetime, timedelta
from flask import Flask, jsonify
import json

# --- START WINDOWS FIX ---
# This forces Windows to use a specific event loop policy to prevent "Event loop is closed" errors
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
# --- END WINDOWS FIX ---

# Setup logging first
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app first
app = Flask(__name__)

# Global variables
bot_stats = {
    "status": "initializing",
    "total_analyses": 0,
    "last_analysis": None,
    "uptime_start": datetime.now().isoformat(),
    "version": "V2.5 Forex Elite"
}
trade_history = []
TRADE_HISTORY_FILE = "forex_trade_history.json"

# Safe imports with error handling
bot = None
exchange = None
scheduler = None

try:
    from dotenv import load_dotenv
    load_dotenv()
    logger.info("✅ Environment loaded")
except Exception as e:
    logger.warning(f"⚠️ dotenv not available: {e}")

try:
    from telegram import Bot
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
    
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        bot = Bot(token=TELEGRAM_BOT_TOKEN)
        logger.info("✅ Telegram bot initialized")
    else:
        logger.warning("⚠️ Telegram credentials missing")
except Exception as e:
    logger.error(f"❌ Telegram init failed: {e}")

try:
    import ccxt
    exchange = ccxt.kraken({'enableRateLimit': True, 'rateLimit': 2000, 'timeout': 30000})
    logger.info("✅ Exchange initialized")
except Exception as e:
    logger.error(f"❌ Exchange init failed: {e}")

try:
    import pandas as pd
    import numpy as np
    logger.info("✅ Data libraries loaded")
except Exception as e:
    logger.error(f"❌ Data libraries failed: {e}")
    pd = None
    np = None

# Configuration
FOREX_PAIRS = [p.strip() for p in os.getenv("FOREX_PAIRS", "EUR/USD,GBP/USD,USD/JPY,AUD/USD,USD/CHF").split(',')]
bot_stats["monitored_assets"] = FOREX_PAIRS

def load_trade_history():
    """Load trade history."""
    global trade_history
    try:
        if os.path.exists(TRADE_HISTORY_FILE):
            with open(TRADE_HISTORY_FILE, 'r') as f:
                trade_history = json.load(f)
                logger.info(f"📊 Loaded {len(trade_history)} trades")
    except Exception as e:
        logger.warning(f"Could not load history: {e}")
        trade_history = []

def save_trade_history():
    """Save trade history."""
    try:
        with open(TRADE_HISTORY_FILE, 'w') as f:
            json.dump(trade_history, f, indent=2)
    except Exception as e:
        logger.error(f"Could not save history: {e}")

def add_trade(symbol, signal, entry_price, tp1, tp2, sl):
    """Add new trade."""
    try:
        trade = {
            "id": len(trade_history) + 1,
            "symbol": symbol,
            "signal": signal,
            "entry_price": float(entry_price),
            "tp1": float(tp1),
            "tp2": float(tp2),
            "sl": float(sl),
            "timestamp": datetime.now().isoformat(),
            "status": "ACTIVE",
            "outcome": None,
            "profit_loss_pips": 0.0
        }
        trade_history.append(trade)
        save_trade_history()
        logger.info(f"Trade added: {symbol} {signal}")
    except Exception as e:
        logger.error(f"Error adding trade: {e}")

def calculate_pips(symbol, price1, price2):
    """Calculate pips."""
    try:
        pip_value = 0.01 if 'JPY' in symbol else 0.0001
        return abs(float(price2) - float(price1)) / pip_value
    except:
        return 0.0

def fetch_data_safe(symbol, timeframe):
    """Fetch market data."""
    if not exchange or not pd:
        return None
        
    try:
        if not exchange.markets:
            exchange.load_markets()
        market_id = exchange.market(symbol)['id']
        ohlcv = exchange.fetch_ohlcv(market_id, timeframe, limit=100)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df['sma9'] = df['close'].rolling(9).mean()
        df['sma20'] = df['close'].rolling(20).mean()
        return df.dropna()
    except Exception as e:
        logger.error(f"Fetch error for {symbol}: {e}")
        return None

def calculate_cpr(df_daily):
    """Calculate pivot points."""
    try:
        if df_daily is None or len(df_daily) < 2:
            return None
        prev = df_daily.iloc[-2]
        H, L, C = float(prev['high']), float(prev['low']), float(prev['close'])
        PP = (H + L + C) / 3.0
        return {
            'PP': PP,
            'R1': 2*PP - L,
            'S1': 2*PP - H,
            'R2': PP + (H - L),
            'S2': PP - (H - L)
        }
    except:
        return None

def generate_signal(symbol):
    """Generate trading signal."""
    global bot_stats
    
    if not bot or not exchange or not pd:
        logger.warning("Required services not available")
        return
        
    try:
        # Fetch data
        df_4h = fetch_data_safe(symbol, '4h')
        df_1h = fetch_data_safe(symbol, '1h')
        
        if df_4h is None or df_1h is None:
            logger.warning(f"No data for {symbol}")
            return
            
        # Get daily data for pivots
        market_id = exchange.market(symbol)['id']
        ohlcv_d = exchange.fetch_ohlcv(market_id, '1d', limit=5)
        df_d = pd.DataFrame(ohlcv_d, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        cpr = calculate_cpr(df_d)
        
        if cpr is None:
            return
            
        # Analyze trends
        price = float(df_4h.iloc[-1]['close'])
        trend_4h = "BULLISH" if df_4h.iloc[-1]['sma9'] > df_4h.iloc[-1]['sma20'] else "BEARISH"
        trend_1h = "BULLISH" if df_1h.iloc[-1]['sma9'] > df_1h.iloc[-1]['sma20'] else "BEARISH"
        
        signal = "HOLD"
        emoji = "⏳"
        
        if trend_4h == "BULLISH" and trend_1h == "BULLISH" and price > cpr['PP']:
            signal = "STRONG BUY"
            emoji = "🚀"
        elif trend_4h == "BEARISH" and trend_1h == "BEARISH" and price < cpr['PP']:
            signal = "STRONG SELL"
            emoji = "🔻"
            
        # Calculate targets
        is_buy = "BUY" in signal
        tp1 = cpr['R1'] if is_buy else cpr['S1']
        tp2 = cpr['R2'] if is_buy else cpr['S2']
        sl = cpr['S1'] if is_buy else cpr['R1']
        
        decimals = 3 if 'JPY' in symbol else 5
        
        # Track trade
        if "BUY" in signal or "SELL" in signal:
            add_trade(symbol, signal, price, tp1, tp2, sl)
            tp1_pips = calculate_pips(symbol, price, tp1)
            tp2_pips = calculate_pips(symbol, price, tp2)
            sl_pips = calculate_pips(symbol, price, sl)
        
        # Build message
        message = (
            f"╔════════════════════════════════╗\n"
            f"  🌍 <b>FOREX AI SIGNAL</b>\n"
            f"╚════════════════════════════════╝\n\n"
            f"<b>Pair:</b> {symbol}\n"
            f"<b>Price:</b> <code>{price:.{decimals}f}</code>\n\n"
            f"--- {emoji} <b>{signal}</b> ---\n\n"
            f"<b>Trends:</b>\n"
            f"• 4H: {trend_4h}\n"
            f"• 1H: {trend_1h}\n\n"
        )
        
        if "BUY" in signal or "SELL" in signal:
            message += (
                f"<b>Targets:</b>\n"
                f"✅ TP1: {tp1:.{decimals}f} (+{tp1_pips:.1f}p)\n"
                f"🔥 TP2: {tp2:.{decimals}f} (+{tp2_pips:.1f}p)\n"
                f"🛑 SL: {sl:.{decimals}f} (-{sl_pips:.1f}p)\n\n"
            )
            
        message += "<i>Forex AI V2.5</i>"
        
        # Send message
        asyncio.run(bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message, parse_mode='HTML'))
        
        bot_stats['total_analyses'] += 1
        bot_stats['last_analysis'] = datetime.now().isoformat()
        bot_stats['status'] = "operational"
        logger.info(f"✅ Signal sent: {symbol}")
        
    except Exception as e:
        logger.error(f"Signal generation error: {e}")
        import traceback
        traceback.print_exc()

def check_trades():
    """Check trade outcomes."""
    if not exchange:
        return
        
    for trade in trade_history:
        if trade['status'] == 'ACTIVE':
            try:
                df = fetch_data_safe(trade['symbol'], '1h')
                if df is None:
                    continue
                    
                current = float(df.iloc[-1]['close'])
                entry = float(trade['entry_price'])
                is_buy = "BUY" in trade['signal']
                
                if is_buy:
                    if current >= float(trade['tp2']):
                        trade['status'] = 'TP2_HIT'
                        trade['outcome'] = 'WIN'
                        trade['profit_loss_pips'] = calculate_pips(trade['symbol'], entry, trade['tp2'])
                    elif current >= float(trade['tp1']):
                        trade['status'] = 'TP1_HIT'
                        trade['outcome'] = 'PARTIAL_WIN'
                        trade['profit_loss_pips'] = calculate_pips(trade['symbol'], entry, trade['tp1'])
                    elif current <= float(trade['sl']):
                        trade['status'] = 'SL_HIT'
                        trade['outcome'] = 'LOSS'
                        trade['profit_loss_pips'] = -calculate_pips(trade['symbol'], entry, trade['sl'])
                else:
                    if current <= float(trade['tp2']):
                        trade['status'] = 'TP2_HIT'
                        trade['outcome'] = 'WIN'
                        trade['profit_loss_pips'] = calculate_pips(trade['symbol'], trade['tp2'], entry)
                    elif current <= float(trade['tp1']):
                        trade['status'] = 'TP1_HIT'
                        trade['outcome'] = 'PARTIAL_WIN'
                        trade['profit_loss_pips'] = calculate_pips(trade['symbol'], trade['tp1'], entry)
                    elif current >= float(trade['sl']):
                        trade['status'] = 'SL_HIT'
                        trade['outcome'] = 'LOSS'
                        trade['profit_loss_pips'] = -calculate_pips(trade['symbol'], trade['sl'], entry)
                        
            except Exception as e:
                logger.error(f"Error checking trade: {e}")
    
    save_trade_history()

def daily_report():
    """Send daily report."""
    if not bot:
        return
        
    try:
        check_trades()
        
        now = datetime.now()
        last_24h = now - timedelta(hours=24)
        recent = [t for t in trade_history if datetime.fromisoformat(t['timestamp']) >= last_24h]
        
        if not recent:
            message = "📊 <b>24H REPORT</b>\n\nNo signals in last 24 hours."
        else:
            wins = len([t for t in recent if t.get('outcome') == 'WIN'])
            partial = len([t for t in recent if t.get('outcome') == 'PARTIAL_WIN'])
            losses = len([t for t in recent if t.get('outcome') == 'LOSS'])
            total_pips = sum([t.get('profit_loss_pips', 0) for t in recent if t.get('outcome')])
            
            message = (
                f"📊 <b>24H FOREX REPORT</b>\n\n"
                f"Signals: {len(recent)}\n"
                f"✅ Wins: {wins}\n"
                f"⚡ Partial: {partial}\n"
                f"❌ Losses: {losses}\n"
                f"Net: {'🟢' if total_pips >= 0 else '🔴'} {total_pips:+.1f} pips"
            )
        
        asyncio.run(bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message, parse_mode='HTML'))
        logger.info("✅ Report sent")
        
    except Exception as e:
        logger.error(f"Report error: {e}")

def start_scheduler():
    """Start background scheduler."""
    global scheduler
    
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        import threading
        
        scheduler = BackgroundScheduler()
        
        # Schedule signals every hour
        for pair in FOREX_PAIRS:
            scheduler.add_job(generate_signal, 'cron', minute='0,30', args=[pair])
        
        # Daily report at 9 AM
        scheduler.add_job(daily_report, 'cron', hour=9, minute=0)
        
        # Check trades every 30 min
        scheduler.add_job(check_trades, 'cron', minute='*/30')
        
        scheduler.start()
        logger.info("✅ Scheduler started")
        
        # Run initial signals
        for pair in FOREX_PAIRS:
            threading.Thread(target=generate_signal, args=(pair,), daemon=True).start()
            
    except Exception as e:
        logger.error(f"Scheduler error: {e}")

# Flask routes
@app.route('/')
def home():
    total = len(trade_history)
    wins = len([t for t in trade_history if t.get('outcome') == 'WIN'])
    losses = len([t for t in trade_history if t.get('outcome') == 'LOSS'])
    wr = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0
    pips = sum([t.get('profit_loss_pips', 0) for t in trade_history if t.get('outcome')])
    
    html = f"""
    <html>
    <head><title>Forex AI Bot</title></head>
    <body style="font-family:Arial; background:#020617; color:#fff; text-align:center; padding:50px;">
        <div style="background:#0f172a; display:inline-block; padding:40px; border-radius:12px; border:1px solid #1e293b;">
            <h1 style="color:#38bdf8;">🌍 Forex AI Dashboard</h1>
            <p style="font-size:1.2em;">Status: <span style="color:#4ade80;">{bot_stats['status']}</span></p>
            <hr style="border-color:#1e293b;">
            <div style="text-align:left; margin-top:20px;">
                <p><b>Analyses:</b> {bot_stats['total_analyses']}</p>
                <p><b>Trades:</b> {total}</p>
                <p><b>Win Rate:</b> {wr:.1f}%</p>
                <p><b>Total Pips:</b> {pips:+.1f}</p>
            </div>
            <hr style="border-color:#1e293b;">
            <p style="font-size:0.9em; color:#94a3b8;">Version: {bot_stats['version']}</p>
        </div>
    </body>
    </html>
    """
    return html

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "time": datetime.now().isoformat()}), 200

@app.route('/stats')
def stats():
    wins = len([t for t in trade_history if t.get('outcome') == 'WIN'])
    losses = len([t for t in trade_history if t.get('outcome') == 'LOSS'])
    pips = sum([t.get('profit_loss_pips', 0) for t in trade_history if t.get('outcome')])
    
    return jsonify({
        "total_trades": len(trade_history),
        "wins": wins,
        "losses": losses,
        "total_pips": round(pips, 1),
        "status": bot_stats['status']
    })

# Initialize on startup
load_trade_history()
start_scheduler()

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    logger.info(f"🚀 Starting on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
