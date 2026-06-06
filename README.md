# BinanceRadarPro v21 ULTRA

**AI-Powered Crypto Trading Radar + Telegram Bot**  
تحليل متعدد الأطر الزمنية، توقعات LSTM، وإشعارات تلغرام فورية

---

## المحتويات

- [نظرة عامة](#نظرة-عامة)
- [المتطلبات](#المتطلبات)
- [التثبيت المحلي](#التثبيت-المحلي)
- [النشر على Railway](#النشر-على-railway)
- [إعداد بوت تلغرام](#إعداد-بوت-تلغرام)
- [أوضاع التشغيل](#أوضاع-التشغيل)
- [أوامر البوت](#أوامر-البوت)
- [المكتبات المستخدمة](#المكتبات-المستخدمة)
- [هيكل المشروع](#هيكل-المشروع)
- [استكشاف الأخطاء](#استكشاف-الأخطاء)

---

## نظرة عامة

BinanceRadarPro هو نظام تحليل تداول متقدم يعمل على بيانات Binance Futures الحية ويرسل السيگنالات عبر تلغرام.

### المميزات الرئيسية

| الميزة | التفاصيل |
|--------|----------|
| **Multi-Timeframe** | تحليل 5 أطر زمنية (5m / 15m / 1h / 4h / 1d) |
| **AI Prediction** | GradientBoosting + XGBoost + LinearRegression ensemble |
| **Order Book** | تحليل 50 مستوى عمق، CVD، اكتشاف spoofing |
| **Indicators** | MACD، RSI، Stoch RSI، EMA، Ichimoku، SuperTrend، OBV |
| **Smart Entry** | مناطق دخول ذكية مع TP1/TP2/TP3 و SL ديناميكي |
| **Telegram Bot** | إشعارات تلقائية + لوحة تحكم كاملة |
| **Backtest Live** | تتبع نتائج السيگنالات في الوقت الحقيقي |
| **Headless Mode** | يعمل على السيرفر بدون واجهة رسومية |

---

## المتطلبات

### Python
```
Python 3.10 أو أحدث
```

### المكتبات

| المكتبة | الإصدار الأدنى | الاستخدام |
|---------|---------------|-----------|
| `requests` | ≥ 2.31.0 | Binance REST API + Telegram API |
| `websocket-client` | ≥ 1.6.0 | WebSocket للسعر الحي |
| `numpy` | ≥ 1.24.0 | العمليات الرياضية والمصفوفات |
| `pandas` | ≥ 2.0.0 | معالجة بيانات الشموع |
| `scikit-learn` | ≥ 1.3.0 | LinearRegression, GradientBoosting |
| `xgboost` | ≥ 2.0.0 | توقعات XGBoost (موصى به) |

### مكتبات مدمجة (لا تحتاج تثبيت)
```
threading, queue, logging, json, time, os,
datetime, collections, dataclasses, typing
```

### ملاحظات المنصة
- **tkinter** — مدمج مع Python، يُتجاهل تلقائياً في headless mode
- **winsound** — Windows فقط، يُتجاهل على Linux/Mac بدون أخطاء

---

## التثبيت المحلي

```bash
# 1. استنساخ أو تحميل الملفات
mkdir binance-radar && cd binance-radar

# 2. تثبيت المكتبات
pip install -r requirements.txt

# 3. تشغيل مع واجهة رسومية (Windows/Mac)
python radar_merged.py --gui

# 4. تشغيل في الخلفية بدون واجهة
python radar_merged.py --headless

# 5. كشف تلقائي للوضع المناسب
python radar_merged.py
```

---

## النشر على Railway

### الخطوة 1 — تجهيز الملفات

تأكد من وجود هذه الملفات الثلاثة في نفس المجلد:

```
📁 المشروع
├── radar_merged.py      ← الملف الرئيسي
├── requirements.txt     ← المكتبات
└── Procfile             ← أمر التشغيل
```

محتوى `Procfile`:
```
worker: python radar_merged.py --headless
```

### الخطوة 2 — رفع على GitHub

```bash
git init
git add radar_merged.py requirements.txt Procfile
git commit -m "BinanceRadarPro v21 ULTRA"
git branch -M main
git remote add origin https://github.com/USERNAME/REPO.git
git push -u origin main
```

### الخطوة 3 — إنشاء مشروع على Railway

1. اذهب إلى [railway.app](https://railway.app)
2. سجّل دخول بـ GitHub
3. اضغط **New Project** ← **Deploy from GitHub repo**
4. اختر الـ repo
5. Railway يكتشف `Procfile` تلقائياً

### الخطوة 4 — التحقق من التشغيل

في تبويب **Deployments** ← **View Logs** يجب أن ترى:
```
✅ [HEADLESS] build_ui skipped — running in background mode
✅ الرادار يعمل: BTCUSDT [1h]
✅ Telegram bot threads launched (polling + notifier)
✅ البوت جاهز! ابدأ المحادثة بـ /start
```

### حدود الاستخدام المجاني
- **500 ساعة/شهر** مجاناً
- بعدها ~**$5/شهر** للاستخدام المستمر
- للتشغيل 24/7 يُنصح بالاشتراك المدفوع

---

## إعداد بوت تلغرام

### الحصول على Token
1. افتح [@BotFather](https://t.me/BotFather) في تلغرام
2. اكتب `/newbot`
3. اختر اسماً للبوت
4. انسخ الـ Token

### تعديل Token في الكود
في `radar_merged.py` ابحث عن:
```python
TG_TOKEN = "YOUR_TOKEN_HERE"
```
واستبدله بـ Token الخاص بك.

### بدء استخدام البوت
1. ابحث عن البوت في تلغرام باسمه
2. اكتب `/start`
3. ستظهر لوحة الأزرار

---

## أوضاع التشغيل

### `--headless` (موصى به على السيرفر)
```bash
python radar_merged.py --headless
```
- بدون واجهة رسومية
- مناسب لـ Railway، Render، VPS
- البوت يعمل بالكامل عبر تلغرام

### `--gui` (للاستخدام المحلي)
```bash
python radar_merged.py --gui
```
- يفتح نافذة Tkinter
- بوت تلغرام يعمل في الخلفية تلقائياً

### كشف تلقائي (بدون argument)
```bash
python radar_merged.py
```
- إذا وُجد `DISPLAY` أو `WAYLAND_DISPLAY` → GUI
- إذا لا → headless تلقائياً

---

## أوامر البوت

| الزر | الوظيفة |
|------|---------|
| 📊 السيگنال الحالي | BUY/SELL/WAIT مع Entry، SL، TP، R:R |
| 📈 MTF كامل | تحليل 5 أطر زمنية مع MASTER signal |
| 🔍 Intel / SR Levels | مستويات دعم/مقاومة، Liquidity، هيكل السوق |
| 📰 الأخبار | أخبار العملة مع sentiment analysis |
| 📂 سجل السيگنالات | آخر 10 سيگنالات مع إحصاءات |
| 🏆 Win Rate AI | دقة توقعات الـ AI |
| ⚙️ تغيير العملة | BTC، ETH، BNB، SOL، XRP، DOGE... |
| ⏱ تغيير الإطار | 5m، 15m، 1h، 4h، 1d |
| 💾 حفظ السيگنال | حفظ في `saved_signals_tg.json` |
| 🔔 تفعيل الإشعارات | إشعارات تلقائية عند كل سيگنال جديد |
| 📉 Backtest | نتائج الصفقات التاريخية |

---

## المكتبات المستخدمة

### `requests` — HTTP Client
```python
import requests
# تُستخدم لـ: Binance REST API, Telegram Bot API
# الوظائف: klines, orderbook, funding, news, tg_send, tg_edit
```

### `websocket-client` — WebSocket
```python
import websocket
# تُستخدم لـ: بث السعر الحي من Binance
# الوظائف: markPrice stream, kline stream
```

### `numpy` — Numerical Computing
```python
import numpy as np
# تُستخدم لـ: حساب المؤشرات الفنية
# EMA, RSI, MACD, Stoch, Bollinger, ATR, ADX
```

### `pandas` — Data Processing
```python
import pandas as pd
# تُستخدم لـ: تحويل klines إلى DataFrame
# معالجة بيانات الشموع، حسابات rolling/ewm
```

### `scikit-learn` — Machine Learning
```python
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
# تُستخدم لـ: AI price prediction ensemble
# Linear trend + GBM ensemble للتوقعات
```

### `xgboost` — Gradient Boosting
```python
import xgboost as xgb
# تُستخدم لـ: XGBoost prediction model
# جزء من ensemble مع LinearRegression و GBM
# اختياري — الكود يعمل بدونه
```

---

## هيكل المشروع

```
radar_merged.py
│
├── Helper Functions          (lines ~1-540)
│   ├── _calc_macd()
│   ├── _calc_vwap()
│   ├── _calc_stoch_rsi()
│   └── Config constants
│
├── Data Classes              (lines ~300-540)
│   ├── Candle
│   ├── AnalysisResult
│   └── NarrativeResult
│
├── BinanceRadarPro Class     (lines ~542-6345)
│   ├── __init__()            — تهيئة كل المتغيرات
│   ├── build_ui()            — واجهة Tkinter (GUI فقط)
│   ├── start_bg()            — تشغيل كل الـ threads
│   ├── load_historical_klines() — تحميل الشموع مع retry
│   ├── connect_websockets()  — اتصال WebSocket
│   ├── analysis_loop()       — حلقة التحليل الرئيسية
│   ├── _deep_analyze()       — محرك التحليل الكامل
│   ├── run_mtf_scan()        — Multi-Timeframe scan
│   ├── _intel_thread()       — تحديث Intel & S/R
│   ├── _news_thread()        — جلب الأخبار
│   └── refresh_ui()          — تحديث الواجهة (GUI فقط)
│
├── Headless Patches          (lines ~6346-6500)
│   ├── FakeRoot              — بديل tk.Tk() بدون نافذة
│   ├── _FakeMock             — بديل صامت لكل الـ widgets
│   ├── _safe_build_ui()      — build_ui آمن للـ headless
│   └── _safe_refresh_ui()    — refresh_ui آمن للـ headless
│
├── Telegram Bot              (lines ~6501-7100)
│   ├── _tg_send/edit/answer  — API helpers
│   ├── Keyboards             — لوحات الأزرار
│   ├── Formatters            — تنسيق الرسائل
│   ├── Handlers              — معالجة الأوامر
│   ├── _tg_polling_loop()    — استقبال التحديثات
│   ├── _tg_notifier_loop()   — إرسال الإشعارات
│   └── start_telegram_bot()  — تشغيل البوت
│
└── Entry Point               (lines ~7101-7245)
    ├── _run_headless()       — تشغيل بدون GUI
    ├── _run_gui()            — تشغيل مع GUI
    └── __main__              — auto-detect mode
```

---

## استكشاف الأخطاء

### `AttributeError: has no attribute '_intel_status_lbl'`
**السبب:** widget غير موجود في headless mode  
**الحل:** محلول في v21 ULTRA — `_safe_build_ui` يُنشئ mock لكل الـ widgets

### `Only 2 candles received`
**السبب:** Binance API بطيء أو محدود  
**الحل:** محلول في v21 ULTRA — 5 محاولات تلقائية مع fallback لـ spot API

### `WebSocket connection failed`
**السبب:** انقطاع الشبكة أو timeout  
**الحل:** `ws_manager_loop` يعيد الاتصال تلقائياً كل 30 ثانية

### `Rate limit (429)`
**السبب:** طلبات كثيرة على Binance API  
**الحل:** الكود يدخل cooldown تلقائي لمدة 60 ثانية

### البوت لا يرد على تلغرام
1. تحقق من الـ Token في `TG_TOKEN`
2. تأكد أن الـ Railway deployment شغّال
3. راجع logs: ابحث عن `Telegram polling بدأ`

### Railway يوقف التشغيل
- تأكد أن الملف `Procfile` يستخدم `worker:` وليس `web:`
- `web:` يتوقف إذا لم يكن هناك HTTP server
- `worker:` يعمل باستمرار في الخلفية

---

## متغيرات البيئة (اختياري)

يمكن تعريفها في Railway بدل تعديل الكود:

```env
TG_TOKEN=your_telegram_bot_token
DEFAULT_SYMBOL=BTCUSDT
DEFAULT_INTERVAL=1h
```

---

## الترخيص

للاستخدام الشخصي فقط.  
لا تشارك الـ Token أو تنشر الكود علنياً.
