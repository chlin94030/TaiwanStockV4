"""
Taiwan Alpha Radar V7.3 Prime
=============================
Mobile-first Streamlit decision dashboard for Taiwan equities.

Core principles
---------------
1. Point-in-time first: optional external data are selected by available_from/as_of,
   never by the latest row when an historical snapshot is requested.
2. Multi-horizon: Short 2-10D / Mid 2-8W / Long 2-9M use different Entry/Risk logic.
3. Soft Mixture-of-Experts: expert applicability gates are blended; no max-score winner's curse.
4. True institutional flow is separated from price-volume accumulation. Missing flow is never fabricated.
5. Volume is context, not direction: supply dry-up is WATCH; reclaim/acceptance is required before entry.
6. New-position Edge and existing-position Hold Edge are different. Entry saturation does not automatically
   destroy a long-term thesis.
7. TDCC is treated as a Shadow Factor until a sufficiently long point-in-time history is available.
8. Edge shown in the UI is a cross-sectional percentile, not a probability of profit.

Optional repository files under data/
-------------------------------------
data/universe.csv
    ticker,name,industry[,market_cap]

data/fundamentals.csv
    ticker,as_of[,available_from],roic,roe,gross_margin,operating_margin,fcf_margin,debt_ratio,
    revenue_yoy,revenue_yoy_prev,eps_yoy,eps_yoy_prev,margin_delta,pe,pb,
    revenue_surprise,information_response,cfo_net_income,inventory_days_delta,
    receivable_days_delta,earnings_stability_score,pe_history_percentile,
    pb_history_percentile,peg,fcf_yield,asset_quality_score,capital_adequacy_score

data/flows.csv
    ticker,as_of[,available_from],foreign_5d_intensity,foreign_20d_intensity,
    trust_5d_intensity,trust_20d_intensity,foreign_ownership_delta,
    dealer_prop_intensity,margin_balance_delta,daytrade_ratio

data/tdcc.csv
    ticker,as_of[,available_from],large_holder_delta,small_holder_delta

Run
---
    pip install -r requirements.txt
    streamlit run app.py

Notes
-----
- Production mode automatically loads the official TWSE + TPEx company universe; the embedded list is emergency fallback only.
- yfinance is convenient for research/prototyping. For live trading decisions, replace RealtimeQuoteProvider
  with a licensed real-time feed/broker API and keep the strategy layer unchanged.
- This app is a research and risk-management tool, not a guarantee of future returns.
"""

from __future__ import annotations

import math
import os
import logging
import time
import re
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, date
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf

# -----------------------------------------------------------------------------
# 0. App + logging
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="Taiwan Alpha Radar V7.3",
    page_icon="📡",
    layout="centered",
    initial_sidebar_state="collapsed",
)

logger = logging.getLogger("alpha_radar")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


# -----------------------------------------------------------------------------
# 1. Mobile-first visual system
# -----------------------------------------------------------------------------

st.markdown(
    """
<style>
:root {
  --ink:#172033; --muted:#667085; --line:#e7ebf2; --paper:#ffffff;
  --bg:#f7f9fc; --blue:#1d4ed8; --cyan:#0e7490; --green:#047857;
  --amber:#b45309; --red:#b91c1c; --violet:#6d28d9;
}
html, body, [class*="css"] {font-size:16px;}
.stApp {background:linear-gradient(180deg,#fbfcff 0%,#f6f8fc 48%,#f8fafc 100%); color:var(--ink);}
.block-container {max-width:930px; padding-top:.55rem; padding-bottom:3rem;}
#MainMenu, footer, header {visibility:hidden;}
.hero {padding:17px 19px 15px; border-radius:22px; color:white; background:linear-gradient(135deg,#111827 0%,#1e3a8a 54%,#0e7490 100%); box-shadow:0 14px 34px rgba(30,58,138,.16); margin-bottom:12px;}
.hero-title {font-size:1.58rem; font-weight:900; letter-spacing:-.025em; margin:0;}
.hero-sub {font-size:.88rem; opacity:.82; margin-top:5px; line-height:1.45;}
.panel,.stock-card,.doctor-card {background:rgba(255,255,255,.98); border:1px solid var(--line); border-radius:18px; padding:15px 16px; box-shadow:0 5px 18px rgba(15,23,42,.035); margin:9px 0 13px;}
.section-title {font-size:1.18rem;font-weight:900;margin:17px 0 5px;letter-spacing:-.01em;}
.section-sub {font-size:.82rem;color:var(--muted);line-height:1.5;margin-bottom:10px;}
.stock-top {display:flex;align-items:flex-start;justify-content:space-between;gap:12px;}
.stock-name {font-size:1.18rem;font-weight:900;line-height:1.2}.stock-code {font-size:.78rem;color:var(--muted);margin-top:4px}
.price {font-weight:900;font-size:1.12rem;text-align:right}.small {font-size:.78rem;color:var(--muted);line-height:1.48}
.pill {display:inline-block;padding:4px 9px;border-radius:999px;font-size:.74rem;font-weight:800;margin:2px 4px 2px 0;}
.pill-blue {background:#dbeafe;color:#1d4ed8}.pill-green {background:#d1fae5;color:#047857}.pill-amber {background:#fef3c7;color:#b45309}.pill-red {background:#fee2e2;color:#b91c1c}.pill-violet {background:#ede9fe;color:#6d28d9}.pill-gray {background:#eef2f7;color:#475569}
.edge-wrap {display:flex;align-items:center;gap:12px;margin:13px 0 9px;}
.edge-number {font-size:2.15rem;font-weight:950;letter-spacing:-.06em;color:#1d4ed8;line-height:1}
.edge-label {font-size:.76rem;color:var(--muted);line-height:1.3}.action {font-weight:900;font-size:.94rem}
.score-grid {display:grid;grid-template-columns:repeat(6,1fr);gap:6px;margin:10px 0;}
.score-box {background:#f8fafc;border:1px solid #edf0f5;border-radius:11px;padding:8px 3px;text-align:center;}
.score-k {font-size:.66rem;color:#7b8494}.score-v {font-size:.96rem;font-weight:900;margin-top:2px}
.level-grid {display:grid;grid-template-columns:repeat(5,1fr);gap:7px;margin-top:9px;}
.level {background:#f8fafc;border-radius:11px;padding:8px 9px;border:1px solid #eef0f4;}
.level-k {font-size:.64rem;color:#778197}.level-v {font-size:.84rem;font-weight:850;margin-top:2px}
.reason {background:#f7faff;border-left:3px solid #2563eb;border-radius:9px;padding:10px 12px;margin-top:10px;font-size:.81rem;line-height:1.58;color:#344054}
.market-grid {display:grid;grid-template-columns:repeat(4,1fr);gap:7px;margin-top:10px;}
.market-box {text-align:center;background:#fff;border:1px solid var(--line);border-radius:11px;padding:9px 4px;}
.market-v {font-size:.96rem;font-weight:900}.market-k {font-size:.66rem;color:#7b8494;margin-top:2px}
.notice {font-size:.80rem;line-height:1.55;color:#5f6978;background:#fff;border:1px solid var(--line);padding:11px 13px;border-radius:12px;}
.chart-key {display:flex;gap:8px;flex-wrap:wrap;margin:2px 0 8px;font-size:.72rem;color:#475467}.chart-key span {display:inline-flex;align-items:center;gap:5px;background:#f8fafc;border:1px solid #edf0f5;border-radius:999px;padding:4px 8px}.chart-dot {width:9px;height:3px;border-radius:2px;display:inline-block}
.scan-strip {font-size:.79rem;color:#475467;background:#fff;border:1px solid var(--line);border-radius:12px;padding:9px 12px;margin:8px 0 12px;}
.stButton>button {border-radius:13px;font-weight:850;min-height:46px;font-size:.94rem;}
div[data-testid="stMetric"] {background:#fff;border:1px solid var(--line);padding:8px 10px;border-radius:12px;}
.stTabs [data-baseweb="tab-list"] {gap:.15rem;overflow-x:auto;scrollbar-width:none;border-bottom:1px solid #e7ebf2;}
.stTabs [data-baseweb="tab"] {font-size:.92rem;font-weight:800;padding:.65rem .72rem;white-space:nowrap;}
.stTabs [aria-selected="true"] {color:#dc2626;}
@media (max-width:650px){
 .block-container{padding-left:.68rem;padding-right:.68rem;padding-top:.35rem}.hero{padding:15px 14px 13px;border-radius:18px}.hero-title{font-size:1.36rem}.hero-sub{font-size:.80rem}
 .score-grid{grid-template-columns:repeat(3,1fr)}.level-grid{grid-template-columns:repeat(2,1fr)} .level-grid .level:last-child{grid-column:1 / -1}.market-grid{grid-template-columns:repeat(2,1fr)}
 .stock-name{font-size:1.10rem}.edge-number{font-size:1.95rem}.action{font-size:.86rem}.stTabs [data-baseweb="tab"] {font-size:.84rem;padding:.60rem .58rem;}
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">
  <div class="hero-title">📡 Taiwan Alpha Radar</div>
  <div class="hero-sub">V7.3 Prime｜全上市櫃 × 短中長獨立雷達 × 個股持股診斷</div>
</div>
""",
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# 2. Configuration
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Config:
    benchmark: str = "^TWII"
    history_period: str = "5y"
    min_history: int = 260
    min_price: float = 10.0
    min_avg_volume20: float = 150_000.0
    min_turnover20: float = 15_000_000.0
    batch_size: int = 80
    download_retries: int = 2
    individual_fallback_limit: int = 60
    finmind_workers: int = 3
    finmind_timeout: int = 18
    benchmark_fallback: str = "0050.TW"
    top_n: int = 3
    universe_cache_hours: int = 6
    quick_screen_period: str = "3mo"
    # Edge is a cross-sectional percentile. Thresholds therefore mean top-X% of the current investable universe.
    short_threshold_bull: float = 86.0
    short_threshold_neutral: float = 90.0
    short_threshold_bear: float = 94.0
    mid_threshold_bull: float = 84.0
    mid_threshold_neutral: float = 88.0
    mid_threshold_bear: float = 92.0
    long_threshold_bull: float = 82.0
    long_threshold_neutral: float = 86.0
    long_threshold_bear: float = 90.0
    tdcc_shadow_weight: float = 0.0  # Shadow until long PIT history is validated.

CFG = Config()
DATA_DIR = Path("data")

# The built-in universe is only a runnable fallback. Put data/universe.csv in the
# repo for a full TWSE + TPEx universe.
DEFAULT_UNIVERSE = [
    ("2330.TW", "台積電", "半導體"), ("2308.TW", "台達電", "電源散熱"),
    ("2454.TW", "聯發科", "IC設計"), ("2317.TW", "鴻海", "電子代工"),
    ("2382.TW", "廣達", "AI伺服器"), ("3231.TW", "緯創", "AI伺服器"),
    ("6669.TW", "緯穎", "AI伺服器"), ("2376.TW", "技嘉", "AI伺服器"),
    ("3017.TW", "奇鋐", "電源散熱"), ("3324.TWO", "雙鴻", "電源散熱"),
    ("2059.TW", "川湖", "伺服器零組件"), ("2383.TW", "台光電", "CCL/PCB"),
    ("2368.TW", "金像電", "CCL/PCB"), ("6274.TWO", "台燿", "CCL/PCB"),
    ("3037.TW", "欣興", "PCB/載板"), ("2449.TW", "京元電子", "半導體封測"),
    ("3711.TW", "日月光投控", "半導體封測"), ("3661.TW", "世芯-KY", "IC設計"),
    ("3443.TW", "創意", "IC設計"), ("3131.TWO", "弘塑", "半導體設備"),
    ("3583.TW", "辛耘", "半導體設備"), ("6187.TWO", "萬潤", "半導體設備"),
    ("3533.TW", "嘉澤", "伺服器零組件"), ("3034.TW", "聯詠", "IC設計"),
    ("2327.TW", "國巨", "被動元件"), ("2357.TW", "華碩", "品牌/伺服器"),
    ("2881.TW", "富邦金", "金融"), ("2882.TW", "國泰金", "金融"),
    ("2891.TW", "中信金", "金融"), ("2412.TW", "中華電", "電信"),
    ("2603.TW", "長榮", "航運"), ("2609.TW", "陽明", "航運"),
    ("2615.TW", "萬海", "航運"), ("1301.TW", "台塑", "塑化"),
    ("1303.TW", "南亞", "塑化"), ("1101.TW", "台泥", "水泥"),
]

CYCLICAL_INDUSTRIES = {"航運", "塑化", "水泥", "鋼鐵", "面板", "記憶體"}


# -----------------------------------------------------------------------------
# 4. Generic helpers
# -----------------------------------------------------------------------------

def clip100(x):
    if pd.isna(x):
        return np.nan
    return float(np.clip(x, 0, 100))


def safe_float(v, default=np.nan):
    try:
        v = float(v)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def pct_rank(s: pd.Series, higher_is_better: bool = True) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    out = pd.Series(np.nan, index=s.index, dtype=float)
    valid = x.notna()
    if valid.sum() == 1:
        out.loc[valid] = 50.0
    elif valid.sum() > 1:
        out.loc[valid] = x.loc[valid].rank(pct=True, ascending=higher_is_better) * 100
    return out


def neutral_rank(df: pd.DataFrame, col: str, higher_is_better: bool = True) -> pd.Series:
    """65% industry rank + 35% market rank; small industries fall back to market."""
    market = pct_rank(df[col], higher_is_better)
    sector_parts = []
    for _, g in df.groupby("industry", dropna=False):
        if len(g) >= 3:
            r = pct_rank(g[col], higher_is_better)
        else:
            r = market.loc[g.index]
        sector_parts.append(r)
    sector = pd.concat(sector_parts).sort_index() if sector_parts else market
    return (0.65 * sector + 0.35 * market).clip(0, 100)


def weighted_geo(row: pd.Series, weights: Dict[str, float], min_completeness: float = 0.5) -> Tuple[float, float]:
    """Weighted geometric mean. Missing factors are not invented; completeness is returned."""
    total = sum(weights.values())
    available = []
    used_w = 0.0
    for k, w in weights.items():
        v = safe_float(row.get(k, np.nan))
        if np.isfinite(v):
            available.append((max(v, 1.0), w))
            used_w += w
    completeness = used_w / total if total else 0.0
    if completeness < min_completeness or not available:
        return np.nan, completeness
    logv = sum((w / used_w) * math.log(v / 100.0) for v, w in available)
    return clip100(100.0 * math.exp(logv)), completeness


def fmt_score(x: float) -> str:
    v = safe_float(x)
    return "—" if not np.isfinite(v) else f"{v:.0f}"


def fmt_price(x: float) -> str:
    if not np.isfinite(safe_float(x)):
        return "—"
    x = float(x)
    if x >= 1000:
        return f"{x:,.0f}"
    if x >= 100:
        return f"{x:,.1f}"
    return f"{x:,.2f}"


def horizon_label(h: str) -> str:
    return {"short": "⚡ 短線 2–10日", "mid": "📈 中線 2–8週", "long": "🧭 長波段 2–9月"}.get(h, h)


def confidence_grade(v: float) -> str:
    if v >= 82: return "A"
    if v >= 68: return "B"
    return "C"


def pill_class(action: str) -> str:
    if any(k in action for k in ["可執行", "續抱", "READY"]): return "pill-green"
    if any(k in action for k in ["不追", "等待", "觀察", "保護"]): return "pill-amber"
    if any(k in action for k in ["退出", "避開", "減碼"]): return "pill-red"
    return "pill-blue"


INDUSTRY_MAP = {
    "01":"水泥", "02":"食品", "03":"塑膠", "04":"紡織", "05":"電機機械", "06":"電器電纜",
    "07":"化學生技醫療", "08":"玻璃陶瓷", "09":"造紙", "10":"鋼鐵", "11":"橡膠", "12":"汽車",
    "14":"建材營造", "15":"航運", "16":"觀光餐旅", "17":"金融", "18":"貿易百貨", "20":"其他",
    "21":"化學", "22":"生技醫療", "23":"油電燃氣", "24":"半導體", "25":"電腦週邊", "26":"光電",
    "27":"通信網路", "28":"電子零組件", "29":"電子通路", "30":"資訊服務", "31":"其他電子",
    "32":"文化創意", "33":"農業科技", "34":"電子商務", "35":"綠能環保", "36":"數位雲端",
    "37":"運動休閒", "38":"居家生活", "39":"其他新產業",
}


def _first_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    if df is None or df.empty:
        return None
    norm = {re.sub(r"\s+", "", str(c)).lower(): c for c in df.columns}
    for cand in candidates:
        key = re.sub(r"\s+", "", str(cand)).lower()
        if key in norm:
            return norm[key]
    return None


def _normalize_roster(df: pd.DataFrame, market: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["ticker","name","industry","market"])
    code_col = _first_col(df,["公司代號","SecuritiesCompanyCode","公司代碼","Code"])
    name_col = _first_col(df,["公司簡稱","CompanyAbbreviation","公司名稱","CompanyName","Name"])
    ind_col = _first_col(df,["產業別","SecuritiesIndustryCode","產業代號","Industry"])
    if not code_col or not name_col:
        return pd.DataFrame(columns=["ticker","name","industry","market"])
    out = pd.DataFrame()
    out["code"] = df[code_col].astype(str).str.strip().str.extract(r"(\d{4})", expand=False)
    out["name"] = df[name_col].astype(str).str.strip()
    if ind_col:
        raw_ind = df[ind_col].astype(str).str.strip()
        def map_ind(v):
            vv = re.sub(r"\.0$", "", str(v)).zfill(2) if re.fullmatch(r"\d+(?:\.0)?", str(v)) else str(v)
            return INDUSTRY_MAP.get(vv, vv if vv and vv.lower() != "nan" else "其他")
        out["industry"] = raw_ind.map(map_ind)
    else:
        out["industry"] = "其他"
    out = out[out["code"].str.fullmatch(r"\d{4}", na=False)].copy()
    out = out[out["name"].ne("") & ~out["name"].str.contains("存託|特別股|權證", na=False)]
    suffix = ".TW" if market == "TWSE" else ".TWO"
    out["ticker"] = out["code"] + suffix
    out["market"] = market
    return out[["ticker","name","industry","market"]].drop_duplicates("ticker")


@st.cache_data(ttl=21600, show_spinner=False)
def fetch_official_universe() -> Tuple[pd.DataFrame, dict]:
    """Build the live TWSE + TPEx company universe from official company rosters."""
    diag = {"source":"官方TWSE+TPEx", "twse":0, "tpex":0, "errors":[]}
    headers = {"User-Agent":"Mozilla/5.0 TaiwanAlphaRadar/7.3"}
    pieces=[]
    try:
        r=requests.get("https://openapi.twse.com.tw/v1/opendata/t187ap03_L",headers=headers,timeout=15)
        r.raise_for_status()
        tw=_normalize_roster(pd.DataFrame(r.json()),"TWSE")
        if not tw.empty:
            pieces.append(tw); diag["twse"]=len(tw)
        else:
            diag["errors"].append("TWSE公司名冊欄位解析失敗")
    except Exception as e:
        diag["errors"].append(f"TWSE名冊 {type(e).__name__}: {str(e)[:100]}")

    ot=pd.DataFrame()
    try:
        r=requests.get("https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O",headers=headers,timeout=15)
        if r.status_code == 200 and "json" in str(r.headers.get("content-type","")):
            ot=_normalize_roster(pd.DataFrame(r.json()),"TPEx")
    except Exception as e:
        diag["errors"].append(f"TPEx OpenAPI {type(e).__name__}: {str(e)[:90]}")
    if ot.empty:
        try:
            from io import StringIO
            r=requests.get("https://mopsfin.twse.com.tw/opendata/t187ap03_O.csv",headers=headers,timeout=15)
            r.raise_for_status()
            text=r.content.decode("utf-8-sig",errors="replace")
            ot=_normalize_roster(pd.read_csv(StringIO(text)),"TPEx")
        except Exception as e:
            diag["errors"].append(f"TPEx名冊備援 {type(e).__name__}: {str(e)[:90]}")
    if not ot.empty:
        pieces.append(ot); diag["tpex"]=len(ot)

    if pieces:
        u=pd.concat(pieces,ignore_index=True).drop_duplicates("ticker")
        if len(u) >= 300:
            diag["total"]=len(u)
            return u.sort_values(["market","ticker"]).reset_index(drop=True), diag

    diag["source"]="內建備援36檔"
    u=pd.DataFrame(DEFAULT_UNIVERSE,columns=["ticker","name","industry"])
    u["market"]=np.where(u["ticker"].str.endswith(".TWO"),"TPEx","TWSE")
    diag["total"]=len(u)
    return u,diag


def load_universe() -> Tuple[pd.DataFrame, dict]:
    # Prefer an explicit local universe only when it is clearly a full-market file.
    # This prevents an old sample/fallback universe from silently shrinking production scans.
    path = DATA_DIR / "universe.csv"
    if path.exists():
        try:
            u=pd.read_csv(path)
            if {"ticker","name","industry"}.issubset(u.columns):
                z=u.copy()
                if "market" not in z.columns:
                    z["market"]=np.where(z["ticker"].astype(str).str.endswith(".TWO"),"TPEx","TWSE")
                z=z[["ticker","name","industry","market"]].drop_duplicates("ticker")
                if len(z) >= 300:
                    return z, {"source":"data/universe.csv","total":len(z),"errors":[]}
        except Exception as e:
            logger.warning("universe.csv error: %s",e)
    return fetch_official_universe()


def read_optional_source(filename: str, uploaded_file=None) -> pd.DataFrame:
    if uploaded_file is not None:
        try:
            return pd.read_csv(uploaded_file)
        except Exception:
            return pd.DataFrame()
    path = DATA_DIR / filename
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception as e:
            logger.warning("%s read error: %s", filename, e)
    return pd.DataFrame()


# -----------------------------------------------------------------------------
# 5. Price downloader
# -----------------------------------------------------------------------------

def _finmind_token() -> str:
    """Optional FinMind token. Anonymous requests are supported for free datasets,
    but a token improves quota/reliability. Put FINMIND_TOKEN in Streamlit secrets
    or environment variables; the app also works without it.
    """
    token = os.getenv("FINMIND_TOKEN", "").strip()
    if token:
        return token
    try:
        token = str(st.secrets.get("FINMIND_TOKEN", "")).strip()
    except Exception:
        token = ""
    return token


def _period_start_date(period: str) -> str:
    now = pd.Timestamp.now().normalize()
    m = re.fullmatch(r"(\d+)y", str(period).lower())
    if m:
        years = int(m.group(1))
        return (now - pd.DateOffset(years=years, days=20)).strftime("%Y-%m-%d")
    m = re.fullmatch(r"(\d+)mo", str(period).lower())
    if m:
        months = int(m.group(1))
        return (now - pd.DateOffset(months=months, days=10)).strftime("%Y-%m-%d")
    return (now - pd.DateOffset(years=5, days=20)).strftime("%Y-%m-%d")


def _normalize_ohlcv_frame(d: pd.DataFrame) -> pd.DataFrame:
    if d is None or not isinstance(d, pd.DataFrame) or d.empty:
        return pd.DataFrame()
    d = d.copy()
    d.columns = [str(c) for c in d.columns]
    need = ["Open", "High", "Low", "Close", "Volume"]
    if not set(need).issubset(d.columns):
        return pd.DataFrame()
    d = d[need]
    for c in need:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["Close"])
    if d.empty:
        return d
    idx = pd.to_datetime(d.index, errors="coerce")
    ok = ~idx.isna()
    d = d.loc[ok].copy(); idx = idx[ok]
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    d.index = idx
    return d[~d.index.duplicated(keep="last")].sort_index()


def _finmind_fetch_one(ticker: str, period: str, token: str = "") -> Tuple[str, pd.DataFrame, str]:
    """Fetch Taiwan daily OHLCV directly from FinMind's HTTP API.

    Uses adjusted prices first so long-horizon returns are not distorted by
    ex-right/ex-dividend gaps. If adjusted data is unavailable, falls back to
    the raw TaiwanStockPrice dataset.
    """
    code = str(ticker).split(".")[0].strip()
    if not code.isdigit():
        return ticker, pd.DataFrame(), "not a Taiwan numeric ticker"
    url = "https://api.finmindtrade.com/api/v4/data"
    headers = {"User-Agent": "TaiwanAlphaRadar/7.2.2"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    base_params = {
        "data_id": code,
        "start_date": _period_start_date(period),
        "end_date": (pd.Timestamp.now().normalize() + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
    }
    last_error = ""
    for dataset in ("TaiwanStockPriceAdj", "TaiwanStockPrice"):
        try:
            params = dict(base_params); params["dataset"] = dataset
            resp = requests.get(url, params=params, headers=headers, timeout=CFG.finmind_timeout)
            if resp.status_code != 200:
                last_error = f"HTTP {resp.status_code}"
                continue
            payload = resp.json()
            rows = payload.get("data", []) if isinstance(payload, dict) else []
            if not rows:
                msg = payload.get("msg", "empty") if isinstance(payload, dict) else "empty"
                last_error = str(msg)[:120]
                continue
            x = pd.DataFrame(rows)
            if "date" not in x.columns:
                last_error = "missing date"
                continue
            # FinMind has used both Trading_Volume and trading_volume naming.
            cmap = {
                "open": "Open", "max": "High", "min": "Low", "close": "Close",
                "Trading_Volume": "Volume", "trading_volume": "Volume",
                "Trading_volume": "Volume", "volume": "Volume",
            }
            x = x.rename(columns={k:v for k,v in cmap.items() if k in x.columns})
            if not {"Open","High","Low","Close","Volume"}.issubset(x.columns):
                last_error = "missing OHLCV columns"
                continue
            x.index = pd.to_datetime(x["date"], errors="coerce")
            d = _normalize_ohlcv_frame(x[["Open","High","Low","Close","Volume"]])
            if len(d) >= 60:
                return ticker, d, dataset
            last_error = f"only {len(d)} rows"
        except Exception as e:
            last_error = f"{type(e).__name__}: {str(e)[:120]}"
    return ticker, pd.DataFrame(), last_error or "no data"


@st.cache_data(ttl=600, show_spinner=False)
def download_prices_resilient(tickers_tuple: tuple, period: str = "5y") -> Tuple[Dict[str, pd.DataFrame], dict]:
    """Full-market optimized loader: Yahoo batch -> FinMind fallback -> Yahoo single."""
    tickers=list(dict.fromkeys(str(t).strip() for t in tickers_tuple if str(t).strip()))
    result: Dict[str,pd.DataFrame]={}; source_by_ticker: Dict[str,str]={}
    diag={"requested":len(tickers),"downloaded":0,"batch_recovered":0,"individual_recovered":0,
          "finmind_recovered":0,"failed_tickers":[],"errors":[],"provider":"Yahoo + FinMind fallback",
          "source_counts":{},"source_by_ticker":source_by_ticker}

    def normalize_yahoo(raw: pd.DataFrame,ticker: str,single: bool=False)->pd.DataFrame:
        if raw is None or not isinstance(raw,pd.DataFrame) or raw.empty:
            return pd.DataFrame()
        d=pd.DataFrame()
        try:
            if isinstance(raw.columns,pd.MultiIndex):
                for level in range(raw.columns.nlevels):
                    vals=raw.columns.get_level_values(level).astype(str)
                    if ticker in set(vals):
                        d=raw.xs(ticker,axis=1,level=level,drop_level=True).copy(); break
                if d.empty and single:
                    d=raw.copy()
                    if isinstance(d.columns,pd.MultiIndex):
                        for level in range(d.columns.nlevels):
                            values=d.columns.get_level_values(level).astype(str)
                            if len(set(values))==1:
                                d.columns=d.columns.droplevel(level); break
            else:
                d=raw.copy() if single else pd.DataFrame()
            if isinstance(d.columns,pd.MultiIndex):
                for level in range(d.columns.nlevels):
                    vals=set(map(str,d.columns.get_level_values(level)))
                    if {"Open","High","Low","Close","Volume"}.issubset(vals):
                        d.columns=d.columns.get_level_values(level); break
            return _normalize_ohlcv_frame(d)
        except Exception as e:
            diag["errors"].append(f"{ticker}: Yahoo parse {type(e).__name__}: {str(e)[:90]}")
            return pd.DataFrame()

    for i in range(0,len(tickers),CFG.batch_size):
        batch=tickers[i:i+CFG.batch_size]
        raw=pd.DataFrame(); last_error=None
        for attempt in range(CFG.download_retries+1):
            try:
                kwargs=dict(tickers=batch,period=period,interval="1d",group_by="ticker",
                            auto_adjust=True,threads=True,progress=False,timeout=25)
                try: raw=yf.download(repair=False,**kwargs)
                except TypeError:
                    kwargs.pop("timeout",None); raw=yf.download(**kwargs)
                if isinstance(raw,pd.DataFrame) and not raw.empty: break
            except Exception as e:
                last_error=e
            if attempt<CFG.download_retries: time.sleep(.6*(attempt+1))
        if last_error is not None and (raw is None or raw.empty):
            diag["errors"].append(f"Yahoo batch {type(last_error).__name__}: {str(last_error)[:110]}")
        before=len(result)
        for t in batch:
            d=normalize_yahoo(raw,t,single=(len(batch)==1))
            if not d.empty:
                result[t]=d; source_by_ticker[t]="Yahoo"
        diag["batch_recovered"]+=len(result)-before

    missing=[t for t in tickers if t not in result]
    token=_finmind_token()
    finmind_targets=[t for t in missing if str(t).split(".")[0].isdigit()][:CFG.individual_fallback_limit]
    if finmind_targets:
        with ThreadPoolExecutor(max_workers=max(1,CFG.finmind_workers)) as ex:
            futs={ex.submit(_finmind_fetch_one,t,period,token):t for t in finmind_targets}
            for fut in as_completed(futs):
                t=futs[fut]
                try: _,d,src=fut.result()
                except Exception as e: d=pd.DataFrame(); src=f"{type(e).__name__}: {str(e)[:80]}"
                if not d.empty:
                    result[t]=d; source_by_ticker[t]="FinMind"; diag["finmind_recovered"]+=1
                elif "HTTP 402" not in str(src):
                    diag["errors"].append(f"{t}: FinMind {src}")

    missing2=[t for t in tickers if t not in result]
    for t in missing2[:CFG.individual_fallback_limit]:
        try:
            kwargs=dict(tickers=t,period=period,interval="1d",group_by="ticker",
                        auto_adjust=True,threads=False,progress=False,timeout=20)
            try: raw=yf.download(repair=False,**kwargs)
            except TypeError:
                kwargs.pop("timeout",None); raw=yf.download(**kwargs)
            d=normalize_yahoo(raw,t,single=True)
            if not d.empty:
                result[t]=d; source_by_ticker[t]="Yahoo-single"; diag["individual_recovered"]+=1
        except Exception as e:
            diag["errors"].append(f"{t}: Yahoo single {type(e).__name__}: {str(e)[:90]}")

    counts={}
    for src in source_by_ticker.values(): counts[src]=counts.get(src,0)+1
    diag["source_counts"]=counts; diag["downloaded"]=len(result)
    diag["failed_tickers"]=[t for t in tickers if t not in result]
    diag["errors"]=diag["errors"][:12]
    return result,diag


def download_prices(tickers_tuple: tuple, period: str = "5y") -> Dict[str, pd.DataFrame]:
    """Compatibility wrapper used by chart/doctor helpers."""
    return download_prices_resilient(tickers_tuple, period)[0]


@st.cache_data(ttl=180, show_spinner=False)
def download_one(code: str, period: str = "2y") -> Tuple[str, pd.DataFrame]:
    candidates = [code]
    if code.isdigit():
        candidates = [f"{code}.TW", f"{code}.TWO"]
    dd, _ = download_prices_resilient(tuple(candidates), period)
    for t in candidates:
        if t in dd and len(dd[t]) >= 60:
            return t, dd[t]
    return candidates[0], pd.DataFrame()


# -----------------------------------------------------------------------------
# 6. Technical / volume / behavioral engine
# -----------------------------------------------------------------------------

def enrich_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy().sort_index()
    for n in [5, 10, 20, 60, 120, 240]:
        x[f"MA{n}"] = x["Close"].rolling(n).mean()
    pc = x["Close"].shift(1)
    tr = pd.concat([(x["High"] - x["Low"]), (x["High"] - pc).abs(), (x["Low"] - pc).abs()], axis=1).max(axis=1)
    x["ATR14"] = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    x["RET1"] = x["Close"].pct_change()
    x["VOL20"] = x["Volume"].shift(1).rolling(20).median()
    x["RVOL20"] = x["Volume"] / (x["VOL20"] + 1e-9)
    x["VDU"] = x["Volume"].rolling(5).mean() / (x["Volume"].rolling(20).mean() + 1e-9)
    x["CLV"] = (x["Close"] - x["Low"]) / (x["High"] - x["Low"] + 1e-9)
    x["HIGH20_PREV"] = x["High"].shift(1).rolling(20).max()
    x["HIGH55_PREV"] = x["High"].shift(1).rolling(55).max()
    x["HIGH120_PREV"] = x["High"].shift(1).rolling(120).max()
    x["LOW20_PREV"] = x["Low"].shift(1).rolling(20).min()
    x["LOW60_PREV"] = x["Low"].shift(1).rolling(60).min()
    x["HIGHEST20"] = x["High"].rolling(20).max()
    x["HIGHEST60"] = x["High"].rolling(60).max()

    # Display-only KD / MACD. They do not drive the production Edge.
    low9 = x["Low"].rolling(9).min(); high9 = x["High"].rolling(9).max()
    rsv = (x["Close"] - low9) / (high9 - low9 + 1e-9) * 100
    x["K"] = rsv.ewm(com=2, adjust=False).mean(); x["D"] = x["K"].ewm(com=2, adjust=False).mean()
    ema12 = x["Close"].ewm(span=12, adjust=False).mean(); ema26 = x["Close"].ewm(span=26, adjust=False).mean()
    x["MACD"] = ema12 - ema26; x["SIGNAL"] = x["MACD"].ewm(span=9, adjust=False).mean(); x["HIST"] = x["MACD"] - x["SIGNAL"]
    return x


def period_return(close: pd.Series, n: int) -> float:
    if len(close) <= n: return np.nan
    a, b = close.iloc[-1], close.iloc[-n-1]
    return (a / b - 1) * 100 if np.isfinite(a) and np.isfinite(b) and b != 0 else np.nan


def max_drawdown(close: pd.Series) -> float:
    if close.empty: return np.nan
    dd = close / close.cummax() - 1
    return abs(float(dd.min())) * 100


def last_percentile(s: pd.Series, window: int = 60) -> float:
    z = pd.to_numeric(s.tail(window), errors="coerce").dropna()
    if len(z) < max(10, window // 3): return np.nan
    return float(z.rank(pct=True).iloc[-1] * 100)


def _event_indices(mask: pd.Series, cooldown: int = 10) -> List[int]:
    raw = np.where(mask.fillna(False).values)[0]
    keep: List[int] = []
    for i in raw:
        if not keep or i - keep[-1] >= cooldown:
            keep.append(int(i))
    return keep


def behavioral_memory(df: pd.DataFrame, setup_family: str, horizon: int = 20) -> Tuple[float, int, float, float]:
    """Bayesian-shrunk conditional edge for the stock's own historical setup behavior."""
    x = enrich_ohlcv(df) if "HIGH55_PREV" not in df.columns else df
    trend = (x["Close"] > x["MA20"]) & (x["MA20"] > x["MA60"])
    if setup_family == "breakout":
        mask = (x["Close"] > x["HIGH55_PREV"] * 1.001) & (x["RVOL20"] >= 1.15) & (x["CLV"] >= 0.55)
    elif setup_family == "pullback":
        mask = trend & (x["Low"] <= x["MA20"] + 0.45*x["ATR14"]) & (x["Close"] >= x["MA20"]) & (x["VDU"] <= 1.05)
    elif setup_family == "reclaim":
        mask = (x["Close"].shift(1) < x["MA20"].shift(1)) & (x["Close"] > x["MA20"]) & (x["CLV"] >= 0.50)
    else:
        price_pct = x["Close"].rolling(60).apply(lambda a: pd.Series(a).rank(pct=True).iloc[-1] if len(a) else np.nan, raw=False)
        vol_pct = x["Volume"].rolling(60).apply(lambda a: pd.Series(a).rank(pct=True).iloc[-1] if len(a) else np.nan, raw=False)
        dry = (price_pct <= .12) & (vol_pct <= .18) & (x["VDU"] <= .9)
        mask = dry.shift(1).rolling(8).max().fillna(0).astype(bool) & (x["Close"] > x["MA10"])
    idxs = _event_indices(mask, cooldown=max(7, horizon//2))
    rets = []
    maes = []
    for i in idxs:
        if i + horizon < len(x):
            entry = float(x["Close"].iloc[i])
            future = x.iloc[i+1:i+horizon+1]
            if entry > 0 and not future.empty:
                rets.append((float(future["Close"].iloc[-1]) / entry - 1) * 100)
                maes.append((float(future["Low"].min()) / entry - 1) * 100)
    n = len(rets)
    succ = sum(r > 0 for r in rets)
    # Conservative prior; tiny samples cannot become 100/0.
    posterior = (succ + 8 * 0.52) / (n + 8)
    median_ret = float(np.median(rets)) if rets else np.nan
    median_mae = float(np.median(maes)) if maes else np.nan
    return posterior * 100, n, median_ret, median_mae


def extract_raw_feature(ticker: str, df: pd.DataFrame, bench: Dict[str, float]) -> Optional[dict]:
    try:
        x = enrich_ohlcv(df)
        if len(x) < CFG.min_history:
            return None
        r = x.iloc[-1]; p = x.iloc[-2]
        required = ["Close", "MA20", "MA60", "MA120", "MA240", "ATR14"]
        if any(pd.isna(r[k]) for k in required):
            return None
        close, atr = float(r["Close"]), float(r["ATR14"])
        if close <= 0 or atr <= 0:
            return None

        avg_vol20 = float(x["Volume"].shift(1).rolling(20).mean().iloc[-1])
        avg_turn20 = float((x["Close"] * x["Volume"]).shift(1).rolling(20).mean().iloc[-1])
        if close < CFG.min_price or avg_vol20 < CFG.min_avg_volume20 or avg_turn20 < CFG.min_turnover20:
            return None

        ma5, ma10, ma20, ma60, ma120, ma240 = [float(r[f"MA{n}"]) for n in [5,10,20,60,120,240]]
        extension = (close - ma20) / atr
        ret20, ret60, ret120 = (period_return(x["Close"], n) for n in [20,60,120])
        excess20 = ret20 - bench.get("ret20", 0); excess60 = ret60 - bench.get("ret60", 0); excess120 = ret120 - bench.get("ret120", 0)
        rvol = float(r["RVOL20"]); vdu = float(r["VDU"]); clv = float(r["CLV"])
        chg = (close / float(p["Close"]) - 1) * 100
        breakout20 = close >= float(r["HIGH20_PREV"]) * 1.001
        breakout55 = close >= float(r["HIGH55_PREV"]) * 1.001
        breakout120 = close >= float(r["HIGH120_PREV"]) * 1.001
        trend_up = close > ma20 > ma60 > ma120
        recent3_low = float(x["Low"].tail(3).min())
        pullback20 = trend_up and recent3_low <= ma20 + 0.45 * atr and close >= ma20 and extension <= 1.6 and vdu <= 1.05
        reclaim20 = float(p["Close"]) < float(p["MA20"]) and close > ma20 and close > float(p["Close"])
        reclaim10 = float(p["Close"]) < float(p["MA10"]) and close > ma10 and close > float(p["Close"])

        price_pct60 = last_percentile(x["Close"], 60)
        volume_pct60 = last_percentile(x["Volume"], 60)
        atr_med60 = float(x["ATR14"].tail(60).median())
        atr_contract = atr <= atr_med60 * 0.95 if np.isfinite(atr_med60) else False
        supply_dryup_today = bool(price_pct60 <= 12 and volume_pct60 <= 18 and vdu <= 0.90 and atr_contract)
        # Historical dry-up proxy used only to decide whether a later reclaim is meaningful.
        price_roll = x["Close"].rolling(60).min()
        vol_roll = x["Volume"].rolling(60).quantile(.18)
        supply_hist = (x["Close"] <= price_roll * 1.025) & (x["Volume"] <= vol_roll) & (x["VDU"] <= .92)
        recent_supply = bool(supply_hist.tail(8).fillna(False).any())
        exhaustion_reclaim = recent_supply and close > ma10 and float(p["Close"]) <= float(p["MA10"])

        upvol = x.loc[x["RET1"] > 0, "Volume"].tail(20).sum()
        downvol = x.loc[x["RET1"] < 0, "Volume"].tail(20).sum()
        updown_ratio = float(upvol / (downvol + 1e-9))
        accumulation_proxy = float(np.clip(50 + 28*np.tanh(np.log(max(updown_ratio, 1e-6))), 0, 100))

        late_churn = bool(rvol >= 1.8 and abs(chg) <= 1.5 and clv < 0.55 and extension > 1.2)
        volume_confirm = np.clip(42 + 24 * min(rvol, 2.2) + 28 * (clv - 0.5), 0, 100)
        dryup_score = np.clip(100 - abs(vdu - 0.72) * 95, 0, 100)
        price_volume_score = 0.45 * volume_confirm + 0.35 * accumulation_proxy + 0.20 * dryup_score
        if late_churn: price_volume_score *= 0.62

        setup = "BASE"
        family = "reclaim"
        if breakout120: setup, family = "BREAKOUT_120", "breakout"
        elif breakout55: setup, family = "BREAKOUT_55", "breakout"
        elif pullback20: setup, family = "PULLBACK_20", "pullback"
        elif reclaim20: setup, family = "RECLAIM_20", "reclaim"
        elif exhaustion_reclaim: setup, family = "EXHAUSTION_RECLAIM", "exhaustion"
        elif reclaim10: setup, family = "RECLAIM_10", "reclaim"
        elif trend_up: setup, family = "TREND", "pullback"
        elif supply_dryup_today: setup, family = "SUPPLY_DRYUP_WATCH", "exhaustion"

        behavior, behavior_n, behavior_med, behavior_mae = behavioral_memory(x, family, 20)
        ma20_prev10 = float(x["MA20"].iloc[-11]); ma60_prev20 = float(x["MA60"].iloc[-21])
        slope20 = (ma20 / ma20_prev10 - 1) * 100 if ma20_prev10 else 0
        slope60 = (ma60 / ma60_prev20 - 1) * 100 if ma60_prev20 else 0
        trend_score = 0
        trend_score += 20 if close > ma20 else 0; trend_score += 15 if close > ma60 else 0
        trend_score += 20 if ma20 > ma60 else 0; trend_score += 20 if ma60 > ma120 else 0
        trend_score += 10 if ma120 > ma240 else 0; trend_score += 10 if slope20 > 0 else 0; trend_score += 5 if slope60 > 0 else 0

        abs_ext = abs(extension)
        if abs_ext <= .5: location = 95
        elif abs_ext <= 1: location = 90
        elif abs_ext <= 1.5: location = 80
        elif abs_ext <= 2: location = 65
        elif abs_ext <= 3: location = 42
        else: location = 15
        if close < ma60: location = min(location, 35)

        # Horizon-specific Entry. Short is execution-heavy; Long tolerates normal short-term extension.
        if setup.startswith("BREAKOUT"):
            short_entry = .30*location + .30*volume_confirm + .20*clv*100 + .20*behavior
            mid_entry = .25*location + .22*volume_confirm + .28*trend_score + .25*behavior
        elif setup == "PULLBACK_20":
            short_entry = .30*location + .30*dryup_score + .25*trend_score + .15*clv*100
            mid_entry = .25*location + .25*dryup_score + .35*trend_score + .15*behavior
        elif "RECLAIM" in setup:
            short_entry = .34*location + .26*price_volume_score + .25*trend_score + .15*clv*100
            mid_entry = .25*location + .22*price_volume_score + .33*trend_score + .20*behavior
        elif setup == "SUPPLY_DRYUP_WATCH":
            short_entry, mid_entry = 38, 45
        else:
            short_entry = .42*location + .33*trend_score + .25*price_volume_score
            mid_entry = .25*location + .45*trend_score + .18*price_volume_score + .12*behavior
        long_entry = .15*location + .50*trend_score + .20*behavior + .15*price_volume_score
        if extension > 2.7:
            short_entry *= .62; mid_entry *= .78; long_entry *= .92
        if late_churn:
            short_entry *= .75; mid_entry *= .84

        daily = x["RET1"].tail(60)
        downside = daily[daily < 0]
        downside_vol = float(downside.std(ddof=0) * np.sqrt(252) * 100) if len(downside) >= 5 else np.nan
        maxdd60 = max_drawdown(x["Close"].tail(60))
        prev_close = x["Close"].shift(1)
        gaps = ((x["Open"] / prev_close - 1).abs() * 100).tail(20)
        gap20 = float(gaps.max())

        return {
            "ticker": ticker, "price": close, "change_pct": chg,
            "ma5": ma5, "ma10": ma10, "ma20": ma20, "ma60": ma60, "ma120": ma120, "ma240": ma240,
            "atr": atr, "atr_pct": atr / close * 100, "extension_atr": extension,
            "ret20": ret20, "ret60": ret60, "ret120": ret120,
            "excess20": excess20, "excess60": excess60, "excess120": excess120,
            "avg_volume20": avg_vol20, "avg_turnover20": avg_turn20,
            "rvol": rvol, "vdu": vdu, "clv": clv, "price_pct60": price_pct60, "volume_pct60": volume_pct60,
            "accumulation_proxy": accumulation_proxy, "price_volume_score": clip100(price_volume_score),
            "volume_context_score": clip100(price_volume_score), "late_churn": late_churn,
            "supply_dryup": supply_dryup_today, "exhaustion_reclaim": exhaustion_reclaim,
            "breakout20": breakout20, "breakout55": breakout55, "breakout120": breakout120,
            "trend_up": trend_up, "trend_score": clip100(trend_score),
            "short_entry_score": clip100(short_entry), "mid_entry_score": clip100(mid_entry), "long_entry_score": clip100(long_entry),
            "entry_score": clip100(mid_entry),  # backward-compatible display fallback
            "setup": setup, "setup_family": family,
            "behavior_score": clip100(behavior), "behavior_n": behavior_n,
            "behavior_median20": behavior_med, "behavior_mae20": behavior_mae,
            "downside_vol60": downside_vol, "maxdd60": maxdd60, "gap20": gap20,
            "high20": float(r["HIGHEST20"]), "high60": float(r["HIGHEST60"]),
            "pivot55": float(r["HIGH55_PREV"]), "pivot120": float(r["HIGH120_PREV"]),
            "low20": float(r["LOW20_PREV"]), "low60": float(r["LOW60_PREV"]),
            "k": safe_float(r["K"]), "d": safe_float(r["D"]), "macd": safe_float(r["MACD"]), "macd_hist": safe_float(r["HIST"]),
            "hist": x,
        }
    except Exception as e:
        logger.exception("feature %s failed: %s", ticker, e)
        return None


# -----------------------------------------------------------------------------
# 7. Point-in-time fundamentals / institutional / TDCC enrichment
# -----------------------------------------------------------------------------

def point_in_time_snapshot(df: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Return one latest-available row per ticker with no look-ahead."""
    if df is None or df.empty or "ticker" not in df.columns:
        return pd.DataFrame()
    x = df.copy()
    x["ticker"] = x["ticker"].astype(str).str.strip()
    date_col = "available_from" if "available_from" in x.columns else ("as_of" if "as_of" in x.columns else None)
    if date_col is None:
        # Live convenience only. Confidence is downgraded because PIT availability cannot be audited.
        x["__pit_verified"] = False
        return x.drop_duplicates("ticker", keep="last").set_index("ticker")
    x["__available_from"] = pd.to_datetime(x[date_col], errors="coerce").dt.tz_localize(None)
    cutoff = pd.Timestamp(as_of).tz_localize(None) if pd.Timestamp(as_of).tzinfo is not None else pd.Timestamp(as_of)
    x = x[x["__available_from"].notna() & (x["__available_from"] <= cutoff)].copy()
    if x.empty:
        return pd.DataFrame()
    x["__pit_verified"] = True
    x = x.sort_values(["ticker", "__available_from"]).drop_duplicates("ticker", keep="last")
    x["__age_days"] = (cutoff.normalize() - x["__available_from"].dt.normalize()).dt.days.clip(lower=0)
    return x.set_index("ticker")


def add_external_factors(features: pd.DataFrame, fundamentals: pd.DataFrame, flows: pd.DataFrame, tdcc: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    x = features.copy()
    f = point_in_time_snapshot(fundamentals, as_of)
    fl = point_in_time_snapshot(flows, as_of)
    td = point_in_time_snapshot(tdcc, as_of)

    fcols = [
        "roic","roe","gross_margin","operating_margin","fcf_margin","debt_ratio",
        "revenue_yoy","revenue_yoy_prev","eps_yoy","eps_yoy_prev","margin_delta","pe","pb",
        "revenue_surprise","information_response","cfo_net_income","inventory_days_delta",
        "receivable_days_delta","earnings_stability_score","pe_history_percentile",
        "pb_history_percentile","peg","fcf_yield","asset_quality_score","capital_adequacy_score",
        "market_cap"
    ]
    for c in fcols:
        x[c] = pd.to_numeric(f[c], errors="coerce").reindex(x.index) if c in f.columns else np.nan
    x["fundamental_pit_verified"] = f["__pit_verified"].reindex(x.index).fillna(False) if "__pit_verified" in f.columns else False
    x["fundamental_age_days"] = pd.to_numeric(f["__age_days"], errors="coerce").reindex(x.index) if "__age_days" in f.columns else np.nan
    x["revenue_accel"] = x["revenue_yoy"] - x["revenue_yoy_prev"]
    x["eps_accel"] = x["eps_yoy"] - x["eps_yoy_prev"]
    f_required = ["roic","roe","operating_margin","fcf_margin","revenue_yoy","eps_yoy"]
    x["fundamental_completeness"] = x[f_required].notna().mean(axis=1)

    rank_high = ["roic","roe","gross_margin","operating_margin","fcf_margin","revenue_yoy","revenue_accel","eps_yoy","eps_accel","margin_delta","revenue_surprise","information_response","cfo_net_income","earnings_stability_score","fcf_yield","asset_quality_score","capital_adequacy_score"]
    for c in rank_high:
        x[c + "_rank"] = neutral_rank(x, c, True) if c in x.columns else np.nan
    x["debt_rank"] = neutral_rank(x, "debt_ratio", False)
    x["inventory_quality_rank"] = neutral_rank(x, "inventory_days_delta", False)
    x["receivable_quality_rank"] = neutral_rank(x, "receivable_days_delta", False)
    x["pe_rank"] = neutral_rank(x.assign(pe_clean=x["pe"].where(x["pe"] > 0)), "pe_clean", False)
    x["pb_rank"] = neutral_rank(x.assign(pb_clean=x["pb"].where(x["pb"] > 0)), "pb_clean", False)
    x["peg_rank"] = neutral_rank(x.assign(peg_clean=x["peg"].where(x["peg"] > 0)), "peg_clean", False)

    # Earnings quality: cash conversion plus working-capital discipline when supplied.
    eq_parts = pd.DataFrame(index=x.index)
    eq_parts["cash"] = x["cfo_net_income_rank"]
    eq_parts["inventory"] = x["inventory_quality_rank"]
    eq_parts["receivable"] = x["receivable_quality_rank"]
    x["earnings_quality_score"] = eq_parts.mean(axis=1, skipna=True).where(eq_parts.notna().sum(axis=1) >= 1)

    quality_parts = pd.DataFrame(index=x.index)
    quality_parts["roic"] = x["roic_rank"]; quality_parts["roe"] = x["roe_rank"]
    quality_parts["opm"] = x["operating_margin_rank"]; quality_parts["fcf"] = x["fcf_margin_rank"]
    quality_parts["debt"] = x["debt_rank"]; quality_parts["eq"] = x["earnings_quality_score"]
    quality_parts["stability"] = x["earnings_stability_score_rank"]
    general_quality = quality_parts.mean(axis=1, skipna=True).where(quality_parts.notna().sum(axis=1) >= 3)

    finance_mask = x["industry"].astype(str).str.contains("金融|銀行|保險", regex=True, na=False)
    finance_parts = pd.DataFrame(index=x.index)
    finance_parts["roe"] = x["roe_rank"]; finance_parts["asset"] = x["asset_quality_score_rank"]
    finance_parts["capital"] = x["capital_adequacy_score_rank"]; finance_parts["stability"] = x["earnings_stability_score_rank"]
    finance_quality = finance_parts.mean(axis=1, skipna=True).where(finance_parts.notna().sum(axis=1) >= 2)
    x["quality_score"] = general_quality
    x.loc[finance_mask & finance_quality.notna(), "quality_score"] = finance_quality.loc[finance_mask & finance_quality.notna()]

    business_parts = pd.DataFrame(index=x.index)
    business_parts["rev"] = x["revenue_yoy_rank"]; business_parts["racc"] = x["revenue_accel_rank"]
    business_parts["eps"] = x["eps_yoy_rank"]; business_parts["eacc"] = x["eps_accel_rank"]
    business_parts["margin"] = x["margin_delta_rank"]; business_parts["surprise"] = x["revenue_surprise_rank"]
    x["business_score"] = business_parts.mean(axis=1, skipna=True).where(business_parts.notna().sum(axis=1) >= 3)

    # Valuation Fit: current cross-section + own-history percentile + growth-adjusted measures when available.
    own_pe = 100 - x["pe_history_percentile"].clip(0,100)
    own_pb = 100 - x["pb_history_percentile"].clip(0,100)
    val_parts = pd.DataFrame(index=x.index)
    val_parts["pe"] = x["pe_rank"]; val_parts["pb"] = x["pb_rank"]; val_parts["own_pe"] = own_pe
    val_parts["own_pb"] = own_pb; val_parts["peg"] = x["peg_rank"]; val_parts["fcf"] = x["fcf_yield_rank"]
    x["valuation_score"] = val_parts.mean(axis=1, skipna=True).where(val_parts.notna().sum(axis=1) >= 2)

    info_cols = ["revenue_surprise_rank", "information_response_rank"]
    x["information_response_score"] = x[info_cols].mean(axis=1, skipna=True).where(x[info_cols].notna().any(axis=1))

    # TRUE institutional flow. Never backfill with price-volume proxies.
    flow_cols = ["foreign_5d_intensity","foreign_20d_intensity","trust_5d_intensity","trust_20d_intensity","foreign_ownership_delta","dealer_prop_intensity","margin_balance_delta","daytrade_ratio"]
    for c in flow_cols:
        x[c] = pd.to_numeric(fl[c], errors="coerce").reindex(x.index) if c in fl.columns else np.nan
    x["flow_pit_verified"] = fl["__pit_verified"].reindex(x.index).fillna(False) if "__pit_verified" in fl.columns else False
    x["flow_age_days"] = pd.to_numeric(fl["__age_days"], errors="coerce").reindex(x.index) if "__age_days" in fl.columns else np.nan
    flow_rank_cols = []
    for c in ["foreign_5d_intensity","foreign_20d_intensity","trust_5d_intensity","trust_20d_intensity","foreign_ownership_delta","dealer_prop_intensity"]:
        rc = c + "_rank"; x[rc] = pct_rank(x[c], True); flow_rank_cols.append(rc)
    x["true_flow_score"] = x[flow_rank_cols].mean(axis=1, skipna=True).where(x[flow_rank_cols].notna().sum(axis=1) >= 2)
    x["smart_money_score"] = x["true_flow_score"]  # UI compatibility; NaN means no real flow data.
    x["flow_data_completeness"] = x[flow_rank_cols].notna().mean(axis=1)

    # TDCC is displayed as a SHADOW factor; by default it does not change production Edge.
    for c in ["large_holder_delta","small_holder_delta"]:
        x[c] = pd.to_numeric(td[c], errors="coerce").reindex(x.index) if c in td.columns else np.nan
    x["large_holder_rank"] = pct_rank(x["large_holder_delta"], True)
    x["small_holder_inverse_rank"] = pct_rank(x["small_holder_delta"], False)
    x["tdcc_shadow_score"] = pd.concat([x["large_holder_rank"], x["small_holder_inverse_rank"]], axis=1).mean(axis=1, skipna=True).where(x[["large_holder_delta","small_holder_delta"]].notna().any(axis=1))
    x["tdcc_age_days"] = pd.to_numeric(td["__age_days"], errors="coerce").reindex(x.index) if "__age_days" in td.columns else np.nan

    return x


# -----------------------------------------------------------------------------
# 8. Cross-sectional scores, sector leadership, horizon risk, regime
# -----------------------------------------------------------------------------

def benchmark_snapshot(df: pd.DataFrame, source: str = "^TWII") -> Dict[str, float]:
    b = enrich_ohlcv(df)
    r = b.iloc[-1]
    return {"close":float(r["Close"]), "ma20":float(r["MA20"]), "ma60":float(r["MA60"]), "ma120":float(r["MA120"]),
            "ret20":period_return(b["Close"],20), "ret60":period_return(b["Close"],60), "ret120":period_return(b["Close"],120),
            "synthetic":False, "source":source}


def synthetic_benchmark(prices: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    """Cross-sectional fallback when both index and ETF benchmark are temporarily unavailable."""
    r20=[]; r60=[]; r120=[]
    for d in prices.values():
        if isinstance(d, pd.DataFrame) and len(d) >= 130 and "Close" in d.columns:
            r20.append(period_return(d["Close"],20)); r60.append(period_return(d["Close"],60)); r120.append(period_return(d["Close"],120))
    med=lambda a: float(np.nanmedian(a)) if len(a) and np.isfinite(np.nanmedian(a)) else 0.0
    return {"ret20":med(r20), "ret60":med(r60), "ret120":med(r120),
            "synthetic":True, "source":"母池中位數備援"}


def market_regime(df: pd.DataFrame, bench: Dict[str, float]) -> Dict[str, float | str]:
    breadth20 = float((df["price"] > df["ma20"]).mean() * 100)
    breadth60 = float((df["price"] > df["ma60"]).mean() * 100)
    highs = int(df["breakout20"].sum()); lows = int((df["price"] < df["low20"] * 1.001).sum())
    nh = 50 if highs + lows == 0 else highs / (highs + lows) * 100
    breadth_score = 0.55*breadth20 + 0.45*breadth60
    if bench.get("synthetic", False):
        # Degrade gracefully: breadth and new-high/new-low are still real market evidence.
        score = 0.72 * breadth_score + 0.28 * nh
        trend = np.nan
    else:
        trend = (40 if bench["close"] > bench["ma20"] else 0) + (35 if bench["ma20"] > bench["ma60"] else 0) + (25 if bench["ma60"] > bench["ma120"] else 0)
        score = 0.40 * trend + 0.40 * breadth_score + 0.20 * nh
    status = "bull" if score >= 65 else ("neutral" if score >= 45 else "bear")
    label = {"bull":"🟢 Risk-On", "neutral":"🟡 Selective", "bear":"🔴 Risk-Off"}[status]
    return {"score":round(score,1), "status":status, "label":label, "breadth20":round(breadth20,1),
            "breadth60":round(breadth60,1), "new_high_ratio":round(nh,1),
            "benchmark_source":bench.get("source","—"), "benchmark_degraded":bool(bench.get("synthetic",False))}


def add_cross_sectional_scores(x: pd.DataFrame) -> pd.DataFrame:
    z = x.copy()
    z["rs20_rank"] = pct_rank(z["excess20"], True); z["rs60_rank"] = pct_rank(z["excess60"], True); z["rs120_rank"] = pct_rank(z["excess120"], True)
    z["rs_short_score"] = (0.40*z["rs20_rank"] + 0.60*z["rs60_rank"]).clip(0,100)
    z["rs_long_score"] = (0.40*z["rs60_rank"] + 0.60*z["rs120_rank"]).clip(0,100)
    z["liquidity_score"] = pct_rank(z["avg_turnover20"], True)

    # Sector leadership uses peer price leadership, breadth, and business breadth when available.
    sector_count = z.groupby("industry")["rs_long_score"].transform("count")
    sector_rs_raw = z.groupby("industry")["rs_long_score"].transform("mean")
    sector_breadth_raw = z.groupby("industry")["trend_up"].transform("mean") * 100
    sector_rs = sector_rs_raw.where(sector_count >= 3, 50.0)
    sector_breadth = sector_breadth_raw.where(sector_count >= 3, 50.0)
    sector_price = (0.65 * sector_rs + 0.35 * sector_breadth).clip(0,100)
    if "business_score" in z.columns and z["business_score"].notna().any():
        sb = z.groupby("industry")["business_score"].transform("mean")
        z["sector_score"] = (0.72*sector_price + 0.28*sb.fillna(sector_price)).clip(0,100)
    else:
        z["sector_score"] = sector_price

    # Stock personality is descriptive; it changes interpretation, not the raw direction of a signal.
    vol_rank = pct_rank(z["atr_pct"], True).fillna(50)
    turn_rank = pct_rank(z["avg_turnover20"], True).fillna(50)
    z["personality"] = np.select(
        [(vol_rank >= 75) & (turn_rank >= 65), (z["trend_score"] >= 75) & (vol_rank < 75), (vol_rank <= 35)],
        ["High-Beta Momentum", "Institutional Trend", "Defensive/Low-Vol"],
        default="Balanced/Cyclical"
    )

    atr_risk = pct_rank(z["atr_pct"], True).fillna(50)
    dd_risk = pct_rank(z["maxdd60"], True).fillna(50)
    gap_risk = pct_rank(z["gap20"], True).fillna(50)
    ext_risk = pd.Series(np.select([z["extension_atr"].abs()<=1, z["extension_atr"].abs()<=2, z["extension_atr"].abs()<=3], [15,35,62], default=90), index=z.index, dtype=float)
    turnover_heat = pct_rank(z["avg_turnover20"], True).fillna(50)
    rvol_heat = pct_rank(z["rvol"], True).fillna(50)
    crowd = (0.42*rvol_heat + 0.28*turnover_heat + 0.30*ext_risk).clip(0,100)
    if "margin_balance_delta" in z.columns and z["margin_balance_delta"].notna().any():
        margin_risk = pct_rank(z["margin_balance_delta"], True)
        crowd = (0.72*crowd + 0.28*margin_risk.fillna(crowd)).clip(0,100)
    if "daytrade_ratio" in z.columns and z["daytrade_ratio"].notna().any():
        day_risk = pct_rank(z["daytrade_ratio"], True)
        crowd = (0.84*crowd + 0.16*day_risk.fillna(crowd)).clip(0,100)
    z["crowding_risk"] = crowd

    illiq = (100-z["liquidity_score"].fillna(50)).clip(0,100)
    z["short_risk_score"] = (0.25*atr_risk + 0.20*gap_risk + 0.20*crowd + 0.20*ext_risk + 0.15*illiq).clip(0,100)
    z["mid_risk_score"] = (0.20*atr_risk + 0.25*dd_risk + 0.12*gap_risk + 0.15*crowd + 0.13*ext_risk + 0.15*illiq).clip(0,100)
    z["long_risk_score"] = (0.18*atr_risk + 0.30*dd_risk + 0.12*gap_risk + 0.10*crowd + 0.08*ext_risk + 0.12*illiq + 0.10*(100-z["quality_score"].fillna(50))).clip(0,100)
    for h in ["short","mid","long"]:
        z[f"{h}_safety_score"] = (100-z[f"{h}_risk_score"]).clip(0,100)
    z["risk_score"] = z["mid_risk_score"]; z["safety_score"] = z["mid_safety_score"]

    # Entry saturation answers "too late to chase?". It is NOT the same as thesis deterioration.
    ret60_heat = pct_rank(z["ret60"], True).fillna(50)
    extension_heat = pct_rank(z["extension_atr"], True).fillna(50)
    churn = z["late_churn"].astype(float) * 100
    z["entry_saturation"] = (0.42*ret60_heat + 0.25*extension_heat + 0.13*rvol_heat + 0.20*churn).clip(0,100)
    if "information_response_score" in z.columns:
        response = z["information_response_score"]
        mask = response.notna()
        z.loc[mask, "entry_saturation"] = (0.82*z.loc[mask,"entry_saturation"] + 0.18*(100-response.loc[mask])).clip(0,100)
    z["saturation_score"] = z["entry_saturation"]  # UI compatibility

    # Thesis deterioration focuses on weakening business/flow/long-price evidence, not the holder's cost basis.
    det = pd.DataFrame(index=z.index)
    det["rev"] = 100-z["revenue_accel_rank"] if "revenue_accel_rank" in z else np.nan
    det["eps"] = 100-z["eps_accel_rank"] if "eps_accel_rank" in z else np.nan
    det["margin"] = 100-z["margin_delta_rank"] if "margin_delta_rank" in z else np.nan
    det["flow"] = 100-z["true_flow_score"] if "true_flow_score" in z else np.nan
    det["trend"] = 100-z["rs_long_score"]
    z["thesis_deterioration"] = det.mean(axis=1, skipna=True).fillna(50).clip(0,100)

    return z


# -----------------------------------------------------------------------------
# 9. V7.2.1 Gated Soft Mixture-of-Experts
# -----------------------------------------------------------------------------

def _gate(v: float, floor: float = .05) -> float:
    return float(np.clip(v, floor, 1.0))


def soft_mixture(scores: Dict[str,float], gates: Dict[str,float]) -> Tuple[float,str,float,float]:
    valid = [(k, safe_float(scores.get(k)), max(0.0, safe_float(gates.get(k),0))) for k in scores]
    valid = [(k,s,g) for k,s,g in valid if np.isfinite(s) and g > 0]
    if not valid: return np.nan, "—", 0.0, 0.0
    weights = np.array([g**1.35 for _,_,g in valid], dtype=float)
    weights = weights / weights.sum()
    vals = np.array([s for _,s,_ in valid], dtype=float)
    mixed = float(np.sum(weights*vals))
    dominant = valid[int(np.argmax(weights))][0]
    agreement = float(np.clip(100 - np.std(vals)*2.2, 0, 100)) if len(vals) > 1 else 70.0
    completeness = float(min(1.0, sum(g for _,_,g in valid) / max(1,len(scores))))
    return clip100(mixed), dominant, agreement, completeness


def expert_scores_and_gates(row: pd.Series) -> Tuple[Dict[str,float],Dict[str,float],Dict[str,float]]:
    scores: Dict[str,float] = {}; gates: Dict[str,float] = {}; comps: Dict[str,float] = {}
    fcomp = safe_float(row.get("fundamental_completeness"),0)
    true_flow = safe_float(row.get("true_flow_score"))
    flow_available = np.isfinite(true_flow)
    setup = str(row.get("setup","BASE"))

    def wg(name: str, weights: Dict[str,float], minc: float):
        score, comp = weighted_geo(row, weights, minc)
        scores[name] = score; comps[name] = comp

    # SHORT: price/volume and true flow are separate inputs.
    wg("S_Momentum", {"rs_short_score":.28,"true_flow_score":.20,"price_volume_score":.18,"behavior_score":.14,"sector_score":.20}, .55)
    gates["S_Momentum"] = _gate(.18 + .35*float(setup.startswith("BREAKOUT")) + .20*safe_float(row.get("trend_score"),0)/100 + .27*safe_float(row.get("rs_short_score"),0)/100)
    wg("S_PullbackReclaim", {"trend_score":.30,"true_flow_score":.18,"price_volume_score":.17,"behavior_score":.15,"sector_score":.20}, .52)
    gates["S_PullbackReclaim"] = _gate(.12 + .58*float(setup in ["PULLBACK_20","RECLAIM_10","RECLAIM_20"]) + .30*safe_float(row.get("trend_score"),0)/100)
    wg("S_Exhaustion", {"price_volume_score":.30,"behavior_score":.25,"rs_short_score":.20,"sector_score":.15,"true_flow_score":.10}, .45)
    gates["S_Exhaustion"] = _gate(.05 + .80*float(setup=="EXHAUSTION_RECLAIM") + .10*float(setup=="SUPPLY_DRYUP_WATCH"))

    # MID experts.
    wg("M_FundamentalMomentum", {"business_score":.35,"true_flow_score":.18,"sector_score":.22,"rs_long_score":.15,"price_volume_score":.10}, .60)
    gates["M_FundamentalMomentum"] = _gate(.08 + .52*min(1,fcomp/.75) + .20*safe_float(row.get("business_score"),0)/100 + .20*safe_float(row.get("sector_score"),0)/100)
    wg("M_SectorRepricing", {"sector_score":.30,"rs_long_score":.25,"true_flow_score":.15,"price_volume_score":.20,"trend_score":.10}, .58)
    gates["M_SectorRepricing"] = _gate(.10 + .38*safe_float(row.get("sector_score"),0)/100 + .30*safe_float(row.get("rs_long_score"),0)/100 + .22*safe_float(row.get("price_volume_score"),0)/100)
    wg("M_Recovery", {"business_score":.22,"true_flow_score":.18,"sector_score":.18,"rs_short_score":.16,"price_volume_score":.26}, .48)
    gates["M_Recovery"] = _gate(.08 + .62*float(setup in ["RECLAIM_10","RECLAIM_20","EXHAUSTION_RECLAIM","PULLBACK_20"]) + .30*safe_float(row.get("business_score"),40)/100)

    # LONG experts.
    wg("L_Compounder", {"quality_score":.30,"business_score":.30,"sector_score":.15,"valuation_score":.15,"rs_long_score":.10}, .68)
    gates["L_Compounder"] = _gate(.05 + .48*min(1,fcomp/.80) + .24*safe_float(row.get("quality_score"),0)/100 + .23*safe_float(row.get("business_score"),0)/100)
    wg("L_QualityCore", {"quality_score":.45,"valuation_score":.25,"sector_score":.10,"rs_long_score":.10,"earnings_quality_score":.10}, .58)
    gates["L_QualityCore"] = _gate(.08 + .52*min(1,fcomp/.70) + .40*safe_float(row.get("quality_score"),0)/100)
    wg("L_EmergingLeader", {"sector_score":.30,"rs_long_score":.30,"true_flow_score":.12,"price_volume_score":.13,"quality_score":.15}, .52)
    gates["L_EmergingLeader"] = _gate(.10 + .32*safe_float(row.get("sector_score"),0)/100 + .32*safe_float(row.get("rs_long_score"),0)/100 + .26*safe_float(row.get("trend_score"),0)/100)
    wg("L_CyclicalRotation", {"business_score":.25,"sector_score":.30,"true_flow_score":.15,"rs_long_score":.15,"valuation_score":.15}, .48)
    gates["L_CyclicalRotation"] = _gate(.75 if row.get("industry") in CYCLICAL_INDUSTRIES else .02)

    # If no TRUE flow is available, experts can still work through other factors, but confidence later is reduced.
    if not flow_available:
        for k in gates:
            gates[k] *= .94
    return scores, gates, comps


def _saturation_penalty(raw: float, sat: float, horizon: str) -> float:
    if not np.isfinite(raw): return np.nan
    if not np.isfinite(sat): return raw
    strength = {"short":.34,"mid":.22,"long":.08}[horizon]
    penalty = np.clip((sat-70)/30,0,1) * strength
    return clip100(raw*(1-penalty))


def add_horizon_scores(df: pd.DataFrame, regime: dict) -> pd.DataFrame:
    z = df.copy()
    rows = []
    for _, row in z.iterrows():
        scores, gates, comps = expert_scores_and_gates(row)
        out = {}
        for h,prefix in [("short","S_"),("mid","M_"),("long","L_")]:
            hs = {k:v for k,v in scores.items() if k.startswith(prefix)}
            hg = {k:gates.get(k,0) for k in hs}
            opp, expert, agreement, expert_comp = soft_mixture(hs,hg)
            entry = safe_float(row.get(f"{h}_entry_score"))
            safety = safe_float(row.get(f"{h}_safety_score"))
            # New-position score: Opportunity + horizon Entry + Safety. Saturation only affects NEW entry.
            temp = pd.Series({"opp":opp,"entry":entry,"safe":safety})
            weights = {"opp":{"short":.52,"mid":.60,"long":.72}[h],
                       "entry":{"short":.30,"mid":.23,"long":.10}[h],
                       "safe":{"short":.18,"mid":.17,"long":.18}[h]}
            new_raw,_ = weighted_geo(temp,weights,.70)
            new_raw = _saturation_penalty(new_raw, safe_float(row.get("entry_saturation")), h)
            # Hold score ignores Entry saturation and instead checks thesis health.
            thesis_health = 100-safe_float(row.get("thesis_deterioration"),50)
            htemp = pd.Series({"opp":opp,"safe":safety,"thesis":thesis_health})
            hold_raw,_ = weighted_geo(htemp,{"opp":.67,"safe":.15,"thesis":.18},.65)
            out[f"{h}_opportunity_raw"] = opp
            out[f"{h}_new_raw"] = new_raw
            out[f"{h}_hold_raw"] = hold_raw
            out[f"{h}_expert"] = expert
            out[f"{h}_agreement"] = agreement
            out[f"{h}_expert_completeness"] = expert_comp
        rows.append(out)
    rz = pd.DataFrame(rows,index=z.index)
    z = pd.concat([z,rz],axis=1)

    # Final Edge is an investable-universe percentile, not a probability.
    for h in ["short","mid","long"]:
        z[f"{h}_edge"] = pct_rank(z[f"{h}_new_raw"], True)
        z[f"{h}_hold_edge"] = pct_rank(z[f"{h}_hold_raw"], True)

    # Data confidence. This is intentionally labeled Data Confidence, not probability of success.
    behavior_q = (z["behavior_n"] / 10).clip(0,1)
    flow_q = np.where(z["flow_data_completeness"] >= .33, .95, .35)
    pit_flow_q = np.where(z["flow_pit_verified"].fillna(False),1.0,.75)
    pit_fund_q = np.where(z["fundamental_pit_verified"].fillna(False),1.0,.75)
    fund_q = z["fundamental_completeness"].fillna(0).clip(0,1)
    for h in ["short","mid","long"]:
        agree = z[f"{h}_agreement"].fillna(50)/100
        comp = z[f"{h}_expert_completeness"].fillna(0).clip(0,1)
        if h == "short":
            conf = .20 + .18*behavior_q + .25*flow_q*pit_flow_q + .22*agree + .15*comp
        elif h == "mid":
            conf = .16 + .12*behavior_q + .20*flow_q*pit_flow_q + .18*agree + .16*comp + .18*fund_q*pit_fund_q
        else:
            conf = .14 + .08*behavior_q + .10*flow_q*pit_flow_q + .16*agree + .15*comp + .37*fund_q*pit_fund_q
        z[f"{h}_conf"] = (100*conf).clip(0,100)

    # Cross-horizon meta selector. The three horizons have already been normalized to percentiles.
    best_h=[]; prime_raw=[]
    for _,r in z.iterrows():
        cand={h:safe_float(r[f"{h}_edge"])*(.72+.28*safe_float(r[f"{h}_conf"])/100) for h in ["short","mid","long"]}
        cand={k:v for k,v in cand.items() if np.isfinite(v)}
        h=max(cand,key=cand.get) if cand else "mid"
        best_h.append(h); prime_raw.append(cand.get(h,np.nan))
    z["best_horizon"] = best_h
    z["prime_raw"] = prime_raw
    z["prime_score"] = pct_rank(z["prime_raw"], True)
    return z


# -----------------------------------------------------------------------------
# 10. Entry / exit / phase / action engine
# -----------------------------------------------------------------------------

def price_phase(row: pd.Series, horizon: str) -> str:
    ext = safe_float(row.get("extension_atr"),0); trend = safe_float(row.get("trend_score"),0)
    if row.get("setup") == "SUPPLY_DRYUP_WATCH": return "P0 供給乾涸觀察"
    if row.get("setup") == "EXHAUSTION_RECLAIM": return "P1 衰竭後Reclaim"
    if str(row.get("setup","")).startswith("BREAKOUT"): return "P2 突破確認"
    if trend >= 75 and ext < 1.8: return "P3 主升趨勢"
    if trend >= 70 and ext >= 1.8: return "P4 趨勢延伸"
    if safe_float(row.get("price")) < safe_float(row.get("ma60")): return "P5 結構轉弱"
    return "P0/P1 築底整理"


def levels(row: pd.Series, horizon: str) -> dict:
    p = safe_float(row["price"]); atr = max(safe_float(row["atr"],0), p*.005); setup = str(row.get("setup","BASE"))
    ma10,ma20,ma60 = (safe_float(row.get(k),p) for k in ["ma10","ma20","ma60"])
    pivot = safe_float(row.get("pivot120" if setup=="BREAKOUT_120" else "pivot55"), p)
    low20, low60 = safe_float(row.get("low20"),p-2*atr), safe_float(row.get("low60"),p-3*atr)
    if setup.startswith("BREAKOUT"):
        ideal_lo, ideal_hi = pivot, pivot + .45*atr; trigger = pivot*1.001; base_invalid = pivot-.85*atr
    elif setup == "PULLBACK_20":
        ideal_lo, ideal_hi = ma20-.30*atr, ma20+.45*atr; trigger = max(ma10,p); base_invalid = low20-.25*atr
    elif "RECLAIM" in setup:
        support = ma20 if setup=="RECLAIM_20" else ma10
        ideal_lo, ideal_hi = support, support+.40*atr; trigger=max(p,ma10); base_invalid=min(low20,support-.65*atr)
    else:
        ideal_lo, ideal_hi = ma20-.25*atr, ma20+.55*atr; trigger=max(p,ma10); base_invalid=low20-.25*atr
    if horizon == "short":
        do_not=trigger+1.35*atr; invalid=max(base_invalid,low20-.20*atr); trailing=max(ma10-.55*atr,safe_float(row.get("high20"),p)-2.5*atr)
    elif horizon == "mid":
        do_not=trigger+2.0*atr; invalid=min(base_invalid,ma20-1.0*atr); trailing=max(ma20-1.0*atr,safe_float(row.get("high20"),p)-3.0*atr)
    else:
        do_not=ma20+3.0*atr; invalid=min(low60-.35*atr,ma60-1.15*atr); trailing=max(ma60-1.1*atr,safe_float(row.get("high60"),p)-4.0*atr)
    ideal_lo=max(.01,ideal_lo); ideal_hi=max(ideal_lo,ideal_hi); invalid=max(.01,min(invalid,p)); trailing=max(invalid,min(trailing,p*1.05))
    add_on=trigger+.25*atr if p<=do_not else np.nan
    return {"ideal_lo":ideal_lo,"ideal_hi":ideal_hi,"trigger":trigger,"do_not_chase":do_not,"add_on":add_on,"trailing":trailing,"invalidation":invalid}


def threshold_for(h: str, regime_status: str) -> float:
    if h=="short": return {"bull":CFG.short_threshold_bull,"neutral":CFG.short_threshold_neutral,"bear":CFG.short_threshold_bear}[regime_status]
    if h=="mid": return {"bull":CFG.mid_threshold_bull,"neutral":CFG.mid_threshold_neutral,"bear":CFG.mid_threshold_bear}[regime_status]
    return {"bull":CFG.long_threshold_bull,"neutral":CFG.long_threshold_neutral,"bear":CFG.long_threshold_bear}[regime_status]


def new_position_action(row: pd.Series, horizon: str, regime_status: str) -> str:
    edge=safe_float(row[f"{horizon}_edge"],0); entry=safe_float(row[f"{horizon}_entry_score"],0); sat=safe_float(row["entry_saturation"],50)
    lev=levels(row,horizon); p=safe_float(row["price"]); threshold=threshold_for(horizon,regime_status)
    if row.get("setup")=="SUPPLY_DRYUP_WATCH": return "🟡 供給乾涸，先等Reclaim"
    if p > lev["do_not_chase"] or sat >= (88 if horizon=="long" else 82): return "🟠 Edge佳但位階偏熱，不追價"
    if regime_status=="bear" and horizon=="short" and str(row.get("setup","")).startswith("BREAKOUT"): return "🟡 Risk-Off，等Reclaim而非追突破"
    if edge >= threshold and entry >= 78: return "🟢 可執行 Setup"
    if edge >= threshold and entry >= 64: return "🔵 可分批／等待小回測"
    if edge >= threshold-5: return "🟡 等待確認"
    return "⚪ 目前不列優先"


def holding_action(row: pd.Series, horizon: str, regime_status: str, cost: Optional[float]=None) -> str:
    p=safe_float(row["price"]); lev=levels(row,horizon); hold=safe_float(row[f"{horizon}_hold_edge"],0)
    risk=safe_float(row[f"{horizon}_risk_score"],50); sat=safe_float(row["entry_saturation"],50); det=safe_float(row["thesis_deterioration"],50)
    profit=(p/cost-1)*100 if cost and cost>0 else np.nan
    if p <= lev["invalidation"]: return "🔴 結構失效，退出優先"
    if det >= 78 or hold < 35: return "🟠 Thesis顯著惡化，考慮大幅減碼／退出"
    if risk >= 86: return "🟠 風險升高，降低部位優先"
    if np.isfinite(profit) and profit>12 and (p <= lev["trailing"] or sat>=86): return "🟡 續抱但上移保護／分批減碼"
    if sat>=82 and det<60: return "🔒 新資金不追；既有趨勢仍可續抱＋保護"
    if hold >= max(70,threshold_for(horizon,regime_status)-8): return "🟢 續抱，主要投資理由仍成立"
    return "🔵 續抱觀察，不建議再加碼"


def reason_text(row: pd.Series, horizon: str) -> str:
    reasons=[]
    if safe_float(row.get("business_score"),0)>=78: reasons.append("營收/EPS/利潤動能位居前段")
    if safe_float(row.get("quality_score"),0)>=78: reasons.append("品質與資本效率佳")
    if np.isfinite(safe_float(row.get("true_flow_score"))) and safe_float(row.get("true_flow_score"))>=75: reasons.append("真實法人籌碼偏正向")
    if safe_float(row.get("price_volume_score"),0)>=75: reasons.append("量價累積/承接結構偏強")
    if safe_float(row.get("sector_score"),0)>=75: reasons.append("所屬產業相對強勢")
    if safe_float(row.get("rs_long_score"),0)>=75: reasons.append("相對強度領先大盤")
    if str(row.get("setup","")).startswith("BREAKOUT"): reasons.append("價格完成突破確認")
    elif row.get("setup")=="PULLBACK_20": reasons.append("強勢趨勢中的健康量縮回測")
    elif "RECLAIM" in str(row.get("setup","")): reasons.append("洗盤後重新站回關鍵結構")
    elif row.get("setup")=="SUPPLY_DRYUP_WATCH": reasons.append("新低低量僅代表供給乾涸觀察，尚非買點")
    if safe_float(row.get("entry_saturation"),0)>=80: reasons.append("但新進場位階已有偏熱風險")
    if safe_float(row.get("thesis_deterioration"),0)>=70: reasons.append("且中長期投資理由正在弱化")
    if not reasons: reasons.append("價格、產業與風險條件綜合排序進入前段")
    return "；".join(reasons[:4])


# -----------------------------------------------------------------------------
# 11. Scan orchestration
# -----------------------------------------------------------------------------

@st.cache_data(ttl=900, show_spinner=False)
def liquidity_prefilter(univ_records: tuple) -> Tuple[pd.DataFrame, dict]:
    """Stage 1: full official universe -> investable/liquid universe."""
    u=pd.DataFrame(list(univ_records),columns=["ticker","name","industry","market"])
    prices,diag=download_prices_resilient(tuple(u["ticker"].astype(str)),CFG.quick_screen_period)
    rows=[]
    for _,r in u.iterrows():
        t=str(r["ticker"]); d=prices.get(t)
        if d is None or len(d)<25: continue
        z=d.tail(25)
        close=safe_float(z["Close"].iloc[-1]); avgv=safe_float(z["Volume"].tail(20).mean())
        avgt=safe_float((z["Close"]*z["Volume"]).tail(20).mean())
        if close>=CFG.min_price and avgv>=CFG.min_avg_volume20 and avgt>=CFG.min_turnover20:
            rows.append((t,r["name"],r["industry"],r["market"],avgt))
    out=pd.DataFrame(rows,columns=["ticker","name","industry","market","screen_turnover"])
    if not out.empty: out=out.sort_values("screen_turnover",ascending=False).reset_index(drop=True)
    diag["liquidity_pass"]=len(out)
    return out,diag


def run_full_market_scan(univ: pd.DataFrame,fund: pd.DataFrame,flow: pd.DataFrame,tdcc: pd.DataFrame,as_of: Optional[pd.Timestamp]=None):
    base=univ[["ticker","name","industry","market"]].drop_duplicates("ticker").copy()
    records=tuple(map(tuple,base.itertuples(index=False,name=None)))
    screened,screen_diag=liquidity_prefilter(records)
    if screened.empty:
        diag={"requested":len(base),"universe_total":len(base),"liquidity_pass":0,"stage":"quick_screen",
              "download_errors":screen_diag.get("errors",[]),"failed_tickers":screen_diag.get("failed_tickers",[]),
              "source_counts":screen_diag.get("source_counts",{}),"price_downloaded":screen_diag.get("downloaded",0),"feature_valid":0}
        return pd.DataFrame(),{}, {},diag
    scan,reg,prices,diag=do_scan(screened[["ticker","name","industry","market"]],fund,flow,tdcc,as_of)
    diag["universe_total"]=len(base); diag["liquidity_pass"]=len(screened)
    diag["quick_downloaded"]=screen_diag.get("downloaded",0); diag["quick_source_counts"]=screen_diag.get("source_counts",{})
    return scan,reg,prices,diag


def do_scan(univ: pd.DataFrame, fund: pd.DataFrame, flow: pd.DataFrame, tdcc: pd.DataFrame, as_of: Optional[pd.Timestamp]=None):
    as_of = pd.Timestamp(as_of or datetime.now()).tz_localize(None)
    tickers = univ["ticker"].astype(str).tolist()
    prices, price_diag = download_prices_resilient(tuple(tickers), CFG.history_period)
    diag = {
        "requested": len(tickers), "price_downloaded": len(prices), "feature_valid": 0,
        "benchmark_source": "—", "benchmark_degraded": False,
        "failed_tickers": price_diag.get("failed_tickers", []),
        "download_errors": price_diag.get("errors", []),
        "individual_recovered": price_diag.get("individual_recovered", 0),
        "source_counts": price_diag.get("source_counts", {}),
        "finmind_recovered": price_diag.get("finmind_recovered", 0),
    }
    if not prices:
        diag["stage"] = "price_download"
        return pd.DataFrame(), {}, prices, diag

    # Preferred benchmark: TAIEX.  If unavailable, 0050 is a useful market proxy.
    bench = None
    bench_dict, bench_diag = download_prices_resilient((CFG.benchmark,), CFG.history_period)
    if CFG.benchmark in bench_dict and len(bench_dict[CFG.benchmark]) >= 130:
        bench = benchmark_snapshot(bench_dict[CFG.benchmark], CFG.benchmark)
        diag["benchmark_source"] = CFG.benchmark
    else:
        proxy_dict, proxy_diag = download_prices_resilient((CFG.benchmark_fallback,), CFG.history_period)
        if CFG.benchmark_fallback in proxy_dict and len(proxy_dict[CFG.benchmark_fallback]) >= 130:
            bench = benchmark_snapshot(proxy_dict[CFG.benchmark_fallback], CFG.benchmark_fallback + " proxy")
            diag["benchmark_source"] = CFG.benchmark_fallback + " proxy"
            diag["benchmark_degraded"] = True
            diag["download_errors"].extend(bench_diag.get("errors", [])[:2])
        else:
            bench = synthetic_benchmark(prices)
            diag["benchmark_source"] = bench["source"]
            diag["benchmark_degraded"] = True
            diag["download_errors"].extend(bench_diag.get("errors", [])[:2])
            diag["download_errors"].extend(proxy_diag.get("errors", [])[:2])

    rows=[]
    for t in tickers:
        if t in prices:
            r=extract_raw_feature(t, prices[t], bench)
            if r:
                rows.append(r)
    diag["feature_valid"] = len(rows)
    if not rows:
        diag["stage"] = "feature_gate"
        return pd.DataFrame(), {}, prices, diag

    x=pd.DataFrame(rows).set_index("ticker").join(univ.set_index("ticker")[[c for c in ["name","industry","market"] if c in univ.columns]],how="left")
    x=add_external_factors(x,fund,flow,tdcc,as_of)
    x=add_cross_sectional_scores(x)
    reg=market_regime(x,bench)
    x=add_horizon_scores(x,reg)
    diag["stage"] = "ok"
    diag["benchmark_source"] = reg.get("benchmark_source",diag["benchmark_source"])
    diag["benchmark_degraded"] = reg.get("benchmark_degraded",diag["benchmark_degraded"])
    diag["download_errors"] = diag["download_errors"][:8]
    return x,reg,prices,diag


# -----------------------------------------------------------------------------
# 12. UI helpers
# -----------------------------------------------------------------------------

EXPERT_LABELS = {
    "S_Momentum":"短線動能", "S_PullbackReclaim":"拉回/Reclaim", "S_Exhaustion":"衰竭反轉",
    "M_FundamentalMomentum":"基本面加速", "M_SectorRepricing":"產業重新定價", "M_Recovery":"轉折復甦",
    "L_Compounder":"成長複利", "L_QualityCore":"核心品質", "L_EmergingLeader":"新興領先", "L_CyclicalRotation":"循環輪動",
}
SETUP_LABELS = {
    "BREAKOUT_120":"120日突破", "BREAKOUT_55":"55日突破", "PULLBACK_20":"20MA健康回測",
    "RECLAIM_20":"20MA Reclaim", "RECLAIM_10":"10MA Reclaim", "EXHAUSTION_RECLAIM":"低量衰竭後Reclaim",
    "SUPPLY_DRYUP_WATCH":"新低低量觀察", "TREND":"趨勢延續", "BASE":"整理/等待",
}


def render_market(reg: dict, diag: dict, fund: pd.DataFrame, flow: pd.DataFrame):
    if not reg: return
    full_n=int(diag.get("universe_total",diag.get("requested",0)))
    liquid_n=int(diag.get("liquidity_pass",diag.get("requested",0)))
    model_n=int(diag.get("feature_valid",0))
    fund_state="✅" if not fund.empty else "○"
    flow_state="✅" if not flow.empty else "○"
    bench_note=(" · 備援" if reg.get("benchmark_degraded") else "")
    st.markdown(f"""<div class='panel'>
      <div style='display:flex;justify-content:space-between;align-items:center;gap:12px'>
        <div><b>市場環境 {reg['label']}</b><div class='small'>Regime {reg['score']:.0f}/100 · Benchmark {reg.get('benchmark_source','—')}{bench_note}</div></div>
        <div class='small' style='text-align:right'>基本面 {fund_state}　法人 {flow_state}</div>
      </div>
      <div class='market-grid'>
        <div class='market-box'><div class='market-v'>{full_n:,}</div><div class='market-k'>全市場母池</div></div>
        <div class='market-box'><div class='market-v'>{liquid_n:,}</div><div class='market-k'>流動性通過</div></div>
        <div class='market-box'><div class='market-v'>{model_n:,}</div><div class='market-k'>有效模型</div></div>
        <div class='market-box'><div class='market-v'>{reg['breadth20']:.0f}%</div><div class='market-k'>站上20MA</div></div>
      </div></div>""",unsafe_allow_html=True)


def render_system_status(diag: dict, universe_diag: dict):
    if not diag: return
    with st.sidebar.expander("系統狀態",expanded=False):
        st.caption(f"母池：{universe_diag.get('source','—')} · {int(universe_diag.get('total',0)):,}檔")
        st.caption(f"流動性初篩：{int(diag.get('liquidity_pass',0)):,}檔 · 有效模型：{int(diag.get('feature_valid',0)):,}檔")
        src=diag.get("source_counts",{}) or {}
        if src: st.caption("深度行情："+"、".join(f"{k} {v}" for k,v in src.items()))
        errs=[e for e in diag.get("download_errors",[]) if e]
        if errs:
            st.caption("資料來源有部分備援/失敗（不影響已成功股票）：")
            st.code("\n".join(errs[:4]),language=None)


def _candidate_frame(scan: pd.DataFrame,h: str,reg_status: str,n: int=3)->pd.DataFrame:
    if scan is None or scan.empty: return pd.DataFrame()
    threshold=threshold_for(h,reg_status)
    c=scan[(scan[f"{h}_edge"]>=threshold)&(scan[f"{h}_conf"]>=45)].copy()
    if h=="short" and reg_status=="bear":
        c=c[~c["setup"].isin(["BREAKOUT_55","BREAKOUT_120"])]
    return c.sort_values([f"{h}_edge",f"{h}_conf"],ascending=False).head(n)


def _prime_frame(scan: pd.DataFrame,n: int=3,max_per_industry: int=2)->pd.DataFrame:
    if scan is None or scan.empty: return pd.DataFrame()
    ranked=scan.sort_values(["prime_score","prime_raw"],ascending=False)
    keep=[]; counts={}
    for idx,row in ranked.iterrows():
        ind=str(row.get("industry","其他")); cnt=counts.get(ind,0)
        if cnt>=max_per_industry: continue
        keep.append(idx); counts[ind]=cnt+1
        if len(keep)>=n: break
    return ranked.loc[keep] if keep else pd.DataFrame()


def render_card(row: pd.Series,horizon: str,reg_status: str,prices: Dict[str,pd.DataFrame],rank_no: int=1,holding: bool=False,cost: Optional[float]=None):
    edge=safe_float(row[f"{horizon}_hold_edge" if holding else f"{horizon}_edge"])
    conf=safe_float(row[f"{horizon}_conf"]); lev=levels(row,horizon)
    expert=EXPERT_LABELS.get(row[f"{horizon}_expert"],row[f"{horizon}_expert"]); setup=SETUP_LABELS.get(row["setup"],row["setup"])
    action=holding_action(row,horizon,reg_status,cost) if holding else new_position_action(row,horizon,reg_status)
    price=safe_float(row["price"]); chg=safe_float(row["change_pct"]); chg_color="#dc2626" if chg>=0 else "#047857"
    vals=[v for v in [safe_float(row.get("quality_score")),safe_float(row.get("business_score"))] if np.isfinite(v)]
    company=float(np.mean(vals)) if vals else np.nan
    flow=safe_float(row.get("true_flow_score")); pv=safe_float(row.get("price_volume_score")); sector=safe_float(row.get("sector_score"))
    entry=safe_float(row.get(f"{horizon}_entry_score")); safe=safe_float(row.get(f"{horizon}_safety_score")); phase=price_phase(row,horizon)
    st.markdown(f"""<div class="stock-card">
      <div class="stock-top"><div><div class="stock-name">#{rank_no} {row.get('name',row.name)}</div>
      <div class="stock-code">{row.name} · {row.get('industry','')} · {row.get('personality','')}</div></div>
      <div class="price">{fmt_price(price)}<br><span style="font-size:.78rem;color:{chg_color}">{chg:+.2f}%</span></div></div>
      <div style="margin-top:8px"><span class="pill pill-blue">{horizon_label(horizon)}</span><span class="pill pill-violet">{expert}</span><span class="pill pill-gray">{setup}</span></div>
      <div class="edge-wrap"><div class="edge-number">{edge:.0f}</div><div class="edge-label">Edge 百分位<br>資料信心 <b>{confidence_grade(conf)}</b> ({conf:.0f})</div>
      <div style="margin-left:auto;text-align:right"><div class="action">{action}</div><div class="small">{phase}</div></div></div>
      <div class="score-grid">
       <div class="score-box"><div class="score-k">🧬 公司</div><div class="score-v">{fmt_score(company)}</div></div>
       <div class="score-box"><div class="score-k">💰 法人</div><div class="score-v">{fmt_score(flow)}</div></div>
       <div class="score-box"><div class="score-k">📊 量價</div><div class="score-v">{fmt_score(pv)}</div></div>
       <div class="score-box"><div class="score-k">🏭 產業</div><div class="score-v">{fmt_score(sector)}</div></div>
       <div class="score-box"><div class="score-k">🎯 買點</div><div class="score-v">{fmt_score(entry)}</div></div>
       <div class="score-box"><div class="score-k">🛡 安全</div><div class="score-v">{fmt_score(safe)}</div></div>
      </div>
      <div class="level-grid">
       <div class="level"><div class="level-k">🟢 理想布局</div><div class="level-v">{fmt_price(lev['ideal_lo'])}–{fmt_price(lev['ideal_hi'])}</div></div>
       <div class="level"><div class="level-k">🚀 確認價</div><div class="level-v">{fmt_price(lev['trigger'])}</div></div>
       <div class="level"><div class="level-k">🟠 不追價</div><div class="level-v">&gt; {fmt_price(lev['do_not_chase'])}</div></div>
       <div class="level"><div class="level-k">🔒 保護價</div><div class="level-v">{fmt_price(lev['trailing'])}</div></div>
       <div class="level"><div class="level-k">🛑 失效價</div><div class="level-v">{fmt_price(lev['invalidation'])}</div></div>
      </div><div class="reason"><b>重點：</b>{reason_text(row,horizon)}</div></div>""",unsafe_allow_html=True)
    with st.expander("K線與進階細節",expanded=False):
        hist=prices.get(row.name)
        if hist is not None and not hist.empty: plot_stock(hist,row,horizon,lev)
        detail=pd.DataFrame({
            "指標":["RS短","RS中長","RVOL","ATR%","20MA乖離(ATR)","Crowding","Thesis惡化","Behavior樣本"],
            "數值":[fmt_score(row.get("rs_short_score")),fmt_score(row.get("rs_long_score")),f"{safe_float(row.get('rvol')):.2f}",f"{safe_float(row.get('atr_pct')):.2f}",f"{safe_float(row.get('extension_atr')):.2f}",fmt_score(row.get("crowding_risk")),fmt_score(row.get("thesis_deterioration")),int(safe_float(row.get("behavior_n"),0))]
        })
        st.dataframe(detail,use_container_width=True,hide_index=True)
        st.caption("KD/MACD不進核心Edge。Edge是全市場有效母池百分位，不是獲利機率。")


def plot_stock(raw: pd.DataFrame,row: pd.Series,horizon: str,lev: dict):
    d=enrich_ohlcv(raw).tail(110).copy(); x=d.index
    st.markdown('''<div class="chart-key">
      <span><i class="chart-dot" style="background:#475569"></i>10MA</span>
      <span><i class="chart-dot" style="background:#2563eb"></i>20MA</span>
      <span><i class="chart-dot" style="background:#7c3aed"></i>60MA</span>
      <span><i class="chart-dot" style="background:#d97706"></i>120MA</span>
      <span><i class="chart-dot" style="background:#f59e0b"></i>確認價</span>
      <span><i class="chart-dot" style="background:#dc2626"></i>失效價</span>
    </div>''',unsafe_allow_html=True)
    fig=make_subplots(rows=2,cols=1,shared_xaxes=True,vertical_spacing=.035,row_heights=[.76,.24])
    fig.add_trace(go.Candlestick(x=x,open=d["Open"],high=d["High"],low=d["Low"],close=d["Close"],showlegend=False,
                                 increasing_line_color="#ef4444",decreasing_line_color="#10b981"),row=1,col=1)
    colors_ma={10:"#475569",20:"#2563eb",60:"#7c3aed",120:"#d97706"}
    widths={10:1.0,20:1.6,60:1.6,120:1.15}
    for n in [10,20,60,120]:
        fig.add_trace(go.Scatter(x=x,y=d[f"MA{n}"],mode="lines",showlegend=False,line=dict(color=colors_ma[n],width=widths[n])),row=1,col=1)
    colors=np.where(d["Close"]>=d["Open"],"#ef4444","#10b981")
    fig.add_trace(go.Bar(x=x,y=d["Volume"],marker_color=colors,showlegend=False,opacity=.82),row=2,col=1)
    fig.add_hrect(y0=lev["ideal_lo"],y1=lev["ideal_hi"],fillcolor="rgba(37,99,235,.07)",line_width=0,row=1,col=1)
    fig.add_hline(y=lev["trigger"],line_dash="dot",line_color="#f59e0b",line_width=1.6,row=1,col=1)
    fig.add_hline(y=lev["invalidation"],line_dash="dash",line_color="#dc2626",line_width=1.6,row=1,col=1)
    fig.update_layout(height=500,margin=dict(l=8,r=8,t=6,b=8),template="plotly_white",showlegend=False,hovermode="x unified",
                      xaxis_rangeslider_visible=False,dragmode=False)
    fig.update_xaxes(showgrid=False,showspikes=True,spikemode="across",spikesnap="cursor",spikethickness=1)
    fig.update_yaxes(gridcolor="rgba(148,163,184,.20)",zeroline=False)
    st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False,"displaylogo":False,"scrollZoom":False,"responsive":True})


# -----------------------------------------------------------------------------
# 13. Production navigation
# -----------------------------------------------------------------------------

universe, universe_diag = load_universe()
up_fund=up_flow=up_tdcc=None
with st.sidebar:
    st.markdown("### 📡 資料狀態")
    st.caption(f"母池：{universe_diag.get('source','—')} · {len(universe):,}檔")
    if universe_diag.get("errors"):
        with st.expander("母池備援紀錄",expanded=False):
            st.write("\n".join(universe_diag.get("errors",[])[:4]))
    if st.button("重新取得官方母池",use_container_width=True,key="refresh_universe"):
        fetch_official_universe.clear(); liquidity_prefilter.clear(); st.session_state.pop("scan_result",None); st.rerun()
    with st.expander("進階資料（選填）",expanded=False):
        st.caption("基本面/法人CSV可提高模型完整度；正式資料請保留 as_of 或 available_from。")
        up_fund=st.file_uploader("fundamentals.csv",type=["csv"],key="fund")
        up_flow=st.file_uploader("flows.csv",type=["csv"],key="flow")
        up_tdcc=st.file_uploader("tdcc.csv（Shadow）",type=["csv"],key="tdcc")

fund=read_optional_source("fundamentals.csv",up_fund)
flow=read_optional_source("flows.csv",up_flow)
tdcc=read_optional_source("tdcc.csv",up_tdcc)
scan_as_of=pd.Timestamp(datetime.now()).tz_localize(None)

c1,c2=st.columns([3,1])
with c1:
    run=st.button("🚀 更新全市場雷達",type="primary",use_container_width=True,key="scan_button")
with c2:
    if st.button("清除快取",use_container_width=True,key="clear_cache"):
        download_prices_resilient.clear(); download_one.clear(); liquidity_prefilter.clear(); st.session_state.pop("scan_result",None); st.rerun()
if run:
    with st.spinner("正在掃描全上市櫃：先做流動性初篩，再執行5年深度量化模型…"):
        scan,reg,prices,diag=run_full_market_scan(universe,fund,flow,tdcc,scan_as_of)
        st.session_state["scan_result"]=(scan,reg,prices,diag)

scan=reg=prices=diag=None
if "scan_result" in st.session_state:
    scan,reg,prices,diag=st.session_state["scan_result"]
    render_system_status(diag,universe_diag)
    if scan is not None and not scan.empty:
        render_market(reg,diag,fund,flow)
        st.markdown(f"<div class='scan-strip'>更新 {datetime.now().strftime('%H:%M')} · 全市場 {int(diag.get('universe_total',len(universe))):,} → 流動性 {int(diag.get('liquidity_pass',0)):,} → 有效模型 {len(scan):,}。Edge 為有效母池橫斷面百分位。</div>",unsafe_allow_html=True)
    else:
        st.error("目前全市場掃描未完成。請開啟側邊欄的「系統狀態」查看行情來源；可先清除快取後重試。")
else:
    st.markdown(f"<div class='notice'>目前母池：<b>{len(universe):,} 檔</b>（{universe_diag.get('source','')}）。按「更新全市場雷達」後，系統會先以價格與流動性門檻篩除不可投資股票，再對通過者執行短／中／長深度模型。</div>",unsafe_allow_html=True)

prime_tab,short_tab,mid_tab,long_tab,doctor_tab=st.tabs(["⭐ 極選","⚡ 短線","📈 中線","🧭 長波段","🔎 個股"])

with prime_tab:
    st.markdown("<div class='section-title'>⭐ 今日極選</div><div class='section-sub'>跨短、中、長三週期比較後的最高 Net Edge；為降低同一產業重複曝險，前三名同產業最多2檔。</div>",unsafe_allow_html=True)
    if scan is None or scan.empty:
        st.info("請先更新全市場雷達。")
    else:
        c=_prime_frame(scan,CFG.top_n,2)
        for i,(t,row) in enumerate(c.iterrows(),1): render_card(row,row["best_horizon"],reg["status"],prices,i)

with short_tab:
    st.markdown("<div class='section-title'>⚡ 短線 2–10日</div><div class='section-sub'>優先量價、相對強度、Reclaim／突破與短期資金效率；Risk-Off 時可以沒有標的。</div>",unsafe_allow_html=True)
    if scan is None or scan.empty: st.info("請先更新全市場雷達。")
    else:
        c=_candidate_frame(scan,"short",reg["status"],CFG.top_n)
        if c.empty: st.info("今天沒有達到短線門檻的A級Setup。")
        for i,(t,row) in enumerate(c.iterrows(),1): render_card(row,"short",reg["status"],prices,i)

with mid_tab:
    st.markdown("<div class='section-title'>📈 中線 2–8週</div><div class='section-sub'>兼顧產業重新定價、基本面動能、趨勢與買點，是波段操作的核心頁。</div>",unsafe_allow_html=True)
    if scan is None or scan.empty: st.info("請先更新全市場雷達。")
    else:
        c=_candidate_frame(scan,"mid",reg["status"],CFG.top_n)
        if c.empty: st.info("今天沒有達到中線門檻的標的。")
        for i,(t,row) in enumerate(c.iterrows(),1): render_card(row,"mid",reg["status"],prices,i)

with long_tab:
    st.markdown("<div class='section-title'>🧭 長波段 2–9月</div><div class='section-sub'>重視公司品質、商業動能、產業領先與長期趨勢；短線過熱不等同既有長線部位必須退出。</div>",unsafe_allow_html=True)
    if scan is None or scan.empty: st.info("請先更新全市場雷達。")
    else:
        c=_candidate_frame(scan,"long",reg["status"],CFG.top_n)
        if c.empty: st.info("今天沒有達到長波段門檻的標的。")
        for i,(t,row) in enumerate(c.iterrows(),1): render_card(row,"long",reg["status"],prices,i)
        if "fundamental_completeness" in scan.columns:
            ec=scan[(scan["long_expert"]=="L_Compounder")&(scan["long_edge"]>=94)&(scan["fundamental_completeness"]>=.75)&(scan["entry_saturation"]<80)].sort_values("long_edge",ascending=False).head(2)
            st.markdown("<div class='section-title'>💎 Exceptional Compounder</div>",unsafe_allow_html=True)
            if ec.empty: st.caption("目前沒有符合極致成長條件者；此訊號刻意保持稀有。")
            for i,(t,row) in enumerate(ec.iterrows(),1): render_card(row,"long",reg["status"],prices,i)

with doctor_tab:
    st.markdown("<div class='section-title'>🔎 個股診斷</div><div class='section-sub'>輸入股票代碼與持股成本，分開回答「現在適不適合新買」與「已持有要不要續抱」。</div>",unsafe_allow_html=True)
    code=st.text_input("股票代碼",value="2330",placeholder="例如 2330 / 2383 / 6187")
    own=st.toggle("目前已有持股",value=False)
    cc1,cc2=st.columns(2)
    with cc1: cost=st.number_input("持股成本",min_value=0.0,value=0.0,step=1.0,disabled=not own)
    with cc2: shares=st.number_input("股數（選填）",min_value=0,value=0,step=100,disabled=not own)
    user_h=st.selectbox("投資週期",["自動判斷","短線 2–10日","中線 2–8週","長波段 2–9月"])
    analyze=st.button("🩺 立即診斷",type="primary",use_container_width=True,key="doctor")
    if analyze:
        ticker,one=download_one(code.strip(),CFG.history_period)
        if one.empty:
            st.error("找不到該股票行情；請確認代碼或稍後再試。")
        elif scan is None or scan.empty:
            st.warning("請先執行一次「更新全市場雷達」，個股Edge才有完整全市場比較基準。")
            st.caption("行情本身已取得，但尚未產生全市場橫斷面排名。")
        elif ticker not in scan.index:
            st.warning("該股不在目前有效投資母池（常見原因為流動性或歷史長度門檻）。目前不提供可能誤導的正式Edge。")
        else:
            row=scan.loc[ticker]
            h=row["best_horizon"] if user_h=="自動判斷" else {"短線 2–10日":"short","中線 2–8週":"mid","長波段 2–9月":"long"}[user_h]
            pmap=dict(prices); pmap[ticker]=one
            st.markdown("<div class='panel'><b>三週期比較</b><div class='small'>新進場 Edge / 持股 Edge</div></div>",unsafe_allow_html=True)
            cols=st.columns(3)
            for col,hh in zip(cols,["short","mid","long"]):
                with col: st.metric(horizon_label(hh),f"{fmt_score(row[f'{hh}_edge'])} / {fmt_score(row[f'{hh}_hold_edge'])}",f"信心 {confidence_grade(safe_float(row[f'{hh}_conf']))}")
            render_card(row,h,reg["status"],pmap,1,holding=own,cost=cost if own and cost>0 else None)
            if own and cost>0:
                pnow=safe_float(row["price"]); pnl=(pnow/cost-1)*100; lev=levels(row,h)
                st.markdown(f"<div class='doctor-card'><b>持股結論：</b>{holding_action(row,h,reg['status'],cost)}<br><span class='small'>成本 {fmt_price(cost)} · 現價 {fmt_price(pnow)} · 未實現 {pnl:+.1f}%</span></div>",unsafe_allow_html=True)
                if shares>0:
                    risk_amt=max(0,(pnow-lev["invalidation"])*shares)
                    st.caption(f"若跌至結構失效點，目前價到失效點的名目風險約 {risk_amt:,.0f} 元（未計滑價/稅費）。")
            st.markdown(f"<div class='notice'><b>如果今天沒有持股：</b>{new_position_action(row,h,reg['status'])}</div>",unsafe_allow_html=True)

st.markdown("---")
st.caption("⚠️ 量化研究與風險管理參考，不保證未來報酬。Edge為相對排名，不是上漲機率；交易前仍須確認行情時間、成本與重大事件。")
